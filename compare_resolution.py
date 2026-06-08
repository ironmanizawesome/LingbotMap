"""Compare streaming 3D reconstruction across input resolutions (does NOT touch demo.py).

Why this exists
---------------
The released checkpoint's patch-embed `pos_embed` was created at a native 518px grid
(37x37 patches). demo.py builds the model with `img_size=--image_size`, which couples the
*constructed* pos_embed size to the requested resolution -> feeding e.g. 728 makes a 52x52
pos_embed that no longer matches the checkpoint (size mismatch on load).

But the architecture already supports variable resolution at *runtime*:
  - prepare_tokens_with_masks() bicubic-interpolates pos_embed to the real H/W
    (lingbot_map/layers/vision_transformer.py:228-235), and
  - the aggregator reads the real H/W from the image tensor and feeds it to 3D RoPE
    (lingbot_map/aggregator/base.py:565,595-596).

So the correct pattern is: BUILD once at the checkpoint-native size (--model_native_size,
default 518) so weights load cleanly, then run inference at each target resolution by only
changing the *preprocessing* size. One model instance serves all resolutions.

Usage
-----
    python compare_resolution.py \
        --model_path checkpoints/lingbot-map-long.pt \
        --image_folder example/courthouse \
        --first_k 32 \
        --image_sizes 518,644,728 \
        --use_sdpa

Outputs, per resolution: peak GPU memory, inference time, and an .npz of predictions in
--out_dir (view locally with view_npz.py to compare point-cloud quality).
"""

import argparse
import datetime
import os
import time

# Must be set before any CUDA init (mirrors demo.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images

# Reuse demo.py's image/video loading and post-processing verbatim so the comparison
# matches the real pipeline exactly. Importing demo only sets an env var + imports.
import demo as demo_mod


# Order of columns written to the markdown benchmark log (one row per resolution run).
LOG_COLUMNS = ["time", "scene", "frames", "fps", "img_size", "WxH", "peak_GB",
               "infer_s", "fps_eff", "offload", "sdpa", "native", "keyframe"]


def append_log(log_path, row):
    """Append one row to a markdown table at log_path, creating header if the file is new.

    Markdown tables grow downward, so appending rows keeps the file a valid, readable table.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if new_file:
            f.write("# Resolution benchmark log\n\n")
            f.write("| " + " | ".join(LOG_COLUMNS) + " |\n")
            f.write("|" + "|".join("---" for _ in LOG_COLUMNS) + "|\n")
        f.write("| " + " | ".join(str(row.get(c, "")) for c in LOG_COLUMNS) + " |\n")


def build_model(model_path, native_size, patch_size, num_scale_frames, use_sdpa, device):
    """Build GCTStream at the checkpoint-native size and load weights.

    Kwargs mirror demo.load_model so the only difference vs. demo is `img_size`, which we
    pin to `native_size` (not the per-run input resolution).
    """
    model = GCTStream(
        img_size=native_size,
        patch_size=patch_size,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=use_sdpa,
        camera_num_iterations=4,
    )
    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    print("  Checkpoint loaded.")
    return model.to(device).eval()


def run_one(model, images, num_scale_frames, keyframe_interval, dtype, device, offload=False):
    """Run streaming inference for a single preprocessed tensor.

    Returns (predictions, images_for_post, peak_GB, secs). When offload=True, per-frame
    predictions are moved to CPU during inference (output_device=cpu) to cut VRAM, and the
    CPU image tensor from predictions["images"] is used for post-processing (mirrors
    demo.py's --offload_to_cpu path).
    """
    model.clean_kv_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    output_device = torch.device("cpu") if offload else None
    images = images.to(device)
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(
            images,
            num_scale_frames=num_scale_frames,
            keyframe_interval=keyframe_interval,
            output_device=output_device,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    if offload:
        # Predictions (incl. "images") are already on CPU; free the GPU input copy so the
        # next (larger) resolution starts from a clean slate.
        images_for_post = predictions["images"]
        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        images_for_post = images
    return predictions, images_for_post, peak, secs


def main():
    p = argparse.ArgumentParser(description="Compare reconstruction across input resolutions")
    p.add_argument("--model_path", required=True)
    p.add_argument("--image_folder", default=None)
    p.add_argument("--video_path", default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--first_k", type=int, default=32,
                   help="Cap frames so the sweep is fast. Use the same value across resolutions.")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--image_sizes", default="518,644,728",
                   help="Comma-separated input widths to compare. Each must be a multiple of "
                        "--patch_size (e.g. 518=37x14, 644=46x14, 728=52x14).")
    p.add_argument("--model_native_size", type=int, default=518,
                   help="Resolution the checkpoint's pos_embed was created at. The model is BUILT "
                        "here so weights load; input resolution is varied via pos-embed interpolation.")
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--num_scale_frames", type=int, default=8)
    p.add_argument("--keyframe_interval", type=int, default=1)
    p.add_argument("--use_sdpa", action="store_true", default=False,
                   help="Use SDPA attention (no FlashInfer dependency).")
    p.add_argument("--offload", action="store_true", default=False,
                   help="Offload per-frame predictions to CPU during inference (cuts VRAM, "
                        "slightly slower). Lets you use more frames / higher res without OOM.")
    p.add_argument("--out_dir", default="predictions/restest")
    p.add_argument("--log_path", default="results/benchmark_log.md",
                   help="Markdown file to append one row per resolution run (conditions + time/mem).")
    args = p.parse_args()

    assert args.image_folder or args.video_path, "Provide --image_folder or --video_path"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse + validate resolutions (snap to a multiple of patch_size).
    sizes = []
    for tok in args.image_sizes.split(","):
        s = int(tok)
        if s % args.patch_size != 0:
            snapped = (s // args.patch_size) * args.patch_size
            print(f"Warning: {s} not divisible by {args.patch_size}; snapping to {snapped}")
            s = snapped
        sizes.append(s)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Resolve frames ONCE (handles video extraction too) and reuse the same paths for
    #    every resolution, so all runs see the identical frame set. ────────────────────
    images0, paths, _ = demo_mod.load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=sizes[0], patch_size=args.patch_size,
    )
    preprocessed = {sizes[0]: images0}
    for s in sizes[1:]:
        preprocessed[s] = load_and_preprocess_images(
            paths, mode="crop", image_size=s, patch_size=args.patch_size,
        )

    # ── Build the model ONCE at native size ───────────────────────────────────────────
    model = build_model(
        args.model_path, args.model_native_size, args.patch_size,
        args.num_scale_frames, args.use_sdpa, device,
    )

    dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
             else (torch.float16 if torch.cuda.is_available() else torch.float32))
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    scene = os.path.splitext(os.path.basename(os.path.normpath(
        args.image_folder or args.video_path)))[0]

    # ── Sweep ─────────────────────────────────────────────────────────────────────────
    results = []
    for s in sizes:
        images = preprocessed[s]
        h, w = images.shape[-2:]
        print(f"\n=== image_size={s}  ({w}x{h}, {images.shape[0]} frames, native={args.model_native_size}) ===")
        # Fields that are known regardless of success/OOM; time/mem are filled in per outcome.
        log_base = {
            "scene": scene, "frames": images.shape[0], "fps": args.fps,
            "img_size": s, "WxH": f"{w}x{h}",
            "offload": "Y" if args.offload else "N",
            "sdpa": "Y" if args.use_sdpa else "N",
            "native": args.model_native_size, "keyframe": args.keyframe_interval,
        }
        now = lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            predictions, images_used, peak, secs = run_one(
                model, images, args.num_scale_frames, args.keyframe_interval, dtype, device,
                offload=args.offload,
            )
            preds, images_cpu = demo_mod.postprocess(predictions, images_used)
            vis = demo_mod.prepare_for_visualization(preds, images_cpu)
            out_path = os.path.join(args.out_dir, f"{scene}_{s}.npz")
            np.savez_compressed(out_path, **vis)
            fps_eff = images.shape[0] / secs if secs > 0 else 0.0
            print(f"  peak={peak:.2f} GB  time={secs:.1f}s  ({fps_eff:.1f} fps)  saved={out_path}")
            results.append((s, f"{w}x{h}", f"{peak:.2f} GB", f"{secs:.1f}s", f"{fps_eff:.1f}", out_path))
            append_log(args.log_path, {**log_base, "time": now(), "peak_GB": f"{peak:.2f}",
                                       "infer_s": f"{secs:.1f}", "fps_eff": f"{fps_eff:.1f}"})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM at image_size={s}")
            results.append((s, f"{w}x{h}", "OOM", "-", "-", "-"))
            append_log(args.log_path, {**log_base, "time": now(), "peak_GB": "OOM",
                                       "infer_s": "-", "fps_eff": "-"})

    # ── Summary ───────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'size':>6} | {'WxH':>10} | {'peak mem':>9} | {'time':>7} | {'fps':>5} | npz")
    print("-" * 78)
    for s, wh, peak, secs, fps_eff, out in results:
        print(f"{s:>6} | {wh:>10} | {peak:>9} | {secs:>7} | {fps_eff:>5} | {out}")
    print("=" * 78)
    print("Visualize each with: python view_npz.py <npz>   (compare detail vs. scale drift)")


if __name__ == "__main__":
    main()
