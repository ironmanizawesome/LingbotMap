"""Compare raw 2D depth maps across resolutions to judge real detail vs. smooth upsampling.

If 728 resolves sharper foliage/edge structure than 518 (not just a blurrier upscale), the
extra points carry genuine geometric information. Per-image robust normalization so structure
is visible regardless of absolute scale.

  results/cmp_depthmap.png  - rows=[RGB, depth], cols=resolutions, for one frame
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZS = {"518": "predictions/restest/outside_lab_518.npz",
        "644": "predictions/restest/outside_lab_644.npz",
        "728": "predictions/restest/outside_lab_728.npz"}


def main(frame=200, crop=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for j, (k, p) in enumerate(NPZS.items()):
        d = np.load(p, allow_pickle=True)
        f = min(frame, d["depth"].shape[0] - 1)
        rgb = d["images"][f].transpose(1, 2, 0)        # (H,W,3)
        dep = d["depth"][f, :, :, 0]                   # (H,W)
        H, W = dep.shape
        if crop:  # crop = (y0,y1,x0,x1) in [0,1] normalized
            y0, y1, x0, x1 = crop
            ys, ye, xs, xe = int(y0*H), int(y1*H), int(x0*W), int(x1*W)
            rgb, dep = rgb[ys:ye, xs:xe], dep[ys:ye, xs:xe]
        vmin, vmax = np.percentile(dep, 5), np.percentile(dep, 95)
        axes[0, j].imshow(np.clip(rgb, 0, 1)); axes[0, j].set_title(f"{k}px RGB ({W}x{H})"); axes[0, j].axis("off")
        im = axes[1, j].imshow(dep, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[1, j].set_title(f"{k}px depth"); axes[1, j].axis("off")
    fig.tight_layout(); fig.savefig("results/cmp_depthmap.png", dpi=120); plt.close(fig)
    print("Saved results/cmp_depthmap.png  (frame", frame, "crop", crop, ")")


if __name__ == "__main__":
    frame = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    # optional normalized crop: pass 4 floats y0 y1 x0 x1
    crop = tuple(map(float, sys.argv[2:6])) if len(sys.argv) >= 6 else None
    main(frame, crop)
