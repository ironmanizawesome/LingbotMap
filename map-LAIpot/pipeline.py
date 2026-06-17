"""
map-LAIpot full LAI estimation pipeline: precomputed NPZ → per-pot LAI.

Stages
------
  1. Load NPZ (precomputed by precompute_npz.py) → filter → leaf points
  2. up_vector — estimate gravity-up from camera poses
  3. pots      — separate pots/plants via footprint connected components
  4. lai       — per-component footprint LAI + column-volume LAI

The --npz flag skips inference entirely; this script assumes the NPZ was already
produced by precompute_npz.py (with --mask_sky recommended for greenhouse scans).

Usage
-----
    python map-LAIpot/pipeline.py \\
        --npz predictions/restest/Tomato_pot_644.npz \\
        --crop tomato \\
        --scale_factor 0.15 \\
        --out_json results/tomato_lai.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_module(name: str):
    """Import a sibling .py file from the same directory without package install."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _unproject(depth: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Depth → world-space 3D points (numpy-only).

    depth:    [S, H, W] or [S, H, W, 1]
    extrinsic:[S, 3, 4] camera-to-world (c2w) — matches postprocess() output in demo.py
    intrinsic:[S, 3, 3]
    returns:  [S, H, W, 3]
    """
    if depth.ndim == 4:
        depth = depth[..., 0]
    S, H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = np.empty((S, H, W, 3), dtype=np.float32)
    for i in range(S):
        K, d = intrinsic[i], depth[i]
        cam = np.stack(
            [(uu - K[0, 2]) * d / K[0, 0], (vv - K[1, 2]) * d / K[1, 1], d], axis=-1
        )
        R, t = extrinsic[i][:3, :3], extrinsic[i][:3, 3]
        # world = R_c2w @ cam + t_c2w  (c2w stored as R | t, so world = cam @ R.T + t)
        out[i] = cam @ R.T + t
    return out


def _load_leaf_points(
    npz_path: str,
    frame_stride: int = 25,
    conf_pct: float = 45.0,
    exg_thr: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Load NPZ and return (leaf_points [N,3], ext4 [S,4,4]).

    Filtering pipeline:
      1. Unproject depth → world points
      2. Validity: finite values + magnitude < 1e4
      3. Confidence: keep top (100 - conf_pct)% of valid points
      4. ExG green-leaf proxy: keep 2G-R-B > exg_thr
    """
    d = np.load(npz_path, allow_pickle=True)

    # ── frame subsampling ────────────────────────────────────────────────────
    n_frames = d["depth"].shape[0]
    idx = np.arange(0, n_frames, frame_stride)

    # ── unproject ────────────────────────────────────────────────────────────
    wp = _unproject(d["depth"][idx], d["extrinsic"][idx], d["intrinsic"][idx]).reshape(-1, 3)

    # ── colour (NCHW [S,3,H,W] or NHWC [S,H,W,3]) ──────────────────────────
    imgs = d["images"][idx]
    if imgs.ndim == 4 and imgs.shape[1] == 3:   # NCHW → NHWC
        imgs = imgs.transpose(0, 2, 3, 1)
    col = np.clip(imgs.reshape(-1, 3), 0.0, 1.0).astype(np.float32)

    # ── confidence ──────────────────────────────────────────────────────────
    cf = d["depth_conf"][idx].reshape(-1)

    # ── validity + confidence gate ───────────────────────────────────────────
    valid = np.isfinite(wp).all(1) & np.isfinite(cf) & (np.abs(wp).max(1) < 1e4)
    thresh = np.percentile(cf[valid], conf_pct) if valid.any() else 0.0
    valid &= cf >= thresh
    wp, col = wp[valid], col[valid]

    # ── ExG leaf proxy ───────────────────────────────────────────────────────
    exg = 2.0 * col[:, 1] - col[:, 0] - col[:, 2]
    leaf = exg > exg_thr
    wp = wp[leaf]

    # ── extrinsics for up_from_cameras (needs [S,4,4]) ───────────────────────
    ext = d["extrinsic"][idx]                        # [S,3,4]
    ext4 = np.zeros((len(idx), 4, 4), dtype=np.float64)
    ext4[:, :3, :4] = ext
    ext4[:, 3, 3] = 1.0

    return wp, ext4


# ---------------------------------------------------------------------------
# Public API (importable from lai-service tasks.py)
# ---------------------------------------------------------------------------

def run_pipeline(
    npz_path: str,
    *,
    crop: str = "strawberry",
    lad: float | None = None,
    scale_factor: float = 1.0,
    frame_stride: int = 25,
    conf_pct: float = 45.0,
    exg_thr: float = 0.06,
    cell: float = 0.02,
    close_iter: int = 2,
    min_area: float = 0.01,
    max_components: int | None = None,
) -> dict:
    """Run the full LAI pipeline on a precomputed NPZ.

    Args:
        npz_path:       Path to .npz produced by precompute_npz.py.
        crop:           Crop type string for DEFAULT_LAD lookup.
        lad:            Override leaf-area density (m²/m³). None → table lookup.
        scale_factor:   Metric rescale for column-volume LAI.
        frame_stride:   Use every Nth frame (reduces memory; 25 ≈ 1 fps @ 25-fps source).
        conf_pct:       Confidence percentile cutoff (keep top (100-conf_pct)%).
        exg_thr:        ExG threshold for green-leaf proxy (2G-R-B > thr).
        cell:           Footprint raster cell size (m).
        close_iter:     Binary-closing iterations for pot segmentation.
        min_area:       Minimum component footprint area (m²).
        max_components: Limit to N largest components (None = all).

    Returns:
        dict with keys: npz_path, crop, lad_m2_m3, scale_factor, n_components,
        n_leaf_points, up_vector, basis_u, basis_v,
        agg_cover_fraction, agg_column_lai, agg_volume_lai, components.
    """
    uv_mod = _load_module("up_vector")
    pots_mod = _load_module("pots")
    lai_mod = _load_module("lai")

    if lad is None:
        lad = lai_mod.DEFAULT_LAD.get(crop, lai_mod.DEFAULT_LAD["_generic"])

    # ── Stage 0: load + filter ────────────────────────────────────────────────
    print(f"[pipeline] Loading {npz_path}")
    leaf_pts, ext4 = _load_leaf_points(
        npz_path, frame_stride=frame_stride, conf_pct=conf_pct, exg_thr=exg_thr
    )
    print(f"[pipeline] {len(leaf_pts):,} leaf points after filtering")
    if len(leaf_pts) < 10:
        raise ValueError(
            f"Too few leaf points ({len(leaf_pts)}); "
            "lower conf_pct or exg_thr, or check NPZ quality."
        )

    # ── Stage 1: up vector ───────────────────────────────────────────────────
    up_prior = uv_mod.up_from_cameras(ext4)
    up = uv_mod.estimate_up_vector(leaf_pts, up_prior)
    print(f"[pipeline] up vector: {[round(float(x), 4) for x in up]}")

    # ── Stage 2: separate pots ───────────────────────────────────────────────
    labels, comps, (u, v) = pots_mod.separate_pots(
        leaf_pts, up, cell=cell, close_iter=close_iter, min_area=min_area
    )
    print(f"[pipeline] {len(comps)} component(s) found")

    if max_components is not None:
        comps = sorted(comps, key=lambda c: -c["footprint_area_m2"])[:max_components]

    # ── Stage 3: LAI ─────────────────────────────────────────────────────────
    fl = lai_mod.footprint_lai(leaf_pts, comps, (u, v), cell=0.01)
    cl = lai_mod.column_volume_lai(
        leaf_pts, comps, (u, v), lad=lad, cell=cell, scale_factor=scale_factor
    )

    components = []
    for i, (comp, f, c) in enumerate(zip(comps, fl, cl)):
        components.append({
            "comp_index": i,
            "n_points": int(comp["n_points"]),
            "footprint_area_m2": float(comp["footprint_area_m2"]),
            # footprint (scale-robust)
            "leaf_proj_area_m2": float(f["leaf_proj_area_m2"]),
            "ground_area_m2": float(f["ground_area_m2"]),
            "cover_fraction": float(f["cover_fraction"]),
            # column-volume (primary volumetric estimate)
            "column_volume_m3": float(c["column_volume_m3"]),
            "mean_height_m": float(c["mean_height_m"]),
            "lad_m2_m3": float(c["lad_m2_m3"]),
            "leaf_area_m2": float(c["leaf_area_m2"]),
            "volume_lai": float(c["volume_lai"]),
            "n_columns": int(c["n_columns"]),
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total_footprint = sum(c["footprint_area_m2"] for c in components)
    total_leaf_proj = sum(c["leaf_proj_area_m2"] for c in components)
    total_leaf_area = sum(c["leaf_area_m2"] for c in components)
    total_volume = sum(c["column_volume_m3"] for c in components)

    return {
        "npz_path": str(npz_path),
        "crop": crop,
        "lad_m2_m3": float(lad),
        "scale_factor": float(scale_factor),
        "n_components": len(components),
        "n_leaf_points": int(len(leaf_pts)),
        "up_vector": [round(float(x), 6) for x in up],
        "basis_u": [round(float(x), 6) for x in u],
        "basis_v": [round(float(x), 6) for x in v],
        # aggregated metrics
        "agg_cover_fraction": round(total_leaf_proj / total_footprint, 4) if total_footprint > 0 else 0.0,
        "agg_column_lai": round(total_leaf_area / total_footprint, 4) if total_footprint > 0 else 0.0,
        "agg_volume_m3": round(total_volume, 5),
        "components": components,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="map-LAIpot pipeline: precomputed NPZ → per-pot LAI (JSON)"
    )
    p.add_argument("--npz", required=True,
                   help="Path to .npz produced by precompute_npz.py")
    p.add_argument("--crop", default="strawberry",
                   help="Crop type for LAD lookup: strawberry / tomato / watermelon / _generic")
    p.add_argument("--lad", type=float, default=None,
                   help="Override LAD (m² leaf/m³ canopy). Default: per-crop table.")
    p.add_argument("--scale_factor", type=float, default=1.0,
                   help="Metric rescale for column-volume LAI (not scale-invariant).")
    p.add_argument("--frame_stride", type=int, default=25,
                   help="Use every Nth frame (reduces memory; default 25).")
    p.add_argument("--conf_pct", type=float, default=45.0,
                   help="Confidence percentile cutoff: keep top (100-conf_pct)%% of points.")
    p.add_argument("--exg_thr", type=float, default=0.06,
                   help="ExG threshold for green-leaf proxy (2G-R-B > thr).")
    p.add_argument("--cell", type=float, default=0.02,
                   help="Footprint raster cell size (m).")
    p.add_argument("--close_iter", type=int, default=2,
                   help="Binary-closing iterations for pot segmentation.")
    p.add_argument("--min_area", type=float, default=0.01,
                   help="Minimum component footprint area (m²).")
    p.add_argument("--max_components", type=int, default=None,
                   help="Limit to N largest components (default: all).")
    p.add_argument("--out_json", default=None,
                   help="Write JSON results to this file (default: print to stdout).")
    args = p.parse_args()

    result = run_pipeline(
        npz_path=args.npz,
        crop=args.crop,
        lad=args.lad,
        scale_factor=args.scale_factor,
        frame_stride=args.frame_stride,
        conf_pct=args.conf_pct,
        exg_thr=args.exg_thr,
        cell=args.cell,
        close_iter=args.close_iter,
        min_area=args.min_area,
        max_components=args.max_components,
    )

    out = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[pipeline] Results saved → {args.out_json}")
    else:
        print(out)


if __name__ == "__main__":
    main()
