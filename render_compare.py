"""Render saved-npz point clouds to PNGs for side-by-side resolution comparison.

Mirrors the viewer's unprojection (lingbot_map ... unproject_depth_map_to_point_map on
pred_dict["extrinsic"]/["intrinsic"]) so what we render matches what view_npz.py shows.

Outputs:
  results/cmp_overview.png  - rows=[top-down X-Z, front X-Y], cols=resolutions, shared axis limits
  results/cmp_detail.png    - one frame, full pixel density, per resolution (shows the density gain)
Also prints per-resolution numeric stats (valid points, conf, scene bbox extent).
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

NPZS = {
    "518": "predictions/restest/outside_lab_518.npz",
    "644": "predictions/restest/outside_lab_644.npz",
    "728": "predictions/restest/outside_lab_728.npz",
}
RNG = np.random.default_rng(0)


def cloud_from_npz(path, frame_stride, conf_pct, max_points):
    d = np.load(path, allow_pickle=True)
    depth, conf = d["depth"], d["depth_conf"]          # (S,H,W,1), (S,H,W)
    extr, intr = d["extrinsic"], d["intrinsic"]        # (S,3,4), (S,3,3)
    images = d["images"]                               # (S,3,H,W)
    S = depth.shape[0]
    idx = np.arange(0, S, frame_stride)
    wp = unproject_depth_map_to_point_map(depth[idx], extr[idx], intr[idx])  # (s,H,W,3)
    col = images[idx].transpose(0, 2, 3, 1).reshape(-1, 3)
    pts = wp.reshape(-1, 3)
    cf = conf[idx].reshape(-1)
    valid = np.isfinite(pts).all(1) & np.isfinite(cf) & (np.abs(pts).max(1) < 1e4)
    thr = np.percentile(cf[valid], conf_pct)
    valid &= cf >= thr
    pts, col, cf = pts[valid], np.clip(col[valid], 0, 1), cf[valid]
    n_valid = len(pts)
    if n_valid > max_points:
        sel = RNG.choice(n_valid, max_points, replace=False)
        pts_r, col_r = pts[sel], col[sel]
    else:
        pts_r, col_r = pts, col
    return pts_r, col_r, n_valid, cf, pts


def robust_lims(pts, lo=2, hi=98, pad=0.05):
    mn, mx = np.percentile(pts, lo, 0), np.percentile(pts, hi, 0)
    c, r = (mn + mx) / 2, (mx - mn) * (1 + pad) / 2
    return c - r, c + r  # min[3], max[3]


def overview():
    clouds, stats = {}, {}
    for k, p in NPZS.items():
        pts_r, col_r, n_valid, cf, pts_all = cloud_from_npz(p, frame_stride=10, conf_pct=30, max_points=400_000)
        clouds[k] = (pts_r, col_r)
        lo, hi = robust_lims(pts_all)
        stats[k] = dict(n_valid=n_valid, conf_med=float(np.median(cf)),
                        ext=(hi - lo), rendered=len(pts_r))
    # shared limits from 518 (reference) so scale/shape diffs are visible
    ref_pts = cloud_from_npz(NPZS["518"], 10, 30, 400_000)[4]
    lo, hi = robust_lims(ref_pts)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for j, k in enumerate(NPZS):
        pts, col = clouds[k]
        # top-down X-Z
        axes[0, j].scatter(pts[:, 0], pts[:, 2], c=col, s=0.4, marker=".", linewidths=0)
        axes[0, j].set_title(f"{k}px  top-down (X-Z)  | valid={stats[k]['n_valid']:,}")
        axes[0, j].set_xlim(lo[0], hi[0]); axes[0, j].set_ylim(lo[2], hi[2]); axes[0, j].set_aspect("equal")
        # front X-Y (y down)
        axes[1, j].scatter(pts[:, 0], pts[:, 1], c=col, s=0.4, marker=".", linewidths=0)
        axes[1, j].set_title(f"{k}px  front (X-Y)")
        axes[1, j].set_xlim(lo[0], hi[0]); axes[1, j].set_ylim(hi[1], lo[1]); axes[1, j].set_aspect("equal")
    fig.tight_layout(); fig.savefig("results/cmp_overview.png", dpi=110); plt.close(fig)

    print("\n=== overview stats (frame_stride=10, conf>=30th pct) ===")
    for k in NPZS:
        s = stats[k]
        print(f"  {k}px: valid_points={s['n_valid']:>9,}  conf_med={s['conf_med']:.2f}  "
              f"scene_extent(X,Y,Z)=({s['ext'][0]:.2f},{s['ext'][1]:.2f},{s['ext'][2]:.2f})")


def detail(frame=200):
    """Single frame, FULL pixel density -> shows the raw density gain per resolution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for j, (k, p) in enumerate(NPZS.items()):
        d = np.load(p, allow_pickle=True)
        f = min(frame, d["depth"].shape[0] - 1)
        wp = unproject_depth_map_to_point_map(d["depth"][f:f+1], d["extrinsic"][f:f+1], d["intrinsic"][f:f+1])[0]
        col = d["images"][f].transpose(1, 2, 0)
        cf = d["depth_conf"][f]
        H, W = cf.shape
        pts = wp.reshape(-1, 3); c = np.clip(col.reshape(-1, 3), 0, 1); cfl = cf.reshape(-1)
        valid = np.isfinite(pts).all(1) & (cfl >= np.percentile(cfl, 20)) & (np.abs(pts).max(1) < 1e4)
        pts, c = pts[valid], c[valid]
        axes[j].scatter(pts[:, 0], pts[:, 2], c=c, s=0.6, marker=".", linewidths=0)
        axes[j].set_title(f"{k}px frame {f}  ({W}x{H} = {H*W:,}px, kept {len(pts):,})")
        axes[j].set_aspect("equal")
    fig.tight_layout(); fig.savefig("results/cmp_detail.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    overview()
    detail(frame=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
    print("\nSaved: results/cmp_overview.png, results/cmp_detail.png")
