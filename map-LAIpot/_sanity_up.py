"""Dev sanity check: run Stage-1 up estimation on a real saved npz (no GT exists, so this
is a *plausibility* check, not an accuracy number).

Signals we look for:
  - camera-prior up  vs  constrained-RANSAC up  should roughly AGREE (small angle) if both valid.
  - height-along-up histogram should show a dominant low band (bench/floor) with canopy above.
  - "green" (leaf) points should sit ABOVE the dominant low plane.

Usage: python map-LAIpot/_sanity_up.py predictions/restest/Strawberry_onebed_644.npz
"""
import importlib.util
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

# load up_vector.py directly (hyphenated package dir isn't importable by name)
spec = importlib.util.spec_from_file_location("up_vector", "map-LAIpot/up_vector.py")
uv = importlib.util.module_from_spec(spec); spec.loader.exec_module(uv)


def ang(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))


def main(path, frame_stride=25, conf_pct=40):
    d = np.load(path, allow_pickle=True)
    depth, conf = d["depth"], d["depth_conf"]
    extr, intr, images = d["extrinsic"], d["intrinsic"], d["images"]
    S = depth.shape[0]
    idx = np.arange(0, S, frame_stride)
    wp = unproject_depth_map_to_point_map(depth[idx], extr[idx], intr[idx]).reshape(-1, 3)
    col = images[idx].transpose(0, 2, 3, 1).reshape(-1, 3)
    cf = conf[idx].reshape(-1)
    valid = np.isfinite(wp).all(1) & np.isfinite(cf) & (np.abs(wp).max(1) < 1e4)
    thr = np.percentile(cf[valid], conf_pct)
    valid &= cf >= thr
    wp, col = wp[valid], np.clip(col[valid], 0, 1)
    print(f"{path}: {len(idx)} frames, {len(wp):,} valid points")

    # extrinsic is c2w (demo.py postprocess); pad to 4x4 for up_from_cameras
    ext4 = np.broadcast_to(np.eye(4), (len(idx), 4, 4)).copy()
    ext4[:, :3, :4] = extr[idx]
    up_cam = uv.up_from_cameras(ext4)
    up_est = uv.estimate_up_vector(wp, up_cam, max_tilt_deg=30)
    print(f"  up_from_cameras = {up_cam.round(3)}")
    print(f"  estimate_up     = {up_est.round(3)}")
    print(f"  agreement (cam vs RANSAC up): {ang(up_cam, up_est):.1f} deg  (작을수록 일관)")

    # height-along-up distribution
    h = wp @ up_est
    h = h - np.percentile(h, 1)
    print(f"  height span (1-99pct): {np.percentile(h,99)-0:.2f} (metric units)")
    # green (leaf) vs all: leaves should be higher than the low band
    green = (col[:, 1] > col[:, 0] + 0.05) & (col[:, 1] > col[:, 2] + 0.05)
    if green.sum() > 100:
        print(f"  green points: {green.sum():,} ({100*green.mean():.1f}%)  "
              f"median height green={np.median(h[green]):.2f} vs all={np.median(h):.2f} "
              f"(green이 더 높아야 정상)")

    # render: side view (u vs height) + top-down (u vs v), colored by RGB
    ref = np.array([1.0, 0, 0]) if abs(up_est[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(up_est, ref); u /= np.linalg.norm(u)
    v = np.cross(up_est, u)
    cu, cv = wp @ u, wp @ v
    sel = np.random.default_rng(0).choice(len(wp), min(300_000, len(wp)), replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].scatter(cu[sel], h[sel], c=col[sel], s=0.4, marker=".", linewidths=0)
    ax[0].set_title("side view (u vs height-along-up)  ← 베드=수평 띠, 잎=위로"); ax[0].set_aspect("equal")
    ax[1].scatter(cu[sel], cv[sel], c=col[sel], s=0.4, marker=".", linewidths=0)
    ax[1].set_title("top-down (u vs v)"); ax[1].set_aspect("equal")
    out = "results/sanity_up_" + path.split("/")[-1].replace(".npz", ".png")
    import os; os.makedirs("results", exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "predictions/restest/Strawberry_onebed_644.npz")
