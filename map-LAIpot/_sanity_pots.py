"""Dev sanity: run Stage 1 (up) + Stage 2 (pot separation) on a real npz and render the
footprint colored by component. Continuous bed -> ~1 component (honest); spaced pots -> many.

Usage: python map-LAIpot/_sanity_pots.py predictions/restest/Strawberry_onebed_644.npz
"""
import importlib.util, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _unproject(depth, extrinsic, intrinsic):
    """Depth -> world points, numpy-only (mirrors lingbot_map.utils.geometry, no torch dep).

    depth: (S,H,W) or (S,H,W,1); extrinsic: (S,3,4) world->cam; intrinsic: (S,3,3).
    """
    if depth.ndim == 4:
        depth = depth[..., 0]
    S, H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    out = np.empty((S, H, W, 3), np.float32)
    for i in range(S):
        K, d = intrinsic[i], depth[i]
        cam = np.stack([(uu - K[0, 2]) * d / K[0, 0],
                        (vv - K[1, 2]) * d / K[1, 1], d], axis=-1)
        R, t = extrinsic[i][:3, :3], extrinsic[i][:3, 3]
        out[i] = cam @ R + (-R.T @ t)      # world = cam @ R_c2w.T + t_c2w; R_c2w=R.T
    return out


def _load(name):
    spec = importlib.util.spec_from_file_location(name, f"map-LAIpot/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


uv, pots = _load("up_vector"), _load("pots")


def main(path, frame_stride=25, conf_pct=45, cell=0.02, close_iter=2, min_area=0.01, exg_thr=0.06):
    d = np.load(path, allow_pickle=True)
    idx = np.arange(0, d["depth"].shape[0], frame_stride)
    wp = _unproject(d["depth"][idx], d["extrinsic"][idx], d["intrinsic"][idx]).reshape(-1, 3)
    col = d["images"][idx].transpose(0, 2, 3, 1).reshape(-1, 3)
    cf = d["depth_conf"][idx].reshape(-1)
    valid = np.isfinite(wp).all(1) & np.isfinite(cf) & (np.abs(wp).max(1) < 1e4)
    valid &= cf >= np.percentile(cf[valid], conf_pct)
    wp, col = wp[valid], np.clip(col[valid], 0, 1)

    # green leaf proxy (Excess Green = 2G-R-B), stand-in for SAM2 LeafSegmentor
    exg = 2 * col[:, 1] - col[:, 0] - col[:, 2]
    for t in (0.02, 0.05, 0.10, 0.15):
        print(f"  ExG>{t}: {100*(exg>t).mean():.1f}% of points")
    leaf = exg > exg_thr
    print(f"  leaf proxy (ExG>{exg_thr}): {leaf.sum():,} of {len(wp):,} pts")
    wp = wp[leaf]

    ext4 = np.broadcast_to(np.eye(4), (len(idx), 4, 4)).copy(); ext4[:, :3, :4] = d["extrinsic"][idx]
    up = uv.estimate_up_vector(wp, uv.up_from_cameras(ext4), max_tilt_deg=30)
    labels, comps, (u, v) = pots.separate_pots(wp, up, cell=cell, close_iter=close_iter, min_area=min_area)
    print(f"{path}: {len(wp):,} pts -> {len(comps)} components (noise {int((labels==-1).sum()):,})")

    lai = _load("lai")
    order = sorted(range(len(comps)), key=lambda i: -comps[i]['footprint_area_m2'])[:5]
    sub = [comps[i] for i in order]
    lad = lai.DEFAULT_LAD.get("strawberry")
    fl = lai.footprint_lai(wp, sub, (u, v), cell=0.01)
    vl = lai.volume_lai(wp, sub, (u, v), lad=lad)                       # convex-hull (diagnostic)
    cl = lai.column_volume_lai(wp, sub, (u, v), lad=lad, cell=cell)     # 2.5D column (primary)
    print("  footprint | cover | HULL vol/mean_h/LAI | COLUMN vol/mean_h/LAI  (LAD=2.0)")
    for cc, fr, hr, cr in zip(sub, fl, vl, cl):
        print(f"   area={cc['footprint_area_m2']:6.3f} | cover={fr['cover_fraction']:.3f} | "
              f"hull {hr['envelope_volume_m3']:.4f}/{hr['mean_height_m']:.3f}/{hr['volume_lai']:.3f} | "
              f"col {cr['column_volume_m3']:.4f}/{cr['mean_height_m']:.3f}/{cr['volume_lai']:.3f}")

    cu, cv = wp @ u, wp @ v
    sel = np.random.default_rng(0).choice(len(wp), min(300_000, len(wp)), replace=False)
    cmap = plt.get_cmap("tab20")
    colors = np.array([[0.6, 0.6, 0.6] if labels[i] < 0 else cmap(labels[i] % 20)[:3] for i in sel])
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(cu[sel], cv[sel], c=colors, s=0.5, marker=".", linewidths=0)
    ax.set_title(f"footprint by component: {len(comps)} comps  (cell={cell}, close={close_iter}, min_area={min_area})")
    ax.set_aspect("equal")
    os.makedirs("results", exist_ok=True)
    out = "results/sanity_pots_" + path.split("/")[-1].replace(".npz", ".png")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "predictions/restest/Strawberry_onebed_644.npz")
