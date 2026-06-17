"""
Stage 2: separate the point cloud into individual pots / plants by footprint.

Only the up vector matters for projection (offset cancels — see DECISIONS.md §2). We project
points along `up` onto the 2D footprint plane, rasterize occupancy, bridge small within-plant
gaps (binary closing), and take connected components:
  - spaced pots  -> separate components (one per pot/plant)
  - continuous bed -> a single component (honest: cannot split what isn't separated)

Occupied-cell count doubles as the per-pot footprint area (m²), which Stage 3 uses as the LAI
denominator. Uses only numpy + scipy.ndimage (no new dependency, no sklearn/DBSCAN).
"""

import numpy as np
from scipy import ndimage


def footprint_basis(up: np.ndarray):
    """Two unit vectors u, v spanning the plane ⊥ up (for projecting to a 2D footprint)."""
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(up, ref); u /= np.linalg.norm(u)
    v = np.cross(up, u)
    return u, v


def separate_pots(
    points: np.ndarray,
    up: np.ndarray,
    cell: float = 0.02,
    close_iter: int = 2,
    min_area: float = 0.01,
):
    """Cluster points into pots/plants via footprint occupancy connected-components.

    Args:
        points: [N,3] world coordinates (e.g. leaf points).
        up: [3] gravity-up (from Stage 1 estimate_up_vector).
        cell: footprint raster cell size (metric units). ~2 cm default.
        close_iter: binary-closing iterations to bridge within-plant gaps (cell units).
        min_area: drop components smaller than this footprint area (m²) as noise.

    Returns:
        labels: [N] int — component id per point (0..k-1), -1 = noise/too-small.
        comps:  list of dicts {indices, n_points, footprint_area_m2, centroid_uv}.
        basis:  (u, v) the footprint basis used.
    """
    pts = np.asarray(points, dtype=np.float64)
    u, v = footprint_basis(up)
    cu, cv = pts @ u, pts @ v

    gu = np.floor((cu - cu.min()) / cell).astype(np.int64)
    gv = np.floor((cv - cv.min()) / cell).astype(np.int64)
    occ = np.zeros((gu.max() + 1, gv.max() + 1), dtype=bool)
    occ[gu, gv] = True
    if close_iter > 0:
        occ = ndimage.binary_closing(occ, iterations=close_iter)

    lab, n = ndimage.label(occ)                       # 8-connectivity off by default (4-conn)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    min_cells = max(1, int(round(min_area / (cell * cell))))
    keep = [i + 1 for i in range(n) if sizes[i] >= min_cells]
    remap = {cl: k for k, cl in enumerate(keep)}

    point_comp = lab[gu, gv]                           # raw component id per point (>=1)
    labels = np.array([remap.get(pc, -1) for pc in point_comp], dtype=np.int64)

    comps = []
    for k, cl in enumerate(keep):
        idx = np.nonzero(labels == k)[0]
        comps.append({
            "indices": idx,
            "n_points": int(idx.size),
            "footprint_area_m2": float((lab == cl).sum() * cell * cell),
            "centroid_uv": (float(cu[idx].mean()), float(cv[idx].mean())),
        })
    return labels, comps, (u, v)
