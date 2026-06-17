"""
Stage 3: two LAI estimates per pot/component, computed and compared (DECISIONS.md §8).

  footprint_lai — projected leaf silhouette area / ground footprint = canopy COVER FRACTION.
                  Scale-robust (area/area cancels scale). Surface-only -> ≤1, underestimates
                  dense (multi-layer) canopies. A direct "what you see from above" baseline.

  volume_lai    — envelope VOLUME × leaf-area-density (LAD), to infer hidden interior biomass.
                  LAI = LAD · (volume / footprint) = LAD · mean canopy height  (= ∫LAD dz for
                  uniform LAD). Scale-SENSITIVE (volume ∝ scale³, footprint ∝ scale² → LAI ∝ scale)
                  → pass `scale_factor` from a known reference (e.g. pot diameter). LAD is
                  crop/stage-specific and MUST be calibrated (allometry) — defaults are placeholders.

Both take per-component indices from pots.separate_pots and the same up-projection basis.
numpy + scipy.spatial.ConvexHull only (no new dependency).
"""

import numpy as np
from scipy.spatial import ConvexHull

# Placeholder LAD (m² leaf / m³ canopy). NOT authoritative — calibrate per crop & growth stage.
DEFAULT_LAD = {"_generic": 2.0, "strawberry": 2.0, "watermelon": 1.5, "tomato": 2.5}


def _project2d(points, basis):
    u, v = basis
    return np.stack([points @ u, points @ v], axis=1)


def _occupied_area(coords2d, cell):
    """Projected coverage = number of occupied raster cells × cell² (the visible leaf silhouette)."""
    g = np.floor(coords2d / cell).astype(np.int64)
    # unique cells via void-view trick (fast set of rows)
    uniq = np.unique(g, axis=0)
    return len(uniq) * cell * cell


def _hull_area(coords2d):
    """2D convex-hull area = the ground patch the component spans (LAI denominator)."""
    if len(coords2d) < 3:
        return 0.0
    try:
        return float(ConvexHull(coords2d).volume)  # in 2D, .volume == area
    except Exception:
        r = coords2d.max(0) - coords2d.min(0)
        return float(r[0] * r[1])


def footprint_lai(points, comps, basis, cell=0.01):
    """Per-component canopy cover fraction (scale-robust, ≤1). See module docstring."""
    out = []
    for c in comps:
        p2 = _project2d(points[c["indices"]], basis)
        leaf = _occupied_area(p2, cell)     # visible projected leaf area (m²)
        ground = _hull_area(p2)             # spanned ground patch (m²)
        out.append({
            "leaf_proj_area_m2": round(leaf, 4),
            "ground_area_m2": round(ground, 4),
            "cover_fraction": round(leaf / ground, 4) if ground > 0 else 0.0,
        })
    return out


def volume_lai(points, comps, basis, lad, scale_factor=1.0):
    """Per-component volumetric LAI = LAD · (envelope_volume / footprint), CONVEX-HULL version.

    `points` are leaf SURFACE points; ConvexHull gives the convex outer envelope. It OVER-estimates
    for concave / sparse / gappy canopies because it bridges empty space and vertical clutter into
    one solid (this is what produced the mean_h≈1.06 m artifact on the continuous strawberry bed).
    Kept as a DIAGNOSTIC: the gap between this and `column_volume_lai` signals how concave/sparse the
    canopy (or how dirty the segmentation) is. `scale_factor` rescales to metric (NOT scale-invariant).
    Prefer `column_volume_lai` as the primary volumetric estimate.
    """
    out = []
    for c in comps:
        P = points[c["indices"]] * scale_factor
        ground = _hull_area(_project2d(P, basis))
        try:
            vol = float(ConvexHull(P).volume) if len(P) >= 4 else 0.0
        except Exception:
            vol = 0.0
        mean_h = vol / ground if ground > 0 else 0.0
        out.append({
            "envelope_volume_m3": round(vol, 5),
            "mean_height_m": round(mean_h, 4),
            "lad_m2_m3": lad,
            "ground_area_m2": round(ground, 4),
            "leaf_area_m2": round(lad * vol, 4),
            "volume_lai": round(lad * mean_h, 4),   # = lad·vol/ground = leaf_area/ground
        })
    return out


def column_volume_lai(points, comps, basis, lad, cell=0.02, base_percentile=5.0, scale_factor=1.0):
    """Per-component volumetric LAI from a 2.5D COLUMN (heightfield) model — robust default.

    Rasterize the component's leaf points onto the footprint grid (basis ⊥ up). Each occupied cell
    is an independent vertical column whose canopy height is `top - base`:
      - top  = MAX leaf height in that cell (canopy upper surface).
      - base = a low percentile of the component's heights (canopy floor ≈ bed surface / pot rim),
               robust to a few stray low points.
    Volume = Σ_cells cell² · max(0, top−base); footprint = (#occupied cells)·cell². Unlike ConvexHull
    this NEVER bridges empty space between cells, so it is not inflated by gaps, concavity, or a
    distant vertical clutter point (which only raises its own column, not the whole bed).

    The up axis is recovered from the basis as u×v (footprint_basis builds v=up×u ⇒ u×v=up), so the
    signature stays parallel to `volume_lai` and the same occupied-cell area is the LAI denominator
    (fixes the hull-denominator / dual-cell-size inconsistencies of the hull path).

    Args:
        cell: column raster size (m). ~2 cm default (matches pots.separate_pots).
        base_percentile: height percentile used as the canopy floor (5 = robust min).
        scale_factor: metric rescale (volume LAI is NOT scale-invariant).

    Caveat: per-cell `top` is a max, so a single high outlier inflates that one column — rely on
    upstream conf/segmentation filtering; a `top_percentile` knob can be added if needed.
    """
    u, v = basis
    up = np.cross(np.asarray(u, float), np.asarray(v, float))
    up = up / np.linalg.norm(up)
    cell_area = cell * cell
    out = []
    for c in comps:
        P = points[c["indices"]] * scale_factor
        if len(P) == 0:
            out.append({"column_volume_m3": 0.0, "mean_height_m": 0.0, "lad_m2_m3": lad,
                        "ground_area_m2": 0.0, "leaf_area_m2": 0.0, "volume_lai": 0.0,
                        "canopy_base_h": 0.0, "n_columns": 0})
            continue
        h = P @ up                                   # height along up
        coords2d = np.stack([P @ u, P @ v], axis=1)
        g = np.floor((coords2d - coords2d.min(0)) / cell).astype(np.int64)
        ids = g[:, 0] * (g[:, 1].max() + 1) + g[:, 1]
        uniq, inv = np.unique(ids, return_inverse=True)
        top = np.full(uniq.shape, -np.inf)
        np.maximum.at(top, inv, h)                   # per-cell canopy top
        base = float(np.percentile(h, base_percentile))
        col_h = np.clip(top - base, 0.0, None)       # per-cell canopy height
        volume = float(col_h.sum() * cell_area)
        footprint = float(len(uniq) * cell_area)     # occupied-cell footprint (consistent denom)
        mean_h = volume / footprint if footprint > 0 else 0.0
        out.append({
            "column_volume_m3": round(volume, 5),
            "mean_height_m": round(mean_h, 4),       # = mean per-cell canopy height
            "lad_m2_m3": lad,
            "ground_area_m2": round(footprint, 4),
            "leaf_area_m2": round(lad * volume, 4),
            "volume_lai": round(lad * mean_h, 4),    # = lad·vol/footprint = leaf_area/footprint
            "canopy_base_h": round(base, 4),
            "n_columns": int(len(uniq)),
        })
    return out
