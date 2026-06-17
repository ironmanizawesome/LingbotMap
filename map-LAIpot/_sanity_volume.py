"""Dev sanity: column (2.5D heightfield) volume vs convex-hull volume on synthetic canopies.

Shows that column_volume_lai recovers the true canopy volume while ConvexHull volume_lai
over-estimates exactly where it hurt us on real data:
  (1) solid box      -> both ~correct (convex case, formula check).
  (2) two slabs+gap  -> hull bridges the empty gap; column does not.
  (3) flat slab + one tall clutter spike -> hull balloons over the whole footprint
      (the mean_h~1.06 m artifact mechanism); column keeps it local.

Usage: python map-LAIpot/_sanity_volume.py
"""
import importlib.util
import numpy as np


def _load(name):
    spec = importlib.util.spec_from_file_location(name, f"map-LAIpot/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


lai = _load("lai")
pots = _load("pots")
rng = np.random.default_rng(0)

UP = np.array([0.0, 0.0, 1.0])
BASIS = pots.footprint_basis(UP)
LAD = 2.0


def _fill_box(u0, u1, v0, v1, h0, h1, n):
    """n random points filling an axis-aligned box in (x=u, y=v, z=h)."""
    x = rng.uniform(u0, u1, n); y = rng.uniform(v0, v1, n); z = rng.uniform(h0, h1, n)
    return np.stack([x, y, z], axis=1)


def _report(name, pts, true_vol, true_mean_h, expect):
    comps = [{"indices": np.arange(len(pts))}]
    h = lai.volume_lai(pts, comps, BASIS, lad=LAD)[0]
    c = lai.column_volume_lai(pts, comps, BASIS, lad=LAD, cell=0.02)[0]
    print(f"\n[{name}]  truth: vol≈{true_vol:.4f} m³, mean_h≈{true_mean_h:.3f} m   ({expect})")
    print(f"  hull  : vol={h['envelope_volume_m3']:.4f}  mean_h={h['mean_height_m']:.3f}  "
          f"LAI={h['volume_lai']:.3f}")
    print(f"  column: vol={c['column_volume_m3']:.4f}  mean_h={c['mean_height_m']:.3f}  "
          f"LAI={c['volume_lai']:.3f}  (base={c['canopy_base_h']:.3f}, cols={c['n_columns']})")
    return h, c


# (1) Solid convex box: 0.5 x 0.5 x 0.3  -> vol 0.075, mean_h 0.30. Both should match (~).
box = _fill_box(0, 0.5, 0, 0.5, 0, 0.3, 80_000)
_report("1 solid box", box, 0.075, 0.30, "both ~correct")

# (2) Two slabs (each 0.2 wide) with a 0.6-wide empty gap between -> true canopy vol = 0.06.
#     Hull bridges the gap (spans full 1.0 width) -> ~0.15.  Column ignores empty cells -> ~0.06.
slabs = np.vstack([
    _fill_box(0.0, 0.2, 0, 0.5, 0, 0.3, 30_000),
    _fill_box(0.8, 1.0, 0, 0.5, 0, 0.3, 30_000),
])
hg, cg = _report("2 two slabs + gap", slabs, 0.06, 0.30, "hull over (~0.15), column ~0.06")

# (3) Flat slab (0.5x0.5x0.3) + a single tall clutter spike to h=1.5 in one corner.
#     Hull lifts an apex over the whole footprint -> mean_h balloons.  Column localizes it.
spike = _fill_box(0.48, 0.50, 0.48, 0.50, 0.0, 1.5, 2_000)
clutter = np.vstack([_fill_box(0, 0.5, 0, 0.5, 0, 0.3, 60_000), spike])
hc, cc = _report("3 slab + tall spike", clutter, 0.075, 0.30,
                 "hull mean_h balloons, column ~0.31")

print("\nchecks:")
print(f"  (2) hull overestimates gap : {hg['envelope_volume_m3'] > 2 * cg['column_volume_m3']}  "
      f"(hull {hg['envelope_volume_m3']:.3f} vs column {cg['column_volume_m3']:.3f})")
print(f"  (3) hull mean_h > 2x column: {hc['mean_height_m'] > 2 * cc['mean_height_m']}  "
      f"(hull {hc['mean_height_m']:.3f} vs column {cc['mean_height_m']:.3f})")
