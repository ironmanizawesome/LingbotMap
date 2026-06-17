"""
Stage 1: robust gravity-UP estimation for greenhouse / potted-crop scenes.

Per-pot LAI needs only the gravity-up DIRECTION, not a ground height: the projected
leaf/pot areas depend solely on the plane normal (the offset cancels in the in-plane
projection onto u,v ⊥ normal). So instead of "fitting the ground plane" we estimate a
single robust up vector.

Greenhouse capture is horizontal-ish and cluttered (walls, bench fronts, pot sides), so a
plain largest-plane RANSAC can lock onto a vertical surface. We therefore CONSTRAIN the
search to planes whose normal is within `max_tilt_deg` of an up prior (from camera poses or
supplied), pick the largest such ~horizontal plane (the bench/floor), and refine its normal
with iterative PCA (total least squares) — same refinement used in map-LAIcrop/ground.py.
"""

import numpy as np


def up_from_cameras(extrinsics_c2w: np.ndarray) -> np.ndarray:
    """Coarse gravity-up prior from camera poses (assumes operator held camera ~upright).

    Args:
        extrinsics_c2w: [S,3,4] or [S,4,4] camera-to-world matrices (as saved by demo.py).

    Returns:
        [3] unit vector — mean world direction of the cameras' image-up axis. Falls back to
        +Z if the cameras average out (e.g. a full orbit).
    """
    R = np.asarray(extrinsics_c2w)[:, :3, :3]
    # OpenCV camera convention: +Y is image-down, so image-up in camera frame is -Y.
    ups = R @ np.array([0.0, -1.0, 0.0])         # [S,3]
    mean_up = ups.mean(axis=0)
    n = np.linalg.norm(mean_up)
    return np.array([0.0, 0.0, 1.0]) if n < 1e-8 else mean_up / n


def estimate_up_vector(
    points: np.ndarray,
    up_prior: np.ndarray,
    max_tilt_deg: float = 25.0,
    ransac_iterations: int = 1000,
    distance_threshold: float = 0.02,
    refit_iters: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Robust gravity-up via tilt-constrained RANSAC + iterative PCA refit.

    Only the normal matters for per-pot area, so we return a unit up vector (no offset).
    Candidate planes whose normal deviates from `up_prior` by more than `max_tilt_deg` are
    rejected — this prevents locking onto vertical walls / pot sides that often dominate
    horizontal greenhouse capture.

    Args:
        points: [N,3] world coordinates (e.g. world_points[conf_mask]).
        up_prior: [3] approximate up direction (e.g. from up_from_cameras, or [0,0,1]).
        max_tilt_deg: reject planes tilted more than this from up_prior.
        distance_threshold: inlier band half-width (metric units).
        refit_iters: PCA refit iterations (converges in 1-2; extra are cheap).

    Returns:
        up: [3] unit vector aligned with up_prior (oriented "up").
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("포인트가 3개 미만입니다.")
    up_prior = np.asarray(up_prior, dtype=np.float64)
    up_prior = up_prior / np.linalg.norm(up_prior)
    cos_max = np.cos(np.radians(max_tilt_deg))
    rng = np.random.default_rng(seed)

    # ── tilt-constrained RANSAC: largest plane whose normal is ~aligned with up_prior ──
    best_normal = up_prior.copy()
    best_d = -float(best_normal @ pts.mean(axis=0))
    best_inliers = 0
    for _ in range(ransac_iterations):
        p0, p1, p2 = pts[rng.choice(len(pts), 3, replace=False)]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-8:
            continue
        nrm = nrm / ln
        if nrm @ up_prior < 0:                      # orient toward prior
            nrm = -nrm
        if nrm @ up_prior < cos_max:                # too tilted → wall/pot side → reject
            continue
        d = -float(nrm @ p0)
        inl = int((np.abs(pts @ nrm + d) < distance_threshold).sum())
        if inl > best_inliers:
            best_inliers, best_normal, best_d = inl, nrm, d

    # ── iterative PCA (TLS) refit on full-extent inliers (lowers variance) ──
    normal, d = best_normal, best_d
    for _ in range(refit_iters):
        inliers = pts[np.abs(pts @ normal + d) < distance_threshold]
        if len(inliers) < 3:
            break
        centroid = inliers.mean(axis=0)
        eigvals, eigvecs = np.linalg.eigh(np.cov(inliers, rowvar=False))
        cand = eigvecs[:, 0]                         # smallest eigenvalue → plane normal
        cn = np.linalg.norm(cand)
        if cn < 1e-8 or not np.isfinite(eigvals).all():
            break
        cand = cand / cn
        if cand @ up_prior < 0:
            cand = -cand
        # guard: a refit that drifts past the tilt gate means inliers got contaminated.
        if cand @ up_prior < cos_max:
            break
        normal, d = cand, -float(cand @ centroid)

    return normal
