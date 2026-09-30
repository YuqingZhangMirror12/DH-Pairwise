"""Weighted flat-kernel mean shift, initialized at EVERY distinct observation.

No seed selection, normal votes, sigma calibration, labels, or edge filtering.
The only geometric scale is radius_px. Tolerance/iteration limit are numerical
convergence checks: a failure raises, rather than silently dropping a mode.
"""
import numpy as np
from scipy.spatial import cKDTree


def weighted_modes(points, weights, radius_px):
    x = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(x) != len(w) or radius_px <= 0 or not np.isfinite(radius_px):
        raise ValueError('invalid dimensions or radius')
    if not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('all input observations must be finite with positive mass')
    if not len(x):
        return []
    tree = cKDTree(x)
    centers = np.unique(x, axis=0)
    finished = {}
    cache = {}
    for iteration in range(1024):
        pending = []
        for center, indices in zip(centers, tree.query_ball_point(centers, radius_px)):
            key = tuple(sorted(indices))
            if not key:
                raise RuntimeError('mean shift lost every observation')
            if key not in cache:
                idx = np.asarray(key)
                cache[key] = np.sum(x[idx] * w[idx, None], axis=0) / w[idx].sum()
            nxt = cache[key]
            if np.max(np.abs(nxt - center)) <= 1e-7:
                finished[key] = nxt
            else:
                pending.append(nxt)
        if not pending:
            break
        centers = np.unique(pending, axis=0)
    else:
        raise RuntimeError('mean shift did not converge; do not hide this mode')
    result = []
    for key, center in finished.items():
        idx = np.asarray(key, dtype=np.int64)
        result.append(dict(center=center, members=idx, mass=float(w[idx].sum())))
    return sorted(result, key=lambda m: (-m['mass'], *m['center']))


def unique_modes(modes, radius_px):
    """Standard fixed-representative mode suppression, not transitive linkage.

    Used for descriptive mode counts; observations are not filtered out of
    the original EdgeCloud. Each retained mode still owns its radial support.
    """
    out = []
    for mode in modes:
        if not any(np.linalg.norm(mode['center'] - old['center']) < radius_px for old in out):
            out.append(mode)
    return out
