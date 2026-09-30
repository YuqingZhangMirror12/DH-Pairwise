"""Target-only correspondence construction for independently cut seams.

All coordinates here are in the common GT frame. Only the resulting target
arrays go to the loss; eligibility/provenance/GT never enter the six inputs.
The generator must project the UNION of either side's mismatch intervals onto
BOTH contours. Damaged seam tokens use the existing ignored-target value -2,
not a forced match and not a fabricated dustbin label. Other unmatched points
use -1, as do all points in a negative pair.
"""
import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import recover_mutual_contour_correspondences


def projected_interval_damage(along, intervals, *, margin_px=3.):
    """Include transition shoulders and smoothing neighbourhoods, not just peaks."""
    along = np.asarray(along, dtype=np.float64)
    if along.ndim != 1 or not np.isfinite(along).all():
        raise ValueError('finite 1-D common-frame seam coordinates required')
    if not np.isfinite(margin_px) or margin_px < 0:
        raise ValueError('nonnegative finite damage margin required')
    damage = np.zeros(len(along), dtype=bool)
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError('explicit interval start/end required')
        start, stop = interval
        if not np.isfinite([start, stop]).all() or stop <= start:
            raise ValueError('invalid mismatch interval')
        damage |= (along >= start - margin_px) & (along <= stop + margin_px)
    return damage


def correspondence_targets(points_a, points_b, eligible_a, eligible_b,
                           mismatch_a, mismatch_b, *, label, uniform_wear=(0., 0.)):
    if type(label) is not bool:
        raise ValueError('explicit boolean label required')
    points = [np.asarray(p, np.float64) for p in (points_a, points_b)]
    eligible = [np.asarray(v) for v in (eligible_a, eligible_b)]
    mismatch = [np.asarray(v) for v in (mismatch_a, mismatch_b)]
    for p, e, m in zip(points, eligible, mismatch):
        if p.ndim != 2 or p.shape[1:] != (2,) or not np.isfinite(p).all():
            raise ValueError('finite contour points required')
        if e.dtype != np.bool_ or m.dtype != np.bool_ or e.shape != (len(p),) or m.shape != (len(p),):
            raise ValueError('boolean per-token seam and damage provenance required')
    if len(uniform_wear) != 2 or not np.isfinite(uniform_wear).all() or not all(0 <= w <= 1.4 for w in uniform_wear):
        raise ValueError('R uniform wear must be independently recorded in [0,1.4] px per side')
    targets = [np.full(len(p), -1, dtype=np.int64) for p in points]
    max_distance = 3. + sum(uniform_wear)
    if not label:
        return (*targets, dict(positive=False, correspondence_count=0,
                              rejected_mismatch_matches=0, target_semantics='all negative tokens dustbin -1'))
    for target, e, m in zip(targets, eligible, mismatch):
        target[e & m] = -2
    candidates = recover_mutual_contour_correspondences(*points, max_distance_px=max_distance)
    retained, rejected_damage = [], 0
    for i, j in candidates:
        if not (eligible[0][i] and eligible[1][j]):
            continue
        if mismatch[0][i] or mismatch[1][j]:
            rejected_damage += 1
            # The intact partner of a damaged token must not become a false
            # dustbin target just because its projected interval is a boundary.
            targets[0][i] = targets[1][j] = -2
            continue
        targets[0][i], targets[1][j] = j, i
        retained.append((int(i), int(j)))
    return (*targets, dict(positive=True, correspondence_count=len(retained),
                          distance_only_candidate_count=len(candidates),
                          rejected_mismatch_matches=rejected_damage,
                          distance_tolerance_px=max_distance,
                          uniform_wear_px=list(uniform_wear),
                          target_semantics='reciprocal healthy seam matches; mismatch -2; other unmatched -1',
                          generator_pixel_provenance_still_requires_independent_audit=True))
