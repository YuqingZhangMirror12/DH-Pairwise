"""Prediction-free post-analysis of extra missing material at the CLEAN GT seam.

This metric never changes inputs, targets, training, layout or an existing
evaluation. A TEST manifest producer supplies the original clean pair and its
fixed stress input; all models can subsequently join the same metric by pair ID.

At an A anchor with outward normal n, probe x_A(s)=anchor+s*n, s in [-8,8]
at 0.25px spacing. The SAME world line uses x_B(s)=x_A(s)+t_GT because placing B
in A coordinates subtracts t_GT. Gap is the signed B-enter minus A-exit offset:
positive means separation, negative means digital-mask overlap. Added gap is
stress minus clean separation, NOT a physical measurement of a real seam width.

Every profile must have exactly one correctly directed transition, with stable
endpoints inside the 800px frame. Multiple intersections, insufficient range,
or negative added separation are explicitly unmeasured, never silently clipped
or resolved by picking a convenient crossing. The 0.25px probing is a numerical
resolution on nearest-pixel masks, not subpixel physical accuracy.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

from .rachel_training_dataset import RachelPairSample


SCHEMA = "rachel-clean-gt-seam-added-gap/v1"
PROBE_OFFSETS_PX = np.arange(-32, 33, dtype=np.float64) * .25
NORMAL_PROBE_DISTANCES_PX = (2., 4., 8.)
MINIMUM_MEASURED_ARC_PX = 8.
DAMAGE_ADDED_GAP_PX = 1.
DAMAGE_CLEAN_SUPPORTED_ARC_FRACTION = .10
_NEGATIVE_TOLERANCE_PX = 1e-9


def _mask(mask):
    value = np.asarray(mask)
    if value.shape != (1, 800, 800) or not np.all((value == 0) | (value == 1)):
        raise ValueError("seam diagnostics require binary [1,800,800] masks")
    return value[0].astype(np.bool_, copy=False)


def _probe(mask, points):
    """Nearest-pixel probes and explicit in-frame flags, preserving leading dims."""
    indices = np.floor(np.asarray(points, np.float64) + .5).astype(np.int64)
    in_frame = np.all((indices >= 0) & (indices < np.asarray(mask.shape)), axis=-1)
    values = np.zeros(indices.shape[:-1], dtype=np.bool_)
    values[in_frame] = mask[indices[..., 0][in_frame], indices[..., 1][in_frame]]
    return values, in_frame


def _clean_supported_anchors(clean, mask_a, mask_b, gt):
    """Independent reconstruction of clean GT seam anchors, not student points.

    Half-adjacent-edge arc cells are disjoint on the original ordered contour;
    each original token contributes once regardless of overlapping patches.
    Normal sign comes from A polygon orientation and is never fitted to a model.
    """
    ids = np.flatnonzero(clean.contour_valid_a)
    points = np.asarray(clean.points_rc_a, np.float64)[ids]
    empty = (np.empty(0, np.int64), np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
    support = dict(seam_token_count=0, seam_arc_px=0., perimeter_px=0.,
                   clean_supported_token_count=0, clean_supported_arc_px=0., clean_support_coverage_fraction=0.)
    if len(points) < 4 or not np.isfinite(points).all():
        return empty, support, "invalid_clean_ordered_contour"
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    area = .5 * np.sum(points[:, 1] * np.roll(-points[:, 0], -1)
                        - np.roll(points[:, 1], -1) * -points[:, 0])
    if np.any(lengths <= 1e-8) or abs(area) <= 1e-8:
        return empty, support, "degenerate_clean_ordered_contour"
    weights = .5 * (lengths + np.roll(lengths, 1))
    targets = np.asarray(clean.target_a)[ids]
    seam = targets >= 0
    if np.any(seam):
        matches = targets[seam]
        if (np.any(matches >= len(clean.target_b)) or np.any(~np.asarray(clean.contour_valid_b)[matches])
                or not np.array_equal(np.asarray(clean.target_b)[matches], ids[seam])):
            raise ValueError("clean GT seam assignments must be valid and reciprocal")
    support.update(seam_token_count=int(seam.sum()), seam_arc_px=float(weights[seam].sum()),
                   perimeter_px=float(lengths.sum()))
    if not np.any(seam):
        return empty, support, "no_clean_reciprocal_gt_seam"
    tangent = edges / lengths[:, None] + np.roll(edges / lengths[:, None], 1, axis=0)
    tangent_length = np.linalg.norm(tangent, axis=1)
    normals = np.stack((tangent[:, 1], -tangent[:, 0]), axis=1) * (1. if area > 0 else -1.)
    normals /= np.maximum(tangent_length[:, None], 1e-12)
    trusted = np.zeros(len(points), dtype=np.bool_)
    for distance in NORMAL_PROBE_DISTANCES_PX:
        plus, minus = points + distance * normals, points - distance * normals
        ap, api = _probe(mask_a, plus)
        am, ami = _probe(mask_a, minus)
        bp, bpi = _probe(mask_b, plus + gt)
        bm, bmi = _probe(mask_b, minus + gt)
        trusted |= ~ap & am & bp & ~bm & api & ami & bpi & bmi
    trusted &= seam & (tangent_length > 1e-8)
    arc = float(weights[trusted].sum())
    support.update(clean_supported_token_count=int(trusted.sum()), clean_supported_arc_px=arc,
                   clean_support_coverage_fraction=arc / max(1e-12, support["seam_arc_px"]))
    rows = (ids[trusted], points[trusted], normals[trusted], weights[trusted])
    return rows, support, None if trusted.any() else "no_clean_gt_supported_normals"


def _single_boundary(profile, *, exiting):
    """A single unambiguous crossing; stable outer two samples at each end."""
    changes = np.flatnonzero(profile[1:] != profile[:-1])
    if len(changes) > 1:
        return None, "multiple_line_crossings"
    if not len(changes):
        return None, "no_line_crossing_within_range"
    first, last = (True, False) if exiting else (False, True)
    if not np.all(profile[:2] == first) or not np.all(profile[-2:] == last):
        return None, "unstable_or_wrong_direction_endpoints"
    index = int(changes[0])
    # Midpoint of the observed pixel-state change; never fit/interpolate a mask.
    return float(.5 * (PROBE_OFFSETS_PX[index] + PROBE_OFFSETS_PX[index + 1])), None


def _weighted_statistics(values, weights):
    if not len(values) or float(weights.sum()) <= 0:
        return dict(weighted_mean_px=None, weighted_p90_px=None)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    at = min(len(order) - 1, int(np.searchsorted(cumulative, .90 * cumulative[-1], side="left")))
    return dict(weighted_mean_px=float(np.dot(values, weights) / weights.sum()),
                weighted_p90_px=float(values[order[at]]))


def _removed_probe_pixels(clean_mask, stress_mask, points):
    """Unique deleted pixels touched by supported rays, not whole-band area."""
    indices = np.floor(points.reshape(-1, 2) + .5).astype(np.int64)
    indices = indices[np.all((indices >= 0) & (indices < 800), axis=1)]
    if not len(indices):
        return 0
    flat = np.unique(indices[:, 0] * 800 + indices[:, 1])
    return int(np.count_nonzero(clean_mask.ravel()[flat] & ~stress_mask.ravel()[flat]))


def seam_gap_diagnostics(clean, student, *, include_samples=False):
    """Return JSON-only GT metric evidence; no prediction or model is accepted.

    actual_seam_damaged is None when unmeasurable, not False. A measurable pair
    qualifies iff >=8px arc was measured AND >=10% of ORIGINAL clean-supported
    seam arc has >=1px added separation. Invalid measurements remain in that
    denominator, preventing a shrunken measurement subset from inflating damage.
    Optional samples (<=512 clean anchors) are exclusively for metric galleries.
    """
    if not isinstance(clean, RachelPairSample) or not isinstance(student, RachelPairSample):
        raise TypeError("clean and student must be RachelPairSample")
    if type(include_samples) is not bool:
        raise TypeError("include_samples must be bool")
    for key in ("pair_id", "fragment_a_token", "fragment_b_token", "label", "translation_valid"):
        if getattr(clean, key) != getattr(student, key):
            raise ValueError("clean/stress pair identity differs: " + key)
    if float(clean.label) not in (0., 1.):
        raise ValueError("pair label must be binary")
    if bool(clean.translation_valid):
        for key in ("translation_a_to_b_rc", "translation_a_to_b_xy_cartesian"):
            if not np.array_equal(getattr(clean, key), getattr(student, key)):
                raise ValueError("original GT placement must be preserved")
    ca, cb, sa, sb = (_mask(getattr(sample, "mask_" + side))
                      for sample in (clean, student) for side in "ab")
    if np.any(sa & ~ca) or np.any(sb & ~cb):
        raise ValueError("stress masks must be inward-only subsets of clean masks")
    output = dict(schema_version=SCHEMA, pair_id=str(clean.pair_id), label=bool(clean.label),
        valid=False, invalid_reason=None, actual_seam_damaged=None,
        zero_identity=bool(np.array_equal(ca, sa) and np.array_equal(cb, sb)),
        physical_gap_width_ground_truth=False, prediction_used=False, model_input=False,
        protocol=dict(probe_range_px=[-8., 8.], probe_step_px=.25,
            sampling="nearest pixel center; boundary is crossing-bracket midpoint",
            endpoint_stability="first and last two probes must have the expected state",
            gt_sign="x_B=x_A+t_GT; B placement in A frame is -t_GT",
            gap_sign="B_enter-A_exit: positive separation, negative digital-mask overlap",
            added_gap="stress_gap-clean_gap; negative values flagged, never clipped",
            arc_definition="one clean-token cell, half each adjacent edge; no overlapping patch votes",
            weighted_p90="first value reaching 90% cumulative independent arc weight",
            normal_probe_distances_px=list(NORMAL_PROBE_DISTANCES_PX),
            minimum_measured_arc_px=MINIMUM_MEASURED_ARC_PX,
            damage_added_gap_px=DAMAGE_ADDED_GAP_PX,
            damage_fraction_of_clean_supported_arc=DAMAGE_CLEAN_SUPPORTED_ARC_FRACTION),
        support=dict(seam_token_count=0, seam_arc_px=0., perimeter_px=0.,
            clean_supported_token_count=0, clean_supported_arc_px=0., clean_support_coverage_fraction=0.,
            measured_token_count=0, measured_arc_px=0., coverage_fraction=0.,
            invalid_token_count=0, invalid_token_reasons={}, invalid_arc_px_by_reason={}),
        clean_gap=_weighted_statistics([], np.empty(0)), stress_gap=_weighted_statistics([], np.empty(0)),
        added_gap=_weighted_statistics([], np.empty(0)), damaged_added_gap=_weighted_statistics([], np.empty(0)),
        damage_added_ge_1px_arc_px=None, damage_added_ge_1px_arc_fraction=None,
        damage_fraction_of_measured_arc=None, stress_positive_separation_arc_fraction=None,
        clean_negative_overlap_arc_fraction=None, negative_added_gap_count=0,
        removed_pixels_touched_by_supported_rays_a=0, removed_pixels_touched_by_supported_rays_b=0)
    if include_samples:
        output["samples"] = []
    if not bool(clean.label):
        output["invalid_reason"] = "negative_pair_has_no_gt_seam"
        return output
    gt = np.asarray(clean.translation_a_to_b_rc, np.float64)
    if not bool(clean.translation_valid) or gt.shape != (2,) or not np.isfinite(gt).all():
        output["invalid_reason"] = "missing_or_nonfinite_gt_translation"
        return output
    (ids, anchors, normals, weights), support, reason = _clean_supported_anchors(clean, ca, cb, gt)
    output["support"].update(support)
    if reason:
        output["invalid_reason"] = reason
        return output
    if len(anchors) > 512:
        raise ValueError("clean metric contour exceeds fixed N512 cap")
    lines_a = anchors[:, None] + PROBE_OFFSETS_PX[None, :, None] * normals[:, None]
    lines_b = lines_a + gt
    profiles = [(*_probe(mask, line), exiting, name) for mask, line, exiting, name in (
        (ca, lines_a, True, "clean_a"), (cb, lines_b, False, "clean_b"),
        (sa, lines_a, True, "stress_a"), (sb, lines_b, False, "stress_b"))]
    output["removed_pixels_touched_by_supported_rays_a"] = _removed_probe_pixels(ca, sa, lines_a)
    output["removed_pixels_touched_by_supported_rays_b"] = _removed_probe_pixels(cb, sb, lines_b)
    valid_indices, gaps_clean, gaps_stress, gaps_added = [], [], [], []
    invalid_counts, invalid_arcs = Counter(), Counter()
    for i in range(len(anchors)):
        row = dict(clean_token_a=int(ids[i]), anchor_a_rc=anchors[i].tolist(), normal_a_rc=normals[i].tolist(),
            arc_weight_px=float(weights[i]), valid=False, invalid_reason=None,
            clean_gap_px=None, stress_gap_px=None, added_gap_px=None)
        boundaries, reasons = {}, []
        for values, in_frame, exiting, name in profiles:
            if not in_frame[i].all():
                position, why = None, "probe_outside_model_frame"
            else:
                position, why = _single_boundary(values[i], exiting=exiting)
            boundaries[name] = position
            if why:
                reasons.append(name + ":" + why)
        if not reasons:
            clean_gap = boundaries["clean_b"] - boundaries["clean_a"]
            stress_gap = boundaries["stress_b"] - boundaries["stress_a"]
            added = stress_gap - clean_gap
            row.update(clean_gap_px=clean_gap, stress_gap_px=stress_gap, added_gap_px=added,
                       boundary_offsets_px=boundaries)
            if added < -_NEGATIVE_TOLERANCE_PX:
                reasons.append("negative_added_gap_inconsistent_with_inward_subset")
                output["negative_added_gap_count"] += 1
            else:
                valid_indices.append(i)
                gaps_clean.append(clean_gap)
                gaps_stress.append(stress_gap)
                gaps_added.append(added)
                row["valid"] = True
        if reasons:
            row["invalid_reason"] = "|".join(reasons)
            # Reasons can overlap; their counts/arcs are not additive totals.
            for why in reasons:
                invalid_counts[why] += 1
                invalid_arcs[why] += float(weights[i])
        if include_samples:
            output["samples"].append(row)
    measured_weights = weights[np.asarray(valid_indices, np.int64)]
    measured_arc = float(measured_weights.sum())
    original_supported_arc = support["clean_supported_arc_px"]
    output["support"].update(measured_token_count=len(valid_indices), measured_arc_px=measured_arc,
        coverage_fraction=measured_arc / max(1e-12, original_supported_arc),
        invalid_token_count=len(anchors) - len(valid_indices), invalid_token_reasons=dict(invalid_counts),
        invalid_arc_px_by_reason=dict(invalid_arcs))
    if not valid_indices:
        output["invalid_reason"] = "no_unambiguous_line_measurements"
        return output
    clean_values, stress_values, added_values = (np.asarray(values, np.float64)
        for values in (gaps_clean, gaps_stress, gaps_added))
    damaged = added_values >= DAMAGE_ADDED_GAP_PX
    damaged_arc = float(measured_weights[damaged].sum())
    damaged_fraction = damaged_arc / original_supported_arc
    output.update(clean_gap=_weighted_statistics(clean_values, measured_weights),
        stress_gap=_weighted_statistics(stress_values, measured_weights),
        added_gap=_weighted_statistics(added_values, measured_weights),
        damaged_added_gap=_weighted_statistics(added_values[damaged], measured_weights[damaged]),
        damage_added_ge_1px_arc_px=damaged_arc, damage_added_ge_1px_arc_fraction=damaged_fraction,
        damage_fraction_of_measured_arc=damaged_arc / measured_arc,
        stress_positive_separation_arc_fraction=float(measured_weights[stress_values > 0].sum() / measured_arc),
        clean_negative_overlap_arc_fraction=float(measured_weights[clean_values < 0].sum() / measured_arc))
    if measured_arc < MINIMUM_MEASURED_ARC_PX:
        output["invalid_reason"] = "insufficient_measured_seam_arc"
        return output
    output.update(valid=True, invalid_reason=None,
        actual_seam_damaged=bool(damaged_fraction >= DAMAGE_CLEAN_SUPPORTED_ARC_FRACTION))
    return output


__all__ = ["SCHEMA", "seam_gap_diagnostics"]
