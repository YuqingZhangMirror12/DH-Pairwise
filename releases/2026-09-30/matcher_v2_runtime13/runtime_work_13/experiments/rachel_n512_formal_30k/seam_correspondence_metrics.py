"""Post-inference seam metrics against *sampled dataset correspondence targets*.

This reference is NOT a human-annotated full seam.  No real-data seam ground
truth is inferred.  Call only after selecting the model/decoder's edges without
targets; this module neither extracts candidates nor selects a mode.  All
distances are Euclidean pixels, in the supplied local row/column coordinates.
The fixed 3/5-pixel tolerances are measurement definitions, not fitted thresholds.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "sampled-reference-seam-metrics/1.0"
REFERENCE = "sampled dataset correspondence targets; NOT human-annotated full seam"
TOLERANCES_PX = (3.0, 5.0)
_METRICS = (
    "displacement_compatibility_precision", "reference_correspondence_recall",
    "reference_arc_coverage_a", "reference_arc_coverage_b", "reference_arc_coverage",
)
_SUPPORTS = (
    "displacement_compatible_edge_count", "displacement_evaluated_edge_count",
    "reference_recalled_edge_count", "reference_edge_count",
    "covered_arc_a_px", "reference_arc_a_px", "covered_arc_b_px", "reference_arc_b_px",
)


def _mask(value, size, name):
    if value is None:
        return np.ones(size, dtype=bool)
    mask = np.asarray(value)
    if mask.shape != (size,) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError(name + " must be a boolean/0-1 vector matching the contour")
    return mask.astype(bool, copy=False)


def _points(value, valid, name):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(name + " must have shape [N,2]")
    mask = _mask(valid, len(points), "valid_" + name)
    if not np.all(np.isfinite(points[mask])):
        raise ValueError(name + " active coordinates must be finite")
    return points, mask


def _edges(value, valid_a, valid_b, name):
    if value is None:
        raise ValueError(name + " must be an explicit edge list (empty is allowed)")
    edges = np.asarray(value)
    if edges.size == 0 and edges.shape in ((0,), (0, 2)):
        edges = np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError(name + " must have integer shape [K,2]")
    if len(edges) and (np.any(edges < 0) or np.any(edges[:, 0] >= len(valid_a))
                       or np.any(edges[:, 1] >= len(valid_b))):
        raise ValueError(name + " contains out-of-range indices")
    edges = edges.astype(np.int64, copy=False)
    active = valid_a[edges[:, 0]] & valid_b[edges[:, 1]]
    unique = np.unique(edges[active], axis=0)
    counts = {"input": int(len(edges)), "padding_excluded": int(np.sum(~active)),
              "duplicates_removed": int(np.sum(active) - len(unique)), "unique_valid": int(len(unique))}
    return unique, counts


def gt_edges_from_targets(target_a, target_b, valid_a=None, valid_b=None):
    """Convert reciprocal RachelBatch targets (-1 dustbin/-2 ignore) to [K,2].

    Padding sources are ignored.  Active non-dustbin targets must be reciprocal
    and reference an active partner; malformed targets raise, not silently
    become a smaller/easier reference.  No model inputs or targets are modified.
    """
    a, b = np.asarray(target_a), np.asarray(target_b)
    for targets in (a, b):
        if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer) or np.any(targets < -2):
            raise ValueError("targets must be integer vectors with -2/-1 or partner indices")
    va = _mask(valid_a, len(a), "valid_a") & (a != -2)
    vb = _mask(valid_b, len(b), "valid_b") & (b != -2)
    ia, ib = np.flatnonzero(va & (a >= 0)), np.flatnonzero(vb & (b >= 0))
    if np.any(a[ia] >= len(b)) or np.any(b[ib] >= len(a)):
        raise ValueError("targets contain out-of-range indices")
    if (not np.all(vb[a[ia]]) or not np.all(va[b[ib]])
            or not np.array_equal(b[a[ia]], ia) or not np.array_equal(a[b[ib]], ib)):
        raise ValueError("active targets must be reciprocal and reference active tokens")
    return np.column_stack((ia, a[ia])).astype(np.int64, copy=False)


def _arc_weights(points, valid):
    """Cyclic Voronoi support: each valid token owns half each adjacent edge."""
    weights = np.zeros(len(points), dtype=np.float64)
    indices = np.flatnonzero(valid)
    if len(indices) >= 2:
        contour = points[indices]
        weights[indices] = .5 * (np.linalg.norm(contour - np.roll(contour, 1, axis=0), axis=1)
                                 + np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1))
    return weights


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator is not None and denominator > 0 else None


def _reference_hits(a, b, predicted, reference):
    # Chunk reference rows to bound temporary memory.  Keep endpoint tests paired
    # along the SAME prediction column, never independently match the marginals.
    hits = {str(int(tol)): np.zeros(len(reference), dtype=bool) for tol in TOLERANCES_PX}
    if not len(predicted):
        return hits
    for start in range(0, len(reference), 128):
        edge = reference[start:start + 128]
        da2 = np.sum((a[edge[:, 0], None] - a[predicted[:, 0]][None]) ** 2, axis=2)
        db2 = np.sum((b[edge[:, 1], None] - b[predicted[:, 1]][None]) ** 2, axis=2)
        for tol in TOLERANCES_PX:
            hits[str(int(tol))][start:start + len(edge)] = np.any((da2 <= tol * tol) & (db2 <= tol * tol), axis=1)
    return hits


def measure_seam_correspondences(points_a_rc, points_b_rc, predicted_edges, gt_edges, *,
                                 valid_a=None, valid_b=None, gt_translation_a_to_b_rc=None,
                                 label=True, pose_valid=True):
    """Measure already-selected integer edges, returning JSON-ready diagnostics.

    Precision tests ``b[j]-a[i]`` against the known A-to-B translation.  Recall
    requires a prediction close to BOTH endpoints of the SAME reference edge;
    it is tolerant reference coverage, not a one-to-one matching assignment.
    Arc coverage counts unique recalled reference endpoints using their cyclic
    contour Voronoi lengths, never multiple overlapping-patch votes.  Predicted
    edges are unique, but multiple distinct edges may share an endpoint.

    ``pose_valid`` means PREDICTED decoder validity: invalid predictions emit no
    edges, retain valid positive reference denominators, and receive zero recall
    and coverage.  Empty emitted precision is undefined (None).  Negative pairs
    have no seam metric.  Missing translation only disables precision; an empty
    or missing reference only disables reference recall/coverage.  Zero-length
    reference arc support has undefined arc coverage.  A method with no defined
    selected edges should be recorded as unavailable by its caller, not passed
    as an empty prediction.  Nonfinite padding is allowed and ignored.
    """
    a, va = _points(points_a_rc, valid_a, "points_a_rc")
    b, vb = _points(points_b_rc, valid_b, "points_b_rc")
    if np.asarray(label).shape != () or label not in (False, True, 0, 1):
        raise ValueError("label must be a scalar binary label")
    if np.asarray(pose_valid).shape != () or pose_valid not in (False, True, 0, 1):
        raise ValueError("pose_valid must be scalar boolean/0-1")
    predicted, pred_counts = _edges(predicted_edges, va, vb, "predicted_edges")
    reference, gt_counts = _edges([] if gt_edges is None else gt_edges, va, vb, "gt_edges")
    issues = []
    if not len(reference):
        issues.append("missing_reference" if gt_edges is None else "empty_reference")
    translation = None
    if gt_translation_a_to_b_rc is None:
        issues.append("missing_translation_gt")
    else:
        value = np.asarray(gt_translation_a_to_b_rc, dtype=np.float64)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            issues.append("invalid_translation_gt")
        else:
            translation = value
    ref_available, translation_available = bool(label and len(reference)), bool(label and translation is not None)
    emitted = predicted if pose_valid else np.empty((0, 2), dtype=np.int64)
    prediction_status = "invalid_prediction" if not pose_valid else ("empty_predictions" if not len(emitted) else "ok")
    target_status = ("available" if ref_available and translation_available else
                     "reference_only" if ref_available else "translation_only" if translation_available else "unavailable")
    status = ("negative_label" if not label else "unavailable_target" if target_status == "unavailable" else
              prediction_status if prediction_status != "ok" else "partial_target" if issues else "ok")
    weights_a, weights_b = _arc_weights(a, va), _arc_weights(b, vb)
    ref_a, ref_b = np.unique(reference[:, 0]), np.unique(reference[:, 1])
    arc_a, arc_b = float(weights_a[ref_a].sum()), float(weights_b[ref_b].sum())
    hits = _reference_hits(a, b, emitted, reference) if ref_available else {}
    displacement_error = (np.linalg.norm(b[emitted[:, 1]] - a[emitted[:, 0]] - translation, axis=1)
                          if translation_available else None)
    by_tolerance = {}
    for tol in TOLERANCES_PX:
        key = str(int(tol))
        item = {name: None for name in _METRICS + _SUPPORTS}
        item["precision_status"] = "negative_label" if not label else "unavailable_translation_gt"
        item["reference_status"] = "negative_label" if not label else "unavailable_reference"
        if translation_available:
            compatible = int(np.sum(displacement_error <= tol))
            item.update(displacement_compatible_edge_count=compatible,
                        displacement_evaluated_edge_count=int(len(emitted)),
                        displacement_compatibility_precision=_ratio(compatible, len(emitted)),
                        precision_status="defined" if len(emitted) else prediction_status)
        if ref_available:
            recalled = reference[hits[key]]
            covered_a = float(weights_a[np.unique(recalled[:, 0])].sum())
            covered_b = float(weights_b[np.unique(recalled[:, 1])].sum())
            item.update(reference_recalled_edge_count=int(len(recalled)), reference_edge_count=int(len(reference)),
                        reference_correspondence_recall=_ratio(len(recalled), len(reference)),
                        covered_arc_a_px=covered_a, reference_arc_a_px=arc_a,
                        covered_arc_b_px=covered_b, reference_arc_b_px=arc_b,
                        reference_arc_coverage_a=_ratio(covered_a, arc_a), reference_arc_coverage_b=_ratio(covered_b, arc_b),
                        reference_arc_coverage=_ratio(covered_a + covered_b, arc_a + arc_b),
                        reference_status="defined" if arc_a > 0 and arc_b > 0 else "defined_recall_degenerate_arc")
        by_tolerance[key] = item
    return {"schema_version": SCHEMA_VERSION, "reference": REFERENCE, "tolerances_px": list(TOLERANCES_PX),
            "status": status, "target_status": target_status, "target_issues": issues,
            "prediction_status": prediction_status, "label": bool(label), "pose_valid": bool(pose_valid),
            "counts": {"predicted_edges": pred_counts, "gt_edges": gt_counts, "emitted_edges": int(len(emitted)),
                       "reference_unique_endpoints_a": int(len(ref_a)), "reference_unique_endpoints_b": int(len(ref_b))},
            "by_tolerance_px": by_tolerance}


def summarize_seam_correspondences(records: Iterable[Mapping]):
    """Aggregate pair macros and additive supports; never fit any threshold.

    Pass measurement records for one model/decoder/population.  Macro recall
    includes empty/invalid predictions with valid references.  Each macro has a
    contributing-pair count; support-weighted ratios are separately labelled.
    None is retained for unavailable/empty aggregate denominators.
    """
    rows = list(records)
    for row in rows:
        if row.get("schema_version") != SCHEMA_VERSION or row.get("tolerances_px") != list(TOLERANCES_PX):
            raise ValueError("records must use this seam metric schema and fixed tolerances")
    result = {"schema_version": SCHEMA_VERSION, "reference": REFERENCE, "tolerances_px": list(TOLERANCES_PX),
              "pair_count": len(rows), "positive_pair_count": sum(bool(row["label"]) for row in rows),
              "negative_pair_count": sum(not row["label"] for row in rows),
              "status_counts": dict(Counter(row["status"] for row in rows)),
              "prediction_status_counts": dict(Counter(row["prediction_status"] for row in rows)),
              "target_status_counts": dict(Counter(row["target_status"] for row in rows)),
              "thresholds_fitted": False, "by_tolerance_px": {}}
    result["count_sums"] = {
        name: sum(row["counts"][name] for row in rows)
        for name in ("emitted_edges", "reference_unique_endpoints_a", "reference_unique_endpoints_b")
    }
    for edge_kind in ("predicted_edges", "gt_edges"):
        result["count_sums"][edge_kind] = {
            name: sum(row["counts"][edge_kind][name] for row in rows)
            for name in ("input", "padding_excluded", "duplicates_removed", "unique_valid")
        }
    for tol in TOLERANCES_PX:
        key = str(int(tol))
        items = [row["by_tolerance_px"][key] for row in rows]
        macro, macro_counts, sums, support_counts = {}, {}, {}, {}
        for metric in _METRICS:
            values = [item[metric] for item in items if item[metric] is not None]
            macro[metric] = float(np.mean(values)) if values else None
            macro_counts[metric] = len(values)
        for support in _SUPPORTS:
            values = [item[support] for item in items if item[support] is not None]
            sums[support] = sum(values) if values else None
            support_counts[support] = len(values)
        micro = {
            "displacement_compatibility_precision": _ratio(sums["displacement_compatible_edge_count"], sums["displacement_evaluated_edge_count"]),
            "reference_correspondence_recall": _ratio(sums["reference_recalled_edge_count"], sums["reference_edge_count"]),
            "reference_arc_coverage_a": _ratio(sums["covered_arc_a_px"], sums["reference_arc_a_px"]),
            "reference_arc_coverage_b": _ratio(sums["covered_arc_b_px"], sums["reference_arc_b_px"]),
            "reference_arc_coverage": None,
        }
        if sums["reference_arc_a_px"] is not None:
            micro["reference_arc_coverage"] = _ratio(sums["covered_arc_a_px"] + sums["covered_arc_b_px"],
                                                      sums["reference_arc_a_px"] + sums["reference_arc_b_px"])
        result["by_tolerance_px"][key] = {"pair_macro": macro, "pair_macro_counts": macro_counts,
                                        "support_sums": sums, "support_pair_counts": support_counts,
                                        "support_weighted": micro}
    return result
