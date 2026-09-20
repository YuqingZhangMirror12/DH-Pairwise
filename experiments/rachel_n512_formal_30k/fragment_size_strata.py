"""Fixed fragment-area ratio strata; no per-stratum fitting or new cutpoints.

Areas count nonzero pixels in the supplied filled binary masks, not contour
length or bounding-box area. The ratio is smaller area / larger area. Metadata
is flattened into each existing prediction row with ``row.update(metadata)``.
"""
from __future__ import annotations

import numpy as np

SIZE_RATIO_STRATA = ("lt_0_25", "0_25_to_lt_0_5", "ge_0_5")
INVALID_SIZE_STRATUM = "invalid_zero_area"


def _mask_area(raw_mask):
    mask = np.asarray(raw_mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError("each filled binary mask must be [H,W] or [1,H,W]")
    if not np.isfinite(mask).all():
        raise ValueError("mask values must be finite")
    return int(np.count_nonzero(mask))


def pair_size_metadata(mask_a, mask_b):
    """Return target-independent areas and the fixed smaller/larger ratio bin.

    Either empty fragment makes the ratio invalid; it is not placed in the
    most-unequal-size bin. Boolean, 0/1, and 0/255 binary masks are equivalent.
    No resizing, threshold fitting, hole filling, or GT access is performed.
    """
    area_a, area_b = _mask_area(mask_a), _mask_area(mask_b)
    smaller, larger = min(area_a, area_b), max(area_a, area_b)
    valid = smaller > 0
    ratio = smaller / larger if valid else None
    stratum = INVALID_SIZE_STRATUM
    if valid:
        stratum = SIZE_RATIO_STRATA[0] if ratio < .25 else (SIZE_RATIO_STRATA[1] if ratio < .5 else SIZE_RATIO_STRATA[2])
    return {"area_a_px": area_a, "area_b_px": area_b, "smaller_area_px": smaller,
            "area_ratio": ratio, "size_ratio_stratum": stratum, "size_metadata_valid": valid}


def _undefined_classification_metrics(metrics, positive_count):
    """Expose undefined denominators instead of the base helper's zero fill."""
    if positive_count == 0:
        metrics["auroc"] = None
        metrics["auprc"] = None
    elif metrics["tn"] + metrics["fp"] == 0:
        metrics["auroc"] = None
        metrics["auprc"] = 1.0  # Every ranked example is a positive.
    if metrics["tp"] + metrics["fp"] == 0:
        metrics["precision"] = None
    if metrics["tp"] + metrics["fn"] == 0:
        metrics["recall"] = None
    if 2 * metrics["tp"] + metrics["fp"] + metrics["fn"] == 0:
        metrics["f1"] = None


def _defined_group_summary(rows, threshold, branch_thresholds):
    # Keep the metadata path NumPy-only; the existing evaluation stack is only
    # imported when its common metrics are actually requested.
    from experiments.rachel_n512_formal_30k.run_layout_decoder_experiment import summarize

    summary = summarize(rows, threshold, branch_thresholds)
    positive = summary["positive_count"]
    summary["negative_count"] = len(rows) - positive
    summary["metrics_status"] = "computed"
    for branch in summary["classification"].values():
        for metrics in branch.values():
            _undefined_classification_metrics(metrics, positive)
    for metrics in summary["layout"].values():
        if positive == 0:
            metrics["positive_pose_coverage"] = None
            metrics["recall"] = {tolerance: None for tolerance in metrics["recall"]}
        for assembly in metrics["assembly"].values():
            if assembly["tp"] + assembly["fp"] == 0:
                assembly["precision"] = None
            if assembly["tp"] + assembly["fn"] == 0:
                assembly["recall"] = None
            if 2 * assembly["tp"] + assembly["fp"] + assembly["fn"] == 0:
                assembly["f1"] = None
    return summary


def summarize_size_strata(rows, threshold, branch_thresholds):
    """Compute existing pairability/layout metrics within the fixed area bins.

    All thresholds are supplied by the parent evaluation and reused unchanged.
    Empty groups and invalid-mask groups have no scores. Single-class groups
    retain accuracy/counts and any defined metrics, but never claim an AUROC.
    """
    if not np.isfinite(threshold):
        raise ValueError("the frozen fused threshold must be finite")
    if branch_thresholds is not None and (
            set(branch_thresholds) != {"coarse", "local", "fused"}
            or not all(np.isfinite(value) for value in branch_thresholds.values())):
        raise ValueError("branch thresholds must be the three finite frozen values")
    grouped = {key: [] for key in SIZE_RATIO_STRATA + (INVALID_SIZE_STRATUM,)}
    for row in rows:
        key = row.get("size_ratio_stratum")
        if key not in grouped:
            raise ValueError("each row needs a known size_ratio_stratum from pair_size_metadata")
        grouped[key].append(row)
    summaries = {}
    for key, group in grouped.items():
        if group and key != INVALID_SIZE_STRATUM:
            summaries[key] = _defined_group_summary(group, threshold, branch_thresholds)
        else:
            positive = sum(bool(row["label"]) for row in group)
            summaries[key] = {"sample_count": len(group), "positive_count": positive,
                "negative_count": len(group) - positive, "classification": None, "layout": None,
                "metrics_status": "invalid_input" if group else "empty"}
    return {"area_ratio_definition": "min(area_a_px,area_b_px)/max(area_a_px,area_b_px)",
        "fixed_bin_edges": [.25, .5], "thresholds_refit": False,
        "frozen_fused_threshold": float(threshold),
        "branch_validation_thresholds": dict(branch_thresholds) if branch_thresholds is not None else None,
        "groups": summaries}


__all__ = ["pair_size_metadata", "summarize_size_strata", "SIZE_RATIO_STRATA", "INVALID_SIZE_STRATUM"]
