"""GT-assisted validation diagnostics for evidence lost by Top2/cap512.

This is NOT inference: ground truth defines the neighborhoods being measured.
Support near GT, or its weighted mean, does not show that a target-blind method
can find that neighborhood.  No decoder is run or selected and no threshold is
fitted.  All computations use NumPy float64 and the original confidence scale.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


TOLERANCES_PX = (3.0, 5.0, 10.0)
SCHEMA_VERSION = "candidate-support-diagnostics/1.0"


def _translation_helpers():
    # Reuse the exact decoder helper without importing models/__init__.py,
    # which eagerly imports torch.  This diagnostic needs only NumPy.
    name = "_candidate_diagnostics_translation_layout"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "staging/pairwise_v0_2/models/translation_layout.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolves its defining module.
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def _translation(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(name + " must be a finite [2] row/column translation")
    return result


def _mass(weights):
    result = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(result):
        raise ValueError("confidence mass must be representable in float64")
    return result


def _fraction(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _neighborhoods(indices, delta, weights, center, gt_translation, total_mass):
    distances = np.linalg.norm(delta - center, axis=1)
    result = {}
    for radius in TOLERANCES_PX:
        selected = distances <= radius
        support_indices = indices[selected]
        support_mass = _mass(weights[selected])
        mean = None
        mean_gt_error = None
        mean_center_error = None
        if len(support_indices):
            mean = np.sum(delta[selected] * (weights[selected] / support_mass)[:, None], axis=0)
            mean_gt_error = float(np.linalg.norm(mean - gt_translation))
            mean_center_error = float(np.linalg.norm(mean - center))
        result[str(int(radius))] = {
            "status": "supported" if len(support_indices) else "no_support",
            "edge_count": int(len(support_indices)),
            "edge_count_fraction": _fraction(len(support_indices), len(indices)),
            "confidence_mass": support_mass,
            "confidence_mass_fraction": _fraction(support_mass, total_mass),
            "unique_endpoints_a": int(len(np.unique(support_indices[:, 0]))),
            "unique_endpoints_b": int(len(np.unique(support_indices[:, 1]))),
            "weighted_mean_translation_a_to_b_rc": None if mean is None else mean.tolist(),
            "weighted_mean_error_to_gt_px": mean_gt_error,
            "weighted_mean_error_to_neighborhood_center_px": mean_center_error,
        }
    return result


def diagnose_candidate_support(points_a_rc, points_b_rc, confidence, valid_a=None, valid_b=None, *,
                               gt_translation_a_to_b_rc, baseline_translation_a_to_b_rc=None):
    """Compare all positive edges, untruncated Top2 union, and Top2 cap512.

    ``confidence`` is [N,M], without dustbins.  Boolean masks may contain padding
    anywhere; padded values (including NaN) are ignored.  Active confidence must
    be finite and nonnegative.  GT and optional baseline use ``b[j]-a[i]`` sign.

    ``sets[name]`` reports total edge count, original confidence mass and unique
    endpoint counts. ``gt_neighborhoods['3'|'5'|'10']`` reports support within the
    fixed Euclidean-pixel radius, fractions relative to that candidate set, and
    the support's weighted-mean translation/error.  Baseline neighborhoods use
    the same statistics centered at the provided prediction, without changing
    GT neighborhoods or choosing between them.  Missing support has mean/error
    None.  Empty-set fractions are None, not fabricated zero/one scores.

    Top2 extraction and tie/cap behavior are exactly ``translation_layout``'s
    ``_candidates``. Its normalized weights are deliberately discarded; masses
    are retrieved from the original matrix.  No GT is passed into extraction.
    """
    helper = _translation_helpers()
    a, b, raw, va, vb = helper._validate_inputs(points_a_rc, points_b_rc, confidence, valid_a, valid_b)
    if np.any(raw[np.ix_(va, vb)] < 0):
        raise ValueError("active confidence must be nonnegative")
    gt = _translation(gt_translation_a_to_b_rc, "gt_translation_a_to_b_rc")
    baseline = (None if baseline_translation_a_to_b_rc is None else
                _translation(baseline_translation_a_to_b_rc, "baseline_translation_a_to_b_rc"))
    config = helper.TranslationLayoutConfig(correspondence_mode="topk_union", top_k=2,
                                           max_candidates=512, min_inliers=3, inlier_radius_px=10.)
    # A Top2 union cannot contain more edges than the active Cartesian product.
    uncapped = helper.TranslationLayoutConfig(correspondence_mode="topk_union", top_k=2,
        max_candidates=max(1, int(va.sum()) * int(vb.sum())), min_inliers=3, inlier_radius_px=10.)
    all_indices = np.argwhere(va[:, None] & vb[None, :] & (raw > 0)).astype(np.int64, copy=False)
    uncapped_indices, _, _ = helper._candidates(a, b, raw, va, vb, uncapped)
    capped_indices, _, _ = helper._candidates(a, b, raw, va, vb, config)
    collections = {"all_positive": all_indices, "top2_uncapped": uncapped_indices,
                   "top2_cap512": capped_indices}
    sets = {}
    for name, indices in collections.items():
        weights = raw[indices[:, 0], indices[:, 1]]
        delta = b[indices[:, 1]] - a[indices[:, 0]]
        if not np.all(np.isfinite(delta)):
            raise ValueError("active candidate displacements must be finite")
        mass = _mass(weights)
        sets[name] = {
            "status": "nonempty" if len(indices) else "empty_set",
            "edge_count": int(len(indices)), "confidence_mass": mass,
            "unique_endpoints_a": int(len(np.unique(indices[:, 0]))),
            "unique_endpoints_b": int(len(np.unique(indices[:, 1]))),
            "gt_neighborhoods": _neighborhoods(indices, delta, weights, gt, gt, mass),
            "baseline_neighborhoods": (None if baseline is None else
                                        _neighborhoods(indices, delta, weights, baseline, gt, mass)),
        }
    return {
        "schema_version": SCHEMA_VERSION, "diagnostic_uses_gt": True,
        "not_target_blind_recovery_evidence": True, "decoder_run_or_selected": False,
        "thresholds_fitted": False, "tolerances_px": list(TOLERANCES_PX),
        "confidence_mass_scale": "original input confidence, without candidate normalization",
        "candidate_config": {"correspondence_mode": "topk_union", "score_mode": "confidence",
                             "top_k": 2, "max_candidates": 512},
        "gt_translation_a_to_b_rc": gt.tolist(),
        "baseline_translation_a_to_b_rc": None if baseline is None else baseline.tolist(),
        "valid_points_a": int(va.sum()), "valid_points_b": int(vb.sum()), "sets": sets,
    }
