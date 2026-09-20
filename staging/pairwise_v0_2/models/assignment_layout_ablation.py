"""CPU-only correspondence selection ablations with the unchanged v2 geometry.

Consume one predicted Sinkhorn real plan Q and, for partial Hungarian, its
per-token dustbin masses. No labels, target translations, masks, normals or
pair scores are used. Hungarian selects a maximum-weight hard one-to-one
matching; it does NOT estimate a pose. Every strategy subsequently uses the
existing weighted displacement-mode consensus and inlier refinement.

``hungarian_full`` maximizes sum(Q_ij), forcing min(Na, Nb) real edges, even
on an unmatchable pair. ``hungarian_partial`` instead maximizes

    sum_matched Q_ij + sum_unmatched_A uA_i + sum_unmatched_B uB_j.

This is an explicitly declared RAW-MASS objective, not a product/log-likelihood
objective and not Hungarian on the original affinity logits. The all-unmatched
baseline is feasible. Each chosen real edge replaces TWO unmatched endpoints,
so its gain is Q_ij - uA_i - uB_j. Zero-gain ties prefer being unmatched.
The OT dustbin corner is aggregate bookkeeping, not a per-edge bonus here.

``hungarian_partial_log`` separately maximizes the sum of log Q for selected
edges and log dustbin mass for unmatched endpoints. It uses these supplied
masses directly, without renormalizing them into another probability matrix.
Zero mass forbids that edge: there is no epsilon floor or fitted threshold.
If no positive-product assignment exists, the result explicitly abstains with
``infeasible_log_objective``. All geometry still uses ORIGINAL Q weights.

Top1/2/5 all use row/column UNION, the same candidate budget and the same geometry;
Top1 here is deliberately not the old reciprocal-top1 (intersection) control.
Hard assignments are optimized before applying the shared geometry budget. The
reported pre-budget matching and post-budget geometric support are distinct.
SciPy is deterministic for a fixed input; tied optima need not select identical
edges across SciPy versions or after swapping the two fragment orderings.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .translation_layout import (
    TranslationLayoutConfig,
    TranslationLayoutResult,
    _validate_inputs,
    estimate_translation_layout,
)


@dataclass(frozen=True)
class AssignmentLayoutConfig:
    strategy: str = "topk_union"
    top_k: int = 2
    max_candidates: int = 512
    inlier_radius_px: float = 10.0
    min_inliers: int = 3
    refinement: str = "weighted_mean"
    refinement_iterations: int = 3

    def __post_init__(self):
        if not isinstance(self.strategy, str) or self.strategy not in (
            "topk_union", "hungarian_full", "hungarian_partial", "hungarian_partial_log",
        ):
            raise ValueError("strategy must be topk_union, hungarian_full, hungarian_partial or hungarian_partial_log")
        self.geometry_config()  # Reuse exactly the old geometry's value checks.

    def geometry_config(self, *, top_k: Optional[int] = None) -> TranslationLayoutConfig:
        return TranslationLayoutConfig(
            correspondence_mode="topk_union", decoder="mode_consensus",
            score_mode="confidence", top_k=self.top_k if top_k is None else top_k,
            max_candidates=self.max_candidates, inlier_radius_px=self.inlier_radius_px,
            min_inliers=self.min_inliers, refinement=self.refinement,
            refinement_iterations=self.refinement_iterations,
        )


@dataclass(frozen=True)
class AssignmentLayoutResult(TranslationLayoutResult):
    """Old layout fields, plus JSON-ready correspondence-objective accounting.

    candidate_indices/inlier_mask retain their old meaning: the geometrically
    evaluated edges AFTER the common budget, not all forced Hungarian edges.
    """

    assignment_diagnostics: dict


def assignment_ablation_configs():
    """Predeclared six-way comparison; only k changes across the Top-k arm trio."""
    common = AssignmentLayoutConfig()
    return {
        "top1_mode": replace(common, top_k=1),
        "top2_mode": common,
        "top5_mode": replace(common, top_k=5),
        "hungarian_full_mode": replace(common, strategy="hungarian_full"),
        "hungarian_partial_mode": replace(common, strategy="hungarian_partial"),
        "hungarian_partial_log_mode": replace(common, strategy="hungarian_partial_log"),
    }


def _dustbin(value, valid, name):
    if value is None:
        raise ValueError(name + " is required for partial Hungarian")
    value = np.asarray(value, dtype=np.float64)
    if value.shape != valid.shape:
        raise ValueError(name + " must be a vector aligned with its contour tokens")
    active = value[valid]
    if not np.all(np.isfinite(active)) or np.any(active < 0):
        raise ValueError(name + " must be finite non-negative mass on valid tokens")
    return active


def _sum_or_none(values):
    # Ordinary OT masses are <= 1. Keep diagnostics JSON-safe even if a caller
    # supplies very large finite weights whose sum cannot be a float64 value.
    total = np.sum(np.asarray(values, dtype=np.longdouble), dtype=np.longdouble)
    return float(total) if np.isfinite(total) and abs(total) <= np.finfo(np.float64).max else None


def _hard_assignment(q, strategy, ua=None, ub=None):
    na, nb = q.shape
    empty = np.empty(0, dtype=np.int64)
    if not na or not nb:
        return empty, empty
    # Common rescaling leaves the linear objective unchanged and prevents
    # overflow while forming Q - uA - uB. Original masses weight geometry.
    maximum = float(q.max())
    if strategy == "hungarian_partial":
        maximum = max(maximum, float(ua.max()), float(ub.max()))
    scale = maximum if maximum > 0 else 1.0
    if strategy == "hungarian_full":
        return linear_sum_assignment(-(q / scale))
    gain = q / scale - (ua / scale)[:, None] - (ub / scale)[None, :]
    # N_A dummy columns let EVERY A remain unmatched. An unused real B retains
    # its baseline uB mass. This rectangular construction is equivalent to an
    # (Na+Nb)^2 private-dummy construction, without the dummy-dummy block.
    cost = np.zeros((na, nb + na), dtype=np.float64)
    cost[:, :nb] = -gain
    rows, columns = linear_sum_assignment(cost)
    real = columns < nb
    rows, columns = rows[real], columns[real]
    improving = gain[rows, columns] > 0.0
    return rows[improving], columns[improving]


def _log_values(values):
    result = np.full(values.shape, -np.inf, dtype=np.float64)
    np.log(values, out=result, where=values > 0.0)
    return result


def _log_hard_assignment(q, ua, ub):
    """Exact private-dummy log objective, including zero-mass impossibilities."""
    na, nb = q.shape
    empty = np.empty(0, dtype=np.int64)
    if na + nb == 0:
        return empty, empty, True
    lq, la, lb = _log_values(q), _log_values(ua), _log_values(ub)
    cost = np.full((na + nb, na + nb), np.inf, dtype=np.float64)
    cost[:na, :nb] = -lq
    cost[np.arange(na), nb + np.arange(na)] = -la
    cost[na + np.arange(nb), np.arange(nb)] = -lb
    cost[na:, nb:] = 0.0
    try:
        rows, columns = linear_sum_assignment(cost)
    except ValueError as error:
        # Input validation has already excluded NaNs. Infinite costs represent
        # forbidden zero-mass choices, so infeasibility is an explicit abstention.
        if "infeasible" not in str(error).lower():
            raise
        return empty, empty, False
    real = (rows < na) & (columns < nb)
    rows, columns = rows[real], columns[real]
    # If leaving BOTH endpoints unmatched is possible, an exact zero-gain tie
    # may be dropped without changing the optimum. No tolerance is fitted.
    finite_baseline = (ua[rows] > 0) & (ub[columns] > 0)
    keep = ~finite_baseline | (lq[rows, columns] > la[rows] + lb[columns])
    return rows[keep], columns[keep], True


def estimate_assignment_layout(
    points_a_rc, points_b_rc, confidence, valid_a=None, valid_b=None, *,
    unmatched_a=None, unmatched_b=None,
    config: AssignmentLayoutConfig = AssignmentLayoutConfig(),
) -> AssignmentLayoutResult:
    """Decode one predicted pair without changing its classifier or old decoder.

    Q has [Na,Nb] shape without dustbins. unmatched_a is the model's
    ``unmatched_a`` / ``transport.dustbin_col``; unmatched_b is ``dustbin_row``.
    Both are REQUIRED for either partial Hungarian, and unused by the other strategies.
    Only valid-token coordinates and masses are checked/used; padding is ignored.
    A negative pair is not special-cased with a label: predicted dustbin evidence
    alone may select no real matches. No negative-label information is accepted.
    """
    if not isinstance(config, AssignmentLayoutConfig):
        raise TypeError("config must be AssignmentLayoutConfig")
    a, b, score, va, vb = _validate_inputs(points_a_rc, points_b_rc, confidence, valid_a, valid_b)
    ai, bi = np.flatnonzero(va), np.flatnonzero(vb)
    q = score[np.ix_(ai, bi)]
    if np.any(q < 0):
        raise ValueError("confidence must be non-negative predicted transport mass")
    diagnostics = {
        "strategy": config.strategy, "top_k": config.top_k if config.strategy == "topk_union" else None,
        "valid_a_count": len(ai), "valid_b_count": len(bi),
        "max_geometry_candidates": config.max_candidates,
        "one_to_one_selection": config.strategy != "topk_union",
        "pair_score_modified": False, "rotation_estimated": False,
    }
    if config.strategy == "topk_union":
        # Direct call: the new Top2 matches the existing Top2 result exactly,
        # including cap sorting/ties, normalized weights and runner-up accounting.
        layout = estimate_translation_layout(a, b, score, va, vb, config=config.geometry_config())
        diagnostics.update(
            objective="row_column_topk_union", selected_edge_count=None,
            selected_zero_weight_count=0, unmatched_a_count=None, unmatched_b_count=None,
            raw_objective=None, all_unmatched_baseline_raw_weight=None,
            geometry_candidate_count=layout.candidate_count,
        )
        return AssignmentLayoutResult(**vars(layout), assignment_diagnostics=diagnostics)

    partial = config.strategy in ("hungarian_partial", "hungarian_partial_log")
    logarithmic = config.strategy == "hungarian_partial_log"
    ua = _dustbin(unmatched_a, va, "unmatched_a") if partial else None
    ub = _dustbin(unmatched_b, vb, "unmatched_b") if partial else None
    if logarithmic:
        rows, columns, feasible = _log_hard_assignment(q, ua, ub)
    else:
        rows, columns = _hard_assignment(q, config.strategy, ua, ub)
        feasible = True
    selected_mass = q[rows, columns]
    unmatched_rows, unmatched_columns = np.ones(len(ai), dtype=bool), np.ones(len(bi), dtype=bool)
    unmatched_rows[rows], unmatched_columns[columns] = False, False
    objective_parts = [selected_mass]
    if partial:
        objective_parts.extend((ua[unmatched_rows], ub[unmatched_columns]))
    # Sparsify only the selected edges. Because this is one-to-one, Top1 union
    # on this matrix retains every positive chosen edge before the old cap.
    # Zero-weight edges forced by naive full matching do NOT become unit votes.
    sparse = np.zeros_like(score)
    sparse[ai[rows], bi[columns]] = selected_mass
    layout = estimate_translation_layout(a, b, sparse, va, vb, config=config.geometry_config(top_k=1))
    if not feasible:
        layout = replace(layout, reason="infeasible_log_objective")
    diagnostics.update(
        objective=("sum_real_plus_unmatched_log_mass" if logarithmic else
                   "sum_real_plus_unmatched_raw_mass" if partial else "sum_real_raw_mass_full_matching"),
        objective_feasible=feasible,
        selected_edge_count=int(len(rows)),
        selected_positive_weight_count=int(np.count_nonzero(selected_mass > 0)),
        selected_zero_weight_count=int(np.count_nonzero(selected_mass == 0)),
        selected_raw_weight_sum=_sum_or_none(selected_mass),
        unmatched_a_count=int(unmatched_rows.sum()), unmatched_b_count=int(unmatched_columns.sum()),
        raw_objective=_sum_or_none(np.concatenate(objective_parts)) if feasible else None,
        all_unmatched_baseline_raw_weight=_sum_or_none(np.concatenate((ua, ub))) if partial else None,
        log_objective=_sum_or_none(_log_values(np.concatenate(objective_parts))) if logarithmic and feasible else None,
        all_unmatched_baseline_log_weight=_sum_or_none(_log_values(np.concatenate((ua, ub)))) if logarithmic else None,
        all_unmatched_baseline_feasible=bool(np.all(ua > 0) and np.all(ub > 0)) if logarithmic else partial,
        all_unmatched_selected=bool(partial and feasible and not len(rows)),
        geometry_candidate_count=layout.candidate_count,
        geometry_positive_edges_dropped_by_budget=int(np.count_nonzero(selected_mass > 0)) - layout.candidate_count,
        partial_zero_gain_ties="prefer_unmatched" if partial else None,
        log_zero_mass_policy="forbidden_edges_no_epsilon" if logarithmic else None,
        dustbin_corner_used=False,
    )
    return AssignmentLayoutResult(**vars(layout), assignment_diagnostics=diagnostics)


__all__ = [
    "AssignmentLayoutConfig", "AssignmentLayoutResult", "assignment_ablation_configs",
    "estimate_assignment_layout",
]
