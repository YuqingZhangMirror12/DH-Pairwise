"""Deterministic, translation-only layout from a learned correspondence matrix.

This CPU decoder consumes predicted evidence only.  It does not alter the
pairability classifier or estimate rotation.  ``t_a_to_b_rc`` maps a point in
A's coordinate frame to B's: ``point_b = point_a + t_a_to_b_rc``.  To place B
on A's canvas, use the opposite translation.

Unlike the original dense transport mean, mode consensus starts at the best
supported displacement hypothesis.  ``median_cauchy`` is a separate sparse
weighted-median / Cauchy-IRLS ablation inspired by the ShreddingNet adapter.
No dustbin probability threshold is applied to either decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TranslationLayoutConfig:
    correspondence_mode: str = "reciprocal_top1"
    decoder: str = "mode_consensus"
    score_mode: str = "confidence"
    top_k: int = 2
    max_candidates: int = 512
    inlier_radius_px: float = 10.0
    min_inliers: int = 3
    cauchy_iterations: int = 20
    refinement: str = "weighted_mean"
    refinement_iterations: int = 3
    affinity_temperature: float = 1.0

    def __post_init__(self) -> None:
        choices = {
            "correspondence_mode": ("reciprocal_top1", "topk_union"),
            "decoder": ("mode_consensus", "median_cauchy"),
            "score_mode": ("confidence", "dual_softmax"),
            "refinement": ("weighted_mean", "weighted_median"),
        }
        for name, values in choices.items():
            if getattr(self, name) not in values:
                raise ValueError("{} must be one of {}".format(name, values))
        for name in ("top_k", "max_candidates", "min_inliers",
                     "cauchy_iterations", "refinement_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        for name in ("inlier_radius_px", "affinity_temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")


@dataclass(frozen=True)
class TranslationLayoutResult:
    t_a_to_b_rc: np.ndarray
    valid: bool
    candidate_count: int
    inlier_count: int
    inlier_fraction: float
    weighted_inlier_fraction: float
    residual_px: Optional[float]
    runner_up_support_ratio: float
    support_weight: float
    runner_up_support_weight: float
    candidate_indices: np.ndarray
    inlier_mask: np.ndarray
    reason: str


def _validate_inputs(
    points_a_rc: np.ndarray,
    points_b_rc: np.ndarray,
    confidence: np.ndarray,
    valid_a: Optional[np.ndarray],
    valid_b: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(points_a_rc, dtype=np.float64)
    b = np.asarray(points_b_rc, dtype=np.float64)
    score = np.asarray(confidence, dtype=np.float64)
    if a.ndim != 2 or a.shape[1:] != (2,) or b.ndim != 2 or b.shape[1:] != (2,):
        raise ValueError("points must have [N,2] and [M,2] shapes")
    if score.shape != (len(a), len(b)):
        raise ValueError("confidence must have [N,M] shape, excluding dustbins")
    masks = []
    for points, mask, name in ((a, valid_a, "valid_a"), (b, valid_b, "valid_b")):
        value = np.ones(len(points), dtype=np.bool_) if mask is None else np.asarray(mask)
        if value.dtype != np.bool_ or value.shape != (len(points),):
            raise ValueError(name + " must be a Boolean vector aligned with points")
        if not np.all(np.isfinite(points[value])):
            raise ValueError("valid point coordinates must be finite")
        masks.append(value)
    va, vb = masks
    if not np.all(np.isfinite(score[np.ix_(va, vb)])):
        raise ValueError("confidence on valid point pairs must be finite")
    return a, b, score, va, vb


def _scores(raw: np.ndarray, config: TranslationLayoutConfig) -> np.ndarray:
    if config.score_mode == "confidence":
        if np.any(raw < 0):
            raise ValueError("confidence must be non-negative; use dual_softmax for logits")
        return raw
    logits = raw / config.affinity_temperature
    if not np.all(np.isfinite(logits)):
        raise ValueError("temperature-scaled affinities must be finite")
    row_exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    col_exp = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    return ((row_exp / row_exp.sum(axis=1, keepdims=True))
            * (col_exp / col_exp.sum(axis=0, keepdims=True)))


def _edge_key(a: np.ndarray, b: np.ndarray, i: int, j: int, weight: float) -> tuple:
    # Sorting by unordered endpoint identity preserves A/B exchange, including
    # confidence ties.  Indistinguishable edges at the cap are dropped together.
    first = (float(a[i, 0]), float(a[i, 1]), int(i))
    second = (float(b[j, 0]), float(b[j, 1]), int(j))
    return (-float(weight),) + min(first, second) + max(first, second)


def _candidates(a, b, raw, va, vb, config):
    ai, bi = np.flatnonzero(va), np.flatnonzero(vb)
    if not len(ai) or not len(bi):
        return np.empty((0, 2), dtype=np.int64), np.empty(0), []
    score = _scores(raw[np.ix_(ai, bi)], config)
    selected = np.zeros(score.shape, dtype=np.bool_)
    if config.correspondence_mode == "reciprocal_top1":
        row_best = np.argmax(score, axis=1)
        col_best = np.argmax(score, axis=0)
        rows = np.arange(len(ai))
        keep = col_best[row_best] == rows
        selected[rows[keep], row_best[keep]] = True
    else:
        row_best = np.argsort(-score, axis=1, kind="stable")[:, :config.top_k]
        col_best = np.argsort(-score, axis=0, kind="stable")[:config.top_k, :]
        selected[np.arange(len(ai))[:, None], row_best] = True
        selected[col_best, np.arange(len(bi))[None, :]] = True
    rows, cols = np.where(selected & (score > 0.0))
    edges = [(int(ai[i]), int(bi[j]), float(score[i, j])) for i, j in zip(rows, cols)]
    ordered = sorted((_edge_key(a, b, i, j, w), i, j, w) for i, j, w in edges)
    end = min(len(ordered), config.max_candidates)
    if end < len(ordered):
        while end > 0 and ordered[end - 1][0] == ordered[end][0]:
            end -= 1
    ordered = ordered[:end]
    indices = np.asarray([(i, j) for _, i, j, _ in ordered], dtype=np.int64).reshape(-1, 2)
    weight = np.asarray([w for _, _, _, w in ordered], dtype=np.float64)
    if len(weight):
        weight /= weight.max()  # Support diagnostics use this normalized scale.
    return indices, weight, [key for key, _, _, _ in ordered]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    half = 0.5 * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, half, side="left")), len(values) - 1)
    # An exact half-mass interval has no preferred endpoint. Its midpoint makes
    # the estimate odd under a sign flip (and hence A/B exchange invariant).
    if index + 1 < len(values) and math.isclose(cumulative[index], half, rel_tol=1e-14):
        return float((values[index] + values[index + 1]) * 0.5)
    return float(values[index])


def _center(delta: np.ndarray, weight: np.ndarray, method: str) -> np.ndarray:
    if method == "weighted_mean":
        return (delta * weight[:, None]).sum(axis=0) / weight.sum()
    return np.asarray([_weighted_median(delta[:, axis], weight) for axis in range(2)])


def estimate_translation_layout(
    points_a_rc: np.ndarray,
    points_b_rc: np.ndarray,
    confidence: np.ndarray,
    valid_a: Optional[np.ndarray] = None,
    valid_b: Optional[np.ndarray] = None,
    *,
    config: TranslationLayoutConfig = TranslationLayoutConfig(),
) -> TranslationLayoutResult:
    """Estimate one A-to-B translation from a single, unbatched pair.

    Padding can appear anywhere; only true mask entries enter matching. Invalid
    padded coordinates/scores are ignored, but non-finite active values raise.
    Confidence zero means no evidence; arbitrarily small positive mass remains
    eligible. ``dual_softmax`` instead interprets the matrix as raw affinities.

    The returned inlier mask aligns with ``candidate_indices``. ``residual_px``
    is confidence-weighted inlier RMS distance. The runner-up is the strongest
    candidate-centered mode more than two inlier radii from the final estimate;
    its support is divided by final support (and may exceed one for the Cauchy
    ablation). No pairability decision is made by this function.
    """
    a, b, score, va, vb = _validate_inputs(points_a_rc, points_b_rc, confidence, valid_a, valid_b)
    indices, weight, keys = _candidates(a, b, score, va, vb, config)
    count = len(indices)
    if not count:
        return TranslationLayoutResult(np.full(2, np.nan), False, 0, 0, 0., 0., None,
                                       0., 0., 0., indices, np.zeros(0, dtype=np.bool_), "no_candidates")
    delta = b[indices[:, 1]] - a[indices[:, 0]]
    if not np.all(np.isfinite(delta)):
        raise ValueError("candidate displacements must be finite")
    radius = config.inlier_radius_px
    squared = np.sum((delta[:, None, :] - delta[None, :, :]) ** 2, axis=2)
    support_mask = squared <= radius * radius
    support = support_mask @ weight
    ambiguous = False
    if config.decoder == "mode_consensus":
        best_support = support.max()
        tied = np.flatnonzero(np.isclose(support, best_support, rtol=1e-12, atol=1e-14))
        cost = ((squared[tied] * support_mask[tied]) @ weight) / support[tied]
        tied = tied[np.isclose(cost, cost.min(), rtol=1e-12, atol=1e-14)]
        best = int(tied[0])
        ambiguous = any(keys[int(other)] == keys[best] and squared[best, other] > 4 * radius * radius
                        for other in tied[1:])
        estimate = delta[best].copy()
    else:
        estimate = _center(delta, weight, "weighted_median")
        for _ in range(config.cauchy_iterations):
            distance = np.linalg.norm(delta - estimate, axis=1)
            robust_weight = weight / (1.0 + (distance / radius) ** 2)
            updated = _center(delta, robust_weight, "weighted_mean")
            if np.linalg.norm(updated - estimate) < 1e-4:
                estimate = updated
                break
            estimate = updated
    for _ in range(config.refinement_iterations):
        inlier = np.linalg.norm(delta - estimate, axis=1) <= radius
        if not np.any(inlier):
            break
        updated = _center(delta[inlier], weight[inlier], config.refinement)
        if np.linalg.norm(updated - estimate) < 1e-8:
            estimate = updated
            break
        estimate = updated
    distance = np.linalg.norm(delta - estimate, axis=1)
    inlier = distance <= radius
    inlier_count = int(inlier.sum())
    final_support = float(weight[inlier].sum())
    separate = distance > 2 * radius
    runner_up = float(support[separate].max()) if np.any(separate) else 0.0
    valid = bool(inlier_count >= config.min_inliers and np.all(np.isfinite(estimate)) and not ambiguous)
    residual = float(np.sqrt(np.sum(weight[inlier] * distance[inlier] ** 2) / final_support)) if final_support else None
    reason = "ok" if valid else ("ambiguous_equal_modes" if ambiguous else "insufficient_inliers")
    return TranslationLayoutResult(
        estimate if valid else np.full(2, np.nan), valid, count, inlier_count,
        inlier_count / count, final_support / float(weight.sum()), residual,
        runner_up / final_support if final_support else 0.0,
        final_support, runner_up, indices, inlier, reason,
    )
