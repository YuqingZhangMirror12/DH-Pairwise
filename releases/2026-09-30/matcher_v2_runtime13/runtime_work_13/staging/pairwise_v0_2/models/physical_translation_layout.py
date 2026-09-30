"""Target-blind translation-mode reranking with normals and deep overlap.

This ablation is independent of the contiguous-seam decoder. Its proposals use
the original Top2 union (cap 512), 10-pixel consensus radius and three weighted
mean refinements. Proposal zero is the *unaltered* original Top2 result; at most
six additional, separated seeds are refined and deduplicated against it and
each other. No rotation, pair-classifier change, or whole-contour matching loss
is introduced.

For a mode with confidence support S, the registered default score is
    S * F_normal * exp(-20 * deep_overlap_ratio).
Each enabled normal vote is .25 + .75*(1-dot(nA,nB))/2; a missing normal is a
neutral vote of one. Votes use the same correspondence-confidence weights.
Normals come from +/-8 physical-pixel arc chords and outer-contour winding.
Deep overlap uses both masks eroded by a fixed 2-pixel disk and is normalized
by the smaller ORIGINAL filled-mask area. B is placed with offset -t_a_to_b.

All intersections are computed in the unbounded common coordinate frame;
neither placed fragment is cropped to an 800-pixel output canvas. Fractional
translation is the bilinear interpolation of four integer cross-correlations,
which also preserves A/B exchange symmetry. Zero physics explicitly retains
the exact baseline, independently of optional support diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy import ndimage

from .translation_layout import (
    TranslationLayoutConfig, _candidates, _center, _validate_inputs,
    estimate_translation_layout,
)


@dataclass(frozen=True)
class PhysicalTranslationConfig:
    top_k: int = 2
    max_candidates: int = 512
    inlier_radius_px: float = 10.0
    min_inliers: int = 3
    refinement_iterations: int = 3
    max_additional_modes: int = 6  # Baseline is additional to this count.
    mode_separation_px: float = 20.0  # Applied before AND after refinement.
    use_normals: bool = True
    use_overlap: bool = True
    normal_arc_half_length_px: float = 8.0
    normal_floor: float = 0.25
    normal_exponent: float = 1.0
    erosion_radius_px: int = 2
    overlap_penalty: float = 20.0
    support_mode: str = "sum"  # Registered four-arm experiment uses sum only.

    def __post_init__(self):
        for name in ("top_k", "max_candidates", "min_inliers", "refinement_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        for name in ("max_additional_modes", "erosion_radius_px"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + " must be a non-negative integer")
        for name in ("use_normals", "use_overlap"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(name + " must be bool")
        for name in ("inlier_radius_px", "mode_separation_px", "normal_arc_half_length_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("normal_exponent", "overlap_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and non-negative")
        if not math.isfinite(self.normal_floor) or not 0 <= self.normal_floor <= 1:
            raise ValueError("normal_floor must be in [0,1]")
        if self.support_mode not in ("sum", "independent"):
            raise ValueError("support_mode must be sum or independent")


@dataclass(frozen=True)
class PhysicalTranslationResult:
    t_a_to_b_rc: np.ndarray
    valid: bool
    baseline_displacement: np.ndarray
    baseline_valid: bool
    selected_mode_index: int
    mode_count: int
    proposed_mode_count: int
    deduplicated_mode_count: int
    mode_diagnostics: tuple
    candidate_count: int
    candidate_indices: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    support_weight: float
    residual_px: Optional[float]
    runner_up_score_ratio: float
    reason: str


def _filled_mask(value, name):
    mask = np.asarray(value)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError(name + " must be a filled binary [H,W] or [1,H,W] mask")
    if not np.all(np.isfinite(mask)) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError(name + " must contain binary 0/1 values")
    return mask.astype(bool, copy=False)


def outward_contour_normals(points, valid, *, arc_half_length_px=8.0):
    """Return RC unit outward normals plus validity, preserving token indices.

    Contours are ordered closed outer boundaries. Winding/cyclic start and
    unequal arc sampling are supported. A degenerate area, zero perimeter or
    zero chord yields unavailable normals, never an arbitrary rejection.
    """

    points = np.asarray(points, dtype=np.float64)
    indices = np.flatnonzero(valid)
    normals = np.zeros_like(points)
    usable = np.zeros(len(points), dtype=bool)
    if len(indices) < 3:
        return normals, usable
    p = points[indices]
    area = .5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    perimeter = float(lengths.sum())
    if abs(area) <= 1e-10 or perimeter <= 1e-10:
        return normals, usable
    cumulative = np.r_[0., np.cumsum(lengths)]
    centers = cumulative[:-1]

    def sample(position):
        location = position % perimeter
        segment = np.searchsorted(cumulative, location, side="right") - 1
        segment = np.clip(segment, 0, len(p) - 1)
        alpha = (location - cumulative[segment]) / np.maximum(lengths[segment], 1e-12)
        return p[segment] + alpha[:, None] * (p[(segment + 1) % len(p)] - p[segment])

    chord = sample(centers + arc_half_length_px) - sample(centers - arc_half_length_px)
    magnitude = np.linalg.norm(chord, axis=1)
    good = np.isfinite(magnitude) & (magnitude > 1e-8)
    # Cartesian CCW tangent's right-hand normal, expressed back in (row,col).
    normal = np.sign(area) * np.column_stack((chord[:, 1], -chord[:, 0]))
    normal[good] /= magnitude[good, None]
    normals[indices[good]] = normal[good]
    usable[indices[good]] = True
    return normals, usable


def _erode(mask, radius):
    if radius == 0:
        return mask
    row, col = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return ndimage.binary_erosion(mask, structure=(row * row + col * col <= radius * radius), border_value=0)


def translated_intersection_area(mask_a, mask_b, t_a_to_b_rc):
    """Full-plane intersection for B offset=-t, with fractional-pixel weights.

    Inputs are binary masks in their own local frames. An A pixel at (r,c)
    samples B at (r+t_r,c+t_c); slicing the common valid domain never discards
    an overlap merely because B extends beyond A's original image canvas.
    """

    a, b = np.asarray(mask_a, dtype=bool), np.asarray(mask_b, dtype=bool)
    t = np.asarray(t_a_to_b_rc, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or t.shape != (2,) or not np.isfinite(t).all():
        raise ValueError("intersection requires 2D masks and finite translation [2]")
    integer = np.floor(t).astype(np.int64)
    fraction = t - integer
    terms = []
    for dr in (0, 1):
        wr = fraction[0] if dr else 1. - fraction[0]
        for dc in (0, 1):
            wc = fraction[1] if dc else 1. - fraction[1]
            if wr * wc == 0:
                continue
            qr, qc = int(integer[0]) + dr, int(integer[1]) + dc
            r0, r1 = max(0, -qr), min(a.shape[0], b.shape[0] - qr)
            c0, c1 = max(0, -qc), min(a.shape[1], b.shape[1] - qc)
            if r1 <= r0 or c1 <= c0:
                continue
            intersection = np.count_nonzero(a[r0:r1, c0:c1] & b[r0 + qr:r1 + qr, c0 + qc:c1 + qc])
            terms.append(float(wr * wc) * int(intersection))
    return float(math.fsum(terms))


def _independent_support(indices, weight, na, nb):
    first, second = np.zeros(na), np.zeros(nb)
    if len(indices):
        np.maximum.at(first, indices[:, 0], weight)
        np.maximum.at(second, indices[:, 1], weight)
    return min(float(first.sum()), float(second.sum()))


def _refine_mode(delta, weight, seed, config):
    estimate = delta[seed].copy()
    for _ in range(config.refinement_iterations):
        inlier = np.linalg.norm(delta - estimate, axis=1) <= config.inlier_radius_px
        if not np.any(inlier):
            break
        updated = _center(delta[inlier], weight[inlier], "weighted_mean")
        if np.linalg.norm(updated - estimate) < 1e-8:
            estimate = updated
            break
        estimate = updated
    distance = np.linalg.norm(delta - estimate, axis=1)
    inlier = distance <= config.inlier_radius_px
    support = float(weight[inlier].sum())
    residual = float(np.sqrt(np.sum(weight[inlier] * distance[inlier] ** 2) / support)) if support else None
    return estimate, inlier, residual


def estimate_physical_translation_layout(
    points_a, points_b, assignment, valid_a=None, valid_b=None, *, mask_a, mask_b,
    config=PhysicalTranslationConfig(),
):
    """Return a pose only; inputs contain predicted evidence and geometry, no GT.

    Candidate zero is always the baseline, even when it is invalid. At most
    1+max_additional_modes proposals exist BEFORE refinement deduplication.
    Mode diagnostics and inlier_mask index the original Top2 candidate array.
    Disabled physics selects the baseline explicitly, so optional independent
    support or additional proposals cannot change the zero-physics control.
    """

    a, b, score, va, vb = _validate_inputs(points_a, points_b, assignment, valid_a, valid_b)
    ma, mb = _filled_mask(mask_a, "mask_a"), _filled_mask(mask_b, "mask_b")
    baseline_config = TranslationLayoutConfig(
        correspondence_mode="topk_union", decoder="mode_consensus", score_mode="confidence",
        top_k=config.top_k, max_candidates=config.max_candidates,
        inlier_radius_px=config.inlier_radius_px, min_inliers=config.min_inliers,
        refinement="weighted_mean", refinement_iterations=config.refinement_iterations,
    )
    baseline = estimate_translation_layout(a, b, score, va, vb, config=baseline_config)
    indices = baseline.candidate_indices
    weight = np.asarray(score[indices[:, 0], indices[:, 1]], dtype=np.float64).copy()
    if len(weight):
        weight /= weight.max()
    count = len(indices)
    delta = b[indices[:, 1]] - a[indices[:, 0]]
    normal_on = config.use_normals and config.normal_exponent > 0 and config.normal_floor < 1
    overlap_on = config.use_overlap and config.overlap_penalty > 0
    if normal_on:
        normals_a, good_a = outward_contour_normals(a, va, arc_half_length_px=config.normal_arc_half_length_px)
        normals_b, good_b = outward_contour_normals(b, vb, arc_half_length_px=config.normal_arc_half_length_px)
        good_edge = good_a[indices[:, 0]] & good_b[indices[:, 1]]
        dot = np.clip(np.sum(normals_a[indices[:, 0]] * normals_b[indices[:, 1]], axis=1), -1., 1.)
        normal_vote = np.where(good_edge, config.normal_floor + (1. - config.normal_floor) * (1. - dot) * .5, 1.)
    else:
        good_edge, dot, normal_vote = np.zeros(count, bool), np.zeros(count), np.ones(count)
    if overlap_on:
        deep_a, deep_b = _erode(ma, config.erosion_radius_px), _erode(mb, config.erosion_radius_px)
    area_a, area_b = int(ma.sum()), int(mb.sum())
    denominator = min(area_a, area_b)
    modes, inliers = [], []

    def append_mode(estimate, inlier, residual, *, is_baseline, seed, valid):
        support = float(weight[inlier].sum())
        independent = _independent_support(indices[inlier], weight[inlier], len(a), len(b))
        factor = float(np.average(normal_vote[inlier], weights=weight[inlier])) if support else 1.
        normal_valid_fraction = float(weight[inlier & good_edge].sum() / support) if support else 0.
        dot_mean = float(np.average(dot[inlier & good_edge], weights=weight[inlier & good_edge])) if np.any(inlier & good_edge) else None
        intersection = translated_intersection_area(deep_a, deep_b, estimate) if overlap_on and np.isfinite(estimate).all() else 0.
        overlap = intersection / denominator if denominator else 0.
        ranking_support = support if config.support_mode == "sum" else independent
        physics_score = ranking_support * factor ** config.normal_exponent * math.exp(-config.overlap_penalty * overlap)
        if not valid:
            physics_score = 0.
        modes.append({
            "mode_index": len(modes), "is_baseline": is_baseline, "seed_candidate_index": seed,
            "translation_rc": estimate.tolist(), "valid": bool(valid), "inlier_count": int(inlier.sum()),
            "support_weight": support, "independent_support_weight": independent,
            "normal_factor": factor, "normal_valid_weight_fraction": normal_valid_fraction,
            "mean_normal_dot": dot_mean, "deep_overlap_area_px": intersection, "deep_overlap_ratio": overlap,
            "overlap_normalizer_area_px": denominator, "overlap_available": bool(denominator),
            "physics_score": physics_score, "residual_px": residual,
        })
        inliers.append(inlier.copy())

    append_mode(baseline.t_a_to_b_rc.copy(), baseline.inlier_mask, baseline.residual_px,
                is_baseline=True, seed=None, valid=baseline.valid)
    proposed, deduplicated = 1, 0
    if count and config.max_additional_modes:
        squared = np.sum((delta[:, None, :] - delta[None, :, :]) ** 2, axis=2)
        neighborhood = squared <= config.inlier_radius_px ** 2
        support = neighborhood @ weight
        cost = ((squared * neighborhood) @ weight) / support
        ranked = np.lexsort((np.arange(count), cost, -support))
        separated_seeds = []
        for seed_value in ranked:
            seed = int(seed_value)
            if baseline.valid and np.linalg.norm(delta[seed] - baseline.t_a_to_b_rc) <= config.mode_separation_px:
                continue
            if any(np.linalg.norm(delta[seed] - delta[old]) <= config.mode_separation_px for old in separated_seeds):
                continue
            separated_seeds.append(seed)
            if len(separated_seeds) >= config.max_additional_modes:
                break
        for seed in separated_seeds:
            proposed += 1
            estimate, inlier, residual = _refine_mode(delta, weight, seed, config)
            if any(np.isfinite(mode["translation_rc"]).all()
                   and np.linalg.norm(estimate - mode["translation_rc"]) <= config.mode_separation_px for mode in modes):
                deduplicated += 1
                continue
            valid = int(inlier.sum()) >= config.min_inliers and np.isfinite(estimate).all()
            append_mode(estimate, inlier, residual, is_baseline=False, seed=seed, valid=valid)
    if not normal_on and not overlap_on:
        selected = 0
    else:
        selected = max(range(len(modes)), key=lambda index: (
            modes[index]["valid"], modes[index]["physics_score"],
            modes[index]["support_weight"],
            -(modes[index]["residual_px"] if modes[index]["residual_px"] is not None else float("inf")),
            -index,
        ))
    chosen = modes[selected]
    runner = max((m["physics_score"] for i, m in enumerate(modes) if i != selected and m["valid"]), default=0.)
    reason = baseline.reason if selected == 0 else ("ok" if chosen["valid"] else "insufficient_inliers")
    return PhysicalTranslationResult(
        t_a_to_b_rc=np.asarray(chosen["translation_rc"]) if chosen["valid"] else np.full(2, np.nan),
        valid=chosen["valid"], baseline_displacement=baseline.t_a_to_b_rc.copy(), baseline_valid=baseline.valid,
        selected_mode_index=selected, mode_count=len(modes), proposed_mode_count=proposed,
        deduplicated_mode_count=deduplicated, mode_diagnostics=tuple(modes),
        candidate_count=count, candidate_indices=indices, inlier_mask=inliers[selected],
        inlier_count=chosen["inlier_count"], support_weight=chosen["support_weight"],
        residual_px=chosen["residual_px"],
        runner_up_score_ratio=runner / chosen["physics_score"] if chosen["physics_score"] else 0., reason=reason,
    )


__all__ = [
    "PhysicalTranslationConfig", "PhysicalTranslationResult", "estimate_physical_translation_layout",
    "outward_contour_normals", "translated_intersection_area",
]
