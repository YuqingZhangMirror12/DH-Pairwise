"""Target-blind candidate-stage views of the existing frozen Matcher cache.

No Matcher forward, labels, GT, model weights, or queue operations occur here.
All three views retain ORIGINAL candidate IDs and (i,j) edge identity. Translation
is A-to-B in row/column coordinates: p_b = p_a + translation. To display B on A,
negate it. Separate displacement modes remain separate groups, never one union.

`single_seed` is the production best-support seed BEFORE three mean refinements.
`multi_modes` replays candidate_modes_v1.propose_modes, including its all-edge
refinement and 2*radius NMS. `final_decoded` uses only cached production VALID
final inliers. Solver validity is not a pairability label. Invalid/ambiguous
cache rows still expose seed/proposals with explicit status and no final group.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


SCHEMA = "matched-only-candidate-stages/1"


@dataclass(frozen=True)
class CandidateGroup:
    stage: str
    group_id: str
    rank: int
    seed_candidate_id: Optional[int]
    candidate_ids: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    translation_rc: np.ndarray
    inlier_count: int
    seed_support_normalized: float
    support_normalized: float
    raw_q_mass: float
    residual_px: float
    status: str


@dataclass(frozen=True)
class CandidateGroups:
    schema: str
    candidate_count: int
    production_status: str
    single_seed: Optional[CandidateGroup]
    multi_modes: Tuple[CandidateGroup, ...]
    final_decoded: Optional[CandidateGroup]
    diagnostics: dict


def _boolean(value, shape, name):
    value = np.asarray(value)
    if value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(name + " must be a Boolean array of shape " + str(shape))
    return value


def _edge_key(a, b, i, j, weight):
    # Exact production tie key; order is NOT recomputed, preserving cap/Top2.
    first = (float(a[i, 0]), float(a[i, 1]), int(i))
    second = (float(b[j, 0]), float(b[j, 1]), int(j))
    return (-float(weight),) + min(first, second) + max(first, second)


def build_candidate_groups(points_a_rc, points_b_rc, candidate_indices,
                           candidate_valid, candidate_weights, final_inliers,
                           final_translation_rc, layout_valid, *, valid_a=None,
                           valid_b=None, radius=10.0, max_modes=5, min_inliers=3,
                           replay_atol=1e-4):
    """Build one sample's candidate side-cache without any target access.

    Arrays are the corresponding unbatched rows of the existing cache:
    points_[ab], candidate_indices/valid/weights/inliers,
    translation_a_to_b_rc and layout_valid. Cache padding may occur anywhere and
    is ignored. Active weights must be positive raw Q values; candidate order
    must be the original production order. This function does not re-extract or
    re-rank the Top2/cap512 correspondences.

    Raises on cache inconsistency, including a production replay mismatch; it
    never silently substitutes a new pose. replay_atol accommodates cached
    float32 translation rounding, not a changed decoder. Returned group arrays
    are independent copies suitable for a per-sample derived cache.
    """
    a, b = np.asarray(points_a_rc, np.float64), np.asarray(points_b_rc, np.float64)
    if a.ndim != 2 or a.shape[1:] != (2,) or b.ndim != 2 or b.shape[1:] != (2,):
        raise ValueError("points must be [N,2] and [M,2]")
    va = np.ones(len(a), bool) if valid_a is None else _boolean(valid_a, (len(a),), "valid_a")
    vb = np.ones(len(b), bool) if valid_b is None else _boolean(valid_b, (len(b),), "valid_b")
    if not np.isfinite(a[va]).all() or not np.isfinite(b[vb]).all():
        raise ValueError("active points must be finite")
    indices = np.asarray(candidate_indices)
    if indices.ndim != 2 or indices.shape[1:] != (2,) or indices.dtype.kind not in "iu":
        raise ValueError("candidate_indices must be an integer [C,2] array")
    if len(indices) > 512:
        raise ValueError("this side-cache is bound to production cap512")
    present = _boolean(candidate_valid, (len(indices),), "candidate_valid")
    final = _boolean(final_inliers, (len(indices),), "final_inliers")
    if (final & ~present).any():
        raise ValueError("final inliers cannot include padding")
    raw = np.asarray(candidate_weights, np.float64)
    if raw.shape != (len(indices),):
        raise ValueError("candidate_weights must align with candidate_indices")
    if not np.isfinite(raw[present]).all() or (raw[present] <= 0).any():
        raise ValueError("active raw Q weights must be finite and positive")
    if (not np.isfinite(radius) or radius <= 0 or type(max_modes) is not int
            or not 1 <= max_modes <= 5 or type(min_inliers) is not int or min_inliers < 1
            or not np.isfinite(replay_atol) or replay_atol < 0):
        raise ValueError("invalid radius, max_modes, min_inliers or replay_atol")
    if np.asarray(layout_valid).shape != () or np.asarray(layout_valid).dtype != np.bool_:
        raise ValueError("layout_valid must be a Boolean scalar")
    cached_translation = np.asarray(final_translation_rc, np.float64)
    if cached_translation.shape != (2,):
        raise ValueError("final_translation_rc must have shape [2]")
    if layout_valid and not np.isfinite(cached_translation).all():
        raise ValueError("valid cached layout needs a finite translation")
    if not layout_valid and not np.isnan(cached_translation).all():
        raise ValueError("invalid production cache layout must retain NaN translation")
    original_ids = np.flatnonzero(present)
    edges, mass = indices[present].astype(np.int64), raw[present]
    if len(edges):
        if ((edges < 0).any() or (edges[:, 0] >= len(a)).any()
                or (edges[:, 1] >= len(b)).any()):
            raise ValueError("active candidate endpoint index out of bounds")
        if not va[edges[:, 0]].all() or not vb[edges[:, 1]].all():
            raise ValueError("candidate references an invalid/padded point")
        if len(np.unique(edges, axis=0)) != len(edges):
            raise ValueError("production candidates must not duplicate an (i,j) edge")
    diagnostics = dict(gt_used=False, mode_groups_merged=False,
                       radius_px=float(radius), refinement_iterations=3,
                       nms_distance_px=float(2 * radius), max_modes=max_modes,
                       first_mode_replay_checked=False, first_mode_replay_l2_px=None,
                       seed_final_edge_jaccard=None, seed_final_edges_equal=None,
                       seed_final_translation_l2_px=None, seed_final_identical=None)
    if not len(edges):
        if layout_valid or final.any():
            raise ValueError("empty candidate cache cannot contain a final layout")
        return CandidateGroups(SCHEMA, 0, "no_candidates", None, (), None, diagnostics)

    delta = b[edges[:, 1]] - a[edges[:, 0]]
    if not np.isfinite(delta).all():
        raise ValueError("candidate displacements must be finite")
    weight = mass / mass.max()
    squared = ((delta[:, None] - delta[None, :]) ** 2).sum(2)
    support_mask = squared <= radius ** 2
    support = support_mask @ weight
    # Key weights in production are raw confidence values, not normalized mass.
    keys = [_edge_key(a, b, int(i), int(j), w) for (i, j), w in zip(edges, mass)]

    def best_seed(available):
        ids = np.flatnonzero(available)
        tied = ids[np.isclose(support[ids], support[ids].max(), rtol=1e-12, atol=1e-14)]
        cost = ((squared[tied] * support_mask[tied]) @ weight) / support[tied]
        tied = tied[np.isclose(cost, cost.min(), rtol=1e-12, atol=1e-14)]
        return int(tied[0]), tied

    def refine(seed):
        center = delta[seed].copy()
        for _ in range(3):
            inside = np.linalg.norm(delta - center, axis=1) <= radius
            if not inside.any():
                break
            updated = (delta[inside] * weight[inside, None]).sum(0) / weight[inside].sum()
            change = np.linalg.norm(updated - center)
            center = updated
            if change < 1e-8:
                break
        distance = np.linalg.norm(delta - center, axis=1)
        return center, distance, distance <= radius

    def group(stage, rank, seed, inside, translation, status):
        distance = np.linalg.norm(delta[inside] - translation, axis=1)
        selected_weight = weight[inside]
        residual = float(np.sqrt((selected_weight * distance ** 2).sum() / selected_weight.sum()))
        return CandidateGroup(stage, stage + ":" + str(rank), rank,
                              int(original_ids[seed]), original_ids[inside].copy(),
                              edges[inside].copy(), mass[inside].copy(), translation.copy(),
                              int(inside.sum()), float(support[seed]),
                              float(selected_weight.sum()), float(mass[inside].sum()),
                              residual, status)

    seed, tied = best_seed(np.ones(len(edges), bool))
    ambiguous = any(keys[int(other)] == keys[seed]
                    and squared[seed, other] > 4 * radius ** 2 for other in tied[1:])
    center, distance, refined = refine(seed)
    production_status = ("ambiguous_equal_modes" if ambiguous else
                         "ok" if int(refined.sum()) >= min_inliers else "insufficient_inliers")
    diagnostics["production_equal_seed_tie_count"] = int(len(tied))
    diagnostics["production_ambiguous"] = bool(ambiguous)
    diagnostics["production_refined_inlier_count"] = int(refined.sum())
    if bool(layout_valid) != (production_status == "ok"):
        raise ValueError("cached production layout_valid fails mode-consensus replay")
    if not np.array_equal(final[present], refined):
        raise ValueError("cached final inlier membership fails mode-consensus replay")

    seed_status = ("ambiguous_equal_modes" if ambiguous else
                   "ok" if int(support_mask[seed].sum()) >= min_inliers else "insufficient_inliers")
    single = group("single_seed", 1, seed, support_mask[seed], delta[seed], seed_status)
    final_group = None
    if layout_valid:
        replay_error = float(np.linalg.norm(center - cached_translation))
        if not np.allclose(center, cached_translation, rtol=0., atol=replay_atol):
            raise ValueError("cached final translation fails mode-consensus replay")
        diagnostics["first_mode_replay_checked"] = True
        diagnostics["first_mode_replay_l2_px"] = replay_error
        # Deliberately retain CACHED production values, rather than their replay.
        final_group = group("final_decoded", 1, seed, final[present], cached_translation, "ok")
        equal_edges = bool(np.array_equal(single.candidate_ids, final_group.candidate_ids))
        shift = float(np.linalg.norm(single.translation_rc - cached_translation))
        diagnostics.update(seed_final_edge_jaccard=float(
            np.logical_and(support_mask[seed], refined).sum() /
            np.logical_or(support_mask[seed], refined).sum()),
            seed_final_edges_equal=equal_edges, seed_final_translation_l2_px=shift,
            seed_final_identical=bool(equal_edges and shift <= replay_atol))

    available, modes = np.ones(len(edges), bool), []
    attempts, insufficient_attempts, suppressed_attempts = 0, 0, 0
    while available.any() and len(modes) < max_modes:
        current, current_ties = best_seed(available)
        refined_center, refined_distance, inside = refine(current)
        available[current] = False
        available[refined_distance <= 2 * radius] = False
        attempts += 1
        if int(inside.sum()) < min_inliers:
            insufficient_attempts += 1
            continue
        if any(np.linalg.norm(refined_center - m.translation_rc) <= 2 * radius for m in modes):
            suppressed_attempts += 1
            continue
        current_ambiguous = any(keys[int(other)] == keys[current]
            and squared[current, other] > 4 * radius ** 2 for other in current_ties[1:])
        modes.append(group("multi_mode", len(modes) + 1, current, inside, refined_center,
                           "ambiguous_equal_modes" if current_ambiguous else "proposal"))
    diagnostics["mode_seed_attempts"] = attempts
    diagnostics["mode_insufficient_attempts"] = insufficient_attempts
    diagnostics["mode_suppressed_attempts"] = suppressed_attempts
    diagnostics["multi_mode_count"] = len(modes)
    diagnostics["multiple_separated_modes"] = len(modes) > 1
    if layout_valid and (not modes or not np.array_equal(modes[0].candidate_ids, final_group.candidate_ids)
            or not np.allclose(modes[0].translation_rc, cached_translation, rtol=0., atol=replay_atol)):
        raise ValueError("first accepted multi-mode must replay valid production final")
    return CandidateGroups(SCHEMA, len(edges), production_status, single, tuple(modes),
                           final_group, diagnostics)
