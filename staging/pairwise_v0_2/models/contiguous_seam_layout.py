"""Translation-mode reranking by one continuous, oppositely traversed seam.

This inference-only ablation reuses v2's sparse Top2 correspondence candidates.
It never changes pairability scores, estimates rotation, or reads targets. Outer
contours are oriented consistently using their signed polygon area. A seam then
advances along A and backwards along B, allowing gaps measured in physical arc
length. A one-to-one chain prevents repeated endpoint votes. No mask normals or
fragment overlap terms are used.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Optional

import numpy as np

from .translation_layout import TranslationLayoutConfig, _candidates, _validate_inputs


@dataclass(frozen=True)
class ContiguousSeamConfig:
    top_k: int = 2
    max_candidates: int = 512
    max_modes: int = 6
    inlier_radius_px: float = 10.0
    max_gap_px: float = 20.0
    max_gap_ratio: float = 3.0
    min_inliers: int = 3
    refinement_iterations: int = 3
    refinement_support: str = "seam"

    def __post_init__(self):
        for name in ("top_k", "max_candidates", "max_modes", "min_inliers", "refinement_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        for name in ("inlier_radius_px", "max_gap_px", "max_gap_ratio"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.max_gap_ratio < 1:
            raise ValueError("max_gap_ratio must be at least one")
        if self.refinement_support not in ("seam", "mode"):
            raise ValueError("refinement_support must be seam or mode")


@dataclass(frozen=True)
class ContiguousSeamResult:
    """Seam diagnostics retain their chain-only meaning in both refinements.

    In particular, inlier_count, residual_px, seam_membership, segment_spans and
    mode_diagnostics describe the selected seam and its chain-only estimate,
    used for ranking. pose_refinement_point_count separately counts the points
    used in the final weighted pose update; it can exceed inlier_count when
    refinement_support='mode'. No successful pose is implied when valid=False.
    """
    t_a_to_b_rc: np.ndarray
    valid: bool
    candidate_count: int
    inlier_count: int
    inlier_fraction: float
    residual_px: Optional[float]
    seam_score: float
    runner_up_support_ratio: float
    candidate_indices: np.ndarray
    seam_membership: np.ndarray
    segment_spans: dict
    mode_diagnostics: tuple
    reason: str
    pose_refinement_point_count: int = 0


def _contour_arc(points, valid):
    indices = np.flatnonzero(valid)
    p = points[indices]
    if len(p) < 3:
        raise ValueError("a closed contour needs at least three valid points")
    area = .5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
    if abs(area) <= 1e-10:
        raise ValueError("outer contour must have nonzero signed area")
    if area < 0:
        indices = indices[::-1]
    p = points[indices]
    # A geometric start makes diagnostics and all ordering independent of the
    # arbitrary cyclic start used by a contour sampler.
    start = int(np.lexsort((p[:, 1], p[:, 0]))[0])
    indices = np.roll(indices, -start)
    p = points[indices]
    steps = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    perimeter = float(steps.sum())
    arc = np.zeros(len(points), dtype=float)
    cell = np.zeros(len(points), dtype=float)
    arc[indices] = np.r_[0., np.cumsum(steps[:-1])]
    # The Voronoi cell of every distinct contour token is counted at most once.
    cell[indices] = .5 * (steps + np.roll(steps, 1))
    return arc, cell, perimeter, bool(area < 0)


def _independent_support(indices, weights, size_a, size_b):
    a, b = np.zeros(size_a), np.zeros(size_b)
    np.maximum.at(a, indices[:, 0], weights)
    np.maximum.at(b, indices[:, 1], weights)
    return min(float(a.sum()), float(b.sum()))


def _chain_dag(adjacency, da, db, values, perimeter_a, perimeter_b, order):
    count = len(values)
    score = values.copy()
    predecessor = np.full(count, -1, dtype=int)
    span_a, span_b = np.zeros(count), np.zeros(count)
    length = np.ones(count, dtype=int)
    for node in order:
        previous = np.flatnonzero(adjacency[:, node])
        if not len(previous):
            continue
        pa, pb = span_a[previous] + da[previous, node], span_b[previous] + db[previous, node]
        eligible = (pa < perimeter_a - 1e-7) & (pb < perimeter_b - 1e-7)
        previous, pa, pb = previous[eligible], pa[eligible], pb[eligible]
        if not len(previous):
            continue
        ranked = np.lexsort((previous, -length[previous], -(score[previous] + values[node])))
        best = int(ranked[0])
        parent = int(previous[best])
        predecessor[node] = parent
        score[node] = score[parent] + values[node]
        span_a[node], span_b[node] = pa[best], pb[best]
        length[node] = length[parent] + 1
    end = int(np.lexsort((np.arange(count), -length, -score))[0])
    chain = []
    while end >= 0:
        chain.append(end)
        end = int(predecessor[end])
    return np.asarray(chain[::-1], dtype=int)


def _best_chain(indices, weights, arc_a, arc_b, cell_a, cell_b, perimeter_a, perimeter_b, config):
    count = len(indices)
    sa, sb = arc_a[indices[:, 0]], (-arc_b[indices[:, 1]]) % perimeter_b
    da = (sa[None, :] - sa[:, None]) % perimeter_a
    db = (sb[None, :] - sb[:, None]) % perimeter_b
    adjacency = ((da > 1e-7) & (db > 1e-7) & (da <= config.max_gap_px)
                 & (db <= config.max_gap_px)
                 & (np.maximum(da, db) <= config.max_gap_ratio * np.minimum(da, db)))
    # Positive progression in each arc forbids repeating an A or B endpoint.
    values = weights * np.minimum(cell_a[indices[:, 0]], cell_b[indices[:, 1]])
    indegree = adjacency.sum(axis=0).astype(int)
    queue = np.flatnonzero(indegree == 0).tolist()
    heapq.heapify(queue)
    order = []
    while queue:
        node = heapq.heappop(queue)
        order.append(node)
        for successor in np.flatnonzero(adjacency[node]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(queue, int(successor))
    if len(order) == count:
        return _chain_dag(adjacency, da, db, values, perimeter_a, perimeter_b, order)
    # Very short / closed cyclic candidate sets require a cut. Try every
    # candidate-origin unwrapping; ordinary fragment seams use the DAG path.
    chains = []
    for start in range(count):
        ua, ub = (sa - sa[start]) % perimeter_a, (sb - sb[start]) % perimeter_b
        cut = adjacency & (ua[None, :] > ua[:, None]) & (ub[None, :] > ub[:, None])
        chains.append(_chain_dag(cut, da, db, values, perimeter_a, perimeter_b, np.argsort(ua, kind="stable")))
    return max(chains, key=lambda chain: (float(values[chain].sum()), len(chain), tuple(-chain)))


def _describe_chain(chain, indices, weights, delta, arc_a, arc_b, cell_a, cell_b, pa, pb):
    chosen = indices[chain]
    ca, cb = float(cell_a[chosen[:, 0]].sum()), float(cell_b[chosen[:, 1]].sum())
    common = 2 * ca * cb / max(ca + cb, 1e-12)
    cell_weight = np.minimum(cell_a[chosen[:, 0]], cell_b[chosen[:, 1]])
    mean_confidence = float(np.average(weights[chain], weights=cell_weight)) if cell_weight.sum() else 0.
    estimate = np.average(delta[chain], axis=0, weights=weights[chain])
    residual = float(np.sqrt(np.average(np.sum((delta[chain] - estimate) ** 2, axis=1), weights=weights[chain])))
    first, last = chosen[0], chosen[-1]
    span_a = float((arc_a[last[0]] - arc_a[first[0]]) % pa)
    span_b = float((arc_b[first[1]] - arc_b[last[1]]) % pb)
    spans = {"a_start_index": int(first[0]), "a_end_index": int(last[0]),
        "b_start_index": int(first[1]), "b_end_index": int(last[1]),
        "a_traversal": "forward_in_canonical_orientation", "b_traversal": "reverse_in_canonical_orientation",
        "a_span_px": span_a, "b_span_px": span_b,
        "a_covered_arc_px": ca, "b_covered_arc_px": cb, "common_covered_arc_px": common,
        "a_unique_point_count": len(set(chosen[:, 0])), "b_unique_point_count": len(set(chosen[:, 1])),
        "mean_confidence": mean_confidence}
    return estimate, residual, common * mean_confidence, spans


def estimate_contiguous_seam_layout(points_a_rc, points_b_rc, confidence, valid_a=None, valid_b=None,
                                    *, config=ContiguousSeamConfig()):
    """Return a target-blind translation and one supported anti-directed arc.

    Candidate and seam indices refer to the original input token arrays, even
    when their winding or cyclic starts differ. Invalid estimates contain NaN.
    ``refinement_support='seam'`` preserves the original chain-only pose.
    ``'mode'`` changes only the final pose refinement: after selecting a mode
    with exactly the same seam ranking, repeat its original full-neighborhood
    weighted refinement from the same seed. It does not reselect a mode or
    change seam scores/membership. All parameters remain target-independent.
    """
    a, b, score, va, vb = _validate_inputs(points_a_rc, points_b_rc, confidence, valid_a, valid_b)
    candidate_config = TranslationLayoutConfig(correspondence_mode="topk_union", top_k=config.top_k,
        max_candidates=config.max_candidates, inlier_radius_px=config.inlier_radius_px, min_inliers=config.min_inliers)
    indices, weight, _ = _candidates(a, b, score, va, vb, candidate_config)
    count = len(indices)
    empty = lambda reason: ContiguousSeamResult(np.full(2, np.nan), False, count, 0, 0., None,
        0., 0., indices, np.zeros(count, dtype=bool), {}, (), reason)
    if not count:
        return empty("no_candidates")
    if np.count_nonzero(va) < 3 or np.count_nonzero(vb) < 3:
        return empty("insufficient_contour_points")
    try:
        arc_a, cell_a, pa, reversed_a = _contour_arc(a, va)
        arc_b, cell_b, pb, reversed_b = _contour_arc(b, vb)
    except ValueError:
        return empty("degenerate_contour")
    delta = b[indices[:, 1]] - a[indices[:, 0]]
    distances = np.linalg.norm(delta[:, None, :] - delta[None, :, :], axis=2)
    neighborhood = distances <= config.inlier_radius_px
    support = np.asarray([_independent_support(indices[m], weight[m], len(a), len(b)) for m in neighborhood])
    ranked = np.lexsort((np.arange(count), -(neighborhood @ weight), -support))
    seeds = []
    for seed in ranked:
        if all(distances[seed, old] > 2 * config.inlier_radius_px for old in seeds):
            seeds.append(int(seed))
        if len(seeds) == config.max_modes:
            break
    modes, solutions = [], []
    for seed in seeds:
        estimate = delta[seed].copy()
        for _ in range(config.refinement_iterations):
            member = np.linalg.norm(delta - estimate, axis=1) <= config.inlier_radius_px
            estimate = np.average(delta[member], axis=0, weights=weight[member])
        member = np.flatnonzero(np.linalg.norm(delta - estimate, axis=1) <= config.inlier_radius_px)
        if not len(member):
            continue
        local_chain = _best_chain(indices[member], weight[member], arc_a, arc_b, cell_a, cell_b, pa, pb, config)
        chain = member[local_chain]
        for _ in range(config.refinement_iterations):
            estimate = np.average(delta[chain], axis=0, weights=weight[chain])
            kept = chain[np.linalg.norm(delta[chain] - estimate, axis=1) <= config.inlier_radius_px]
            if len(kept) == len(chain) or not len(kept):
                break
            chain = kept[_best_chain(indices[kept], weight[kept], arc_a, arc_b, cell_a, cell_b, pa, pb, config)]
        estimate, residual, seam_score, spans = _describe_chain(chain, indices, weight, delta,
            arc_a, arc_b, cell_a, cell_b, pa, pb)
        valid = len(chain) >= config.min_inliers
        modes.append({"seed_candidate_index": seed, "translation_rc": estimate.tolist(),
            "mode_candidate_count": len(member), "independent_point_support": float(support[seed]),
            "seam_score": seam_score, "inlier_count": len(chain), "residual_px": residual,
            "valid": valid, "segment_spans": spans})
        solutions.append((valid, seam_score, len(chain), -residual, estimate, chain, spans))
    if not solutions:
        return empty("no_modes")
    ranking = sorted(range(len(solutions)), key=lambda i: solutions[i][:4], reverse=True)
    best = solutions[ranking[0]]
    valid, value, inliers, negative_residual, estimate, chain, spans = best
    runner = solutions[ranking[1]][1] if len(ranking) > 1 else 0.
    membership = np.zeros(count, dtype=bool)
    membership[chain] = True
    spans = dict(spans, a_input_winding_reversed=reversed_a, b_input_winding_reversed=reversed_b)
    pose_refinement_point_count = inliers
    if config.refinement_support == "mode":
        # The selected seed is fixed by the unmodified seam-only ranking.
        # Reproduce its pre-chain refinement, keeping all correspondence votes
        # in that translation neighborhood rather than only its longest chain.
        # Restarting from that seed (not the chain-biased center) also makes
        # this identical to the full-mode estimate used during mode generation.
        seed = modes[ranking[0]]["seed_candidate_index"]
        estimate = delta[seed].copy()
        for _ in range(config.refinement_iterations):
            pose_member = np.linalg.norm(delta - estimate, axis=1) <= config.inlier_radius_px
            estimate = np.average(delta[pose_member], axis=0, weights=weight[pose_member])
        # This is the support actually used by the last update, not a new
        # post-refinement inlier recount, and not the seam's inlier_count.
        pose_refinement_point_count = int(np.count_nonzero(pose_member))
    return ContiguousSeamResult(estimate if valid else np.full(2, np.nan), valid, count, inliers,
        inliers / count, -negative_residual, value, runner / value if value else 0.,
        indices, membership, spans, tuple(modes), "ok" if valid else "insufficient_contiguous_support",
        pose_refinement_point_count=pose_refinement_point_count)
