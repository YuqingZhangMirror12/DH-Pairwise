"""Exact parent-canvas seam targets for contour-keypoint Sinkhorn training.

The supplied ``shredding_pipeline`` writes every fragment mask in one aligned
parent canvas.  Its current Voronoi-fill implementation makes the masks a
disjoint, exhaustive partition, so an A/B seam is recoverable without RGB or
text: every horizontal or vertical grid edge whose two incident pixels belong
to different fragments is one exact correspondence.

This module keeps two concepts deliberately separate:

* :class:`AlignedMaskSeam` is exact pixel geometry in the generator's parent
  canvas;
* :class:`KeypointAssignmentTarget` is a deterministic, conservative
  discretisation of that seam onto the inference-time keypoint tokens.

Ground truth is never placed in ``KeypointPairCandidate.correspondence_mask``.
That mask remains a label-blind allowed-edge graph (normally same-scale only),
so training and inference receive identical model inputs.  Targets use the
compact partial-assignment convention ``>=0`` = opposite token index, ``-1`` =
dustbin, and ``-2`` = ignore/padding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import Tensor

from staging.pairwise_v0_2.geometry.keypoint_candidates import (
    KeypointPairCandidate,
    KeypointPairResult,
    KeypointScaleToken,
)
from staging.pairwise_v0_2.geometry.schema import DEFAULT_DIRECTION_ORDER, PairDirection
from staging.pairwise_v0_2.models.local_matcher import LocalMatcherOutput


IGNORE_ASSIGNMENT = -2
DUSTBIN_ASSIGNMENT = -1
EXACT_SEAM_SCHEMA_VERSION = "aligned-mask-exact-seam-keypoint-target/v0.1"


class SyntheticMaskPairLike(Protocol):
    """Dependency-light contract used by the optional synthetic-pair adapter."""

    group_key: str
    fragment_a_id: str
    fragment_b_id: str
    label: bool

    def load_masks(self) -> Tuple[np.ndarray, np.ndarray]: ...


class ExactSeamSupervisionError(ValueError):
    """Raised when aligned masks cannot support an exact supervision claim."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _binary_mask(value: np.ndarray, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.size == 0:
        raise ExactSeamSupervisionError(name + " must be a non-empty 2D mask")
    if not np.issubdtype(source.dtype, np.bool_) and not np.issubdtype(
        source.dtype, np.number
    ):
        raise ExactSeamSupervisionError(name + " must be bool or numeric")
    if source.dtype == np.bool_:
        result = source
    else:
        finite = np.isfinite(source)
        if not finite.all():
            raise ExactSeamSupervisionError(name + " contains non-finite values")
        threshold = 0.0 if float(np.max(source)) <= 1.0 else 127.0
        result = source > threshold
    if not result.any():
        raise ExactSeamSupervisionError(name + " contains no foreground")
    return np.asarray(result, dtype=np.bool_)


@dataclass(frozen=True)
class AlignedMaskSeam:
    """Ordered exact cross-fragment grid edges in one parent canvas.

    ``segments_rc`` are oriented along deterministic maximal paths.  The
    adjacent A/B foreground pixels remain paired one-to-one regardless of path
    orientation.  ``path_arclength_px`` is the midpoint arclength of each unit
    grid edge, making it a stable coordinate for projecting sparse keypoints.
    """

    a_pixels_rc: np.ndarray
    b_pixels_rc: np.ndarray
    a_points_rc: np.ndarray
    b_points_rc: np.ndarray
    segments_rc: np.ndarray
    edge_midpoints_rc: np.ndarray
    path_indices: np.ndarray
    path_arclength_px: np.ndarray
    path_arc_fractions: np.ndarray
    path_lengths_px: np.ndarray
    parent_shape: Tuple[int, int]
    has_branch_vertex: bool
    schema_version: str = EXACT_SEAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        count = int(self.a_pixels_rc.shape[0])
        if count < 1:
            raise ValueError("an exact seam must contain at least one grid edge")
        for name in ("a_pixels_rc", "b_pixels_rc"):
            value = getattr(self, name)
            if value.dtype != np.int64 or tuple(value.shape) != (count, 2):
                raise TypeError(name + " must be int64 [K,2]")
        for name in ("a_points_rc", "b_points_rc", "edge_midpoints_rc"):
            value = getattr(self, name)
            if value.dtype != np.float64 or tuple(value.shape) != (count, 2):
                raise TypeError(name + " must be float64 [K,2]")
        if self.segments_rc.dtype != np.float64 or tuple(self.segments_rc.shape) != (
            count,
            2,
            2,
        ):
            raise TypeError("segments_rc must be float64 [K,2,2]")
        if self.path_indices.dtype != np.int64 or tuple(self.path_indices.shape) != (
            count,
        ):
            raise TypeError("path_indices must be int64 [K]")
        for name in ("path_arclength_px", "path_arc_fractions"):
            value = getattr(self, name)
            if value.dtype != np.float64 or tuple(value.shape) != (count,):
                raise TypeError(name + " must be float64 [K]")
        if self.path_lengths_px.dtype != np.float64 or self.path_lengths_px.ndim != 1:
            raise TypeError("path_lengths_px must be float64 [P]")
        if len(self.path_lengths_px) < 1:
            raise ValueError("an exact seam needs at least one ordered path")
        if int(self.path_indices.max()) >= len(self.path_lengths_px):
            raise ValueError("path index exceeds path_lengths_px")
        for name in (
            "a_pixels_rc",
            "b_pixels_rc",
            "a_points_rc",
            "b_points_rc",
            "segments_rc",
            "edge_midpoints_rc",
            "path_indices",
            "path_arclength_px",
            "path_arc_fractions",
            "path_lengths_px",
        ):
            if getattr(self, name).flags.writeable:
                raise ValueError(name + " must be read-only")

    @property
    def correspondence_count(self) -> int:
        return int(self.a_pixels_rc.shape[0])

    @property
    def path_count(self) -> int:
        return int(self.path_lengths_px.shape[0])


_Vertex = Tuple[int, int]


@dataclass(frozen=True)
class _RawEdge:
    first: _Vertex
    second: _Vertex
    a_pixel: _Vertex
    b_pixel: _Vertex

    @property
    def key(self) -> Tuple[_Vertex, _Vertex]:
        return tuple(sorted((self.first, self.second)))  # type: ignore[return-value]


def _cross_child_edges(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[_RawEdge, ...]:
    edges: List[_RawEdge] = []
    # Scan the full canvas with vectorised boolean operations; Python work is
    # proportional only to seam length, not to all 640k pixels of an 800² mask.
    horizontal_a_left = np.argwhere(mask_a[:, :-1] & mask_b[:, 1:])
    horizontal_b_left = np.argwhere(mask_b[:, :-1] & mask_a[:, 1:])
    vertical_a_top = np.argwhere(mask_a[:-1, :] & mask_b[1:, :])
    vertical_b_top = np.argwhere(mask_b[:-1, :] & mask_a[1:, :])
    for row, column in horizontal_a_left:
        r, c = int(row), int(column)
        edges.append(_RawEdge((r, c + 1), (r + 1, c + 1), (r, c), (r, c + 1)))
    for row, column in horizontal_b_left:
        r, c = int(row), int(column)
        edges.append(_RawEdge((r, c + 1), (r + 1, c + 1), (r, c + 1), (r, c)))
    for row, column in vertical_a_top:
        r, c = int(row), int(column)
        edges.append(_RawEdge((r + 1, c), (r + 1, c + 1), (r, c), (r + 1, c)))
    for row, column in vertical_b_top:
        r, c = int(row), int(column)
        edges.append(_RawEdge((r + 1, c), (r + 1, c + 1), (r + 1, c), (r, c)))
    edges.sort(key=lambda value: (value.key, value.a_pixel, value.b_pixel))
    return tuple(edges)


def _ordered_edge_paths(
    edges: Sequence[_RawEdge],
) -> Tuple[Tuple[Tuple[int, _Vertex, _Vertex], ...], bool]:
    """Split an undirected grid-edge graph into deterministic maximal paths."""

    incident: Dict[_Vertex, List[int]] = {}
    for index, edge in enumerate(edges):
        incident.setdefault(edge.first, []).append(index)
        incident.setdefault(edge.second, []).append(index)
    for values in incident.values():
        values.sort()
    branch = any(len(values) > 2 for values in incident.values())
    unused = set(range(len(edges)))
    paths: List[Tuple[Tuple[int, _Vertex, _Vertex], ...]] = []

    def walk(
        start: _Vertex, first_index: int
    ) -> Tuple[Tuple[int, _Vertex, _Vertex], ...]:
        current = start
        edge_index = first_index
        path: List[Tuple[int, _Vertex, _Vertex]] = []
        while edge_index in unused:
            unused.remove(edge_index)
            edge = edges[edge_index]
            other = edge.second if current == edge.first else edge.first
            path.append((edge_index, current, other))
            current = other
            candidates = [value for value in incident[current] if value in unused]
            if len(incident[current]) != 2 or not candidates:
                break
            edge_index = min(candidates)
        return tuple(path)

    terminals = sorted(
        vertex for vertex, values in incident.items() if len(values) != 2
    )
    for vertex in terminals:
        for edge_index in incident[vertex]:
            if edge_index in unused:
                paths.append(walk(vertex, edge_index))
    while unused:
        first_index = min(unused)
        edge = edges[first_index]
        paths.append(walk(min(edge.first, edge.second), first_index))
    paths.sort(key=lambda path: (path[0][1], path[0][2], path[0][0]))
    return tuple(paths), branch


def extract_aligned_mask_seam(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    require_disjoint: bool = True,
) -> Optional[AlignedMaskSeam]:
    """Recover exact 4-neighbour A/B correspondences from aligned masks.

    ``None`` means the fragments share no unit grid edge.  Dilation proximity
    is intentionally not accepted as exact correspondence.
    """

    first = _binary_mask(mask_a, "mask_a")
    second = _binary_mask(mask_b, "mask_b")
    if first.shape != second.shape:
        raise ExactSeamSupervisionError("aligned masks must have the same shape")
    if require_disjoint and np.logical_and(first, second).any():
        raise ExactSeamSupervisionError("aligned fragment masks overlap")
    edges = _cross_child_edges(first, second)
    if not edges:
        return None
    paths, branch = _ordered_edge_paths(edges)
    ordered_records: List[Tuple[_RawEdge, _Vertex, _Vertex, int, float, float]] = []
    path_lengths = []
    for path_index, path in enumerate(paths):
        length = float(len(path))
        path_lengths.append(length)
        for ordinal, (edge_index, start, end) in enumerate(path):
            ordered_records.append(
                (
                    edges[edge_index],
                    start,
                    end,
                    path_index,
                    ordinal + 0.5,
                    (ordinal + 0.5) / length,
                )
            )
    a_pixels = np.asarray(
        [value[0].a_pixel for value in ordered_records], dtype=np.int64
    )
    b_pixels = np.asarray(
        [value[0].b_pixel for value in ordered_records], dtype=np.int64
    )
    segments = np.asarray(
        [[value[1], value[2]] for value in ordered_records], dtype=np.float64
    )
    return AlignedMaskSeam(
        a_pixels_rc=_readonly(a_pixels, np.int64),
        b_pixels_rc=_readonly(b_pixels, np.int64),
        a_points_rc=_readonly(a_pixels.astype(np.float64) + 0.5, np.float64),
        b_points_rc=_readonly(b_pixels.astype(np.float64) + 0.5, np.float64),
        segments_rc=_readonly(segments, np.float64),
        edge_midpoints_rc=_readonly(segments.mean(axis=1), np.float64),
        path_indices=_readonly(
            np.asarray([value[3] for value in ordered_records]), np.int64
        ),
        path_arclength_px=_readonly(
            np.asarray([value[4] for value in ordered_records]), np.float64
        ),
        path_arc_fractions=_readonly(
            np.asarray([value[5] for value in ordered_records]), np.float64
        ),
        path_lengths_px=_readonly(np.asarray(path_lengths), np.float64),
        parent_shape=tuple(int(value) for value in first.shape),
        has_branch_vertex=branch,
    )


@dataclass(frozen=True)
class ExactSeamTargetConfig:
    """Conservative seam-to-keypoint quantisation thresholds."""

    max_token_to_seam_distance_px: float = 1.0
    max_match_arclength_gap_px: float = 24.0
    projection_tie_tolerance_px: float = 1e-6
    positive_without_seam: str = "error"
    positive_candidate_without_match: str = "ignore"
    unmatched_near_seam: str = "ignore"

    def __post_init__(self) -> None:
        for name in (
            "max_token_to_seam_distance_px",
            "max_match_arclength_gap_px",
            "projection_tie_tolerance_px",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(name + " must be finite and non-negative")
        if self.max_token_to_seam_distance_px <= 0.0:
            raise ValueError("max_token_to_seam_distance_px must be positive")
        if self.max_match_arclength_gap_px <= 0.0:
            raise ValueError("max_match_arclength_gap_px must be positive")
        if self.positive_without_seam not in {"error", "ignore"}:
            raise ValueError("positive_without_seam must be error or ignore")
        if self.positive_candidate_without_match not in {"ignore", "dustbin"}:
            raise ValueError(
                "positive_candidate_without_match must be ignore or dustbin"
            )
        if self.unmatched_near_seam not in {"ignore", "dustbin"}:
            raise ValueError("unmatched_near_seam must be ignore or dustbin")


@dataclass(frozen=True)
class _TokenProjection:
    path_index: int
    arclength_px: float
    distance_px: float
    ambiguous: bool


def _project_tokens(
    tokens: Sequence[KeypointScaleToken],
    seam: Optional[AlignedMaskSeam],
    tie_tolerance: float,
) -> Tuple[_TokenProjection, ...]:
    if seam is None:
        return tuple(_TokenProjection(-1, math.nan, math.inf, False) for _ in tokens)
    starts = seam.segments_rc[:, 0]
    deltas = seam.segments_rc[:, 1] - starts
    squared_lengths = np.sum(deltas * deltas, axis=1)
    output = []
    for token in tokens:
        point = np.asarray(token.center_row_col, dtype=np.float64)
        parameters = np.clip(
            np.sum((point[None, :] - starts) * deltas, axis=1) / squared_lengths,
            0.0,
            1.0,
        )
        projected = starts + parameters[:, None] * deltas
        distances = np.linalg.norm(projected - point[None, :], axis=1)
        minimum = float(np.min(distances))
        tied = np.flatnonzero(distances <= minimum + tie_tolerance)
        best = int(tied[0])
        tied_paths = {int(seam.path_indices[index]) for index in tied}
        output.append(
            _TokenProjection(
                path_index=int(seam.path_indices[best]),
                arclength_px=float(
                    seam.path_arclength_px[best] - 0.5 + parameters[best]
                ),
                distance_px=minimum,
                ambiguous=len(tied_paths) > 1,
            )
        )
    return tuple(output)


def _maximum_cardinality_min_cost_matches(
    candidate: KeypointPairCandidate,
    projected_a: Sequence[_TokenProjection],
    projected_b: Sequence[_TokenProjection],
    config: ExactSeamTargetConfig,
) -> Tuple[Tuple[int, int], ...]:
    count_a = len(candidate.tokens_a)
    count_b = len(candidate.tokens_b)
    size = count_a + count_b
    cost = np.zeros((size, size), dtype=np.float64)
    cost[:count_a, :count_b] = 1e6
    valid_costs: Dict[Tuple[int, int], float] = {}
    for i, first in enumerate(projected_a):
        if first.ambiguous or first.distance_px > config.max_token_to_seam_distance_px:
            continue
        for j, second in enumerate(projected_b):
            if (
                not candidate.correspondence_mask[i, j]
                or second.ambiguous
                or second.distance_px > config.max_token_to_seam_distance_px
                or first.path_index != second.path_index
            ):
                continue
            gap = abs(first.arclength_px - second.arclength_px)
            if gap <= config.max_match_arclength_gap_px:
                valid_costs[(i, j)] = gap
    if not valid_costs:
        return ()
    # A negative unit reward makes cardinality dominate; the small normalised
    # term then selects the minimum-arclength solution among equal cardinalities.
    epsilon = 0.25 / float(max(1, min(count_a, count_b)))
    for (i, j), gap in valid_costs.items():
        cost[i, j] = -1.0 + epsilon * (gap / config.max_match_arclength_gap_px)
    rows, columns = linear_sum_assignment(cost)
    matches = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if row < count_a and column < count_b and (int(row), int(column)) in valid_costs
    ]
    matches.sort()
    return tuple(matches)


@dataclass(frozen=True)
class KeypointAssignmentTarget:
    """Tri-state partial-assignment target for one direction candidate."""

    candidate_id: str
    direction: PairDirection
    assignment_target_a: np.ndarray
    assignment_target_b: np.ndarray
    distance_to_seam_a_px: np.ndarray
    distance_to_seam_b_px: np.ndarray
    exact_seam_edge_count: int
    matched_token_pair_count: int

    def __post_init__(self) -> None:
        if (
            self.assignment_target_a.dtype != np.int64
            or self.assignment_target_a.ndim != 1
        ):
            raise TypeError("assignment_target_a must be int64 [La]")
        if (
            self.assignment_target_b.dtype != np.int64
            or self.assignment_target_b.ndim != 1
        ):
            raise TypeError("assignment_target_b must be int64 [Lb]")
        if (
            self.distance_to_seam_a_px.dtype != np.float64
            or self.distance_to_seam_a_px.shape != self.assignment_target_a.shape
        ):
            raise TypeError("distance_to_seam_a_px must be float64 [La]")
        if (
            self.distance_to_seam_b_px.dtype != np.float64
            or self.distance_to_seam_b_px.shape != self.assignment_target_b.shape
        ):
            raise TypeError("distance_to_seam_b_px must be float64 [Lb]")
        for i, value in enumerate(self.assignment_target_a):
            target = int(value)
            if target >= 0:
                if (
                    target >= len(self.assignment_target_b)
                    or int(self.assignment_target_b[target]) != i
                ):
                    raise ValueError("A/B assignment targets are not reciprocal")
            elif target not in {IGNORE_ASSIGNMENT, DUSTBIN_ASSIGNMENT}:
                raise ValueError("invalid A-side assignment sentinel")
        for j, value in enumerate(self.assignment_target_b):
            target = int(value)
            if target >= 0:
                if (
                    target >= len(self.assignment_target_a)
                    or int(self.assignment_target_a[target]) != j
                ):
                    raise ValueError("B/A assignment targets are not reciprocal")
            elif target not in {IGNORE_ASSIGNMENT, DUSTBIN_ASSIGNMENT}:
                raise ValueError("invalid B-side assignment sentinel")
        for name in (
            "assignment_target_a",
            "assignment_target_b",
            "distance_to_seam_a_px",
            "distance_to_seam_b_px",
        ):
            if getattr(self, name).flags.writeable:
                raise ValueError(name + " must be read-only")

    @property
    def supervised_token_count(self) -> int:
        # Count every match once from A, plus explicit dustbins on both sides.
        return int(
            np.count_nonzero(self.assignment_target_a >= 0)
            + np.count_nonzero(self.assignment_target_a == DUSTBIN_ASSIGNMENT)
            + np.count_nonzero(self.assignment_target_b == DUSTBIN_ASSIGNMENT)
        )


def build_keypoint_assignment_target(
    candidate: KeypointPairCandidate,
    seam: Optional[AlignedMaskSeam],
    *,
    pair_is_adjacent: bool,
    config: Optional[ExactSeamTargetConfig] = None,
) -> KeypointAssignmentTarget:
    """Project one exact aligned seam onto one label-blind candidate graph."""

    if not isinstance(candidate, KeypointPairCandidate):
        raise TypeError("candidate must be KeypointPairCandidate")
    if type(pair_is_adjacent) is not bool:
        raise TypeError("pair_is_adjacent must be bool")
    settings = config or ExactSeamTargetConfig()
    if pair_is_adjacent and seam is None and settings.positive_without_seam == "error":
        raise ExactSeamSupervisionError(
            "positive pair has no exact 4-neighbour seam; dilation is not mapping"
        )
    if not pair_is_adjacent and seam is not None:
        raise ExactSeamSupervisionError(
            "negative pair exposes an exact cross-fragment seam"
        )

    projected_a = _project_tokens(
        candidate.tokens_a, seam, settings.projection_tie_tolerance_px
    )
    projected_b = _project_tokens(
        candidate.tokens_b, seam, settings.projection_tie_tolerance_px
    )
    matches = (
        _maximum_cardinality_min_cost_matches(
            candidate, projected_a, projected_b, settings
        )
        if pair_is_adjacent and seam is not None
        else ()
    )
    if not pair_is_adjacent:
        target_a = np.full(len(candidate.tokens_a), DUSTBIN_ASSIGNMENT, np.int64)
        target_b = np.full(len(candidate.tokens_b), DUSTBIN_ASSIGNMENT, np.int64)
    elif not matches:
        fill = (
            IGNORE_ASSIGNMENT
            if settings.positive_candidate_without_match == "ignore"
            else DUSTBIN_ASSIGNMENT
        )
        target_a = np.full(len(candidate.tokens_a), fill, np.int64)
        target_b = np.full(len(candidate.tokens_b), fill, np.int64)
    else:
        target_a = np.full(len(candidate.tokens_a), DUSTBIN_ASSIGNMENT, np.int64)
        target_b = np.full(len(candidate.tokens_b), DUSTBIN_ASSIGNMENT, np.int64)
        if settings.unmatched_near_seam == "ignore":
            for index, projection in enumerate(projected_a):
                if (
                    projection.ambiguous
                    or projection.distance_px <= settings.max_token_to_seam_distance_px
                ):
                    target_a[index] = IGNORE_ASSIGNMENT
            for index, projection in enumerate(projected_b):
                if (
                    projection.ambiguous
                    or projection.distance_px <= settings.max_token_to_seam_distance_px
                ):
                    target_b[index] = IGNORE_ASSIGNMENT
        for i, j in matches:
            target_a[i] = j
            target_b[j] = i
    for i, j in matches:
        if not candidate.correspondence_mask[i, j]:
            raise RuntimeError(
                "exact target escaped the label-blind allowed-edge graph"
            )
    return KeypointAssignmentTarget(
        candidate_id=candidate.candidate_id,
        direction=candidate.direction,
        assignment_target_a=_readonly(target_a, np.int64),
        assignment_target_b=_readonly(target_b, np.int64),
        distance_to_seam_a_px=_readonly(
            np.asarray([value.distance_px for value in projected_a]), np.float64
        ),
        distance_to_seam_b_px=_readonly(
            np.asarray([value.distance_px for value in projected_b]), np.float64
        ),
        exact_seam_edge_count=0 if seam is None else seam.correspondence_count,
        matched_token_pair_count=len(matches),
    )


@dataclass(frozen=True)
class ExactSeamPairTargets:
    pair_key: str
    pair_is_adjacent: bool
    seam: Optional[AlignedMaskSeam]
    candidates: Tuple[KeypointAssignmentTarget, ...]


def build_pair_exact_seam_targets(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    keypoints: KeypointPairResult,
    *,
    pair_key: str,
    pair_is_adjacent: bool,
    config: Optional[ExactSeamTargetConfig] = None,
) -> ExactSeamPairTargets:
    """Build targets in exactly the candidate order used by keypoint batching."""

    if not isinstance(keypoints, KeypointPairResult):
        raise TypeError("keypoints must be KeypointPairResult")
    if not pair_key:
        raise ValueError("pair_key is required")
    seam = extract_aligned_mask_seam(mask_a, mask_b)
    targets = tuple(
        build_keypoint_assignment_target(
            candidate,
            seam,
            pair_is_adjacent=pair_is_adjacent,
            config=config,
        )
        for candidate in keypoints.candidates
    )
    return ExactSeamPairTargets(
        pair_key=pair_key,
        pair_is_adjacent=pair_is_adjacent,
        seam=seam,
        candidates=targets,
    )


def build_synthetic_pair_exact_seam_targets(
    pair: SyntheticMaskPairLike,
    keypoints: KeypointPairResult,
    config: Optional[ExactSeamTargetConfig] = None,
) -> ExactSeamPairTargets:
    """Thin adapter from the existing extra-synthetic pair record."""

    if not callable(getattr(pair, "load_masks", None)):
        raise TypeError("pair must provide load_masks()")
    for field in ("group_key", "fragment_a_id", "fragment_b_id", "label"):
        if not hasattr(pair, field):
            raise TypeError("pair is missing required field: " + field)
    mask_a, mask_b = pair.load_masks()
    pair_key = "{}/{}--{}".format(
        pair.group_key, pair.fragment_a_id, pair.fragment_b_id
    )
    return build_pair_exact_seam_targets(
        mask_a,
        mask_b,
        keypoints,
        pair_key=pair_key,
        pair_is_adjacent=pair.label,
        config=config,
    )


@dataclass(frozen=True)
class ExactSeamTensorBatch:
    assignment_target_a: Tensor
    assignment_target_b: Tensor
    sample_index: Tensor
    direction_index: Tensor
    candidate_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.assignment_target_a.dtype != torch.long
            or self.assignment_target_a.ndim != 2
        ):
            raise TypeError("assignment_target_a must be int64 [N,La]")
        if (
            self.assignment_target_b.dtype != torch.long
            or self.assignment_target_b.ndim != 2
        ):
            raise TypeError("assignment_target_b must be int64 [N,Lb]")
        count = int(self.assignment_target_a.shape[0])
        if int(self.assignment_target_b.shape[0]) != count:
            raise ValueError("A/B target candidate counts differ")
        for name, value in (
            ("sample_index", self.sample_index),
            ("direction_index", self.direction_index),
        ):
            if value.dtype != torch.long or tuple(value.shape) != (count,):
                raise TypeError(name + " must be int64 [N]")
        if len(self.candidate_ids) != count or len(set(self.candidate_ids)) != count:
            raise ValueError("candidate_ids must be unique and match N")
        if (self.assignment_target_a < IGNORE_ASSIGNMENT).any().item() or (
            self.assignment_target_b < IGNORE_ASSIGNMENT
        ).any().item():
            raise ValueError("assignment target contains an invalid sentinel")
        if (self.assignment_target_a >= self.assignment_target_b.shape[1]).any().item():
            raise ValueError("A target points beyond padded B tokens")
        if (self.assignment_target_b >= self.assignment_target_a.shape[1]).any().item():
            raise ValueError("B target points beyond padded A tokens")
        for candidate_index in range(count):
            for a_index in torch.nonzero(
                self.assignment_target_a[candidate_index] >= 0, as_tuple=False
            ).flatten():
                b_index = int(
                    self.assignment_target_a[candidate_index, int(a_index)].item()
                )
                if int(
                    self.assignment_target_b[candidate_index, b_index].item()
                ) != int(a_index):
                    raise ValueError(
                        "batched A/B assignment targets are not reciprocal"
                    )

    @property
    def candidate_count(self) -> int:
        return int(self.assignment_target_a.shape[0])

    def to(self, device: torch.device) -> "ExactSeamTensorBatch":
        return replace(
            self,
            assignment_target_a=self.assignment_target_a.to(device),
            assignment_target_b=self.assignment_target_b.to(device),
            sample_index=self.sample_index.to(device),
            direction_index=self.direction_index.to(device),
        )

    def assert_aligned_keypoint_batch(self, batch: Any) -> None:
        """Fail if targets and model inputs were flattened/padded differently."""

        if tuple(getattr(batch, "candidate_ids", ())) != self.candidate_ids:
            raise ValueError("seam targets do not align with keypoint candidate ids")
        if tuple(batch.local_a.shape[:2]) != tuple(self.assignment_target_a.shape):
            raise ValueError("seam A targets do not align with keypoint padding")
        if tuple(batch.local_b.shape[:2]) != tuple(self.assignment_target_b.shape):
            raise ValueError("seam B targets do not align with keypoint padding")
        if not torch.equal(batch.sample_index.cpu(), self.sample_index.cpu()):
            raise ValueError("seam targets changed sample ownership")
        if not torch.equal(batch.direction_index.cpu(), self.direction_index.cpu()):
            raise ValueError("seam targets changed direction order")


def build_exact_seam_tensor_batch(
    results: Sequence[ExactSeamPairTargets],
) -> ExactSeamTensorBatch:
    """Pad targets in the same sample/candidate order as keypoint tensor inputs."""

    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise TypeError("results must be a finite sequence")
    if not results or any(
        not isinstance(value, ExactSeamPairTargets) for value in results
    ):
        raise ValueError("results must contain ExactSeamPairTargets")
    flattened = [
        (sample_index, candidate)
        for sample_index, result in enumerate(results)
        for candidate in result.candidates
    ]
    if not flattened:
        raise ValueError("exact seam results contain no candidates")
    max_a = max(len(value.assignment_target_a) for _, value in flattened)
    max_b = max(len(value.assignment_target_b) for _, value in flattened)
    target_a = torch.full((len(flattened), max_a), IGNORE_ASSIGNMENT, dtype=torch.long)
    target_b = torch.full((len(flattened), max_b), IGNORE_ASSIGNMENT, dtype=torch.long)
    sample_indices = []
    direction_indices = []
    candidate_ids = []
    for index, (sample_index, candidate) in enumerate(flattened):
        target_a[index, : len(candidate.assignment_target_a)].copy_(
            torch.from_numpy(candidate.assignment_target_a.copy())
        )
        target_b[index, : len(candidate.assignment_target_b)].copy_(
            torch.from_numpy(candidate.assignment_target_b.copy())
        )
        sample_indices.append(sample_index)
        direction_indices.append(DEFAULT_DIRECTION_ORDER.index(candidate.direction))
        candidate_ids.append(
            "sample/{:06d}:{}".format(sample_index, candidate.candidate_id)
        )
    return ExactSeamTensorBatch(
        assignment_target_a=target_a,
        assignment_target_b=target_b,
        sample_index=torch.tensor(sample_indices, dtype=torch.long),
        direction_index=torch.tensor(direction_indices, dtype=torch.long),
        candidate_ids=tuple(candidate_ids),
    )


@dataclass(frozen=True)
class ExactPartialAssignmentLoss:
    total: Tensor
    match_nll: Tensor
    dustbin_a_nll: Tensor
    dustbin_b_nll: Tensor
    supervised_match_count: int
    supervised_dustbin_a_count: int
    supervised_dustbin_b_count: int


def _sample_balanced_masked_mean(
    value: Tensor, mask: Tensor, sample_index: Tensor
) -> Tensor:
    """Mean tokens per candidate, candidates per pair, then pairs per batch."""

    if tuple(value.shape) != tuple(mask.shape) or value.ndim != 2:
        raise ValueError("sample-balanced values and mask must be [N,L]")
    if sample_index.dtype != torch.long or tuple(sample_index.shape) != (
        int(value.shape[0]),
    ):
        raise TypeError("sample_index must be int64 [N]")
    counts = mask.sum(dim=1)
    candidate_valid = counts > 0
    candidate_mean = (value * mask.to(value.dtype)).sum(dim=1) / counts.clamp_min(1)
    pair_means = []
    for sample in torch.unique(sample_index, sorted=True):
        selected = (sample_index == sample) & candidate_valid
        if selected.any().item():
            pair_means.append(candidate_mean[selected].mean())
    return torch.stack(pair_means).mean() if pair_means else value.sum() * 0.0


def exact_partial_assignment_nll(
    output: LocalMatcherOutput,
    targets: ExactSeamTensorBatch,
    *,
    match_weight: float = 1.0,
    dustbin_weight: float = 1.0,
    epsilon: float = 1e-8,
) -> ExactPartialAssignmentLoss:
    """Supervise Sinkhorn real matches and both dustbins without teacher forcing."""

    if output.matcher_mode != "dustbin_sinkhorn" or output.transport is None:
        raise ValueError("exact partial assignment NLL requires dustbin Sinkhorn")
    for name, value in (
        ("match_weight", match_weight),
        ("dustbin_weight", dustbin_weight),
        ("epsilon", epsilon),
    ):
        if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0.0:
            raise ValueError(name + " must be finite and non-negative")
    if epsilon <= 0.0 or epsilon >= 1.0:
        raise ValueError("epsilon must be in (0,1)")
    assignment = output.assignment
    if tuple(targets.assignment_target_a.shape) != tuple(assignment.shape[:2]):
        raise ValueError("A targets do not match local assignment shape")
    if tuple(targets.assignment_target_b.shape) != (
        int(assignment.shape[0]),
        int(assignment.shape[2]),
    ):
        raise ValueError("B targets do not match local assignment shape")
    target_a = targets.assignment_target_a.to(device=assignment.device)
    target_b = targets.assignment_target_b.to(device=assignment.device)
    sample_index = targets.sample_index.to(device=assignment.device)
    valid = output.training_valid[:, None]

    match_mask = (target_a >= 0) & valid
    safe_indices = target_a.clamp(min=0, max=max(0, assignment.shape[2] - 1))
    match_probability = assignment.gather(2, safe_indices[:, :, None]).squeeze(2)
    match_nll_values = -torch.log(match_probability.clamp_min(epsilon))
    match_nll = _sample_balanced_masked_mean(match_nll_values, match_mask, sample_index)

    dustbin_a_mask = (target_a == DUSTBIN_ASSIGNMENT) & valid
    dustbin_b_mask = (target_b == DUSTBIN_ASSIGNMENT) & valid
    dustbin_a_values = -torch.log(output.unmatched_a.clamp_min(epsilon))
    dustbin_b_values = -torch.log(output.unmatched_b.clamp_min(epsilon))
    dustbin_a_nll = _sample_balanced_masked_mean(
        dustbin_a_values, dustbin_a_mask, sample_index
    )
    dustbin_b_nll = _sample_balanced_masked_mean(
        dustbin_b_values, dustbin_b_mask, sample_index
    )

    dustbin_parts = []
    if dustbin_a_mask.any().item():
        dustbin_parts.append(dustbin_a_nll)
    if dustbin_b_mask.any().item():
        dustbin_parts.append(dustbin_b_nll)
    dustbin_nll = (
        torch.stack(dustbin_parts).mean() if dustbin_parts else assignment.sum() * 0.0
    )
    total = float(match_weight) * match_nll + float(dustbin_weight) * dustbin_nll
    return ExactPartialAssignmentLoss(
        total=total,
        match_nll=match_nll,
        dustbin_a_nll=dustbin_a_nll,
        dustbin_b_nll=dustbin_b_nll,
        supervised_match_count=int(match_mask.sum().item()),
        supervised_dustbin_a_count=int(dustbin_a_mask.sum().item()),
        supervised_dustbin_b_count=int(dustbin_b_mask.sum().item()),
    )


def export_group_exact_seams(
    group_dir: Path,
    masks: Mapping[str, np.ndarray],
    *,
    filename: str = "exact_seams.npz",
) -> Path:
    """Persist compact parent-coordinate/arclength maps beside generated masks.

    The minimal generator integration point is immediately after
    ``get_submasks(mask)`` succeeds and before/after the existing PNG loop in
    ``iterate_genxvoronoi``.  The PNG masks remain the model input; this file is
    optional target metadata and can always be regenerated from aligned masks.
    """

    destination = Path(group_dir)
    if not destination.is_dir():
        raise ExactSeamSupervisionError("group_dir must already exist")
    identifiers = tuple(sorted(masks))
    if len(identifiers) < 2:
        raise ExactSeamSupervisionError("a group needs at least two masks")
    payload: Dict[str, np.ndarray] = {}
    pairs = []
    for first_id, second_id in combinations(identifiers, 2):
        seam = extract_aligned_mask_seam(masks[first_id], masks[second_id])
        if seam is None:
            continue
        prefix = "pair_{}_{}".format(first_id, second_id)
        payload[prefix + "_a_pixels_rc"] = seam.a_pixels_rc
        payload[prefix + "_b_pixels_rc"] = seam.b_pixels_rc
        payload[prefix + "_segments_rc"] = seam.segments_rc
        payload[prefix + "_path_indices"] = seam.path_indices
        payload[prefix + "_path_arclength_px"] = seam.path_arclength_px
        payload[prefix + "_path_lengths_px"] = seam.path_lengths_px
        pairs.append(
            {
                "fragment_a_id": first_id,
                "fragment_b_id": second_id,
                "prefix": prefix,
                "edge_count": seam.correspondence_count,
                "path_count": seam.path_count,
                "has_branch_vertex": seam.has_branch_vertex,
            }
        )
    metadata = {
        "schema_version": EXACT_SEAM_SCHEMA_VERSION,
        "coordinate_order": "row_col",
        "seam_definition": "all_4_neighbor_cross_child_grid_edges",
        "pairs": pairs,
    }
    payload["metadata_json_utf8"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    )
    output = destination / filename
    np.savez_compressed(str(output), **payload)
    return output


__all__ = [
    "AlignedMaskSeam",
    "DUSTBIN_ASSIGNMENT",
    "EXACT_SEAM_SCHEMA_VERSION",
    "ExactPartialAssignmentLoss",
    "ExactSeamPairTargets",
    "ExactSeamSupervisionError",
    "ExactSeamTargetConfig",
    "ExactSeamTensorBatch",
    "IGNORE_ASSIGNMENT",
    "KeypointAssignmentTarget",
    "build_exact_seam_tensor_batch",
    "build_keypoint_assignment_target",
    "build_pair_exact_seam_targets",
    "build_synthetic_pair_exact_seam_targets",
    "exact_partial_assignment_nll",
    "export_group_exact_seams",
    "extract_aligned_mask_seam",
]
