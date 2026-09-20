"""Deterministic mask-to-side preprocessing for B1 ``benchmark_v0_1``.

The implementation is deliberately independent of OpenCV and model code.  It
accepts only an in-memory Pillow image or NumPy array, resolves foreground
polarity with an explicit fail-closed policy, selects the largest connected
component, traces its external pixel-cell boundary, canonicalises the contour,
and partitions it into cardinal sides using bounding-box proximity.

All coordinates are ``[row, column]``.  Numeric preprocessing choices live in
``ContourPreprocessConfig`` so a later benchmark runner can select them on
train/validation and freeze them before test evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from PIL import Image


CONTOUR_VERSION = "largest-external-bbox-sides/v1"
SIDE_RUNS_VERSION = "cardinal-side-multi-run/0.1"
COORDINATE_ORDER = "row_col"


class ForegroundPolarity(str, Enum):
    """Which threshold class represents the manuscript fragment."""

    AUTO = "auto"
    BRIGHT = "bright"
    DARK = "dark"


class BoundarySaddlePolicy(str, Enum):
    """How the pixel-cell tracer handles checkerboard saddle vertices.

    ``REJECT`` preserves the frozen v0.1 behavior.  The opt-in
    ``FOREGROUND_4_BACKGROUND_8`` rule keeps diagonally touching foreground
    cells locally separate by turning each clockwise boundary edge to the
    right.  This is the deterministic digital-topology pairing for a
    4-connected foreground and an 8-connected background.
    """

    REJECT = "reject"
    FOREGROUND_4_BACKGROUND_8 = "foreground_4_background_8"


class ContourStatus(str, Enum):
    """B1-compatible preprocessing statuses."""

    OK = "ok"
    NO_FOREGROUND = "no_foreground"
    INVALID_CONTOUR = "invalid_contour"
    INSUFFICIENT_POINTS = "insufficient_points"


class CardinalSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


class SideRunExtractionStatus(str, Enum):
    """Fail-closed status for label-free cardinal candidate arcs."""

    OK = "ok"
    INVALID_CONTOUR = "invalid_contour"
    NO_USABLE_RUNS = "no_usable_runs"


MaskInput = Union[np.ndarray, Image.Image]


@dataclass(frozen=True)
class ContourPreprocessConfig:
    """Caller-visible pilot parameters; benchmark callers must freeze them."""

    polarity: ForegroundPolarity = ForegroundPolarity.AUTO
    threshold: Optional[float] = None
    auto_border_confidence: float = 0.75
    connectivity: int = 4
    saddle_policy: BoundarySaddlePolicy = BoundarySaddlePolicy.REJECT
    min_component_pixels: int = 4
    min_contour_points: int = 8
    min_side_points: int = 2
    contour_resample_count: int = 128
    side_resample_count: int = 32

    def __post_init__(self) -> None:
        try:
            ForegroundPolarity(self.polarity)
        except ValueError as exc:
            raise ValueError("unsupported foreground polarity") from exc
        if self.threshold is not None and not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if not 0.5 < self.auto_border_confidence <= 1.0:
            raise ValueError("auto_border_confidence must be in (0.5, 1]")
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        try:
            saddle_policy = BoundarySaddlePolicy(self.saddle_policy)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported boundary saddle policy") from exc
        if (
            saddle_policy is BoundarySaddlePolicy.FOREGROUND_4_BACKGROUND_8
            and self.connectivity != 4
        ):
            raise ValueError(
                "foreground_4_background_8 saddle policy requires connectivity=4"
            )
        if self.min_component_pixels < 1:
            raise ValueError("min_component_pixels must be positive")
        if self.min_contour_points < 4:
            raise ValueError("min_contour_points must be at least four")
        if self.min_side_points < 2:
            raise ValueError("min_side_points must be at least two")
        if self.contour_resample_count < 4:
            raise ValueError("contour_resample_count must be at least four")
        if self.side_resample_count < 2:
            raise ValueError("side_resample_count must be at least two")


@dataclass(frozen=True)
class ResampledPath:
    """Equidistant points with provenance back to the canonical contour."""

    points: np.ndarray
    source_contour_indices: np.ndarray
    contour_arc_fractions: np.ndarray
    closed: bool

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class CanonicalContour:
    """Largest external contour in canonical image-clockwise order."""

    points: np.ndarray
    contour_indices: np.ndarray
    arc_fractions: np.ndarray
    perimeter_px: float
    signed_area_px2: float
    winding: str
    start_rule: str
    version: str
    coordinate_order: str
    resampled: ResampledPath

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class ContourSide:
    """One contiguous bounding-box side with full-contour provenance."""

    side: CardinalSide
    points: np.ndarray
    contour_indices: np.ndarray
    contour_arc_fractions: np.ndarray
    wraps_contour_start: bool
    resampled: ResampledPath

    @property
    def original_point_count(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class ContourQuality:
    """Explicit, JSON-friendly quality and preprocessing metadata."""

    input_shape: Tuple[int, ...]
    input_dtype: str
    requested_polarity: str
    resolved_polarity: Optional[str] = None
    threshold: Optional[float] = None
    threshold_source: Optional[str] = None
    border_bright_fraction: Optional[float] = None
    foreground_pixel_count: int = 0
    foreground_fraction: float = 0.0
    component_count: int = 0
    component_areas_desc: Tuple[int, ...] = ()
    largest_component_pixels: int = 0
    discarded_foreground_pixels: int = 0
    largest_component_bbox_rc_exclusive: Optional[Tuple[int, int, int, int]] = None
    external_loop_count: int = 0
    contour_original_points: int = 0
    contour_resampled_points: int = 0
    contour_perimeter_px: Optional[float] = None
    side_original_points: Tuple[Tuple[str, int], ...] = ()
    side_resampled_points: Tuple[Tuple[str, int], ...] = ()
    side_run_counts: Tuple[Tuple[str, int], ...] = ()
    connectivity: int = 4
    contour_version: str = CONTOUR_VERSION
    coordinate_order: str = COORDINATE_ORDER
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "requested_polarity": self.requested_polarity,
            "resolved_polarity": self.resolved_polarity,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "border_bright_fraction": self.border_bright_fraction,
            "foreground_pixel_count": self.foreground_pixel_count,
            "foreground_fraction": self.foreground_fraction,
            "component_count": self.component_count,
            "component_areas_desc": list(self.component_areas_desc),
            "largest_component_pixels": self.largest_component_pixels,
            "discarded_foreground_pixels": self.discarded_foreground_pixels,
            "largest_component_bbox_rc_exclusive": (
                list(self.largest_component_bbox_rc_exclusive)
                if self.largest_component_bbox_rc_exclusive is not None
                else None
            ),
            "external_loop_count": self.external_loop_count,
            "contour_original_points": self.contour_original_points,
            "contour_resampled_points": self.contour_resampled_points,
            "contour_perimeter_px": self.contour_perimeter_px,
            "side_original_points": dict(self.side_original_points),
            "side_resampled_points": dict(self.side_resampled_points),
            "side_run_counts": dict(self.side_run_counts),
            "connectivity": self.connectivity,
            "contour_version": self.contour_version,
            "coordinate_order": self.coordinate_order,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ContourPreprocessResult:
    """Fail-closed B1 contour result; failures never expose partial geometry."""

    status: ContourStatus
    failure_reason: Optional[str]
    contour: Optional[CanonicalContour]
    sides: Mapping[CardinalSide, ContourSide]
    quality: ContourQuality

    def __post_init__(self) -> None:
        object.__setattr__(self, "sides", MappingProxyType(dict(self.sides)))
        if self.status is ContourStatus.OK:
            if self.failure_reason is not None or self.contour is None:
                raise ValueError("successful contour result requires geometry")
            if set(self.sides) != set(CardinalSide):
                raise ValueError("successful contour result requires four sides")
        elif self.contour is not None or self.sides:
            raise ValueError("failed contour result cannot expose geometry")

    @property
    def ok(self) -> bool:
        return self.status is ContourStatus.OK


@dataclass(frozen=True)
class ContourOnlyPreprocessResult:
    """Largest canonical contour without the strict four-side requirement."""

    status: ContourStatus
    failure_reason: Optional[str]
    contour: Optional[CanonicalContour]
    quality: ContourQuality

    def __post_init__(self) -> None:
        if self.status is ContourStatus.OK:
            if self.failure_reason is not None or self.contour is None:
                raise ValueError("successful contour-only result requires geometry")
        else:
            if self.contour is not None:
                raise ValueError("failed contour-only result cannot expose geometry")
            if self.failure_reason is None:
                raise ValueError("failed contour-only result requires a reason")

    @property
    def ok(self) -> bool:
        return self.status is ContourStatus.OK


@dataclass(frozen=True)
class SideRunExtractionQuality:
    """Auditable raw/retained cardinal-run counts without label selection."""

    contour_point_count: int
    min_side_points: int
    side_resample_count: int
    raw_run_counts: Tuple[Tuple[str, int], ...] = ()
    retained_run_counts: Tuple[Tuple[str, int], ...] = ()
    dropped_short_run_counts: Tuple[Tuple[str, int], ...] = ()
    raw_point_counts: Tuple[Tuple[str, int], ...] = ()
    retained_point_counts: Tuple[Tuple[str, int], ...] = ()
    dropped_short_point_counts: Tuple[Tuple[str, int], ...] = ()
    tie_order: Tuple[str, ...] = ("top", "right", "bottom", "left")
    coordinate_order: str = COORDINATE_ORDER
    version: str = SIDE_RUNS_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contour_point_count": self.contour_point_count,
            "min_side_points": self.min_side_points,
            "side_resample_count": self.side_resample_count,
            "raw_run_counts": dict(self.raw_run_counts),
            "retained_run_counts": dict(self.retained_run_counts),
            "dropped_short_run_counts": dict(self.dropped_short_run_counts),
            "raw_point_counts": dict(self.raw_point_counts),
            "retained_point_counts": dict(self.retained_point_counts),
            "dropped_short_point_counts": dict(
                self.dropped_short_point_counts
            ),
            "tie_order": list(self.tie_order),
            "coordinate_order": self.coordinate_order,
            "version": self.version,
        }


@dataclass(frozen=True)
class CardinalSideRunsResult:
    """All retained contiguous runs for every cardinal contour label."""

    status: SideRunExtractionStatus
    failure_reason: Optional[str]
    runs: Mapping[CardinalSide, Tuple[ContourSide, ...]]
    quality: SideRunExtractionQuality
    version: str = SIDE_RUNS_VERSION

    def __post_init__(self) -> None:
        copied = {
            side: tuple(side_runs) for side, side_runs in self.runs.items()
        }
        object.__setattr__(self, "runs", MappingProxyType(copied))
        if self.status is SideRunExtractionStatus.OK:
            if self.failure_reason is not None:
                raise ValueError("successful side-run result cannot have a reason")
            if set(copied) != set(CardinalSide):
                raise ValueError("successful side-run result requires four keys")
            if not any(copied.values()):
                raise ValueError("successful side-run result requires a retained run")
            for side, side_runs in copied.items():
                if any(run.side is not side for run in side_runs):
                    raise ValueError("side-run mapping contains a mismatched side")
        else:
            if copied:
                raise ValueError("failed side-run result cannot expose partial runs")
            if self.failure_reason is None:
                raise ValueError("failed side-run result requires a reason")

    @property
    def ok(self) -> bool:
        return self.status is SideRunExtractionStatus.OK

    @property
    def total_retained_run_count(self) -> int:
        return sum(len(side_runs) for side_runs in self.runs.values())


def _readonly(array: np.ndarray, dtype: Any) -> np.ndarray:
    output = np.asarray(array, dtype=dtype).copy()
    output.setflags(write=False)
    return output


def _failure(
    status: ContourStatus, reason: str, quality: ContourQuality
) -> ContourPreprocessResult:
    return ContourPreprocessResult(
        status=status,
        failure_reason=reason,
        contour=None,
        sides={},
        quality=quality,
    )


def _contour_only_failure(
    status: ContourStatus, reason: str, quality: ContourQuality
) -> ContourOnlyPreprocessResult:
    return ContourOnlyPreprocessResult(
        status=status,
        failure_reason=reason,
        contour=None,
        quality=quality,
    )


def _mask_array(mask: MaskInput) -> np.ndarray:
    if isinstance(mask, Image.Image):
        return np.asarray(mask)
    return np.asarray(mask)


def _otsu_or_midpoint(values: np.ndarray) -> Tuple[Optional[float], str]:
    unique, counts = np.unique(values, return_counts=True)
    if unique.size < 2:
        return None, "constant"
    if unique.size == 2:
        return float((float(unique[0]) + float(unique[1])) / 2.0), "binary_midpoint"

    weights = counts.astype(np.float64)
    cumulative_weight = np.cumsum(weights)
    cumulative_mean = np.cumsum(weights * unique.astype(np.float64))
    total_weight = cumulative_weight[-1]
    total_mean = cumulative_mean[-1]
    left_weight = cumulative_weight[:-1]
    right_weight = total_weight - left_weight
    valid = (left_weight > 0.0) & (right_weight > 0.0)
    between = np.full(unique.size - 1, -np.inf, dtype=np.float64)
    left_mean = cumulative_mean[:-1] / left_weight
    right_mean = (total_mean - cumulative_mean[:-1]) / right_weight
    between[valid] = (
        left_weight[valid]
        * right_weight[valid]
        * np.square(left_mean[valid] - right_mean[valid])
    )
    index = int(np.argmax(between))
    threshold = (float(unique[index]) + float(unique[index + 1])) / 2.0
    return threshold, "otsu"


def _border_values(bright: np.ndarray) -> np.ndarray:
    rows, columns = bright.shape
    if rows == 1 or columns == 1:
        return bright.reshape(-1)
    return np.concatenate(
        (
            bright[0, :],
            bright[-1, :],
            bright[1:-1, 0],
            bright[1:-1, -1],
        )
    )


def _connected_components(
    foreground: np.ndarray, connectivity: int
) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[int, int, int, int]]:
    rows, columns = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    if connectivity == 4:
        neighbours = ((-1, 0), (0, 1), (1, 0), (0, -1))
    else:
        neighbours = (
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        )

    best_pixels: List[int] = []
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    areas: List[int] = []
    for row, column in np.argwhere(foreground):
        row_i = int(row)
        column_i = int(column)
        if visited[row_i, column_i]:
            continue
        stack = [row_i * columns + column_i]
        visited[row_i, column_i] = True
        pixels: List[int] = []
        min_row = max_row = row_i
        min_column = max_column = column_i
        while stack:
            flat = stack.pop()
            current_row, current_column = divmod(flat, columns)
            pixels.append(flat)
            min_row = min(min_row, current_row)
            max_row = max(max_row, current_row)
            min_column = min(min_column, current_column)
            max_column = max(max_column, current_column)
            for delta_row, delta_column in neighbours:
                next_row = current_row + delta_row
                next_column = current_column + delta_column
                if not (0 <= next_row < rows and 0 <= next_column < columns):
                    continue
                if foreground[next_row, next_column] and not visited[
                    next_row, next_column
                ]:
                    visited[next_row, next_column] = True
                    stack.append(next_row * columns + next_column)
        bbox = (min_row, min_column, max_row + 1, max_column + 1)
        areas.append(len(pixels))
        if len(pixels) > len(best_pixels) or (
            len(pixels) == len(best_pixels)
            and (best_bbox is None or bbox < best_bbox)
        ):
            best_pixels = pixels
            best_bbox = bbox

    if best_bbox is None:
        raise RuntimeError("connected component search received an empty mask")
    largest = np.zeros_like(foreground, dtype=bool)
    largest.reshape(-1)[best_pixels] = True
    return largest, tuple(sorted(areas, reverse=True)), best_bbox


GridPoint = Tuple[int, int]
BoundaryEdge = Tuple[GridPoint, GridPoint]


def _add_boundary_edge(
    outgoing: Dict[GridPoint, GridPoint],
    incoming: Dict[GridPoint, GridPoint],
    start: GridPoint,
    end: GridPoint,
) -> bool:
    old_end = outgoing.get(start)
    old_start = incoming.get(end)
    if (old_end is not None and old_end != end) or (
        old_start is not None and old_start != start
    ):
        return False
    outgoing[start] = end
    incoming[end] = start
    return True


def _signed_area_row_col(points: np.ndarray) -> float:
    rows = points[:, 0]
    columns = points[:, 1]
    return float(
        0.5
        * np.sum(columns * np.roll(rows, -1) - np.roll(columns, -1) * rows)
    )


def _external_boundary(
    component: np.ndarray,
    saddle_policy: BoundarySaddlePolicy = BoundarySaddlePolicy.REJECT,
) -> Tuple[Optional[np.ndarray], Optional[str], int]:
    policy = BoundarySaddlePolicy(saddle_policy)
    if policy is BoundarySaddlePolicy.FOREGROUND_4_BACKGROUND_8:
        return _external_boundary_foreground_4_background_8(component)

    # This is the original v0.1 tracer, intentionally kept behavior-identical
    # behind the default REJECT policy so existing golden geometry and failure
    # receipts do not change.
    rows, columns = component.shape
    outgoing: Dict[GridPoint, GridPoint] = {}
    incoming: Dict[GridPoint, GridPoint] = {}
    for row, column in np.argwhere(component):
        row_i = int(row)
        column_i = int(column)
        edges: List[Tuple[GridPoint, GridPoint]] = []
        if row_i == 0 or not component[row_i - 1, column_i]:
            edges.append(((row_i, column_i), (row_i, column_i + 1)))
        if column_i + 1 == columns or not component[row_i, column_i + 1]:
            edges.append(
                ((row_i, column_i + 1), (row_i + 1, column_i + 1))
            )
        if row_i + 1 == rows or not component[row_i + 1, column_i]:
            edges.append(
                ((row_i + 1, column_i + 1), (row_i + 1, column_i))
            )
        if column_i == 0 or not component[row_i, column_i - 1]:
            edges.append(((row_i + 1, column_i), (row_i, column_i)))
        for start, end in edges:
            if not _add_boundary_edge(outgoing, incoming, start, end):
                return None, "non_manifold_boundary", 0

    if not outgoing or set(outgoing) != set(incoming):
        return None, "open_boundary", 0

    unused = set(outgoing)
    loops: List[np.ndarray] = []
    while unused:
        start = min(unused)
        current = start
        loop: List[GridPoint] = []
        for _ in range(len(outgoing) + 1):
            if current == start and loop:
                break
            if current not in unused:
                return None, "self_intersecting_boundary_graph", len(loops)
            unused.remove(current)
            loop.append(current)
            try:
                current = outgoing[current]
            except KeyError:
                return None, "open_boundary", len(loops)
        else:
            return None, "boundary_trace_limit_exceeded", len(loops)
        if current != start or len(loop) < 4:
            return None, "invalid_boundary_loop", len(loops)
        loops.append(np.asarray(loop, dtype=np.float64))

    external = [loop for loop in loops if _signed_area_row_col(loop) > 0.0]
    if not external:
        return None, "no_external_boundary", len(loops)
    external.sort(
        key=lambda loop: (
            -_signed_area_row_col(loop),
            float(np.min(loop[:, 0])),
            float(np.min(loop[:, 1])),
        )
    )
    return external[0], None, len(loops)


def _pixel_cell_boundary_edges(component: np.ndarray) -> Tuple[BoundaryEdge, ...]:
    """Return every clockwise exposed pixel-cell edge in canonical order."""

    rows, columns = component.shape
    edges: List[BoundaryEdge] = []
    for row, column in np.argwhere(component):
        row_i = int(row)
        column_i = int(column)
        if row_i == 0 or not component[row_i - 1, column_i]:
            edges.append(((row_i, column_i), (row_i, column_i + 1)))
        if column_i + 1 == columns or not component[row_i, column_i + 1]:
            edges.append(((row_i, column_i + 1), (row_i + 1, column_i + 1)))
        if row_i + 1 == rows or not component[row_i + 1, column_i]:
            edges.append(((row_i + 1, column_i + 1), (row_i + 1, column_i)))
        if column_i == 0 or not component[row_i, column_i - 1]:
            edges.append(((row_i + 1, column_i), (row_i, column_i)))
    return tuple(edges)


_CLOCKWISE_GRID_DIRECTION = {
    (0, 1): 0,
    (1, 0): 1,
    (0, -1): 2,
    (-1, 0): 3,
}


def _trace_foreground_4_background_8_loops(
    boundary_edges: Tuple[BoundaryEdge, ...],
) -> Tuple[Optional[List[np.ndarray]], Optional[str]]:
    """Trace edge cycles while resolving degree-four checkerboard saddles.

    Edges are sorted before graph construction, making the result independent
    of foreground-pixel or edge insertion order.  At a saddle, the unique
    clockwise right turn preserves the foreground on the right of each edge;
    equivalently, diagonal foreground cells remain 4-disconnected locally and
    diagonal background cells remain 8-connected.
    """

    edges = tuple(sorted(boundary_edges))
    if not edges:
        return None, "open_boundary"
    if len(set(edges)) != len(edges):
        return None, "duplicate_boundary_edge"

    outgoing: Dict[GridPoint, List[int]] = {}
    incoming: Dict[GridPoint, List[int]] = {}
    for edge_index, (start, end) in enumerate(edges):
        outgoing.setdefault(start, []).append(edge_index)
        incoming.setdefault(end, []).append(edge_index)
    if set(outgoing) != set(incoming):
        return None, "open_boundary"

    successors: Dict[int, int] = {}
    for vertex in sorted(outgoing):
        outgoing_indices = sorted(outgoing[vertex], key=lambda index: edges[index])
        incoming_indices = sorted(incoming[vertex], key=lambda index: edges[index])
        if len(outgoing_indices) != len(incoming_indices):
            return None, "open_boundary"
        if len(outgoing_indices) == 1:
            successors[incoming_indices[0]] = outgoing_indices[0]
            continue
        if len(outgoing_indices) != 2:
            return None, "unsupported_boundary_vertex_degree"

        used_outgoing = set()
        for incoming_index in incoming_indices:
            start, end = edges[incoming_index]
            incoming_direction = (end[0] - start[0], end[1] - start[1])
            try:
                incoming_clockwise = _CLOCKWISE_GRID_DIRECTION[incoming_direction]
            except KeyError:
                return None, "non_unit_boundary_edge"
            right_turns = []
            for outgoing_index in outgoing_indices:
                next_start, next_end = edges[outgoing_index]
                outgoing_direction = (
                    next_end[0] - next_start[0],
                    next_end[1] - next_start[1],
                )
                try:
                    outgoing_clockwise = _CLOCKWISE_GRID_DIRECTION[
                        outgoing_direction
                    ]
                except KeyError:
                    return None, "non_unit_boundary_edge"
                if (outgoing_clockwise - incoming_clockwise) % 4 == 1:
                    right_turns.append(outgoing_index)
            if len(right_turns) != 1 or right_turns[0] in used_outgoing:
                return None, "unresolvable_checkerboard_saddle"
            successors[incoming_index] = right_turns[0]
            used_outgoing.add(right_turns[0])
        if used_outgoing != set(outgoing_indices):
            return None, "unresolvable_checkerboard_saddle"

    if set(successors) != set(range(len(edges))) or set(successors.values()) != set(
        range(len(edges))
    ):
        return None, "open_boundary"

    unused = set(range(len(edges)))
    loops: List[np.ndarray] = []
    while unused:
        start_edge = min(unused, key=lambda index: edges[index])
        current_edge = start_edge
        loop: List[GridPoint] = []
        for _ in range(len(edges) + 1):
            if current_edge == start_edge and loop:
                break
            if current_edge not in unused:
                return None, "self_intersecting_boundary_graph"
            unused.remove(current_edge)
            loop.append(edges[current_edge][0])
            current_edge = successors[current_edge]
        else:
            return None, "boundary_trace_limit_exceeded"
        if current_edge != start_edge or len(loop) < 4:
            return None, "invalid_boundary_loop"
        points = np.asarray(loop, dtype=np.float64)
        if np.any(np.all(points == np.roll(points, -1, axis=0), axis=1)):
            return None, "duplicate_consecutive_boundary_point"
        loops.append(points)
    return loops, None


def _external_boundary_foreground_4_background_8(
    component: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[str], int]:
    loops, reason = _trace_foreground_4_background_8_loops(
        _pixel_cell_boundary_edges(component)
    )
    if loops is None:
        return None, reason or "external_boundary_failed", 0
    external = [loop for loop in loops if _signed_area_row_col(loop) > 0.0]
    if not external:
        return None, "no_external_boundary", len(loops)
    external.sort(
        key=lambda loop: (
            -_signed_area_row_col(loop),
            float(np.min(loop[:, 0])),
            float(np.min(loop[:, 1])),
        )
    )
    return external[0], None, len(loops)


def _canonicalize(points: np.ndarray) -> Tuple[np.ndarray, float]:
    output = np.asarray(points, dtype=np.float64)
    area = _signed_area_row_col(output)
    if area < 0.0:
        output = output[::-1]
        area = -area
    order = np.lexsort((output[:, 1], output[:, 0]))
    start_index = int(order[0])
    output = np.concatenate((output[start_index:], output[:start_index]), axis=0)
    return output, float(area)


def _arc_metadata(points: np.ndarray) -> Tuple[np.ndarray, float]:
    deltas = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(deltas, axis=1)
    perimeter = float(np.sum(lengths))
    if not np.isfinite(perimeter) or perimeter <= 0.0:
        raise ValueError("contour has zero or invalid perimeter")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    return cumulative / perimeter, perimeter


def _resample(
    points: np.ndarray,
    count: int,
    *,
    closed: bool,
    source_contour_indices: np.ndarray,
    contour_arc_fractions: np.ndarray,
) -> ResampledPath:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    minimum = 3 if closed else 2
    if len(points) < minimum:
        raise ValueError("path has insufficient points for resampling")
    if count < minimum:
        raise ValueError("resample count is too small")
    starts = points if closed else points[:-1]
    ends = np.roll(points, -1, axis=0) if closed else points[1:]
    lengths = np.linalg.norm(ends - starts, axis=1)
    if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
        raise ValueError("path contains duplicate or invalid consecutive points")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    targets = np.linspace(0.0, total, count, endpoint=not closed)
    segments = np.searchsorted(cumulative, targets, side="right") - 1
    segments = np.clip(segments, 0, len(lengths) - 1)
    alpha = (targets - cumulative[segments]) / lengths[segments]
    samples = starts[segments] + alpha[:, None] * (ends[segments] - starts[segments])

    fractions = np.asarray(contour_arc_fractions, dtype=np.float64).copy()
    for index in range(1, len(fractions)):
        while fractions[index] < fractions[index - 1]:
            fractions[index] += 1.0
    if closed:
        fraction_ends = np.concatenate((fractions[1:], [fractions[0] + 1.0]))
        fraction_starts = fractions
    else:
        fraction_starts = fractions[:-1]
        fraction_ends = fractions[1:]
    sample_fractions = (
        fraction_starts[segments]
        + alpha * (fraction_ends[segments] - fraction_starts[segments])
    ) % 1.0
    source_indices = source_contour_indices[segments]
    return ResampledPath(
        points=_readonly(samples, np.float64),
        source_contour_indices=_readonly(source_indices, np.int64),
        contour_arc_fractions=_readonly(sample_fractions, np.float64),
        closed=closed,
    )


def arc_length_resample(
    points: np.ndarray, count: int, *, closed: bool = False
) -> np.ndarray:
    """Public geometry helper for deterministic equidistant resampling.

    Standalone paths have local arc fractions and local point indices.  The
    richer provenance used by ``preprocess_mask`` is retained internally.
    """

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if closed:
        fractions, _ = _arc_metadata(array)
    else:
        if len(array) < 2:
            raise ValueError("open path needs at least two points")
        lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
        total = float(np.sum(lengths))
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("path has zero or invalid length")
        fractions = np.concatenate(([0.0], np.cumsum(lengths))) / total
    return _resample(
        array,
        count,
        closed=closed,
        source_contour_indices=np.arange(len(array), dtype=np.int64),
        contour_arc_fractions=fractions,
    ).points


def _circular_run_indices(selected: np.ndarray) -> List[np.ndarray]:
    count = len(selected)
    if not np.any(selected):
        return []
    if np.all(selected):
        return [np.arange(count, dtype=np.int64)]
    starts = [
        index
        for index in range(count)
        if selected[index] and not selected[(index - 1) % count]
    ]
    runs: List[np.ndarray] = []
    for start in starts:
        indices = []
        current = start
        while selected[current]:
            indices.append(current)
            current = (current + 1) % count
        runs.append(np.asarray(indices, dtype=np.int64))
    return runs


_LABEL_BY_SIDE = {
    CardinalSide.TOP: 0,
    CardinalSide.RIGHT: 1,
    CardinalSide.BOTTOM: 2,
    CardinalSide.LEFT: 3,
}


def _cardinal_labels(
    points: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Apply the versioned bbox-proximity rule shared by strict/multi-run APIs."""

    min_row = float(np.min(points[:, 0]))
    max_row = float(np.max(points[:, 0]))
    min_column = float(np.min(points[:, 1]))
    max_column = float(np.max(points[:, 1]))
    if min_row == max_row or min_column == max_column:
        return None, "degenerate_contour_bbox"

    # Tie order is versioned and deliberate: top, right, bottom, left.  It
    # assigns rectangle corners consistently while keeping each side circularly
    # contiguous under the canonical clockwise contour order when possible.
    distances = np.stack(
        (
            points[:, 0] - min_row,
            max_column - points[:, 1],
            max_row - points[:, 0],
            points[:, 1] - min_column,
        ),
        axis=1,
    )
    return np.argmin(distances, axis=1), None


def _partition_sides(
    points: np.ndarray,
    arc_fractions: np.ndarray,
    config: ContourPreprocessConfig,
) -> Tuple[
    Optional[Dict[CardinalSide, ContourSide]],
    Dict[CardinalSide, int],
    Dict[CardinalSide, int],
    Optional[str],
]:
    labels, label_reason = _cardinal_labels(points)
    if labels is None:
        return None, {}, {}, label_reason
    run_counts: Dict[CardinalSide, int] = {}
    point_counts: Dict[CardinalSide, int] = {}
    run_by_side: Dict[CardinalSide, np.ndarray] = {}
    for side in CardinalSide:
        runs = _circular_run_indices(labels == _LABEL_BY_SIDE[side])
        run_counts[side] = len(runs)
        point_counts[side] = sum(len(run) for run in runs)
        if len(runs) != 1:
            return None, point_counts, run_counts, "non_contiguous_bbox_side"
        if len(runs[0]) < config.min_side_points:
            return None, point_counts, run_counts, "side_has_insufficient_points"
        run_by_side[side] = runs[0]

    sides: Dict[CardinalSide, ContourSide] = {}
    for side in CardinalSide:
        indices = run_by_side[side]
        side_points = points[indices]
        side_fractions = arc_fractions[indices]
        try:
            resampled = _resample(
                side_points,
                config.side_resample_count,
                closed=False,
                source_contour_indices=indices,
                contour_arc_fractions=side_fractions,
            )
        except ValueError:
            return None, point_counts, run_counts, "side_resampling_failed"
        sides[side] = ContourSide(
            side=side,
            points=_readonly(side_points, np.float64),
            contour_indices=_readonly(indices, np.int64),
            contour_arc_fractions=_readonly(side_fractions, np.float64),
            wraps_contour_start=bool(
                len(indices) > 1 and np.any(np.diff(indices) < 0)
            ),
            resampled=resampled,
        )
    return sides, point_counts, run_counts, None


def _pairs(values: Mapping[CardinalSide, int]) -> Tuple[Tuple[str, int], ...]:
    return tuple((side.value, int(values.get(side, 0))) for side in CardinalSide)


def _side_run_failure(
    status: SideRunExtractionStatus,
    reason: str,
    quality: SideRunExtractionQuality,
) -> CardinalSideRunsResult:
    return CardinalSideRunsResult(
        status=status,
        failure_reason=reason,
        runs={},
        quality=quality,
    )


def extract_cardinal_side_runs(
    canonical_contour: CanonicalContour,
    *,
    min_side_points: int = 2,
    side_resample_count: int = 32,
) -> CardinalSideRunsResult:
    """Extract every ordered contiguous cardinal run without choosing one.

    The bbox-proximity labels and their tie order are identical to the strict
    historical B1 partition.  Runs shorter than ``min_side_points`` are
    counted and dropped; every other run is returned, in canonical contour
    traversal order, as an independent :class:`ContourSide`.  No pair label,
    ground-truth direction, score, or longest-run heuristic is consulted.
    """

    if not isinstance(canonical_contour, CanonicalContour):
        raise TypeError("canonical_contour must be CanonicalContour")
    if isinstance(min_side_points, bool) or not isinstance(min_side_points, int):
        raise TypeError("min_side_points must be an integer")
    if min_side_points < 2:
        raise ValueError("min_side_points must be at least two")
    if isinstance(side_resample_count, bool) or not isinstance(
        side_resample_count, int
    ):
        raise TypeError("side_resample_count must be an integer")
    if side_resample_count < 2:
        raise ValueError("side_resample_count must be at least two")

    points = np.asarray(canonical_contour.points)
    contour_indices = np.asarray(canonical_contour.contour_indices)
    arc_fractions = np.asarray(canonical_contour.arc_fractions)
    base_quality = SideRunExtractionQuality(
        contour_point_count=(int(points.shape[0]) if points.ndim > 0 else 0),
        min_side_points=min_side_points,
        side_resample_count=side_resample_count,
    )
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 4:
        return _side_run_failure(
            SideRunExtractionStatus.INVALID_CONTOUR,
            "contour_points_must_have_shape_n_by_2",
            base_quality,
        )
    if not np.issubdtype(points.dtype, np.number) or not np.isfinite(points).all():
        return _side_run_failure(
            SideRunExtractionStatus.INVALID_CONTOUR,
            "contour_points_must_be_finite_numeric",
            base_quality,
        )
    if contour_indices.shape != (len(points),) or arc_fractions.shape != (
        len(points),
    ):
        return _side_run_failure(
            SideRunExtractionStatus.INVALID_CONTOUR,
            "contour_provenance_shape_mismatch",
            base_quality,
        )
    if (
        not np.issubdtype(contour_indices.dtype, np.integer)
        or not np.issubdtype(arc_fractions.dtype, np.number)
        or not np.isfinite(arc_fractions).all()
        or not np.array_equal(
            contour_indices, np.arange(len(points), dtype=contour_indices.dtype)
        )
    ):
        return _side_run_failure(
            SideRunExtractionStatus.INVALID_CONTOUR,
            "contour_provenance_is_invalid",
            base_quality,
        )
    if canonical_contour.coordinate_order != COORDINATE_ORDER:
        return _side_run_failure(
            SideRunExtractionStatus.INVALID_CONTOUR,
            "unsupported_coordinate_order",
            base_quality,
        )

    labels, label_reason = _cardinal_labels(np.asarray(points, dtype=np.float64))
    if labels is None:
        return _side_run_failure(
            SideRunExtractionStatus.INVALID_CONTOUR,
            label_reason or "cardinal_labeling_failed",
            base_quality,
        )

    raw_runs: Dict[CardinalSide, List[np.ndarray]] = {}
    retained_runs: Dict[CardinalSide, List[np.ndarray]] = {}
    dropped_runs: Dict[CardinalSide, List[np.ndarray]] = {}
    for side in CardinalSide:
        side_runs = _circular_run_indices(labels == _LABEL_BY_SIDE[side])
        raw_runs[side] = side_runs
        retained_runs[side] = [
            run for run in side_runs if len(run) >= min_side_points
        ]
        dropped_runs[side] = [
            run for run in side_runs if len(run) < min_side_points
        ]

    quality = SideRunExtractionQuality(
        contour_point_count=len(points),
        min_side_points=min_side_points,
        side_resample_count=side_resample_count,
        raw_run_counts=_pairs(
            {side: len(raw_runs[side]) for side in CardinalSide}
        ),
        retained_run_counts=_pairs(
            {side: len(retained_runs[side]) for side in CardinalSide}
        ),
        dropped_short_run_counts=_pairs(
            {side: len(dropped_runs[side]) for side in CardinalSide}
        ),
        raw_point_counts=_pairs(
            {
                side: sum(len(run) for run in raw_runs[side])
                for side in CardinalSide
            }
        ),
        retained_point_counts=_pairs(
            {
                side: sum(len(run) for run in retained_runs[side])
                for side in CardinalSide
            }
        ),
        dropped_short_point_counts=_pairs(
            {
                side: sum(len(run) for run in dropped_runs[side])
                for side in CardinalSide
            }
        ),
    )
    if not any(retained_runs.values()):
        return _side_run_failure(
            SideRunExtractionStatus.NO_USABLE_RUNS,
            "all_cardinal_runs_below_minimum_points",
            quality,
        )

    output: Dict[CardinalSide, Tuple[ContourSide, ...]] = {}
    try:
        for side in CardinalSide:
            segments: List[ContourSide] = []
            for positions in retained_runs[side]:
                run_points = np.asarray(points[positions], dtype=np.float64)
                run_indices = np.asarray(
                    contour_indices[positions], dtype=np.int64
                )
                run_fractions = np.asarray(
                    arc_fractions[positions], dtype=np.float64
                )
                resampled = _resample(
                    run_points,
                    side_resample_count,
                    closed=False,
                    source_contour_indices=run_indices,
                    contour_arc_fractions=run_fractions,
                )
                segments.append(
                    ContourSide(
                        side=side,
                        points=_readonly(run_points, np.float64),
                        contour_indices=_readonly(run_indices, np.int64),
                        contour_arc_fractions=_readonly(
                            run_fractions, np.float64
                        ),
                        wraps_contour_start=bool(
                            len(positions) > 1
                            and np.any(np.diff(positions) < 0)
                        ),
                        resampled=resampled,
                    )
                )
            output[side] = tuple(segments)
    except (IndexError, ValueError):
        return _side_run_failure(
            SideRunExtractionStatus.INVALID_CONTOUR,
            "side_run_resampling_failed",
            quality,
        )

    return CardinalSideRunsResult(
        status=SideRunExtractionStatus.OK,
        failure_reason=None,
        runs=output,
        quality=quality,
    )


def preprocess_mask_contour_only(
    mask: MaskInput,
    config: Optional[ContourPreprocessConfig] = None,
) -> ContourOnlyPreprocessResult:
    """Extract the canonical full contour without requiring cardinal sides.

    Mask/data failures return a fail-closed result.  Invalid configuration is a
    programmer error and raises during config construction.  Thresholding,
    polarity resolution, largest-component selection, boundary tracing, and
    contour resampling are shared verbatim with :func:`preprocess_mask`.
    """

    settings = config or ContourPreprocessConfig()
    polarity = ForegroundPolarity(settings.polarity)
    try:
        raw = _mask_array(mask)
    except (TypeError, ValueError):
        raw = np.asarray([], dtype=np.float64)
        quality = ContourQuality(
            input_shape=(),
            input_dtype="unreadable",
            requested_polarity=polarity.value,
            connectivity=settings.connectivity,
        )
        return _contour_only_failure(
            ContourStatus.INVALID_CONTOUR, "mask_conversion_failed", quality
        )

    quality = ContourQuality(
        input_shape=tuple(int(value) for value in raw.shape),
        input_dtype=str(raw.dtype),
        requested_polarity=polarity.value,
        connectivity=settings.connectivity,
    )
    if raw.ndim != 2 or not raw.shape[0] or not raw.shape[1]:
        return _contour_only_failure(
            ContourStatus.INVALID_CONTOUR, "mask_must_be_nonempty_2d", quality
        )
    if raw.dtype.kind not in "buif":
        return _contour_only_failure(
            ContourStatus.INVALID_CONTOUR, "mask_dtype_is_not_numeric", quality
        )
    numeric = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        return _contour_only_failure(
            ContourStatus.INVALID_CONTOUR, "mask_contains_nonfinite_values", quality
        )

    if settings.threshold is None:
        threshold, threshold_source = _otsu_or_midpoint(numeric.reshape(-1))
    else:
        threshold = float(settings.threshold)
        threshold_source = "configured"
    if threshold is None:
        quality = replace(quality, threshold_source=threshold_source)
        return _contour_only_failure(
            ContourStatus.NO_FOREGROUND,
            "constant_mask_has_no_separable_foreground",
            quality,
        )

    bright = numeric > threshold
    border_bright_fraction = float(np.mean(_border_values(bright)))
    if polarity is ForegroundPolarity.AUTO:
        if border_bright_fraction >= settings.auto_border_confidence:
            resolved = ForegroundPolarity.DARK
        elif border_bright_fraction <= 1.0 - settings.auto_border_confidence:
            resolved = ForegroundPolarity.BRIGHT
        else:
            quality = replace(
                quality,
                threshold=threshold,
                threshold_source=threshold_source,
                border_bright_fraction=border_bright_fraction,
            )
            return _contour_only_failure(
                ContourStatus.INVALID_CONTOUR,
                "ambiguous_foreground_polarity",
                quality,
            )
    else:
        resolved = polarity

    foreground = bright if resolved is ForegroundPolarity.BRIGHT else ~bright
    foreground_pixels = int(np.count_nonzero(foreground))
    foreground_fraction = float(foreground_pixels / foreground.size)
    quality = replace(
        quality,
        resolved_polarity=resolved.value,
        threshold=threshold,
        threshold_source=threshold_source,
        border_bright_fraction=border_bright_fraction,
        foreground_pixel_count=foreground_pixels,
        foreground_fraction=foreground_fraction,
    )
    if foreground_pixels == 0:
        return _contour_only_failure(
            ContourStatus.NO_FOREGROUND, "no_foreground_pixels", quality
        )

    largest, component_areas, bbox = _connected_components(
        foreground, settings.connectivity
    )
    largest_pixels = int(component_areas[0])
    warnings: List[str] = []
    if len(component_areas) > 1:
        warnings.append("multiple_components_largest_selected")
    if bbox[0] == 0 or bbox[1] == 0 or bbox[2] == raw.shape[0] or bbox[3] == raw.shape[1]:
        warnings.append("largest_component_touches_image_border")
    if polarity is not ForegroundPolarity.AUTO:
        warnings.append("foreground_polarity_forced_by_config")
    quality = replace(
        quality,
        component_count=len(component_areas),
        component_areas_desc=component_areas,
        largest_component_pixels=largest_pixels,
        discarded_foreground_pixels=foreground_pixels - largest_pixels,
        largest_component_bbox_rc_exclusive=bbox,
        warnings=tuple(warnings),
    )
    if largest_pixels < settings.min_component_pixels:
        return _contour_only_failure(
            ContourStatus.INSUFFICIENT_POINTS,
            "largest_component_below_minimum_pixels",
            quality,
        )

    boundary, boundary_reason, loop_count = _external_boundary(
        largest, BoundarySaddlePolicy(settings.saddle_policy)
    )
    quality = replace(quality, external_loop_count=loop_count)
    if boundary is None:
        return _contour_only_failure(
            ContourStatus.INVALID_CONTOUR,
            boundary_reason or "external_boundary_failed",
            quality,
        )
    points, area = _canonicalize(boundary)
    quality = replace(quality, contour_original_points=len(points))
    if len(points) < settings.min_contour_points:
        return _contour_only_failure(
            ContourStatus.INSUFFICIENT_POINTS,
            "contour_below_minimum_points",
            quality,
        )
    try:
        arc_fractions, perimeter = _arc_metadata(points)
        indices = np.arange(len(points), dtype=np.int64)
        resampled_contour = _resample(
            points,
            settings.contour_resample_count,
            closed=True,
            source_contour_indices=indices,
            contour_arc_fractions=arc_fractions,
        )
    except ValueError:
        return _contour_only_failure(
            ContourStatus.INVALID_CONTOUR, "contour_resampling_failed", quality
        )
    quality = replace(
        quality,
        contour_resampled_points=resampled_contour.point_count,
        contour_perimeter_px=perimeter,
    )

    contour = CanonicalContour(
        points=_readonly(points, np.float64),
        contour_indices=_readonly(indices, np.int64),
        arc_fractions=_readonly(arc_fractions, np.float64),
        perimeter_px=perimeter,
        signed_area_px2=area,
        winding="clockwise_image_coordinates",
        start_rule="lexicographic_min_row_then_column",
        version=CONTOUR_VERSION,
        coordinate_order=COORDINATE_ORDER,
        resampled=resampled_contour,
    )
    return ContourOnlyPreprocessResult(
        status=ContourStatus.OK,
        failure_reason=None,
        contour=contour,
        quality=quality,
    )


def preprocess_mask(
    mask: MaskInput,
    config: Optional[ContourPreprocessConfig] = None,
) -> ContourPreprocessResult:
    """Extract strict historical B1 contour sides from an in-memory mask.

    This remains the backward-compatible single-run baseline.  It reuses the
    contour-only stage and then requires exactly one sufficiently long run for
    every cardinal side, preserving the prior fail-closed behavior.
    """

    settings = config or ContourPreprocessConfig()
    contour_only = preprocess_mask_contour_only(mask, settings)
    if not contour_only.ok:
        assert contour_only.failure_reason is not None
        return _failure(
            contour_only.status,
            contour_only.failure_reason,
            contour_only.quality,
        )
    assert contour_only.contour is not None
    contour = contour_only.contour
    sides, point_counts, run_counts, side_reason = _partition_sides(
        contour.points, contour.arc_fractions, settings
    )
    quality = replace(
        contour_only.quality,
        side_original_points=_pairs(point_counts),
        side_resampled_points=(
            _pairs(
                {
                    side: sides[side].resampled.point_count
                    for side in CardinalSide
                }
            )
            if sides is not None
            else ()
        ),
        side_run_counts=_pairs(run_counts),
    )
    if sides is None:
        status = (
            ContourStatus.INSUFFICIENT_POINTS
            if side_reason == "side_has_insufficient_points"
            else ContourStatus.INVALID_CONTOUR
        )
        return _failure(status, side_reason or "side_partition_failed", quality)
    return ContourPreprocessResult(
        status=ContourStatus.OK,
        failure_reason=None,
        contour=contour,
        sides=sides,
        quality=quality,
    )


__all__ = [
    "COORDINATE_ORDER",
    "CONTOUR_VERSION",
    "SIDE_RUNS_VERSION",
    "BoundarySaddlePolicy",
    "CardinalSideRunsResult",
    "CanonicalContour",
    "CardinalSide",
    "ContourOnlyPreprocessResult",
    "ContourPreprocessConfig",
    "ContourPreprocessResult",
    "ContourQuality",
    "ContourSide",
    "ContourStatus",
    "ForegroundPolarity",
    "ResampledPath",
    "SideRunExtractionQuality",
    "SideRunExtractionStatus",
    "arc_length_resample",
    "extract_cardinal_side_runs",
    "preprocess_mask",
    "preprocess_mask_contour_only",
]
