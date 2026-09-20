"""Rachel-only mask, contour, correspondence, and translation preprocessing.

This module deliberately has no dependency on the Shredding data or labels.
Rachel RGB files and the colocated ``label.csv`` are the only source inputs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy import ndimage
from scipy.spatial import cKDTree


class RachelPreprocessError(ValueError):
    """Rachel input or derived geometry violates the preprocessing contract."""


@dataclass(frozen=True)
class RachelPreprocessConfig:
    canvas_size: int = 800
    threshold: int = 8
    contour_cap: int = 512
    minimum_positive_seam: int = 64
    seam_bridge_radius: int = 2
    contour_smoothing_sigma: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "canvas_size",
            "contour_cap",
            "minimum_positive_seam",
            "seam_bridge_radius",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        if type(self.threshold) is not int or not 0 <= self.threshold <= 254:  # noqa: E721
            raise ValueError("threshold must be an integer in [0, 254]")
        if self.contour_cap < 4:
            raise ValueError("contour_cap must be at least four")
        if not np.isfinite(self.contour_smoothing_sigma):
            raise ValueError("contour_smoothing_sigma must be finite")


@dataclass(frozen=True)
class CenteredMask:
    model_mask: np.ndarray
    bbox_min_rc: Tuple[int, int]
    pad_start_rc: Tuple[int, int]
    parent_to_model_offset_rc: Tuple[int, int]


@dataclass(frozen=True)
class RachelFragment:
    fragment_id: str
    mask_id: str
    jpeg_path: Path
    parent_mask: np.ndarray
    model: CenteredMask
    dense_parent_contour: np.ndarray
    model_contour: np.ndarray
    contour_valid: np.ndarray
    foreground_area: int
    bbox_aspect_ratio: float


@dataclass(frozen=True)
class RachelPair:
    fragment_a_id: str
    fragment_b_id: str
    label: bool
    dense_correspondences: np.ndarray
    token_correspondences: np.ndarray
    seam_length_px: float
    accepted_seam_match_count: int
    main_training_eligible: bool
    selection_exclusion_reason: Optional[str]
    translation_a_to_b_rc: Optional[Tuple[float, float]]
    translation_a_to_b_xy: Optional[Tuple[float, float]]
    status: str = "accepted"
    quarantine_reason: Optional[str] = None


@dataclass(frozen=True)
class RachelGroup:
    generator: str
    group_id: str
    image_name: str
    fragments: Tuple[RachelFragment, ...]
    neighbor_edges: Tuple[Tuple[str, str], ...]
    pairs: Tuple[RachelPair, ...]


__all__ = [
    "CenteredMask",
    "RachelFragment",
    "RachelGroup",
    "RachelPair",
    "RachelPreprocessConfig",
    "RachelPreprocessError",
    "assign_lineage_splits",
    "centerpad_mask_with_transform",
    "extract_ordered_outer_contour",
    "read_rachel_group",
    "recover_mutual_contour_correspondences",
    "rgb_to_binary_mask",
    "translation_a_to_b",
]


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    output = np.ascontiguousarray(value, dtype=dtype)
    output.setflags(write=False)
    return output


def rgb_to_binary_mask(
    rgb: np.ndarray,
    config: RachelPreprocessConfig,
) -> np.ndarray:
    """Convert one Rachel RGB fragment to its filled paper-support silhouette."""

    array = np.asarray(rgb)
    if array.shape != (config.canvas_size, config.canvas_size, 3):
        raise RachelPreprocessError(
            "RGB fragment must have shape ({0}, {0}, 3)".format(config.canvas_size)
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise RachelPreprocessError("RGB fragment must use an integer dtype")
    foreground = np.max(array, axis=2) > config.threshold
    foreground = ndimage.binary_closing(
        foreground,
        structure=np.ones((3, 3), dtype=np.bool_),
        iterations=1,
        border_value=0,
    )
    labels, count = ndimage.label(
        foreground,
        structure=np.ones((3, 3), dtype=np.bool_),
    )
    if count < 1:
        raise RachelPreprocessError("RGB fragment has no foreground component")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    if sizes[largest] <= 0:
        raise RachelPreprocessError("RGB fragment has no non-empty component")
    mask = labels == largest
    mask = ndimage.binary_fill_holes(mask)
    return _readonly(mask, np.bool_)


GridPoint = Tuple[int, int]
BoundaryEdge = Tuple[GridPoint, GridPoint]


def _pixel_cell_boundary_edges(mask: np.ndarray) -> Tuple[BoundaryEdge, ...]:
    rows, columns = mask.shape
    edges = []
    top = mask & ~np.pad(mask[:-1], ((1, 0), (0, 0)), constant_values=False)
    right = mask & ~np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    bottom = mask & ~np.pad(mask[1:], ((0, 1), (0, 0)), constant_values=False)
    left = mask & ~np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    for row, column in np.argwhere(top):
        r, c = int(row), int(column)
        edges.append(((r, c), (r, c + 1)))
    for row, column in np.argwhere(right):
        r, c = int(row), int(column)
        edges.append(((r, c + 1), (r + 1, c + 1)))
    for row, column in np.argwhere(bottom):
        r, c = int(row), int(column)
        edges.append(((r + 1, c + 1), (r + 1, c)))
    for row, column in np.argwhere(left):
        r, c = int(row), int(column)
        edges.append(((r + 1, c), (r, c)))
    if rows == 0 or columns == 0:
        return ()
    return tuple(edges)


_CLOCKWISE_DIRECTION = {
    (0, 1): 0,
    (1, 0): 1,
    (0, -1): 2,
    (-1, 0): 3,
}


def _trace_boundary_loops(edges_value: Tuple[BoundaryEdge, ...]) -> Tuple[np.ndarray, ...]:
    edges = tuple(sorted(edges_value))
    if not edges or len(set(edges)) != len(edges):
        raise RachelPreprocessError("invalid or empty external boundary")
    outgoing: Dict[GridPoint, list] = {}
    incoming: Dict[GridPoint, list] = {}
    for index, (start, end) in enumerate(edges):
        outgoing.setdefault(start, []).append(index)
        incoming.setdefault(end, []).append(index)
    if set(outgoing) != set(incoming):
        raise RachelPreprocessError("open external boundary")
    successor: Dict[int, int] = {}
    for vertex in sorted(outgoing):
        outs = sorted(outgoing[vertex], key=lambda index: edges[index])
        ins = sorted(incoming[vertex], key=lambda index: edges[index])
        if len(outs) != len(ins) or len(outs) not in {1, 2}:
            raise RachelPreprocessError("unsupported external boundary vertex")
        if len(outs) == 1:
            successor[ins[0]] = outs[0]
            continue
        used = set()
        for incoming_index in ins:
            start, end = edges[incoming_index]
            direction = (end[0] - start[0], end[1] - start[1])
            incoming_clockwise = _CLOCKWISE_DIRECTION[direction]
            choices = []
            for outgoing_index in outs:
                next_start, next_end = edges[outgoing_index]
                next_direction = (
                    next_end[0] - next_start[0],
                    next_end[1] - next_start[1],
                )
                outgoing_clockwise = _CLOCKWISE_DIRECTION[next_direction]
                if (outgoing_clockwise - incoming_clockwise) % 4 == 1:
                    choices.append(outgoing_index)
            if len(choices) != 1 or choices[0] in used:
                raise RachelPreprocessError("ambiguous checkerboard boundary")
            successor[incoming_index] = choices[0]
            used.add(choices[0])
    unused = set(range(len(edges)))
    loops = []
    while unused:
        start = min(unused, key=lambda index: edges[index])
        current = start
        loop = []
        for _ in range(len(edges) + 1):
            if current == start and loop:
                break
            if current not in unused:
                raise RachelPreprocessError("self-intersecting external boundary")
            unused.remove(current)
            loop.append(current)
            current = successor[current]
        if current != start or len(loop) < 4:
            raise RachelPreprocessError("invalid external boundary loop")
        loops.append(np.asarray(loop, dtype=np.int64))
    return tuple(loops)


def _cartesian_signed_area(points_rc: np.ndarray) -> float:
    rows = points_rc[:, 0]
    columns = points_rc[:, 1]
    return float(
        0.5
        * np.sum(
            columns * np.roll(-rows, -1)
            - np.roll(columns, -1) * (-rows)
        )
    )


def _edge_foreground_pixel(edge: BoundaryEdge) -> GridPoint:
    (row, column), (next_row, next_column) = edge
    direction = (next_row - row, next_column - column)
    if direction == (0, 1):
        return row, column
    if direction == (1, 0):
        return row, column - 1
    if direction == (0, -1):
        return row - 1, column - 1
    if direction == (-1, 0):
        return row - 1, column
    raise RachelPreprocessError("non-unit boundary edge")


def _dense_external_contour(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=np.bool_)
    if array.ndim != 2 or not np.any(array):
        raise RachelPreprocessError("mask must be a non-empty 2-D array")
    edges = _pixel_cell_boundary_edges(array)
    loops = _trace_boundary_loops(edges)
    edge_lookup = tuple(sorted(edges))
    candidates = []
    for loop in loops:
        vertices = np.asarray([edge_lookup[int(index)][0] for index in loop])
        area = abs(_cartesian_signed_area(vertices.astype(np.float64)))
        candidates.append((area, loop))
    _, external = max(candidates, key=lambda item: item[0])
    points = []
    for edge_index in external:
        point = _edge_foreground_pixel(edge_lookup[int(edge_index)])
        if not points or point != points[-1]:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    # Thin/saddle pixels can appear twice on a cell-edge trace.  Keep their
    # first traversal; ordinary paper silhouettes remain unchanged.
    unique_points = []
    seen = set()
    for point in points:
        if point not in seen:
            unique_points.append(point)
            seen.add(point)
    output = np.asarray(unique_points, dtype=np.float64)
    if len(output) < 4:
        raise RachelPreprocessError("external contour has fewer than four points")
    if _cartesian_signed_area(output) < 0.0:
        output = output[::-1]
    order = np.lexsort((output[:, 1], output[:, 0]))
    start = int(order[0])
    output = np.concatenate((output[start:], output[:start]), axis=0)
    return output


def _arc_resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    starts = np.asarray(points, dtype=np.float64)
    ends = np.roll(starts, -1, axis=0)
    lengths = np.linalg.norm(ends - starts, axis=1)
    if np.any(lengths <= 0.0):
        raise RachelPreprocessError("contour has duplicate consecutive points")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    targets = np.arange(count, dtype=np.float64) * total / float(count)
    segments = np.searchsorted(cumulative, targets, side="right") - 1
    segments = np.clip(segments, 0, len(lengths) - 1)
    alpha = (targets - cumulative[segments]) / lengths[segments]
    return starts[segments] + alpha[:, None] * (ends[segments] - starts[segments])


def extract_ordered_outer_contour(
    mask: np.ndarray,
    *,
    cap: int = 512,
    smoothing_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the complete external CCW contour, uniformly capped by arc length."""

    if type(cap) is not int or cap < 4:  # noqa: E721
        raise ValueError("cap must be an integer of at least four")
    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0.0:
        raise ValueError("smoothing_sigma must be finite and non-negative")
    dense = _dense_external_contour(mask)
    if smoothing_sigma > 0.0:
        dense = ndimage.gaussian_filter1d(
            dense,
            sigma=float(smoothing_sigma),
            axis=0,
            mode="wrap",
        )
    points = dense if len(dense) <= cap else _arc_resample_closed(dense, cap)
    valid = np.ones(len(points), dtype=np.bool_)
    return _readonly(points, np.float32), _readonly(valid, np.bool_)


def recover_mutual_contour_correspondences(
    contour_a: np.ndarray,
    contour_b: np.ndarray,
    *,
    max_distance_px: float = 3.0,
) -> np.ndarray:
    """Return reciprocal nearest-neighbour pairs on two complete contours."""

    first = np.asarray(contour_a, dtype=np.float64)
    second = np.asarray(contour_b, dtype=np.float64)
    if first.ndim != 2 or first.shape[1:] != (2,):
        raise ValueError("contour_a must have shape [N,2]")
    if second.ndim != 2 or second.shape[1:] != (2,):
        raise ValueError("contour_b must have shape [M,2]")
    if len(first) == 0 or len(second) == 0:
        return _readonly(np.empty((0, 2), dtype=np.int64), np.int64)
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("contours must be finite")
    if not np.isfinite(max_distance_px) or max_distance_px <= 0.0:
        raise ValueError("max_distance_px must be positive and finite")
    distance_ab, index_ab = cKDTree(second).query(first, k=1)
    _, index_ba = cKDTree(first).query(second, k=1)
    matches = [
        (index_a, int(index_b))
        for index_a, (distance, index_b) in enumerate(zip(distance_ab, index_ab))
        if float(distance) <= float(max_distance_px)
        and int(index_ba[int(index_b)]) == index_a
    ]
    return _readonly(np.asarray(matches, dtype=np.int64).reshape(-1, 2), np.int64)


def _accepted_continuous_seam(
    matches: np.ndarray,
    contour_a: np.ndarray,
    contour_b: np.ndarray,
    *,
    minimum_run: int = 4,
) -> np.ndarray:
    """Remove isolated/corner coincidences while retaining reverse seam runs."""

    pairs = np.asarray(matches, dtype=np.int64)
    if len(pairs) < minimum_run:
        return _readonly(np.empty((0, 2), dtype=np.int64), np.int64)
    size_a, size_b = len(contour_a), len(contour_b)
    pair_set = {tuple(pair) for pair in pairs.tolist()}
    retained = set()
    for pair in sorted(pair_set):
        if ((pair[0] - 1) % size_a, (pair[1] + 1) % size_b) in pair_set:
            continue
        run = []
        current = pair
        for _ in range(len(pair_set) + 1):
            if current not in pair_set or current in run:
                break
            run.append(current)
            current = ((current[0] + 1) % size_a, (current[1] - 1) % size_b)
        if len(run) >= minimum_run:
            retained.update(run)
    # Also accept the opposite traversal convention.  The canonical contour
    # is CCW, so complementary seams normally use the branch above, but this
    # keeps the function stable for externally supplied contours.
    for pair in sorted(pair_set):
        if ((pair[0] - 1) % size_a, (pair[1] - 1) % size_b) in pair_set:
            continue
        run = []
        current = pair
        for _ in range(len(pair_set) + 1):
            if current not in pair_set or current in run:
                break
            run.append(current)
            current = ((current[0] + 1) % size_a, (current[1] + 1) % size_b)
        if len(run) >= minimum_run:
            retained.update(run)
    return _readonly(np.asarray(sorted(retained), dtype=np.int64).reshape(-1, 2), np.int64)


def centerpad_mask_with_transform(
    parent_mask: np.ndarray,
    *,
    canvas_size: int = 800,
) -> CenteredMask:
    """Tight-crop and centre-pad without resizing or exposing parent origin."""

    mask = np.asarray(parent_mask, dtype=np.bool_)
    if mask.ndim != 2 or not np.any(mask):
        raise RachelPreprocessError("parent mask must be a non-empty 2-D array")
    if type(canvas_size) is not int or canvas_size <= 0:  # noqa: E721
        raise ValueError("canvas_size must be a positive integer")
    coordinates = np.argwhere(mask)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0) + 1
    crop = mask[minimum[0] : maximum[0], minimum[1] : maximum[1]]
    if crop.shape[0] > canvas_size or crop.shape[1] > canvas_size:
        raise RachelPreprocessError("tight crop does not fit model canvas")
    pad_start = np.asarray(
        ((canvas_size - crop.shape[0]) // 2, (canvas_size - crop.shape[1]) // 2),
        dtype=np.int64,
    )
    output = np.zeros((canvas_size, canvas_size), dtype=np.bool_)
    output[
        pad_start[0] : pad_start[0] + crop.shape[0],
        pad_start[1] : pad_start[1] + crop.shape[1],
    ] = crop
    offset = pad_start - minimum
    return CenteredMask(
        model_mask=_readonly(output, np.bool_),
        bbox_min_rc=(int(minimum[0]), int(minimum[1])),
        pad_start_rc=(int(pad_start[0]), int(pad_start[1])),
        parent_to_model_offset_rc=(int(offset[0]), int(offset[1])),
    )


def translation_a_to_b(first: CenteredMask, second: CenteredMask) -> np.ndarray:
    """Return A→B translation in image coordinates ``(delta_row, delta_col)``."""

    return _readonly(
        np.asarray(second.parent_to_model_offset_rc, dtype=np.float64)
        - np.asarray(first.parent_to_model_offset_rc, dtype=np.float64),
        np.float64,
    )


def assign_lineage_splits(
    image_names: Iterable[str],
    *,
    seed: str = "rachel-pairwise-n512-v1",
) -> Dict[str, str]:
    """Assign unique source manuscripts to an exact count-based 80/10/10 split."""

    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a non-empty string")
    names = sorted(set(image_names))
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("image_names must contain non-empty strings")
    ranked = sorted(
        names,
        key=lambda name: (
            hashlib.sha256((seed + "\0" + name).encode("utf-8")).digest(),
            name,
        ),
    )
    train_count = int(0.8 * len(ranked))
    val_count = int(0.1 * len(ranked))
    assignments = {}
    for index, name in enumerate(ranked):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        assignments[name] = split
    return dict(sorted(assignments.items()))


_REQUIRED_CSV_FIELDS = {
    "id",
    "mask_id",
    "image_name",
    "center_x",
    "center_y",
    "width",
    "height",
    "neighbors",
}


def _natural_key(value: str) -> Tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
        if part
    )


def _parse_neighbors(value: object) -> Tuple[str, ...]:
    text = "" if value is None else str(value).strip()
    if not text:
        return ()
    text = text.strip("[](){}")
    tokens = [token.strip(" \t\r\n'\"") for token in re.split(r"[,;|\s]+", text)]
    return tuple(sorted((token for token in tokens if token), key=_natural_key))


def _read_csv_rows(group_path: Path) -> Tuple[dict, ...]:
    csv_path = group_path / "label.csv"
    if not csv_path.is_file():
        raise RachelPreprocessError("group is missing label.csv")
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or set(reader.fieldnames) != _REQUIRED_CSV_FIELDS:
                raise RachelPreprocessError("label.csv has an invalid schema")
            rows = tuple(dict(row) for row in reader)
    except (OSError, csv.Error) as error:
        raise RachelPreprocessError("cannot read label.csv") from error
    if len(rows) < 2:
        raise RachelPreprocessError("label.csv must contain at least two fragments")
    return rows


def _validated_csv_graph(
    rows: Tuple[dict, ...],
) -> Tuple[str, Dict[str, dict], Tuple[Tuple[str, str], ...]]:
    by_id: Dict[str, dict] = {}
    mask_ids: Dict[str, str] = {}
    image_names = set()
    for row in rows:
        fragment_id = str(row.get("id", "")).strip()
        mask_id = str(row.get("mask_id", "")).strip()
        image_name = str(row.get("image_name", "")).strip()
        if not fragment_id or fragment_id in by_id:
            raise RachelPreprocessError("CSV id is missing or duplicated")
        if not mask_id or mask_id in mask_ids:
            raise RachelPreprocessError("CSV mask_id is missing or duplicated")
        if not image_name:
            raise RachelPreprocessError("CSV image_name is missing")
        by_id[fragment_id] = row
        mask_ids[mask_id] = fragment_id
        image_names.add(image_name)
    if len(image_names) != 1:
        raise RachelPreprocessError("CSV image_name is inconsistent within group")

    directed = set()
    for fragment_id, row in by_id.items():
        for neighbor in _parse_neighbors(row.get("neighbors")):
            if neighbor == fragment_id:
                raise RachelPreprocessError("neighbor self-loop is forbidden")
            if neighbor not in by_id:
                raise RachelPreprocessError("neighbor references an unknown id")
            directed.add((fragment_id, neighbor))
    for first, second in directed:
        if (second, first) not in directed:
            raise RachelPreprocessError("asymmetric neighbor relation in label.csv")
    undirected = tuple(
        sorted(
            {
                tuple(sorted((first, second), key=_natural_key))
                for first, second in directed
            },
            key=lambda edge: (_natural_key(edge[0]), _natural_key(edge[1])),
        )
    )
    return next(iter(image_names)), by_id, undirected


def _load_fragment(
    row: Mapping[str, str],
    *,
    jpeg_path: Path,
    config: RachelPreprocessConfig,
) -> RachelFragment:
    try:
        with Image.open(jpeg_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as error:
        raise RachelPreprocessError("cannot decode JPEG fragment") from error
    parent_mask = rgb_to_binary_mask(rgb, config)
    model = centerpad_mask_with_transform(parent_mask, canvas_size=config.canvas_size)
    dense = _readonly(_dense_external_contour(parent_mask), np.float32)
    model_contour, contour_valid = extract_ordered_outer_contour(
        model.model_mask,
        cap=config.contour_cap,
        smoothing_sigma=config.contour_smoothing_sigma,
    )
    coordinates = np.argwhere(parent_mask)
    extent = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
    aspect = float(extent[1]) / float(extent[0])
    return RachelFragment(
        fragment_id=str(row["id"]).strip(),
        mask_id=str(row["mask_id"]).strip(),
        jpeg_path=jpeg_path,
        parent_mask=parent_mask,
        model=model,
        dense_parent_contour=dense,
        model_contour=model_contour,
        contour_valid=contour_valid,
        foreground_area=int(np.count_nonzero(parent_mask)),
        bbox_aspect_ratio=aspect,
    )


def _token_correspondences(
    first: RachelFragment,
    second: RachelFragment,
    *,
    max_distance_px: float,
    dense_matches: np.ndarray,
) -> np.ndarray:
    first_parent = first.model_contour - np.asarray(
        first.model.parent_to_model_offset_rc, dtype=np.float32
    )
    second_parent = second.model_contour - np.asarray(
        second.model.parent_to_model_offset_rc, dtype=np.float32
    )
    if len(dense_matches) == 0:
        return _readonly(np.empty((0, 2), dtype=np.int64), np.int64)
    seam_a = first.dense_parent_contour[dense_matches[:, 0]]
    seam_b = second.dense_parent_contour[dense_matches[:, 1]]
    distance_a, token_a = cKDTree(first_parent).query(seam_a, k=1)
    distance_b, token_b = cKDTree(second_parent).query(seam_b, k=1)
    candidates = sorted(
        (
            (float(error_a + error_b), int(index_a), int(index_b))
            for error_a, error_b, index_a, index_b in zip(
                distance_a, distance_b, token_a, token_b
            )
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    used_a = set()
    used_b = set()
    selected = []
    for _, index_a, index_b in candidates:
        if index_a in used_a or index_b in used_b:
            continue
        selected.append((index_a, index_b))
        used_a.add(index_a)
        used_b.add(index_b)
    return _readonly(
        np.asarray(sorted(selected), dtype=np.int64).reshape(-1, 2),
        np.int64,
    )


def _seam_arc_length(contour: np.ndarray, matches: np.ndarray) -> float:
    """Measure accepted seam support in physical parent-canvas pixels."""

    if len(matches) == 0:
        return 0.0
    points = np.asarray(contour, dtype=np.float64)
    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    point_weights = 0.5 * (
        np.linalg.norm(points - previous, axis=1)
        + np.linalg.norm(following - points, axis=1)
    )
    return float(np.sum(point_weights[np.unique(matches[:, 0])]))


def _make_pair(
    first: RachelFragment,
    second: RachelFragment,
    *,
    label: bool,
    config: RachelPreprocessConfig,
) -> RachelPair:
    tolerance = float(config.seam_bridge_radius + 1)
    raw = recover_mutual_contour_correspondences(
        first.dense_parent_contour,
        second.dense_parent_contour,
        max_distance_px=tolerance,
    )
    accepted = _accepted_continuous_seam(
        raw,
        first.dense_parent_contour,
        second.dense_parent_contour,
    )
    seam_length = _seam_arc_length(first.dense_parent_contour, accepted)
    status = "accepted"
    quarantine_reason = None
    exclusion_reason = None
    eligible = True
    if label and len(accepted) == 0:
        status = "quarantined"
        quarantine_reason = "positive has no reliable local contour correspondence"
        exclusion_reason = "quarantined_positive_without_correspondence"
        eligible = False
    elif not label and len(accepted) > 0:
        status = "quarantined"
        quarantine_reason = "negative has reliable local contour correspondence"
        exclusion_reason = "quarantined_negative_with_correspondence"
        eligible = False
    elif label and seam_length < float(config.minimum_positive_seam):
        eligible = False
        exclusion_reason = "positive_seam_shorter_than_{}_pixels".format(
            config.minimum_positive_seam
        )

    token_matches = (
        _token_correspondences(
            first,
            second,
            max_distance_px=tolerance,
            dense_matches=accepted,
        )
        if label and len(accepted) > 0
        else _readonly(np.empty((0, 2), dtype=np.int64), np.int64)
    )
    if label and status == "accepted" and eligible and len(token_matches) == 0:
        eligible = False
        exclusion_reason = "positive_has_no_n512_correspondence"
    translation_rc_value = translation_a_to_b(first.model, second.model)
    translation_rc = (
        (float(translation_rc_value[0]), float(translation_rc_value[1]))
        if label
        else None
    )
    translation_xy = (
        (float(translation_rc_value[1]), float(-translation_rc_value[0]))
        if label
        else None
    )
    return RachelPair(
        fragment_a_id=first.fragment_id,
        fragment_b_id=second.fragment_id,
        label=label,
        dense_correspondences=accepted,
        token_correspondences=token_matches,
        seam_length_px=seam_length,
        accepted_seam_match_count=len(accepted),
        main_training_eligible=eligible,
        selection_exclusion_reason=exclusion_reason,
        translation_a_to_b_rc=translation_rc,
        translation_a_to_b_xy=translation_xy,
        status=status,
        quarantine_reason=quarantine_reason,
    )


def read_rachel_group(
    group_path: Path,
    *,
    generator: str,
    config: RachelPreprocessConfig = RachelPreprocessConfig(),
) -> RachelGroup:
    """Decode and validate one Rachel folder without consulting Shredding data."""

    group = Path(group_path)
    if not group.is_dir():
        raise RachelPreprocessError("Rachel group directory does not exist")
    if not isinstance(generator, str) or not generator.strip():
        raise ValueError("generator must be a non-empty string")
    rows = _read_csv_rows(group)
    image_name, by_id, neighbor_edges = _validated_csv_graph(rows)

    expected_by_generator = {
        "gen2voronoi_1": 2,
        "gen2voronoi_2": 2,
        "gen3voronoi": 3,
        "gen4voronoi": 4,
        "gen4voronoi_1_3": 4,
        "gen5voronoi_1_1_3": 5,
    }
    if generator not in expected_by_generator:
        raise RachelPreprocessError("unsupported Rachel generator: " + generator)
    expected_fragments = expected_by_generator[generator]
    if len(rows) != expected_fragments:
        raise RachelPreprocessError(
            "generator {} requires exactly {} fragments".format(
                generator, expected_fragments
            )
        )

    jpeg_paths = tuple(
        sorted(
            (
                path
                for path in group.iterdir()
                if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}
            ),
            key=lambda path: _natural_key(path.stem),
        )
    )
    jpeg_by_stem = {path.stem: path for path in jpeg_paths}
    if len(jpeg_by_stem) != len(jpeg_paths):
        raise RachelPreprocessError("JPEG stems are duplicated")
    expected_mask_ids = {str(row["mask_id"]).strip() for row in rows}
    if set(jpeg_by_stem) != expected_mask_ids:
        raise RachelPreprocessError(
            "JPEG inventory does not exactly match CSV mask_id values"
        )

    fragments = tuple(
        _load_fragment(
            by_id[fragment_id],
            jpeg_path=jpeg_by_stem[str(by_id[fragment_id]["mask_id"]).strip()],
            config=config,
        )
        for fragment_id in sorted(by_id, key=_natural_key)
    )
    fragment_by_id = {fragment.fragment_id: fragment for fragment in fragments}
    positive_keys = {frozenset(edge) for edge in neighbor_edges}
    pairs = []
    ids = sorted(fragment_by_id, key=_natural_key)
    for first_index, first_id in enumerate(ids):
        for second_id in ids[first_index + 1 :]:
            pairs.append(
                _make_pair(
                    fragment_by_id[first_id],
                    fragment_by_id[second_id],
                    label=frozenset((first_id, second_id)) in positive_keys,
                    config=config,
                )
            )
    return RachelGroup(
        generator=generator,
        group_id=group.name,
        image_name=image_name,
        fragments=fragments,
        neighbor_edges=neighbor_edges,
        pairs=tuple(pairs),
    )
