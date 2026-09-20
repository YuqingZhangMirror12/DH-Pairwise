"""Build the single-source gen4 Pairwise 30k mask manifest.

The input is an extracted directory with the shape::

    gen4voronoi/no_erode/<parent>/<fragment>.png

Only scalar binary masks are decoded.  Parent groups are assigned to an exact
80/10/10 split before pairs are made.  A positive is an unordered fragment
pair that shares at least one exact four-neighbour grid edge in the aligned
parent canvas.  Positive pairs shorter than the configured clean-training seam
threshold retain their positive label but are excluded before main selection.
Negatives come only from different parents in the same split and are matched
on foreground area and bounding-box aspect ratio.

This module writes metadata only.  Filled-mask and 10-pixel internal-contour
views intentionally share these pair IDs and are materialized separately.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


SCHEMA_VERSION = "dunhuang-gen4-pairwise30k/0.1"
DEFAULT_SEED = "gen4-pairwise30k-v1"
SOURCE_ID = "shredding_pipeline_gen4_no_erode"
SPLITS = ("train", "val", "test")
_SCALAR_MODES = frozenset({"1", "L", "I", "I;16", "I;16B", "I;16L"})
_OTHER_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
)


class Gen4Pairwise30kError(ValueError):
    """The gen4 source or requested balanced population is invalid."""


class Gen4Pairwise30kShortfall(Gen4Pairwise30kError):
    """A split/label quota cannot be filled without weakening the contract."""


@dataclass(frozen=True)
class SplitQuota:
    positive: int
    negative: int

    def __post_init__(self) -> None:
        if type(self.positive) is not int or self.positive <= 0:  # noqa: E721
            raise ValueError("positive quota must be a positive integer")
        if type(self.negative) is not int or self.negative <= 0:  # noqa: E721
            raise ValueError("negative quota must be a positive integer")
        if self.positive != self.negative:
            raise ValueError("each split must be exactly 1:1 positive/negative")

    def to_dict(self) -> Dict[str, int]:
        return {"positive": self.positive, "negative": self.negative}


DEFAULT_SPLIT_QUOTAS = MappingProxyType(
    {
        "train": SplitQuota(positive=12_000, negative=12_000),
        "val": SplitQuota(positive=1_500, negative=1_500),
        "test": SplitQuota(positive=1_500, negative=1_500),
    }
)


def _normalize_quotas(
    value: Mapping[str, SplitQuota],
) -> Mapping[str, SplitQuota]:
    if not isinstance(value, Mapping) or set(value) != set(SPLITS):
        raise ValueError("split_quotas must define train, val, and test exactly")
    normalized: Dict[str, SplitQuota] = {}
    for split in SPLITS:
        quota = value[split]
        if not isinstance(quota, SplitQuota):
            raise TypeError("split_quotas values must be SplitQuota")
        normalized[split] = quota
    totals = [
        normalized[split].positive + normalized[split].negative for split in SPLITS
    ]
    if totals[0] != 8 * totals[1] or totals[1] != totals[2]:
        raise ValueError("split quotas must have an exact 8:1:1 ratio")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class Gen4Pairwise30kConfig:
    mask_root: Path
    seed: str = DEFAULT_SEED
    split_quotas: Mapping[str, SplitQuota] = field(
        default_factory=lambda: DEFAULT_SPLIT_QUOTAS
    )
    expected_parent_count: Optional[int] = 8_000
    fragments_per_parent: int = 4
    parent_canvas_size: int = 800
    model_canvas_size: int = 800
    contour_width: int = 10
    minimum_positive_seam_edge_count: int = 64
    max_foreground_area_ratio: float = 2.0
    max_bbox_aspect_ratio_ratio: float = 2.0
    negative_neighbor_window: int = 64
    max_negative_degree_per_fragment: int = 2
    max_negative_degree_per_parent: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask_root", Path(self.mask_root))
        if not isinstance(self.seed, str) or not self.seed.strip():
            raise ValueError("seed must be a non-empty string")
        object.__setattr__(self, "split_quotas", _normalize_quotas(self.split_quotas))
        if self.expected_parent_count is not None and (
            type(self.expected_parent_count) is not int
            or self.expected_parent_count <= 0
        ):
            raise ValueError("expected_parent_count must be positive or None")
        for name in (
            "fragments_per_parent",
            "parent_canvas_size",
            "model_canvas_size",
            "contour_width",
            "minimum_positive_seam_edge_count",
            "negative_neighbor_window",
            "max_negative_degree_per_fragment",
            "max_negative_degree_per_parent",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        for name in (
            "max_foreground_area_ratio",
            "max_bbox_aspect_ratio_ratio",
        ):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise ValueError(name + " must be finite")
            if float(value) < 1.0:
                raise ValueError(name + " must be at least 1.0")


@dataclass(frozen=True)
class _Fragment:
    token: str
    relative_path: str
    parent_id: str
    split: str
    foreground_area: int
    bbox_aspect_ratio: float


@dataclass(frozen=True)
class Gen4PairRow:
    pair_id: str
    split: str
    label: bool
    direction_b_wrt_a: Optional[str]
    fragment_a_relative_path: str
    fragment_b_relative_path: str
    parent_a_id: str
    parent_b_id: str
    label_origin: str
    negative_origin: Optional[str]
    seam_edge_count: Optional[int] = None
    foreground_area_ratio: Optional[float] = None
    bbox_aspect_ratio_ratio: Optional[float] = None

    @property
    def canonical_pair_key(self) -> Tuple[str, str]:
        return tuple(
            sorted((self.fragment_a_relative_path, self.fragment_b_relative_path))
        )  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "source_id": SOURCE_ID,
            "split": self.split,
            "label": self.label,
            "direction_b_wrt_a": self.direction_b_wrt_a,
            "fragment_a_relative_path": self.fragment_a_relative_path,
            "fragment_b_relative_path": self.fragment_b_relative_path,
            "parent_a_id": self.parent_a_id,
            "parent_b_id": self.parent_b_id,
            "label_origin": self.label_origin,
            "negative_origin": self.negative_origin,
            "seam_edge_count": self.seam_edge_count,
            "scale_match": None
            if self.label
            else {
                "foreground_area_ratio": self.foreground_area_ratio,
                "bbox_aspect_ratio_ratio": self.bbox_aspect_ratio_ratio,
            },
        }


@dataclass(frozen=True)
class Gen4Pairwise30kManifest:
    config: Gen4Pairwise30kConfig
    parent_assignments: Mapping[str, str]
    positive_candidate_counts: Mapping[str, int]
    ignored_same_parent_nonadjacent_counts: Mapping[str, int]
    rows: Tuple[Gen4PairRow, ...]
    negative_degree_statistics: Mapping[str, Mapping[str, object]]
    excluded_short_seam_positive_rows: Tuple[Gen4PairRow, ...] = ()

    def rows_for_split(self, split: str) -> Tuple[Gen4PairRow, ...]:
        if split not in SPLITS:
            raise ValueError("split must be train, val, or test")
        return tuple(row for row in self.rows if row.split == split)

    def protocol_dict(self) -> Dict[str, object]:
        split_counts = {}
        for split in SPLITS:
            split_rows = self.rows_for_split(split)
            split_counts[split] = {
                "parent_groups": sum(
                    value == split for value in self.parent_assignments.values()
                ),
                "positive": sum(row.label for row in split_rows),
                "negative": sum(not row.label for row in split_rows),
                "total": len(split_rows),
                "positive_candidates": self.positive_candidate_counts[split],
                "excluded_short_seam_positive": sum(
                    row.split == split
                    for row in self.excluded_short_seam_positive_rows
                ),
                "ignored_same_parent_nonadjacent": (
                    self.ignored_same_parent_nonadjacent_counts[split]
                ),
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "source_id": SOURCE_ID,
            "seed": self.config.seed,
            "source": {
                "mask_root": self.config.mask_root.as_posix(),
                "generator": "gen4voronoi",
                "erosion_profile": "no_erode",
                "rgb_used": False,
            },
            "input_contract": {
                "storage": "scalar PNG",
                "canonical_pixel_values": [0, 255],
                "parent_canvas_shape": [
                    self.config.parent_canvas_size,
                    self.config.parent_canvas_size,
                ],
                "fragments_per_parent": self.config.fragments_per_parent,
                "known_orientation": True,
                "rotation_search": False,
                "positive_definition": "exact aligned-mask 4-neighbor seam",
                "main_training_positive_minimum_seam_edge_count": (
                    self.config.minimum_positive_seam_edge_count
                ),
                "short_seam_policy": (
                    "retain positive semantics; exclude before main selection "
                    "and representation materialization"
                ),
                "negative_definition": (
                    "different parent, same split, foreground-area and "
                    "bbox-aspect-ratio matched"
                ),
            },
            "representations": {
                "canonical_source": "filled binary mask",
                "model_canvas_shape": [
                    self.config.model_canvas_size,
                    self.config.model_canvas_size,
                ],
                "spatial_normalization": (
                    "shared-parent-scale tight crop then center pad; no bbox origin"
                ),
                "contour10": {
                    "materialized_by_this_module": False,
                    "width_pixels_after_shared_parent_scale": (
                        self.config.contour_width
                    ),
                    "definition": (
                        "M & ~binary_erosion(M, 3x3, iterations={}, "
                        "border_value=0)".format(self.config.contour_width)
                    ),
                    "shares_pair_ids_splits_labels_and_order": True,
                },
            },
            "split_policy": {
                "unit": "parent folder",
                "method": "seeded stable rank then exact contiguous 80/10/10",
                "parent_leakage": False,
                "assignments": dict(sorted(self.parent_assignments.items())),
            },
            "split_quotas": {
                split: self.config.split_quotas[split].to_dict() for split in SPLITS
            },
            "split_counts": split_counts,
            "scale_matching": {
                "max_foreground_area_ratio": (self.config.max_foreground_area_ratio),
                "max_bbox_aspect_ratio_ratio": (
                    self.config.max_bbox_aspect_ratio_ratio
                ),
                "neighbor_window": self.config.negative_neighbor_window,
            },
            "negative_reuse_limits": {
                "max_degree_per_fragment": (
                    self.config.max_negative_degree_per_fragment
                ),
                "max_degree_per_parent": self.config.max_negative_degree_per_parent,
                "observed": {
                    split: dict(self.negative_degree_statistics[split])
                    for split in SPLITS
                },
            },
            "artifacts": {
                "protocol": "protocol.json",
                "splits": {split: "splits/{}.jsonl".format(split) for split in SPLITS},
            },
            "large_file_hashes_computed": False,
        }


def _digest(seed: str, namespace: str, *values: str) -> str:
    payload = "\0".join((seed, namespace) + values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parent_directories(root: Path) -> Tuple[Path, ...]:
    if not root.is_dir():
        raise Gen4Pairwise30kError(
            "mask_root must be the extracted gen4voronoi/no_erode directory"
        )
    values = tuple(
        sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
    )
    if not values:
        raise Gen4Pairwise30kError("mask_root contains no parent folders")
    if len({path.name for path in values}) != len(values):
        raise Gen4Pairwise30kError("parent folder IDs are not unique")
    return values


def _assign_parent_splits(parents: Sequence[Path], seed: str) -> Mapping[str, str]:
    if len(parents) % 10 != 0:
        raise Gen4Pairwise30kError(
            "exact 80/10/10 parent split requires a parent count divisible by 10"
        )
    ranked = sorted(
        (path.name for path in parents),
        key=lambda parent_id: (
            _digest(seed, "parent-split-rank", parent_id),
            parent_id,
        ),
    )
    train_end = len(ranked) * 8 // 10
    val_end = len(ranked) * 9 // 10
    result = {}
    for index, parent_id in enumerate(ranked):
        if index < train_end:
            split = "train"
        elif index < val_end:
            split = "val"
        else:
            split = "test"
        result[parent_id] = split
    return MappingProxyType(result)


def _png_paths(parent: Path, expected_count: int) -> Tuple[Path, ...]:
    other_images = sorted(
        path.name
        for path in parent.iterdir()
        if path.is_file() and path.suffix.casefold() in _OTHER_IMAGE_SUFFIXES
    )
    if other_images:
        raise Gen4Pairwise30kError(
            "RGB/non-PNG image is not a mask input: {}/{}".format(
                parent.name, other_images[0]
            )
        )
    values = tuple(
        sorted(
            (
                path
                for path in parent.iterdir()
                if path.is_file() and path.suffix.casefold() == ".png"
            ),
            key=lambda path: path.name,
        )
    )
    if len(values) != expected_count:
        raise Gen4Pairwise30kError(
            "parent {!r} must contain exactly {} PNG masks, observed {}".format(
                parent.name, expected_count, len(values)
            )
        )
    return values


def _load_binary_scalar_mask(path: Path, canvas_size: int) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.mode not in _SCALAR_MODES:
                raise Gen4Pairwise30kError(
                    "RGB/palette image rejected; expected scalar binary mask: "
                    + path.as_posix()
                )
            value = np.asarray(image)
    except (OSError, UnidentifiedImageError) as error:
        raise Gen4Pairwise30kError("cannot decode mask: " + path.as_posix()) from error
    if value.ndim != 2 or value.shape != (canvas_size, canvas_size):
        raise Gen4Pairwise30kError(
            "mask must be a scalar {}x{} parent canvas: {}".format(
                canvas_size, canvas_size, path.as_posix()
            )
        )
    if value.dtype == np.bool_:
        mask = value
    elif np.issubdtype(value.dtype, np.integer):
        if not np.logical_or(value == 0, value == 255).all():
            raise Gen4Pairwise30kError(
                "integer mask must contain only canonical 0/255 pixels: "
                + path.as_posix()
            )
        mask = value == 255
    else:
        raise Gen4Pairwise30kError(
            "mask dtype must be bool or integer 0/255: " + path.as_posix()
        )
    if not mask.any():
        raise Gen4Pairwise30kError("mask contains no foreground: " + path.as_posix())
    return np.ascontiguousarray(mask, dtype=np.bool_)


def _fragment_metadata(
    *, root: Path, path: Path, parent_id: str, split: str, mask: np.ndarray
) -> _Fragment:
    rows, columns = np.nonzero(mask)
    height = int(rows.max()) - int(rows.min()) + 1
    width = int(columns.max()) - int(columns.min()) + 1
    relative = path.relative_to(root).as_posix()
    return _Fragment(
        token="{}/{}".format(SOURCE_ID, relative),
        relative_path=relative,
        parent_id=parent_id,
        split=split,
        foreground_area=int(mask.sum()),
        bbox_aspect_ratio=width / float(height),
    )


def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    rows, columns = np.nonzero(mask)
    return float(rows.mean()), float(columns.mean())


def _direction(center_a: Tuple[float, float], center_b: Tuple[float, float]) -> str:
    row_delta = center_b[0] - center_a[0]
    column_delta = center_b[1] - center_a[1]
    if abs(column_delta) >= abs(row_delta):
        return "right" if column_delta >= 0.0 else "left"
    return "below" if row_delta >= 0.0 else "above"


def _seam_counts(label_canvas: np.ndarray) -> Mapping[Tuple[int, int], int]:
    chunks = []
    for first, second in (
        (label_canvas[:, :-1], label_canvas[:, 1:]),
        (label_canvas[:-1, :], label_canvas[1:, :]),
    ):
        valid = (first >= 0) & (second >= 0) & (first != second)
        if valid.any():
            lower = np.minimum(first[valid], second[valid])
            upper = np.maximum(first[valid], second[valid])
            chunks.append(np.stack((lower, upper), axis=1))
    if not chunks:
        return MappingProxyType({})
    pairs, counts = np.unique(
        np.concatenate(chunks, axis=0), axis=0, return_counts=True
    )
    return MappingProxyType(
        {(int(pair[0]), int(pair[1])): int(count) for pair, count in zip(pairs, counts)}
    )


def _pair_id(seed: str, label: bool, first: str, second: str) -> str:
    endpoints = tuple(sorted((first, second)))
    digest = _digest(
        seed,
        "positive-pair" if label else "negative-pair",
        endpoints[0],
        endpoints[1],
    )[:24]
    return "gen4-{}-{}".format("pos" if label else "neg", digest)


def _read_parent(
    root: Path,
    parent: Path,
    *,
    split: str,
    config: Gen4Pairwise30kConfig,
) -> Tuple[
    Tuple[_Fragment, ...],
    Tuple[Gen4PairRow, ...],
    Tuple[Gen4PairRow, ...],
    int,
]:
    paths = _png_paths(parent, config.fragments_per_parent)
    masks = tuple(
        _load_binary_scalar_mask(path, config.parent_canvas_size) for path in paths
    )
    label_canvas = np.full(masks[0].shape, -1, dtype=np.int8)
    for index, mask in enumerate(masks):
        if (label_canvas[mask] >= 0).any():
            raise Gen4Pairwise30kError(
                "fragment masks overlap in parent {!r}".format(parent.name)
            )
        label_canvas[mask] = index
    fragments = tuple(
        _fragment_metadata(
            root=root,
            path=path,
            parent_id=parent.name,
            split=split,
            mask=mask,
        )
        for path, mask in zip(paths, masks)
    )
    centers = tuple(_centroid(mask) for mask in masks)
    seams = _seam_counts(label_canvas)
    positives = []
    excluded_short_positives = []
    for (first_index, second_index), seam_edge_count in seams.items():
        first = fragments[first_index]
        second = fragments[second_index]
        row = Gen4PairRow(
            pair_id=_pair_id(
                config.seed, True, first.relative_path, second.relative_path
            ),
            split=split,
            label=True,
            direction_b_wrt_a=_direction(
                centers[first_index], centers[second_index]
            ),
            fragment_a_relative_path=first.relative_path,
            fragment_b_relative_path=second.relative_path,
            parent_a_id=parent.name,
            parent_b_id=parent.name,
            label_origin="exact_aligned_mask_4_neighbor_seam",
            negative_origin=None,
            seam_edge_count=seam_edge_count,
        )
        if seam_edge_count >= config.minimum_positive_seam_edge_count:
            positives.append(row)
        else:
            excluded_short_positives.append(row)
    total_unordered = len(fragments) * (len(fragments) - 1) // 2
    ignored_nonadjacent = total_unordered - len(seams)
    return (
        fragments,
        tuple(positives),
        tuple(excluded_short_positives),
        ignored_nonadjacent,
    )


def _ratio(first: float, second: float) -> float:
    return max(first, second) / min(first, second)


def _negative_degree_statistics(
    fragments: Sequence[_Fragment], rows: Sequence[Gen4PairRow]
) -> Mapping[str, object]:
    fragment_degree = Counter(fragment.relative_path for fragment in fragments)
    fragment_degree.clear()
    parent_degree: Counter = Counter()
    for row in rows:
        fragment_degree[row.fragment_a_relative_path] += 1
        fragment_degree[row.fragment_b_relative_path] += 1
        parent_degree[row.parent_a_id] += 1
        parent_degree[row.parent_b_id] += 1
    fragment_values = [
        fragment_degree[fragment.relative_path] for fragment in fragments
    ]
    parents = sorted({fragment.parent_id for fragment in fragments})
    parent_values = [parent_degree[parent] for parent in parents]

    def summary(values: Sequence[int]) -> Dict[str, object]:
        return {
            "count": len(values),
            "used_count": sum(value > 0 for value in values),
            "max": max(values, default=0),
            "mean": sum(values) / float(len(values)) if values else 0.0,
            "degree_histogram": {
                str(key): count for key, count in sorted(Counter(values).items())
            },
        }

    return MappingProxyType(
        {"fragment": summary(fragment_values), "parent": summary(parent_values)}
    )


def _scale_distance(
    first: _Fragment, second: _Fragment, config: Gen4Pairwise30kConfig
) -> Tuple[float, float, float]:
    area_ratio = _ratio(first.foreground_area, second.foreground_area)
    aspect_ratio = _ratio(first.bbox_aspect_ratio, second.bbox_aspect_ratio)
    area_denominator = max(math.log(config.max_foreground_area_ratio), 1e-12)
    aspect_denominator = max(math.log(config.max_bbox_aspect_ratio_ratio), 1e-12)
    area_distance = math.log(area_ratio) / area_denominator
    aspect_distance = math.log(aspect_ratio) / aspect_denominator
    return (
        area_ratio,
        aspect_ratio,
        area_distance * area_distance + aspect_distance * aspect_distance,
    )


def _select_negative_rows(
    fragments: Sequence[_Fragment],
    *,
    split: str,
    target: int,
    config: Gen4Pairwise30kConfig,
) -> Tuple[Gen4PairRow, ...]:
    values = tuple(sorted(fragments, key=lambda value: value.token))
    if any(fragment.split != split for fragment in values):
        raise Gen4Pairwise30kError(
            "negative candidate fragments must all belong to split " + split
        )
    if len({fragment.parent_id for fragment in values}) < 2:
        raise Gen4Pairwise30kShortfall(
            "negative shortfall for {}: different-parent negatives require at "
            "least two parent groups".format(split)
        )
    orderings = (
        tuple(
            sorted(
                values,
                key=lambda value: (
                    value.foreground_area,
                    value.bbox_aspect_ratio,
                    value.token,
                ),
            )
        ),
        tuple(
            sorted(
                values,
                key=lambda value: (
                    value.bbox_aspect_ratio,
                    value.foreground_area,
                    value.token,
                ),
            )
        ),
    )
    positions = tuple(
        {fragment.token: index for index, fragment in enumerate(ordering)}
        for ordering in orderings
    )
    fragment_degree: Counter = Counter()
    parent_degree: Counter = Counter()
    selected_keys = set()
    selected: List[Gen4PairRow] = []
    round_index = 0
    while len(selected) < target:
        made_progress = False
        visit_order = sorted(
            values,
            key=lambda value: (
                fragment_degree[value.token],
                parent_degree[value.parent_id],
                _digest(
                    config.seed,
                    "negative-visit-{}-{}".format(split, round_index),
                    value.token,
                ),
                value.token,
            ),
        )
        for first in visit_order:
            if len(selected) >= target:
                break
            if (
                fragment_degree[first.token] >= config.max_negative_degree_per_fragment
                or parent_degree[first.parent_id]
                >= config.max_negative_degree_per_parent
            ):
                continue
            nearby: Dict[str, _Fragment] = {}
            for ordering, position_by_token in zip(orderings, positions):
                center = position_by_token[first.token]
                start = max(0, center - config.negative_neighbor_window)
                stop = min(len(ordering), center + config.negative_neighbor_window + 1)
                for second in ordering[start:stop]:
                    if second.token != first.token:
                        nearby[second.token] = second
            eligible = []
            for second in nearby.values():
                if second.parent_id == first.parent_id:
                    continue
                key = tuple(sorted((first.token, second.token)))
                if key in selected_keys:
                    continue
                if (
                    fragment_degree[second.token]
                    >= config.max_negative_degree_per_fragment
                    or parent_degree[second.parent_id]
                    >= config.max_negative_degree_per_parent
                ):
                    continue
                area_ratio, aspect_ratio, distance = _scale_distance(
                    first, second, config
                )
                if area_ratio > config.max_foreground_area_ratio or (
                    aspect_ratio > config.max_bbox_aspect_ratio_ratio
                ):
                    continue
                eligible.append(
                    (
                        max(
                            fragment_degree[first.token],
                            fragment_degree[second.token],
                        ),
                        fragment_degree[first.token] + fragment_degree[second.token],
                        max(
                            parent_degree[first.parent_id],
                            parent_degree[second.parent_id],
                        ),
                        parent_degree[first.parent_id]
                        + parent_degree[second.parent_id],
                        distance,
                        _digest(
                            config.seed,
                            "negative-choice-{}".format(split),
                            key[0],
                            key[1],
                        ),
                        second.token,
                        second,
                        area_ratio,
                        aspect_ratio,
                    )
                )
            if not eligible:
                continue
            choice = min(eligible)
            second = choice[7]
            area_ratio = choice[8]
            aspect_ratio = choice[9]
            endpoint_a, endpoint_b = sorted(
                (first, second), key=lambda value: value.relative_path
            )
            key = tuple(sorted((first.token, second.token)))
            selected_keys.add(key)
            fragment_degree[first.token] += 1
            fragment_degree[second.token] += 1
            parent_degree[first.parent_id] += 1
            parent_degree[second.parent_id] += 1
            selected.append(
                Gen4PairRow(
                    pair_id=_pair_id(
                        config.seed,
                        False,
                        endpoint_a.relative_path,
                        endpoint_b.relative_path,
                    ),
                    split=split,
                    label=False,
                    direction_b_wrt_a=None,
                    fragment_a_relative_path=endpoint_a.relative_path,
                    fragment_b_relative_path=endpoint_b.relative_path,
                    parent_a_id=endpoint_a.parent_id,
                    parent_b_id=endpoint_b.parent_id,
                    label_origin="constructed_non_match",
                    negative_origin=(
                        "cross_parent_same_split_scale_matched_low_degree_greedy"
                    ),
                    foreground_area_ratio=area_ratio,
                    bbox_aspect_ratio_ratio=aspect_ratio,
                )
            )
            made_progress = True
        if not made_progress:
            break
        round_index += 1
    if len(selected) < target:
        raise Gen4Pairwise30kShortfall(
            "negative shortfall for {}: requested {}, selected {}; scale "
            "limits area<={} aspect<={}, reuse limits fragment<={} parent<={}, "
            "neighbor_window={}".format(
                split,
                target,
                len(selected),
                config.max_foreground_area_ratio,
                config.max_bbox_aspect_ratio_ratio,
                config.max_negative_degree_per_fragment,
                config.max_negative_degree_per_parent,
                config.negative_neighbor_window,
            )
        )
    return tuple(selected)


def _select_positive_rows(
    rows: Iterable[Gen4PairRow], *, split: str, target: int, seed: str
) -> Tuple[Gen4PairRow, ...]:
    ranked = sorted(
        rows,
        key=lambda row: (
            _digest(seed, "positive-select-{}".format(split), row.pair_id),
            row.pair_id,
        ),
    )
    if len(ranked) < target:
        raise Gen4Pairwise30kShortfall(
            "positive shortfall for {}: requested {}, available {} exact-seam "
            "pairs".format(split, target, len(ranked))
        )
    return tuple(ranked[:target])


def build_gen4_pairwise30k_manifest(
    config: Gen4Pairwise30kConfig,
) -> Gen4Pairwise30kManifest:
    """Read gen4 masks once per parent and build the exact balanced manifest."""

    if not isinstance(config, Gen4Pairwise30kConfig):
        raise TypeError("config must be Gen4Pairwise30kConfig")
    parents = _parent_directories(config.mask_root)
    if config.expected_parent_count is not None and (
        len(parents) != config.expected_parent_count
    ):
        raise Gen4Pairwise30kError(
            "expected {} parent folders, observed {}".format(
                config.expected_parent_count, len(parents)
            )
        )
    assignments = _assign_parent_splits(parents, config.seed)
    fragments_by_split: Dict[str, List[_Fragment]] = {split: [] for split in SPLITS}
    positives_by_split: Dict[str, List[Gen4PairRow]] = {split: [] for split in SPLITS}
    excluded_short_positives: List[Gen4PairRow] = []
    ignored_by_split = Counter()
    for parent in sorted(parents, key=lambda path: path.name):
        split = assignments[parent.name]
        fragments, positives, excluded, ignored = _read_parent(
            config.mask_root,
            parent,
            split=split,
            config=config,
        )
        fragments_by_split[split].extend(fragments)
        positives_by_split[split].extend(positives)
        excluded_short_positives.extend(excluded)
        ignored_by_split[split] += ignored

    selected_rows = []
    degree_stats = {}
    for split in SPLITS:
        quota = config.split_quotas[split]
        positives = _select_positive_rows(
            positives_by_split[split],
            split=split,
            target=quota.positive,
            seed=config.seed,
        )
        negatives = _select_negative_rows(
            fragments_by_split[split],
            split=split,
            target=quota.negative,
            config=config,
        )
        degree_stats[split] = _negative_degree_statistics(
            fragments_by_split[split], negatives
        )
        split_rows = list(positives + negatives)
        split_rows.sort(
            key=lambda row: (
                _digest(config.seed, "row-order-{}".format(split), row.pair_id),
                row.pair_id,
            )
        )
        selected_rows.extend(split_rows)

    pair_ids = [row.pair_id for row in selected_rows]
    pair_keys = [row.canonical_pair_key for row in selected_rows]
    if len(set(pair_ids)) != len(pair_ids) or len(set(pair_keys)) != len(pair_keys):
        raise AssertionError("internal error: selected pair identity is not unique")
    return Gen4Pairwise30kManifest(
        config=config,
        parent_assignments=assignments,
        positive_candidate_counts=MappingProxyType(
            {split: len(positives_by_split[split]) for split in SPLITS}
        ),
        ignored_same_parent_nonadjacent_counts=MappingProxyType(
            {split: ignored_by_split[split] for split in SPLITS}
        ),
        rows=tuple(selected_rows),
        negative_degree_statistics=MappingProxyType(degree_stats),
        excluded_short_seam_positive_rows=tuple(excluded_short_positives),
    )


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Gen4PairRow]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def write_gen4_pairwise30k_manifest(
    manifest: Gen4Pairwise30kManifest, output_root: Path
) -> Path:
    """Write ``protocol.json`` and exactly three split JSONL files."""

    if not isinstance(manifest, Gen4Pairwise30kManifest):
        raise TypeError("manifest must be Gen4Pairwise30kManifest")
    root = Path(output_root)
    split_root = root / "splits"
    destinations = (root / "protocol.json",) + tuple(
        split_root / (split + ".jsonl") for split in SPLITS
    )
    existing = [path.as_posix() for path in destinations if path.exists()]
    if existing:
        raise Gen4Pairwise30kError(
            "refusing to overwrite existing manifest artifacts: " + repr(existing)
        )
    split_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "protocol.json", manifest.protocol_dict())
    for split in SPLITS:
        _atomic_jsonl(split_root / (split + ".jsonl"), manifest.rows_for_split(split))
    return root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--expected-parent-count", type=int, default=8_000)
    parser.add_argument("--parent-canvas-size", type=int, default=800)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    config = Gen4Pairwise30kConfig(
        mask_root=arguments.mask_root,
        seed=arguments.seed,
        expected_parent_count=arguments.expected_parent_count,
        parent_canvas_size=arguments.parent_canvas_size,
    )
    manifest = build_gen4_pairwise30k_manifest(config)
    output = write_gen4_pairwise30k_manifest(manifest, arguments.output_root)
    print(output.as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_SPLIT_QUOTAS",
    "Gen4PairRow",
    "Gen4Pairwise30kConfig",
    "Gen4Pairwise30kError",
    "Gen4Pairwise30kManifest",
    "Gen4Pairwise30kShortfall",
    "SCHEMA_VERSION",
    "SOURCE_ID",
    "SPLITS",
    "SplitQuota",
    "build_gen4_pairwise30k_manifest",
    "main",
    "write_gen4_pairwise30k_manifest",
]
