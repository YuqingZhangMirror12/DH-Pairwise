"""Source-neutral pair candidates for all shredding_pipeline generators.

The source masks remain aligned on their 800x800 parent canvas only while
adjacency and cardinal direction labels are derived.  A positive is any pair
with at least one shared four-neighbour grid edge.  Positives whose total seam
is shorter than 64 pixels retain that label but are explicitly ineligible for
the clean main-training selection.  A same-folder pair is a hard negative only
when its seam count is exactly zero.

Cross-folder negative candidates are built separately as a deterministic,
bounded-degree graph within the same generator and split.  Both foreground
area and bbox-aspect ratios must be at most 2; this module deliberately does
not choose the later hard-negative/cross-negative sampling ratio.

Model-facing filled-mask and contour tensors are prepared separately by
``mask_representations``; pair metadata never contains a fragment bbox origin.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import combinations
import math
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from .pairwise_30k_protocol import PairCandidate, stable_split_for_unit


SOURCE_ID_PREFIX = "shredding_pipeline_no_erode"
DEFAULT_MINIMUM_POSITIVE_SEAM = 64
DEFAULT_PARENT_CANVAS_SIZE = 800
_SCALAR_MODES = frozenset({"1", "L", "I", "I;16", "I;16B", "I;16L"})


class ShreddingPipelinePairError(ValueError):
    """One source parent cannot satisfy the unified mask/pair contract."""


@dataclass(frozen=True)
class GeneratorSpec:
    """One no-erode generator stratum under the common voronoi-mask root."""

    generator_id: str
    fragments_per_parent: int

    def __post_init__(self) -> None:
        if not isinstance(self.generator_id, str) or not self.generator_id:
            raise ValueError("generator_id must be a non-empty string")
        if type(self.fragments_per_parent) is not int or not (
            2 <= self.fragments_per_parent <= 5
        ):
            raise ValueError("fragments_per_parent must be an integer from 2 to 5")

    @property
    def relative_root(self) -> Path:
        return Path(self.generator_id) / "no_erode"

    @property
    def source_id(self) -> str:
        return "{}/n{}".format(SOURCE_ID_PREFIX, self.fragments_per_parent)


DEFAULT_GENERATOR_SPECS = (
    GeneratorSpec("gen2voronoi_1", 2),
    GeneratorSpec("gen2voronoi_2", 2),
    GeneratorSpec("gen3voronoi", 3),
    GeneratorSpec("gen4voronoi", 4),
    GeneratorSpec("gen4voronoi_1_3", 4),
    GeneratorSpec("gen5voronoi_1_1_3", 5),
)


@dataclass(frozen=True)
class FragmentGeometry:
    """Origin-free fragment metadata used only for pair construction."""

    token: str
    relative_path: str
    parent_group_id: str
    split_unit_id: str
    foreground_area: int
    bbox_aspect_ratio: float


@dataclass(frozen=True)
class ParentPairInventory:
    """All semantic within-parent pairs, including excluded short positives."""

    generator: GeneratorSpec
    parent_group_id: str
    split: str
    fragments: Tuple[FragmentGeometry, ...]
    pair_candidates: Tuple[PairCandidate, ...]

    @property
    def selectable_positives(self) -> Tuple[PairCandidate, ...]:
        return tuple(
            candidate
            for candidate in self.pair_candidates
            if candidate.label and candidate.main_training_eligible
        )

    @property
    def excluded_short_seam_positives(self) -> Tuple[PairCandidate, ...]:
        return tuple(
            candidate
            for candidate in self.pair_candidates
            if candidate.label and not candidate.main_training_eligible
        )

    @property
    def same_parent_hard_negatives(self) -> Tuple[PairCandidate, ...]:
        return tuple(
            candidate for candidate in self.pair_candidates if not candidate.label
        )


def _digest(seed: str, namespace: str, *values: str) -> str:
    payload = "\0".join((seed, namespace) + values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_binary_mask(path: Path, canvas_size: int) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.mode not in _SCALAR_MODES:
                raise ShreddingPipelinePairError(
                    "RGB/palette input is not a scalar binary mask: "
                    + path.as_posix()
                )
            value = np.asarray(image)
    except (OSError, UnidentifiedImageError) as error:
        raise ShreddingPipelinePairError(
            "cannot decode mask: " + path.as_posix()
        ) from error
    if value.ndim != 2 or value.shape != (canvas_size, canvas_size):
        raise ShreddingPipelinePairError(
            "mask must be a scalar {0}x{0} parent canvas: {1}".format(
                canvas_size, path.as_posix()
            )
        )
    if value.dtype == np.bool_:
        mask = value
    elif np.issubdtype(value.dtype, np.integer):
        if not np.logical_or(value == 0, value == 255).all():
            raise ShreddingPipelinePairError(
                "integer mask must contain only 0/255: " + path.as_posix()
            )
        mask = value == 255
    else:
        raise ShreddingPipelinePairError(
            "mask dtype must be bool or integer 0/255: " + path.as_posix()
        )
    if not mask.any():
        raise ShreddingPipelinePairError(
            "mask contains no foreground: " + path.as_posix()
        )
    return np.ascontiguousarray(mask, dtype=np.bool_)


def _mask_paths(parent: Path, expected_count: int) -> Tuple[Path, ...]:
    paths = tuple(
        sorted(
            (
                path
                for path in parent.iterdir()
                if path.is_file() and path.suffix.casefold() == ".png"
            ),
            key=lambda path: path.name,
        )
    )
    if len(paths) != expected_count:
        raise ShreddingPipelinePairError(
            "parent {!r} must contain exactly {} PNG masks, observed {}".format(
                parent.as_posix(), expected_count, len(paths)
            )
        )
    return paths


def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    rows, columns = np.nonzero(mask)
    return float(rows.mean()), float(columns.mean())


def _cardinal_direction(
    center_a: Tuple[float, float], center_b: Tuple[float, float]
) -> str:
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
        {
            (int(pair[0]), int(pair[1])): int(count)
            for pair, count in zip(pairs, counts)
        }
    )


def _fragment_geometry(
    *,
    pipeline_root: Path,
    path: Path,
    parent_group_id: str,
    mask: np.ndarray,
) -> FragmentGeometry:
    rows, columns = np.nonzero(mask)
    height = int(rows.max()) - int(rows.min()) + 1
    width = int(columns.max()) - int(columns.min()) + 1
    relative_path = path.relative_to(pipeline_root).as_posix()
    token = "{}/{}".format(SOURCE_ID_PREFIX, relative_path)
    return FragmentGeometry(
        token=token,
        relative_path=relative_path,
        parent_group_id=parent_group_id,
        split_unit_id=parent_group_id,
        foreground_area=int(mask.sum()),
        bbox_aspect_ratio=width / float(height),
    )


def read_parent_pair_inventory(
    pipeline_root: Path,
    parent: Path,
    generator: GeneratorSpec,
    *,
    seed: str,
    parent_canvas_size: int = DEFAULT_PARENT_CANVAS_SIZE,
    minimum_positive_seam: int = DEFAULT_MINIMUM_POSITIVE_SEAM,
    split: Optional[str] = None,
) -> ParentPairInventory:
    """Derive labels in aligned coordinates without exposing bbox origins."""

    root = Path(pipeline_root)
    parent_path = Path(parent)
    if type(parent_canvas_size) is not int or parent_canvas_size <= 0:
        raise ValueError("parent_canvas_size must be a positive integer")
    if type(minimum_positive_seam) is not int or minimum_positive_seam <= 0:
        raise ValueError("minimum_positive_seam must be a positive integer")
    expected_parent_root = root / generator.relative_root
    try:
        relative_parent = parent_path.relative_to(expected_parent_root)
    except ValueError as error:
        raise ShreddingPipelinePairError(
            "parent is outside its generator no_erode root"
        ) from error
    if len(relative_parent.parts) != 1 or not parent_path.is_dir():
        raise ShreddingPipelinePairError("parent must be one direct child folder")
    parent_group_id = "{}/{}/{}".format(
        SOURCE_ID_PREFIX, generator.generator_id, relative_parent.as_posix()
    )
    resolved_split = split or stable_split_for_unit(parent_group_id, seed=seed)
    if resolved_split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")

    paths = _mask_paths(parent_path, generator.fragments_per_parent)
    masks = tuple(_load_binary_mask(path, parent_canvas_size) for path in paths)
    label_canvas = np.full(masks[0].shape, -1, dtype=np.int16)
    for index, mask in enumerate(masks):
        if (label_canvas[mask] >= 0).any():
            raise ShreddingPipelinePairError(
                "fragment masks overlap in parent " + repr(parent_group_id)
            )
        label_canvas[mask] = index
    seams = _seam_counts(label_canvas)
    centers = tuple(_centroid(mask) for mask in masks)
    fragments = tuple(
        _fragment_geometry(
            pipeline_root=root,
            path=path,
            parent_group_id=parent_group_id,
            mask=mask,
        )
        for path, mask in zip(paths, masks)
    )

    candidates = []
    for first_index, second_index in combinations(range(len(fragments)), 2):
        first = fragments[first_index]
        second = fragments[second_index]
        seam_edge_count = seams.get((first_index, second_index), 0)
        is_positive = seam_edge_count > 0
        is_eligible = not is_positive or seam_edge_count >= minimum_positive_seam
        pair_key = tuple(sorted((first.token, second.token)))
        pair_id = "shredding-{}-{}".format(
            "pos" if is_positive else "hard-neg",
            _digest(seed, "within-parent-pair", pair_key[0], pair_key[1])[:24],
        )
        metadata: Dict[str, object] = {
            "generator_id": generator.generator_id,
            "fragments_per_parent": generator.fragments_per_parent,
            "parent_canvas_shape": [parent_canvas_size, parent_canvas_size],
            "seam_edge_count": seam_edge_count,
        }
        candidates.append(
            PairCandidate(
                pair_id=pair_id,
                source_id=generator.source_id,
                fragment_a_parent_group_id=parent_group_id,
                fragment_b_parent_group_id=parent_group_id,
                fragment_a_split_unit_id=parent_group_id,
                fragment_b_split_unit_id=parent_group_id,
                fragment_a_token=first.token,
                fragment_b_token=second.token,
                label=is_positive,
                label_origin=(
                    "exact_aligned_mask_4_neighbor_seam"
                    if is_positive
                    else "exact_aligned_mask_zero_seam"
                ),
                direction_b_wrt_a=(
                    _cardinal_direction(centers[first_index], centers[second_index])
                    if is_positive
                    else None
                ),
                negative_origin=(
                    None if is_positive else "same_parent_nonadjacent_hard"
                ),
                metadata=metadata,
                main_training_eligible=is_eligible,
                selection_exclusion_reason=(
                    None
                    if is_eligible
                    else "positive_seam_shorter_than_{}_pixels".format(
                        minimum_positive_seam
                    )
                ),
            )
        )
    return ParentPairInventory(
        generator=generator,
        parent_group_id=parent_group_id,
        split=resolved_split,
        fragments=fragments,
        pair_candidates=tuple(candidates),
    )


def iter_usable_parent_directories(
    pipeline_root: Path,
    generators: Sequence[GeneratorSpec] = DEFAULT_GENERATOR_SPECS,
) -> Iterator[Tuple[GeneratorSpec, Path]]:
    """Yield complete 2/3/4/5-fragment parents; known empty folders are skipped."""

    root = Path(pipeline_root)
    if not root.is_dir():
        raise ShreddingPipelinePairError(
            "pipeline_root must be the extracted voronoi_masks directory"
        )
    for generator in generators:
        generator_root = root / generator.relative_root
        if not generator_root.is_dir():
            raise ShreddingPipelinePairError(
                "missing generator root: " + generator_root.as_posix()
            )
        usable_count = 0
        for parent in sorted(
            (path for path in generator_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            png_count = sum(
                path.is_file() and path.suffix.casefold() == ".png"
                for path in parent.iterdir()
            )
            if png_count == 0:
                continue
            if png_count != generator.fragments_per_parent:
                raise ShreddingPipelinePairError(
                    "partial parent {!r}: expected {} PNG masks, observed {}".format(
                        parent.as_posix(), generator.fragments_per_parent, png_count
                    )
                )
            usable_count += 1
            yield generator, parent
        if usable_count == 0:
            raise ShreddingPipelinePairError(
                "generator contains no usable parent folders: "
                + generator_root.as_posix()
            )


def iter_shredding_pipeline_pair_inventories(
    pipeline_root: Path,
    *,
    seed: str,
    generators: Sequence[GeneratorSpec] = DEFAULT_GENERATOR_SPECS,
    parent_canvas_size: int = DEFAULT_PARENT_CANVAS_SIZE,
    minimum_positive_seam: int = DEFAULT_MINIMUM_POSITIVE_SEAM,
) -> Iterator[ParentPairInventory]:
    """Lazily cover every usable parent across all six 2/3/4/5 generators."""

    for generator, parent in iter_usable_parent_directories(
        pipeline_root, generators
    ):
        yield read_parent_pair_inventory(
            pipeline_root,
            parent,
            generator,
            seed=seed,
            parent_canvas_size=parent_canvas_size,
            minimum_positive_seam=minimum_positive_seam,
        )


def iter_pair_candidates(
    inventories: Iterable[ParentPairInventory],
) -> Iterator[PairCandidate]:
    """Flatten semantic candidates; the protocol filters explicit exclusions."""

    for inventory in inventories:
        if not isinstance(inventory, ParentPairInventory):
            raise TypeError("inventories must contain ParentPairInventory values")
        yield from inventory.pair_candidates


def _ratio(first: float, second: float) -> float:
    return max(first, second) / min(first, second)


def _cross_negative_candidate(
    *,
    first: FragmentGeometry,
    second: FragmentGeometry,
    generator: GeneratorSpec,
    split: str,
    seed: str,
    foreground_area_ratio: float,
    bbox_aspect_ratio_ratio: float,
) -> PairCandidate:
    endpoint_a, endpoint_b = sorted((first, second), key=lambda value: value.token)
    pair_key = (endpoint_a.token, endpoint_b.token)
    return PairCandidate(
        pair_id="shredding-cross-neg-{}".format(
            _digest(seed, "cross-parent-negative", pair_key[0], pair_key[1])[:24]
        ),
        source_id=generator.source_id,
        fragment_a_parent_group_id=endpoint_a.parent_group_id,
        fragment_b_parent_group_id=endpoint_b.parent_group_id,
        fragment_a_split_unit_id=endpoint_a.split_unit_id,
        fragment_b_split_unit_id=endpoint_b.split_unit_id,
        fragment_a_token=endpoint_a.token,
        fragment_b_token=endpoint_b.token,
        label=False,
        label_origin="constructed_non_match",
        direction_b_wrt_a=None,
        negative_origin="cross_parent_same_generator_same_split_scale_matched",
        metadata={
            "generator_id": generator.generator_id,
            "fragments_per_parent": generator.fragments_per_parent,
            "assigned_split": split,
            "foreground_area_ratio": foreground_area_ratio,
            "bbox_aspect_ratio_ratio": bbox_aspect_ratio_ratio,
        },
    )


def build_cross_parent_scale_matched_negatives(
    inventories: Iterable[ParentPairInventory],
    *,
    seed: str,
    max_foreground_area_ratio: float = 2.0,
    max_bbox_aspect_ratio_ratio: float = 2.0,
    neighbor_window: int = 64,
    max_degree_per_fragment: int = 2,
    max_degree_per_parent: int = 8,
) -> Tuple[PairCandidate, ...]:
    """Build a bounded candidate graph without choosing a final sample ratio.

    Edges are restricted to different parents in the same generator and split.
    Two scale-sorted neighbourhoods avoid an all-pairs expansion on the full
    392k-fragment source.  Greedy degree caps keep any fragment or parent from
    dominating the later sampler; no requested 30k quota is encoded here.
    """

    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a non-empty string")
    for name, value in (
        ("max_foreground_area_ratio", max_foreground_area_ratio),
        ("max_bbox_aspect_ratio_ratio", max_bbox_aspect_ratio_ratio),
    ):
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise ValueError(name + " must be finite")
        if float(value) < 1.0:
            raise ValueError(name + " must be at least 1.0")
    for name, value in (
        ("neighbor_window", neighbor_window),
        ("max_degree_per_fragment", max_degree_per_fragment),
        ("max_degree_per_parent", max_degree_per_parent),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(name + " must be a positive integer")

    inventory_values = tuple(inventories)
    cells: Dict[Tuple[str, str], List[ParentPairInventory]] = {}
    parent_keys = set()
    fragment_tokens = set()
    for inventory in inventory_values:
        if not isinstance(inventory, ParentPairInventory):
            raise TypeError("inventories must contain ParentPairInventory values")
        parent_key = (inventory.generator.generator_id, inventory.parent_group_id)
        if parent_key in parent_keys:
            raise ShreddingPipelinePairError(
                "duplicate parent inventory: " + repr(parent_key)
            )
        parent_keys.add(parent_key)
        for fragment in inventory.fragments:
            if fragment.parent_group_id != inventory.parent_group_id:
                raise ShreddingPipelinePairError(
                    "fragment parent does not match its inventory"
                )
            if fragment.token in fragment_tokens:
                raise ShreddingPipelinePairError(
                    "duplicate fragment token: " + fragment.token
                )
            fragment_tokens.add(fragment.token)
        cells.setdefault(
            (inventory.generator.generator_id, inventory.split), []
        ).append(inventory)

    selected: List[PairCandidate] = []
    for (generator_id, split), cell_inventories in sorted(cells.items()):
        if len(cell_inventories) < 2:
            continue
        generator = cell_inventories[0].generator
        if any(value.generator != generator for value in cell_inventories):
            raise ShreddingPipelinePairError(
                "one generator ID maps to inconsistent fragment counts"
            )
        fragments = tuple(
            sorted(
                (
                    fragment
                    for inventory in cell_inventories
                    for fragment in inventory.fragments
                ),
                key=lambda value: value.token,
            )
        )
        orderings = (
            tuple(
                sorted(
                    fragments,
                    key=lambda value: (
                        value.foreground_area,
                        value.bbox_aspect_ratio,
                        value.token,
                    ),
                )
            ),
            tuple(
                sorted(
                    fragments,
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
        round_index = 0
        while True:
            made_progress = False
            visit_order = sorted(
                fragments,
                key=lambda value: (
                    fragment_degree[value.token],
                    parent_degree[value.parent_group_id],
                    _digest(
                        seed,
                        "cross-negative-visit-{}-{}-{}".format(
                            generator_id, split, round_index
                        ),
                        value.token,
                    ),
                    value.token,
                ),
            )
            for first in visit_order:
                if (
                    fragment_degree[first.token] >= max_degree_per_fragment
                    or parent_degree[first.parent_group_id]
                    >= max_degree_per_parent
                ):
                    continue
                nearby: Dict[str, FragmentGeometry] = {}
                for ordering, position_by_token in zip(orderings, positions):
                    center = position_by_token[first.token]
                    start = max(0, center - neighbor_window)
                    stop = min(len(ordering), center + neighbor_window + 1)
                    for second in ordering[start:stop]:
                        if second.token != first.token:
                            nearby[second.token] = second
                eligible = []
                for second in nearby.values():
                    if second.parent_group_id == first.parent_group_id:
                        continue
                    key = tuple(sorted((first.token, second.token)))
                    if key in selected_keys:
                        continue
                    if (
                        fragment_degree[second.token] >= max_degree_per_fragment
                        or parent_degree[second.parent_group_id]
                        >= max_degree_per_parent
                    ):
                        continue
                    area_ratio = _ratio(
                        first.foreground_area, second.foreground_area
                    )
                    aspect_ratio = _ratio(
                        first.bbox_aspect_ratio, second.bbox_aspect_ratio
                    )
                    if area_ratio > max_foreground_area_ratio or (
                        aspect_ratio > max_bbox_aspect_ratio_ratio
                    ):
                        continue
                    scale_distance = (
                        math.log(area_ratio) ** 2 + math.log(aspect_ratio) ** 2
                    )
                    eligible.append(
                        (
                            max(
                                fragment_degree[first.token],
                                fragment_degree[second.token],
                            ),
                            fragment_degree[first.token]
                            + fragment_degree[second.token],
                            max(
                                parent_degree[first.parent_group_id],
                                parent_degree[second.parent_group_id],
                            ),
                            parent_degree[first.parent_group_id]
                            + parent_degree[second.parent_group_id],
                            scale_distance,
                            _digest(
                                seed,
                                "cross-negative-choice-{}-{}".format(
                                    generator_id, split
                                ),
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
                key = tuple(sorted((first.token, second.token)))
                selected_keys.add(key)
                fragment_degree[first.token] += 1
                fragment_degree[second.token] += 1
                parent_degree[first.parent_group_id] += 1
                parent_degree[second.parent_group_id] += 1
                selected.append(
                    _cross_negative_candidate(
                        first=first,
                        second=second,
                        generator=generator,
                        split=split,
                        seed=seed,
                        foreground_area_ratio=area_ratio,
                        bbox_aspect_ratio_ratio=aspect_ratio,
                    )
                )
                made_progress = True
            if not made_progress:
                break
            round_index += 1
    selected.sort(key=lambda candidate: (candidate.source_id, candidate.pair_id))
    return tuple(selected)


def build_shredding_pipeline_candidate_pool(
    inventories: Iterable[ParentPairInventory],
    *,
    seed: str,
    max_foreground_area_ratio: float = 2.0,
    max_bbox_aspect_ratio_ratio: float = 2.0,
    neighbor_window: int = 64,
    max_degree_per_fragment: int = 2,
    max_degree_per_parent: int = 8,
) -> Tuple[PairCandidate, ...]:
    """Combine semantic within-folder pairs and bounded cross-folder negatives."""

    values = tuple(inventories)
    within = tuple(iter_pair_candidates(values))
    cross = build_cross_parent_scale_matched_negatives(
        values,
        seed=seed,
        max_foreground_area_ratio=max_foreground_area_ratio,
        max_bbox_aspect_ratio_ratio=max_bbox_aspect_ratio_ratio,
        neighbor_window=neighbor_window,
        max_degree_per_fragment=max_degree_per_fragment,
        max_degree_per_parent=max_degree_per_parent,
    )
    combined = within + cross
    pair_ids = [candidate.pair_id for candidate in combined]
    pair_keys = [candidate.canonical_pair_key for candidate in combined]
    if len(set(pair_ids)) != len(pair_ids) or len(set(pair_keys)) != len(pair_keys):
        raise AssertionError("internal error: candidate pool contains a duplicate")
    return tuple(sorted(combined, key=lambda candidate: candidate.pair_id))


__all__ = [
    "DEFAULT_GENERATOR_SPECS",
    "DEFAULT_MINIMUM_POSITIVE_SEAM",
    "DEFAULT_PARENT_CANVAS_SIZE",
    "FragmentGeometry",
    "GeneratorSpec",
    "ParentPairInventory",
    "SOURCE_ID_PREFIX",
    "ShreddingPipelinePairError",
    "build_cross_parent_scale_matched_negatives",
    "build_shredding_pipeline_candidate_pool",
    "iter_pair_candidates",
    "iter_shredding_pipeline_pair_inventories",
    "iter_usable_parent_directories",
    "read_parent_pair_inventory",
]
