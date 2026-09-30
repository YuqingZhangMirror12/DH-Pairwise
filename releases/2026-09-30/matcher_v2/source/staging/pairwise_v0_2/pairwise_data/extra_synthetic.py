"""Thin directory adapter for additional synthetic fragment data.

This module intentionally bypasses the frozen LOCAL-Q1 input machinery.  It is
for optional training expansion after the baseline run:

* ``iter_mask_pairs`` turns an extracted ``voronoi_masks`` tree directly into
  mask pairs and derives the legacy adjacency label on demand.
* ``iter_rgb_attachments`` exposes already-rendered RGB fragments as views of
  the corresponding mask group.  It does not emit pairs, so the 2,000 RGB
  groups in ``dunhuang_datas.zip`` cannot accidentally duplicate the existing
  5,000 mask groups in the geometry sampler.

Expected layouts are the ones used by both supplied generators::

    voronoi_masks/<generator>/no_erode/<group_id>/<fragment_id>.png
    datasets/<generator>/no_erode/<group_id>/<fragment_id>.jpg
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np


class ExtraSyntheticError(ValueError):
    """Raised when one requested synthetic group is malformed."""


def _numeric_path_key(path: Path) -> Tuple[int, object]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def _mask_paths(group_dir: Path) -> Tuple[Path, ...]:
    paths = tuple(sorted(group_dir.glob("*.png"), key=_numeric_path_key))
    if not 2 <= len(paths) <= 5:
        raise ExtraSyntheticError(
            f"mask group must contain 2--5 PNG files: {group_dir}"
        )
    if len({path.stem for path in paths}) != len(paths):
        raise ExtraSyntheticError(f"duplicate fragment id in {group_dir}")
    return paths


def load_binary_mask(path: Path) -> np.ndarray:
    """Decode one scalar/colour mask as a contiguous boolean array."""

    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None or value.ndim != 2 or value.size == 0:
        raise ExtraSyntheticError(f"cannot decode mask: {path}")
    output = np.ascontiguousarray(value > 127, dtype=np.bool_)
    output.setflags(write=False)
    return output


def legacy_pair_label(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    kernel_size: int = 3,
    iterations: int = 10,
    minimum_overlap_pixels: int = 30,
) -> Tuple[bool, int]:
    """Apply the label rule used by both supplied shredding generators."""

    first = np.asarray(mask_a, dtype=np.bool_)
    second = np.asarray(mask_b, dtype=np.bool_)
    if first.ndim != 2 or first.shape != second.shape or first.size == 0:
        raise ExtraSyntheticError("pair masks must be non-empty aligned 2D arrays")
    if kernel_size <= 0 or iterations < 0 or minimum_overlap_pixels <= 0:
        raise ValueError("invalid dilation label parameters")
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(first.astype(np.uint8), kernel, iterations=iterations)
    overlap = int(np.count_nonzero((dilated > 0) & second))
    return overlap >= minimum_overlap_pixels, overlap


@dataclass(frozen=True)
class SyntheticMaskPair:
    dataset_id: str
    generator: str
    group_id: str
    fragment_a_id: str
    fragment_b_id: str
    fragment_a_path: Path
    fragment_b_path: Path
    label: bool
    dilated_overlap_pixels: int

    @property
    def group_key(self) -> str:
        return f"{self.dataset_id}/{self.generator}/no_erode/{self.group_id}"

    def load_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        return load_binary_mask(self.fragment_a_path), load_binary_mask(
            self.fragment_b_path
        )


def iter_mask_group_dirs(
    mask_root: Path,
    *,
    generators: Optional[Sequence[str]] = None,
    max_groups: Optional[int] = None,
) -> Iterator[Tuple[str, str, Path]]:
    """Yield ``(generator, group_id, path)`` without pre-building an index."""

    root = Path(mask_root)
    if not root.is_dir():
        raise ExtraSyntheticError(f"mask root is not a directory: {root}")
    allowed = None if generators is None else frozenset(generators)
    emitted = 0
    for generator_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if allowed is not None and generator_dir.name not in allowed:
            continue
        profile_dir = generator_dir / "no_erode"
        if not profile_dir.is_dir():
            continue
        for group_dir in sorted(
            (path for path in profile_dir.iterdir() if path.is_dir()),
            key=_numeric_path_key,
        ):
            if max_groups is not None and emitted >= max_groups:
                return
            yield generator_dir.name, group_dir.name, group_dir
            emitted += 1


def iter_mask_pairs(
    mask_root: Path,
    *,
    dataset_id: str = "turfan_shredding_extra",
    generators: Optional[Sequence[str]] = None,
    max_groups: Optional[int] = None,
) -> Iterator[SyntheticMaskPair]:
    """Stream all unordered pairs from extracted masks with on-demand labels."""

    if not dataset_id.strip():
        raise ValueError("dataset_id is required")
    for generator, group_id, group_dir in iter_mask_group_dirs(
        mask_root, generators=generators, max_groups=max_groups
    ):
        yield from load_mask_group_pairs(
            group_dir,
            dataset_id=dataset_id,
            generator=generator,
            group_id=group_id,
        )


def load_mask_group_pairs(
    group_dir: Path,
    *,
    dataset_id: str,
    generator: str,
    group_id: str,
) -> Tuple[SyntheticMaskPair, ...]:
    """Materialize one selected group's pairs without scanning other groups."""

    if not dataset_id.strip() or not generator.strip() or not group_id.strip():
        raise ValueError("dataset_id, generator, and group_id are required")
    paths = _mask_paths(Path(group_dir))
    masks = tuple(load_binary_mask(path) for path in paths)
    if len({mask.shape for mask in masks}) != 1:
        raise ExtraSyntheticError(f"unaligned masks in {group_dir}")
    rows = []
    for left, right in combinations(range(len(paths)), 2):
        label, overlap = legacy_pair_label(masks[left], masks[right])
        rows.append(
            SyntheticMaskPair(
                dataset_id=dataset_id,
                generator=generator,
                group_id=group_id,
                fragment_a_id=paths[left].stem,
                fragment_b_id=paths[right].stem,
                fragment_a_path=paths[left],
                fragment_b_path=paths[right],
                label=label,
                dilated_overlap_pixels=overlap,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class RGBGroupAttachment:
    """RGB views joined to an existing mask group; never a new pair source."""

    generator: str
    group_id: str
    fragment_ids: Tuple[str, ...]
    rgb_paths: Tuple[Path, ...]
    mask_paths: Tuple[Path, ...]
    neighbor_ids: Tuple[Tuple[str, Tuple[str, ...]], ...]

    @property
    def group_key(self) -> str:
        return f"{self.generator}/no_erode/{self.group_id}"


def _read_neighbors(label_path: Path) -> Dict[str, Tuple[str, ...]]:
    with label_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    result: Dict[str, Tuple[str, ...]] = {}
    for row in rows:
        fragment_id = (row.get("id") or "").strip()
        if not fragment_id or fragment_id in result:
            raise ExtraSyntheticError(f"invalid fragment id in {label_path}")
        result[fragment_id] = tuple(
            item for item in (row.get("neighbors") or "").split(";") if item
        )
    return result


def iter_rgb_attachments(
    dataset_root: Path,
    mask_root: Path,
    *,
    generators: Optional[Sequence[str]] = None,
    max_groups: Optional[int] = None,
) -> Iterator[RGBGroupAttachment]:
    """Join rendered JPEGs to masks while emitting no duplicate pair samples."""

    rgb_root = Path(dataset_root)
    masks = Path(mask_root)
    if not rgb_root.is_dir() or not masks.is_dir():
        raise ExtraSyntheticError("dataset_root and mask_root must be directories")
    allowed = None if generators is None else frozenset(generators)
    emitted = 0
    for generator_dir in sorted(path for path in rgb_root.iterdir() if path.is_dir()):
        if allowed is not None and generator_dir.name not in allowed:
            continue
        profile_dir = generator_dir / "no_erode"
        if not profile_dir.is_dir():
            continue
        for group_dir in sorted(
            (path for path in profile_dir.iterdir() if path.is_dir()),
            key=_numeric_path_key,
        ):
            if max_groups is not None and emitted >= max_groups:
                return
            label_path = group_dir / "label.csv"
            if not label_path.is_file():
                continue
            neighbors = _read_neighbors(label_path)
            fragment_ids = tuple(
                sorted(
                    neighbors,
                    key=lambda item: (
                        not item.isdigit(),
                        int(item) if item.isdigit() else item,
                    ),
                )
            )
            rgb_paths = tuple(
                group_dir / f"{fragment_id}.jpg" for fragment_id in fragment_ids
            )
            mask_group = masks / generator_dir.name / "no_erode" / group_dir.name
            mask_paths = tuple(
                mask_group / f"{fragment_id}.png" for fragment_id in fragment_ids
            )
            if not all(path.is_file() for path in rgb_paths + mask_paths):
                raise ExtraSyntheticError(f"incomplete RGB/mask group: {group_dir}")
            yield RGBGroupAttachment(
                generator=generator_dir.name,
                group_id=group_dir.name,
                fragment_ids=fragment_ids,
                rgb_paths=rgb_paths,
                mask_paths=mask_paths,
                neighbor_ids=tuple((key, neighbors[key]) for key in fragment_ids),
            )
            emitted += 1


__all__ = [
    "ExtraSyntheticError",
    "RGBGroupAttachment",
    "SyntheticMaskPair",
    "iter_mask_group_dirs",
    "iter_mask_pairs",
    "iter_rgb_attachments",
    "legacy_pair_label",
    "load_mask_group_pairs",
    "load_binary_mask",
]
