"""Materialize matched filled-mask and contour10 views for a gen4 manifest.

The pair manifest is the authority for selection and ordering.  This module
does not sample, relabel, reorder, or otherwise rewrite pair rows.  It decodes
only the four scalar binary masks in each selected gen4 parent, constructs the
two representations at one shared parent-canvas scale, and writes only the
fragments referenced by the selected rows.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
from types import MappingProxyType
from typing import Dict, Mapping, Sequence

import numpy as np
from PIL import Image

from .gen4_pairwise30k import (
    SOURCE_ID,
    Gen4PairRow,
    Gen4Pairwise30kManifest,
    _load_binary_scalar_mask,
    _png_paths,
)
from .mask_representations import prepare_parent_mask_representations


SCHEMA_VERSION = "dunhuang-gen4-representations/0.1"
TARGET_LONG_SIDE = 800
CONTOUR_WIDTH = 10


class Gen4RepresentationError(ValueError):
    """The selected manifest cannot be safely materialized."""


def _selected_fragments(
    rows: Sequence[Gen4PairRow],
) -> Mapping[str, str]:
    """Return ``relative mask path -> parent ID`` without changing row order."""

    selected: Dict[str, str] = {}
    for row in rows:
        for relative_path, parent_id in (
            (row.fragment_a_relative_path, row.parent_a_id),
            (row.fragment_b_relative_path, row.parent_b_id),
        ):
            if not isinstance(relative_path, str) or "\\" in relative_path:
                raise Gen4RepresentationError(
                    "fragment paths must be canonical POSIX relative paths"
                )
            path = PurePosixPath(relative_path)
            if (
                path.is_absolute()
                or len(path.parts) != 2
                or any(part in {"", ".", ".."} for part in path.parts)
                or path.suffix.casefold() != ".png"
            ):
                raise Gen4RepresentationError(
                    "invalid gen4 fragment relative path: " + repr(relative_path)
                )
            if not isinstance(parent_id, str) or not parent_id:
                raise Gen4RepresentationError("parent IDs must be non-empty strings")
            if path.parts[0] != parent_id:
                raise Gen4RepresentationError(
                    "fragment path parent does not match pair-row parent ID: "
                    + repr(relative_path)
                )
            previous = selected.get(relative_path)
            if previous is not None and previous != parent_id:
                raise Gen4RepresentationError(
                    "one fragment path is associated with multiple parents: "
                    + repr(relative_path)
                )
            selected[relative_path] = parent_id
    if not selected:
        raise Gen4RepresentationError("manifest contains no selected fragments")
    return MappingProxyType(dict(sorted(selected.items())))


def _read_parent_masks(
    *,
    mask_root: Path,
    parent_id: str,
    fragments_per_parent: int,
    parent_canvas_size: int,
) -> Mapping[str, np.ndarray]:
    parent = mask_root / parent_id
    try:
        paths = _png_paths(parent, fragments_per_parent)
        values = {
            path.relative_to(mask_root).as_posix(): _load_binary_scalar_mask(
                path, parent_canvas_size
            )
            for path in paths
        }
    except (OSError, ValueError) as error:
        raise Gen4RepresentationError(
            "cannot read scalar gen4 masks for parent {!r}: {}".format(parent_id, error)
        ) from error
    return MappingProxyType(values)


def _write_binary_png(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(value, dtype=np.uint8) * np.uint8(255)
    Image.fromarray(pixels, mode="L").save(path, format="PNG")


def _verify_selected_rows(
    rows: Sequence[Gen4PairRow], *, filled_root: Path, contour_root: Path
) -> None:
    for row in rows:
        for relative_path in (
            row.fragment_a_relative_path,
            row.fragment_b_relative_path,
        ):
            if not (filled_root / relative_path).is_file():
                raise Gen4RepresentationError(
                    "missing filled representation for pair {} endpoint {}".format(
                        row.pair_id, relative_path
                    )
                )
            if not (contour_root / relative_path).is_file():
                raise Gen4RepresentationError(
                    "missing contour10 representation for pair {} endpoint {}".format(
                        row.pair_id, relative_path
                    )
                )


def _receipt(
    *,
    mask_root: Path,
    pair_count: int,
    fragment_count: int,
    parent_count: int,
) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_id": SOURCE_ID,
        "counts": {
            "selected_pair_rows": pair_count,
            "selected_unique_fragments": fragment_count,
            "parent_groups_read": parent_count,
            "filled_png": fragment_count,
            "contour10_png": fragment_count,
        },
        "roots": {
            "filled": "representations/filled",
            "contour10": "representations/contour10",
        },
        "config": {
            "source_mask_root": mask_root.as_posix(),
            "input": "scalar binary PNG with canonical pixels 0/255",
            "output": "PNG mode L with canonical pixels 0/255",
            "target_long_side": TARGET_LONG_SIDE,
            "model_canvas_shape": [TARGET_LONG_SIDE, TARGET_LONG_SIDE],
            "shared_parent_canvas_scale": True,
            "tight_crop_after_shared_scale": True,
            "center_pad_after_tight_crop": True,
            "absolute_parent_canvas_origin_exposed_to_model": False,
            "contour_width_pixels": CONTOUR_WIDTH,
            "contour_definition": (
                "M & ~binary_erosion(M, 3x3, iterations=10, border_value=0)"
            ),
            "rgb_used": False,
        },
        "pair_manifest_modified": False,
        "large_file_hashes_computed": False,
    }


def materialize_gen4_representations(
    manifest: Gen4Pairwise30kManifest,
    *,
    mask_root: Path,
    output_root: Path,
) -> Path:
    """Write filled and contour10 PNGs for selected gen4 fragments only.

    ``output_root`` must not already exist.  If decoding, representation, or
    output verification fails, the newly-created output is removed rather than
    leaving a partial package.  The supplied manifest object is never written.
    """

    if not isinstance(manifest, Gen4Pairwise30kManifest):
        raise TypeError("manifest must be Gen4Pairwise30kManifest")
    source_root = Path(mask_root)
    destination = Path(output_root)
    if not source_root.is_dir():
        raise Gen4RepresentationError(
            "mask_root must be an extracted gen4 no_erode directory"
        )
    if destination.exists():
        raise Gen4RepresentationError(
            "refusing to overwrite existing output_root: " + destination.as_posix()
        )
    if manifest.config.contour_width != CONTOUR_WIDTH:
        raise Gen4RepresentationError(
            "manifest contour_width must be exactly {}".format(CONTOUR_WIDTH)
        )
    if manifest.config.model_canvas_size != TARGET_LONG_SIDE:
        raise Gen4RepresentationError(
            "manifest model_canvas_size must be exactly {}".format(TARGET_LONG_SIDE)
        )

    rows_before = tuple(row.to_dict() for row in manifest.rows)
    selected = _selected_fragments(manifest.rows)
    selected_by_parent: Dict[str, set] = {}
    for relative_path, parent_id in selected.items():
        selected_by_parent.setdefault(parent_id, set()).add(relative_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise Gen4RepresentationError(
            "refusing to overwrite existing output_root: " + destination.as_posix()
        ) from error

    filled_root = destination / "representations" / "filled"
    contour_root = destination / "representations" / "contour10"
    try:
        for parent_id in sorted(selected_by_parent):
            parent_masks = _read_parent_masks(
                mask_root=source_root,
                parent_id=parent_id,
                fragments_per_parent=manifest.config.fragments_per_parent,
                parent_canvas_size=manifest.config.parent_canvas_size,
            )
            available = set(parent_masks)
            missing = selected_by_parent[parent_id] - available
            if missing:
                raise Gen4RepresentationError(
                    "selected masks are absent from parent {!r}: {}".format(
                        parent_id, sorted(missing)
                    )
                )
            representations = prepare_parent_mask_representations(
                parent_masks,
                target_long_side=TARGET_LONG_SIDE,
                contour_width=CONTOUR_WIDTH,
            )
            if tuple(representations.filled) != tuple(representations.contour):
                raise Gen4RepresentationError(
                    "filled and contour10 fragment keys differ for parent "
                    + repr(parent_id)
                )
            for relative_path in sorted(selected_by_parent[parent_id]):
                filled = representations.filled[relative_path]
                contour = representations.contour[relative_path]
                if filled.shape != contour.shape:
                    raise Gen4RepresentationError(
                        "filled and contour10 shapes differ for " + repr(relative_path)
                    )
                if np.any(contour & ~filled):
                    raise Gen4RepresentationError(
                        "contour10 is not a subset of filled mask for "
                        + repr(relative_path)
                    )
                _write_binary_png(filled_root / relative_path, filled)
                _write_binary_png(contour_root / relative_path, contour)

        _verify_selected_rows(
            manifest.rows, filled_root=filled_root, contour_root=contour_root
        )
        rows_after = tuple(row.to_dict() for row in manifest.rows)
        if rows_after != rows_before:
            raise Gen4RepresentationError(
                "pair rows changed while representations were materialized"
            )
        receipt = _receipt(
            mask_root=source_root,
            pair_count=len(manifest.rows),
            fragment_count=len(selected),
            parent_count=len(selected_by_parent),
        )
        (destination / "representation_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


__all__ = [
    "CONTOUR_WIDTH",
    "Gen4RepresentationError",
    "SCHEMA_VERSION",
    "TARGET_LONG_SIDE",
    "materialize_gen4_representations",
]
