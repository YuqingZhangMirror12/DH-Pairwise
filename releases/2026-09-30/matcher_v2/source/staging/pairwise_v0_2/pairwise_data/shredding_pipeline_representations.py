"""Materialize leakage-safe model inputs for selected shredding pairs only."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil
from typing import Dict, Mapping, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from .mask_representations import prepare_parent_mask_representations
from .pairwise_30k_protocol import PairCandidate, SelectedPair
from .shredding_pipeline_pairs import (
    DEFAULT_GENERATOR_SPECS,
    DEFAULT_PARENT_CANVAS_SIZE,
    GeneratorSpec,
    SOURCE_ID_PREFIX,
    _load_binary_mask,
    _mask_paths,
)


MODEL_CANVAS_SIZE = 800
CONTOUR_WIDTH = 10


class ShreddingRepresentationError(ValueError):
    """Selected rows cannot safely yield source-neutral model tensors."""


def _candidate(value: object) -> PairCandidate:
    if isinstance(value, PairCandidate):
        return value
    if isinstance(value, SelectedPair):
        return value.candidate
    raise TypeError("rows must contain PairCandidate or SelectedPair values")


def _relative_path_from_token(token: str) -> str:
    prefix = SOURCE_ID_PREFIX + "/"
    if not token.startswith(prefix):
        raise ShreddingRepresentationError(
            "fragment token is outside shredding_pipeline: " + repr(token)
        )
    relative = token[len(prefix) :]
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or len(path.parts) != 4
        or path.parts[1] != "no_erode"
        or path.suffix.casefold() != ".png"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ShreddingRepresentationError(
            "invalid shredding fragment token: " + repr(token)
        )
    return path.as_posix()


def _selected_paths(
    rows: Sequence[object],
) -> Mapping[Tuple[str, str], Set[str]]:
    selected: Dict[Tuple[str, str], Set[str]] = {}
    if not rows:
        raise ShreddingRepresentationError("selected rows cannot be empty")
    for value in rows:
        candidate = _candidate(value)
        if not candidate.main_training_eligible:
            raise ShreddingRepresentationError(
                "excluded candidate cannot enter representation preprocessing: "
                + candidate.pair_id
            )
        for token in (candidate.fragment_a_token, candidate.fragment_b_token):
            relative = _relative_path_from_token(token)
            parts = PurePosixPath(relative).parts
            selected.setdefault((parts[0], parts[2]), set()).add(relative)
    return selected


def _write_binary_png(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(value, dtype=np.uint8) * np.uint8(255)
    Image.fromarray(pixels, mode="L").save(path, format="PNG")


def materialize_shredding_pipeline_representations(
    rows: Sequence[object],
    *,
    pipeline_root: Path,
    output_root: Path,
    generators: Sequence[GeneratorSpec] = DEFAULT_GENERATOR_SPECS,
    parent_canvas_size: int = DEFAULT_PARENT_CANVAS_SIZE,
) -> Path:
    """Write 800x800 filled/contour10 PNGs for selected endpoints only.

    Labels and directions must already have been computed in aligned parent
    coordinates.  This function supplies only tight-cropped, centred tensors;
    it never copies a parent-canvas bbox origin into a model artifact.
    """

    source = Path(pipeline_root)
    destination = Path(output_root)
    if not source.is_dir():
        raise ShreddingRepresentationError(
            "pipeline_root must be an extracted voronoi_masks directory"
        )
    if destination.exists():
        raise ShreddingRepresentationError(
            "refusing to overwrite output_root: " + destination.as_posix()
        )
    generator_by_id = {spec.generator_id: spec for spec in generators}
    if len(generator_by_id) != len(tuple(generators)):
        raise ShreddingRepresentationError("generator IDs must be unique")
    selected = _selected_paths(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    filled_root = destination / "filled"
    contour_root = destination / "contour10"
    try:
        for (generator_id, parent_id), relative_paths in sorted(selected.items()):
            try:
                spec = generator_by_id[generator_id]
            except KeyError as error:
                raise ShreddingRepresentationError(
                    "selected row references unknown generator: " + generator_id
                ) from error
            parent = source / spec.relative_root / parent_id
            paths = _mask_paths(parent, spec.fragments_per_parent)
            masks = {
                path.relative_to(source).as_posix(): _load_binary_mask(
                    path, parent_canvas_size
                )
                for path in paths
            }
            missing = relative_paths - set(masks)
            if missing:
                raise ShreddingRepresentationError(
                    "selected masks are missing: " + repr(sorted(missing))
                )
            representations = prepare_parent_mask_representations(
                masks,
                target_long_side=MODEL_CANVAS_SIZE,
                contour_width=CONTOUR_WIDTH,
            )
            for relative in sorted(relative_paths):
                filled = representations.filled[relative]
                contour = representations.contour[relative]
                if filled.shape != (MODEL_CANVAS_SIZE, MODEL_CANVAS_SIZE):
                    raise AssertionError("internal error: filled model shape")
                if contour.shape != filled.shape or np.any(contour & ~filled):
                    raise AssertionError("internal error: contour model shape")
                _write_binary_png(filled_root / relative, filled)
                _write_binary_png(contour_root / relative, contour)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


__all__ = [
    "CONTOUR_WIDTH",
    "MODEL_CANVAS_SIZE",
    "ShreddingRepresentationError",
    "materialize_shredding_pipeline_representations",
]
