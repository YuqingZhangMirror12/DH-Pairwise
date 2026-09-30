from __future__ import annotations

from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
import pytest

from staging.pairwise_v0_2.pairwise_data.shredding_pipeline_pairs import (
    GeneratorSpec,
    read_parent_pair_inventory,
)
from staging.pairwise_v0_2.pairwise_data.shredding_pipeline_representations import (
    ShreddingRepresentationError,
    materialize_shredding_pipeline_representations,
)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _rectangle(rows, columns):
    mask = np.zeros((80, 80), dtype=np.bool_)
    mask[rows, columns] = True
    return mask


def _read_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        assert image.mode == "L"
        pixels = np.asarray(image)
    assert set(np.unique(pixels)).issubset({0, 255})
    return pixels


def test_selected_model_views_are_exact_800_binary_centered_and_origin_free(
    tmp_path: Path,
) -> None:
    root = tmp_path / "voronoi_masks"
    spec = GeneratorSpec("gen2fixture", 2)
    parent = root / spec.relative_root / "case-1"
    _write_mask(parent / "0.png", _rectangle(slice(5, 75), slice(5, 25)))
    _write_mask(parent / "1.png", _rectangle(slice(5, 75), slice(25, 35)))
    inventory = read_parent_pair_inventory(
        root,
        parent,
        spec,
        seed="fixture",
        parent_canvas_size=80,
        split="train",
    )
    candidate = inventory.selectable_positives[0]
    output = materialize_shredding_pipeline_representations(
        [candidate],
        pipeline_root=root,
        output_root=tmp_path / "output",
        generators=(spec,),
        parent_canvas_size=80,
    )

    paths = []
    for token in (candidate.fragment_a_token, candidate.fragment_b_token):
        relative = PurePosixPath(token.split("/", 1)[1])
        paths.append(output / "filled" / relative)
    first, second = (_read_pixels(path) for path in paths)
    assert first.shape == second.shape == (800, 800)
    first_rows, first_columns = np.nonzero(first)
    second_rows, second_columns = np.nonzero(second)
    assert (np.ptp(first_rows) + 1, np.ptp(first_columns) + 1) == (700, 200)
    assert (np.ptp(second_rows) + 1, np.ptp(second_columns) + 1) == (700, 100)
    assert (first_rows.mean(), first_columns.mean()) == (399.5, 399.5)
    assert (second_rows.mean(), second_columns.mean()) == (399.5, 399.5)
    assert candidate.direction_b_wrt_a == "right"
    assert all("origin" not in key for key in candidate.metadata)


def test_excluded_short_positive_is_rejected_before_any_output_is_written(
    tmp_path: Path,
) -> None:
    root = tmp_path / "voronoi_masks"
    spec = GeneratorSpec("gen2fixture", 2)
    parent = root / spec.relative_root / "case-short"
    _write_mask(parent / "0.png", _rectangle(slice(0, 63), slice(0, 20)))
    _write_mask(parent / "1.png", _rectangle(slice(0, 63), slice(20, 40)))
    inventory = read_parent_pair_inventory(
        root,
        parent,
        spec,
        seed="fixture",
        parent_canvas_size=80,
        split="train",
    )
    short = inventory.excluded_short_seam_positives[0]
    output = tmp_path / "output"

    with pytest.raises(ShreddingRepresentationError, match="excluded candidate"):
        materialize_shredding_pipeline_representations(
            [short],
            pipeline_root=root,
            output_root=output,
            generators=(spec,),
            parent_canvas_size=80,
        )

    assert short.label is True
    assert not output.exists()
