from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
from PIL import Image
import pytest
from scipy import ndimage

from staging.pairwise_v0_2.pairwise_data.gen4_pairwise30k import (
    Gen4PairRow,
    Gen4Pairwise30kConfig,
    Gen4Pairwise30kManifest,
    SplitQuota,
)
from staging.pairwise_v0_2.pairwise_data.gen4_representations import (
    Gen4RepresentationError,
    materialize_gen4_representations,
)


def _write_mask(path: Path, mask: np.ndarray, *, mode: str = "L") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = mask.astype(np.uint8) * 255
    if mode == "RGB":
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    Image.fromarray(pixels, mode=mode).save(path)


def _source(root: Path) -> Path:
    boxes = {
        "parent-a": (
            (slice(2, 22), slice(3, 13)),
            (slice(4, 14), slice(24, 29)),
            (slice(25, 35), slice(2, 12)),
            (slice(26, 36), slice(25, 35)),
        ),
        "parent-b": (
            (slice(1, 16), slice(2, 17)),
            (slice(1, 16), slice(20, 35)),
            (slice(20, 35), slice(2, 17)),
            (slice(20, 35), slice(20, 35)),
        ),
    }
    for parent, parent_boxes in boxes.items():
        for index, box in enumerate(parent_boxes):
            mask = np.zeros((40, 40), dtype=np.bool_)
            mask[box] = True
            _write_mask(root / parent / (str(index) + ".png"), mask)
    return root


def _manifest(mask_root: Path) -> Gen4Pairwise30kManifest:
    quotas = {
        "train": SplitQuota(8, 8),
        "val": SplitQuota(1, 1),
        "test": SplitQuota(1, 1),
    }
    config = Gen4Pairwise30kConfig(
        mask_root=mask_root,
        split_quotas=quotas,
        expected_parent_count=None,
        parent_canvas_size=40,
    )
    rows = (
        Gen4PairRow(
            pair_id="pair-first",
            split="train",
            label=True,
            direction_b_wrt_a="right",
            fragment_a_relative_path="parent-a/0.png",
            fragment_b_relative_path="parent-a/1.png",
            parent_a_id="parent-a",
            parent_b_id="parent-a",
            label_origin="fixture",
            negative_origin=None,
        ),
        Gen4PairRow(
            pair_id="pair-second",
            split="train",
            label=False,
            direction_b_wrt_a=None,
            fragment_a_relative_path="parent-a/0.png",
            fragment_b_relative_path="parent-b/2.png",
            parent_a_id="parent-a",
            parent_b_id="parent-b",
            label_origin="fixture",
            negative_origin="cross_parent_fixture",
        ),
    )
    empty_counts = MappingProxyType({"train": 0, "val": 0, "test": 0})
    empty_stats = MappingProxyType(
        {split: MappingProxyType({}) for split in ("train", "val", "test")}
    )
    return Gen4Pairwise30kManifest(
        config=config,
        parent_assignments=MappingProxyType({"parent-a": "train", "parent-b": "train"}),
        positive_candidate_counts=empty_counts,
        ignored_same_parent_nonadjacent_counts=empty_counts,
        rows=rows,
        negative_degree_statistics=empty_stats,
    )


def _read_bool(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        assert image.mode == "L"
        value = np.asarray(image)
    assert set(np.unique(value)).issubset({0, 255})
    return value == 255


def test_materializes_selected_fragments_once_without_changing_pair_rows(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "masks")
    manifest = _manifest(source)
    before = tuple(row.to_dict() for row in manifest.rows)

    output = materialize_gen4_representations(
        manifest, mask_root=source, output_root=tmp_path / "output"
    )

    assert tuple(row.to_dict() for row in manifest.rows) == before
    assert [row.pair_id for row in manifest.rows] == ["pair-first", "pair-second"]
    selected = {"parent-a/0.png", "parent-a/1.png", "parent-b/2.png"}
    for representation in ("filled", "contour10"):
        observed = {
            path.relative_to(output / "representations" / representation).as_posix()
            for path in (output / "representations" / representation).rglob("*.png")
        }
        assert observed == selected
    assert not (output / "representations" / "filled" / "parent-a" / "2.png").exists()

    receipt = json.loads(
        (output / "representation_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["counts"] == {
        "selected_pair_rows": 2,
        "selected_unique_fragments": 3,
        "parent_groups_read": 2,
        "filled_png": 3,
        "contour10_png": 3,
    }
    assert receipt["config"]["rgb_used"] is False
    assert receipt["config"]["target_long_side"] == 800
    assert receipt["config"]["model_canvas_shape"] == [800, 800]
    assert receipt["config"]["center_pad_after_tight_crop"] is True
    assert (
        receipt["config"]["absolute_parent_canvas_origin_exposed_to_model"]
        is False
    )
    assert receipt["config"]["contour_width_pixels"] == 10
    assert receipt["pair_manifest_modified"] is False
    assert receipt["large_file_hashes_computed"] is False


def test_written_views_share_scale_shape_and_exact_internal_contour(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "masks")
    manifest = _manifest(source)
    output = materialize_gen4_representations(
        manifest, mask_root=source, output_root=tmp_path / "output"
    )

    large = _read_bool(output / "representations" / "filled" / "parent-a" / "0.png")
    small = _read_bool(output / "representations" / "filled" / "parent-a" / "1.png")
    assert large.shape == small.shape == (800, 800)
    large_rows, large_columns = np.nonzero(large)
    small_rows, small_columns = np.nonzero(small)
    assert (large_rows.ptp() + 1, large_columns.ptp() + 1) == (400, 200)
    assert (small_rows.ptp() + 1, small_columns.ptp() + 1) == (200, 100)
    assert (large_rows.mean(), large_columns.mean()) == (399.5, 399.5)
    assert (small_rows.mean(), small_columns.mean()) == (399.5, 399.5)

    structure = np.ones((3, 3), dtype=np.bool_)
    for relative_path in (
        Path("parent-a/0.png"),
        Path("parent-a/1.png"),
        Path("parent-b/2.png"),
    ):
        filled = _read_bool(output / "representations" / "filled" / relative_path)
        contour = _read_bool(output / "representations" / "contour10" / relative_path)
        expected = filled & ~ndimage.binary_erosion(
            filled, structure=structure, iterations=10, border_value=0
        )
        assert filled.shape == contour.shape
        assert np.all(contour <= filled)
        assert np.array_equal(contour, expected)


def test_existing_output_is_rejected_without_modification(tmp_path: Path) -> None:
    source = _source(tmp_path / "masks")
    manifest = _manifest(source)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "belongs-to-user.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(Gen4RepresentationError, match="refusing to overwrite"):
        materialize_gen4_representations(manifest, mask_root=source, output_root=output)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output / "representation_receipt.json").exists()


def test_rgb_parent_member_is_rejected_and_partial_output_removed(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "masks")
    rgb = np.zeros((40, 40), dtype=np.bool_)
    rgb[1:12, 1:12] = True
    _write_mask(source / "parent-a" / "3.png", rgb, mode="RGB")
    manifest = _manifest(source)
    output = tmp_path / "output"

    with pytest.raises(Gen4RepresentationError, match="scalar gen4 masks"):
        materialize_gen4_representations(manifest, mask_root=source, output_root=output)

    assert not output.exists()
