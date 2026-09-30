from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from staging.pairwise_v0_2.pairwise_data.gen4_pairwise30k import (
    Gen4Pairwise30kConfig,
    Gen4Pairwise30kError,
    Gen4Pairwise30kShortfall,
    SplitQuota,
    build_gen4_pairwise30k_manifest,
    write_gen4_pairwise30k_manifest,
)


TINY_QUOTAS = {
    "train": SplitQuota(8, 8),
    "val": SplitQuota(1, 1),
    "test": SplitQuota(1, 1),
}


def _write_mask(path: Path, mask: np.ndarray, *, mode: str = "L") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "RGB":
        value = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    else:
        value = mask.astype(np.uint8) * 255
    Image.fromarray(value, mode=mode).save(path)


def _four_fragment_parent(parent: Path, *, size: int = 16) -> None:
    half = size // 2
    slices = (
        (slice(0, half), slice(0, half)),
        (slice(0, half), slice(half, size)),
        (slice(half, size), slice(0, half)),
        (slice(half, size), slice(half, size)),
    )
    for index, (rows, columns) in enumerate(slices):
        mask = np.zeros((size, size), dtype=np.bool_)
        mask[rows, columns] = True
        _write_mask(parent / (str(index) + ".png"), mask)


def _source(root: Path, count: int = 20) -> Path:
    no_erode = root / "gen4voronoi" / "no_erode"
    for index in reversed(range(count)):
        _four_fragment_parent(no_erode / "parent-{:03d}".format(index))
    return no_erode


def _config(mask_root: Path, **changes) -> Gen4Pairwise30kConfig:
    values = {
        "mask_root": mask_root,
        "seed": "fixture-seed",
        "split_quotas": TINY_QUOTAS,
        "expected_parent_count": 20,
        "parent_canvas_size": 16,
        "minimum_positive_seam_edge_count": 1,
        "negative_neighbor_window": 16,
    }
    values.update(changes)
    return Gen4Pairwise30kConfig(**values)


def test_exact_counts_parent_split_and_exact_seam_labels(tmp_path: Path) -> None:
    config = _config(_source(tmp_path))

    manifest = build_gen4_pairwise30k_manifest(config)

    counts = Counter((row.split, row.label) for row in manifest.rows)
    assert counts == {
        ("train", True): 8,
        ("train", False): 8,
        ("val", True): 1,
        ("val", False): 1,
        ("test", True): 1,
        ("test", False): 1,
    }
    assert Counter(manifest.parent_assignments.values()) == {
        "train": 16,
        "val": 2,
        "test": 2,
    }
    parent_splits = defaultdict(set)
    for row in manifest.rows:
        parent_splits[row.parent_a_id].add(row.split)
        parent_splits[row.parent_b_id].add(row.split)
        if row.label:
            assert row.parent_a_id == row.parent_b_id
            assert row.direction_b_wrt_a in {"left", "right", "above", "below"}
            assert row.seam_edge_count == 8
        else:
            assert row.parent_a_id != row.parent_b_id
            assert row.direction_b_wrt_a is None
            assert (
                manifest.parent_assignments[row.parent_a_id]
                == manifest.parent_assignments[row.parent_b_id]
                == row.split
            )
            assert row.foreground_area_ratio == 1.0
            assert row.bbox_aspect_ratio_ratio == 1.0
    assert all(len(splits) == 1 for splits in parent_splits.values())
    # A four-quadrant parent has four edge-neighbour positives and two diagonal
    # non-neighbours.  The latter are ignored, never relabelled positive/negative.
    assert sum(manifest.positive_candidate_counts.values()) == 20 * 4
    assert sum(manifest.ignored_same_parent_nonadjacent_counts.values()) == 20 * 2
    for row in (value for value in manifest.rows if value.label):
        stems = {
            Path(row.fragment_a_relative_path).stem,
            Path(row.fragment_b_relative_path).stem,
        }
        assert stems not in ({"0", "3"}, {"1", "2"})
    assert all(
        row.parent_a_id != row.parent_b_id for row in manifest.rows if not row.label
    )


def test_deterministic_unordered_identity_low_reuse_and_output(tmp_path: Path) -> None:
    config = _config(_source(tmp_path / "input"))

    first = build_gen4_pairwise30k_manifest(config)
    second = build_gen4_pairwise30k_manifest(config)

    assert [row.to_dict() for row in first.rows] == [
        row.to_dict() for row in second.rows
    ]
    keys = [row.canonical_pair_key for row in first.rows]
    assert len(keys) == len(set(keys))
    assert len({row.pair_id for row in first.rows}) == len(first.rows)
    for split in ("train", "val", "test"):
        observed = first.negative_degree_statistics[split]
        assert observed["fragment"]["max"] <= config.max_negative_degree_per_fragment
        assert observed["parent"]["max"] <= config.max_negative_degree_per_parent

    output = write_gen4_pairwise30k_manifest(first, tmp_path / "manifest")
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["source"]["rgb_used"] is False
    assert protocol["input_contract"]["canonical_pixel_values"] == [0, 255]
    assert (
        protocol["representations"]["contour10"][
            "shares_pair_ids_splits_labels_and_order"
        ]
        is True
    )
    assert protocol["large_file_hashes_computed"] is False
    for split, expected in (("train", 16), ("val", 2), ("test", 2)):
        rows = [
            json.loads(line)
            for line in (output / "splits" / (split + ".jsonl"))
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == expected
        assert all(row["split"] == split for row in rows)


def test_rgb_png_is_rejected_instead_of_converted_to_mask(tmp_path: Path) -> None:
    root = _source(tmp_path)
    rgb_path = root / "parent-007" / "2.png"
    mask = np.zeros((16, 16), dtype=np.bool_)
    mask[:8, :8] = True
    _write_mask(rgb_path, mask, mode="RGB")

    with pytest.raises(Gen4Pairwise30kError, match="RGB/palette image rejected"):
        build_gen4_pairwise30k_manifest(_config(root))


def test_clear_shortfall_when_a_split_has_only_one_parent(tmp_path: Path) -> None:
    root = _source(tmp_path, count=10)
    config = _config(root, expected_parent_count=10)

    with pytest.raises(Gen4Pairwise30kShortfall, match="at least two parent groups"):
        build_gen4_pairwise30k_manifest(config)
