from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from staging.pairwise_v0_2.pairwise_data.shredding_pipeline_pairs import (
    GeneratorSpec,
    build_cross_parent_scale_matched_negatives,
    build_shredding_pipeline_candidate_pool,
    iter_shredding_pipeline_pair_inventories,
    read_parent_pair_inventory,
)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _rectangle(shape, rows, columns):
    mask = np.zeros(shape, dtype=np.bool_)
    mask[rows, columns] = True
    return mask


def _write_parent(parent: Path, masks) -> None:
    for index, mask in enumerate(masks):
        _write_mask(parent / (str(index) + ".png"), mask)


def test_short_seam_retains_positive_semantics_but_is_not_selectable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "voronoi_masks"
    spec = GeneratorSpec("gen4fixture", 4)
    parent = root / spec.relative_root / "case-1"
    _write_parent(
        parent,
        (
            _rectangle((80, 80), slice(0, 63), slice(0, 10)),
            _rectangle((80, 80), slice(0, 63), slice(10, 20)),
            _rectangle((80, 80), slice(70, 75), slice(60, 65)),
            _rectangle((80, 80), slice(70, 75), slice(70, 75)),
        ),
    )

    inventory = read_parent_pair_inventory(
        root,
        parent,
        spec,
        seed="fixture",
        parent_canvas_size=80,
        split="train",
    )

    assert len(inventory.pair_candidates) == 6
    assert inventory.selectable_positives == ()
    assert len(inventory.excluded_short_seam_positives) == 1
    short = inventory.excluded_short_seam_positives[0]
    assert short.label is True
    assert short.metadata["seam_edge_count"] == 63
    assert short.direction_b_wrt_a == "right"
    assert short.selection_exclusion_reason == (
        "positive_seam_shorter_than_64_pixels"
    )
    assert len(inventory.same_parent_hard_negatives) == 5
    assert all(
        candidate.label is False
        and candidate.metadata["seam_edge_count"] == 0
        and candidate.negative_origin == "same_parent_nonadjacent_hard"
        and candidate.direction_b_wrt_a is None
        for candidate in inventory.same_parent_hard_negatives
    )
    assert all("origin" not in key for key in short.metadata)


def test_seam_threshold_is_inclusive_at_64(tmp_path: Path) -> None:
    root = tmp_path / "voronoi_masks"
    spec = GeneratorSpec("gen2fixture", 2)
    parent = root / spec.relative_root / "case-1"
    _write_parent(
        parent,
        (
            _rectangle((80, 80), slice(0, 64), slice(0, 20)),
            _rectangle((80, 80), slice(0, 64), slice(20, 40)),
        ),
    )

    inventory = read_parent_pair_inventory(
        root,
        parent,
        spec,
        seed="fixture",
        parent_canvas_size=80,
        split="val",
    )

    assert len(inventory.selectable_positives) == 1
    candidate = inventory.selectable_positives[0]
    assert candidate.label is True
    assert candidate.main_training_eligible is True
    assert candidate.metadata["seam_edge_count"] == 64
    assert inventory.excluded_short_seam_positives == ()


def test_iterator_covers_2_3_4_5_fragment_generators_and_skips_empty_parents(
    tmp_path: Path,
) -> None:
    root = tmp_path / "voronoi_masks"
    specs = tuple(GeneratorSpec("gen{}fixture".format(count), count) for count in range(2, 6))
    for spec in specs:
        parent = root / spec.relative_root / "case-1"
        width = 80 // spec.fragments_per_parent
        masks = []
        for index in range(spec.fragments_per_parent):
            start = index * width
            stop = 80 if index + 1 == spec.fragments_per_parent else start + width
            masks.append(_rectangle((80, 80), slice(0, 80), slice(start, stop)))
        _write_parent(parent, masks)
    (root / specs[0].relative_root / "known-empty").mkdir()

    inventories = tuple(
        iter_shredding_pipeline_pair_inventories(
            root,
            seed="fixture",
            generators=specs,
            parent_canvas_size=80,
        )
    )

    assert [value.generator.fragments_per_parent for value in inventories] == [
        2,
        3,
        4,
        5,
    ]
    assert all(value.selectable_positives for value in inventories)
    assert all(
        candidate.metadata["parent_canvas_shape"] == [80, 80]
        for inventory in inventories
        for candidate in inventory.pair_candidates
    )


def test_cross_folder_pool_is_scale_matched_low_degree_and_split_safe(
    tmp_path: Path,
) -> None:
    root = tmp_path / "voronoi_masks"
    specs = tuple(
        GeneratorSpec("gen{}fixture".format(count), count)
        for count in range(2, 6)
    )
    inventories = []
    for spec in specs:
        width = 80 // spec.fragments_per_parent
        for parent_index in range(2):
            parent = root / spec.relative_root / "train-{}".format(parent_index)
            masks = []
            for fragment_index in range(spec.fragments_per_parent):
                start = fragment_index * width
                stop = (
                    80
                    if fragment_index + 1 == spec.fragments_per_parent
                    else start + width
                )
                masks.append(
                    _rectangle((80, 80), slice(0, 80), slice(start, stop))
                )
            _write_parent(parent, masks)
            inventories.append(
                read_parent_pair_inventory(
                    root,
                    parent,
                    spec,
                    seed="fixture",
                    parent_canvas_size=80,
                    split="train",
                )
            )
    val_spec = specs[0]
    val_parent = root / val_spec.relative_root / "val-only"
    _write_parent(
        val_parent,
        (
            _rectangle((80, 80), slice(0, 80), slice(0, 40)),
            _rectangle((80, 80), slice(0, 80), slice(40, 80)),
        ),
    )
    inventories.append(
        read_parent_pair_inventory(
            root,
            val_parent,
            val_spec,
            seed="fixture",
            parent_canvas_size=80,
            split="val",
        )
    )

    cross = build_cross_parent_scale_matched_negatives(
        inventories,
        seed="fixture",
        neighbor_window=32,
    )
    reversed_cross = build_cross_parent_scale_matched_negatives(
        reversed(inventories),
        seed="fixture",
        neighbor_window=32,
    )
    assert [value.to_dict() for value in cross] == [
        value.to_dict() for value in reversed_cross
    ]

    token_lineage = {
        fragment.token: (
            inventory.generator.generator_id,
            inventory.split,
            inventory.parent_group_id,
        )
        for inventory in inventories
        for fragment in inventory.fragments
    }
    observed_generator_counts = {spec.generator_id: 0 for spec in specs}
    fragment_degree = {}
    parent_degree = {}
    positive_keys = {
        candidate.canonical_pair_key
        for inventory in inventories
        for candidate in inventory.pair_candidates
        if candidate.label
    }
    for candidate in cross:
        lineage_a = token_lineage[candidate.fragment_a_token]
        lineage_b = token_lineage[candidate.fragment_b_token]
        assert lineage_a[0] == lineage_b[0]
        assert lineage_a[1] == lineage_b[1]
        assert lineage_a[2] != lineage_b[2]
        assert candidate.fragment_a_parent_group_id != (
            candidate.fragment_b_parent_group_id
        )
        assert candidate.negative_origin == (
            "cross_parent_same_generator_same_split_scale_matched"
        )
        assert candidate.direction_b_wrt_a is None
        assert candidate.metadata["foreground_area_ratio"] <= 2.0
        assert candidate.metadata["bbox_aspect_ratio_ratio"] <= 2.0
        assert candidate.canonical_pair_key not in positive_keys
        observed_generator_counts[lineage_a[0]] += 1
        for token, parent_id in (
            (candidate.fragment_a_token, candidate.fragment_a_parent_group_id),
            (candidate.fragment_b_token, candidate.fragment_b_parent_group_id),
        ):
            fragment_degree[token] = fragment_degree.get(token, 0) + 1
            parent_degree[parent_id] = parent_degree.get(parent_id, 0) + 1
    assert all(observed_generator_counts[spec.generator_id] > 0 for spec in specs)
    assert max(fragment_degree.values()) <= 2
    assert max(parent_degree.values()) <= 8
    assert all("val-only" not in token for candidate in cross for token in candidate.canonical_pair_key)

    pool = build_shredding_pipeline_candidate_pool(
        inventories,
        seed="fixture",
        neighbor_window=32,
    )
    assert len({candidate.canonical_pair_key for candidate in pool}) == len(pool)
    for fragment_count in (4, 5):
        source_id = next(
            spec.source_id
            for spec in specs
            if spec.fragments_per_parent == fragment_count
        )
        origins = {
            candidate.negative_origin
            for candidate in pool
            if candidate.source_id == source_id and not candidate.label
        }
        assert "same_parent_nonadjacent_hard" in origins
        assert "cross_parent_same_generator_same_split_scale_matched" in origins
