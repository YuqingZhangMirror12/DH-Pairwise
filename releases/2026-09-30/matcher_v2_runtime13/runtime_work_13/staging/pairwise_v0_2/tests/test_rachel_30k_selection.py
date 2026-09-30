import json
from collections import Counter

import pytest

from staging.pairwise_v0_2.pairwise_data.rachel_30k_selection import (
    FragmentRecord,
    RachelSelectionError,
    RachelSplitQuota,
    WithinPairCandidate,
    build_rachel_30k_selection,
)


SMALL_QUOTAS = {
    "train": RachelSplitQuota(16, 8, 8),
    "val": RachelSplitQuota(2, 1, 1),
    "test": RachelSplitQuota(2, 1, 1),
}


def _population():
    fragments = []
    candidates = []
    assignments = {}
    split_counts = {"train": 16, "val": 2, "test": 2}
    cursor = 0
    for split, count in split_counts.items():
        for _ in range(count):
            image_name = "source-image-{:03d}".format(cursor)
            generator = "gen{}".format(2 + cursor % 4)
            group = "{}/group-{:03d}".format(generator, cursor)
            tokens = []
            for fragment_index in range(3):
                token = "{}/fragment-{}".format(group, fragment_index)
                tokens.append(token)
                fragments.append(
                    FragmentRecord(
                        fragment_token=token,
                        parent_group_id=group,
                        split_unit_id=image_name,
                        generator=generator,
                        fragment_id=str(fragment_index),
                        foreground_area=1000 + 10 * fragment_index + cursor,
                        bbox_aspect_ratio=1.0 + 0.01 * fragment_index,
                        model_mask_path="masks/{}.png".format(token),
                        contour_path="contours/{}.npz".format(token),
                    )
                )
            candidates.append(
                WithinPairCandidate(
                    pair_id=group + "/positive",
                    fragment_a_token=tokens[0],
                    fragment_b_token=tokens[1],
                    label=True,
                    label_origin="csv_neighbor_reciprocal_contour",
                    seam_length_px=128,
                    translation_a_to_b_rc=(0, 10),
                    correspondence_path="targets/{:03d}.npz".format(cursor),
                )
            )
            candidates.append(
                WithinPairCandidate(
                    pair_id=group + "/hard",
                    fragment_a_token=tokens[0],
                    fragment_b_token=tokens[2],
                    label=False,
                    label_origin="csv_nonneighbor_zero_seam",
                    negative_origin="same_folder_hard",
                    metadata={"seam_match_count": 0},
                )
            )
            assignments[image_name] = split
            cursor += 1
    return fragments, candidates, assignments


def test_exact_30k_shape_scaled_fixture_and_negative_origin_balance():
    fragments, candidates, assignments = _population()
    result = build_rachel_30k_selection(
        fragments,
        candidates,
        seed="fixture",
        split_quotas=SMALL_QUOTAS,
        lineage_assignments=assignments,
    )

    counts = Counter(
        (
            row.split,
            "positive" if row.candidate.label else row.candidate.negative_origin,
        )
        for row in result.rows
    )
    assert len(result.rows) == 40
    assert counts == {
        ("train", "positive"): 16,
        ("train", "same_folder_hard"): 8,
        ("train", "cross_folder_scale_matched"): 8,
        ("val", "positive"): 2,
        ("val", "same_folder_hard"): 1,
        ("val", "cross_folder_scale_matched"): 1,
        ("test", "positive"): 2,
        ("test", "same_folder_hard"): 1,
        ("test", "cross_folder_scale_matched"): 1,
    }


def test_cross_negatives_are_same_split_different_lineage_and_scale_matched():
    fragments, candidates, assignments = _population()
    result = build_rachel_30k_selection(
        fragments,
        candidates,
        seed="fixture",
        split_quotas=SMALL_QUOTAS,
        lineage_assignments=assignments,
    )
    unordered = set()
    for row in result.rows:
        key = row.candidate.canonical_pair_key
        assert key not in unordered
        unordered.add(key)
        assert assignments[row.fragment_a.split_unit_id] == row.split
        assert assignments[row.fragment_b.split_unit_id] == row.split
        if row.candidate.negative_origin == "cross_folder_scale_matched":
            assert row.fragment_a.split_unit_id != row.fragment_b.split_unit_id
            assert row.candidate.metadata["area_ratio"] <= 2
            assert row.candidate.metadata["bbox_aspect_ratio_ratio"] <= 2


def test_lineage_assignment_is_input_order_independent_and_never_splits_image_name():
    fragments, candidates, _ = _population()
    first = build_rachel_30k_selection(
        fragments, candidates, seed="fixture", split_quotas=SMALL_QUOTAS
    )
    second = build_rachel_30k_selection(
        reversed(fragments),
        reversed(candidates),
        seed="fixture",
        split_quotas=SMALL_QUOTAS,
    )
    assert dict(first.lineage_assignments) == dict(second.lineage_assignments)
    assert [row.to_dict() for row in first.rows] == [row.to_dict() for row in second.rows]
    assert Counter(first.lineage_assignments.values()) == {
        "train": 16,
        "val": 2,
        "test": 2,
    }


def test_reversed_pair_and_cross_lineage_within_candidate_are_rejected():
    fragments, candidates, assignments = _population()
    first = candidates[0]
    reversed_pair = WithinPairCandidate(
        pair_id="reversed",
        fragment_a_token=first.fragment_b_token,
        fragment_b_token=first.fragment_a_token,
        label=True,
        label_origin="csv_neighbor_reciprocal_contour",
        seam_length_px=128,
    )
    with pytest.raises(RachelSelectionError, match="duplicate/reversed"):
        build_rachel_30k_selection(
            fragments,
            candidates + [reversed_pair],
            split_quotas=SMALL_QUOTAS,
            lineage_assignments=assignments,
        )

    crossed = WithinPairCandidate(
        pair_id="crossed-positive",
        fragment_a_token=fragments[0].fragment_token,
        fragment_b_token=fragments[-1].fragment_token,
        label=True,
        label_origin="bad",
        seam_length_px=128,
    )
    with pytest.raises(RachelSelectionError, match="crosses image_name"):
        build_rachel_30k_selection(
            fragments,
            candidates + [crossed],
            split_quotas=SMALL_QUOTAS,
            lineage_assignments=assignments,
        )


def test_preprocess_json_aliases_remain_compatible():
    fragment = FragmentRecord.from_dict(
        {
            "fragment_token": "g/f0",
            "group_id": "g",
            "lineage_id": "source.jpg",
            "image_name": "source.jpg",
            "generator": "gen2",
            "fragment_id": "0",
            "foreground_area": 100,
            "bbox_aspect_ratio": 1.25,
            "model_mask_path": "m.png",
            "contour_path": "c.npz",
            "parent_mask_path": "p.png",
            "parent_to_model_offset_rc": [2, 3],
        }
    )
    candidate = WithinPairCandidate.from_dict(
        {
            "candidate_id": "p0",
            "fragment_a_token": "g/f0",
            "fragment_b_token": "g/f1",
            "label": False,
            "label_origin": "csv_nonneighbor_zero_seam",
            "negative_origin": "same_folder_hard",
            "seam_match_count": 0,
        }
    )
    assert fragment.split_unit_id == "source.jpg"
    assert fragment.metadata["parent_mask_path"] == "p.png"
    assert candidate.pair_id == "p0"
    assert json.loads(json.dumps(candidate.to_dict()))["negative_origin"] == "same_folder_hard"

