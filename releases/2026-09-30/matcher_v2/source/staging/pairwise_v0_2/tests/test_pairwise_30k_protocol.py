import json
from collections import Counter, defaultdict

import pytest

from staging.pairwise_v0_2.pairwise_data.pairwise_30k_protocol import (
    CandidateShortfallError,
    LabelQuota,
    PairCandidate,
    Pairwise30kProtocolError,
    build_pairwise_30k_protocol,
    stable_split_for_unit,
)


SMALL_QUOTAS = {
    "train": LabelQuota(8, 8),
    "val": LabelQuota(1, 1),
    "test": LabelQuota(1, 1),
}


def _candidate(
    index,
    *,
    source="mm",
    split="train",
    label=True,
    unit=None,
    parent=None,
    b_unit=None,
    b_parent=None,
):
    polarity = "p" if label else "n"
    unit = unit or "{}/{}/{}/unit-{}".format(source, split, polarity, index)
    parent = parent or "{}/{}/{}/parent-{}".format(source, split, polarity, index)
    b_unit = b_unit or unit
    b_parent = b_parent or parent
    return PairCandidate(
        pair_id="{}/{}/{}/pair-{}".format(source, split, polarity, index),
        source_id=source,
        fragment_a_parent_group_id=parent,
        fragment_b_parent_group_id=b_parent,
        fragment_a_split_unit_id=unit,
        fragment_b_split_unit_id=b_unit,
        fragment_a_token="{}/{}/{}/a-{}".format(source, split, polarity, index),
        fragment_b_token="{}/{}/{}/b-{}".format(source, split, polarity, index),
        label=label,
        label_origin="fixture-adjacency",
        negative_origin=None if label else "fixture-negative",
        metadata={"representation": "representation-independent"},
    )


def _single_source_pool(extra=0):
    candidates = []
    assignments = {}
    counts = {"train": 8 + extra, "val": 1 + extra, "test": 1 + extra}
    for split, count in counts.items():
        for label in (True, False):
            for index in range(count):
                candidate = _candidate(index, split=split, label=label)
                candidates.append(candidate)
                assignments[candidate.fragment_a_split_unit_id] = split
    return candidates, assignments


def test_exact_balanced_counts_are_deterministic_and_order_independent():
    candidates, assignments = _single_source_pool(extra=3)

    first = build_pairwise_30k_protocol(
        candidates,
        seed="fixture-seed",
        split_quotas=SMALL_QUOTAS,
        unit_assignments=assignments,
    )
    second = build_pairwise_30k_protocol(
        reversed(candidates),
        seed="fixture-seed",
        split_quotas=SMALL_QUOTAS,
        unit_assignments=dict(reversed(list(assignments.items()))),
    )

    assert first.to_dict() == second.to_dict()
    assert len(first.rows) == 20
    counts = Counter((row.split, row.candidate.label) for row in first.rows)
    assert counts == {
        ("train", True): 8,
        ("train", False): 8,
        ("val", True): 1,
        ("val", False): 1,
        ("test", True): 1,
        ("test", False): 1,
    }
    assert len({row.candidate.pair_id for row in first.rows}) == len(first.rows)
    assert [json.loads(line) for line in first.to_jsonl().splitlines()] == [
        row.to_dict() for row in first.rows
    ]


def test_explicitly_excluded_positive_keeps_label_but_never_enters_selection():
    candidates, assignments = _single_source_pool(extra=1)
    short = PairCandidate(
        pair_id="mm/train/short-positive",
        source_id="mm",
        fragment_a_parent_group_id="mm/train/short-parent",
        fragment_b_parent_group_id="mm/train/short-parent",
        fragment_a_split_unit_id="mm/train/short-unit",
        fragment_b_split_unit_id="mm/train/short-unit",
        fragment_a_token="mm/train/short-a",
        fragment_b_token="mm/train/short-b",
        label=True,
        label_origin="exact_aligned_mask_4_neighbor_seam",
        direction_b_wrt_a="right",
        metadata={"seam_edge_count": 63},
        main_training_eligible=False,
        selection_exclusion_reason="positive_seam_shorter_than_64_pixels",
    )
    candidates.append(short)

    result = build_pairwise_30k_protocol(
        candidates,
        seed="fixture-seed",
        split_quotas=SMALL_QUOTAS,
        unit_assignments=assignments,
    )

    assert short.label is True
    assert short in result.excluded_before_selection
    assert all(row.candidate.pair_id != short.pair_id for row in result.rows)
    assert short.fragment_a_split_unit_id not in result.unit_assignments


def test_parent_and_split_units_never_cross_splits_and_cross_parent_is_supported():
    candidates, assignments = _single_source_pool()
    replaced = candidates.index(
        next(
            candidate
            for candidate in candidates
            if candidate.source_id == "mm"
            and candidate.label is False
            and assignments[candidate.fragment_a_split_unit_id] == "train"
        )
    )
    cross = _candidate(
        99,
        split="train",
        label=False,
        unit="mm/train/unit-cross-a",
        parent="mm/train/parent-cross-a",
        b_unit="mm/train/unit-cross-b",
        b_parent="mm/train/parent-cross-b",
    )
    assignments.pop(candidates[replaced].fragment_a_split_unit_id)
    candidates[replaced] = cross
    assignments[cross.fragment_a_split_unit_id] = "train"
    assignments[cross.fragment_b_split_unit_id] = "train"

    result = build_pairwise_30k_protocol(
        candidates,
        split_quotas=SMALL_QUOTAS,
        unit_assignments=assignments,
    )

    assert any(row.candidate.pair_id == cross.pair_id for row in result.rows)
    parent_splits = defaultdict(set)
    unit_splits = defaultdict(set)
    for row in result.rows:
        candidate = row.candidate
        parent_splits[candidate.fragment_a_parent_group_id].add(row.split)
        parent_splits[candidate.fragment_b_parent_group_id].add(row.split)
        unit_splits[candidate.fragment_a_split_unit_id].add(row.split)
        unit_splits[candidate.fragment_b_split_unit_id].add(row.split)
    assert all(len(splits) == 1 for splits in parent_splits.values())
    assert all(len(splits) == 1 for splits in unit_splits.values())


def test_cross_split_endpoint_pair_is_rejected():
    candidates, assignments = _single_source_pool()
    cross = _candidate(
        80,
        split="train",
        label=False,
        unit="mm/train/cross-a",
        b_unit="mm/test/cross-b",
        parent="mm/train/cross-parent-a",
        b_parent="mm/test/cross-parent-b",
    )
    candidates.append(cross)
    assignments[cross.fragment_a_split_unit_id] = "train"
    assignments[cross.fragment_b_split_unit_id] = "test"

    with pytest.raises(Pairwise30kProtocolError, match="different splits"):
        build_pairwise_30k_protocol(
            candidates,
            split_quotas=SMALL_QUOTAS,
            unit_assignments=assignments,
        )


@pytest.mark.parametrize(
    "second_label,match", [(True, "duplicate"), (False, "conflict")]
)
def test_reversed_pair_is_not_a_second_sample(second_label, match):
    first = _candidate(0)
    reversed_candidate = PairCandidate(
        pair_id="mm/train/reversed",
        source_id=first.source_id,
        fragment_a_parent_group_id=first.fragment_b_parent_group_id,
        fragment_b_parent_group_id=first.fragment_a_parent_group_id,
        fragment_a_split_unit_id=first.fragment_b_split_unit_id,
        fragment_b_split_unit_id=first.fragment_a_split_unit_id,
        fragment_a_token=first.fragment_b_token,
        fragment_b_token=first.fragment_a_token,
        label=second_label,
        label_origin="fixture-adjacency",
        negative_origin=None if second_label else "fixture-negative",
    )

    with pytest.raises(Pairwise30kProtocolError, match=match):
        build_pairwise_30k_protocol([first, reversed_candidate])


def test_self_pair_and_parent_with_multiple_units_are_rejected():
    first = _candidate(0)
    with pytest.raises(Pairwise30kProtocolError, match="self pair"):
        PairCandidate(
            pair_id="self",
            source_id="mm",
            fragment_a_parent_group_id="parent",
            fragment_b_parent_group_id="parent",
            fragment_a_split_unit_id="unit",
            fragment_b_split_unit_id="unit",
            fragment_a_token="fragment",
            fragment_b_token="fragment",
            label=True,
            label_origin="fixture-adjacency",
        )

    inconsistent = _candidate(
        1,
        unit="mm/other-unit",
        parent=first.fragment_a_parent_group_id,
    )
    with pytest.raises(Pairwise30kProtocolError, match="parent group spans"):
        build_pairwise_30k_protocol([first, inconsistent])


def test_shortfall_names_source_split_and_label():
    candidates, assignments = _single_source_pool()
    missing = next(
        candidate
        for candidate in candidates
        if candidate.label is False
        and assignments[candidate.fragment_a_split_unit_id] == "test"
    )
    candidates.remove(missing)
    assignments.pop(missing.fragment_a_split_unit_id)

    with pytest.raises(CandidateShortfallError) as raised:
        build_pairwise_30k_protocol(
            candidates,
            split_quotas=SMALL_QUOTAS,
            unit_assignments=assignments,
        )
    assert raised.value.shortfalls == (
        {
            "source": "mm",
            "split": "test",
            "label": "negative",
            "requested": 1,
            "available": 0,
        },
    )


def test_multiple_sources_require_and_honor_explicit_cell_quotas():
    candidates = []
    assignments = {}
    source_counts = {
        "mm": {"train": 4, "val": 1, "test": 0},
        "eccv": {"train": 4, "val": 0, "test": 1},
    }
    source_quotas = {}
    for source, split_counts in source_counts.items():
        source_quotas[source] = {}
        for split, count in split_counts.items():
            source_quotas[source][split] = LabelQuota(count, count)
            for label in (True, False):
                for index in range(count):
                    candidate = _candidate(
                        index,
                        source=source,
                        split=split,
                        label=label,
                    )
                    candidates.append(candidate)
                    assignments[candidate.fragment_a_split_unit_id] = split

    with pytest.raises(Pairwise30kProtocolError, match="multiple sources"):
        build_pairwise_30k_protocol(
            candidates,
            split_quotas=SMALL_QUOTAS,
            unit_assignments=assignments,
        )

    result = build_pairwise_30k_protocol(
        candidates,
        split_quotas=SMALL_QUOTAS,
        source_quotas=source_quotas,
        unit_assignments=assignments,
    )
    counts = Counter(
        (row.candidate.source_id, row.split, row.candidate.label) for row in result.rows
    )
    for source, split_counts in source_counts.items():
        for split, expected in split_counts.items():
            assert counts[(source, split, True)] == expected
            assert counts[(source, split, False)] == expected


def test_stable_hash_assignment_is_seeded_and_reproducible():
    observed = stable_split_for_unit("mm/component/42", seed="seed-a")
    assert observed == stable_split_for_unit("mm/component/42", seed="seed-a")
    assert observed in {"train", "val", "test"}
