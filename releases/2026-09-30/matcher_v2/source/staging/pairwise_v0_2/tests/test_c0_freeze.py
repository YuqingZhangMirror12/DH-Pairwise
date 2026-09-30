from __future__ import annotations

import json
from pathlib import Path

import pytest

from staging.pairwise_v0_2.pairwise_data import historical_identity as module
from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    HISTORICAL_TEST_ACCESS_EVIDENCE,
    HistoricalIdentityError,
    HistoricalIdentityIndex,
    assert_portable_receipt,
    freeze_c0_n_q1,
)
from staging.pairwise_v0_2.pairwise_data.sampling import (
    validation_stream_fingerprint,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)


def _fixture(tmp_path: Path):
    cache_groups = {"mm": [], "eccv": []}
    split_groups = {}
    split_components = {}
    group_components = {}
    identities = []

    def add_group(dataset: str, split: str, index: int, hashes: list[str]):
        if dataset == "mm":
            source = "[S.{}{}]".format(index, "v" if split == "val" else "t")
            base = "root/{}/001".format(source)
            pair_group = "mm/base/{}".format(base)
            component = "mm/source/{}".format(source)
            members = ["{}/0/{}.png".format(base, n + 1) for n in range(len(hashes))]
            group = {
                "profile": "mm",
                "canonical_group_id": base + "/0",
                "base_group_id": base,
                "source_id": source,
                "group_id": "001",
                "group_signature": "1" * 64,
            }
            binding = MM_CANONICAL_BINDING
            fragment_ids = [member[:-4] for member in members]
        else:
            group_number = "{}{}".format(index, "9" if split == "val" else "8")
            pair_group = "eccv/2x2/{}".format(group_number)
            signature = "{:064x}".format(index + (1 if split == "val" else 20))
            component = "eccv/signature/{}".format(signature)
            prefix = "archive/siamese/2x2/{}".format(group_number)
            members = ["{}/{}.png".format(prefix, n + 1) for n in range(len(hashes))]
            group = {
                "profile": "siamese/2x2",
                "canonical_group_id": "siamese/2x2/{}".format(group_number),
                "base_group_id": None,
                "source_id": None,
                "group_id": group_number,
                "group_signature": signature,
            }
            binding = ECCV_CANONICAL_BINDING
            fragment_ids = [
                "2x2/{}/{}.png".format(group_number, n + 1) for n in range(len(hashes))
            ]
        group["fragments"] = [
            {
                "canonical_fragment_id": str(n + 1),
                "member_path": member,
                "content_sha256": content,
                "byte_count": 3,
            }
            for n, (member, content) in enumerate(zip(members, hashes))
        ]
        cache_groups[dataset].append(group)
        split_groups[pair_group] = split
        split_components[component] = split
        group_components[pair_group] = component
        identities.append(
            (dataset, split, pair_group, component, binding, members, fragment_ids)
        )

    # One val and two train groups per dataset.  The first train group reuses a
    # val endpoint hash and is quarantined; the second remains selectable.
    for dataset_index, dataset in enumerate(("mm", "eccv")):
        prefix = 10 + dataset_index * 20
        val_hashes = [
            "{:064x}".format(prefix),
            "{:064x}".format(prefix + 1),
            "{:064x}".format(prefix + 2),
        ]
        add_group(dataset, "val", 1, val_hashes)
        # The reused hash belongs to a val-assigned fragment that is not an
        # endpoint in any usable validation pair.  It must still quarantine.
        add_group(
            dataset,
            "train",
            2,
            [val_hashes[2], "{:064x}".format(prefix + 5)],
        )
        add_group(
            dataset,
            "train",
            3,
            ["{:064x}".format(prefix + 3), "{:064x}".format(prefix + 4)],
        )

    def cache(binding: ArchiveBinding, groups):
        return {
            "cache_schema_version": "pairwise-group-fingerprint-cache/0.1",
            "archive": {
                "sha256": binding.sha256,
                "expected_sha256": binding.sha256,
            },
            "fingerprints": {
                "schema_version": "pairwise-group-fingerprint/0.1",
                "groups": groups,
            },
        }

    split = {
        "schema_version": "pairwise-real-data-gate/0.1",
        "status": "provisional",
        "authorization": "fixture",
        "candidate_id": "70_15_15__pairwise-v0.1-expanded-056",
        "seed": "pairwise-v0.1-expanded-056",
        "pair_rows_materialized": False,
        "excluded_groups": [],
        "assignments": {
            "groups": split_groups,
            "components": split_components,
            "group_components": group_components,
        },
    }
    for name, payload in (
        ("mm.json", cache(MM_CANONICAL_BINDING, cache_groups["mm"])),
        ("eccv.json", cache(ECCV_CANONICAL_BINDING, cache_groups["eccv"])),
        ("split.json", split),
    ):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    index = HistoricalIdentityIndex.from_files(
        mm_cache_path=tmp_path / "mm.json",
        eccv_cache_path=tmp_path / "eccv.json",
        split_path=tmp_path / "split.json",
    )

    records = {"train": [], "val": []}
    for (
        dataset,
        split_name,
        group,
        component,
        binding,
        members,
        fragment_ids,
    ) in identities:
        dataset_id = "mm_augmented" if dataset == "mm" else "eccv_1113data"

        def ref(position: int):
            return MaskMemberRef(
                binding=binding,
                archive_member=members[position],
                fragment_id=fragment_ids[position],
                dataset_id=dataset_id,
                canonical_group_id=group,
                component_id=component,
                split=split_name,
                threshold_rule="binary_brighter_value",
            )

        for label in (False, True):
            record = TrainingPairRecord(
                fragment_a=ref(0),
                fragment_b=ref(1),
                label=label,
                direction_b_wrt_a="right" if label else None,
                dataset_id=dataset_id,
                canonical_group_id=group,
                component_id=component,
                split=split_name,
                canonical_pair_key=tuple(sorted((fragment_ids[0], fragment_ids[1]))),
                label_origin="fixture",
            )
            records[split_name].append(record)
    return index, records


def test_freeze_quarantines_content_overlap_and_is_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, records = _fixture(tmp_path)
    val_fingerprint = validation_stream_fingerprint(records["val"])["sha256"]
    mm_archive = tmp_path / "mm.zip"
    eccv_archive = tmp_path / "eccv.tar"
    mm_archive.write_bytes(b"fixture-mm")
    eccv_archive.write_bytes(b"fixture-eccv")

    def fake_sha(path: Path) -> str:
        return (
            MM_CANONICAL_BINDING.sha256
            if Path(path) == mm_archive
            else ECCV_CANONICAL_BINDING.sha256
        )

    monkeypatch.setattr(module, "sha256_file", fake_sha)
    expected = {
        ("mm_augmented", False): 1,
        ("mm_augmented", True): 1,
        ("eccv_1113data", False): 1,
        ("eccv_1113data", True): 1,
    }
    result = freeze_c0_n_q1(
        identity_index=index,
        validation_records=records["val"],
        training_records=records["train"],
        archive_paths={
            "mm_augmented": mm_archive,
            "eccv_1113data": eccv_archive,
        },
        target_per_dataset_label=1,
        max_per_component_label=1,
        expected_validation_fingerprint=val_fingerprint,
        expected_validation_counts=expected,
    )
    assert len(result.selected_train_records) == 4
    assert result.receipt["scope"]["historical_test_access"] == dict(
        HISTORICAL_TEST_ACCESS_EVIDENCE
    )
    assert result.receipt["scope"]["pair_stream_splits_read"] == ["train", "val"]
    assert "historical_test_read" not in result.receipt["scope"]
    quarantine = result.receipt["training"]["content_quarantine"]
    assert quarantine["raw_row_count"] == 4
    assert quarantine["unique_physical_pair_count"] == 2
    assert quarantine["prefilter_or_deduplication_before_quarantine"] is False
    assert result.receipt["validation"]["identity_universe_fragment_count"] == 6
    assert (
        result.receipt["validation"][
            "pair_stream_endpoint_unique_physical_member_count"
        ]
        == 4
    )
    assert result.receipt["selected_train_vs_full_validation_overlap"] == {
        "component": 0,
        "member": 0,
        "content_sha256": 0,
    }
    assert_portable_receipt(result.receipt)
    text = json.dumps(result.receipt)
    assert str(tmp_path) not in text
    assert "archive/siamese" not in text


def test_changed_validation_order_fingerprint_fails_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, records = _fixture(tmp_path)
    mm_archive = tmp_path / "mm.zip"
    eccv_archive = tmp_path / "eccv.tar"
    mm_archive.write_bytes(b"m")
    eccv_archive.write_bytes(b"e")
    monkeypatch.setattr(
        module,
        "sha256_file",
        lambda path: (
            MM_CANONICAL_BINDING.sha256
            if Path(path) == mm_archive
            else ECCV_CANONICAL_BINDING.sha256
        ),
    )
    expected = {
        ("mm_augmented", False): 1,
        ("mm_augmented", True): 1,
        ("eccv_1113data", False): 1,
        ("eccv_1113data", True): 1,
    }
    with pytest.raises(HistoricalIdentityError, match="fingerprint changed"):
        freeze_c0_n_q1(
            identity_index=index,
            validation_records=reversed(records["val"]),
            training_records=records["train"],
            archive_paths={
                "mm_augmented": mm_archive,
                "eccv_1113data": eccv_archive,
            },
            target_per_dataset_label=1,
            expected_validation_fingerprint="0" * 64,
            expected_validation_counts=expected,
        )
