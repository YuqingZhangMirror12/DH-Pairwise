from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    HistoricalIdentityError,
    HistoricalIdentityIndex,
    historical_identity_index_content_sha256,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MM_CANONICAL_BINDING,
    MaskMemberRef,
    TrainingPairRecord,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _write_fixture(
    tmp_path: Path,
) -> tuple[HistoricalIdentityIndex, TrainingPairRecord]:
    mm_group = {
        "profile": "mm",
        "canonical_group_id": "root/[S.1]/001/0",
        "base_group_id": "root/[S.1]/001",
        "source_id": "[S.1]",
        "group_id": "001",
        "group_signature": "c" * 64,
        "fragments": [
            {
                "canonical_fragment_id": "1",
                "member_path": "root/[S.1]/001/0/1.png",
                "content_sha256": SHA_A,
                "byte_count": 7,
            },
            {
                "canonical_fragment_id": "2",
                "member_path": "root/[S.1]/001/0/2.png",
                "content_sha256": SHA_B,
                "byte_count": 9,
            },
        ],
    }
    mm_cache = {
        "cache_schema_version": "pairwise-group-fingerprint-cache/0.1",
        "archive": {
            "sha256": MM_CANONICAL_BINDING.sha256,
            "expected_sha256": MM_CANONICAL_BINDING.sha256,
        },
        "fingerprints": {
            "schema_version": "pairwise-group-fingerprint/0.1",
            "groups": [mm_group],
        },
    }
    eccv_cache = {
        "cache_schema_version": "pairwise-group-fingerprint-cache/0.1",
        "archive": {
            "sha256": (
                "98042e1b2068500f803be817a17e09470aefa34fb432047bebc50bebc06d9e72"
            ),
            "expected_sha256": (
                "98042e1b2068500f803be817a17e09470aefa34fb432047bebc50bebc06d9e72"
            ),
        },
        "fingerprints": {
            "schema_version": "pairwise-group-fingerprint/0.1",
            "groups": [],
        },
    }
    group = "mm/base/root/[S.1]/001"
    component = "mm/source/[S.1]"
    split = {
        "schema_version": "pairwise-real-data-gate/0.1",
        "status": "provisional",
        "authorization": "fixture",
        "candidate_id": "70_15_15__pairwise-v0.1-expanded-056",
        "seed": "pairwise-v0.1-expanded-056",
        "pair_rows_materialized": False,
        "excluded_groups": [],
        "assignments": {
            "groups": {group: "train"},
            "components": {component: "train"},
            "group_components": {group: component},
        },
    }
    paths = []
    for name, payload in (
        ("mm.json", mm_cache),
        ("eccv.json", eccv_cache),
        ("split.json", split),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    index = HistoricalIdentityIndex.from_files(
        mm_cache_path=paths[0], eccv_cache_path=paths[1], split_path=paths[2]
    )

    def ref(number: str) -> MaskMemberRef:
        return MaskMemberRef(
            binding=MM_CANONICAL_BINDING,
            archive_member="root/[S.1]/001/0/{}.png".format(number),
            fragment_id="root/[S.1]/001/0/{}".format(number),
            dataset_id="mm_augmented",
            canonical_group_id=group,
            component_id=component,
            split="train",
            threshold_rule="binary_brighter_value",
        )

    record = TrainingPairRecord(
        fragment_a=ref("1"),
        fragment_b=ref("2"),
        label=True,
        direction_b_wrt_a="right",
        dataset_id="mm_augmented",
        canonical_group_id=group,
        component_id=component,
        split="train",
        canonical_pair_key=(
            "root/[S.1]/001/0/1",
            "root/[S.1]/001/0/2",
        ),
        label_origin="fixture",
    )
    return index, record


def test_cache_and_split_independently_enrich_content_hashes(tmp_path: Path) -> None:
    index, record = _write_fixture(tmp_path)
    verified = index.verify_record(record)
    assert verified.record.fragment_a.content_sha256 == SHA_A
    assert verified.record.fragment_b.content_sha256 == SHA_B
    assert record.fragment_a.content_sha256 is None


def test_immutable_payload_constructor_matches_file_constructor(tmp_path: Path) -> None:
    from_files, record = _write_fixture(tmp_path)
    from_payloads = HistoricalIdentityIndex.from_payloads(
        mm_cache_payload=(tmp_path / "mm.json").read_bytes(),
        eccv_cache_payload=(tmp_path / "eccv.json").read_bytes(),
        split_payload=(tmp_path / "split.json").read_bytes(),
    )

    assert from_payloads.content_sha256 == from_files.content_sha256
    assert dict(from_payloads.artifact_locks) == dict(from_files.artifact_locks)
    assert from_payloads.verify_record(record) == from_files.verify_record(record)

    with pytest.raises(HistoricalIdentityError, match="immutable bytes"):
        HistoricalIdentityIndex.from_payloads(
            mm_cache_payload=bytearray((tmp_path / "mm.json").read_bytes()),
            eccv_cache_payload=(tmp_path / "eccv.json").read_bytes(),
            split_payload=(tmp_path / "split.json").read_bytes(),
        )

    class EvilBytes(bytes):
        def decode(self, *_args, **_kwargs):
            return "{}"

    with pytest.raises(HistoricalIdentityError, match="immutable bytes"):
        HistoricalIdentityIndex.from_payloads(
            mm_cache_payload=EvilBytes((tmp_path / "mm.json").read_bytes()),
            eccv_cache_payload=(tmp_path / "eccv.json").read_bytes(),
            split_payload=(tmp_path / "split.json").read_bytes(),
        )


def test_from_files_excludes_historical_test_groups_from_runtime_identity_digest(
    tmp_path: Path,
) -> None:
    baseline, _record = _write_fixture(tmp_path)
    mm_path = tmp_path / "mm.json"
    split_path = tmp_path / "split.json"
    mm_payload = json.loads(mm_path.read_text(encoding="utf-8"))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))

    test_group = copy.deepcopy(mm_payload["fingerprints"]["groups"][0])
    test_group.update(
        {
            "canonical_group_id": "root/[S.9]/999/0",
            "base_group_id": "root/[S.9]/999",
            "source_id": "[S.9]",
            "group_id": "999",
            "group_signature": "d" * 64,
        }
    )
    for index, fragment in enumerate(test_group["fragments"], start=1):
        fragment["canonical_fragment_id"] = str(index)
        fragment["member_path"] = "root/[S.9]/999/0/{}.png".format(index)
        fragment["content_sha256"] = "{:064x}".format(100 + index)
    mm_payload["fingerprints"]["groups"].append(test_group)

    group_id = "mm/base/root/[S.9]/999"
    component_id = "mm/source/[S.9]"
    assignments = split_payload["assignments"]
    assignments["groups"][group_id] = "test"
    assignments["components"][component_id] = "test"
    assignments["group_components"][group_id] = component_id
    mm_path.write_text(json.dumps(mm_payload), encoding="utf-8")
    split_path.write_text(json.dumps(split_payload), encoding="utf-8")

    rebuilt = HistoricalIdentityIndex.from_files(
        mm_cache_path=mm_path,
        eccv_cache_path=tmp_path / "eccv.json",
        split_path=split_path,
    )
    assert rebuilt.identity_count == baseline.identity_count == 2
    assert rebuilt.content_sha256 == baseline.content_sha256
    assert all(
        identity.split in {"train", "val"} for identity in rebuilt._by_member.values()
    )
    assert all("[S.9]" not in key[1] for key in rebuilt._by_member)


def test_identity_mapping_commitment_is_order_independent_and_recomputed(
    tmp_path: Path,
) -> None:
    index, _record = _write_fixture(tmp_path)
    reversed_mapping = dict(reversed(tuple(index._by_member.items())))
    rebuilt = HistoricalIdentityIndex(
        by_member=reversed_mapping,
        split_candidate_id=index.split_candidate_id,
        split_seed=index.split_seed,
        artifact_locks=index.artifact_locks,
    )

    assert rebuilt.identity_count == index.identity_count == 2
    assert rebuilt.content_sha256 == index.content_sha256
    assert historical_identity_index_content_sha256(rebuilt) == rebuilt.content_sha256


def test_identity_mapping_recompute_rejects_rekeyed_primitive_row(
    tmp_path: Path,
) -> None:
    index, _record = _write_fixture(tmp_path)
    rows = dict(index._by_member._rows)
    key = sorted(rows)[0]
    row = rows.pop(key)
    rekeyed_member = "root/[S.1]/001/0/1-rekeyed.png"
    rekeyed_key = (key[0], rekeyed_member)
    rows[rekeyed_key] = row
    assert sorted(rows).index(rekeyed_key) == 0
    object.__setattr__(index._by_member, "_rows", MappingProxyType(rows))

    with pytest.raises(HistoricalIdentityError, match="key disagrees"):
        historical_identity_index_content_sha256(index)


def test_identity_index_deep_copies_inputs_and_exposes_read_only_state(
    tmp_path: Path,
) -> None:
    index, _record = _write_fixture(tmp_path)
    caller_mapping = dict(index._by_member)
    caller_locks = {role: dict(lock) for role, lock in index.artifact_locks.items()}
    rebuilt = HistoricalIdentityIndex(
        by_member=caller_mapping,
        split_candidate_id=index.split_candidate_id,
        split_seed=index.split_seed,
        artifact_locks=caller_locks,
    )
    expected = rebuilt.content_sha256
    key = sorted(caller_mapping)[0]
    caller_mapping[key] = replace(caller_mapping[key], split="val")
    caller_locks["historical_split"]["sha256"] = "f" * 64

    assert rebuilt.content_sha256 == expected
    assert historical_identity_index_content_sha256(rebuilt) == expected
    assert rebuilt._by_member[key].split == "train"
    assert rebuilt.artifact_locks["historical_split"]["sha256"] != "f" * 64
    with pytest.raises(TypeError):
        rebuilt._by_member[key] = caller_mapping[key]
    with pytest.raises(TypeError):
        rebuilt.artifact_locks["historical_split"]["sha256"] = "e" * 64
    with pytest.raises(AttributeError):
        rebuilt.split_seed = "mutated"
    with pytest.raises(AttributeError):
        rebuilt.content_sha256 = "0" * 64
    with pytest.raises(AttributeError):
        rebuilt._by_member = caller_mapping

    # Returned values are fresh views over immutable primitive rows.
    exposed = rebuilt._by_member[key]
    object.__setattr__(exposed, "split", "val")
    assert rebuilt._by_member[key].split == "train"
    assert historical_identity_index_content_sha256(rebuilt) == expected


def test_identity_mapping_commitment_binds_key_member_and_population(
    tmp_path: Path,
) -> None:
    index, _record = _write_fixture(tmp_path)
    original = dict(index._by_member)
    key = sorted(original)[0]

    removed = dict(original)
    removed.pop(key)
    removed_index = HistoricalIdentityIndex(
        by_member=removed,
        split_candidate_id=index.split_candidate_id,
        split_seed=index.split_seed,
        artifact_locks=index.artifact_locks,
    )
    assert removed_index.content_sha256 != index.content_sha256

    moved = dict(original)
    identity = moved.pop(key)
    moved_member = "root/[S.1]/001/0/moved.png"
    moved[(identity.dataset_id, moved_member)] = replace(
        identity,
        archive_member=moved_member,
        fragment_id="root/[S.1]/001/0/moved",
    )
    moved_index = HistoricalIdentityIndex(
        by_member=moved,
        split_candidate_id=index.split_candidate_id,
        split_seed=index.split_seed,
        artifact_locks=index.artifact_locks,
    )
    assert moved_index.content_sha256 != index.content_sha256


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("fragment_id", "root/[S.1]/001/0/forged"),
        ("pair_group_id", "mm/base/forged"),
        ("component_id", "mm/source/forged"),
        ("split", "val"),
        ("content_sha256", "c" * 64),
    ),
)
def test_identity_mapping_commitment_binds_private_row_fields(
    tmp_path: Path, field: str, value: str
) -> None:
    index, _record = _write_fixture(tmp_path)
    by_member = dict(index._by_member)
    key = sorted(by_member)[0]
    by_member[key] = replace(by_member[key], **{field: value})
    changed = HistoricalIdentityIndex(
        by_member=by_member,
        split_candidate_id=index.split_candidate_id,
        split_seed=index.split_seed,
        artifact_locks=index.artifact_locks,
    )
    assert changed.content_sha256 != index.content_sha256


def test_identity_mapping_rejects_noncanonical_key_hash_and_member_schema(
    tmp_path: Path,
) -> None:
    index, _record = _write_fixture(tmp_path)
    original = dict(index._by_member)
    key = sorted(original)[0]

    invalid_values = (
        {("mm_augmented", "wrong.png"): original[key]},
        {**original, key: replace(original[key], content_sha256="A" * 64)},
        {**original, key: replace(original[key], archive_member="../escape.png")},
    )
    for by_member in invalid_values:
        with pytest.raises(HistoricalIdentityError):
            HistoricalIdentityIndex(
                by_member=by_member,
                split_candidate_id=index.split_candidate_id,
                split_seed=index.split_seed,
                artifact_locks=index.artifact_locks,
            )


@pytest.mark.parametrize("field", ["member", "fragment", "component", "split"])
def test_identity_disagreement_fails_closed(tmp_path: Path, field: str) -> None:
    index, record = _write_fixture(tmp_path)
    fragment = record.fragment_a
    if field == "member":
        fragment = replace(fragment, archive_member="root/[S.1]/001/0/9.png")
    elif field == "fragment":
        fragment = replace(fragment, fragment_id="root/[S.1]/001/0/9")
    elif field == "component":
        fragment = replace(fragment, component_id="mm/source/[S.9]")
    else:
        fragment = replace(fragment, split="val")
    changes = {"fragment_a": fragment}
    if field == "fragment":
        changes["canonical_pair_key"] = tuple(
            sorted((fragment.fragment_id, record.fragment_b.fragment_id))
        )
    elif field == "component":
        changes.update(
            {
                "fragment_b": replace(
                    record.fragment_b, component_id="mm/source/[S.9]"
                ),
                "component_id": "mm/source/[S.9]",
            }
        )
    elif field == "split":
        changes.update(
            {"fragment_b": replace(record.fragment_b, split="val"), "split": "val"}
        )
    tampered = replace(record, **changes)
    with pytest.raises(HistoricalIdentityError):
        index.verify_record(tampered)


def test_cache_archive_binding_substitution_fails(tmp_path: Path) -> None:
    index, _record = _write_fixture(tmp_path)
    assert index.artifact_locks["mm_fingerprint_cache"]["sha256"]
    payload_path = tmp_path / "mm.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["archive"]["sha256"] = "0" * 64
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HistoricalIdentityError, match="archive SHA"):
        HistoricalIdentityIndex.from_files(
            mm_cache_path=payload_path,
            eccv_cache_path=tmp_path / "eccv.json",
            split_path=tmp_path / "split.json",
        )
