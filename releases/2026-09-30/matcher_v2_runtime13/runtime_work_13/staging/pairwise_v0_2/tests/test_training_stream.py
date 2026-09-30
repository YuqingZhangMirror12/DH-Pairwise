import json
from pathlib import Path

import pytest

from staging.pairwise_v0_1.baselines.real_pair_stream import (
    ArchiveImageRef,
    ECCVSplitExposureAudit,
    RealPairRecord,
    SplitManifestView,
)
from staging.pairwise_v0_1.pairwise_data.schema import DirectedRelation
from staging.pairwise_v0_2.pairwise_data import training_stream
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingDataError,
    TrainingPairRecord,
    direction_to_geometry_relation,
    iter_historical_pair_records,
    iter_synthetic_pair_records,
    iter_synthetic_pair_records_payload,
    iter_training_pair_records,
)


def _synthetic_group(group_id="dataset/family/no_erode/1", status="retained"):
    archive = {
        "logical_id": "local_asset://fixture_masks",
        "format": "zip",
        "sha256": "a" * 64,
    }
    members = [
        {
            "archive_member": "output/voronoi_masks/family/no_erode/1/{}.png".format(i),
            "content_sha256": str(i + 1) * 64,
            "fragment_id": i,
            "threshold_rule": "grayscale_uint8_gt_127",
        }
        for i in range(3)
    ]
    measurements = [
        {
            "fragment_a": 0,
            "fragment_b": 1,
            "is_neighbor": True,
            "legacy_dilated_overlap_pixels": 50,
            "raw_overlap_pixels": 0,
        },
        {
            "fragment_a": 0,
            "fragment_b": 2,
            "is_neighbor": False,
            "legacy_dilated_overlap_pixels": 12,
            "raw_overlap_pixels": 0,
        },
        {
            "fragment_a": 1,
            "fragment_b": 2,
            "is_neighbor": True,
            "legacy_dilated_overlap_pixels": 40,
            "raw_overlap_pixels": 0,
        },
    ]
    return {
        "schema_version": "dunhuang-pairwise-synthetic-group-manifest/0.2",
        "archive": archive,
        "dataset_id": "fixture_synthetic",
        "group_id": group_id,
        "source_group_id": "1",
        "generator_family": "family",
        "erosion_profile": "no_erode",
        "no_erode": True,
        "fragment_count": 3,
        "members": members,
        "pair_measurements": measurements,
        "quarantine": {"status": status, "reasons": []},
    }


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_synthetic_stream_derives_all_pairs_and_skips_quarantine(tmp_path):
    manifest = tmp_path / "groups.jsonl"
    quarantined = _synthetic_group("dataset/family/no_erode/2", "quarantined")
    _write_jsonl(manifest, [_synthetic_group(), quarantined])

    records = list(iter_synthetic_pair_records(manifest))

    assert len(records) == 3
    assert [record.label for record in records] == [True, False, True]
    negative = records[1]
    assert type(negative.label) is bool
    assert negative.static_hard_negative_score == pytest.approx(12.0 / 30.0)
    assert negative.split == "train"
    assert negative.direction_b_wrt_a is None
    assert negative.provenance["source_lineage_status"] == "unavailable_training_only"
    assert all(record.fragment_a.binding.sha256 == "a" * 64 for record in records)


def test_synthetic_payload_stream_matches_file_stream_and_rejects_mutable_bytes(
    tmp_path,
):
    manifest = tmp_path / "groups.jsonl"
    _write_jsonl(manifest, [_synthetic_group()])

    expected = list(iter_synthetic_pair_records(manifest))
    observed = list(iter_synthetic_pair_records_payload(manifest.read_bytes()))
    assert observed == expected

    with pytest.raises(TrainingDataError, match="immutable bytes"):
        list(iter_synthetic_pair_records_payload(bytearray(manifest.read_bytes())))

    class EvilBytes(bytes):
        def decode(self, *_args, **_kwargs):
            return ""

    with pytest.raises(TrainingDataError, match="immutable bytes"):
        list(iter_synthetic_pair_records_payload(EvilBytes(manifest.read_bytes())))


def test_synthetic_stream_rejects_non_bool_labels(tmp_path):
    manifest = tmp_path / "groups.jsonl"
    group = _synthetic_group()
    group["pair_measurements"][0]["is_neighbor"] = 1
    _write_jsonl(manifest, [group])

    with pytest.raises(TrainingDataError, match="explicit bool"):
        list(iter_synthetic_pair_records(manifest))


def test_synthetic_data_is_never_exposed_by_val_composer(tmp_path):
    manifest = tmp_path / "groups.jsonl"
    _write_jsonl(manifest, [_synthetic_group()])

    assert (
        list(
            iter_training_pair_records(
                split="val",
                split_manifest={},
                synthetic_manifest=manifest,
            )
        )
        == []
    )


def test_training_record_requires_exact_bool():
    binding = ArchiveBinding("local_asset://fixture", "zip", "b" * 64)
    common = dict(
        binding=binding,
        dataset_id="fixture",
        canonical_group_id="fixture/group/1",
        component_id="fixture/component/1",
        split="train",
        threshold_rule="grayscale_uint8_gt_127",
        content_sha256="c" * 64,
    )
    first = MaskMemberRef(
        archive_member="masks/0.png", fragment_id="fragment/0", **common
    )
    second = MaskMemberRef(
        archive_member="masks/1.png", fragment_id="fragment/1", **common
    )
    with pytest.raises(TypeError, match="explicit built-in bool"):
        TrainingPairRecord(
            fragment_a=first,
            fragment_b=second,
            label=1,
            direction_b_wrt_a=None,
            dataset_id="fixture",
            canonical_group_id="fixture/group/1",
            component_id="fixture/component/1",
            split="train",
            canonical_pair_key=("fragment/0", "fragment/1"),
            label_origin="fixture",
        )


def test_historical_adapter_preserves_frozen_component_and_direction(monkeypatch):
    fragment_common = dict(
        dataset_id="mm_augmented",
        archive_format="zip",
        canonical_group_id="mm/base/root/source/group",
        cluster_id="mm/source/source",
        split="val",
        profile="mm-augmented",
        source_id="source",
        variant_id="0",
    )
    first = ArchiveImageRef(
        archive_member="root/source/group/0/0.png",
        fragment_id="root/source/group/0/0",
        **fragment_common,
    )
    second = ArchiveImageRef(
        archive_member="root/source/group/0/1.png",
        fragment_id="root/source/group/0/1",
        **fragment_common,
    )
    historical = RealPairRecord(
        fragment_a=first,
        fragment_b=second,
        is_adjacent=True,
        direction_b_wrt_a=DirectedRelation.RIGHT,
        dataset_id="mm_augmented",
        canonical_group_id="mm/base/root/source/group",
        cluster_id="mm/source/source",
        split="val",
        profile="mm-augmented",
        source_id="source",
        variant_id="0",
        canonical_pair_key=tuple(sorted((first.fragment_id, second.fragment_id))),
        source_member="root/source/group/pair.csv",
        source_row_number=2,
        condition_raw="left-right",
        manifest_candidate_id="fixture-candidate",
        manifest_status="complete",
        manifest_authorization="fixture",
    )
    monkeypatch.setattr(
        training_stream,
        "iter_mm_pair_records",
        lambda *args, **kwargs: iter((historical,)),
    )
    manifest = SplitManifestView(
        candidate_id="fixture-candidate",
        status="complete",
        authorization="fixture",
        group_assignments={"mm/base/root/source/group": "val"},
        group_components={"mm/base/root/source/group": "mm/source/source"},
    )

    records = list(
        iter_historical_pair_records(
            split_manifest=manifest,
            split="val",
            mm_archive=object(),
        )
    )

    assert len(records) == 1
    assert records[0].label is True
    assert records[0].direction_b_wrt_a == "right"
    assert (
        direction_to_geometry_relation(records[0].direction_b_wrt_a) == "b_right_of_a"
    )
    assert records[0].component_id == "mm/source/source"
    assert records[0].fragment_a.threshold_rule == "binary_brighter_value"
    assert records[0].provenance["real_dunhuang_sealed_test"] is False


def test_historical_stream_propagates_eccv_exposure_audit(monkeypatch):
    observed = {}

    def fake_eccv_stream(*args, **kwargs):
        observed.update(kwargs)
        return iter(())

    monkeypatch.setattr(training_stream, "iter_eccv_pair_records", fake_eccv_stream)
    audit = ECCVSplitExposureAudit()
    manifest = SplitManifestView(
        candidate_id="fixture-candidate",
        status="complete",
        authorization="fixture",
        group_assignments={"eccv/2x2/1": "val"},
        group_components={"eccv/2x2/1": "eccv/signature/fixture"},
    )

    assert (
        list(
            iter_historical_pair_records(
                split_manifest=manifest,
                split="val",
                eccv_archive=object(),
                eccv_exposure_audit=audit,
            )
        )
        == []
    )
    assert observed["splits"] == "val"
    assert observed["exposure_audit"] is audit


def test_historical_stream_rejects_eccv_audit_without_archive() -> None:
    with pytest.raises(TrainingDataError, match="requires an ECCV archive"):
        list(
            iter_historical_pair_records(
                split_manifest={},
                split="val",
                mm_archive=object(),
                eccv_exposure_audit=ECCVSplitExposureAudit(),
            )
        )


@pytest.mark.parametrize(
    "source, expected",
    [
        ("left", "b_left_of_a"),
        ("right", "b_right_of_a"),
        ("above", "b_above_a"),
        ("below", "b_below_a"),
        (None, None),
    ],
)
def test_direction_mapping_is_explicit_b_wrt_a(source, expected):
    assert direction_to_geometry_relation(source) == expected


def test_direction_mapping_fails_closed():
    with pytest.raises(TrainingDataError, match="unsupported"):
        direction_to_geometry_relation("clockwise")
