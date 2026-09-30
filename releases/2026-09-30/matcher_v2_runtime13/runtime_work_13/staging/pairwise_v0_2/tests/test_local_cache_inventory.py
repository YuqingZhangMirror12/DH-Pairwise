import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from staging.pairwise_v0_2.geometry import CandidateBuilderConfig
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.geometry_cache import (
    FrozenReceiptTrust,
    GeometryArtifactCache,
)
from staging.pairwise_v0_2.training.local_cache_inventory import (
    LocalCacheInventoryConfig,
    LocalCacheInventoryError,
    build_local_cache_inventory,
    reopen_and_verify_local_cache_inventory,
    write_portable_inventory,
)


def _mask(*, left, top=12, notch=False):
    value = np.zeros((96, 112), dtype=bool)
    value[top : top + 66, left : left + 38] = True
    if notch:
        value[top + 21 : top + 41, left + 27 : left + 38] = False
    return value


def _geometry():
    return CandidateBuilderConfig(
        window_scale_fractions=(0.16,),
        window_min_px=2.0,
        window_max_px=64.0,
        output_size=(10, 12),
        min_run_length_fraction=0.0,
        min_run_length_px=2.0,
        side_resample_count=20,
    )


def _reference(name, *, split="train", dataset="fixture_mm", content_digit="1"):
    binding = ArchiveBinding(
        logical_id="fixture://local-cache/masks",
        archive_format="zip",
        sha256="a" * 64,
    )
    return MaskMemberRef(
        binding=binding,
        archive_member="masks/{}.png".format(name),
        fragment_id="fragment/{}".format(name),
        dataset_id=dataset,
        canonical_group_id="group/{}".format(split),
        component_id="component/{}".format(split),
        split=split,
        threshold_rule="grayscale_uint8_gt_127",
        content_sha256=content_digit * 64,
    )


def _record(first, second, *, label, direction, split="train", dataset="fixture_mm"):
    fragment_a = _reference(
        first,
        split=split,
        dataset=dataset,
        content_digit={"a": "1", "b": "2", "c": "3"}[first],
    )
    fragment_b = _reference(
        second,
        split=split,
        dataset=dataset,
        content_digit={"a": "1", "b": "2", "c": "3"}[second],
    )
    return TrainingPairRecord(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        label=label,
        direction_b_wrt_a=direction,
        dataset_id=dataset,
        canonical_group_id="group/{}".format(split),
        component_id="component/{}".format(split),
        split=split,
        canonical_pair_key=tuple(
            sorted((fragment_a.fragment_id, fragment_b.fragment_id))
        ),
        label_origin="fixture",
        provenance={"sealed_real": False},
    )


class _Loader:
    def __init__(self, masks):
        self.masks = masks
        self.requests = []

    def __call__(self, reference):
        self.requests.append(reference.fragment_id)
        return self.masks[reference.fragment_id]


def _records():
    return (
        _record("a", "b", label=True, direction="right"),
        _record("b", "c", label=False, direction=None),
    )


def _masks():
    return {
        "fragment/a": _mask(left=9, notch=True),
        "fragment/b": _mask(left=34),
        "fragment/c": _mask(left=61, top=17, notch=True),
    }


def _config(**kwargs):
    values = {
        "geometry": _geometry(),
        "max_records": 8,
        "max_unique_references": 8,
        "max_mask_pixels": 96 * 112,
    }
    values.update(kwargs)
    return LocalCacheInventoryConfig(**values)


def _canonical_sha(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_inventory_build_is_label_blind_bounded_and_warm_stable(tmp_path):
    records = _records()
    loader = _Loader(_masks())
    cache = GeometryArtifactCache(tmp_path / "cache")

    cold = build_local_cache_inventory(records, loader, cache, _config())
    operational = cold.receipt["operational_counts"]
    semantic = cold.receipt["semantic_inventory"]

    assert operational == {
        "record_count": 2,
        "endpoint_occurrence_count": 4,
        "unique_physical_reference_count": 3,
        "unique_canonical_fragment_count": 3,
        "canonical_duplicate_reference_count": 0,
        "mask_load_count": 3,
        "cache_hit_count": 0,
        "cache_miss_count": 3,
    }
    assert cold.cache_receipt["artifact_count"] == 3
    assert semantic["pair_status_counts"] == {"ok": 2}
    assert semantic["all_four_direction_pair_count"] == 2
    assert semantic["train_val_canonical_fragment_overlap_count"] == 0
    assert cold.receipt["generation_contract"]["supervision_fields_read"] == []
    assert (
        cold.receipt["generation_contract"]["candidate_generation_direction_argument"]
        is None
    )

    # Change all supervision while retaining the exact geometry population.
    changed_records = (
        _record("a", "b", label=False, direction=None),
        _record("b", "c", label=True, direction="above"),
    )
    warm = build_local_cache_inventory(
        changed_records, _Loader(_masks()), cache, _config()
    )
    assert warm.semantic_commitment_sha256 == cold.semantic_commitment_sha256
    assert warm.receipt["operational_counts"]["cache_miss_count"] == 0
    assert warm.receipt["operational_counts"]["cache_hit_count"] == 3

    encoded = warm.portable_json()
    for forbidden in (
        "masks/a.png",
        "fragment/a",
        "pair_id",
        "archive_member",
        '"label"',
        "/tmp/",
    ):
        assert forbidden not in encoded


def test_frozen_reopen_binds_external_receipt_and_semantic_commitment(tmp_path):
    records = _records()
    masks = _masks()
    root = tmp_path / "cache"
    built = build_local_cache_inventory(
        records, _Loader(masks), GeometryArtifactCache(root), _config()
    )
    trust = FrozenReceiptTrust(
        expected_content_sha256=built.cache_receipt["content_sha256"],
        expected_file_sha256=_canonical_sha(built.cache_receipt),
    )

    replay = reopen_and_verify_local_cache_inventory(
        root=root,
        cache_receipt=built.cache_receipt,
        cache_trust=trust,
        records=records,
        mask_loader=_Loader(masks),
        expected_semantic_commitment_sha256=(built.semantic_commitment_sha256),
        config=_config(),
    )

    assert replay.semantic_commitment_sha256 == built.semantic_commitment_sha256
    assert replay.receipt["operational_counts"]["cache_miss_count"] == 0
    assert replay.receipt["operational_counts"]["cache_hit_count"] == 3

    with pytest.raises(LocalCacheInventoryError, match="experiment lock"):
        reopen_and_verify_local_cache_inventory(
            root=root,
            cache_receipt=built.cache_receipt,
            cache_trust=trust,
            records=records,
            mask_loader=_Loader(masks),
            expected_semantic_commitment_sha256="f" * 64,
            config=_config(),
        )


def test_train_validation_content_overlap_fails_closed(tmp_path):
    train = _record("a", "b", label=True, direction="right")
    val = _record(
        "a",
        "c",
        label=False,
        direction=None,
        split="val",
        dataset="fixture_eccv",
    )
    with pytest.raises(LocalCacheInventoryError, match="overlaps train"):
        build_local_cache_inventory(
            (train, val),
            _Loader(_masks()),
            GeometryArtifactCache(tmp_path / "cache"),
            _config(),
        )


def test_bounds_fail_before_geometry_and_writer_is_non_overwriting(tmp_path):
    records = _records()
    loader = _Loader(_masks())
    with pytest.raises(LocalCacheInventoryError, match="record population"):
        build_local_cache_inventory(
            records,
            loader,
            GeometryArtifactCache(tmp_path / "cache"),
            _config(max_records=1),
        )
    assert loader.requests == []

    receipt = {"schema_version": "fixture/0.1", "ok": True}
    output = tmp_path / "inventory.json"
    digest = write_portable_inventory(output, receipt)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    with pytest.raises(LocalCacheInventoryError, match="already exists"):
        write_portable_inventory(output, receipt)


def test_config_and_mask_guards_are_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="unique increasing"):
        _config(sequence_length_bucket_edges=(16, 16))

    records = _records()
    masks = _masks()
    masks["fragment/a"] = masks["fragment/a"].astype(np.uint8)
    with pytest.raises(LocalCacheInventoryError, match="2D bool"):
        build_local_cache_inventory(
            records,
            _Loader(masks),
            GeometryArtifactCache(tmp_path / "cache"),
            _config(),
        )

    huge_guard = replace(_config(), max_mask_pixels=10)
    with pytest.raises(LocalCacheInventoryError, match="configured bound"):
        build_local_cache_inventory(
            records,
            _Loader(_masks()),
            GeometryArtifactCache(tmp_path / "cache2"),
            huge_guard,
        )
