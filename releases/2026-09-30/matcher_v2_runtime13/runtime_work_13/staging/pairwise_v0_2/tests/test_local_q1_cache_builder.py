from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace

import numpy as np
import pytest

from staging.pairwise_v0_2.geometry import CandidateBuilderConfig
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.preflight.freeze_local_q1 import (
    SCHEMA_VERSION as LOCAL_Q1_FREEZE_SCHEMA_VERSION,
    LocalQ1FreezeResult,
)
from staging.pairwise_v0_2.training.geometry_cache import GeometryCacheError
from staging.pairwise_v0_2.training.local_cache_inventory import (
    LocalCacheInventoryConfig,
)
from staging.pairwise_v0_2.training.local_q1_cache_builder import (
    BUILD_RECEIPT_NAME,
    CACHE_DIRECTORY_NAME,
    CACHE_RECEIPT_NAME,
    INVENTORY_RECEIPT_NAME,
    LocalQ1CacheBuilderError,
    LocalQ1CacheTrust,
    LocalQ1FileLock,
    LocalQ1Population,
    LocalQ1PrecacheAuthority,
    attest_rebuilt_local_q1_population,
    build_local_q1_cache,
    reopen_local_q1_cache,
)


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mask(*, left: int, top: int = 9, notch: bool = False) -> np.ndarray:
    value = np.zeros((72, 88), dtype=bool)
    value[top : top + 50, left : left + 30] = True
    if notch:
        value[top + 15 : top + 30, left + 20 : left + 30] = False
    return value


_BINDING = ArchiveBinding(
    logical_id="fixture://local-q1-cache/masks",
    archive_format="zip",
    sha256="a" * 64,
)


def _reference(name: str, *, split: str) -> MaskMemberRef:
    return MaskMemberRef(
        binding=_BINDING,
        archive_member="masks/{}.png".format(name),
        fragment_id="fragment/{}".format(name),
        dataset_id="fixture_mm" if split == "train" else "fixture_eccv",
        canonical_group_id="group/{}/{}".format(split, name),
        component_id="component/{}/{}".format(split, name),
        split=split,
        threshold_rule="grayscale_uint8_gt_127",
        content_sha256=_sha("source-bytes/{}".format(name)),
    )


def _record(first: str, second: str, *, split: str, label: bool):
    fragment_a = _reference(first, split=split)
    fragment_b = _reference(second, split=split)
    dataset = fragment_a.dataset_id
    group = "group/{}/{}-{}".format(split, first, second)
    component = "component/{}/{}-{}".format(split, first, second)
    fragment_a = replace(fragment_a, canonical_group_id=group, component_id=component)
    fragment_b = replace(fragment_b, canonical_group_id=group, component_id=component)
    return TrainingPairRecord(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        label=label,
        direction_b_wrt_a="right" if label else None,
        dataset_id=dataset,
        canonical_group_id=group,
        component_id=component,
        split=split,
        canonical_pair_key=tuple(
            sorted((fragment_a.fragment_id, fragment_b.fragment_id))
        ),
        label_origin="fixture",
        provenance={"sealed_real": False},
    )


def _records():
    return (
        (
            _record("a", "b", split="train", label=True),
            _record("a_duplicate", "c", split="train", label=False),
        ),
        (_record("d", "e", split="val", label=True),),
    )


def _masks():
    first = _mask(left=7, notch=True)
    return {
        "fragment/a": first,
        # A distinct physical member intentionally has identical canonical mask.
        "fragment/a_duplicate": first.copy(),
        "fragment/b": _mask(left=21),
        "fragment/c": _mask(left=43, top=12, notch=True),
        "fragment/d": _mask(left=12, top=14),
        "fragment/e": _mask(left=49, top=7, notch=True),
    }


class _MaskLoader:
    def __init__(self, masks):
        self._masks = masks
        self.closed = False

    def __call__(self, reference):
        if self.closed:
            raise RuntimeError("fixture loader is closed")
        return self._masks[reference.fragment_id]

    def close(self):
        self.closed = True


@dataclass(frozen=True)
class _MaskLoaderFactory:
    masks: dict

    def __call__(self):
        # Every invocation owns detached masks, making spawn semantics explicit.
        return _MaskLoader({name: value.copy() for name, value in self.masks.items()})


def _freeze_receipt(train_count: int, validation_count: int):
    receipt = {
        "schema_version": LOCAL_Q1_FREEZE_SCHEMA_VERSION,
        "status": "pass_metadata_only_population_frozen_no_model_execution",
        "scope": {
            "experiment": "LOCAL-Q1",
            "pair_stream_splits_read": ["train", "val"],
            "sealed_real_read": False,
            "archive_mask_members_opened": False,
            "model_executed": False,
        },
        "locks": {
            "historical_identity_index": {
                "member_count": 6,
                "content_sha256": _sha("fixture-identity-index"),
            }
        },
        "training": {"population": {"count": train_count}},
        "validation": {"population": {"count": validation_count}},
        "portable_privacy": {
            "absolute_paths_present": False,
            "aggregate_commitments_only": True,
        },
    }
    receipt["content_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return receipt


def _role_locks(freeze_file_sha: str):
    names = (
        "freeze_receipt",
        "mm_archive",
        "eccv_archive",
        "mm_fingerprint_cache",
        "eccv_fingerprint_cache",
        "historical_split",
        "synthetic_manifest",
        "synthetic_archive",
    )
    return {
        name: LocalQ1FileLock(
            byte_count=index + 1,
            sha256=freeze_file_sha if name == "freeze_receipt" else _sha(name),
        )
        for index, name in enumerate(names)
    }


def _authority():
    return LocalQ1PrecacheAuthority(
        run_plan_file_sha256=_sha("fixture-run-plan-file"),
        run_plan_content_sha256=_sha("fixture-run-plan-content"),
        source_bundle_manifest_sha256=_sha("fixture-source-bundle-manifest"),
    )


def _population():
    train, validation = _records()
    receipt = _freeze_receipt(len(train), len(validation))
    freeze_file_sha = _sha("fixture-freeze-file")
    return LocalQ1Population(
        training_records=train,
        validation_records=validation,
        freeze_receipt=receipt,
        freeze_file_sha256=freeze_file_sha,
        freeze_content_sha256=receipt["content_sha256"],
        input_role_locks=_role_locks(freeze_file_sha),
        precache_authority=_authority(),
    )


def _config():
    return LocalCacheInventoryConfig(
        geometry=CandidateBuilderConfig(
            window_scale_fractions=(0.16,),
            window_min_px=2.0,
            window_max_px=48.0,
            output_size=(8, 10),
            min_run_length_fraction=0.0,
            min_run_length_px=2.0,
            side_resample_count=16,
        ),
        max_records=8,
        max_unique_references=16,
        max_mask_pixels=72 * 88,
    )


def _trust(artifacts):
    return LocalQ1CacheTrust(
        expected_freeze_file_sha256=_population().freeze_file_sha256,
        expected_freeze_content_sha256=_population().freeze_content_sha256,
        expected_run_plan_file_sha256=_authority().run_plan_file_sha256,
        expected_run_plan_content_sha256=_authority().run_plan_content_sha256,
        expected_source_bundle_manifest_sha256=(
            _authority().source_bundle_manifest_sha256
        ),
        expected_cache_receipt_file_sha256=(artifacts.cache_receipt_file_sha256),
        expected_cache_receipt_content_sha256=artifacts.cache_receipt["content_sha256"],
        expected_inventory_receipt_file_sha256=(
            artifacts.inventory_receipt_file_sha256
        ),
        expected_inventory_receipt_content_sha256=artifacts.inventory_receipt[
            "content_sha256"
        ],
        expected_inventory_semantic_sha256=artifacts.inventory_receipt[
            "semantic_commitment_sha256"
        ],
        expected_build_receipt_file_sha256=artifacts.build_receipt_file_sha256,
        expected_build_receipt_content_sha256=artifacts.build_receipt["content_sha256"],
    )


@pytest.fixture(scope="module")
def built_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("local-q1-cache-builder") / "formal-cache"
    population = _population()
    factory = _MaskLoaderFactory(_masks())
    artifacts = build_local_q1_cache(
        population=population,
        loader_factory=factory,
        output_dir=root,
        inventory_config=_config(),
        producer_workers=2,
    )
    return population, factory, artifacts, _trust(artifacts)


def test_metadata_rebuild_requires_external_file_content_and_exact_object():
    train, validation = _records()
    receipt = _freeze_receipt(len(train), len(validation))
    file_bytes = json.dumps(
        receipt, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8")
    file_sha = hashlib.sha256(file_bytes).hexdigest()
    locks = _role_locks(file_sha)
    rebuilt = LocalQ1FreezeResult(train, validation, receipt)

    population = attest_rebuilt_local_q1_population(
        canonical_freeze_file_bytes=file_bytes,
        expected_freeze_file_sha256=file_sha,
        expected_freeze_content_sha256=receipt["content_sha256"],
        rebuilt=rebuilt,
        input_role_locks=locks,
        precache_authority=_authority(),
    )
    assert population.training_records == train
    assert population.training_records is population.training_records
    assert population.validation_records == validation
    assert population.records == train + validation

    changed = dict(receipt)
    changed["status"] = "forged"
    with pytest.raises(LocalQ1CacheBuilderError, match="reconstruction differs"):
        attest_rebuilt_local_q1_population(
            canonical_freeze_file_bytes=file_bytes,
            expected_freeze_file_sha256=file_sha,
            expected_freeze_content_sha256=receipt["content_sha256"],
            rebuilt=LocalQ1FreezeResult(train, validation, changed),
            input_role_locks=locks,
            precache_authority=_authority(),
        )
    with pytest.raises(LocalQ1CacheBuilderError, match="file hash mismatch"):
        attest_rebuilt_local_q1_population(
            canonical_freeze_file_bytes=file_bytes,
            expected_freeze_file_sha256="f" * 64,
            expected_freeze_content_sha256=receipt["content_sha256"],
            rebuilt=rebuilt,
            input_role_locks=locks,
            precache_authority=_authority(),
        )


def test_parallel_process_local_build_is_canonical_portable_and_producer_once(
    built_case,
):
    _population_value, _factory, artifacts, _trust_value = built_case
    producer = artifacts.build_receipt["producer_contract"]
    inventory = artifacts.inventory_receipt["operational_counts"]
    cache = artifacts.build_receipt["cache"]

    assert producer["mode"] == ("process_local_deterministic_physical_reference_shards")
    assert producer["active_worker_count"] == 2
    assert producer["shared_python_loader_or_cache_objects"] is False
    assert producer["producer_cache_miss_count"] == cache["artifact_count"] == 5
    assert producer["producer_cache_hit_count"] == 1
    assert inventory["unique_physical_reference_count"] == 6
    assert inventory["unique_canonical_fragment_count"] == 5
    assert inventory["canonical_duplicate_reference_count"] == 1
    assert artifacts.read_only_cache.read_only is True
    assert artifacts.read_only_cache.frozen_receipt_trust is not None
    assert artifacts.build_receipt["precache_authority"] == (
        _authority().portable_dict()
    )

    encoded = json.dumps(
        {
            "cache": dict(artifacts.cache_receipt),
            "inventory": dict(artifacts.inventory_receipt),
            "build": dict(artifacts.build_receipt),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    for forbidden in (
        "/Users/",
        "/tmp/",
        "fragment/a",
        "masks/a.png",
        '"archive_member":',
        '"pair_id":',
        '"secret"',
    ):
        assert forbidden not in encoded


def test_external_trust_reopen_returns_same_read_only_cache_and_zero_miss(built_case):
    population, factory, artifacts, trust = built_case
    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=factory,
        output_dir=artifacts.output_dir,
        trust=trust,
        inventory_config=_config(),
    )

    assert opened.cache.read_only is True
    assert opened.cache is opened.cache
    assert opened.cache.frozen_receipt_trust is not None
    counts = opened.inventory_replay.receipt["operational_counts"]
    assert counts["cache_miss_count"] == 0
    assert counts["cache_hit_count"] == counts["unique_canonical_fragment_count"]
    assert opened.population.training_records == population.training_records

    forged_semantic = replace(trust, expected_inventory_semantic_sha256="f" * 64)
    with pytest.raises(LocalQ1CacheBuilderError, match="semantic commitment"):
        reopen_local_q1_cache(
            population=population,
            loader_factory=factory,
            output_dir=artifacts.output_dir,
            trust=forged_semantic,
            inventory_config=_config(),
        )

    for field in (
        "expected_run_plan_file_sha256",
        "expected_run_plan_content_sha256",
        "expected_source_bundle_manifest_sha256",
    ):
        forged_authority = replace(trust, **{field: "f" * 64})
        with pytest.raises(
            LocalQ1CacheBuilderError, match="plan/source/freeze anchors"
        ):
            reopen_local_q1_cache(
                population=population,
                loader_factory=factory,
                output_dir=artifacts.output_dir,
                trust=forged_authority,
                inventory_config=_config(),
            )


def test_fast_reopen_reuses_frozen_inventory_without_source_mask_loader(built_case):
    population, _factory, artifacts, trust = built_case

    def forbidden_loader_factory():
        raise AssertionError("fast reopen must not construct the source mask loader")

    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=forbidden_loader_factory,
        output_dir=artifacts.output_dir,
        trust=trust,
        inventory_config=_config(),
        replay_source_masks=False,
    )

    assert opened.cache.read_only is True
    assert dict(opened.inventory_replay.receipt) == dict(artifacts.inventory_receipt)
    counts = opened.inventory_replay.receipt["operational_counts"]
    assert counts["cache_miss_count"] == 0
    assert counts["cache_hit_count"] == counts["unique_canonical_fragment_count"]


def _copy_case(tmp_path, built_case, name):
    _population_value, _factory, artifacts, _trust_value = built_case
    destination = tmp_path / name
    shutil.copytree(artifacts.output_dir, destination)
    return destination


def _reopen_copy(path, built_case):
    population, factory, _artifacts, trust = built_case
    return reopen_local_q1_cache(
        population=population,
        loader_factory=factory,
        output_dir=path,
        trust=trust,
        inventory_config=_config(),
    )


def test_missing_extra_and_tampered_cache_artifacts_fail_closed(tmp_path, built_case):
    missing = _copy_case(tmp_path, built_case, "missing")
    missing_artifact = next((missing / CACHE_DIRECTORY_NAME).rglob("*.npz"))
    missing_artifact.unlink()
    with pytest.raises(GeometryCacheError, match="inventory differs"):
        _reopen_copy(missing, built_case)

    extra = _copy_case(tmp_path, built_case, "extra")
    source = next((extra / CACHE_DIRECTORY_NAME).rglob("*.npz"))
    extra_path = extra / CACHE_DIRECTORY_NAME / "00" / ("0" * 64 + ".npz")
    extra_path.parent.mkdir(exist_ok=True)
    shutil.copyfile(source, extra_path)
    with pytest.raises(GeometryCacheError, match="inventory differs"):
        _reopen_copy(extra, built_case)

    tampered = _copy_case(tmp_path, built_case, "tampered")
    target = next((tampered / CACHE_DIRECTORY_NAME).rglob("*.npz"))
    payload = bytearray(target.read_bytes())
    payload[len(payload) // 2] ^= 1
    target.write_bytes(payload)
    with pytest.raises(GeometryCacheError):
        _reopen_copy(tampered, built_case)


def test_missing_extra_and_forged_receipts_fail_external_anchors(tmp_path, built_case):
    missing = _copy_case(tmp_path, built_case, "receipt-missing")
    (missing / INVENTORY_RECEIPT_NAME).unlink()
    with pytest.raises(LocalQ1CacheBuilderError, match="top-level entries"):
        _reopen_copy(missing, built_case)

    extra = _copy_case(tmp_path, built_case, "receipt-extra")
    (extra / "unreviewed.json").write_text("{}", encoding="utf-8")
    with pytest.raises(LocalQ1CacheBuilderError, match="top-level entries"):
        _reopen_copy(extra, built_case)

    planning_sidecar = _copy_case(tmp_path, built_case, "planning-sidecar")
    (planning_sidecar / "local_q1_planning_config.json").write_text(
        "{}", encoding="utf-8"
    )
    assert _reopen_copy(planning_sidecar, built_case).cache.read_only is True

    forged = _copy_case(tmp_path, built_case, "receipt-forged")
    path = forged / INVENTORY_RECEIPT_NAME
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["semantic_commitment_sha256"] = "f" * 64
    receipt.pop("content_sha256")
    receipt["content_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    path.write_bytes(_canonical(receipt))
    with pytest.raises(LocalQ1CacheBuilderError, match="external file hash"):
        _reopen_copy(forged, built_case)

    # A cache-receipt self-resign is equally unable to forge external trust.
    forged_cache = _copy_case(tmp_path, built_case, "cache-receipt-forged")
    cache_path = forged_cache / CACHE_RECEIPT_NAME
    cache_receipt = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_receipt["path_policy"] = "forged-but-self-signed"
    cache_receipt.pop("content_sha256")
    cache_receipt["content_sha256"] = hashlib.sha256(
        _canonical(cache_receipt)
    ).hexdigest()
    cache_path.write_bytes(_canonical(cache_receipt))
    with pytest.raises(LocalQ1CacheBuilderError, match="external file hash"):
        _reopen_copy(forged_cache, built_case)


def test_duplicate_producer_refuses_before_loader_or_receipt_overwrite(built_case):
    population, _factory, artifacts, _trust_value = built_case

    def forbidden_factory():
        raise AssertionError("duplicate producer must not construct a loader")

    before = {
        name: (artifacts.output_dir / name).read_bytes()
        for name in (CACHE_RECEIPT_NAME, INVENTORY_RECEIPT_NAME, BUILD_RECEIPT_NAME)
    }
    with pytest.raises(LocalQ1CacheBuilderError, match="duplicate producer"):
        build_local_q1_cache(
            population=population,
            loader_factory=forbidden_factory,
            output_dir=artifacts.output_dir,
            inventory_config=_config(),
            producer_workers=1,
        )
    after = {
        name: (artifacts.output_dir / name).read_bytes()
        for name in (CACHE_RECEIPT_NAME, INVENTORY_RECEIPT_NAME, BUILD_RECEIPT_NAME)
    }
    assert after == before


def test_unpickleable_parallel_factory_fails_closed_with_single_writer_guidance(
    tmp_path,
):
    masks = _masks()
    factory = lambda: _MaskLoader(masks)  # noqa: E731 - intentionally unpickleable
    with pytest.raises(LocalQ1CacheBuilderError, match="producer_workers=1"):
        build_local_q1_cache(
            population=_population(),
            loader_factory=factory,
            output_dir=tmp_path / "unpickleable",
            inventory_config=_config(),
            producer_workers=2,
        )


def test_population_has_no_sealed_or_test_record_capability():
    train, validation = _records()
    receipt = _freeze_receipt(len(train), len(validation))
    freeze_file_sha = _sha("fixture-freeze-file")
    tainted = replace(train[0], provenance={"real_dunhuang_sealed_test": True})
    with pytest.raises(LocalQ1CacheBuilderError, match="sealed/test provenance"):
        LocalQ1Population(
            training_records=(tainted,) + train[1:],
            validation_records=validation,
            freeze_receipt=receipt,
            freeze_file_sha256=freeze_file_sha,
            freeze_content_sha256=receipt["content_sha256"],
            input_role_locks=_role_locks(freeze_file_sha),
            precache_authority=_authority(),
        )
