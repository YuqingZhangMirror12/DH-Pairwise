from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import zipfile
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch
from PIL import Image

from staging.pairwise_v0_2.models.local_matcher import MatcherMode
from staging.pairwise_v0_2.models.pairwise import ArcPoolingConfig
from staging.pairwise_v0_2.pairwise_data import historical_identity as identity_module
from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    FREEZE_SCHEMA_VERSION,
    HISTORICAL_TEST_ACCESS_EVIDENCE,
    FragmentIdentity,
    HistoricalIdentityIndex,
    VerifiedPair,
)
from staging.pairwise_v0_2.pairwise_data.lazy_mask_loader import (
    ArchiveSourceSpec,
    LazyMaskArchiveLoader,
    LazyMaskLoaderConfig,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training import c0_coarse_provider as provider_module
from staging.pairwise_v0_2.training.c0_coarse_backend import (
    C0_COARSE_MODEL_CONFIG,
    C0_COARSE_OPTIMIZER_CONFIG,
    C0CoarsePayload,
    c0_coarse_payload_content_sha256,
)
from staging.pairwise_v0_2.training.c0_coarse_provider import (
    C0_COARSE_PROVIDER_RECEIPT_VERSION,
    C0_COARSE_PROVIDER_VERSION,
    C0_COARSE_TENSOR_BYTES,
    C0_PRODUCTION_MAX_CACHE_BYTES,
    C0_PRODUCTION_MAX_CACHED_FRAGMENTS,
    C0CoarseProvider,
    C0CoarseProviderConfig,
    C0CoarseProviderError,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    EvidenceMode,
    record_sequence_fingerprint,
)


def _png(mask: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(stream, format="PNG")
    return stream.getvalue()


def _zip(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w") as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)
    return stream.getvalue()


def _tar(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def _receipt(
    mm_binding,
    eccv_binding,
    mm_bytes,
    eccv_bytes,
    *,
    artifact_locks,
    split_candidate_id,
    split_seed,
):
    value = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "pass_metadata_only_no_model_execution",
        "scope": {
            "experiment": "C0-N-Q1",
            "datasets": ["mm_augmented", "eccv_1113data"],
            "pair_stream_splits_read": ["train", "val"],
            "historical_test_access": dict(HISTORICAL_TEST_ACCESS_EVIDENCE),
            "sealed_real_read": False,
            "mask_pixels_decoded": False,
            "model_executed": False,
        },
        "locks": {
            "archives": {
                "mm_augmented": {
                    "format": "zip",
                    "bytes": len(mm_bytes),
                    "sha256": mm_binding.sha256,
                },
                "eccv_1113data": {
                    "format": "tar",
                    "bytes": len(eccv_bytes),
                    "sha256": eccv_binding.sha256,
                },
            },
            "metadata_artifacts": artifact_locks,
            "split_candidate_sha256": hashlib.sha256(
                split_candidate_id.encode("utf-8")
            ).hexdigest(),
            "split_seed_sha256": hashlib.sha256(split_seed.encode("utf-8")).hexdigest(),
        },
    }
    value["content_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return value


def _arm() -> AblationArm:
    return AblationArm(
        name=AblationArmName.COARSE_ONLY,
        evidence=EvidenceMode.COARSE,
        matcher_mode=None,
        model_config=C0_COARSE_MODEL_CONFIG,
        optimizer_config=C0_COARSE_OPTIMIZER_CONFIG,
        aggregation_config={"evidence": "coarse", "matcher_mode": None},
        arc_pooling=None,
    )


@pytest.fixture
def provider_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    large = np.zeros((24, 32), dtype=bool)
    large[5:17, 8:26] = True
    # Smaller disconnected component proves that only the largest survives.
    large[1:3, 1:3] = True
    tall = np.zeros((30, 18), dtype=bool)
    tall[4:27, 5:13] = True
    wide = np.zeros((18, 34), dtype=bool)
    wide[6:13, 2:31] = True
    square = np.zeros((20, 20), dtype=bool)
    square[3:17, 3:17] = True
    empty = np.zeros((12, 12), dtype=bool)
    full = np.ones((12, 12), dtype=bool)

    mm_payloads = {
        "mm/group/a.png": _png(large),
        "mm/group/b.png": _png(tall),
        "mm/group/empty.png": _png(empty),
        "mm/group/full.png": _png(full),
    }
    eccv_payloads = {
        "eccv/group/a.png": _png(wide),
        "eccv/group/b.png": _png(square),
    }
    mm_bytes = _zip(mm_payloads)
    eccv_bytes = _tar(eccv_payloads)
    mm_binding = ArchiveBinding(
        "canonical://fixture/mm", "zip", hashlib.sha256(mm_bytes).hexdigest()
    )
    eccv_binding = ArchiveBinding(
        "canonical://fixture/eccv", "tar", hashlib.sha256(eccv_bytes).hexdigest()
    )
    monkeypatch.setattr(provider_module, "MM_CANONICAL_BINDING", mm_binding)
    monkeypatch.setattr(provider_module, "ECCV_CANONICAL_BINDING", eccv_binding)
    monkeypatch.setattr(identity_module, "MM_CANONICAL_BINDING", mm_binding)
    monkeypatch.setattr(identity_module, "ECCV_CANONICAL_BINDING", eccv_binding)

    by_member = {}
    records = {}

    def add_dataset(
        dataset_id: str,
        binding: ArchiveBinding,
        members: dict[str, bytes],
        split: str,
    ) -> None:
        group = "{}/group".format(dataset_id)
        component = "{}/component".format(dataset_id)
        identities = {}
        for index, (member, payload) in enumerate(sorted(members.items())):
            identity = FragmentIdentity(
                dataset_id=dataset_id,
                binding=binding,
                archive_member=member,
                fragment_id="{}/fragment/{}".format(dataset_id, index),
                pair_group_id=group,
                component_id=component,
                split=split,
                content_sha256=hashlib.sha256(payload).hexdigest(),
            )
            by_member[(dataset_id, member)] = identity
            identities[member.rsplit("/", 1)[-1][:-4]] = identity

        def reference(name: str, *, threshold: str = "binary_brighter_value"):
            identity = identities[name]
            return MaskMemberRef(
                binding=binding,
                archive_member=identity.archive_member,
                fragment_id=identity.fragment_id,
                dataset_id=dataset_id,
                canonical_group_id=group,
                component_id=component,
                split=split,
                threshold_rule=threshold,
            )

        def record(name: str, first: str, second: str, label: bool):
            a = reference(first)
            b = reference(second)
            records[name] = TrainingPairRecord(
                fragment_a=a,
                fragment_b=b,
                label=label,
                direction_b_wrt_a="right" if label else None,
                dataset_id=dataset_id,
                canonical_group_id=group,
                component_id=component,
                split=split,
                canonical_pair_key=tuple(sorted((a.fragment_id, b.fragment_id))),
                label_origin="provider_fixture",
                provenance={"real_dunhuang_sealed_test": False},
            )

        if dataset_id == "mm_augmented":
            record("mm_negative", "a", "b", False)
            record("mm_positive", "a", "b", True)
            record("mm_empty", "empty", "b", False)
            record("mm_full", "full", "b", False)
        else:
            record("eccv_val", "a", "b", True)

    add_dataset("mm_augmented", mm_binding, mm_payloads, "train")
    add_dataset("eccv_1113data", eccv_binding, eccv_payloads, "val")
    artifact_locks = {
        "mm_fingerprint_cache": {"bytes": 101, "sha256": "1" * 64},
        "eccv_fingerprint_cache": {"bytes": 102, "sha256": "2" * 64},
        "historical_split": {"bytes": 103, "sha256": "3" * 64},
    }
    split_candidate_id = "fixture-split"
    split_seed = "fixture-seed"
    index = HistoricalIdentityIndex(
        by_member=by_member,
        split_candidate_id=split_candidate_id,
        split_seed=split_seed,
        artifact_locks=artifact_locks,
    )
    loader = LazyMaskArchiveLoader(
        {
            mm_binding.logical_id: ArchiveSourceSpec(mm_binding, mm_bytes),
            eccv_binding.logical_id: ArchiveSourceSpec(eccv_binding, eccv_bytes),
        },
        config=LazyMaskLoaderConfig(cache_size=32),
    )
    receipt = _receipt(
        mm_binding,
        eccv_binding,
        mm_bytes,
        eccv_bytes,
        artifact_locks=artifact_locks,
        split_candidate_id=split_candidate_id,
        split_seed=split_seed,
    )

    def make_provider(config=C0CoarseProviderConfig()):
        return C0CoarseProvider(
            identity_index=index,
            mask_loader=loader,
            freeze_receipt=receipt,
            expected_freeze_content_sha256=receipt["content_sha256"],
            expected_identity_index_content_sha256=index.content_sha256,
            config=config,
        )

    return {
        "index": index,
        "loader": loader,
        "receipt": receipt,
        "records": records,
        "make_provider": make_provider,
        "tmp_path": tmp_path,
        "mm_binding": mm_binding,
        "eccv_binding": eccv_binding,
        "mm_bytes": mm_bytes,
        "eccv_bytes": eccv_bytes,
    }


def test_constructor_completes_full_archive_preflight_before_any_member_decode(
    provider_fixture,
):
    loader = provider_fixture["loader"]
    assert loader.stats.archive_verifications == 0
    provider = provider_fixture["make_provider"]()

    assert loader.stats.archive_verifications == 2
    assert loader.stats.archive_opens == 0
    assert loader.stats.requests == 0
    assert loader.stats.decoded_masks == 0
    assert {item.logical_id for item in provider.archive_verification_receipts} == {
        provider_fixture["mm_binding"].logical_id,
        provider_fixture["eccv_binding"].logical_id,
    }
    assert provider.contract.coarse_only_geometry_free is True
    assert provider.contract.provider_version == C0_COARSE_PROVIDER_VERSION


def test_default_memo_bounds_are_exact_production_capacity(provider_fixture):
    config = C0CoarseProviderConfig()
    assert C0_COARSE_PROVIDER_VERSION == "c0-n-q1-coarse-provider/0.5"
    assert C0_COARSE_PROVIDER_RECEIPT_VERSION == "c0-n-q1-coarse-provider-receipt/0.5"
    assert config.max_cached_fragments == C0_PRODUCTION_MAX_CACHED_FRAGMENTS
    assert config.max_cache_bytes == C0_PRODUCTION_MAX_CACHE_BYTES
    assert C0_COARSE_TENSOR_BYTES == 128 * 128 * 4
    assert (
        config.max_cached_fragments * C0_COARSE_TENSOR_BYTES
        == config.max_cache_bytes
        == 8 * 1024**3
    )

    receipt = provider_fixture["make_provider"]().receipt()
    assert receipt["schema_version"] == C0_COARSE_PROVIDER_RECEIPT_VERSION
    assert receipt["provider_version"] == C0_COARSE_PROVIDER_VERSION
    assert receipt["memo_bounds"] == {
        "max_batch_size": 256,
        "max_cached_fragments": C0_PRODUCTION_MAX_CACHED_FRAGMENTS,
        "max_cache_bytes": C0_PRODUCTION_MAX_CACHE_BYTES,
    }


@pytest.mark.parametrize("tamper", ("artifact", "candidate", "seed"))
def test_identity_index_locks_are_bound_before_archive_preflight(
    provider_fixture,
    tamper,
):
    trusted = provider_fixture["index"]
    artifact_locks = {
        role: dict(value) for role, value in trusted.artifact_locks.items()
    }
    candidate = trusted.split_candidate_id
    seed = trusted.split_seed
    if tamper == "artifact":
        artifact_locks["historical_split"]["sha256"] = "f" * 64
    elif tamper == "candidate":
        candidate += "-forged"
    else:
        seed += "-forged"
    forged = HistoricalIdentityIndex(
        by_member=trusted._by_member,
        split_candidate_id=candidate,
        split_seed=seed,
        artifact_locks=artifact_locks,
    )

    with pytest.raises(C0CoarseProviderError, match="historical"):
        C0CoarseProvider(
            identity_index=forged,
            mask_loader=provider_fixture["loader"],
            freeze_receipt=provider_fixture["receipt"],
            expected_freeze_content_sha256=provider_fixture["receipt"][
                "content_sha256"
            ],
            expected_identity_index_content_sha256=(
                provider_fixture["index"].content_sha256
            ),
        )
    assert provider_fixture["loader"].stats.archive_verifications == 0
    assert provider_fixture["loader"].stats.requests == 0


def test_forged_identity_mapping_fails_external_content_lock_before_archive(
    provider_fixture,
):
    trusted = provider_fixture["index"]
    by_member = dict(trusted._by_member)
    key = next(key for key in sorted(by_member) if key[0] == "eccv_1113data")
    by_member[key] = replace(by_member[key], split="train")
    forged = HistoricalIdentityIndex(
        by_member=by_member,
        split_candidate_id=trusted.split_candidate_id,
        split_seed=trusted.split_seed,
        artifact_locks=trusted.artifact_locks,
    )
    assert forged.content_sha256 != trusted.content_sha256

    with pytest.raises(C0CoarseProviderError, match="content differs"):
        C0CoarseProvider(
            identity_index=forged,
            mask_loader=provider_fixture["loader"],
            freeze_receipt=provider_fixture["receipt"],
            expected_freeze_content_sha256=provider_fixture["receipt"][
                "content_sha256"
            ],
            expected_identity_index_content_sha256=trusted.content_sha256,
        )
    assert provider_fixture["loader"].stats.archive_verifications == 0
    assert provider_fixture["loader"].stats.archive_opens == 0
    assert provider_fixture["loader"].stats.requests == 0


def test_rekeyed_private_identity_row_fails_before_archive_preflight(
    provider_fixture,
):
    index = provider_fixture["index"]
    rows = dict(index._by_member._rows)
    key = sorted(rows)[0]
    row = rows.pop(key)
    rows[(key[0], key[1].replace(".png", "-rekeyed.png"))] = row
    object.__setattr__(index._by_member, "_rows", MappingProxyType(rows))

    with pytest.raises(C0CoarseProviderError, match="index content is invalid"):
        C0CoarseProvider(
            identity_index=index,
            mask_loader=provider_fixture["loader"],
            freeze_receipt=provider_fixture["receipt"],
            expected_freeze_content_sha256=provider_fixture["receipt"][
                "content_sha256"
            ],
            expected_identity_index_content_sha256=index.content_sha256,
        )
    assert provider_fixture["loader"].stats.archive_verifications == 0
    assert provider_fixture["loader"].stats.archive_opens == 0
    assert provider_fixture["loader"].stats.requests == 0


def test_resigned_metadata_and_forged_reported_digest_cannot_hide_mapping_change(
    provider_fixture,
):
    trusted = provider_fixture["index"]
    by_member = dict(trusted._by_member)
    key = next(key for key in sorted(by_member) if key[0] == "eccv_1113data")
    by_member[key] = replace(by_member[key], split="train")
    forged_artifacts = {
        role: dict(lock) for role, lock in trusted.artifact_locks.items()
    }
    forged_artifacts["historical_split"]["sha256"] = "f" * 64
    forged = HistoricalIdentityIndex(
        by_member=by_member,
        split_candidate_id=trusted.split_candidate_id,
        split_seed=trusted.split_seed,
        artifact_locks=forged_artifacts,
    )
    # Simulate a forged object that re-labels its cached digest.  Provider must
    # recompute the primitive mapping rather than trust this field.
    object.__setattr__(forged, "_content_sha256", trusted.content_sha256)

    resigned = json.loads(json.dumps(provider_fixture["receipt"]))
    resigned["locks"]["metadata_artifacts"] = forged_artifacts
    unsigned = dict(resigned)
    unsigned.pop("content_sha256")
    resigned["content_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with pytest.raises(C0CoarseProviderError, match="self-reported digest"):
        C0CoarseProvider(
            identity_index=forged,
            mask_loader=provider_fixture["loader"],
            freeze_receipt=resigned,
            expected_freeze_content_sha256=resigned["content_sha256"],
            expected_identity_index_content_sha256=trusted.content_sha256,
        )
    assert provider_fixture["loader"].stats.archive_verifications == 0
    assert provider_fixture["loader"].stats.archive_opens == 0
    assert provider_fixture["loader"].stats.requests == 0


def test_registered_archive_format_is_bound_before_archive_verification(
    provider_fixture,
):
    mm = provider_fixture["mm_binding"]
    eccv = provider_fixture["eccv_binding"]
    wrong_mm = ArchiveBinding(mm.logical_id, "tar", mm.sha256)
    loader = LazyMaskArchiveLoader(
        {
            wrong_mm.logical_id: ArchiveSourceSpec(
                wrong_mm, provider_fixture["mm_bytes"]
            ),
            eccv.logical_id: ArchiveSourceSpec(eccv, provider_fixture["eccv_bytes"]),
        }
    )
    with pytest.raises(C0CoarseProviderError, match="binding format/identity"):
        C0CoarseProvider(
            identity_index=provider_fixture["index"],
            mask_loader=loader,
            freeze_receipt=provider_fixture["receipt"],
            expected_freeze_content_sha256=provider_fixture["receipt"][
                "content_sha256"
            ],
            expected_identity_index_content_sha256=(
                provider_fixture["index"].content_sha256
            ),
        )
    assert loader.stats.archive_verifications == 0
    assert loader.stats.archive_opens == 0


def test_prepare_enriches_physical_identity_and_builds_exact_coarse_payload(
    provider_fixture,
):
    provider = provider_fixture["make_provider"]()
    records = [
        provider_fixture["records"]["mm_negative"],
        provider_fixture["records"]["mm_positive"],
    ]
    prepared = provider.prepare(records, arm=_arm(), phase="train")

    assert isinstance(prepared.payload, C0CoarsePayload)
    assert prepared.payload.coarse_a.shape == (2, 1, 128, 128)
    assert prepared.payload.coarse_b.shape == (2, 1, 128, 128)
    assert prepared.payload.coarse_a.dtype == torch.float32
    assert prepared.payload.labels.tolist() == [0.0, 1.0]
    assert prepared.prepared_input_sha256 == prepared.payload.payload_content_sha256
    assert prepared.prepared_input_sha256 == c0_coarse_payload_content_sha256(
        prepared.payload.coarse_a,
        prepared.payload.coarse_b,
        prepared.payload.labels,
    )
    verified_records = [
        provider_fixture["index"].verify_record(record).record for record in records
    ]
    assert prepared.record_sequence_sha256 == record_sequence_fingerprint(
        verified_records
    )
    assert all(record.fragment_a.content_sha256 is None for record in records)
    assert prepared.local_candidate_sha256 is None
    assert prepared.geometry_config_sha256 is None
    assert prepared.processing_counts == {
        "mask_load_count": 2,
        "coarse_preprocess_count": 2,
        "geometry_build_count": 0,
        "geometry_cache_read_count": 0,
        "geometry_cache_write_count": 0,
        "local_candidate_count": 0,
    }

    support = torch.nonzero(prepared.payload.coarse_a[0, 0] > 0.0, as_tuple=False)
    row_min, col_min = support.min(dim=0).values.tolist()
    row_max, col_max = (support.max(dim=0).values + 1).tolist()
    assert max(row_max - row_min, col_max - col_min) == 112
    assert sorted((row_max - row_min, col_max - col_min)) == [75, 112]
    assert abs(row_min - (128 - row_max)) <= 1
    assert abs(col_min - (128 - col_max)) <= 1


def test_bounded_fragment_memo_avoids_duplicate_decode_and_preprocess(
    provider_fixture,
):
    provider = provider_fixture["make_provider"]()
    records = [
        provider_fixture["records"]["mm_negative"],
        provider_fixture["records"]["mm_positive"],
    ]
    first = provider.prepare(records, arm=_arm(), phase="train")
    second = provider.prepare(records, arm=_arm(), phase="train")

    assert first.processing_counts["mask_load_count"] == 2
    assert second.processing_counts["mask_load_count"] == 0
    assert second.processing_counts["coarse_preprocess_count"] == 0
    stats = provider.stats_snapshot()
    assert stats.fragment_request_count == 8
    assert stats.memo_miss_count == 2
    assert stats.memo_hit_count == 6
    assert stats.mask_loader_call_count == 2
    assert stats.archive_decode_count == 2
    assert stats.coarse_preprocess_count == 2
    assert stats.memo_entry_count == 2
    assert stats.memo_byte_count == 2 * 128 * 128 * 4


def test_memo_eviction_is_bounded_and_loader_hits_are_accounted(provider_fixture):
    provider = provider_fixture["make_provider"](
        C0CoarseProviderConfig(
            max_batch_size=256,
            max_cached_fragments=1,
            max_cache_bytes=128 * 128 * 4,
        )
    )
    record = provider_fixture["records"]["mm_negative"]
    provider.prepare([record], arm=_arm(), phase="train")
    provider.prepare([record], arm=_arm(), phase="train")
    stats = provider.stats_snapshot()

    assert stats.memo_entry_count == 1
    assert stats.memo_byte_count <= 128 * 128 * 4
    assert stats.memo_eviction_count == 3
    assert stats.memo_miss_count == 4
    assert stats.archive_decode_count == 2
    assert stats.loader_cache_hit_count == 2


def test_external_loader_use_during_all_memo_hit_batch_taints_and_fails(
    provider_fixture,
    monkeypatch,
):
    provider = provider_fixture["make_provider"]()
    record = provider_fixture["records"]["mm_negative"]
    provider.prepare([record], arm=_arm(), phase="train")
    verified = provider_fixture["index"].verify_record(record)
    original = provider._coarse_fragment
    injected = False

    def raced(identity, reference):
        nonlocal injected
        result = original(identity, reference)
        if not injected:
            injected = True
            # This is a real external loader call during a provider batch.  It
            # hits the loader cache while every provider endpoint hits its own
            # coarse memo, so no provider-owned loader delta may absorb it.
            provider_fixture["loader"](verified.record.fragment_a)
        return result

    monkeypatch.setattr(provider, "_coarse_fragment", raced)
    with pytest.raises(C0CoarseProviderError, match="raced the C0 batch"):
        provider.prepare([record], arm=_arm(), phase="train")
    with pytest.raises(C0CoarseProviderError, match="tainted"):
        provider.receipt()


def test_external_loader_use_inside_provider_miss_is_not_absorbed(
    provider_fixture,
    monkeypatch,
):
    provider = provider_fixture["make_provider"]()
    record = provider_fixture["records"]["mm_negative"]
    verified = provider_fixture["index"].verify_record(record)
    original = LazyMaskArchiveLoader.__call__
    injected = False

    def raced(loader, reference):
        nonlocal injected
        result = original(loader, reference)
        if loader is provider_fixture["loader"] and not injected:
            injected = True
            original(loader, verified.record.fragment_b)
        return result

    monkeypatch.setattr(LazyMaskArchiveLoader, "__call__", raced)
    with pytest.raises(C0CoarseProviderError, match="external mask loader request"):
        provider.prepare([record], arm=_arm(), phase="train")
    with pytest.raises(C0CoarseProviderError, match="tainted"):
        provider.receipt()


def test_every_loader_stat_delta_is_reconciled_even_on_all_memo_hits(
    provider_fixture,
    monkeypatch,
):
    provider = provider_fixture["make_provider"]()
    record = provider_fixture["records"]["mm_negative"]
    provider.prepare([record], arm=_arm(), phase="train")
    original = provider._coarse_fragment
    injected = False

    def raced(identity, reference):
        nonlocal injected
        result = original(identity, reference)
        if not injected:
            injected = True
            stats = provider_fixture["loader"].stats
            stats.archive_verifications += 1
            stats.archive_verified_bytes += 7
            stats.archive_hash_failures += 1
        return result

    monkeypatch.setattr(provider, "_coarse_fragment", raced)
    with pytest.raises(C0CoarseProviderError, match="raced the C0 batch"):
        provider.prepare([record], arm=_arm(), phase="train")
    assert provider._failed is True


def test_receipt_and_stats_snapshot_fail_nonblocking_during_prepare(
    provider_fixture,
    monkeypatch,
):
    provider = provider_fixture["make_provider"]()
    record = provider_fixture["records"]["mm_negative"]
    entered = threading.Event()
    release = threading.Event()
    original = provider._coarse_fragment
    failures = []

    def paused(identity, reference):
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test synchronization timed out")
        return original(identity, reference)

    monkeypatch.setattr(provider, "_coarse_fragment", paused)

    def run_prepare():
        try:
            provider.prepare([record], arm=_arm(), phase="train")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=run_prepare)
    worker.start()
    assert entered.wait(timeout=5)
    with pytest.raises(C0CoarseProviderError, match="receipt during a batch"):
        provider.receipt()
    with pytest.raises(C0CoarseProviderError, match="statistics during a batch"):
        provider.stats_snapshot()
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert failures == []
    assert provider.receipt()["counters"]["batch_count"] == 1


def test_receipt_is_portable_and_contains_exact_counters(provider_fixture):
    provider = provider_fixture["make_provider"]()
    provider.prepare(
        [provider_fixture["records"]["mm_negative"]],
        arm=_arm(),
        phase="train",
    )
    receipt = provider.receipt()
    text = json.dumps(receipt, sort_keys=True)

    assert receipt["status"] == "archive_preflight_complete"
    assert (
        receipt["identity_index_content_sha256"]
        == provider_fixture["index"].content_sha256
    )
    assert receipt["geometry_cache_local_sinkhorn_calls"] == 0
    assert receipt["historical_test_access"] == dict(HISTORICAL_TEST_ACCESS_EVIDENCE)
    assert receipt["sealed_real_read"] is False
    assert receipt["counters"]["archive_decode_count"] == 2
    assert {item["archive_format"] for item in receipt["archive_verification"]} == {
        "zip",
        "tar",
    }
    assert str(provider_fixture["tmp_path"]) not in text
    assert "mm/group/a.png" not in text
    assert "mm_augmented/component" not in text


def test_record_order_attestation_is_order_sensitive_and_accepts_verified_pairs(
    provider_fixture,
):
    provider = provider_fixture["make_provider"]()
    negative = provider_fixture["index"].verify_record(
        provider_fixture["records"]["mm_negative"]
    )
    positive = provider_fixture["index"].verify_record(
        provider_fixture["records"]["mm_positive"]
    )
    first = provider.attest_record_order([negative, positive], phase="train")
    second = provider.attest_record_order([positive, negative], phase="train")
    assert first.record_sequence_sha256 != second.record_sequence_sha256
    assert all(isinstance(item, VerifiedPair) for item in first.verified_pairs)


def test_validation_phase_accepts_only_full_validation_split_records(
    provider_fixture,
):
    provider = provider_fixture["make_provider"]()
    val_record = provider_fixture["records"]["eccv_val"]
    prepared = provider.prepare([val_record], arm=_arm(), phase="validation")
    assert prepared.payload.labels.tolist() == [1.0]
    with pytest.raises(C0CoarseProviderError, match="requested phase"):
        provider.attest_record_order([val_record], phase="train")


def test_freeze_lock_size_mismatch_and_receipt_tampering_fail_before_decode(
    provider_fixture,
):
    receipt = json.loads(json.dumps(provider_fixture["receipt"]))
    receipt["locks"]["archives"]["mm_augmented"]["bytes"] += 1
    unsigned = dict(receipt)
    unsigned.pop("content_sha256")
    receipt["content_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with pytest.raises(C0CoarseProviderError, match="run-plan binding"):
        C0CoarseProvider(
            identity_index=provider_fixture["index"],
            mask_loader=provider_fixture["loader"],
            freeze_receipt=receipt,
            expected_freeze_content_sha256=provider_fixture["receipt"][
                "content_sha256"
            ],
            expected_identity_index_content_sha256=(
                provider_fixture["index"].content_sha256
            ),
        )
    with pytest.raises(C0CoarseProviderError, match="SHA/size"):
        C0CoarseProvider(
            identity_index=provider_fixture["index"],
            mask_loader=provider_fixture["loader"],
            freeze_receipt=receipt,
            expected_freeze_content_sha256=receipt["content_sha256"],
            expected_identity_index_content_sha256=(
                provider_fixture["index"].content_sha256
            ),
        )
    assert provider_fixture["loader"].stats.decoded_masks == 0

    tampered = json.loads(json.dumps(provider_fixture["receipt"]))
    tampered["scope"]["sealed_real_read"] = True
    with pytest.raises(C0CoarseProviderError, match="content digest"):
        C0CoarseProvider(
            identity_index=provider_fixture["index"],
            mask_loader=provider_fixture["loader"],
            freeze_receipt=tampered,
            expected_freeze_content_sha256=provider_fixture["receipt"][
                "content_sha256"
            ],
            expected_identity_index_content_sha256=(
                provider_fixture["index"].content_sha256
            ),
        )


def test_provider_rejects_loader_that_already_read_a_member(provider_fixture):
    verified = provider_fixture["index"].verify_record(
        provider_fixture["records"]["mm_negative"]
    )
    provider_fixture["loader"](verified.record.fragment_a)
    with pytest.raises(C0CoarseProviderError, match="precede every member"):
        provider_fixture["make_provider"]()


def test_forged_identity_sealed_real_and_polarity_fail_closed(
    provider_fixture,
):
    provider = provider_fixture["make_provider"]()
    record = provider_fixture["records"]["mm_negative"]
    verified = provider_fixture["index"].verify_record(record)
    forged = replace(
        verified,
        fragment_a=replace(verified.fragment_a, content_sha256="0" * 64),
    )
    with pytest.raises(C0CoarseProviderError, match="not canonical"):
        provider.attest_record_order([forged], phase="train")

    sealed = replace(record, provenance={"real_dunhuang_sealed_test": True})
    with pytest.raises(C0CoarseProviderError, match="sealed"):
        provider.attest_record_order([sealed], phase="train")

    grayscale_a = replace(record.fragment_a, threshold_rule="grayscale_uint8_gt_127")
    grayscale = replace(record, fragment_a=grayscale_a)
    with pytest.raises(C0CoarseProviderError, match="polarity"):
        provider.attest_record_order([grayscale], phase="train")


@pytest.mark.parametrize(
    ("record_name", "message"),
    (("mm_empty", "empty"), ("mm_full", "polarity")),
)
def test_empty_and_unidentifiable_polarity_masks_taint_provider_and_block_receipt(
    provider_fixture,
    record_name,
    message,
):
    provider = provider_fixture["make_provider"]()
    with pytest.raises(C0CoarseProviderError, match=message):
        provider.prepare(
            [provider_fixture["records"][record_name]],
            arm=_arm(),
            phase="train",
        )
    with pytest.raises(C0CoarseProviderError, match="tainted"):
        provider.receipt()


@pytest.mark.parametrize(
    "bad_tensor",
    (
        torch.zeros(1, 127, 128, dtype=torch.float32),
        torch.full((1, 128, 128), float("nan"), dtype=torch.float32),
        torch.ones(1, 128, 128, dtype=torch.float32),
    ),
)
def test_preprocessing_shape_numeric_and_letterbox_contracts_fail_closed(
    provider_fixture,
    monkeypatch,
    bad_tensor,
):
    provider = provider_fixture["make_provider"]()
    monkeypatch.setattr(
        provider_module,
        "preprocess_coarse_mask",
        lambda _mask, _config: bad_tensor,
    )
    with pytest.raises(C0CoarseProviderError):
        provider.prepare(
            [provider_fixture["records"]["mm_negative"]],
            arm=_arm(),
            phase="train",
        )
    with pytest.raises(C0CoarseProviderError, match="tainted"):
        provider.receipt()


def test_letterbox_validator_uses_allocated_frame_not_resampled_support():
    # The four one-pixel tendrils make the tight bbox 400x400, hence the
    # canonical allocated frame is exactly 112x112.  Bilinear downsampling
    # misses those sparse extrema and leaves only a 56x56 nonzero support.
    component = np.zeros((400, 400), dtype=np.bool_)
    component[100:300, 100:300] = True
    component[:101, 200] = True
    component[299:, 200] = True
    component[200, :101] = True
    component[200, 299:] = True
    selected = provider_module._select_largest_four_connected(component)
    coarse = provider_module.preprocess_coarse_mask(
        selected, provider_module._exact_preprocessing()[0]
    )
    support = torch.nonzero(coarse[0] > 0.0, as_tuple=False)
    support_extent = support.max(dim=0).values - support.min(dim=0).values + 1

    assert support_extent.tolist() == [56, 56]
    assert provider_module._expected_c0_letterbox_frame(selected) == (8, 120, 8, 120)
    assert torch.equal(
        provider_module._validate_coarse_tensor(coarse, selected), coarse
    )


def test_letterbox_validator_rejects_values_outside_source_bound_frame():
    component = np.zeros((400, 200), dtype=np.bool_)
    component[1:399, 1:199] = True
    selected = provider_module._select_largest_four_connected(component)
    coarse = provider_module.preprocess_coarse_mask(
        selected, provider_module._exact_preprocessing()[0]
    )
    assert provider_module._expected_c0_letterbox_frame(selected) == (8, 120, 36, 92)

    outside = coarse.clone()
    outside[0, 64, 35] = 0.25
    with pytest.raises(C0CoarseProviderError, match="outside.*letterbox frame"):
        provider_module._validate_coarse_tensor(outside, selected)

    one_ulp_high = coarse.clone()
    one_ulp_high[0, 64, 64] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    with pytest.raises(C0CoarseProviderError, match=r"\[0,1\]"):
        provider_module._validate_coarse_tensor(one_ulp_high, selected)


def test_wrong_arm_and_batch_bounds_fail_without_member_access(provider_fixture):
    provider = provider_fixture["make_provider"](
        C0CoarseProviderConfig(
            max_batch_size=1,
            max_cached_fragments=2,
            max_cache_bytes=2 * 128 * 128 * 4,
        )
    )
    local_arm = AblationArm(
        name=AblationArmName.LOCAL_DUAL_SOFTMAX,
        evidence=EvidenceMode.LOCAL,
        matcher_mode=MatcherMode.DUAL_SOFTMAX.value,
        model_config={
            "matcher_mode": MatcherMode.DUAL_SOFTMAX.value,
            "arc_pooling": {
                "mode": "log_mean_exp",
                "temperature": 0.25,
                "top_k": 3,
            },
        },
        optimizer_config={"name": "AdamW"},
        aggregation_config={"evidence": "local"},
        arc_pooling=ArcPoolingConfig(),
    )
    record = provider_fixture["records"]["mm_negative"]
    with pytest.raises(C0CoarseProviderError, match="coarse-only"):
        provider.prepare([record], arm=local_arm, phase="train")
    with pytest.raises(C0CoarseProviderError, match="batch exceeds"):
        provider.prepare([record, record], arm=_arm(), phase="train")
    assert provider_fixture["loader"].stats.requests == 0
