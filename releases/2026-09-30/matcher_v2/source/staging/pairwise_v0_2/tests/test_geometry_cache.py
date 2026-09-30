import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from staging.pairwise_v0_2.geometry import CandidateBuilderConfig
from staging.pairwise_v0_2.training.geometry_cache import (
    GeometryArtifactCache,
    GeometryCacheConflictError,
    GeometryCacheCorruptionError,
    GeometryCacheError,
    GeometryCacheLimits,
    FrozenReceiptTrust,
    canonical_receipt_file_sha256,
    estimate_uncached_geometry_cost,
    fragment_cache_identity,
)


def _mask(offset=0):
    value = np.zeros((40, 44), dtype=bool)
    value[5 + offset : 30, 8:35] = True
    return value


def _trust(receipt):
    return FrozenReceiptTrust(
        expected_content_sha256=receipt["content_sha256"],
        expected_file_sha256=canonical_receipt_file_sha256(receipt),
    )


def test_fragment_key_binds_content_full_config_threshold_and_recipe():
    config = CandidateBuilderConfig(output_size=(12, 14))
    identity = fragment_cache_identity(_mask(), "grayscale_uint8_gt_127", config)
    assert identity == fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", config
    )
    assert (
        identity.key
        != fragment_cache_identity(_mask(1), "grayscale_uint8_gt_127", config).key
    )
    assert (
        identity.key
        != fragment_cache_identity(_mask(), "binary_brighter_value", config).key
    )
    assert (
        identity.key
        != fragment_cache_identity(
            _mask(),
            "grayscale_uint8_gt_127",
            replace(config, foreground_polarity="dark"),
        ).key
    )
    saddle_identity = fragment_cache_identity(
        _mask(),
        "grayscale_uint8_gt_127",
        replace(config, saddle_policy="reject"),
    )
    assert identity.geometry_version == "upright-facing-multirun-patches/v0.3"
    assert identity.geometry_config_sha256 != saddle_identity.geometry_config_sha256
    assert identity.key != saddle_identity.key
    # Pair IDs and filesystem paths are not accepted by the key API at all.
    assert set(identity.to_dict()) == {
        "canonical_mask_sha256",
        "threshold_rule",
        "geometry_config_sha256",
        "key",
        "cache_schema_version",
        "recipe_version",
        "geometry_version",
    }


def test_atomic_roundtrip_get_or_compute_and_concurrent_reads(tmp_path):
    cache = GeometryArtifactCache(tmp_path)
    identity = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    calls = []

    def produce():
        calls.append(True)
        return {
            "patches": np.arange(48, dtype=np.float32).reshape(2, 3, 2, 4),
            "valid": np.ones(2, dtype=bool),
        }, {"side": "left", "recipe": "fixture"}

    first = cache.get_or_compute(identity, produce)
    second = cache.get_or_compute(identity, produce)
    assert not first.cache_hit and second.cache_hit
    assert len(calls) == 1
    assert not second.artifact.arrays["patches"].flags.writeable
    assert np.array_equal(
        first.artifact.arrays["patches"], second.artifact.arrays["patches"]
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(
            pool.map(
                lambda _: cache.get(identity).logical_payload_sha256,
                range(32),
            )
        )
    assert len(set(values)) == 1
    assert not list(tmp_path.rglob("*.tmp"))
    receipt = cache.portable_receipt([identity])
    serialized = str(receipt)
    assert str(tmp_path) not in serialized
    assert receipt["artifact_count"] == 1


def test_tamper_and_size_bounds_fail_closed(tmp_path):
    identity = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    cache = GeometryArtifactCache(tmp_path)
    cache.put(identity, {"x": np.ones((4, 4), dtype=np.float32)})
    entry = tmp_path / identity.key[:2] / (identity.key + ".npz")
    entry.write_bytes(b"not-an-npz")
    with pytest.raises(GeometryCacheCorruptionError):
        cache.get(identity)

    bounded = GeometryArtifactCache(
        tmp_path / "bounded",
        GeometryCacheLimits(max_total_array_bytes=8),
    )
    with pytest.raises(Exception, match="byte bound"):
        bounded.put(identity, {"x": np.ones(4, dtype=np.float32)})


def test_entries_are_write_once_and_idempotent_payload_is_allowed(tmp_path):
    identity = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    cache = GeometryArtifactCache(tmp_path)
    first = cache.put(identity, {"x": np.ones((3, 4), dtype=np.float32)})
    same = cache.put(identity, {"x": np.ones((3, 4), dtype=np.float32)})
    assert same.logical_payload_sha256 == first.logical_payload_sha256
    with pytest.raises(GeometryCacheConflictError, match="different logical payload"):
        cache.put(identity, {"x": np.full((3, 4), 42.0, dtype=np.float32)})
    assert np.array_equal(cache.get(identity).arrays["x"], np.ones((3, 4)))


def test_read_side_enforces_metadata_element_and_byte_limits(tmp_path):
    identity = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    root = tmp_path / "metadata"
    GeometryArtifactCache(root).put(
        identity,
        {"x": np.ones((8, 8), dtype=np.float32)},
        {"description": "x" * 512},
    )
    metadata_limited = GeometryArtifactCache(
        root, GeometryCacheLimits(max_metadata_bytes=64)
    )
    with pytest.raises(GeometryCacheCorruptionError, match="metadata byte bound"):
        metadata_limited.get(identity)

    root = tmp_path / "arrays"
    GeometryArtifactCache(root).put(identity, {"x": np.ones((8, 8), dtype=np.float32)})
    array_limited = GeometryArtifactCache(
        root,
        GeometryCacheLimits(
            max_array_elements=1,
            max_total_array_bytes=1,
        ),
    )
    with pytest.raises(GeometryCacheCorruptionError, match="bound"):
        array_limited.get(identity)


def test_quota_ledger_avoids_tree_scan_per_put(tmp_path, monkeypatch):
    cache = GeometryArtifactCache(tmp_path)
    first = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    second = fragment_cache_identity(
        _mask(1), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )

    def forbidden_scan():
        raise AssertionError("put performed an O(N) cache scan")

    monkeypatch.setattr(cache, "_scan_cache_unlocked", forbidden_scan)
    cache.put(first, {"x": np.ones(2, dtype=np.float32)})
    cache.put(second, {"x": np.ones(3, dtype=np.float32)})
    assert cache.portable_receipt([first, second])["artifact_count"] == 2


def test_concurrent_first_miss_invokes_producer_once(tmp_path):
    identity = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    cache = GeometryArtifactCache(tmp_path)
    calls = []

    def produce():
        calls.append(True)
        return {"x": np.ones((4, 4), dtype=np.float32)}, {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(
            pool.map(lambda _: cache.get_or_compute(identity, produce), range(8))
        )
    assert len(calls) == 1
    assert sum(not output.cache_hit for output in outputs) == 1
    assert len({output.artifact.logical_payload_sha256 for output in outputs}) == 1


def test_read_only_mode_requires_precomputed_entries_and_never_calls_producer(
    tmp_path,
):
    first = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    missing = fragment_cache_identity(
        _mask(1), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    GeometryArtifactCache(tmp_path).put(first, {"x": np.ones((2, 3), dtype=np.float32)})
    frozen = GeometryArtifactCache(tmp_path, read_only=True)
    called = []

    def forbidden_producer():
        called.append(True)
        return {"x": np.zeros(1, dtype=np.float32)}, {}

    hit = frozen.get_or_compute(first, forbidden_producer)
    assert hit.cache_hit and not called
    with pytest.raises(GeometryCacheError, match="read-only cache"):
        frozen.get_or_compute(missing, forbidden_producer)
    with pytest.raises(GeometryCacheError, match="read-only cache"):
        frozen.put(missing, {"x": np.ones(1, dtype=np.float32)})
    assert not called
    assert frozen.portable_receipt([first])["cache_mode"] == "read_only_unbound"


def test_frozen_receipt_binds_exact_inventory_and_logical_payload(tmp_path):
    first = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    second = fragment_cache_identity(
        _mask(1), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    trusted_root = tmp_path / "trusted"
    trusted = GeometryArtifactCache(trusted_root)
    trusted.put(first, {"x": np.ones((2, 3), dtype=np.float32)})
    receipt = trusted.portable_receipt([first])
    trust = _trust(receipt)
    frozen = GeometryArtifactCache.from_frozen_receipt(
        trusted_root, receipt, trust=trust
    )
    assert np.array_equal(frozen.get(first).arrays["x"], np.ones((2, 3)))
    assert frozen.portable_receipt([first])["cache_mode"] == ("read_only_frozen_bound")
    assert frozen.frozen_receipt_trust is trust

    tampered_receipt = dict(receipt)
    tampered_receipt["artifact_count"] = 2
    with pytest.raises(GeometryCacheError, match="file hash mismatch"):
        GeometryArtifactCache.from_frozen_receipt(
            trusted_root, tampered_receipt, trust=trust
        )

    malicious_root = tmp_path / "malicious"
    malicious = GeometryArtifactCache(malicious_root)
    malicious.put(first, {"x": np.full((2, 3), 0.5, dtype=np.float32)})
    with pytest.raises(
        GeometryCacheCorruptionError, match="differs from frozen receipt"
    ):
        GeometryArtifactCache.from_frozen_receipt(malicious_root, receipt, trust=trust)

    extra_root = tmp_path / "extra"
    extra = GeometryArtifactCache(extra_root)
    extra.put(first, {"x": np.ones((2, 3), dtype=np.float32)})
    extra.put(second, {"x": np.ones((2, 3), dtype=np.float32)})
    with pytest.raises(GeometryCacheError, match="inventory differs"):
        GeometryArtifactCache.from_frozen_receipt(extra_root, receipt, trust=trust)


def test_frozen_receipt_requires_external_hashes_and_eagerly_checks_array_summary(
    tmp_path,
):
    identity = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    cache = GeometryArtifactCache(tmp_path)
    cache.put(identity, {"x": np.ones((2, 3), dtype=np.float32)})
    receipt = cache.portable_receipt([identity])
    with pytest.raises(TypeError, match="trust"):
        GeometryArtifactCache.from_frozen_receipt(tmp_path, receipt)

    resigned = json.loads(json.dumps(receipt))
    resigned["artifacts"][0]["array_bytes"] += 4
    body = dict(resigned)
    body.pop("content_sha256")
    resigned["content_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(GeometryCacheCorruptionError, match="counts or bytes"):
        GeometryArtifactCache.from_frozen_receipt(
            tmp_path, resigned, trust=_trust(resigned)
        )


def test_portable_receipt_rejects_duplicate_identity_and_frozen_roundtrips(tmp_path):
    identity = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    cache = GeometryArtifactCache(tmp_path)
    cache.put(identity, {"x": np.ones((2, 3), dtype=np.float32)})
    with pytest.raises(GeometryCacheError, match="repeat"):
        cache.portable_receipt([identity, identity])
    receipt = cache.portable_receipt([identity])
    frozen = GeometryArtifactCache.from_frozen_receipt(
        tmp_path, receipt, trust=_trust(receipt)
    )
    roundtrip = frozen.portable_receipt([identity])
    reopened = GeometryArtifactCache.from_frozen_receipt(
        tmp_path, roundtrip, trust=_trust(roundtrip)
    )
    assert reopened.get(identity).logical_payload_sha256 == (
        cache.get(identity).logical_payload_sha256
    )


def test_returned_metadata_and_arrays_are_deeply_immutable(tmp_path):
    identity = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    artifact = GeometryArtifactCache(tmp_path).put(
        identity,
        {"x": np.ones((2, 3), dtype=np.float32)},
        {"nested": {"items": [{"value": 7}]}},
    )
    with pytest.raises(TypeError):
        artifact.metadata["nested"]["items"][0]["value"] = 8
    with pytest.raises(ValueError):
        artifact.arrays["x"].setflags(write=True)
    detached = artifact.metadata_dict()
    detached["nested"]["items"][0]["value"] = 8
    assert artifact.metadata["nested"]["items"][0]["value"] == 7
    artifact.verify_logical_payload()


def test_cache_open_reconciles_quota_and_never_trusts_lower_ledger(
    tmp_path,
):
    first = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    second = fragment_cache_identity(
        _mask(1), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    cache = GeometryArtifactCache(tmp_path)
    cache.put(first, {"x": np.ones(64, dtype=np.float32)})
    actual = sum(path.stat().st_size for path in tmp_path.rglob("*.npz"))
    with pytest.raises(GeometryCacheError, match="existing cache exceeds"):
        GeometryArtifactCache(
            tmp_path,
            GeometryCacheLimits(max_cache_file_bytes=actual - 1),
            read_only=True,
        )

    ledger_path = tmp_path / ".quota-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["total_file_bytes"] = 0
    ledger["entry_count"] = 0
    ledger_path.write_text(
        json.dumps(ledger, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    cache.put(second, {"x": np.ones(32, dtype=np.float32)})
    repaired = json.loads(ledger_path.read_text(encoding="utf-8"))
    observed = sum(path.stat().st_size for path in tmp_path.rglob("*.npz"))
    assert repaired["total_file_bytes"] >= observed
    assert repaired["entry_count"] >= 2


def test_npy_headers_and_central_sizes_are_checked_before_payload_decode(
    tmp_path, monkeypatch
):
    identity = fragment_cache_identity(
        _mask(), "grayscale_uint8_gt_127", CandidateBuilderConfig()
    )
    root = tmp_path / "cache"
    GeometryArtifactCache(root).put(
        identity,
        {"x": np.ones((8, 8), dtype=np.float32)},
        {"description": "x" * 512},
    )
    from staging.pairwise_v0_2.training import geometry_cache as module

    def forbidden_payload(*args, **kwargs):
        raise AssertionError("payload decompression occurred before header gates")

    monkeypatch.setattr(module, "_read_npy_payload", forbidden_payload)
    limited = GeometryArtifactCache(root, GeometryCacheLimits(max_metadata_bytes=64))
    with pytest.raises(GeometryCacheCorruptionError, match="metadata byte bound"):
        limited.get(identity)

    array_limited = GeometryArtifactCache(
        root,
        GeometryCacheLimits(
            max_metadata_bytes=4096,
            max_array_elements=1,
            max_total_array_bytes=1,
        ),
    )
    with pytest.raises(GeometryCacheCorruptionError, match="bound"):
        array_limited.get(identity)


def test_cost_estimator_proves_per_epoch_pair_preprocessing_is_infeasible():
    one = estimate_uncached_geometry_cost(1_291_809, 1)
    twenty = estimate_uncached_geometry_cost(1_291_809, 20)
    assert one.uncached_days_low == pytest.approx(44.854479, rel=1e-5)
    assert one.uncached_days_high == pytest.approx(74.757465, rel=1e-5)
    assert twenty.uncached_days_low > 897.0
    assert twenty.uncached_days_high > 1495.0
