import hashlib
import json
from dataclasses import replace

import pytest

from staging.pairwise_v0_2.pairwise_data.training_stream import ArchiveBinding
from staging.pairwise_v0_2.training.data_lock import (
    ExperimentDataLock,
    ExperimentDataLockError,
    build_experiment_data_lock,
    require_experiment_data_lock,
    require_locked_artifacts,
)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _fixtures(tmp_path):
    manifest = tmp_path / "synthetic_groups.jsonl"
    manifest.write_bytes(b'{"group_id":"fixture"}\n')
    manifest_sha = _sha(manifest.read_bytes())
    bindings = (
        ArchiveBinding("canonical://fixture/a", "zip", "a" * 64),
        ArchiveBinding("canonical://fixture/b", "tar", "b" * 64),
    )
    archives = [
        {
            "logical_id": binding.logical_id,
            "format": binding.archive_format,
            "sha256": binding.sha256,
        }
        for binding in bindings
    ]
    split_receipt = tmp_path / "split_receipt.json"
    split_receipt.write_text(
        json.dumps(
            {
                "upstream_artifacts": {
                    "synthetic_manifest": {"sha256": manifest_sha},
                    "archives": archives,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    stream_audit = tmp_path / "stream_audit.json"
    stream_audit.write_text(
        json.dumps(
            {"inputs": {"synthetic_manifest": {"sha256": manifest_sha}}},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return split_receipt, stream_audit, manifest, bindings


def _build(tmp_path):
    split_receipt, stream_audit, manifest, bindings = _fixtures(tmp_path)
    lock = build_experiment_data_lock(
        split_receipt_path=split_receipt,
        stream_audit_path=stream_audit,
        synthetic_manifest_path=manifest,
        archive_bindings=bindings,
    )
    return lock, split_receipt, stream_audit, manifest, bindings


def test_data_lock_is_exact_portable_and_round_trips(tmp_path):
    lock, split_receipt, stream_audit, manifest, bindings = _build(tmp_path)
    payload = lock.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert str(tmp_path) not in serialized
    assert "file://" not in serialized
    assert payload["artifacts"]["split_receipt"]["sha256"] == _sha(
        split_receipt.read_bytes()
    )
    assert payload["artifacts"]["stream_audit"]["sha256"] == _sha(
        stream_audit.read_bytes()
    )
    assert payload["artifacts"]["synthetic_manifest"]["sha256"] == _sha(
        manifest.read_bytes()
    )
    assert [item["sha256"] for item in payload["archives"]] == [
        binding.sha256 for binding in bindings
    ]
    assert ExperimentDataLock.from_dict(payload) == lock
    assert require_locked_artifacts(
        required_lock=lock,
        split_receipt_path=split_receipt,
        stream_audit_path=stream_audit,
        synthetic_manifest_path=manifest,
        archive_bindings=bindings,
    ) == lock


def test_tampered_artifact_cannot_match_explicit_required_lock(tmp_path):
    lock, split_receipt, stream_audit, manifest, bindings = _build(tmp_path)
    manifest.write_bytes(manifest.read_bytes() + b"tamper\n")

    with pytest.raises(ExperimentDataLockError, match="does not bind"):
        require_locked_artifacts(
            required_lock=lock,
            split_receipt_path=split_receipt,
            stream_audit_path=stream_audit,
            synthetic_manifest_path=manifest,
            archive_bindings=bindings,
        )


def test_archive_hash_or_lock_digest_substitution_fails_closed(tmp_path):
    lock, split_receipt, stream_audit, manifest, bindings = _build(tmp_path)
    substituted_bindings = (
        ArchiveBinding(bindings[0].logical_id, "zip", "c" * 64),
        bindings[1],
    )
    with pytest.raises(ExperimentDataLockError, match="archive bindings"):
        require_locked_artifacts(
            required_lock=lock,
            split_receipt_path=split_receipt,
            stream_audit_path=stream_audit,
            synthetic_manifest_path=manifest,
            archive_bindings=substituted_bindings,
        )

    changed = replace(lock, stream_audit_sha256="d" * 64)
    with pytest.raises(ExperimentDataLockError, match="does not match"):
        require_experiment_data_lock(lock, changed)

    forged_payload = lock.to_dict()
    forged_payload["artifacts"]["stream_audit"]["sha256"] = "e" * 64
    with pytest.raises(ExperimentDataLockError, match="digest is inconsistent"):
        ExperimentDataLock.from_dict(forged_payload)


def test_data_lock_parser_rejects_file_locator_even_with_valid_self_shape(tmp_path):
    lock, *_ = _build(tmp_path)
    payload = lock.to_dict()
    payload["archives"][0]["logical_id"] = "file:///private/training.zip"

    with pytest.raises(ValueError, match="portable non-file locator"):
        ExperimentDataLock.from_dict(payload)
