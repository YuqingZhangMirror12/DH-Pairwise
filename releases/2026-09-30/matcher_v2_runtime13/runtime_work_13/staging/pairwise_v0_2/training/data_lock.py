"""Portable, fail-closed data identity lock for Pairwise v0.2 experiments.

The lock contains digests and portable archive identities only.  Runtime paths
are inputs to verification and are never serialized.  A runner must possess an
explicit expected lock and call :func:`require_locked_artifacts` before it may
construct train/validation streams.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from staging.pairwise_v0_2.pairwise_data.training_stream import ArchiveBinding


DATA_LOCK_SCHEMA_VERSION = "dunhuang-pairwise-experiment-data-lock/0.2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_ARTIFACT_ROLES = (
    "split_receipt",
    "stream_audit",
    "synthetic_manifest",
)


class ExperimentDataLockError(ValueError):
    """Raised when a runner cannot prove its exact data identity."""


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ExperimentDataLockError(
            "{} must be 64 lowercase hex characters".format(name)
        )


def _sha256_path(path: Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise ExperimentDataLockError("data-lock artifact must be a regular file")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_json_object(path: Path, role: str) -> Mapping[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentDataLockError("{} is not valid JSON".format(role)) from exc
    if not isinstance(value, Mapping):
        raise ExperimentDataLockError("{} root must be an object".format(role))
    return value


def _nested_sha256(value: Mapping[str, Any], keys: Sequence[str], role: str) -> str:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ExperimentDataLockError(
                "{} lacks required digest binding".format(role)
            )
        current = current[key]
    _validate_sha256(current, "{} linked SHA-256".format(role))
    return current


@dataclass(frozen=True, order=True)
class LockedArchive:
    """Portable archive identity embedded in an experiment data lock."""

    logical_id: str
    archive_format: str
    sha256: str

    def __post_init__(self) -> None:
        # Reuse the authoritative binding validation, including the file/path
        # locator prohibition.
        ArchiveBinding(self.logical_id, self.archive_format, self.sha256)

    @classmethod
    def from_binding(cls, binding: ArchiveBinding) -> "LockedArchive":
        if not isinstance(binding, ArchiveBinding):
            raise TypeError("archive bindings must be ArchiveBinding instances")
        return cls(binding.logical_id, binding.archive_format, binding.sha256)

    def to_dict(self) -> Dict[str, str]:
        return {
            "logical_id": self.logical_id,
            "format": self.archive_format,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ExperimentDataLock:
    """Exact portable identity of all Pairwise v0.2 training inputs."""

    split_receipt_sha256: str
    stream_audit_sha256: str
    synthetic_manifest_sha256: str
    archives: Tuple[LockedArchive, ...]
    schema_version: str = DATA_LOCK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATA_LOCK_SCHEMA_VERSION:
            raise ExperimentDataLockError("unsupported experiment data-lock schema")
        for role, digest in (
            ("split_receipt", self.split_receipt_sha256),
            ("stream_audit", self.stream_audit_sha256),
            ("synthetic_manifest", self.synthetic_manifest_sha256),
        ):
            _validate_sha256(digest, role + " SHA-256")
        archives = tuple(self.archives)
        if not archives:
            raise ExperimentDataLockError("at least one archive binding is required")
        if not all(isinstance(item, LockedArchive) for item in archives):
            raise TypeError("archives must contain LockedArchive instances")
        canonical = tuple(sorted(archives))
        if archives != canonical:
            raise ExperimentDataLockError("archive bindings must be canonically sorted")
        logical_ids = [item.logical_id for item in archives]
        if len(logical_ids) != len(set(logical_ids)):
            raise ExperimentDataLockError("archive logical IDs must be unique")
        object.__setattr__(self, "archives", archives)

    def _payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifacts": {
                "split_receipt": {"sha256": self.split_receipt_sha256},
                "stream_audit": {"sha256": self.stream_audit_sha256},
                "synthetic_manifest": {
                    "sha256": self.synthetic_manifest_sha256
                },
            },
            "archives": [archive.to_dict() for archive in self.archives],
        }

    @property
    def lock_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self._payload())).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        value = self._payload()
        value["lock_sha256"] = self.lock_sha256
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentDataLock":
        if not isinstance(value, Mapping):
            raise ExperimentDataLockError("experiment data lock must be an object")
        if set(value) != {"schema_version", "artifacts", "archives", "lock_sha256"}:
            raise ExperimentDataLockError("experiment data lock has unexpected fields")
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(_ARTIFACT_ROLES):
            raise ExperimentDataLockError("experiment data lock artifacts are invalid")

        def artifact_sha(role: str) -> str:
            artifact = artifacts[role]
            if not isinstance(artifact, Mapping) or set(artifact) != {"sha256"}:
                raise ExperimentDataLockError("locked artifact entry is invalid")
            return artifact["sha256"]

        archive_values = value.get("archives")
        if not isinstance(archive_values, list):
            raise ExperimentDataLockError("locked archives must be a list")
        archives = []
        for item in archive_values:
            if not isinstance(item, Mapping) or set(item) != {
                "logical_id",
                "format",
                "sha256",
            }:
                raise ExperimentDataLockError("locked archive entry is invalid")
            archives.append(
                LockedArchive(
                    str(item["logical_id"]),
                    str(item["format"]),
                    str(item["sha256"]),
                )
            )
        lock = cls(
            split_receipt_sha256=artifact_sha("split_receipt"),
            stream_audit_sha256=artifact_sha("stream_audit"),
            synthetic_manifest_sha256=artifact_sha("synthetic_manifest"),
            archives=tuple(archives),
            schema_version=value.get("schema_version"),
        )
        claimed = value.get("lock_sha256")
        _validate_sha256(claimed, "lock SHA-256")
        if not hmac.compare_digest(lock.lock_sha256, claimed):
            raise ExperimentDataLockError("experiment data-lock digest is inconsistent")
        return lock


def build_experiment_data_lock(
    *,
    split_receipt_path: Path,
    stream_audit_path: Path,
    synthetic_manifest_path: Path,
    archive_bindings: Iterable[ArchiveBinding],
) -> ExperimentDataLock:
    """Observe artifacts and construct a portable internally linked lock."""

    split_receipt_sha = _sha256_path(Path(split_receipt_path))
    stream_audit_sha = _sha256_path(Path(stream_audit_path))
    synthetic_manifest_sha = _sha256_path(Path(synthetic_manifest_path))
    split_receipt = _load_json_object(Path(split_receipt_path), "split receipt")
    stream_audit = _load_json_object(Path(stream_audit_path), "stream audit")
    if _nested_sha256(
        split_receipt,
        ("upstream_artifacts", "synthetic_manifest", "sha256"),
        "split receipt",
    ) != synthetic_manifest_sha:
        raise ExperimentDataLockError(
            "split receipt does not bind the observed synthetic manifest"
        )
    if _nested_sha256(
        stream_audit,
        ("inputs", "synthetic_manifest", "sha256"),
        "stream audit",
    ) != synthetic_manifest_sha:
        raise ExperimentDataLockError(
            "stream audit does not bind the observed synthetic manifest"
        )

    locked_archives = tuple(
        sorted(LockedArchive.from_binding(binding) for binding in archive_bindings)
    )
    upstream = split_receipt.get("upstream_artifacts")
    declared_values = upstream.get("archives") if isinstance(upstream, Mapping) else None
    if not isinstance(declared_values, list):
        raise ExperimentDataLockError("split receipt lacks archive bindings")
    try:
        declared_archives = tuple(
            sorted(
                LockedArchive(
                    str(item["logical_id"]),
                    str(item["format"]),
                    str(item["sha256"]),
                )
                for item in declared_values
                if isinstance(item, Mapping)
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentDataLockError(
            "split receipt archive binding is malformed"
        ) from exc
    if len(declared_archives) != len(declared_values):
        raise ExperimentDataLockError("split receipt archive binding is malformed")
    if declared_archives != locked_archives:
        raise ExperimentDataLockError(
            "runtime archive bindings do not match the split receipt"
        )
    return ExperimentDataLock(
        split_receipt_sha256=split_receipt_sha,
        stream_audit_sha256=stream_audit_sha,
        synthetic_manifest_sha256=synthetic_manifest_sha,
        archives=locked_archives,
    )


def require_experiment_data_lock(
    required_lock: ExperimentDataLock,
    observed_lock: ExperimentDataLock,
) -> ExperimentDataLock:
    """Fail unless a runner's observed lock exactly matches its required lock."""

    if not isinstance(required_lock, ExperimentDataLock):
        raise TypeError("required_lock must be an ExperimentDataLock")
    if not isinstance(observed_lock, ExperimentDataLock):
        raise TypeError("observed_lock must be an ExperimentDataLock")
    if not hmac.compare_digest(required_lock.lock_sha256, observed_lock.lock_sha256):
        raise ExperimentDataLockError("observed experiment data lock does not match")
    if required_lock.to_dict() != observed_lock.to_dict():
        raise ExperimentDataLockError("observed experiment data lock does not match")
    return observed_lock


def require_locked_artifacts(
    *,
    required_lock: ExperimentDataLock,
    split_receipt_path: Path,
    stream_audit_path: Path,
    synthetic_manifest_path: Path,
    archive_bindings: Iterable[ArchiveBinding],
) -> ExperimentDataLock:
    """Observe runtime inputs and require an explicit exact lock match."""

    observed = build_experiment_data_lock(
        split_receipt_path=split_receipt_path,
        stream_audit_path=stream_audit_path,
        synthetic_manifest_path=synthetic_manifest_path,
        archive_bindings=archive_bindings,
    )
    return require_experiment_data_lock(required_lock, observed)


__all__ = [
    "DATA_LOCK_SCHEMA_VERSION",
    "ExperimentDataLock",
    "ExperimentDataLockError",
    "LockedArchive",
    "build_experiment_data_lock",
    "require_experiment_data_lock",
    "require_locked_artifacts",
]
