"""Dedicated, fail-closed C0-N-Q1 training runner.

The production contract is intentionally narrow: one frozen 16,384-row
historical training selection, five epochs, and a complete source-ordered
273,046-row validation replay after every epoch.  This module has no API for
historical test or sealed-real records.
"""

from __future__ import annotations

import hmac
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import nn

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    HISTORICAL_TEST_ACCESS_EVIDENCE,
    IDENTITY_INDEX_CONTENT_SCHEMA,
)
from staging.pairwise_v0_2.pairwise_data.sampling import (
    validation_stream_fingerprint,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.training.c0_coarse_provider import (
    C0_PRODUCTION_MAX_CACHE_BYTES,
    C0_PRODUCTION_MAX_CACHED_FRAGMENTS,
)
from staging.pairwise_v0_2.training.checkpoint import (
    canonical_config_hash,
    load_trusted_checkpoint,
    save_checkpoint,
)
from staging.pairwise_v0_2.training.evaluation import (
    evaluate_pairwise,
    fit_pairwise_threshold,
)


PRODUCTION_VALIDATION_FINGERPRINT = (
    "23c7c7446adbddf205ff815240973d2d6e1e89c1c77f27f5a2330b11ccd51078"
)
PRODUCTION_FREEZE_FILE_SHA256 = (
    "eb9b8bcab71661c42aab73b1cb9d117c15af145bf5e0a7b897115824ce4a44c2"
)
PRODUCTION_FREEZE_CONTENT_SHA256 = (
    "19950f4f8d20f556de819bc558c59746c02db32f769ce9f54bfb96d82b133e6c"
)
PRODUCTION_RUN_PLAN_FILE_SHA256 = (
    "bd1b9b96fb686241fe560c0e001d7057833f7781a0f9b13e72b700515b0c0cae"
)
PRODUCTION_RUN_PLAN_CONTENT_SHA256 = (
    "acacf7d3611e551750cb5f859f805fba78b15f31f185a396cab3111e9ef5b94e"
)
RUNNER_SCHEMA_VERSION = "dunhuang-pairwise-c0-run/0.3"
FREEZE_SCHEMA_VERSION = "dunhuang-pairwise-c0-n-q1-freeze/0.3"
RUN_PLAN_SCHEMA_VERSION = "dunhuang-pairwise-c0-production-run-plan/0.3"
RUNNER_ANCHOR_NORMALIZED_HASH_MODE = "python_c0_plan_anchor_normalized_v1"
SOURCE_BUNDLE_HASH_MODE = "python_source_bundle_manifest_v1"
_SHA256_CHARS = frozenset("0123456789abcdef")
_DATASETS = ("mm_augmented", "eccv_1113data")
_RUNTIME_ENVIRONMENT_SCHEMA_VERSION = "dunhuang-pairwise-c0-runtime-environment/0.4"
_CUDA_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})


def _role(
    logical_id: str, kind: str, hash_mode: str = "sha256_bytes"
) -> Mapping[str, str]:
    return MappingProxyType(
        {"logical_id": logical_id, "kind": kind, "hash_mode": hash_mode}
    )


PRODUCTION_RUN_PLAN_ROLE_SPECS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "freeze_receipt": _role("artifact://c0-n-q1/freeze", "metadata"),
        "mm_archive": _role("canonical://mm_augmented/archive", "archive"),
        "eccv_archive": _role("canonical://eccv_1113data/archive", "archive"),
        "mm_fingerprint_cache": _role(
            "artifact://v0.1/mm-fingerprint-cache", "metadata"
        ),
        "eccv_fingerprint_cache": _role(
            "artifact://v0.1/eccv-fingerprint-cache", "metadata"
        ),
        "historical_split": _role("artifact://v0.1/expanded-056-split", "metadata"),
        "source_bundle": _role(
            "source://staging/pairwise-runtime-python",
            "source_bundle",
            SOURCE_BUNDLE_HASH_MODE,
        ),
    }
)


class C0RunnerError(RuntimeError):
    """Raised before or during C0 when an attestation cannot be proven."""


def _require_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not set(value).issubset(_SHA256_CHARS)
    ):
        raise C0RunnerError("{} must be lowercase SHA-256".format(name))
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _portable_string(value: str) -> bool:
    text = str(value)
    return not (
        text.startswith(("/", "~/", "file:", "\\\\"))
        or (len(text) >= 3 and text[1:3] in {":\\", ":/"})
    )


def _assert_portable(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in {
                "archive_member",
                "canonical_group_id",
                "component_id",
                "fragment_id",
                "pair_id",
                "path",
            }:
                raise C0RunnerError("portable receipt exposes identity/path field")
            _assert_portable(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_portable(item)
    elif isinstance(value, str) and not _portable_string(value):
        raise C0RunnerError("portable receipt contains a local path")


@dataclass(frozen=True)
class SourceFileLock:
    logical_id: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not self.logical_id or not _portable_string(self.logical_id):
            raise C0RunnerError("source lock logical_id must be portable")
        object.__setattr__(self, "path", Path(self.path))
        _require_sha256(self.sha256, "source lock SHA-256")


@dataclass(frozen=True)
class FrozenReceiptLock:
    path: Path
    file_sha256: str = PRODUCTION_FREEZE_FILE_SHA256
    content_sha256: str = PRODUCTION_FREEZE_CONTENT_SHA256

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        _require_sha256(self.file_sha256, "freeze file SHA-256")
        _require_sha256(self.content_sha256, "freeze content SHA-256")


@dataclass(frozen=True)
class ProductionRunPlanBinding:
    """Runtime paths only; all expected hashes live in the canonical plan."""

    plan_path: Path
    role_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_path", Path(self.plan_path))
        if not isinstance(self.role_paths, Mapping):
            raise C0RunnerError("production run-plan role_paths must be a mapping")
        normalized = {str(role): Path(path) for role, path in self.role_paths.items()}
        if set(normalized) != set(PRODUCTION_RUN_PLAN_ROLE_SPECS):
            raise C0RunnerError("production run-plan role path set is not exact")
        object.__setattr__(self, "role_paths", MappingProxyType(normalized))


@dataclass(frozen=True)
class C0RunnerContract:
    epochs: int = 5
    batch_size: int = 256
    train_count: int = 16_384
    train_per_dataset_label: int = 4_096
    validation_count: int = 273_046
    validation_fingerprint: str = PRODUCTION_VALIDATION_FINGERPRINT
    seed: str = "260828"
    source_locks: Tuple[SourceFileLock, ...] = ()
    production: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_locks", tuple(self.source_locks))
        if not self.seed:
            raise C0RunnerError("runner seed is required")
        _require_sha256(self.validation_fingerprint, "validation fingerprint")
        for name in (
            "epochs",
            "batch_size",
            "train_count",
            "train_per_dataset_label",
            "validation_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise C0RunnerError("{} must be a positive int".format(name))
        if len({lock.logical_id for lock in self.source_locks}) != len(
            self.source_locks
        ):
            raise C0RunnerError("source lock logical IDs must be unique")
        if self.production:
            expected = {
                "epochs": 5,
                "batch_size": 256,
                "train_count": 16_384,
                "train_per_dataset_label": 4_096,
                "validation_count": 273_046,
                "validation_fingerprint": PRODUCTION_VALIDATION_FINGERPRINT,
                "seed": "260828",
            }
            for name, value in expected.items():
                if getattr(self, name) != value:
                    raise C0RunnerError("production C0 constant changed: " + name)
            if self.source_locks:
                raise C0RunnerError(
                    "production C0 source hashes must come only from canonical plan"
                )

    @property
    def batches_per_epoch(self) -> int:
        if self.train_count % self.batch_size:
            raise C0RunnerError("train count must divide exactly into batches")
        return self.train_count // self.batch_size

    @property
    def total_steps(self) -> int:
        return self.epochs * self.batches_per_epoch

    def portable_dict(self) -> Mapping[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "train_count": self.train_count,
            "train_per_dataset_label": self.train_per_dataset_label,
            "validation_count": self.validation_count,
            "validation_fingerprint": self.validation_fingerprint,
            "seed_sha256": hashlib.sha256(self.seed.encode("utf-8")).hexdigest(),
            "batches_per_epoch": self.batches_per_epoch,
            "total_steps": self.total_steps,
            "production": self.production,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "C0RunnerContract":
        allowed = {
            "epochs",
            "batch_size",
            "train_count",
            "train_per_dataset_label",
            "validation_count",
            "validation_fingerprint",
            "seed",
            "source_locks",
            "production",
        }
        extra = set(value) - allowed
        if extra:
            raise C0RunnerError("unknown C0 run-plan field: " + sorted(extra)[0])
        locks = []
        for item in value.get("source_locks", ()):
            locks.append(
                item if isinstance(item, SourceFileLock) else SourceFileLock(**item)
            )
        fields = dict(value)
        fields["source_locks"] = tuple(locks)
        return cls(**fields)


@dataclass(frozen=True)
class ProviderAttestation:
    provider_version: str
    preprocessing_sha256: str
    archive_locks_sha256: str
    archives_verified: bool
    identity_index_content_sha256: Optional[str] = None
    mask_pixels_loaded_during_attestation: bool = False
    sealed_real_test_capability: bool = False

    def __post_init__(self) -> None:
        if not self.provider_version or not _portable_string(self.provider_version):
            raise C0RunnerError("provider version must be portable")
        _require_sha256(self.preprocessing_sha256, "preprocessing SHA-256")
        _require_sha256(self.archive_locks_sha256, "archive-lock SHA-256")
        if self.archives_verified is not True:
            raise C0RunnerError("provider did not attest archives")
        if self.identity_index_content_sha256 is not None:
            _require_sha256(
                self.identity_index_content_sha256,
                "provider identity-index content SHA-256",
            )
        if self.mask_pixels_loaded_during_attestation:
            raise C0RunnerError("provider loaded pixels during archive attestation")
        if self.sealed_real_test_capability:
            raise C0RunnerError("provider exposes sealed-real capability")

    def portable_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendAttestation:
    backend_version: str
    model_family: str
    device_type: str
    deterministic_reload: bool = True
    sealed_real_test_capability: bool = False

    def __post_init__(self) -> None:
        if not self.backend_version or not self.model_family:
            raise C0RunnerError("backend identity is required")
        if self.device_type not in {"cpu", "cuda", "mps"}:
            raise C0RunnerError("unsupported backend device")
        if not self.deterministic_reload:
            raise C0RunnerError("backend does not attest deterministic reload")
        if self.sealed_real_test_capability:
            raise C0RunnerError("backend exposes sealed-real capability")

    def portable_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class C0PreparedBatch:
    payload: Any
    sample_count: int
    record_sequence_sha256: str
    prepared_input_sha256: str

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count <= 0:  # noqa: E721
            raise C0RunnerError("prepared batch count must be positive")
        _require_sha256(self.record_sequence_sha256, "batch sequence SHA-256")
        _require_sha256(self.prepared_input_sha256, "prepared input SHA-256")


@dataclass(frozen=True)
class C0TrainBatchResult:
    loss: float
    valid_count: int

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.loss)):
            raise C0RunnerError("training loss is non-finite")
        if type(self.valid_count) is not int or self.valid_count < 0:  # noqa: E721
            raise C0RunnerError("training valid_count is invalid")


@dataclass(frozen=True)
class C0PredictionBatch:
    probability: Tuple[float, ...]
    valid: Tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "probability", tuple(self.probability))
        object.__setattr__(self, "valid", tuple(self.valid))
        if len(self.probability) != len(self.valid) or not self.probability:
            raise C0RunnerError("prediction batch cardinality is invalid")
        if any(type(value) is not bool for value in self.valid):  # noqa: E721
            raise C0RunnerError("prediction validity must be explicit bool")
        for probability, valid in zip(self.probability, self.valid):
            if valid and (
                not math.isfinite(float(probability))
                or not 0.0 <= float(probability) <= 1.0
            ):
                raise C0RunnerError("valid probability is outside [0, 1]")


class C0Provider(Protocol):
    attestation: ProviderAttestation

    def prepare(
        self, records: Sequence[TrainingPairRecord], *, phase: str
    ) -> C0PreparedBatch: ...

    def final_receipt(self) -> Mapping[str, Any]: ...


class C0Session(Protocol):
    model: nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    model_config: Mapping[str, Any]
    optimizer_config: Mapping[str, Any]

    def train_batch(self, batch: C0PreparedBatch) -> C0TrainBatchResult: ...

    def predict_batch(self, batch: C0PreparedBatch) -> C0PredictionBatch: ...


class C0Backend(Protocol):
    attestation: BackendAttestation

    def create_session(self, *, seed: int, purpose: str) -> C0Session: ...


@dataclass(frozen=True)
class C0RunArtifacts:
    run_directory: Path
    receipt_path: Path
    checkpoint_sidecar_path: Path
    receipt: Mapping[str, Any]


def _coerce_contract(
    value: Union[C0RunnerContract, Mapping[str, Any]],
) -> C0RunnerContract:
    if isinstance(value, C0RunnerContract):
        return value
    if not isinstance(value, Mapping):
        raise C0RunnerError("run_plan must be C0RunnerContract or mapping")
    return C0RunnerContract.from_mapping(value)


def _coerce_freeze_lock(
    value: Union[FrozenReceiptLock, Mapping[str, Any]],
) -> FrozenReceiptLock:
    if isinstance(value, FrozenReceiptLock):
        return value
    if not isinstance(value, Mapping):
        raise C0RunnerError("freeze_receipt must be a locked external file")
    try:
        return FrozenReceiptLock(**dict(value))
    except (TypeError, ValueError) as exc:
        raise C0RunnerError("invalid freeze receipt lock") from exc


def _load_and_verify_freeze(
    lock: FrozenReceiptLock, contract: C0RunnerContract
) -> Mapping[str, Any]:
    if not lock.path.is_file():
        raise C0RunnerError("freeze receipt path is not a regular file")
    observed_file = _sha256_file(lock.path)
    if not hmac.compare_digest(observed_file, lock.file_sha256):
        raise C0RunnerError("freeze receipt file SHA-256 mismatch")
    try:
        payload = json.loads(lock.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise C0RunnerError("freeze receipt is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise C0RunnerError("freeze receipt root must be an object")
    stored_content = payload.get("content_sha256")
    content_view = dict(payload)
    content_view.pop("content_sha256", None)
    observed_content = _content_sha256(content_view)
    if not isinstance(stored_content, str) or not hmac.compare_digest(
        observed_content, stored_content
    ):
        raise C0RunnerError("freeze receipt semantic content hash is inconsistent")
    if not hmac.compare_digest(observed_content, lock.content_sha256):
        raise C0RunnerError("freeze receipt semantic content lock mismatch")
    if payload.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise C0RunnerError("unsupported freeze receipt schema")
    if payload.get("status") != "pass_metadata_only_no_model_execution":
        raise C0RunnerError("freeze receipt status is not runnable")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise C0RunnerError("freeze receipt scope is missing")
    if set(scope) != {
        "experiment",
        "datasets",
        "pair_stream_splits_read",
        "historical_test_access",
        "sealed_real_read",
        "mask_pixels_decoded",
        "model_executed",
    }:
        raise C0RunnerError("freeze receipt scope schema changed")
    if scope.get("historical_test_access") != dict(HISTORICAL_TEST_ACCESS_EVIDENCE):
        raise C0RunnerError("freeze historical-test access evidence changed")
    if set(scope.get("pair_stream_splits_read", ())) != {"train", "val"}:
        raise C0RunnerError("freeze pair-stream split evidence changed")
    if any(
        scope.get(key) is not False
        for key in ("sealed_real_read", "mask_pixels_decoded", "model_executed")
    ):
        raise C0RunnerError("freeze receipt scope admits forbidden data/execution")
    validation = payload.get("validation")
    training = payload.get("training")
    overlap = payload.get("selected_train_vs_full_validation_overlap")
    if not isinstance(validation, Mapping) or not isinstance(training, Mapping):
        raise C0RunnerError("freeze receipt lacks train/validation evidence")
    selection = training.get("selection")
    if not isinstance(selection, Mapping):
        raise C0RunnerError("freeze receipt lacks selected training evidence")
    if validation.get("count") != contract.validation_count:
        raise C0RunnerError("freeze validation count differs from run contract")
    if validation.get("frozen_order_sha256") != contract.validation_fingerprint:
        raise C0RunnerError("freeze validation fingerprint differs from contract")
    if selection.get("count") != contract.train_count:
        raise C0RunnerError("freeze train count differs from run contract")
    if selection.get("target_per_dataset_label") != contract.train_per_dataset_label:
        raise C0RunnerError("freeze stratum target differs from run contract")
    if selection.get("max_per_component_label") != 32 and contract.production:
        raise C0RunnerError("production freeze component cap changed")
    for field in (
        "record_order_commitment_sha256",
        "record_set_commitment_sha256",
    ):
        _require_sha256(str(selection.get(field, "")), "freeze selection " + field)
    _require_sha256(
        str(validation.get("record_sequence_commitment_sha256", "")),
        "freeze validation record commitment",
    )
    if not isinstance(overlap, Mapping) or set(overlap) != {
        "component",
        "member",
        "content_sha256",
    }:
        raise C0RunnerError("freeze overlap evidence is missing")
    if any(value != 0 for value in overlap.values()):
        raise C0RunnerError("freeze reports train/validation overlap")
    locks = payload.get("locks")
    if not isinstance(locks, Mapping) or not isinstance(locks.get("archives"), Mapping):
        raise C0RunnerError("freeze archive locks are missing")
    _assert_portable(payload)
    return payload


def _verify_source_locks(
    locks: Sequence[SourceFileLock], *, production: bool
) -> Tuple[Mapping[str, Any], ...]:
    if production and not locks:
        raise C0RunnerError("production source locks are required")
    portable = []
    seen_paths = set()
    for lock in sorted(locks, key=lambda item: item.logical_id):
        resolved = lock.path.resolve()
        if resolved in seen_paths:
            raise C0RunnerError("the same source path has multiple lock identities")
        seen_paths.add(resolved)
        if not lock.path.is_file():
            raise C0RunnerError("locked source is not a regular file")
        observed = _sha256_file(lock.path)
        if not hmac.compare_digest(observed, lock.sha256):
            raise C0RunnerError("source lock mismatch: " + lock.logical_id)
        portable.append(
            {
                "logical_id": lock.logical_id,
                "bytes": lock.path.stat().st_size,
                "sha256": observed,
            }
        )
    return tuple(portable)


def _normalized_runner_anchor_bytes(path: Path) -> bytes:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise C0RunnerError("normalized runner source must be UTF-8") from exc
    names = (
        "PRODUCTION_RUN_PLAN_FILE_SHA256",
        "PRODUCTION_RUN_PLAN_CONTENT_SHA256",
    )
    for name in names:
        pattern = r'({}\s*=\s*(?:\(\s*)?")[0-9a-f]{{64}}(")'.format(name)
        text, count = re.subn(pattern, r"\g<1>" + ("0" * 64) + r"\g<2>", text)
        if count != 1:
            raise C0RunnerError("runner plan anchor normalization failed: " + name)
    return text.encode("utf-8")


def c0_source_bundle_manifest(root: Path) -> Mapping[str, Any]:
    """Return the exact non-test Python source bundle used by C0.

    Locking the full v0.1/v0.2 non-test Python trees avoids a hand-maintained
    import allowlist silently missing a transitive runtime dependency.  The
    runner source itself uses the narrowly normalized anchor mode documented
    above; every other source is hashed byte-for-byte.
    """

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise C0RunnerError("production source-bundle root is not a directory")
    files = []
    for package in ("pairwise_v0_1", "pairwise_v0_2"):
        package_root = root / package
        if package_root.is_symlink() or not package_root.is_dir():
            raise C0RunnerError("production source bundle lacks " + package)
        if any(candidate.is_symlink() for candidate in package_root.rglob("*")):
            raise C0RunnerError("production source bundle contains a symlink")
        for path in package_root.rglob("*.py"):
            if not path.is_file():
                raise C0RunnerError("production source bundle has a non-file source")
            relative = path.relative_to(root)
            if "tests" in relative.parts or "__pycache__" in relative.parts:
                continue
            relative_text = relative.as_posix()
            mode = (
                RUNNER_ANCHOR_NORMALIZED_HASH_MODE
                if relative_text == "pairwise_v0_2/training/c0_runner.py"
                else "sha256_bytes"
            )
            files.append(
                {
                    "relative_path": relative_text,
                    "bytes": path.stat().st_size,
                    "hash_mode": mode,
                    "sha256": c0_plan_role_file_sha256(path, mode),
                }
            )
    files.sort(key=lambda item: item["relative_path"])
    if not files:
        raise C0RunnerError("production source bundle is empty")
    paths = [item["relative_path"] for item in files]
    if len(paths) != len(set(paths)):
        raise C0RunnerError("production source bundle paths are not unique")
    manifest = {
        "policy": "all_non_test_python_under_pairwise_v0_1_and_v0_2",
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
    }
    manifest["manifest_sha256"] = _content_sha256(manifest)
    return manifest


def c0_plan_role_file_sha256(path: Path, hash_mode: str) -> str:
    """Hash one run-plan role under its preregistered byte policy."""

    if hash_mode == "sha256_bytes":
        return _sha256_file(path)
    if hash_mode == RUNNER_ANCHOR_NORMALIZED_HASH_MODE:
        return hashlib.sha256(_normalized_runner_anchor_bytes(path)).hexdigest()
    if hash_mode == SOURCE_BUNDLE_HASH_MODE:
        return str(c0_source_bundle_manifest(path)["manifest_sha256"])
    raise C0RunnerError("unsupported production run-plan hash mode")


def verify_c0_production_plan(
    binding: ProductionRunPlanBinding,
) -> Tuple[Mapping[str, Any], Tuple[Mapping[str, Any], ...]]:
    """Verify the canonical portable plan and every caller-supplied role path.

    The binding supplies paths only.  Expected role hashes and byte counts come
    exclusively from the plan whose external file/content digests are compiled
    into this module.  The runner-source role normalizes only these two anchor
    constants, avoiding a plan/source hash cycle while binding all logic bytes.
    """

    if not isinstance(binding, ProductionRunPlanBinding):
        raise C0RunnerError("production C0 requires ProductionRunPlanBinding")
    loaded_source_root = Path(__file__).resolve().parents[2]
    if binding.role_paths["source_bundle"].resolve() != loaded_source_root:
        raise C0RunnerError(
            "production source-bundle path differs from the executing runner tree"
        )
    for value, name in (
        (PRODUCTION_RUN_PLAN_FILE_SHA256, "canonical plan file SHA-256"),
        (PRODUCTION_RUN_PLAN_CONTENT_SHA256, "canonical plan content SHA-256"),
    ):
        _require_sha256(value, name)
        if value == "0" * 64:
            raise C0RunnerError("canonical production run plan has not been frozen")
    if not binding.plan_path.is_file():
        raise C0RunnerError("production run-plan path is not a regular file")
    if not hmac.compare_digest(
        _sha256_file(binding.plan_path), PRODUCTION_RUN_PLAN_FILE_SHA256
    ):
        raise C0RunnerError("production run-plan file SHA-256 mismatch")
    try:
        plan = json.loads(binding.plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise C0RunnerError("production run plan is not valid JSON") from exc
    if not isinstance(plan, Mapping):
        raise C0RunnerError("production run plan root must be an object")
    stored_content = plan.get("content_sha256")
    unsigned = dict(plan)
    unsigned.pop("content_sha256", None)
    observed_content = _content_sha256(unsigned)
    if (
        stored_content != observed_content
        or observed_content != PRODUCTION_RUN_PLAN_CONTENT_SHA256
    ):
        raise C0RunnerError("production run-plan content SHA-256 mismatch")
    if (
        plan.get("schema_version") != RUN_PLAN_SCHEMA_VERSION
        or plan.get("status") != "frozen_no_execution"
    ):
        raise C0RunnerError("production run-plan status/schema changed")
    plan_scope = plan.get("scope")
    if not isinstance(plan_scope, Mapping) or set(plan_scope) != {
        "archive_bytes_hashed",
        "archive_members_opened",
        "mask_pixels_decoded",
        "model_or_backend_created",
        "historical_test_access",
        "sealed_real_read",
    }:
        raise C0RunnerError("production run-plan scope schema changed")
    if (
        plan_scope.get("historical_test_access")
        != dict(HISTORICAL_TEST_ACCESS_EVIDENCE)
        or plan_scope.get("archive_bytes_hashed") is not True
        or any(
            plan_scope.get(name) is not False
            for name in (
                "archive_members_opened",
                "mask_pixels_decoded",
                "model_or_backend_created",
                "sealed_real_read",
            )
        )
    ):
        raise C0RunnerError("production run-plan scope evidence changed")
    planned_bundle = plan.get("source_bundle")
    if not isinstance(planned_bundle, Mapping):
        raise C0RunnerError("production run plan lacks source bundle manifest")
    identity_lock = plan.get("identity_index")
    if not isinstance(identity_lock, Mapping) or set(identity_lock) != {
        "schema_version",
        "member_count",
        "content_sha256",
    }:
        raise C0RunnerError("production identity-index lock schema changed")
    if identity_lock.get("schema_version") != IDENTITY_INDEX_CONTENT_SCHEMA:
        raise C0RunnerError("production identity-index schema changed")
    if (
        type(identity_lock.get("member_count")) is not int
        or identity_lock["member_count"] <= 0
    ):
        raise C0RunnerError("production identity-index member count is invalid")
    _require_sha256(
        str(identity_lock.get("content_sha256", "")),
        "production identity-index content SHA-256",
    )
    roles = plan.get("roles")
    if not isinstance(roles, list) or len(roles) != len(PRODUCTION_RUN_PLAN_ROLE_SPECS):
        raise C0RunnerError("production run-plan role list changed")
    by_role = {}
    for row in roles:
        if not isinstance(row, Mapping) or set(row) != {
            "role",
            "logical_id",
            "kind",
            "hash_mode",
            "bytes",
            "sha256",
        }:
            raise C0RunnerError("production run-plan role schema changed")
        role = str(row["role"])
        if role in by_role:
            raise C0RunnerError("duplicate production run-plan role")
        by_role[role] = row
    if set(by_role) != set(PRODUCTION_RUN_PLAN_ROLE_SPECS):
        raise C0RunnerError("production run-plan role names changed")

    verified = []
    seen_paths = set()
    for role in sorted(PRODUCTION_RUN_PLAN_ROLE_SPECS):
        spec = PRODUCTION_RUN_PLAN_ROLE_SPECS[role]
        row = by_role[role]
        if any(row[name] != spec[name] for name in ("logical_id", "kind", "hash_mode")):
            raise C0RunnerError("production role policy changed: " + role)
        path = binding.role_paths[role]
        if path.is_symlink():
            raise C0RunnerError("production role path cannot be a symlink: " + role)
        is_bundle = row["hash_mode"] == SOURCE_BUNDLE_HASH_MODE
        if (is_bundle and not path.is_dir()) or (not is_bundle and not path.is_file()):
            raise C0RunnerError("production role path has wrong kind: " + role)
        resolved = path.resolve()
        if resolved in seen_paths:
            raise C0RunnerError("production role paths must be one-to-one")
        seen_paths.add(resolved)
        observed_bytes = (
            int(c0_source_bundle_manifest(path)["total_bytes"])
            if is_bundle
            else path.stat().st_size
        )
        if type(row["bytes"]) is not int or row["bytes"] != observed_bytes:
            raise C0RunnerError("production role byte count changed: " + role)
        observed = c0_plan_role_file_sha256(path, str(row["hash_mode"]))
        if not hmac.compare_digest(observed, str(row["sha256"])):
            raise C0RunnerError("production role SHA-256 changed: " + role)
        verified.append(
            {
                "role": role,
                "logical_id": row["logical_id"],
                "kind": row["kind"],
                "hash_mode": row["hash_mode"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
        )
    observed_bundle = c0_source_bundle_manifest(binding.role_paths["source_bundle"])
    if _canonical_json(observed_bundle) != _canonical_json(planned_bundle):
        raise C0RunnerError("production source bundle exact manifest changed")
    _assert_portable(plan)
    return plan, tuple(verified)


def _physical_member_key(record: TrainingPairRecord, side: str) -> str:
    fragment = record.fragment_a if side == "a" else record.fragment_b
    return "{}\0{}".format(fragment.binding.sha256, fragment.archive_member)


def _record_token(record: TrainingPairRecord) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "dataset": record.dataset_id,
                "label": record.label,
                "component": record.component_id,
                "pair": list(record.canonical_pair_key),
                "members": [
                    _physical_member_key(record, "a"),
                    _physical_member_key(record, "b"),
                ],
                "content": [
                    record.fragment_a.content_sha256,
                    record.fragment_b.content_sha256,
                ],
            }
        )
    ).hexdigest()


def _ordered_commit(tokens: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _set_commit(tokens: Iterable[str]) -> str:
    return _ordered_commit(sorted(tokens))


def _sequence_commit(records: Sequence[TrainingPairRecord]) -> str:
    return _ordered_commit(_record_token(record) for record in records)


def c0_record_sequence_commitment(
    records: Sequence[TrainingPairRecord],
) -> str:
    """Return the runner's physical/content-bound ordered commitment."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise C0RunnerError("C0 record commitment requires a finite sequence")
    if not records:
        raise C0RunnerError("C0 record commitment cannot be empty")
    return _sequence_commit(records)


def _verify_record_scope(
    record: TrainingPairRecord,
    *,
    split: str,
    freeze: Mapping[str, Any],
    production: bool,
) -> None:
    if not isinstance(record, TrainingPairRecord) or record.split != split:
        raise C0RunnerError("record is outside the required {} split".format(split))
    if record.dataset_id not in _DATASETS:
        raise C0RunnerError("C0 admits only historical MM/ECCV")
    if record.provenance.get("real_dunhuang_sealed_test") is not False:
        raise C0RunnerError("record lacks explicit sealed-real exclusion")
    if production and record.label_origin != "historical_explicit_csv":
        raise C0RunnerError("production C0 admits only historical explicit labels")
    archives = freeze["locks"]["archives"]
    archive = archives.get(record.dataset_id)
    if not isinstance(archive, Mapping):
        raise C0RunnerError("record dataset lacks a frozen archive lock")
    for fragment in (record.fragment_a, record.fragment_b):
        _require_sha256(str(fragment.content_sha256 or ""), "fragment content SHA-256")
        if fragment.binding.sha256 != archive.get("sha256"):
            raise C0RunnerError("record binding differs from frozen archive")


def _attest_records(
    *,
    selected_train_records: Sequence[TrainingPairRecord],
    validation_records: Sequence[TrainingPairRecord],
    freeze: Mapping[str, Any],
    contract: C0RunnerContract,
) -> Mapping[str, Any]:
    if len(selected_train_records) != contract.train_count:
        raise C0RunnerError("selected train cardinality changed")
    if len(validation_records) != contract.validation_count:
        raise C0RunnerError("validation cardinality changed")
    train_counts: Dict[Tuple[str, bool], int] = {}
    for record in selected_train_records:
        _verify_record_scope(
            record, split="train", freeze=freeze, production=contract.production
        )
        key = (record.dataset_id, record.label)
        train_counts[key] = train_counts.get(key, 0) + 1
    expected_train_counts = {
        (dataset, label): contract.train_per_dataset_label
        for dataset in _DATASETS
        for label in (False, True)
    }
    if train_counts != expected_train_counts:
        raise C0RunnerError("selected train dataset/label counts changed")
    train_tokens = [_record_token(record) for record in selected_train_records]
    if len(set(train_tokens)) != len(train_tokens):
        raise C0RunnerError("selected training records are not physically unique")
    selection = freeze["training"]["selection"]
    if _ordered_commit(train_tokens) != selection["record_order_commitment_sha256"]:
        raise C0RunnerError("selected train order commitment changed")
    if _set_commit(train_tokens) != selection["record_set_commitment_sha256"]:
        raise C0RunnerError("selected train set commitment changed")

    validation_counts: Dict[Tuple[str, bool], int] = {}
    for record in validation_records:
        _verify_record_scope(
            record, split="val", freeze=freeze, production=contract.production
        )
        key = (record.dataset_id, record.label)
        validation_counts[key] = validation_counts.get(key, 0) + 1
    validation_fingerprint = validation_stream_fingerprint(validation_records)
    if validation_fingerprint["sha256"] != contract.validation_fingerprint:
        raise C0RunnerError("validation source order/fingerprint changed")
    validation_tokens = [_record_token(record) for record in validation_records]
    expected_validation_commit = freeze["validation"][
        "record_sequence_commitment_sha256"
    ]
    if _ordered_commit(validation_tokens) != expected_validation_commit:
        raise C0RunnerError("validation physical/content commitment changed")

    train_components = {record.component_id for record in selected_train_records}
    validation_components = {record.component_id for record in validation_records}
    train_members = {
        _physical_member_key(record, side)
        for record in selected_train_records
        for side in ("a", "b")
    }
    validation_members = {
        _physical_member_key(record, side)
        for record in validation_records
        for side in ("a", "b")
    }
    train_content = {
        fragment.content_sha256
        for record in selected_train_records
        for fragment in (record.fragment_a, record.fragment_b)
    }
    validation_content = {
        fragment.content_sha256
        for record in validation_records
        for fragment in (record.fragment_a, record.fragment_b)
    }
    observed_overlap = {
        "component": len(train_components & validation_components),
        "member": len(train_members & validation_members),
        "content_sha256": len(train_content & validation_content),
    }
    if any(observed_overlap.values()):
        raise C0RunnerError("runtime train/validation records overlap")
    return {
        "train_count": len(selected_train_records),
        "validation_count": len(validation_records),
        "train_order_commitment_sha256": _ordered_commit(train_tokens),
        "validation_order_commitment_sha256": _ordered_commit(validation_tokens),
        "runtime_overlap": observed_overlap,
        "train_by_dataset_label": {
            "{}|{}".format(dataset, "positive" if label else "negative"): count
            for (dataset, label), count in sorted(train_counts.items())
        },
        "validation_by_dataset_label": {
            "{}|{}".format(dataset, "positive" if label else "negative"): count
            for (dataset, label), count in sorted(validation_counts.items())
        },
    }


def _seed_int(seed: str, namespace: str) -> int:
    return int.from_bytes(
        hashlib.sha256(_canonical_json([seed, namespace])).digest()[:8], "big"
    )


def c0_runner_session_seed(seed: str, purpose: str) -> int:
    """Bind runner session purposes to deterministic integer seeds."""

    namespaces = {
        "train": "training-session",
        "winner_reload": "winner-reload-session",
    }
    try:
        namespace = namespaces[purpose]
    except KeyError as exc:
        raise C0RunnerError("unsupported C0 session purpose") from exc
    return _seed_int(seed, namespace)


def _epoch_records(
    records: Sequence[TrainingPairRecord], *, seed: str, epoch: int
) -> Tuple[TrainingPairRecord, ...]:
    decorated = []
    for record in records:
        token = _record_token(record)
        priority = hashlib.sha256(_canonical_json([seed, epoch, token])).hexdigest()
        decorated.append((priority, token, record))
    decorated.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in decorated)


def _batches(
    records: Sequence[TrainingPairRecord], batch_size: int
) -> Iterable[Tuple[int, Tuple[TrainingPairRecord, ...]]]:
    for index, start in enumerate(range(0, len(records), batch_size)):
        yield index, tuple(records[start : start + batch_size])


def _prepare_checked(
    provider: C0Provider,
    records: Tuple[TrainingPairRecord, ...],
    *,
    phase: str,
) -> C0PreparedBatch:
    batch = provider.prepare(records, phase=phase)
    if not isinstance(batch, C0PreparedBatch):
        raise C0RunnerError("provider returned an invalid prepared batch")
    if batch.sample_count != len(records):
        raise C0RunnerError("provider changed batch cardinality")
    if batch.record_sequence_sha256 != _sequence_commit(records):
        raise C0RunnerError("provider changed batch record order/content")
    return batch


def _prediction_commit(probabilities: Sequence[float], valid: Sequence[bool]) -> str:
    return _ordered_commit(
        json.dumps([float(probability).hex(), bool(is_valid)], separators=(",", ":"))
        for probability, is_valid in zip(probabilities, valid)
    )


def _evaluate_full_validation(
    *,
    provider: C0Provider,
    session: C0Session,
    records: Tuple[TrainingPairRecord, ...],
    batch_size: int,
    phase: str,
    stable_input_digests: Dict[int, str],
) -> Tuple[Mapping[str, Any], Tuple[float, ...], Tuple[bool, ...]]:
    probabilities = []
    valid = []
    for batch_index, batch_records in _batches(records, batch_size):
        prepared = _prepare_checked(
            provider, batch_records, phase="{}-batch-{}".format(phase, batch_index)
        )
        prior_digest = stable_input_digests.setdefault(
            batch_index, prepared.prepared_input_sha256
        )
        if prior_digest != prepared.prepared_input_sha256:
            raise C0RunnerError("validation provider inputs changed across replays")
        prediction = session.predict_batch(prepared)
        if not isinstance(prediction, C0PredictionBatch):
            raise C0RunnerError("backend returned an invalid prediction batch")
        if len(prediction.probability) != len(batch_records):
            raise C0RunnerError("backend changed validation cardinality")
        probabilities.extend(float(value) for value in prediction.probability)
        valid.extend(prediction.valid)
    if len(probabilities) != len(records) or not all(valid):
        raise C0RunnerError("full validation requires every row to be valid")

    labels = [record.label for record in records]
    clusters = [record.component_id for record in records]
    by_dataset = {}
    for dataset in _DATASETS:
        indices = [
            index
            for index, record in enumerate(records)
            if record.dataset_id == dataset
        ]
        if not indices:
            raise C0RunnerError("validation dataset is missing: " + dataset)
        by_dataset[dataset] = evaluate_pairwise(
            [probabilities[index] for index in indices],
            [labels[index] for index in indices],
            [valid[index] for index in indices],
            [clusters[index] for index in indices],
            threshold=0.5,
        )
    macro_cluster_auroc = float(
        np.mean(
            [
                float(by_dataset[dataset]["cluster_balanced"]["auroc"])
                for dataset in _DATASETS
            ]
        )
    )
    macro_cluster_auprc = float(
        np.mean(
            [
                float(by_dataset[dataset]["cluster_balanced"]["auprc"])
                for dataset in _DATASETS
            ]
        )
    )
    report = {
        "count": len(records),
        "all_rows_valid": True,
        "by_dataset": by_dataset,
        "equal_domain_macro_cluster": {
            "auroc": macro_cluster_auroc,
            "auprc": macro_cluster_auprc,
        },
        "prediction_commitment_sha256": _prediction_commit(probabilities, valid),
        "selection_threshold_used": False,
    }
    return report, tuple(probabilities), tuple(valid)


def _backend_checked(value: Any) -> C0Backend:
    attestation = getattr(value, "attestation", None)
    if not isinstance(attestation, BackendAttestation):
        raise C0RunnerError("backend lacks a typed attestation")
    return value


def _provider_checked(value: Any, freeze: Mapping[str, Any]) -> C0Provider:
    attestation = getattr(value, "attestation", None)
    if not isinstance(attestation, ProviderAttestation):
        raise C0RunnerError("provider lacks a typed attestation")
    expected = _content_sha256(freeze["locks"]["archives"])
    if not hmac.compare_digest(attestation.archive_locks_sha256, expected):
        raise C0RunnerError("provider archive attestation differs from freeze")
    return value


def _validate_final_provider_evidence(
    provider: C0Provider,
    *,
    contract: C0RunnerContract,
    freeze: Mapping[str, Any],
    expected_identity_index_content_sha256: Optional[str],
) -> Optional[Mapping[str, Any]]:
    """Bind the final receipt to every data access made by the runner."""

    receipt_method = getattr(provider, "final_receipt", None)
    if not callable(receipt_method):
        if contract.production:
            raise C0RunnerError("production provider lacks a final receipt")
        return None
    receipt = receipt_method()
    if not isinstance(receipt, Mapping):
        raise C0RunnerError("provider final receipt is invalid")
    if (
        receipt.get("freeze_content_sha256") != freeze.get("content_sha256")
        or "historical_test_read" in receipt
        or receipt.get("historical_test_access")
        != dict(HISTORICAL_TEST_ACCESS_EVIDENCE)
        or receipt.get("sealed_real_read") is not False
        or receipt.get("geometry_cache_local_sinkhorn_calls") != 0
    ):
        raise C0RunnerError("provider final receipt violates C0 scope")
    if expected_identity_index_content_sha256 is not None and not hmac.compare_digest(
        str(receipt.get("identity_index_content_sha256", "")),
        expected_identity_index_content_sha256,
    ):
        raise C0RunnerError("provider final receipt identity index changed")
    validation_batches = math.ceil(contract.validation_count / contract.batch_size)
    expected_batches = contract.total_steps + (contract.epochs + 1) * validation_batches
    expected_records = (
        contract.epochs * contract.train_count
        + (contract.epochs + 1) * contract.validation_count
    )
    counters = receipt.get("counters")
    if not isinstance(counters, Mapping) or any(
        counters.get(name) != expected
        for name, expected in {
            "batch_count": expected_batches,
            "record_count": expected_records,
            "fragment_request_count": 2 * expected_records,
        }.items()
    ):
        raise C0RunnerError("provider final access counts differ from C0 schedule")
    if contract.production:
        memo_bounds = receipt.get("memo_bounds")
        expected_memo_bounds = {
            "max_batch_size": contract.batch_size,
            "max_cached_fragments": C0_PRODUCTION_MAX_CACHED_FRAGMENTS,
            "max_cache_bytes": C0_PRODUCTION_MAX_CACHE_BYTES,
        }
        if not isinstance(memo_bounds, Mapping) or dict(memo_bounds) != (
            expected_memo_bounds
        ):
            raise C0RunnerError("production provider memo bounds changed")
        if counters.get("memo_eviction_count") != 0:
            raise C0RunnerError("production provider coarse memo evicted fragments")
    archive_verification = receipt.get("archive_verification")
    expected_archives = sorted(
        (str(row["format"]), str(row["sha256"]), int(row["bytes"]))
        for row in freeze["locks"]["archives"].values()
    )
    observed_archives = (
        sorted(
            (
                str(row.get("archive_format")),
                str(row.get("observed_sha256")),
                int(row.get("byte_count", -1)),
            )
            for row in archive_verification
            if isinstance(row, Mapping)
            and row.get("expected_sha256") == row.get("observed_sha256")
        )
        if isinstance(archive_verification, list)
        else []
    )
    if observed_archives != expected_archives:
        raise C0RunnerError("provider final archive evidence changed")
    preprocessing = receipt.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise C0RunnerError("provider final preprocessing evidence is absent")
    for key in ("preprocessing_sha256", "geometry_config_sha256"):
        _require_sha256(str(preprocessing.get(key, "")), "provider final " + key)
    _assert_portable(receipt)
    return dict(receipt)


def _session_config(
    *,
    contract: C0RunnerContract,
    provider: C0Provider,
    backend: C0Backend,
    session: C0Session,
    environment: Mapping[str, Any],
) -> Mapping[str, Any]:
    model_config = dict(session.model_config)
    optimizer_config = dict(session.optimizer_config)
    config = {
        "runner": dict(contract.portable_dict()),
        "provider": dict(provider.attestation.portable_dict()),
        "backend": dict(backend.attestation.portable_dict()),
        "environment": dict(environment),
        "model": model_config,
        "optimizer": optimizer_config,
    }
    _assert_portable(config)
    return config


def _nvidia_driver_evidence(device_index: int) -> Mapping[str, Any]:
    """Query one CUDA device without a shell or caller-controlled arguments."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--id={}".format(device_index),
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return {"status": "nvidia_smi_not_found", "version": None}
    except subprocess.TimeoutExpired:
        return {"status": "nvidia_smi_timeout", "version": None}
    except subprocess.CalledProcessError:
        return {"status": "nvidia_smi_failed", "version": None}
    except OSError:
        return {"status": "nvidia_smi_unavailable", "version": None}
    versions = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(versions) != 1 or re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", versions[0]) is None:
        return {"status": "invalid_output", "version": None}
    return {"status": "observed", "version": versions[0]}


def _runtime_environment(backend: C0Backend) -> Mapping[str, Any]:
    """Return portable runtime evidence without host, path, or env disclosure."""

    device_type = backend.attestation.device_type
    if device_type == "cuda":
        cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cublas_workspace_config not in _CUDA_CUBLAS_WORKSPACE_CONFIGS:
            raise C0RunnerError("CUDA CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
        cublas_evidence = {
            "status": "validated",
            "value": cublas_workspace_config,
        }
    else:
        cublas_evidence = {"status": "not_applicable", "value": None}
    cuda_available = bool(torch.cuda.is_available())
    if device_type == "cuda":
        if not cuda_available:
            raise C0RunnerError("CUDA backend has no available CUDA runtime")
        device_index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        device = {
            "type": device_type,
            "index": device_index,
            "name": str(properties.name),
            "compute_capability": {
                "major": int(capability[0]),
                "minor": int(capability[1]),
            },
            "total_memory_bytes": int(properties.total_memory),
        }
        driver = _nvidia_driver_evidence(device_index)
    else:
        device = {
            "type": device_type,
            "index": None,
            "name": None,
            "compute_capability": None,
            "total_memory_bytes": None,
        }
        driver = {"status": "not_applicable", "version": None}
    cudnn_version = torch.backends.cudnn.version()
    evidence = {
        "schema_version": _RUNTIME_ENVIRONMENT_SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "torch": {"version": str(torch.__version__)},
        "cuda": {
            "available": cuda_available,
            "runtime_version": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
            "driver": driver,
        },
        "cudnn": {
            "available": bool(torch.backends.cudnn.is_available()),
            "version": None if cudnn_version is None else int(cudnn_version),
        },
        "device": device,
        "determinism": {"cublas_workspace_config": cublas_evidence},
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    _assert_portable(evidence)
    return evidence


def _fresh_output_directory(path: Path) -> Path:
    output = Path(path)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise C0RunnerError("C0 output directory must be new or empty")
    else:
        output.mkdir(parents=True)
    return output


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_c0_n_q1(
    *,
    run_plan: Union[C0RunnerContract, Mapping[str, Any]],
    freeze_receipt: Union[FrozenReceiptLock, Mapping[str, Any]],
    selected_train_records: Sequence[TrainingPairRecord],
    validation_records: Sequence[TrainingPairRecord],
    provider_factory: Callable[[], C0Provider],
    backend_factory: Callable[[], C0Backend],
    output_dir: Path,
    production_plan: Optional[ProductionRunPlanBinding] = None,
) -> C0RunArtifacts:
    """Train, select and clean-replay C0-N-Q1 under frozen metadata.

    There is intentionally no test-record argument.  Thresholds are fitted
    separately per validation dataset *after* winner selection and are marked
    diagnostic-only; this runner emits neither a pooled nor production cutoff.
    """

    contract = _coerce_contract(run_plan)
    locked_freeze = _coerce_freeze_lock(freeze_receipt)
    if contract.production and (
        locked_freeze.file_sha256 != PRODUCTION_FREEZE_FILE_SHA256
        or locked_freeze.content_sha256 != PRODUCTION_FREEZE_CONTENT_SHA256
    ):
        raise C0RunnerError("production C0 freeze lock differs from canonical freeze")

    # Every external trust boundary is checked before either factory.  The
    # production caller supplies paths only; hashes come from the compiled,
    # preregistered plan and its exact source-bundle manifest.
    verified_plan: Optional[Mapping[str, Any]] = None
    if contract.production:
        if production_plan is None:
            raise C0RunnerError("production C0 requires the canonical run plan")
        verified_plan, source_locks = verify_c0_production_plan(production_plan)
        if (
            production_plan.role_paths["freeze_receipt"].resolve()
            != locked_freeze.path.resolve()
        ):
            raise C0RunnerError("freeze path differs from production plan binding")
        plan_by_role = {row["role"]: row for row in verified_plan["roles"]}
        freeze_role = plan_by_role["freeze_receipt"]
        if (
            freeze_role["sha256"] != PRODUCTION_FREEZE_FILE_SHA256
            or verified_plan.get("freeze_content_sha256")
            != PRODUCTION_FREEZE_CONTENT_SHA256
        ):
            raise C0RunnerError("production plan does not bind the canonical freeze")
        expected_contract = dict(contract.portable_dict())
        if verified_plan.get("contract") != expected_contract:
            raise C0RunnerError("production plan contract differs from runner")
    else:
        if production_plan is not None:
            raise C0RunnerError("fixture C0 cannot accept a production plan")
        source_locks = _verify_source_locks(contract.source_locks, production=False)

    freeze = _load_and_verify_freeze(locked_freeze, contract)
    source_lock_commitment = _content_sha256(source_locks)

    # Pure record metadata must match the freeze before either factory is
    # allowed to attest archives or construct a model/backend.
    train_records = tuple(selected_train_records)
    val_records = tuple(validation_records)
    metadata_attestation = _attest_records(
        selected_train_records=train_records,
        validation_records=val_records,
        freeze=freeze,
        contract=contract,
    )

    provider = _provider_checked(provider_factory(), freeze)
    if contract.production and not hmac.compare_digest(
        str(provider.attestation.identity_index_content_sha256 or ""),
        str(verified_plan["identity_index"]["content_sha256"]),
    ):
        raise C0RunnerError("provider identity index differs from production plan")

    backend = _backend_checked(backend_factory())
    runtime_environment = _runtime_environment(backend)
    runtime_environment_sha256 = _content_sha256(runtime_environment)
    session = backend.create_session(
        seed=c0_runner_session_seed(contract.seed, "train"), purpose="train"
    )
    if not isinstance(session.model, nn.Module):
        raise C0RunnerError("backend session model must be torch.nn.Module")
    checkpoint_config = _session_config(
        contract=contract,
        provider=provider,
        backend=backend,
        session=session,
        environment=runtime_environment,
    )
    output = _fresh_output_directory(Path(output_dir))

    epoch_rows = []
    checkpoint_runtime = []
    stable_validation_inputs: Dict[int, str] = {}
    observed_steps = 0
    for epoch in range(1, contract.epochs + 1):
        epoch_records = _epoch_records(train_records, seed=contract.seed, epoch=epoch)
        if len(epoch_records) != contract.train_count:
            raise C0RunnerError("epoch order changed train cardinality")
        losses = []
        batch_count = 0
        for batch_index, batch_records in _batches(epoch_records, contract.batch_size):
            prepared = _prepare_checked(
                provider,
                batch_records,
                phase="train-epoch-{}-batch-{}".format(epoch, batch_index),
            )
            result = session.train_batch(prepared)
            if not isinstance(result, C0TrainBatchResult):
                raise C0RunnerError("backend returned an invalid training result")
            if result.valid_count != len(batch_records):
                raise C0RunnerError("training backend dropped records")
            losses.append(float(result.loss))
            batch_count += 1
            observed_steps += 1
        if batch_count != contract.batches_per_epoch:
            raise C0RunnerError("epoch batch count differs from contract")

        validation_report, _probability, _valid = _evaluate_full_validation(
            provider=provider,
            session=session,
            records=val_records,
            batch_size=contract.batch_size,
            phase="validation-epoch-{}".format(epoch),
            stable_input_digests=stable_validation_inputs,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train": {
                "batch_count": batch_count,
                "step_count_cumulative": observed_steps,
                "mean_loss": float(np.mean(losses)),
                "epoch_order_commitment_sha256": _sequence_commit(epoch_records),
            },
            "validation": validation_report,
        }
        checkpoint_path = output / "checkpoint-epoch-{:02d}.pt".format(epoch)
        checkpoint = save_checkpoint(
            checkpoint_path,
            session.model,
            config=checkpoint_config,
            epoch=epoch,
            optimizer=session.optimizer,
            metrics=epoch_metrics,
            provenance={
                "freeze_file_sha256": locked_freeze.file_sha256,
                "freeze_content_sha256": locked_freeze.content_sha256,
                "source_lock_commitment_sha256": source_lock_commitment,
                "environment_sha256": runtime_environment_sha256,
                "train_order_commitment_sha256": metadata_attestation[
                    "train_order_commitment_sha256"
                ],
                "validation_order_commitment_sha256": metadata_attestation[
                    "validation_order_commitment_sha256"
                ],
            },
        )
        portable_checkpoint = {
            "epoch": epoch,
            "file_sha256": checkpoint.file_sha256,
            "canonical_content_sha256": checkpoint.canonical_content_sha256,
            "model_state_sha256": checkpoint.model_state_sha256,
            "optimizer_state_sha256": checkpoint.optimizer_state_sha256,
            "config_hash": checkpoint.config_hash,
        }
        epoch_rows.append(
            {
                "epoch": epoch,
                "train": epoch_metrics["train"],
                "validation": validation_report,
                "checkpoint": portable_checkpoint,
            }
        )
        checkpoint_runtime.append((checkpoint_path, checkpoint, checkpoint_config))

    if observed_steps != contract.total_steps:
        raise C0RunnerError("observed optimizer steps differ from contract")
    winner_row = max(
        epoch_rows,
        key=lambda row: (
            float(row["validation"]["equal_domain_macro_cluster"]["auroc"]),
            float(row["validation"]["equal_domain_macro_cluster"]["auprc"]),
            -int(row["epoch"]),
        ),
    )
    winner_epoch = int(winner_row["epoch"])
    winner_path, winner_checkpoint, winner_config = checkpoint_runtime[winner_epoch - 1]

    # A second backend factory and session form the clean reload boundary.
    reload_backend = _backend_checked(backend_factory())
    if reload_backend.attestation != backend.attestation:
        raise C0RunnerError("reload backend attestation changed")
    if _runtime_environment(reload_backend) != runtime_environment:
        raise C0RunnerError("reload runtime environment evidence changed")
    reload_session = reload_backend.create_session(
        seed=c0_runner_session_seed(contract.seed, "winner_reload"),
        purpose="winner_reload",
    )
    reload_config = _session_config(
        contract=contract,
        provider=provider,
        backend=reload_backend,
        session=reload_session,
        environment=runtime_environment,
    )
    if canonical_config_hash(reload_config) != canonical_config_hash(winner_config):
        raise C0RunnerError("fresh reload session config differs from training")
    loaded = load_trusted_checkpoint(
        winner_path,
        reload_session.model,
        expected_config=winner_config,
        expected_file_sha256=winner_checkpoint.file_sha256,
        expected_canonical_content_sha256=(winner_checkpoint.canonical_content_sha256),
        map_location="cpu",
        trusted=True,
    )
    if loaded["epoch"] != winner_epoch:
        raise C0RunnerError("restricted reload returned the wrong epoch")
    replay_report, replay_probability, replay_valid = _evaluate_full_validation(
        provider=provider,
        session=reload_session,
        records=val_records,
        batch_size=contract.batch_size,
        phase="winner-reload-validation",
        stable_input_digests=stable_validation_inputs,
    )
    if (
        replay_report["prediction_commitment_sha256"]
        != winner_row["validation"]["prediction_commitment_sha256"]
    ):
        raise C0RunnerError("winner reload predictions differ from winning epoch")
    if _content_sha256(replay_report) != _content_sha256(winner_row["validation"]):
        raise C0RunnerError("winner reload validation metrics differ")

    labels = [record.label for record in val_records]
    clusters = [record.component_id for record in val_records]
    model_config_sha256 = canonical_config_hash(dict(reload_session.model_config))
    diagnostic_aggregation_sha256 = canonical_config_hash(
        {
            "aggregation": "symmetric_coarse_probability",
            "threshold_scope": "per_dataset_validation_diagnostic_only",
        }
    )
    diagnostic_thresholds = {}
    for dataset in _DATASETS:
        indices = [
            index
            for index, record in enumerate(val_records)
            if record.dataset_id == dataset
        ]
        artifact = fit_pairwise_threshold(
            [replay_probability[index] for index in indices],
            [labels[index] for index in indices],
            [replay_valid[index] for index in indices],
            [clusters[index] for index in indices],
            source_split="validation",
            validation_fingerprint_sha256=contract.validation_fingerprint,
            checkpoint_sha256=winner_checkpoint.file_sha256,
            model_config_sha256=model_config_sha256,
            aggregation_config_sha256=diagnostic_aggregation_sha256,
        )
        diagnostic_thresholds[dataset] = artifact.to_dict()

    expected_identity_index_content_sha256 = (
        str(verified_plan["identity_index"]["content_sha256"])
        if verified_plan is not None
        else None
    )
    final_provider_evidence = _validate_final_provider_evidence(
        provider,
        contract=contract,
        freeze=freeze,
        expected_identity_index_content_sha256=(expected_identity_index_content_sha256),
    )

    checkpoint_sidecar = {
        "schema_version": "dunhuang-pairwise-c0-checkpoint-sidecar-local/0.2",
        "local_only": True,
        "checkpoints": [
            {
                "epoch": checkpoint.epoch,
                "path": str(path.resolve()),
                "file_sha256": checkpoint.file_sha256,
                "canonical_content_sha256": checkpoint.canonical_content_sha256,
            }
            for path, checkpoint, _config in checkpoint_runtime
        ],
        "winner_epoch": winner_epoch,
        "winner_path": str(winner_path.resolve()),
    }
    sidecar_path = output / "checkpoint_sidecar.local.json"
    _atomic_json(sidecar_path, checkpoint_sidecar)

    receipt: Dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "complete_validation_selected_no_test_evaluation",
        "scope": {
            "experiment": "C0-N-Q1",
            "historical_test_records_accepted": False,
            "sealed_real_records_accepted": False,
            "threshold_used_for_epoch_selection": False,
        },
        "contract": dict(contract.portable_dict()),
        "locks": {
            "freeze_file_sha256": locked_freeze.file_sha256,
            "freeze_content_sha256": locked_freeze.content_sha256,
            "source_files": list(source_locks),
            "source_lock_commitment_sha256": source_lock_commitment,
            "production_run_plan_file_sha256": (
                PRODUCTION_RUN_PLAN_FILE_SHA256 if contract.production else None
            ),
            "production_run_plan_content_sha256": (
                PRODUCTION_RUN_PLAN_CONTENT_SHA256 if contract.production else None
            ),
        },
        "provider": dict(provider.attestation.portable_dict()),
        "provider_final_evidence": final_provider_evidence,
        "backend": dict(backend.attestation.portable_dict()),
        "environment": dict(runtime_environment),
        "environment_sha256": runtime_environment_sha256,
        "metadata_attestation": dict(metadata_attestation),
        "epochs": epoch_rows,
        "observed": {
            "epoch_count": len(epoch_rows),
            "checkpoint_count": len(checkpoint_runtime),
            "batches_per_epoch": contract.batches_per_epoch,
            "total_steps": observed_steps,
            "full_validation_replay_count": contract.epochs + 1,
        },
        "winner": {
            "epoch": winner_epoch,
            "selection_policy": (
                "max_equal_domain_macro_cluster_auroc_then_auprc_then_earlier_epoch"
            ),
            "checkpoint": winner_row["checkpoint"],
            "restricted_fresh_session_reload": True,
            "full_validation_replay_match": True,
            "environment_sha256": runtime_environment_sha256,
            "validation": replay_report,
        },
        "thresholds": {
            "status": "validation_diagnostic_only_after_winner_selection",
            "production_threshold": None,
            "pooled_threshold": None,
            "by_dataset": diagnostic_thresholds,
        },
        "local_checkpoint_sidecar_written": True,
    }
    _assert_portable(receipt)
    receipt["content_sha256"] = _content_sha256(receipt)
    receipt_path = output / "c0_run_receipt.json"
    _atomic_json(receipt_path, receipt)
    return C0RunArtifacts(
        run_directory=output,
        receipt_path=receipt_path,
        checkpoint_sidecar_path=sidecar_path,
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "BackendAttestation",
    "C0Backend",
    "C0PredictionBatch",
    "C0PreparedBatch",
    "C0Provider",
    "C0RunArtifacts",
    "C0RunnerContract",
    "C0RunnerError",
    "C0Session",
    "C0TrainBatchResult",
    "FrozenReceiptLock",
    "PRODUCTION_FREEZE_CONTENT_SHA256",
    "PRODUCTION_FREEZE_FILE_SHA256",
    "PRODUCTION_VALIDATION_FINGERPRINT",
    "ProviderAttestation",
    "RUNNER_SCHEMA_VERSION",
    "SourceFileLock",
    "c0_record_sequence_commitment",
    "c0_runner_session_seed",
    "run_c0_n_q1",
]
