"""Independent, read-only acceptance checks for a completed C0 production run.

The validator deliberately treats ``*.pt`` checkpoints as opaque byte strings.
It verifies their byte hashes against both the portable run receipt and the
local sidecar, but it never imports PyTorch, calls ``torch.load``, or claims to
have recomputed checkpoint canonical-content hashes.  A trusted, config-bound
checkpoint loader remains a separate execution boundary.

The public :func:`validate_c0_run_artifacts` entry point is production-only and
has no argument that can weaken the frozen constants.  The private fixture
entry point exists solely so the runner's tiny test fixture can exercise the
same artifact checker without pretending its small counts are production.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


VALIDATION_RECEIPT_SCHEMA_VERSION = "dunhuang-pairwise-c0-artifact-validation/0.2"
RUN_SCHEMA_VERSION = "dunhuang-pairwise-c0-run/0.3"
RUN_PLAN_SCHEMA_VERSION = "dunhuang-pairwise-c0-production-run-plan/0.3"
SIDECAR_SCHEMA_VERSION = "dunhuang-pairwise-c0-checkpoint-sidecar-local/0.2"
ENVIRONMENT_SCHEMA_VERSION = "dunhuang-pairwise-c0-runtime-environment/0.4"
PROVIDER_RECEIPT_SCHEMA_VERSION = "c0-n-q1-coarse-provider-receipt/0.5"
PROVIDER_BASE_VERSION = "c0-n-q1-coarse-provider/0.5"
PROVIDER_ADAPTER_VERSION = "dunhuang-pairwise-c0-runtime-adapter/0.3"
PROVIDER_ATTESTATION_VERSION = PROVIDER_BASE_VERSION + "+" + PROVIDER_ADAPTER_VERSION
BACKEND_BASE_VERSION = "c0-n-q1-coarse-backend/0.2"
BACKEND_ATTESTATION_VERSION = BACKEND_BASE_VERSION + "+" + PROVIDER_ADAPTER_VERSION
BACKEND_MODEL_FAMILY = "SymmetricCoarseSiamese-C0-N-Q1"

PRODUCTION_RUN_PLAN_FILE_SHA256 = (
    "bd1b9b96fb686241fe560c0e001d7057833f7781a0f9b13e72b700515b0c0cae"
)
PRODUCTION_RUN_PLAN_CONTENT_SHA256 = (
    "acacf7d3611e551750cb5f859f805fba78b15f31f185a396cab3111e9ef5b94e"
)
PRODUCTION_FREEZE_FILE_SHA256 = (
    "eb9b8bcab71661c42aab73b1cb9d117c15af145bf5e0a7b897115824ce4a44c2"
)
PRODUCTION_FREEZE_CONTENT_SHA256 = (
    "19950f4f8d20f556de819bc558c59746c02db32f769ce9f54bfb96d82b133e6c"
)
PRODUCTION_VALIDATION_FINGERPRINT = (
    "23c7c7446adbddf205ff815240973d2d6e1e89c1c77f27f5a2330b11ccd51078"
)
PRODUCTION_IDENTITY_INDEX_CONTENT_SHA256 = (
    "50c51783b20b375564290ce426e33204816959203062b22e32081af4a8078158"
)
PRODUCTION_TRAIN_ORDER_COMMITMENT_SHA256 = (
    "7bfe04dc61c425a2865c5cef4f0c69f9fc4a4812c78cf119240aff3d796840a6"
)
PRODUCTION_VALIDATION_ORDER_COMMITMENT_SHA256 = (
    "a7280965527d90d367443b4dbfbad4f18a6bfb04ad8ef80f202394446da36668"
)
PRODUCTION_GEOMETRY_CONFIG_SHA256 = (
    "dc994bf15596eea1b3d9245ac41505b1e72739eb49a91fef7a02012c744dcfd3"
)
PRODUCTION_PREPROCESSING_SHA256 = (
    "fa2b31e263b575a4e9794ce54af29cfd67d095bd8417526b9d6e23a830a4db32"
)

_CHECKPOINT_COUNT = 5
_PRODUCTION_BATCHES_PER_EPOCH = 64
_PRODUCTION_TOTAL_STEPS = 320
_PRODUCTION_FULL_VALIDATION_REPLAYS = 6
_PRODUCTION_PROVIDER_BATCHES = 6_722
_PRODUCTION_PROVIDER_RECORDS = 1_720_196
_PRODUCTION_PROVIDER_ENDPOINTS = 3_440_392
_PRODUCTION_MEMO_FRAGMENTS = 131_072
_PRODUCTION_MEMO_BYTES = 8 * 1024**3
_PRODUCTION_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_MAX_CHECKPOINT_BYTES = 256 * 1024**2
_DATASETS = ("mm_augmented", "eccv_1113data")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DRIVER_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")

_HISTORICAL_TEST_ACCESS = {
    "assignment_split_and_cache_metadata_parsed_for_exclusion": True,
    "pair_stream_read": False,
    "archive_members_opened": False,
    "mask_pixels_decoded": False,
    "identity_entered_runtime_index": False,
}

_PRODUCTION_CONTRACT = {
    "epochs": 5,
    "batch_size": 256,
    "train_count": 16_384,
    "train_per_dataset_label": 4_096,
    "validation_count": 273_046,
    "validation_fingerprint": PRODUCTION_VALIDATION_FINGERPRINT,
    "seed_sha256": ("28f137e69811095dcb232573647132134a1f812f38fba039446f5961b5d66e15"),
    "batches_per_epoch": _PRODUCTION_BATCHES_PER_EPOCH,
    "total_steps": _PRODUCTION_TOTAL_STEPS,
    "production": True,
}

_PRODUCTION_TRAIN_COUNTS = {
    "eccv_1113data|negative": 4_096,
    "eccv_1113data|positive": 4_096,
    "mm_augmented|negative": 4_096,
    "mm_augmented|positive": 4_096,
}
_PRODUCTION_VALIDATION_COUNTS = {
    "eccv_1113data|negative": 1_288,
    "eccv_1113data|positive": 9_768,
    "mm_augmented|negative": 208_502,
    "mm_augmented|positive": 53_488,
}
_PRODUCTION_CLUSTER_COUNTS = {
    "eccv_1113data": 533,
    "mm_augmented": 55,
}
_PRODUCTION_PREPROCESSING = {
    "component_connectivity": 4,
    "content_fraction": 0.875,
    "geometry_config_sha256": PRODUCTION_GEOMETRY_CONFIG_SHA256,
    "mode": "tight_crop_letterbox",
    "numeric_contract": (
        "finite_float32_contiguous_reject_beyond_2eps_then_clamp_unit_interval/0.1"
    ),
    "output_size": [128, 128],
    "preprocessing_sha256": PRODUCTION_PREPROCESSING_SHA256,
    "resize_mode": "bilinear",
    "spatial_contract": (
        "single_fragment_largest_connected_component_foreground_bbox_then_"
        "aspect_preserving_centered_letterbox_no_common_canvas_position"
    ),
}
_FROZEN_MODEL_CONFIG = {
    "embedding_dim": 96,
    "hidden_dim": 96,
    "input_channels": 1,
    "input_shape": [1, 128, 128],
    "name": "SymmetricCoarseSiamese",
    "widths": [16, 32, 64],
}
_FROZEN_OPTIMIZER_CONFIG = {
    "amsgrad": False,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "foreach": False,
    "fused": None,
    "lr": 3e-4,
    "name": "AdamW",
    "weight_decay": 1e-4,
}
_DIAGNOSTIC_AGGREGATION_CONFIG = {
    "aggregation": "symmetric_coarse_probability",
    "threshold_scope": "per_dataset_validation_diagnostic_only",
}
_PRODUCTION_ARCHIVE_ROLE_TO_PROVIDER_LOGICAL_ID = MappingProxyType(
    {
        "eccv_archive": "canonical://eccv_1113data/1113data",
        "mm_archive": "canonical://mm_augmented/dunhuang_augmented_data",
    }
)


class C0ArtifactValidationError(RuntimeError):
    """Raised when a completed-run attestation cannot be proven."""


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    sha256: str
    byte_count: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _ValidationPolicy:
    mode: str
    plan_file_sha256: str
    plan_content_sha256: str
    contract: Mapping[str, Any]
    provider_counts: Mapping[str, int]
    memo_bounds: Mapping[str, int]
    device_type: str
    cublas_workspace_config: Optional[str]
    enforce_production_plan_locks: bool
    expected_backend_version: str
    expected_backend_model_family: str
    expected_model_config: Mapping[str, Any]
    expected_optimizer_config: Mapping[str, Any]
    expected_archive_role_to_provider_logical_id: Mapping[str, str]
    expected_train_counts: Optional[Mapping[str, int]] = None
    expected_validation_counts: Optional[Mapping[str, int]] = None
    expected_cluster_counts: Optional[Mapping[str, int]] = None
    expected_train_order_commitment_sha256: Optional[str] = None
    expected_validation_order_commitment_sha256: Optional[str] = None
    require_optimizer_state_sha256: bool = False
    expected_preprocessing: Optional[Mapping[str, Any]] = None
    expected_archive_evidence: Optional[Mapping[str, Tuple[str, str]]] = None


def _production_policy() -> _ValidationPolicy:
    return _ValidationPolicy(
        mode="production",
        plan_file_sha256=PRODUCTION_RUN_PLAN_FILE_SHA256,
        plan_content_sha256=PRODUCTION_RUN_PLAN_CONTENT_SHA256,
        contract=dict(_PRODUCTION_CONTRACT),
        provider_counts={
            "batch_count": _PRODUCTION_PROVIDER_BATCHES,
            "record_count": _PRODUCTION_PROVIDER_RECORDS,
            "fragment_request_count": _PRODUCTION_PROVIDER_ENDPOINTS,
        },
        memo_bounds={
            "max_batch_size": 256,
            "max_cached_fragments": _PRODUCTION_MEMO_FRAGMENTS,
            "max_cache_bytes": _PRODUCTION_MEMO_BYTES,
        },
        device_type="cuda",
        cublas_workspace_config=_PRODUCTION_CUBLAS_WORKSPACE_CONFIG,
        enforce_production_plan_locks=True,
        expected_backend_version=BACKEND_ATTESTATION_VERSION,
        expected_backend_model_family=BACKEND_MODEL_FAMILY,
        expected_model_config=dict(_FROZEN_MODEL_CONFIG),
        expected_optimizer_config=dict(_FROZEN_OPTIMIZER_CONFIG),
        expected_archive_role_to_provider_logical_id=dict(
            _PRODUCTION_ARCHIVE_ROLE_TO_PROVIDER_LOGICAL_ID
        ),
        expected_train_counts=dict(_PRODUCTION_TRAIN_COUNTS),
        expected_validation_counts=dict(_PRODUCTION_VALIDATION_COUNTS),
        expected_cluster_counts=dict(_PRODUCTION_CLUSTER_COUNTS),
        expected_train_order_commitment_sha256=(
            PRODUCTION_TRAIN_ORDER_COMMITMENT_SHA256
        ),
        expected_validation_order_commitment_sha256=(
            PRODUCTION_VALIDATION_ORDER_COMMITMENT_SHA256
        ),
        require_optimizer_state_sha256=True,
        expected_preprocessing=dict(_PRODUCTION_PREPROCESSING),
        expected_archive_evidence={
            "eccv_archive": (
                "tar",
                "observed_open_file_stream",
            ),
            "mm_archive": (
                "zip",
                "observed_open_file_stream",
            ),
        },
    )


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C0ArtifactValidationError(name + " must be lowercase SHA-256")
    return value


def _require_builtin_int(value: Any, expected: int, name: str) -> None:
    if type(value) is not int or value != expected:  # noqa: E721
        raise C0ArtifactValidationError(
            "{} differs: expected {}, observed {!r}".format(name, expected, value)
        )


def _require_exact_keys(value: Any, expected: Sequence[str], name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise C0ArtifactValidationError(name + " must be an object")
    observed = set(value)
    required = set(expected)
    if observed != required:
        raise C0ArtifactValidationError(
            "{} key set differs (missing={}, extra={})".format(
                name, sorted(required - observed), sorted(observed - required)
            )
        )
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise C0ArtifactValidationError("value is not canonical JSON") from exc


def _canonical_pretty_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise C0ArtifactValidationError("value is not canonical JSON") from exc


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stat_identity(metadata: os.stat_result) -> Tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _open_regular_file(
    path: Path, *, name: str, max_bytes: Optional[int] = None
) -> Tuple[int, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise C0ArtifactValidationError(name + " is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise C0ArtifactValidationError(name + " must be a non-symlink regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise C0ArtifactValidationError(name + " cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stat_identity(before) != _stat_identity(
            opened
        ):
            raise C0ArtifactValidationError(name + " changed while it was opened")
        if max_bytes is not None and (
            opened.st_size <= 0 or opened.st_size > max_bytes
        ):
            raise C0ArtifactValidationError(name + " byte count is outside its bound")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, opened


def _snapshot_from_stat(
    path: Path, digest: str, metadata: os.stat_result
) -> _FileSnapshot:
    return _FileSnapshot(
        path=path,
        sha256=digest,
        byte_count=int(metadata.st_size),
        device=int(metadata.st_dev),
        inode=int(metadata.st_ino),
        mtime_ns=int(metadata.st_mtime_ns),
        ctime_ns=int(metadata.st_ctime_ns),
    )


def _assert_snapshot_current(snapshot: _FileSnapshot, name: str) -> None:
    try:
        current = snapshot.path.lstat()
    except OSError as exc:
        raise C0ArtifactValidationError(name + " disappeared after validation") from exc
    observed = _stat_identity(current)
    expected = (
        snapshot.device,
        snapshot.inode,
        snapshot.byte_count,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or observed != expected
    ):
        raise C0ArtifactValidationError(name + " changed during validation")


def _read_regular_file_snapshot(
    path: Path, *, max_bytes: int, name: str
) -> Tuple[bytes, _FileSnapshot]:
    descriptor, opened = _open_regular_file(path, name=name, max_bytes=max_bytes)
    digest = hashlib.sha256()
    chunks: List[bytes] = []
    observed_bytes = 0
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > max_bytes:
                raise C0ArtifactValidationError(
                    name + " exceeded its byte bound while being read"
                )
            digest.update(chunk)
            chunks.append(chunk)
        finished = os.fstat(descriptor)
        if _stat_identity(finished) != _stat_identity(opened):
            raise C0ArtifactValidationError(name + " changed while it was read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) != opened.st_size:
        raise C0ArtifactValidationError(name + " byte count changed while it was read")
    snapshot = _snapshot_from_stat(path, digest.hexdigest(), opened)
    _assert_snapshot_current(snapshot, name)
    return raw, snapshot


def _hash_regular_file_snapshot(
    path: Path, *, name: str, max_bytes: int
) -> _FileSnapshot:
    descriptor, opened = _open_regular_file(path, name=name, max_bytes=max_bytes)
    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > max_bytes:
                raise C0ArtifactValidationError(
                    name + " exceeded its byte bound while being hashed"
                )
            digest.update(chunk)
        finished = os.fstat(descriptor)
        if _stat_identity(finished) != _stat_identity(opened):
            raise C0ArtifactValidationError(name + " changed while it was hashed")
    finally:
        os.close(descriptor)
    if observed_bytes != opened.st_size:
        raise C0ArtifactValidationError(
            name + " byte count changed while it was hashed"
        )
    snapshot = _snapshot_from_stat(path, digest.hexdigest(), opened)
    _assert_snapshot_current(snapshot, name)
    return snapshot


def _reject_json_constant(value: str) -> None:
    raise C0ArtifactValidationError("non-finite JSON number is forbidden: " + value)


def _reject_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise C0ArtifactValidationError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _parse_canonical_json(raw: bytes, *, name: str) -> Mapping:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise C0ArtifactValidationError(name + " must be UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except C0ArtifactValidationError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise C0ArtifactValidationError(name + " is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise C0ArtifactValidationError(name + " root must be an object")
    if raw != _canonical_pretty_json(value):
        raise C0ArtifactValidationError(name + " is not canonical pretty JSON")
    return value


def _load_canonical_json_snapshot(
    path: Path, *, max_bytes: int, name: str
) -> Tuple[Mapping, _FileSnapshot]:
    raw, snapshot = _read_regular_file_snapshot(
        Path(path), max_bytes=max_bytes, name=name
    )
    return _parse_canonical_json(raw, name=name), snapshot


def _load_canonical_json(path: Path, *, max_bytes: int, name: str) -> Mapping:
    value, _snapshot = _load_canonical_json_snapshot(
        Path(path), max_bytes=max_bytes, name=name
    )
    return value


def _json_identical(left: Any, right: Any) -> bool:
    return hmac.compare_digest(_canonical_json(left), _canonical_json(right))


def _assert_no_nonfinite(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_nonfinite(item, "{}.{}".format(name, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_nonfinite(item, "{}[{}]".format(name, index))
    elif isinstance(value, float) and not math.isfinite(value):
        raise C0ArtifactValidationError(name + " is non-finite")


def _portable_string(value: str) -> bool:
    if value != value.strip() or any(ord(character) < 32 for character in value):
        return False
    normalized = value.replace("\\", "/")
    lowered = normalized.casefold()
    return not (
        normalized.startswith(("/", "~/", "//"))
        or lowered.startswith("file:")
        or normalized == ".."
        or normalized.startswith("../")
        or normalized.endswith("/..")
        or "/../" in normalized
        or _WINDOWS_ABSOLUTE_RE.match(value) is not None
    )


def _assert_portable(value: Any, name: str = "aggregate") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in {
                "archive_member",
                "canonical_group_id",
                "component_id",
                "fragment_id",
                "pair_id",
                "path",
            }:
                raise C0ArtifactValidationError(name + " exposes an identity/path key")
            _assert_portable(item, "{}.{}".format(name, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_portable(item, "{}[{}]".format(name, index))
    elif isinstance(value, str) and not _portable_string(value):
        raise C0ArtifactValidationError(name + " contains an absolute/local path")


def _validate_self_hash(value: Mapping, name: str) -> str:
    stored = _require_sha256(value.get("content_sha256"), name + " content hash")
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    observed = _content_sha256(unsigned)
    if not hmac.compare_digest(stored, observed):
        raise C0ArtifactValidationError(name + " self content hash mismatch")
    return observed


def _checkpoint_names() -> Tuple[str, ...]:
    return tuple(
        "checkpoint-epoch-{:02d}.pt".format(epoch)
        for epoch in range(1, _CHECKPOINT_COUNT + 1)
    )


def _validate_run_directory(path: Path) -> Tuple[str, ...]:
    run_dir = Path(path)
    try:
        metadata = run_dir.lstat()
    except OSError as exc:
        raise C0ArtifactValidationError("run directory is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise C0ArtifactValidationError("run directory must be a non-symlink directory")
    expected = set(_checkpoint_names()).union(
        {"checkpoint_sidecar.local.json", "c0_run_receipt.json"}
    )
    try:
        entries = list(os.scandir(str(run_dir)))
    except OSError as exc:
        raise C0ArtifactValidationError("run directory cannot be listed") from exc
    observed = {entry.name for entry in entries}
    if len(entries) != len(observed) or observed != expected:
        raise C0ArtifactValidationError(
            "run artifact allowlist differs (missing={}, extra={})".format(
                sorted(expected - observed), sorted(observed - expected)
            )
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise C0ArtifactValidationError(
                "run artifact is not a non-symlink regular file: " + entry.name
            )
    return tuple(sorted(observed))


def _validate_plan(plan: Mapping, policy: _ValidationPolicy) -> None:
    _require_exact_keys(
        plan,
        (
            "content_sha256",
            "contract",
            "experiment",
            "freeze_content_sha256",
            "identity_index",
            "roles",
            "schema_version",
            "scope",
            "source_bundle",
            "status",
        ),
        "run plan",
    )
    _validate_self_hash(plan, "run plan")
    if plan["content_sha256"] != policy.plan_content_sha256:
        raise C0ArtifactValidationError("run-plan semantic content lock mismatch")
    if (
        plan["schema_version"] != RUN_PLAN_SCHEMA_VERSION
        or plan["status"] != "frozen_no_execution"
        or plan["experiment"] != "C0-N-Q1"
    ):
        raise C0ArtifactValidationError("run-plan schema/status/experiment changed")
    if not _json_identical(plan["contract"], policy.contract):
        raise C0ArtifactValidationError("run-plan contract differs from expected mode")
    plan_scope = _require_exact_keys(
        plan["scope"],
        (
            "archive_bytes_hashed",
            "archive_members_opened",
            "historical_test_access",
            "mask_pixels_decoded",
            "model_or_backend_created",
            "sealed_real_read",
        ),
        "run-plan scope",
    )
    if (
        plan_scope["archive_bytes_hashed"] is not True
        or plan_scope["archive_members_opened"] is not False
        or plan_scope["mask_pixels_decoded"] is not False
        or plan_scope["model_or_backend_created"] is not False
        or plan_scope["sealed_real_read"] is not False
        or not _json_identical(
            plan_scope["historical_test_access"], _HISTORICAL_TEST_ACCESS
        )
    ):
        raise C0ArtifactValidationError("run-plan scope admits forbidden access")
    _require_sha256(plan["freeze_content_sha256"], "run-plan freeze content hash")
    identity = _require_exact_keys(
        plan["identity_index"],
        ("content_sha256", "member_count", "schema_version"),
        "run-plan identity index",
    )
    _require_sha256(identity["content_sha256"], "identity-index content hash")
    if type(identity["member_count"]) is not int or identity["member_count"] <= 0:
        raise C0ArtifactValidationError("identity-index member count is invalid")

    roles = plan["roles"]
    if not isinstance(roles, list) or len(roles) != 7:
        raise C0ArtifactValidationError("run plan must contain exactly seven roles")
    role_names = set()
    for index, role in enumerate(roles):
        row = _require_exact_keys(
            role,
            ("bytes", "hash_mode", "kind", "logical_id", "role", "sha256"),
            "run-plan role {}".format(index),
        )
        if not isinstance(row["role"], str) or not row["role"]:
            raise C0ArtifactValidationError("run-plan role name is invalid")
        if row["role"] in role_names:
            raise C0ArtifactValidationError("run-plan roles are duplicated")
        role_names.add(row["role"])
        if type(row["bytes"]) is not int or row["bytes"] <= 0:
            raise C0ArtifactValidationError("run-plan role byte count is invalid")
        _require_sha256(row["sha256"], "run-plan role SHA-256")
        for field in ("hash_mode", "kind", "logical_id"):
            if not isinstance(row[field], str) or not row[field]:
                raise C0ArtifactValidationError(
                    "run-plan role field is invalid: " + field
                )
    required_roles = {
        "freeze_receipt",
        "mm_archive",
        "eccv_archive",
        "mm_fingerprint_cache",
        "eccv_fingerprint_cache",
        "historical_split",
        "source_bundle",
    }
    if role_names != required_roles:
        raise C0ArtifactValidationError("run-plan role set changed")

    if policy.enforce_production_plan_locks:
        if plan["freeze_content_sha256"] != PRODUCTION_FREEZE_CONTENT_SHA256:
            raise C0ArtifactValidationError("production freeze content lock changed")
        if identity["content_sha256"] != PRODUCTION_IDENTITY_INDEX_CONTENT_SHA256:
            raise C0ArtifactValidationError("production identity lock changed")
        freeze_role = next(row for row in roles if row["role"] == "freeze_receipt")
        if freeze_role["sha256"] != PRODUCTION_FREEZE_FILE_SHA256:
            raise C0ArtifactValidationError("production freeze file lock changed")


def _validate_environment(receipt: Mapping, policy: _ValidationPolicy) -> None:
    environment = _require_exact_keys(
        receipt["environment"],
        (
            "cuda",
            "cudnn",
            "determinism",
            "device",
            "os",
            "python",
            "schema_version",
            "torch",
        ),
        "runtime environment",
    )
    if environment["schema_version"] != ENVIRONMENT_SCHEMA_VERSION:
        raise C0ArtifactValidationError("runtime environment schema changed")
    observed_hash = _content_sha256(environment)
    expected_hash = _require_sha256(
        receipt["environment_sha256"], "runtime environment hash"
    )
    if not hmac.compare_digest(observed_hash, expected_hash):
        raise C0ArtifactValidationError("runtime environment hash mismatch")
    device = _require_exact_keys(
        environment["device"],
        (
            "compute_capability",
            "index",
            "name",
            "total_memory_bytes",
            "type",
        ),
        "runtime device",
    )
    if device["type"] != policy.device_type:
        raise C0ArtifactValidationError(
            "runtime device type differs from expected mode"
        )
    backend = receipt["backend"]
    if backend.get("device_type") != device["type"]:
        raise C0ArtifactValidationError("backend and environment device types differ")
    determinism = _require_exact_keys(
        environment["determinism"],
        ("cublas_workspace_config",),
        "runtime determinism",
    )
    cublas = _require_exact_keys(
        determinism["cublas_workspace_config"],
        ("status", "value"),
        "CUBLAS workspace evidence",
    )
    if policy.device_type == "cuda":
        if (
            cublas["status"] != "validated"
            or cublas["value"] != policy.cublas_workspace_config
        ):
            raise C0ArtifactValidationError("production CUBLAS evidence changed")
        cuda = _require_exact_keys(
            environment["cuda"],
            ("available", "driver", "runtime_version"),
            "CUDA environment",
        )
        if cuda["available"] is not True or not isinstance(
            cuda["runtime_version"], str
        ):
            raise C0ArtifactValidationError("CUDA runtime evidence is incomplete")
        driver = _require_exact_keys(
            cuda["driver"], ("status", "version"), "CUDA driver evidence"
        )
        if (
            driver["status"] != "observed"
            or not isinstance(driver["version"], str)
            or _DRIVER_RE.fullmatch(driver["version"]) is None
        ):
            raise C0ArtifactValidationError("CUDA driver was not observed")
        if (
            type(device["index"]) is not int
            or device["index"] < 0
            or not isinstance(device["name"], str)
            or not device["name"]
            or type(device["total_memory_bytes"]) is not int
            or device["total_memory_bytes"] <= 0
        ):
            raise C0ArtifactValidationError("CUDA device evidence is invalid")
        capability = _require_exact_keys(
            device["compute_capability"], ("major", "minor"), "compute capability"
        )
        if any(
            type(capability[key]) is not int or capability[key] < 0
            for key in capability
        ):
            raise C0ArtifactValidationError("CUDA compute capability is invalid")
    else:
        if cublas != {"status": "not_applicable", "value": None}:
            raise C0ArtifactValidationError(
                "CPU fixture has unexpected CUBLAS evidence"
            )


def _validate_metrics(metrics: Any, name: str) -> Mapping:
    metric_keys = (
        "accuracy",
        "auprc",
        "auroc",
        "brier",
        "ece",
        "f1",
        "false_positive_rate",
        "fn_weight",
        "fp_weight",
        "precision",
        "recall",
        "specificity",
        "tn_weight",
        "tp_weight",
    )
    value = _require_exact_keys(metrics, metric_keys, name)
    for key in metric_keys:
        observed = value[key]
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            raise C0ArtifactValidationError("{}.{} is not numeric".format(name, key))
        number = float(observed)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise C0ArtifactValidationError("{}.{} is outside [0,1]".format(name, key))
    return value


def _validate_validation_report(
    report: Any,
    *,
    expected_count: int,
    expected_dataset_counts: Optional[Mapping[str, int]],
    expected_cluster_counts: Optional[Mapping[str, int]],
    name: str,
) -> Mapping:
    value = _require_exact_keys(
        report,
        (
            "all_rows_valid",
            "by_dataset",
            "count",
            "equal_domain_macro_cluster",
            "prediction_commitment_sha256",
            "selection_threshold_used",
        ),
        name,
    )
    _require_builtin_int(value["count"], expected_count, name + " count")
    if value["all_rows_valid"] is not True:
        raise C0ArtifactValidationError(name + " does not attest all rows valid")
    if value["selection_threshold_used"] is not False:
        raise C0ArtifactValidationError(name + " used a selection threshold")
    _require_sha256(value["prediction_commitment_sha256"], name + " prediction hash")
    by_dataset = _require_exact_keys(value["by_dataset"], _DATASETS, name + " datasets")
    cluster_values: List[Tuple[float, float]] = []
    sample_total = 0
    for dataset in _DATASETS:
        row = _require_exact_keys(
            by_dataset[dataset],
            (
                "cluster_balanced",
                "cluster_count",
                "negative_count",
                "positive_count",
                "row",
                "sample_count",
                "schema_version",
                "threshold",
            ),
            "{} {}".format(name, dataset),
        )
        if row["schema_version"] != "dunhuang-pairwise-evaluation/0.2":
            raise C0ArtifactValidationError(name + " evaluation schema changed")
        for count_name in (
            "cluster_count",
            "negative_count",
            "positive_count",
            "sample_count",
        ):
            if type(row[count_name]) is not int or row[count_name] <= 0:
                raise C0ArtifactValidationError(name + " dataset count is invalid")
        if row["positive_count"] + row["negative_count"] != row["sample_count"]:
            raise C0ArtifactValidationError(name + " dataset labels do not sum")
        if expected_cluster_counts is not None:
            _require_builtin_int(
                row["cluster_count"],
                int(expected_cluster_counts[dataset]),
                name + " " + dataset + " cluster count",
            )
        if row["threshold"] != 0.5:
            raise C0ArtifactValidationError(name + " evaluation threshold changed")
        if expected_dataset_counts is not None:
            positive_key = dataset + "|positive"
            negative_key = dataset + "|negative"
            _require_builtin_int(
                row["positive_count"],
                int(expected_dataset_counts[positive_key]),
                name + " " + positive_key,
            )
            _require_builtin_int(
                row["negative_count"],
                int(expected_dataset_counts[negative_key]),
                name + " " + negative_key,
            )
        _validate_metrics(row["row"], "{} {} row metrics".format(name, dataset))
        cluster = _validate_metrics(
            row["cluster_balanced"],
            "{} {} cluster metrics".format(name, dataset),
        )
        cluster_values.append((float(cluster["auroc"]), float(cluster["auprc"])))
        sample_total += row["sample_count"]
    if sample_total != expected_count:
        raise C0ArtifactValidationError(name + " dataset samples do not sum")
    macro = _require_exact_keys(
        value["equal_domain_macro_cluster"], ("auprc", "auroc"), name + " macro"
    )
    expected_auroc = sum(item[0] for item in cluster_values) / len(cluster_values)
    expected_auprc = sum(item[1] for item in cluster_values) / len(cluster_values)
    for metric, expected in (("auroc", expected_auroc), ("auprc", expected_auprc)):
        observed = macro[metric]
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not math.isfinite(float(observed))
            or not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise C0ArtifactValidationError(name + " macro " + metric + " mismatch")
    return value


def _validate_checkpoint_claim(
    value: Any,
    epoch: int,
    name: str,
    *,
    optimizer_required: bool = False,
) -> Mapping:
    row = _require_exact_keys(
        value,
        (
            "canonical_content_sha256",
            "config_hash",
            "epoch",
            "file_sha256",
            "model_state_sha256",
            "optimizer_state_sha256",
        ),
        name,
    )
    _require_builtin_int(row["epoch"], epoch, name + " epoch")
    for key in (
        "canonical_content_sha256",
        "config_hash",
        "file_sha256",
        "model_state_sha256",
    ):
        _require_sha256(row[key], name + " " + key)
    if optimizer_required and row["optimizer_state_sha256"] is None:
        raise C0ArtifactValidationError(
            name + " optimizer_state_sha256 is required for production AdamW"
        )
    if row["optimizer_state_sha256"] is not None:
        _require_sha256(row["optimizer_state_sha256"], name + " optimizer hash")
    return row


def _expected_checkpoint_config_hash(
    receipt: Mapping, policy: _ValidationPolicy
) -> str:
    """Rebuild the runner's exact portable ``_session_config`` commitment."""

    return _content_sha256(
        {
            "runner": receipt["contract"],
            "provider": receipt["provider"],
            "backend": receipt["backend"],
            "environment": receipt["environment"],
            "model": policy.expected_model_config,
            "optimizer": policy.expected_optimizer_config,
        }
    )


def _validate_provider(
    receipt: Mapping, plan: Mapping, policy: _ValidationPolicy
) -> None:
    provider = _require_exact_keys(
        receipt["provider"],
        (
            "archive_locks_sha256",
            "archives_verified",
            "identity_index_content_sha256",
            "mask_pixels_loaded_during_attestation",
            "preprocessing_sha256",
            "provider_version",
            "sealed_real_test_capability",
        ),
        "provider attestation",
    )
    for field in ("archive_locks_sha256", "preprocessing_sha256"):
        _require_sha256(provider[field], "provider " + field)
    plan_by_role = {row["role"]: row for row in plan["roles"]}
    reconstructed_archive_locks = {
        "eccv_1113data": {
            "bytes": plan_by_role["eccv_archive"]["bytes"],
            "format": "tar",
            "sha256": plan_by_role["eccv_archive"]["sha256"],
        },
        "mm_augmented": {
            "bytes": plan_by_role["mm_archive"]["bytes"],
            "format": "zip",
            "sha256": plan_by_role["mm_archive"]["sha256"],
        },
    }
    if provider["archive_locks_sha256"] != _content_sha256(reconstructed_archive_locks):
        raise C0ArtifactValidationError(
            "provider archive-lock hash differs from reconstructed plan locks"
        )
    if provider["provider_version"] != PROVIDER_ATTESTATION_VERSION:
        raise C0ArtifactValidationError("provider attestation version changed")
    if (
        provider["archives_verified"] is not True
        or provider["mask_pixels_loaded_during_attestation"] is not False
        or provider["sealed_real_test_capability"] is not False
    ):
        raise C0ArtifactValidationError("provider attestation is unsafe")
    expected_identity = plan["identity_index"]["content_sha256"]
    if provider["identity_index_content_sha256"] != expected_identity:
        raise C0ArtifactValidationError("provider identity-index lock mismatch")

    final = _require_exact_keys(
        receipt["provider_final_evidence"],
        (
            "archive_verification",
            "counters",
            "freeze_content_sha256",
            "geometry_cache_local_sinkhorn_calls",
            "historical_test_access",
            "identity_index_content_sha256",
            "memo_bounds",
            "preprocessing",
            "provider_version",
            "schema_version",
            "sealed_real_read",
            "status",
        ),
        "provider final evidence",
    )
    if (
        final["provider_version"] != PROVIDER_BASE_VERSION
        or provider["provider_version"]
        != final["provider_version"] + "+" + PROVIDER_ADAPTER_VERSION
    ):
        raise C0ArtifactValidationError("provider base/adapter version mapping changed")
    if (
        final["schema_version"] != PROVIDER_RECEIPT_SCHEMA_VERSION
        or final["status"] != "archive_preflight_complete"
        or final["freeze_content_sha256"] != plan["freeze_content_sha256"]
        or final["identity_index_content_sha256"] != expected_identity
        or final["geometry_cache_local_sinkhorn_calls"] != 0
        or final["sealed_real_read"] is not False
        or not _json_identical(final["historical_test_access"], _HISTORICAL_TEST_ACCESS)
    ):
        raise C0ArtifactValidationError("provider final evidence violates C0 scope")
    if "historical_test_read" in final:
        raise C0ArtifactValidationError(
            "provider exposes historical-test read evidence"
        )
    preprocessing = final["preprocessing"]
    if not isinstance(preprocessing, Mapping):
        raise C0ArtifactValidationError("provider preprocessing evidence is absent")
    for key in ("geometry_config_sha256", "preprocessing_sha256"):
        _require_sha256(preprocessing.get(key), "provider preprocessing " + key)
    if preprocessing["preprocessing_sha256"] != provider["preprocessing_sha256"]:
        raise C0ArtifactValidationError("provider preprocessing locks differ")
    if policy.expected_preprocessing is not None and not _json_identical(
        preprocessing, policy.expected_preprocessing
    ):
        raise C0ArtifactValidationError(
            "provider preprocessing differs from frozen tight-crop letterbox"
        )

    memo = _require_exact_keys(
        final["memo_bounds"],
        ("max_batch_size", "max_cache_bytes", "max_cached_fragments"),
        "provider memo bounds",
    )
    if not _json_identical(memo, policy.memo_bounds):
        raise C0ArtifactValidationError("provider memo bounds changed")
    counters = _require_exact_keys(
        final["counters"],
        (
            "archive_decode_count",
            "batch_count",
            "coarse_preprocess_count",
            "fragment_request_count",
            "loader_cache_hit_count",
            "mask_loader_call_count",
            "memo_byte_count",
            "memo_entry_count",
            "memo_eviction_count",
            "memo_hit_count",
            "memo_miss_count",
            "record_count",
        ),
        "provider counters",
    )
    for key, value in counters.items():
        if type(value) is not int or value < 0:
            raise C0ArtifactValidationError("provider counter is invalid: " + key)
    for key, expected in policy.provider_counts.items():
        _require_builtin_int(counters[key], int(expected), "provider " + key)
    if counters["memo_eviction_count"] != 0:
        raise C0ArtifactValidationError("provider memo eviction count is not zero")
    if counters["memo_entry_count"] != counters["memo_miss_count"]:
        raise C0ArtifactValidationError(
            "zero-eviction provider memo entries must equal memo misses"
        )
    if counters["memo_byte_count"] != counters["memo_entry_count"] * 65_536:
        raise C0ArtifactValidationError(
            "provider memo bytes differ from fixed float32[1,128,128] entries"
        )
    if (
        counters["memo_hit_count"] + counters["memo_miss_count"]
        != counters["fragment_request_count"]
    ):
        raise C0ArtifactValidationError("provider memo hit/miss counts do not sum")
    if not (
        counters["memo_miss_count"]
        == counters["mask_loader_call_count"]
        == counters["coarse_preprocess_count"]
    ):
        raise C0ArtifactValidationError(
            "provider miss/decode preprocessing counts differ"
        )
    if (
        counters["archive_decode_count"] + counters["loader_cache_hit_count"]
        != counters["mask_loader_call_count"]
    ):
        raise C0ArtifactValidationError("provider loader counts do not sum")
    if (
        counters["memo_entry_count"] > memo["max_cached_fragments"]
        or counters["memo_byte_count"] > memo["max_cache_bytes"]
    ):
        raise C0ArtifactValidationError("provider final memo footprint exceeds bounds")

    archive_roles = {
        row["role"]: row
        for row in plan["roles"]
        if row["role"] in {"mm_archive", "eccv_archive"}
    }
    expected_provider_ids = policy.expected_archive_role_to_provider_logical_id
    expected_archive_roles = {"mm_archive", "eccv_archive"}
    if (
        not isinstance(expected_provider_ids, Mapping)
        or set(expected_provider_ids) != expected_archive_roles
        or any(
            not isinstance(logical_id, str) or not logical_id
            for logical_id in expected_provider_ids.values()
        )
        or len(set(expected_provider_ids.values())) != len(expected_provider_ids)
    ):
        raise C0ArtifactValidationError(
            "provider archive role/logical-ID policy is invalid"
        )
    provider_id_to_role = {
        logical_id: role for role, logical_id in expected_provider_ids.items()
    }
    if set(archive_roles) != expected_archive_roles:
        raise C0ArtifactValidationError("run-plan archive role set changed")
    if (
        policy.expected_archive_evidence is not None
        and set(policy.expected_archive_evidence) != expected_archive_roles
    ):
        raise C0ArtifactValidationError(
            "provider archive format/mode policy role set changed"
        )
    archive_rows = final["archive_verification"]
    if not isinstance(archive_rows, list) or len(archive_rows) != 2:
        raise C0ArtifactValidationError("provider must attest exactly two archives")
    seen_roles = set()
    for row in archive_rows:
        item = _require_exact_keys(
            row,
            (
                "archive_format",
                "byte_count",
                "expected_sha256",
                "logical_id",
                "observed_sha256",
                "verification_mode",
            ),
            "provider archive verification",
        )
        logical_id = item["logical_id"]
        role_name = provider_id_to_role.get(logical_id)
        if role_name is None or role_name in seen_roles:
            raise C0ArtifactValidationError("provider archive logical ID changed")
        seen_roles.add(role_name)
        role = archive_roles[role_name]
        if (
            item["expected_sha256"] != role["sha256"]
            or item["observed_sha256"] != role["sha256"]
            or item["byte_count"] != role["bytes"]
        ):
            raise C0ArtifactValidationError(
                "provider archive evidence differs from plan"
            )
        if policy.expected_archive_evidence is not None:
            expected_format, expected_mode = policy.expected_archive_evidence[role_name]
            if (
                item["archive_format"] != expected_format
                or item["verification_mode"] != expected_mode
            ):
                raise C0ArtifactValidationError(
                    "provider archive format/verification mode changed"
                )
    if seen_roles != expected_archive_roles:
        raise C0ArtifactValidationError("provider archive role evidence is incomplete")


def _validate_metadata(receipt: Mapping, policy: _ValidationPolicy) -> None:
    metadata = _require_exact_keys(
        receipt["metadata_attestation"],
        (
            "runtime_overlap",
            "train_by_dataset_label",
            "train_count",
            "train_order_commitment_sha256",
            "validation_by_dataset_label",
            "validation_count",
            "validation_order_commitment_sha256",
        ),
        "metadata attestation",
    )
    _require_builtin_int(
        metadata["train_count"], int(policy.contract["train_count"]), "train count"
    )
    _require_builtin_int(
        metadata["validation_count"],
        int(policy.contract["validation_count"]),
        "validation count",
    )
    for key in ("train_order_commitment_sha256", "validation_order_commitment_sha256"):
        _require_sha256(metadata[key], "metadata " + key)
    if (
        policy.expected_train_order_commitment_sha256 is not None
        and metadata["train_order_commitment_sha256"]
        != policy.expected_train_order_commitment_sha256
    ):
        raise C0ArtifactValidationError(
            "training order commitment differs from frozen population"
        )
    if (
        policy.expected_validation_order_commitment_sha256 is not None
        and metadata["validation_order_commitment_sha256"]
        != policy.expected_validation_order_commitment_sha256
    ):
        raise C0ArtifactValidationError(
            "validation order commitment differs from frozen population"
        )
    overlap = _require_exact_keys(
        metadata["runtime_overlap"],
        ("component", "content_sha256", "member"),
        "runtime train/validation overlap",
    )
    if any(type(value) is not int or value != 0 for value in overlap.values()):
        raise C0ArtifactValidationError("runtime train/validation overlap is nonzero")
    if policy.expected_train_counts is not None and not _json_identical(
        metadata["train_by_dataset_label"], policy.expected_train_counts
    ):
        raise C0ArtifactValidationError("training dataset/label counts changed")
    if policy.expected_validation_counts is not None and not _json_identical(
        metadata["validation_by_dataset_label"], policy.expected_validation_counts
    ):
        raise C0ArtifactValidationError("validation dataset/label counts changed")


def _validate_receipt_locks(
    receipt: Mapping, plan: Mapping, policy: _ValidationPolicy
) -> None:
    locks = _require_exact_keys(
        receipt["locks"],
        (
            "freeze_content_sha256",
            "freeze_file_sha256",
            "production_run_plan_content_sha256",
            "production_run_plan_file_sha256",
            "source_files",
            "source_lock_commitment_sha256",
        ),
        "run receipt locks",
    )
    if (
        locks["production_run_plan_file_sha256"] != policy.plan_file_sha256
        or locks["production_run_plan_content_sha256"] != policy.plan_content_sha256
    ):
        raise C0ArtifactValidationError("run receipt does not bind canonical run plan")
    freeze_role = next(row for row in plan["roles"] if row["role"] == "freeze_receipt")
    if (
        locks["freeze_file_sha256"] != freeze_role["sha256"]
        or locks["freeze_content_sha256"] != plan["freeze_content_sha256"]
    ):
        raise C0ArtifactValidationError("run receipt freeze locks differ from plan")
    source_files = locks["source_files"]
    if not isinstance(source_files, list) or not source_files:
        raise C0ArtifactValidationError("run receipt source locks are absent")
    source_commit = _require_sha256(
        locks["source_lock_commitment_sha256"], "source lock commitment"
    )
    if source_commit != _content_sha256(source_files):
        raise C0ArtifactValidationError("source lock commitment is inconsistent")
    if policy.enforce_production_plan_locks and not _json_identical(
        source_files, sorted(plan["roles"], key=lambda row: row["role"])
    ):
        raise C0ArtifactValidationError("run receipt source locks differ from run plan")


def _validate_epochs_and_winner(
    receipt: Mapping, policy: _ValidationPolicy
) -> Tuple[List[Mapping], Mapping]:
    expected_checkpoint_config_hash = _expected_checkpoint_config_hash(receipt, policy)
    observed = _require_exact_keys(
        receipt["observed"],
        (
            "batches_per_epoch",
            "checkpoint_count",
            "epoch_count",
            "full_validation_replay_count",
            "total_steps",
        ),
        "observed schedule",
    )
    expected_schedule = {
        "batches_per_epoch": int(policy.contract["batches_per_epoch"]),
        "checkpoint_count": _CHECKPOINT_COUNT,
        "epoch_count": _CHECKPOINT_COUNT,
        "full_validation_replay_count": _CHECKPOINT_COUNT + 1,
        "total_steps": int(policy.contract["total_steps"]),
    }
    if not _json_identical(observed, expected_schedule):
        raise C0ArtifactValidationError("observed epoch/step/replay schedule changed")
    epochs = receipt["epochs"]
    if not isinstance(epochs, list) or len(epochs) != _CHECKPOINT_COUNT:
        raise C0ArtifactValidationError("run receipt must contain exactly five epochs")
    checkpoint_config_hash: Optional[str] = None
    for expected_epoch, row_value in enumerate(epochs, 1):
        row = _require_exact_keys(
            row_value,
            ("checkpoint", "epoch", "train", "validation"),
            "epoch row {}".format(expected_epoch),
        )
        _require_builtin_int(row["epoch"], expected_epoch, "epoch row number")
        train = _require_exact_keys(
            row["train"],
            (
                "batch_count",
                "epoch_order_commitment_sha256",
                "mean_loss",
                "step_count_cumulative",
            ),
            "epoch {} train".format(expected_epoch),
        )
        _require_builtin_int(
            train["batch_count"],
            int(policy.contract["batches_per_epoch"]),
            "epoch {} batch count".format(expected_epoch),
        )
        _require_builtin_int(
            train["step_count_cumulative"],
            expected_epoch * int(policy.contract["batches_per_epoch"]),
            "epoch {} cumulative steps".format(expected_epoch),
        )
        _require_sha256(
            train["epoch_order_commitment_sha256"], "epoch order commitment"
        )
        if (
            not isinstance(train["mean_loss"], (int, float))
            or isinstance(train["mean_loss"], bool)
            or not math.isfinite(float(train["mean_loss"]))
            or float(train["mean_loss"]) < 0.0
        ):
            raise C0ArtifactValidationError("epoch mean loss is invalid")
        checkpoint = _validate_checkpoint_claim(
            row["checkpoint"],
            expected_epoch,
            "epoch checkpoint",
            optimizer_required=policy.require_optimizer_state_sha256,
        )
        if checkpoint["config_hash"] != expected_checkpoint_config_hash:
            raise C0ArtifactValidationError(
                "checkpoint config hash differs from reconstructed frozen session config"
            )
        if checkpoint_config_hash is None:
            checkpoint_config_hash = checkpoint["config_hash"]
        elif checkpoint["config_hash"] != checkpoint_config_hash:
            raise C0ArtifactValidationError("checkpoint configs differ across epochs")
        _validate_validation_report(
            row["validation"],
            expected_count=int(policy.contract["validation_count"]),
            expected_dataset_counts=policy.expected_validation_counts,
            expected_cluster_counts=policy.expected_cluster_counts,
            name="epoch {} validation".format(expected_epoch),
        )

    selected = max(
        epochs,
        key=lambda row: (
            float(row["validation"]["equal_domain_macro_cluster"]["auroc"]),
            float(row["validation"]["equal_domain_macro_cluster"]["auprc"]),
            -int(row["epoch"]),
        ),
    )
    winner = _require_exact_keys(
        receipt["winner"],
        (
            "checkpoint",
            "environment_sha256",
            "epoch",
            "full_validation_replay_match",
            "restricted_fresh_session_reload",
            "selection_policy",
            "validation",
        ),
        "winner",
    )
    _require_builtin_int(winner["epoch"], int(selected["epoch"]), "winner epoch")
    if winner["selection_policy"] != (
        "max_equal_domain_macro_cluster_auroc_then_auprc_then_earlier_epoch"
    ):
        raise C0ArtifactValidationError("winner selection policy changed")
    if (
        winner["restricted_fresh_session_reload"] is not True
        or winner["full_validation_replay_match"] is not True
    ):
        raise C0ArtifactValidationError("winner was not cleanly reloaded/replayed")
    if winner["environment_sha256"] != receipt["environment_sha256"]:
        raise C0ArtifactValidationError("winner environment hash changed")
    if not _json_identical(winner["checkpoint"], selected["checkpoint"]):
        raise C0ArtifactValidationError("winner checkpoint differs from selected epoch")
    if not _json_identical(winner["validation"], selected["validation"]):
        raise C0ArtifactValidationError("winner validation differs from selected epoch")
    _validate_checkpoint_claim(
        winner["checkpoint"],
        int(winner["epoch"]),
        "winner checkpoint",
        optimizer_required=policy.require_optimizer_state_sha256,
    )
    _validate_validation_report(
        winner["validation"],
        expected_count=int(policy.contract["validation_count"]),
        expected_dataset_counts=policy.expected_validation_counts,
        expected_cluster_counts=policy.expected_cluster_counts,
        name="winner validation",
    )
    return epochs, winner


def _validate_thresholds(
    receipt: Mapping, winner: Mapping, policy: _ValidationPolicy
) -> None:
    thresholds = _require_exact_keys(
        receipt["thresholds"],
        ("by_dataset", "pooled_threshold", "production_threshold", "status"),
        "thresholds",
    )
    if (
        thresholds["status"] != "validation_diagnostic_only_after_winner_selection"
        or thresholds["production_threshold"] is not None
        or thresholds["pooled_threshold"] is not None
    ):
        raise C0ArtifactValidationError("pooled/production threshold scope changed")
    by_dataset = _require_exact_keys(
        thresholds["by_dataset"], _DATASETS, "diagnostic thresholds"
    )
    expected_model_config_hash = _content_sha256(policy.expected_model_config)
    expected_aggregation_config_hash = _content_sha256(_DIAGNOSTIC_AGGREGATION_CONFIG)
    for dataset in _DATASETS:
        row = _require_exact_keys(
            by_dataset[dataset],
            (
                "achieved_cluster_balanced_f1",
                "achieved_cluster_balanced_precision",
                "achieved_cluster_balanced_recall",
                "aggregation_config_sha256",
                "checkpoint_sha256",
                "cluster_count",
                "fit_method",
                "model_config_sha256",
                "sample_count",
                "schema_version",
                "source_split",
                "threshold",
                "validation_fingerprint_sha256",
            ),
            "diagnostic threshold " + dataset,
        )
        for field in (
            "aggregation_config_sha256",
            "checkpoint_sha256",
            "model_config_sha256",
            "validation_fingerprint_sha256",
        ):
            _require_sha256(row[field], "diagnostic threshold " + field)
        if (
            row.get("schema_version") != "dunhuang-pairwise-threshold/0.2"
            or row.get("fit_method") != "maximize_cluster_balanced_f1"
            or row.get("source_split") != "validation"
            or row.get("checkpoint_sha256") != winner["checkpoint"]["file_sha256"]
            or row.get("validation_fingerprint_sha256")
            != policy.contract["validation_fingerprint"]
            or row.get("model_config_sha256") != expected_model_config_hash
            or row.get("aggregation_config_sha256") != expected_aggregation_config_hash
        ):
            raise C0ArtifactValidationError("diagnostic threshold binding changed")
        validation_dataset = winner["validation"]["by_dataset"][dataset]
        _require_builtin_int(
            row["sample_count"],
            int(validation_dataset["sample_count"]),
            "diagnostic threshold sample count",
        )
        _require_builtin_int(
            row["cluster_count"],
            int(validation_dataset["cluster_count"]),
            "diagnostic threshold cluster count",
        )
        threshold = row.get("threshold")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise C0ArtifactValidationError("diagnostic threshold is invalid")
        for metric in (
            "achieved_cluster_balanced_f1",
            "achieved_cluster_balanced_precision",
            "achieved_cluster_balanced_recall",
        ):
            value = row[metric]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise C0ArtifactValidationError(
                    "diagnostic threshold achieved metric is invalid: " + metric
                )


def _validate_checkpoint_files_and_sidecar(
    run_dir: Path, epochs: Sequence[Mapping], winner: Mapping
) -> Tuple[List[Mapping[str, Any]], _FileSnapshot, Tuple[_FileSnapshot, ...]]:
    sidecar_path = run_dir / "checkpoint_sidecar.local.json"
    sidecar, sidecar_snapshot = _load_canonical_json_snapshot(
        sidecar_path, max_bytes=1024 * 1024, name="checkpoint sidecar"
    )
    _require_exact_keys(
        sidecar,
        (
            "checkpoints",
            "local_only",
            "schema_version",
            "winner_epoch",
            "winner_path",
        ),
        "checkpoint sidecar",
    )
    if (
        sidecar["schema_version"] != SIDECAR_SCHEMA_VERSION
        or sidecar["local_only"] is not True
    ):
        raise C0ArtifactValidationError("checkpoint sidecar schema/scope changed")
    rows = sidecar["checkpoints"]
    if not isinstance(rows, list) or len(rows) != _CHECKPOINT_COUNT:
        raise C0ArtifactValidationError("checkpoint sidecar must contain five rows")
    artifact_rows: List[Mapping[str, Any]] = []
    checkpoint_snapshots: List[_FileSnapshot] = []
    for epoch, (epoch_row, sidecar_value) in enumerate(zip(epochs, rows), 1):
        sidecar_row = _require_exact_keys(
            sidecar_value,
            ("canonical_content_sha256", "epoch", "file_sha256", "path"),
            "checkpoint sidecar row",
        )
        _require_builtin_int(sidecar_row["epoch"], epoch, "sidecar epoch")
        filename = "checkpoint-epoch-{:02d}.pt".format(epoch)
        if (
            not isinstance(sidecar_row["path"], str)
            or Path(sidecar_row["path"]).name != filename
        ):
            raise C0ArtifactValidationError("checkpoint sidecar filename changed")
        claim = epoch_row["checkpoint"]
        if (
            sidecar_row["file_sha256"] != claim["file_sha256"]
            or sidecar_row["canonical_content_sha256"]
            != claim["canonical_content_sha256"]
        ):
            raise C0ArtifactValidationError(
                "checkpoint receipt/sidecar claims are inconsistent"
            )
        checkpoint_path = run_dir / filename
        checkpoint_snapshot = _hash_regular_file_snapshot(
            checkpoint_path,
            name="checkpoint " + filename,
            max_bytes=_MAX_CHECKPOINT_BYTES,
        )
        if not hmac.compare_digest(checkpoint_snapshot.sha256, claim["file_sha256"]):
            raise C0ArtifactValidationError(
                "checkpoint file SHA-256 mismatch: " + filename
            )
        checkpoint_snapshots.append(checkpoint_snapshot)
        artifact_rows.append(
            {
                "bytes": checkpoint_snapshot.byte_count,
                "file": filename,
                "file_sha256": checkpoint_snapshot.sha256,
            }
        )
    if sidecar["winner_epoch"] != winner["epoch"]:
        raise C0ArtifactValidationError("checkpoint sidecar winner epoch changed")
    expected_winner_name = "checkpoint-epoch-{:02d}.pt".format(winner["epoch"])
    if (
        not isinstance(sidecar["winner_path"], str)
        or Path(sidecar["winner_path"]).name != expected_winner_name
    ):
        raise C0ArtifactValidationError("checkpoint sidecar winner filename changed")
    return artifact_rows, sidecar_snapshot, tuple(checkpoint_snapshots)


def _validate_receipt_shape(receipt: Mapping, policy: _ValidationPolicy) -> None:
    _require_exact_keys(
        receipt,
        (
            "backend",
            "content_sha256",
            "contract",
            "environment",
            "environment_sha256",
            "epochs",
            "local_checkpoint_sidecar_written",
            "locks",
            "metadata_attestation",
            "observed",
            "provider",
            "provider_final_evidence",
            "schema_version",
            "scope",
            "status",
            "thresholds",
            "winner",
        ),
        "run receipt",
    )
    _validate_self_hash(receipt, "run receipt")
    _assert_portable(receipt, "run receipt")
    if (
        receipt["schema_version"] != RUN_SCHEMA_VERSION
        or receipt["status"] != "complete_validation_selected_no_test_evaluation"
        or receipt["local_checkpoint_sidecar_written"] is not True
    ):
        raise C0ArtifactValidationError(
            "run receipt is not a completed C0 validation run"
        )
    if not _json_identical(receipt["contract"], policy.contract):
        raise C0ArtifactValidationError(
            "run receipt contract differs from expected mode"
        )
    scope = _require_exact_keys(
        receipt["scope"],
        (
            "experiment",
            "historical_test_records_accepted",
            "sealed_real_records_accepted",
            "threshold_used_for_epoch_selection",
        ),
        "run receipt scope",
    )
    if scope != {
        "experiment": "C0-N-Q1",
        "historical_test_records_accepted": False,
        "sealed_real_records_accepted": False,
        "threshold_used_for_epoch_selection": False,
    }:
        raise C0ArtifactValidationError(
            "run receipt scope admits test/sealed/threshold use"
        )
    backend = _require_exact_keys(
        receipt["backend"],
        (
            "backend_version",
            "deterministic_reload",
            "device_type",
            "model_family",
            "sealed_real_test_capability",
        ),
        "backend attestation",
    )
    if (
        backend["backend_version"] != policy.expected_backend_version
        or backend["model_family"] != policy.expected_backend_model_family
        or backend["device_type"] != policy.device_type
        or backend["deterministic_reload"] is not True
        or backend["sealed_real_test_capability"] is not False
    ):
        raise C0ArtifactValidationError("backend attestation identity/safety changed")
    _assert_no_nonfinite(receipt, "run receipt")


def _validate_with_policy(
    run_dir: Path,
    canonical_run_plan_path: Path,
    *,
    policy: _ValidationPolicy,
    aggregate_receipt_path: Optional[Path],
) -> Mapping[str, Any]:
    run_dir = Path(run_dir)
    plan_path = Path(canonical_run_plan_path)
    allowlist = _validate_run_directory(run_dir)
    plan, plan_snapshot = _load_canonical_json_snapshot(
        plan_path, max_bytes=16 * 1024 * 1024, name="run plan"
    )
    observed_plan_file_hash = plan_snapshot.sha256
    if not hmac.compare_digest(observed_plan_file_hash, policy.plan_file_sha256):
        raise C0ArtifactValidationError("canonical run-plan file SHA-256 mismatch")
    _validate_plan(plan, policy)

    receipt_path = run_dir / "c0_run_receipt.json"
    receipt, receipt_snapshot = _load_canonical_json_snapshot(
        receipt_path, max_bytes=32 * 1024 * 1024, name="run receipt"
    )
    _validate_receipt_shape(receipt, policy)
    _validate_receipt_locks(receipt, plan, policy)
    _validate_provider(receipt, plan, policy)
    _validate_environment(receipt, policy)
    _validate_metadata(receipt, policy)
    epochs, winner = _validate_epochs_and_winner(receipt, policy)
    _validate_thresholds(receipt, winner, policy)
    (
        checkpoint_artifacts,
        sidecar_snapshot,
        checkpoint_snapshots,
    ) = _validate_checkpoint_files_and_sidecar(run_dir, epochs, winner)

    receipt_file_hash = receipt_snapshot.sha256
    artifacts = checkpoint_artifacts + [
        {
            "bytes": receipt_snapshot.byte_count,
            "file": "c0_run_receipt.json",
            "file_sha256": receipt_file_hash,
        },
        {
            "bytes": sidecar_snapshot.byte_count,
            "file": "checkpoint_sidecar.local.json",
            "file_sha256": sidecar_snapshot.sha256,
        },
    ]
    artifacts.sort(key=lambda row: row["file"])
    aggregate: Dict[str, Any] = {
        "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
        "status": "pass_structural_artifact_consistency_only",
        "validation_mode": policy.mode,
        "scope": {
            "artifact_allowlist_exact": list(allowlist),
            "artifact_provenance_authenticated": False,
            "checkpoint_deserialized": False,
            "checkpoint_payload_authenticated": False,
            "historical_test_read": False,
            "model_executed": False,
            "raw_predictions_recomputed": False,
            "receipt_signature_verified": False,
            "result_validity_certified": False,
            "sealed_real_read": False,
        },
        "run_plan": {
            "content_sha256": policy.plan_content_sha256,
            "file_sha256": observed_plan_file_hash,
        },
        "run_receipt": {
            "content_sha256": receipt["content_sha256"],
            "file_sha256": receipt_file_hash,
        },
        "contract": dict(policy.contract),
        "observed": dict(receipt["observed"]),
        "winner": {
            "checkpoint_file_sha256": winner["checkpoint"]["file_sha256"],
            "epoch": winner["epoch"],
            "equal_domain_macro_cluster": dict(
                winner["validation"]["equal_domain_macro_cluster"]
            ),
        },
        "thresholds": {
            "fit_recomputed": False,
            "metrics_recomputed": False,
            "pooled_threshold": None,
            "production_threshold": None,
        },
        "artifacts": artifacts,
        "checkpoint_content_assurance": {
            "canonical_content_recomputed": False,
            "canonical_content_scope": (
                "receipt_sidecar_cross_consistency_only_not_checkpoint_self_proof"
            ),
            "artifact_provenance_authenticated": False,
            "checkpoint_payload_authenticated": False,
            "checkpoint_bytes_hashed": True,
            "deserialization_api_used": None,
            "filesystem_snapshot_scope": (
                "single_descriptor_reads_with_post_validation_identity_recheck"
            ),
            "raw_predictions_recomputed": False,
            "receipt_signature_verified": False,
            "result_validity_certified": False,
            "threshold_fit_recomputed": False,
        },
    }
    _assert_portable(aggregate)
    aggregate["content_sha256"] = _content_sha256(aggregate)

    # Keep every file bound to the same inode/content metadata from its single
    # descriptor-backed read/hash, then re-check the exact run-directory set.
    for snapshot, name in (
        (plan_snapshot, "run plan"),
        (receipt_snapshot, "run receipt"),
        (sidecar_snapshot, "checkpoint sidecar"),
    ):
        _assert_snapshot_current(snapshot, name)
    for snapshot in checkpoint_snapshots:
        _assert_snapshot_current(snapshot, "checkpoint " + snapshot.path.name)
    if _validate_run_directory(run_dir) != allowlist:
        raise C0ArtifactValidationError("run artifact set changed during validation")

    if aggregate_receipt_path is not None:
        _write_aggregate_receipt(
            Path(aggregate_receipt_path), aggregate, run_dir=run_dir
        )
    return aggregate


def _write_aggregate_receipt(
    path: Path, value: Mapping[str, Any], *, run_dir: Path
) -> None:
    candidate = Path(path)
    try:
        candidate.resolve().relative_to(run_dir.resolve())
    except ValueError:
        pass
    else:
        raise C0ArtifactValidationError(
            "aggregate receipt must remain outside the exact run artifact directory"
        )
    parent = candidate.parent
    if not parent.is_dir() or parent.is_symlink():
        raise C0ArtifactValidationError("aggregate receipt parent is not a directory")
    payload = _canonical_pretty_json(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(str(candidate), flags, 0o600)
    except OSError as exc:
        raise C0ArtifactValidationError(
            "aggregate receipt output must be a new file"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            candidate.unlink()
        except OSError:
            pass
        raise


def validate_c0_run_artifacts(
    run_dir: Path,
    canonical_run_plan_path: Path,
    *,
    aggregate_receipt_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Validate one production C0 run against immutable production constants.

    There is intentionally no profile, fixture, relaxed-count, or skip-check
    parameter on this public entry point.
    """

    return _validate_with_policy(
        Path(run_dir),
        Path(canonical_run_plan_path),
        policy=_production_policy(),
        aggregate_receipt_path=aggregate_receipt_path,
    )


def _validate_fixture_c0_run_artifacts(
    run_dir: Path,
    canonical_run_plan_path: Path,
    *,
    expected_plan_file_sha256: str,
    expected_plan_content_sha256: str,
    expected_contract: Mapping[str, Any],
    expected_provider_counts: Mapping[str, int],
    expected_memo_bounds: Mapping[str, int],
    expected_backend_version: str,
    expected_backend_model_family: str,
    expected_model_config: Mapping[str, Any],
    expected_optimizer_config: Mapping[str, Any],
    expected_archive_role_to_provider_logical_id: Mapping[str, str],
    expected_device_type: str = "cpu",
    expected_cluster_counts: Optional[Mapping[str, int]] = None,
    expected_train_order_commitment_sha256: Optional[str] = None,
    expected_validation_order_commitment_sha256: Optional[str] = None,
    expected_preprocessing: Optional[Mapping[str, Any]] = None,
    expected_archive_evidence: Optional[Mapping[str, Tuple[str, str]]] = None,
) -> Mapping[str, Any]:
    """Private test-only entry point with explicit non-production expectations."""

    if expected_contract.get("production") is not False:
        raise C0ArtifactValidationError(
            "fixture contract must explicitly be non-production"
        )
    return _validate_with_policy(
        Path(run_dir),
        Path(canonical_run_plan_path),
        policy=_ValidationPolicy(
            mode="fixture_non_production",
            plan_file_sha256=_require_sha256(
                expected_plan_file_sha256, "fixture plan file SHA-256"
            ),
            plan_content_sha256=_require_sha256(
                expected_plan_content_sha256, "fixture plan content SHA-256"
            ),
            contract=dict(expected_contract),
            provider_counts=dict(expected_provider_counts),
            memo_bounds=dict(expected_memo_bounds),
            device_type=expected_device_type,
            cublas_workspace_config=None,
            enforce_production_plan_locks=False,
            expected_backend_version=str(expected_backend_version),
            expected_backend_model_family=str(expected_backend_model_family),
            expected_model_config=dict(expected_model_config),
            expected_optimizer_config=dict(expected_optimizer_config),
            expected_archive_role_to_provider_logical_id=dict(
                expected_archive_role_to_provider_logical_id
            ),
            expected_cluster_counts=(
                None
                if expected_cluster_counts is None
                else dict(expected_cluster_counts)
            ),
            expected_train_order_commitment_sha256=(
                None
                if expected_train_order_commitment_sha256 is None
                else _require_sha256(
                    expected_train_order_commitment_sha256,
                    "fixture train order commitment",
                )
            ),
            expected_validation_order_commitment_sha256=(
                None
                if expected_validation_order_commitment_sha256 is None
                else _require_sha256(
                    expected_validation_order_commitment_sha256,
                    "fixture validation order commitment",
                )
            ),
            expected_preprocessing=(
                None if expected_preprocessing is None else dict(expected_preprocessing)
            ),
            expected_archive_evidence=(
                None
                if expected_archive_evidence is None
                else dict(expected_archive_evidence)
            ),
        ),
        aggregate_receipt_path=None,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only, production-strict C0 run artifact validation"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        receipt = validate_c0_run_artifacts(
            args.run_dir,
            args.run_plan,
            aggregate_receipt_path=args.output,
        )
    except C0ArtifactValidationError as exc:
        print("C0 artifact validation failed: {}".format(exc), file=sys.stderr)
        return 2
    if args.output is None:
        sys.stdout.buffer.write(_canonical_pretty_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "C0ArtifactValidationError",
    "PRODUCTION_RUN_PLAN_CONTENT_SHA256",
    "PRODUCTION_RUN_PLAN_FILE_SHA256",
    "VALIDATION_RECEIPT_SCHEMA_VERSION",
    "validate_c0_run_artifacts",
]
