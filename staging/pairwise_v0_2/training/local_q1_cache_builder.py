#!/usr/bin/env python3
"""Production boundary for the frozen LOCAL-Q1 fragment-geometry cache.

This module deliberately stops before any model/backend is imported.  It
reconstructs the exact metadata-only LOCAL-Q1 population, materializes each
role-neutral fragment geometry once, freezes canonical cache/inventory
receipts, and proves a zero-miss read-only replay.

Parallel production is process-local: the parent freezes and deterministically
shards unique physical fragment references, while every worker constructs its
own mask loader and :class:`GeometryArtifactCache` handle.  The generic cache's
cross-process key and commit locks serialize publication.  No archive handle,
loader, or Python cache object is shared between processes, and only the
parent writes receipts after eager verification succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import multiprocessing
import os
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    HistoricalIdentityIndex,
)
from staging.pairwise_v0_2.pairwise_data.lazy_mask_loader import (
    ArchiveSourceSpec,
    LazyMaskArchiveLoader,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    MaskMemberRef,
    TrainingPairRecord,
    iter_historical_pair_records,
    iter_synthetic_pair_records,
)
from staging.pairwise_v0_2.preflight.freeze_local_q1 import (
    ROUTE_A_SCHEMA_VERSION as LOCAL_Q1_ROUTE_A_FREEZE_SCHEMA_VERSION,
    SCHEMA_VERSION as LOCAL_Q1_FREEZE_SCHEMA_VERSION,
    SYNTHETIC_ARCHIVE_BINDING,
    LocalQ1FreezeResult,
    assert_portable_local_q1_receipt,
    freeze_local_q1,
)
from staging.pairwise_v0_2.training.fragment_geometry_cache import (
    load_or_build_fragment_geometry,
)
from staging.pairwise_v0_2.training.geometry_cache import (
    FrozenReceiptTrust,
    GeometryArtifactCache,
    GeometryCacheLimits,
    canonical_receipt_file_sha256,
)
from staging.pairwise_v0_2.training.local_cache_inventory import (
    LocalCacheInventoryConfig,
    LocalCacheInventoryResult,
    build_local_cache_inventory,
)


LOCAL_Q1_CACHE_BUILDER_VERSION = "dunhuang-local-q1-cache-builder/0.2"
LOCAL_Q1_CACHE_BUILD_RECEIPT_VERSION = "dunhuang-local-q1-cache-build/0.2"
LOCAL_Q1_MM30K_POPULATION_SCHEMA_VERSION = "dunhuang-pairwise-mm30k-population/0.1"
LOCAL_Q1_RUN_PLAN_SCHEMA_VERSION = "dunhuang-local-q1-precache-run-plan/0.1"
LOCAL_Q1_SOURCE_BUNDLE_HASH_MODE = "python_local_q1_source_bundle_manifest_v1"
LOCAL_Q1_PLAN_ANCHOR_NORMALIZED_HASH_MODE = "python_local_q1_plan_anchor_normalized_v1"
PRODUCTION_LOCAL_Q1_FREEZE_FILE_SHA256 = (
    "6fe1c1eeb923d2205b1eee88c6c505485463e1a96b48eac403d1d7df402e870e"
)
PRODUCTION_LOCAL_Q1_FREEZE_CONTENT_SHA256 = (
    "9bd019aac4be41263ca916a25822a3b8232aaf52f47115dc2bd3c4afc2ed405b"
)
PRODUCTION_LOCAL_Q1_IDENTITY_MEMBER_COUNT = 294_492
PRODUCTION_LOCAL_Q1_IDENTITY_CONTENT_SHA256 = (
    "50c51783b20b375564290ce426e33204816959203062b22e32081af4a8078158"
)
PRODUCTION_LOCAL_Q1_RUN_PLAN_FILE_SHA256 = (
    "b04691ceed75a67edf0d3ab44be397cb9fe5c10d9f6d671738fda4fdede5eb12"
)
PRODUCTION_LOCAL_Q1_RUN_PLAN_CONTENT_SHA256 = (
    "c2ec31ecbad6491b4ff48762cdc4604a1fa3365f5a1ee236bfad2105f00af609"
)

CACHE_DIRECTORY_NAME = "fragment_cache"
CACHE_RECEIPT_NAME = "fragment_cache_receipt.json"
INVENTORY_RECEIPT_NAME = "local_cache_inventory_receipt.json"
BUILD_RECEIPT_NAME = "local_q1_cache_build_receipt.json"
PLANNING_CONFIG_NAME = "local_q1_planning_config.json"
CANONICAL_LOCAL_Q1_RUN_PLAN_PATH = (
    Path(__file__).resolve().parents[1] / "preflight/local_q1_run_plan.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_NAMES = (
    "freeze_receipt",
    "mm_archive",
    "eccv_archive",
    "mm_fingerprint_cache",
    "eccv_fingerprint_cache",
    "historical_split",
    "synthetic_manifest",
    "synthetic_archive",
)
_MM30K_ROLE_NAMES = ("freeze_receipt", "mm_archive", "historical_split")


def _role(
    logical_id: str, kind: str, hash_mode: str = "sha256_bytes"
) -> Mapping[str, str]:
    return MappingProxyType(
        {"logical_id": logical_id, "kind": kind, "hash_mode": hash_mode}
    )


LOCAL_Q1_RUN_PLAN_ROLE_SPECS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "freeze_receipt": _role("artifact://local-q1/freeze", "metadata"),
        "mm_archive": _role(
            "canonical://mm_augmented/dunhuang_augmented_data", "archive"
        ),
        "eccv_archive": _role("canonical://eccv_1113data/1113data", "archive"),
        "mm_fingerprint_cache": _role(
            "artifact://v0.1/mm-fingerprint-cache", "metadata"
        ),
        "eccv_fingerprint_cache": _role(
            "artifact://v0.1/eccv-fingerprint-cache", "metadata"
        ),
        "historical_split": _role("artifact://v0.1/expanded-056-split", "metadata"),
        "synthetic_manifest": _role(
            "artifact://pairwise-mask-subset-v0.2/synthetic-manifest", "metadata"
        ),
        "synthetic_archive": _role(
            "local_asset://pairwise_mask_subset_v0_2", "archive"
        ),
    }
)
_FORBIDDEN_PORTABLE_KEYS = frozenset(
    {
        "absolute_path",
        "archive_member",
        "canonical_group_id",
        "component_id",
        "fragment_id",
        "local_path",
        "member_path",
        "pair_id",
        "path",
        "secret",
    }
)


class LocalQ1CacheBuilderError(RuntimeError):
    """The LOCAL-Q1 cache build or replay cannot be trusted."""


@dataclass(frozen=True)
class LocalQ1FileLock:
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count <= 0
        ):
            raise ValueError("file-lock byte_count must be a positive integer")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("file-lock sha256 must be lowercase SHA-256")

    def portable_dict(self) -> Dict[str, Any]:
        return {"bytes": self.byte_count, "sha256": self.sha256}


PRODUCTION_LOCAL_Q1_ROLE_LOCKS = MappingProxyType(
    {
        "freeze_receipt": LocalQ1FileLock(
            9_534, PRODUCTION_LOCAL_Q1_FREEZE_FILE_SHA256
        ),
        "mm_archive": LocalQ1FileLock(
            200_771_997,
            "b66dd129e7c7877a2a595c5a6ea748e81dfa5f6f67d4f7f154a6b2c90869e8d2",
        ),
        "eccv_archive": LocalQ1FileLock(
            153_496_003,
            "98042e1b2068500f803be817a17e09470aefa34fb432047bebc50bebc06d9e72",
        ),
        "mm_fingerprint_cache": LocalQ1FileLock(
            61_050_674,
            "6dc757928116c45bdb9373613fea8db83f8d37b788be3b947d7d08fe891174ea",
        ),
        "eccv_fingerprint_cache": LocalQ1FileLock(
            26_190_650,
            "8387145119fe83945c43e5395cd9d4b43769df3996a4d1569b418026a25b6c20",
        ),
        "historical_split": LocalQ1FileLock(
            5_331_015,
            "07b0cc9a96e7cb29d6ca6da03d252c457f7e6046da929def52c34a9629762301",
        ),
        "synthetic_manifest": LocalQ1FileLock(
            14_276_668,
            "e9772ec8e074873e4343ca42906a4056ea382536d2cbdf6ec881c21891587326",
        ),
        "synthetic_archive": LocalQ1FileLock(
            161_535_087,
            "e44e4c0e5825d8577861d79eaf4063888a349c6ba3e531be6e41f2b7dde505df",
        ),
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("{} must be lowercase SHA-256".format(name))
    return value


def _strict_json_bytes(payload: bytes, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError("{} bytes are required".format(name))

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise LocalQ1CacheBuilderError("{} is not strict JSON".format(name)) from exc
    if not isinstance(value, Mapping):
        raise LocalQ1CacheBuilderError("{} root must be an object".format(name))
    return value


def _portable(value: Any, location: str = "receipt") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and (
            value.startswith(("/", "~/", "file://"))
            or re.match(r"^[A-Za-z]:[\\/]", value)
        ):
            raise LocalQ1CacheBuilderError(
                "{} contains a machine-local path".format(location)
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LocalQ1CacheBuilderError("{} contains NaN/Inf".format(location))
        return value
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LocalQ1CacheBuilderError(
                    "{} contains a non-string key".format(location)
                )
            if key.casefold() in _FORBIDDEN_PORTABLE_KEYS:
                raise LocalQ1CacheBuilderError(
                    "{} exposes a local path, row identity, or secret".format(location)
                )
            output[key] = _portable(item, "{}.{}".format(location, key))
        return output
    if isinstance(value, (list, tuple)):
        return [
            _portable(item, "{}[{}]".format(location, index))
            for index, item in enumerate(value)
        ]
    raise LocalQ1CacheBuilderError("{} is not JSON-portable".format(location))


def _receipt_content_sha256(receipt: Mapping[str, Any], name: str) -> str:
    value = dict(receipt)
    claimed = value.pop("content_sha256", None)
    if not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed):
        raise LocalQ1CacheBuilderError("{} content hash is missing".format(name))
    observed = _sha256(_canonical_json(value))
    if not hmac.compare_digest(claimed, observed):
        raise LocalQ1CacheBuilderError("{} content hash mismatch".format(name))
    return claimed


def _hash_file(path: Path) -> Tuple[int, str]:
    try:
        if path.is_symlink() or not path.is_file():
            raise LocalQ1CacheBuilderError("input role must be a regular file")
        byte_count = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise LocalQ1CacheBuilderError("cannot hash input role") from exc
    return byte_count, digest.hexdigest()


def _normalized_local_q1_anchor_bytes(path: Path) -> bytes:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 normalized source must be UTF-8"
        ) from exc
    names = (
        "PRODUCTION_LOCAL_Q1_RUN_PLAN_FILE_SHA256",
        "PRODUCTION_LOCAL_Q1_RUN_PLAN_CONTENT_SHA256",
    )
    for name in names:
        pattern = r'({}\s*=\s*(?:\(\s*)?")[0-9a-f]{{64}}(")'.format(name)
        text, count = re.subn(pattern, r"\g<1>" + ("0" * 64) + r"\g<2>", text)
        if count != 1:
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 plan-anchor normalization failed: " + name
            )
    return text.encode("utf-8")


def local_q1_source_bundle_manifest(root: Path) -> Mapping[str, Any]:
    """Hash the complete non-test v0.1/v0.2 Python source tree.

    The two compiled plan anchors in this module are narrowly normalized to
    break the plan/source digest cycle.  Every other source byte is exact.
    """

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 production source-bundle root is not a directory"
        )
    files = []
    for package in ("pairwise_v0_1", "pairwise_v0_2"):
        package_root = root / package
        if package_root.is_symlink() or not package_root.is_dir():
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 production source bundle lacks " + package
            )
        for candidate in package_root.rglob("*"):
            if candidate.is_symlink():
                raise LocalQ1CacheBuilderError(
                    "LOCAL-Q1 production source bundle contains a symlink"
                )
            if candidate.name == "__pycache__" or candidate.suffix in {".pyc", ".pyo"}:
                raise LocalQ1CacheBuilderError(
                    "LOCAL-Q1 production source bundle contains bytecode cache"
                )
        for path in package_root.rglob("*.py"):
            if not path.is_file():
                raise LocalQ1CacheBuilderError(
                    "LOCAL-Q1 production source bundle has a non-file source"
                )
            relative = path.relative_to(root)
            if "tests" in relative.parts or "__pycache__" in relative.parts:
                continue
            relative_text = relative.as_posix()
            mode = (
                LOCAL_Q1_PLAN_ANCHOR_NORMALIZED_HASH_MODE
                if relative_text == "pairwise_v0_2/training/local_q1_cache_builder.py"
                else "sha256_bytes"
            )
            files.append(
                {
                    "relative_path": relative_text,
                    "bytes": path.stat().st_size,
                    "hash_mode": mode,
                    "sha256": local_q1_plan_role_file_sha256(path, mode),
                }
            )
    files.sort(key=lambda item: item["relative_path"])
    if not files:
        raise LocalQ1CacheBuilderError("LOCAL-Q1 production source bundle is empty")
    paths = [item["relative_path"] for item in files]
    if len(paths) != len(set(paths)):
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 production source-bundle paths are not unique"
        )
    manifest: Dict[str, Any] = {
        "policy": "all_non_test_python_under_pairwise_v0_1_and_v0_2",
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
    }
    manifest["manifest_sha256"] = _sha256(_canonical_json(manifest))
    return manifest


def local_q1_plan_role_file_sha256(path: Path, hash_mode: str) -> str:
    """Hash one plan role or source under its declared byte policy."""

    if hash_mode == "sha256_bytes":
        return _hash_file(path)[1]
    if hash_mode == LOCAL_Q1_PLAN_ANCHOR_NORMALIZED_HASH_MODE:
        return _sha256(_normalized_local_q1_anchor_bytes(path))
    if hash_mode == LOCAL_Q1_SOURCE_BUNDLE_HASH_MODE:
        return str(local_q1_source_bundle_manifest(path)["manifest_sha256"])
    raise LocalQ1CacheBuilderError("unsupported LOCAL-Q1 run-plan hash mode")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _reference_key(reference: MaskMemberRef) -> Tuple[str, ...]:
    """Runtime-only physical reference key; never serialized in a receipt."""

    return (
        reference.binding.logical_id,
        reference.binding.sha256,
        reference.archive_member,
        reference.threshold_rule,
    )


@dataclass(frozen=True)
class LocalQ1ProductionBinding:
    plan_path: Path
    source_bundle: Path
    role_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_path", Path(self.plan_path))
        object.__setattr__(self, "source_bundle", Path(self.source_bundle))
        if not isinstance(self.role_paths, Mapping) or set(self.role_paths) != set(
            _ROLE_NAMES
        ):
            raise ValueError("production binding requires every LOCAL-Q1 role exactly")
        normalized = {name: Path(self.role_paths[name]) for name in _ROLE_NAMES}
        resolved = [str(path.resolve()) for path in normalized.values()]
        if len(resolved) != len(set(resolved)):
            raise ValueError("production LOCAL-Q1 roles must be distinct files")
        object.__setattr__(self, "role_paths", MappingProxyType(normalized))


@dataclass(frozen=True)
class LocalQ1PrecacheAuthority:
    """Portable external anchors authorizing one pre-cache population/build."""

    run_plan_file_sha256: str
    run_plan_content_sha256: str
    source_bundle_manifest_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_sha256(value, name)

    def portable_dict(self) -> Dict[str, str]:
        return dict(asdict(self))


@dataclass(frozen=True)
class LocalQ1Population:
    """Exact, restartable in-memory train/validation record population."""

    training_records: Tuple[TrainingPairRecord, ...]
    validation_records: Tuple[TrainingPairRecord, ...]
    freeze_receipt: Mapping[str, Any]
    freeze_file_sha256: str
    freeze_content_sha256: str
    input_role_locks: Mapping[str, LocalQ1FileLock]
    precache_authority: LocalQ1PrecacheAuthority

    def __post_init__(self) -> None:
        _require_sha256(self.freeze_file_sha256, "freeze file sha256")
        _require_sha256(self.freeze_content_sha256, "freeze content sha256")
        if not isinstance(self.freeze_receipt, Mapping):
            raise TypeError("freeze_receipt must be a mapping")
        if not isinstance(self.precache_authority, LocalQ1PrecacheAuthority):
            raise TypeError("precache_authority must be LocalQ1PrecacheAuthority")
        receipt = _portable(dict(self.freeze_receipt), "freeze_receipt")
        schema_version = receipt.get("schema_version")
        status = receipt.get("status")
        legacy = (
            schema_version == LOCAL_Q1_FREEZE_SCHEMA_VERSION
            and status == "pass_metadata_only_population_frozen_no_model_execution"
        )
        route_a = (
            schema_version == LOCAL_Q1_ROUTE_A_FREEZE_SCHEMA_VERSION
            and status
            == "pass_route_a_geometry_qualified_population_refrozen_no_model_no_test"
        )
        mm30k = (
            schema_version == LOCAL_Q1_MM30K_POPULATION_SCHEMA_VERSION
            and status == "mm30k_geometry_qualified_fixed_seed_no_test"
        )
        if not legacy and not route_a and not mm30k:
            raise LocalQ1CacheBuilderError("LOCAL-Q1 freeze is not qualified")
        scope = receipt.get("scope")
        if not isinstance(scope, Mapping) or scope.get("sealed_real_read") is not False:
            raise LocalQ1CacheBuilderError("LOCAL-Q1 freeze scope is unsafe")
        if scope.get("model_executed") is not False or scope.get(
            "pair_stream_splits_read"
        ) != ["train", "val"]:
            raise LocalQ1CacheBuilderError("LOCAL-Q1 freeze scope is unsafe")
        if mm30k:
            if (
                scope.get("datasets") != ["mm_augmented"]
                or scope.get("historical_test_read") is not False
                or scope.get("archive_mask_members_opened") is not True
                or scope.get("mask_pixels_decoded") is not True
                or scope.get("model_imported") is not False
                or scope.get("geometry_qualification_executed") is not True
            ):
                raise LocalQ1CacheBuilderError("MM30K population scope is unsafe")
        elif scope.get("archive_mask_members_opened") is not False:
            raise LocalQ1CacheBuilderError("LOCAL-Q1 freeze scope is unsafe")
        if route_a and (
            scope.get("mask_pixels_decoded") is not False
            or scope.get("model_imported") is not False
            or scope.get("geometry_eligibility_authority_consumed") is not True
            or scope.get("geometry_qualification_executed_by_refreeze") is not False
            or scope.get("partial_cache_tree_accessed_by_refreeze") is not False
            or scope.get("fixture_scale_test_only") is not False
        ):
            raise LocalQ1CacheBuilderError("Route-A freeze scope is unsafe")
        if _receipt_content_sha256(receipt, "LOCAL-Q1 freeze") != (
            self.freeze_content_sha256
        ):
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 freeze differs from external content lock"
            )
        train = tuple(self.training_records)
        validation = tuple(self.validation_records)
        if not train or not validation:
            raise LocalQ1CacheBuilderError("LOCAL-Q1 populations cannot be empty")
        if any(record.split != "train" for record in train) or any(
            record.split != "val" for record in validation
        ):
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 populations may contain only their frozen split"
            )
        for record in train + validation:
            for key in (
                "sealed_real",
                "real_dunhuang_sealed_test",
                "historical_test",
            ):
                provenance_value = record.provenance.get(key)
                if provenance_value is not None and provenance_value is not False:
                    raise LocalQ1CacheBuilderError(
                        "sealed/test provenance cannot enter LOCAL-Q1 cache"
                    )
        try:
            expected_train = receipt["training"]["population"]["count"]
            expected_validation = receipt["validation"]["population"]["count"]
        except (KeyError, TypeError) as exc:
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 freeze population counts are missing"
            ) from exc
        if expected_train != len(train) or expected_validation != len(validation):
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 records differ from frozen population counts"
            )
        if route_a:
            proof = receipt.get("route_a_frontier_selection_proof")
            authority = receipt.get("geometry_eligibility_authority")
            if (
                not isinstance(proof, Mapping)
                or proof.get("selected_training_count") != len(train)
                or proof.get("selected_validation_count") != len(validation)
                or proof.get("unexplored_higher_priority_record_count") != 0
                or proof.get("arbitrary_fixed_reserve_cutoff_used") is not False
                or proof.get("selected_failure_counts")
                != {
                    "fragment": 0,
                    "pair": 0,
                    "direction_coverage": 0,
                    "resource": 0,
                }
                or not isinstance(authority, Mapping)
                or authority.get("production_authorized") is not True
                or authority.get("selection_replayed_exactly") is not True
                or authority.get("partial_cache_as_membership_input_permitted")
                is not False
                or authority.get("partial_cache_role_count") != 0
                or authority.get("partial_cache_tree_accessed") is not False
            ):
                raise LocalQ1CacheBuilderError(
                    "Route-A freeze lacks the exact complete-frontier selection proof"
                )
        if mm30k:
            selection = receipt.get("selection")
            train_quotas = (
                selection.get("train_quotas")
                if isinstance(selection, Mapping)
                else None
            )
            validation_quotas = (
                selection.get("validation_quotas")
                if isinstance(selection, Mapping)
                else None
            )
            production_train_quotas = {
                "generated_negative": 5000,
                "generated_positive": 5000,
                "original_negative": 10000,
                "original_positive": 10000,
            }
            production_validation_quotas = {
                "generated_negative": 1000,
                "generated_positive": 1000,
                "original_negative": 2000,
                "original_positive": 2000,
            }
            if (
                not isinstance(selection, Mapping)
                or selection.get("all_selected_pairs_geometry_qualified") is not True
                or selection.get("candidate_truncation_used") is not False
                or selection.get("train_validation_component_overlap_count") != 0
                or not isinstance(train_quotas, Mapping)
                or not isinstance(validation_quotas, Mapping)
                or set(train_quotas) != set(production_train_quotas)
                or set(validation_quotas) != set(production_validation_quotas)
                or any(
                    type(value) is not int or value <= 0
                    for value in train_quotas.values()
                )
                or any(
                    type(value) is not int or value <= 0
                    for value in validation_quotas.values()
                )
                or sum(train_quotas.values()) != len(train)
                or sum(validation_quotas.values()) != len(validation)
            ):
                raise LocalQ1CacheBuilderError("MM30K selection contract is incomplete")
            if scope.get("fixture_scale_test_only") is False and (
                dict(train_quotas) != production_train_quotas
                or dict(validation_quotas) != production_validation_quotas
            ):
                raise LocalQ1CacheBuilderError("MM30K production quotas changed")
        required_roles = _MM30K_ROLE_NAMES if mm30k else _ROLE_NAMES
        if not isinstance(self.input_role_locks, Mapping) or set(
            self.input_role_locks
        ) != set(required_roles):
            raise LocalQ1CacheBuilderError("LOCAL-Q1 input-role locks are incomplete")
        role_locks = {}
        for name in required_roles:
            lock = self.input_role_locks[name]
            if not isinstance(lock, LocalQ1FileLock):
                raise TypeError("input role locks must be LocalQ1FileLock")
            role_locks[name] = lock
        if role_locks["freeze_receipt"].sha256 != self.freeze_file_sha256:
            raise LocalQ1CacheBuilderError("freeze role/file lock mismatch")
        object.__setattr__(self, "training_records", train)
        object.__setattr__(self, "validation_records", validation)
        object.__setattr__(self, "freeze_receipt", MappingProxyType(dict(receipt)))
        object.__setattr__(self, "input_role_locks", MappingProxyType(role_locks))

    @property
    def records(self) -> Tuple[TrainingPairRecord, ...]:
        """Return the same immutable train-then-validation sequence each call."""

        return self.training_records + self.validation_records


@dataclass(frozen=True)
class ProductionMaskLoaderFactory:
    """Pickle-safe factory; each call owns independent archive handles."""

    mm_archive: Path
    eccv_archive: Path
    synthetic_archive: Path
    source_bundle: Path
    expected_source_bundle_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "mm_archive",
            "eccv_archive",
            "synthetic_archive",
            "source_bundle",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        _require_sha256(
            self.expected_source_bundle_manifest_sha256,
            "production loader source-bundle manifest sha256",
        )

    def __call__(self) -> LazyMaskArchiveLoader:
        observed_source = local_q1_source_bundle_manifest(self.source_bundle)
        if not hmac.compare_digest(
            str(observed_source["manifest_sha256"]),
            self.expected_source_bundle_manifest_sha256,
        ):
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 source bundle changed before loader construction"
            )
        return LazyMaskArchiveLoader(
            {
                MM_CANONICAL_BINDING.logical_id: ArchiveSourceSpec(
                    binding=MM_CANONICAL_BINDING, source=self.mm_archive
                ),
                ECCV_CANONICAL_BINDING.logical_id: ArchiveSourceSpec(
                    binding=ECCV_CANONICAL_BINDING, source=self.eccv_archive
                ),
                SYNTHETIC_ARCHIVE_BINDING.logical_id: ArchiveSourceSpec(
                    binding=SYNTHETIC_ARCHIVE_BINDING,
                    source=self.synthetic_archive,
                ),
            }
        )


@dataclass(frozen=True)
class _ProducerShardResult:
    task_count: int
    cache_hit_count: int
    cache_miss_count: int
    decoded_mask_pixels: int


@dataclass(frozen=True)
class LocalQ1CacheBuildArtifacts:
    output_dir: Path
    cache_receipt_path: Path
    inventory_receipt_path: Path
    build_receipt_path: Path
    cache_receipt: Mapping[str, Any]
    inventory_receipt: Mapping[str, Any]
    build_receipt: Mapping[str, Any]
    cache_receipt_file_sha256: str
    inventory_receipt_file_sha256: str
    build_receipt_file_sha256: str
    read_only_cache: GeometryArtifactCache


@dataclass(frozen=True)
class LocalQ1CacheTrust:
    """All anchors must be copied into the experiment lock before reopen."""

    expected_freeze_file_sha256: str
    expected_freeze_content_sha256: str
    expected_run_plan_file_sha256: str
    expected_run_plan_content_sha256: str
    expected_source_bundle_manifest_sha256: str
    expected_cache_receipt_file_sha256: str
    expected_cache_receipt_content_sha256: str
    expected_inventory_receipt_file_sha256: str
    expected_inventory_receipt_content_sha256: str
    expected_inventory_semantic_sha256: str
    expected_build_receipt_file_sha256: str
    expected_build_receipt_content_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_sha256(value, name)


@dataclass(frozen=True)
class OpenedLocalQ1Cache:
    population: LocalQ1Population
    cache: GeometryArtifactCache
    inventory_replay: LocalCacheInventoryResult
    build_receipt: Mapping[str, Any]


def verify_local_q1_production_plan(
    binding: LocalQ1ProductionBinding,
    *,
    expected_run_plan_file_sha256: Optional[str] = None,
    expected_run_plan_content_sha256: Optional[str] = None,
    expected_freeze_file_sha256: Optional[str] = None,
    expected_freeze_content_sha256: Optional[str] = None,
    expected_identity_member_count: Optional[int] = None,
    expected_identity_content_sha256: Optional[str] = None,
) -> Tuple[
    Mapping[str, Any],
    Mapping[str, LocalQ1FileLock],
    LocalQ1PrecacheAuthority,
]:
    """Verify the externally anchored plan, all eight roles, and source tree."""

    if not isinstance(binding, LocalQ1ProductionBinding):
        raise TypeError("binding must be LocalQ1ProductionBinding")
    # Resolve legacy production anchors at call time so fixture overrides and
    # existing callers retain their original behavior.  Route-A callers pass
    # the freshly locked values explicitly through the external-plan wrapper.
    if expected_run_plan_file_sha256 is None:
        expected_run_plan_file_sha256 = PRODUCTION_LOCAL_Q1_RUN_PLAN_FILE_SHA256
    if expected_run_plan_content_sha256 is None:
        expected_run_plan_content_sha256 = PRODUCTION_LOCAL_Q1_RUN_PLAN_CONTENT_SHA256
    if expected_freeze_file_sha256 is None:
        expected_freeze_file_sha256 = PRODUCTION_LOCAL_Q1_FREEZE_FILE_SHA256
    if expected_freeze_content_sha256 is None:
        expected_freeze_content_sha256 = PRODUCTION_LOCAL_Q1_FREEZE_CONTENT_SHA256
    if expected_identity_member_count is None:
        expected_identity_member_count = PRODUCTION_LOCAL_Q1_IDENTITY_MEMBER_COUNT
    if expected_identity_content_sha256 is None:
        expected_identity_content_sha256 = PRODUCTION_LOCAL_Q1_IDENTITY_CONTENT_SHA256
    executing_source_root = Path(__file__).resolve().parents[2]
    if binding.source_bundle.resolve() != executing_source_root:
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 source bundle differs from the executing cache-builder tree"
        )
    for value, name in (
        (
            expected_run_plan_file_sha256,
            "LOCAL-Q1 canonical plan file sha256",
        ),
        (
            expected_run_plan_content_sha256,
            "LOCAL-Q1 canonical plan content sha256",
        ),
        (expected_freeze_file_sha256, "LOCAL-Q1 freeze file sha256"),
        (expected_freeze_content_sha256, "LOCAL-Q1 freeze content sha256"),
        (
            expected_identity_content_sha256,
            "LOCAL-Q1 identity-index content sha256",
        ),
    ):
        _require_sha256(value, name)
        if value == "0" * 64:
            raise LocalQ1CacheBuilderError(
                "canonical LOCAL-Q1 production plan has not been frozen"
            )
    if (
        isinstance(expected_identity_member_count, bool)
        or not isinstance(expected_identity_member_count, int)
        or expected_identity_member_count <= 0
    ):
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 identity-index member count must be positive"
        )
    try:
        if binding.plan_path.is_symlink() or not binding.plan_path.is_file():
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 production plan is not a regular file"
            )
        plan_bytes = binding.plan_path.read_bytes()
    except OSError as exc:
        raise LocalQ1CacheBuilderError("cannot read LOCAL-Q1 production plan") from exc
    plan_file_sha = _sha256(plan_bytes)
    if not hmac.compare_digest(plan_file_sha, expected_run_plan_file_sha256):
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 production plan external file hash mismatch"
        )
    plan = _strict_json_bytes(plan_bytes, "LOCAL-Q1 production plan")
    stored_content = plan.get("content_sha256")
    unsigned = dict(plan)
    unsigned.pop("content_sha256", None)
    observed_content = _sha256(_canonical_json(unsigned))
    if (
        stored_content != observed_content
        or observed_content != expected_run_plan_content_sha256
    ):
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 production plan external content hash mismatch"
        )
    expected_root_keys = {
        "schema_version",
        "status",
        "experiment",
        "freeze_content_sha256",
        "identity_index",
        "roles",
        "source_bundle",
        "scope",
        "content_sha256",
    }
    if set(plan) != expected_root_keys:
        raise LocalQ1CacheBuilderError("LOCAL-Q1 production plan schema changed")
    if (
        plan.get("schema_version") != LOCAL_Q1_RUN_PLAN_SCHEMA_VERSION
        or plan.get("status") != "frozen_no_execution"
        or plan.get("experiment") != "LOCAL-Q1-PRECACHE"
        or plan.get("freeze_content_sha256") != expected_freeze_content_sha256
    ):
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 production plan status or freeze anchor changed"
        )
    identity = plan.get("identity_index")
    if identity != {
        "member_count": expected_identity_member_count,
        "content_sha256": expected_identity_content_sha256,
    }:
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 production plan identity-index lock changed"
        )
    scope = plan.get("scope")
    if scope != {
        "archive_bytes_hashed": True,
        "archive_members_opened": False,
        "mask_pixels_decoded": False,
        "model_or_backend_created": False,
        "historical_test_access": {
            "split_container_bytes_hashed_without_parsing": True,
            "assignment_or_pair_records_parsed": False,
            "pair_stream_read": False,
            "archive_members_opened": False,
            "mask_pixels_decoded": False,
        },
        "sealed_real_read": False,
    }:
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 production plan scope evidence changed"
        )

    rows = plan.get("roles")
    if not isinstance(rows, list) or len(rows) != len(LOCAL_Q1_RUN_PLAN_ROLE_SPECS):
        raise LocalQ1CacheBuilderError("LOCAL-Q1 production plan role list changed")
    by_role: Dict[str, Mapping[str, Any]] = {}
    role_schema = {
        "role",
        "logical_id",
        "kind",
        "hash_mode",
        "bytes",
        "sha256",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != role_schema:
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 production plan role schema changed"
            )
        role = str(row["role"])
        if role in by_role:
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 production plan has a duplicate role"
            )
        by_role[role] = row
    if set(by_role) != set(LOCAL_Q1_RUN_PLAN_ROLE_SPECS):
        raise LocalQ1CacheBuilderError("LOCAL-Q1 production plan role names changed")

    observed_locks: Dict[str, LocalQ1FileLock] = {}
    for role in sorted(LOCAL_Q1_RUN_PLAN_ROLE_SPECS):
        spec = LOCAL_Q1_RUN_PLAN_ROLE_SPECS[role]
        row = by_role[role]
        if any(row[name] != spec[name] for name in spec):
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 production role policy changed: " + role
            )
        byte_count, digest = _hash_file(binding.role_paths[role])
        if (
            type(row["bytes"]) is not int  # noqa: E721
            or row["bytes"] != byte_count
            or not hmac.compare_digest(str(row["sha256"]), digest)
        ):
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 production input role changed: " + role
            )
        observed_locks[role] = LocalQ1FileLock(byte_count, digest)
    planned_bundle = plan.get("source_bundle")
    if not isinstance(planned_bundle, Mapping):
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 production plan lacks a source-bundle manifest"
        )
    observed_bundle = local_q1_source_bundle_manifest(binding.source_bundle)
    if not hmac.compare_digest(
        _canonical_json(planned_bundle), _canonical_json(observed_bundle)
    ):
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 production source-bundle exact manifest changed"
        )
    _portable(plan, "LOCAL-Q1 production plan")
    authority = LocalQ1PrecacheAuthority(
        run_plan_file_sha256=plan_file_sha,
        run_plan_content_sha256=observed_content,
        source_bundle_manifest_sha256=str(observed_bundle["manifest_sha256"]),
    )
    return (
        MappingProxyType(dict(plan)),
        MappingProxyType(observed_locks),
        authority,
    )


def verify_external_local_q1_run_plan(
    binding: LocalQ1ProductionBinding,
    *,
    expected_run_plan_file_sha256: str,
    expected_run_plan_content_sha256: str,
    expected_freeze_file_sha256: str,
    expected_freeze_content_sha256: str,
    expected_identity_member_count: int,
    expected_identity_content_sha256: str,
) -> Tuple[
    Mapping[str, Any],
    Mapping[str, LocalQ1FileLock],
    LocalQ1PrecacheAuthority,
]:
    """Verify a newly frozen Route-A run plan without legacy hash constants."""

    return verify_local_q1_production_plan(
        binding,
        expected_run_plan_file_sha256=expected_run_plan_file_sha256,
        expected_run_plan_content_sha256=expected_run_plan_content_sha256,
        expected_freeze_file_sha256=expected_freeze_file_sha256,
        expected_freeze_content_sha256=expected_freeze_content_sha256,
        expected_identity_member_count=expected_identity_member_count,
        expected_identity_content_sha256=expected_identity_content_sha256,
    )


def attest_rebuilt_local_q1_population(
    *,
    canonical_freeze_file_bytes: bytes,
    expected_freeze_file_sha256: str,
    expected_freeze_content_sha256: str,
    rebuilt: LocalQ1FreezeResult,
    input_role_locks: Mapping[str, LocalQ1FileLock],
    precache_authority: LocalQ1PrecacheAuthority,
) -> LocalQ1Population:
    """Bind a metadata reconstruction to external bytes and exact object equality."""

    expected_file = _require_sha256(
        expected_freeze_file_sha256, "expected freeze file sha256"
    )
    expected_content = _require_sha256(
        expected_freeze_content_sha256, "expected freeze content sha256"
    )
    if not hmac.compare_digest(_sha256(canonical_freeze_file_bytes), expected_file):
        raise LocalQ1CacheBuilderError("LOCAL-Q1 freeze file hash mismatch")
    canonical = _strict_json_bytes(
        canonical_freeze_file_bytes, "canonical LOCAL-Q1 freeze"
    )
    if _receipt_content_sha256(canonical, "canonical LOCAL-Q1 freeze") != (
        expected_content
    ):
        raise LocalQ1CacheBuilderError("LOCAL-Q1 freeze content lock mismatch")
    if not isinstance(rebuilt, LocalQ1FreezeResult):
        raise TypeError("rebuilt must be LocalQ1FreezeResult")
    if dict(rebuilt.receipt) != dict(canonical):
        raise LocalQ1CacheBuilderError(
            "metadata reconstruction differs from canonical LOCAL-Q1 freeze"
        )
    assert_portable_local_q1_receipt(rebuilt.receipt)
    return LocalQ1Population(
        training_records=tuple(rebuilt.training_records),
        validation_records=tuple(rebuilt.validation_records),
        freeze_receipt=canonical,
        freeze_file_sha256=expected_file,
        freeze_content_sha256=expected_content,
        input_role_locks=input_role_locks,
        precache_authority=precache_authority,
    )


def rebuild_production_local_q1_population(
    binding: LocalQ1ProductionBinding,
) -> LocalQ1Population:
    """Verify every role, rerun metadata freeze, and return restartable tuples."""

    if not isinstance(binding, LocalQ1ProductionBinding):
        raise TypeError("binding must be LocalQ1ProductionBinding")
    _plan, role_locks, precache_authority = verify_local_q1_production_plan(binding)
    freeze_bytes = binding.role_paths["freeze_receipt"].read_bytes()
    identity_index = HistoricalIdentityIndex.from_files(
        mm_cache_path=binding.role_paths["mm_fingerprint_cache"],
        eccv_cache_path=binding.role_paths["eccv_fingerprint_cache"],
        split_path=binding.role_paths["historical_split"],
    )
    if (
        identity_index.identity_count != PRODUCTION_LOCAL_Q1_IDENTITY_MEMBER_COUNT
        or identity_index.content_sha256 != PRODUCTION_LOCAL_Q1_IDENTITY_CONTENT_SHA256
    ):
        raise LocalQ1CacheBuilderError(
            "rebuilt historical identity index differs from external lock"
        )

    def historical(split: str):
        return iter_historical_pair_records(
            split_manifest=binding.role_paths["historical_split"],
            split=split,
            mm_archive=binding.role_paths["mm_archive"],
            eccv_archive=binding.role_paths["eccv_archive"],
        )

    rebuilt = freeze_local_q1(
        identity_index=identity_index,
        validation_records=historical("val"),
        historical_training_records=historical("train"),
        synthetic_training_records=iter_synthetic_pair_records(
            binding.role_paths["synthetic_manifest"]
        ),
        historical_archive_locks={
            "mm_augmented": {
                "format": MM_CANONICAL_BINDING.archive_format,
                "logical_id": MM_CANONICAL_BINDING.logical_id,
                **role_locks["mm_archive"].portable_dict(),
            },
            "eccv_1113data": {
                "format": ECCV_CANONICAL_BINDING.archive_format,
                "logical_id": ECCV_CANONICAL_BINDING.logical_id,
                **role_locks["eccv_archive"].portable_dict(),
            },
        },
        synthetic_manifest_lock=role_locks["synthetic_manifest"].portable_dict(),
    )
    return attest_rebuilt_local_q1_population(
        canonical_freeze_file_bytes=freeze_bytes,
        expected_freeze_file_sha256=PRODUCTION_LOCAL_Q1_FREEZE_FILE_SHA256,
        expected_freeze_content_sha256=PRODUCTION_LOCAL_Q1_FREEZE_CONTENT_SHA256,
        rebuilt=rebuilt,
        input_role_locks=role_locks,
        precache_authority=precache_authority,
    )


def _unique_fragment_tasks(
    records: Sequence[TrainingPairRecord],
) -> Tuple[MaskMemberRef, ...]:
    by_key: Dict[Tuple[str, ...], MaskMemberRef] = {}
    for record in records:
        if not isinstance(record, TrainingPairRecord):
            raise TypeError("every LOCAL-Q1 record must be TrainingPairRecord")
        if record.split not in {"train", "val"}:
            raise LocalQ1CacheBuilderError("test records cannot enter LOCAL-Q1 cache")
        for reference in (record.fragment_a, record.fragment_b):
            key = _reference_key(reference)
            existing = by_key.setdefault(key, reference)
            if existing != reference:
                raise LocalQ1CacheBuilderError(
                    "one physical reference has inconsistent metadata"
                )
    return tuple(by_key[key] for key in sorted(by_key))


def _close_loader(loader: Any) -> None:
    closer = getattr(loader, "close", None)
    if callable(closer):
        closer()


def _load_mask(
    loader: Callable[[MaskMemberRef], np.ndarray],
    reference: MaskMemberRef,
    max_mask_pixels: int,
) -> np.ndarray:
    value = np.asarray(loader(reference))
    if value.ndim != 2 or value.size == 0 or value.dtype != np.bool_:
        raise LocalQ1CacheBuilderError(
            "mask loader must return a non-empty 2D bool array"
        )
    if value.size > max_mask_pixels:
        raise LocalQ1CacheBuilderError("decoded mask exceeds configured bound")
    return np.ascontiguousarray(value, dtype=np.bool_)


def _produce_fragment_shard(
    references: Tuple[MaskMemberRef, ...],
    loader_factory: Callable[[], Callable[[MaskMemberRef], np.ndarray]],
    cache_root: Path,
    inventory_config: LocalCacheInventoryConfig,
    cache_limits: GeometryCacheLimits,
) -> _ProducerShardResult:
    """Worker entrypoint; all state is constructed inside this process."""

    loader = loader_factory()
    cache = GeometryArtifactCache(cache_root, limits=cache_limits)
    hit_count = 0
    miss_count = 0
    pixel_count = 0
    try:
        for reference in references:
            mask = _load_mask(loader, reference, inventory_config.max_mask_pixels)
            pixel_count += int(mask.size)
            lookup = load_or_build_fragment_geometry(
                mask,
                reference.threshold_rule,
                inventory_config.geometry,
                cache,
            )
            if lookup.cache_hit:
                hit_count += 1
            else:
                miss_count += 1
    finally:
        _close_loader(loader)
    return _ProducerShardResult(
        task_count=len(references),
        cache_hit_count=hit_count,
        cache_miss_count=miss_count,
        decoded_mask_pixels=pixel_count,
    )


def _produce_fragments(
    *,
    references: Tuple[MaskMemberRef, ...],
    loader_factory: Callable[[], Callable[[MaskMemberRef], np.ndarray]],
    cache_root: Path,
    inventory_config: LocalCacheInventoryConfig,
    cache_limits: GeometryCacheLimits,
    producer_workers: int,
) -> Tuple[_ProducerShardResult, ...]:
    if (
        isinstance(producer_workers, bool)
        or not isinstance(producer_workers, int)
        or producer_workers <= 0
        or producer_workers > 64
    ):
        raise ValueError("producer_workers must be an integer in [1, 64]")
    active_workers = min(producer_workers, len(references))
    if active_workers == 1:
        return (
            _produce_fragment_shard(
                references,
                loader_factory,
                cache_root,
                inventory_config,
                cache_limits,
            ),
        )
    try:
        pickle.dumps(loader_factory, protocol=pickle.HIGHEST_PROTOCOL)
    except (AttributeError, pickle.PickleError, TypeError) as exc:
        raise LocalQ1CacheBuilderError(
            "multi-process production requires a pickle-safe loader factory; "
            "use producer_workers=1 for explicit single-writer mode"
        ) from exc
    # ``references`` is already sorted by archive identity/member.  Contiguous
    # balanced shards preserve that order inside each worker, which is
    # materially faster for the compressed ECCV tar stream than round-robin
    # seeks across the whole archive while retaining exact task coverage.
    quotient, remainder = divmod(len(references), active_workers)
    shards = []
    start = 0
    for index in range(active_workers):
        size = quotient + (1 if index < remainder else 0)
        stop = start + size
        shards.append(tuple(references[start:stop]))
        start = stop
    shards = tuple(shards)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=active_workers, mp_context=context
    ) as executor:
        futures = [
            executor.submit(
                _produce_fragment_shard,
                shard,
                loader_factory,
                cache_root,
                inventory_config,
                cache_limits,
            )
            for shard in shards
        ]
        return tuple(future.result() for future in futures)


def _inventory_with_fresh_loader(
    population: LocalQ1Population,
    loader_factory: Callable[[], Callable[[MaskMemberRef], np.ndarray]],
    cache: GeometryArtifactCache,
    inventory_config: LocalCacheInventoryConfig,
) -> LocalCacheInventoryResult:
    loader = loader_factory()
    try:
        return build_local_cache_inventory(
            population.records, loader, cache, inventory_config
        )
    finally:
        _close_loader(loader)


def _write_new_canonical(path: Path, receipt: Mapping[str, Any]) -> str:
    payload = _canonical_json(_portable(dict(receipt)))
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise LocalQ1CacheBuilderError("refusing to overwrite a receipt") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return _sha256(payload)


def _cache_disk_inventory(cache_root: Path) -> Tuple[int, int]:
    count = 0
    byte_count = 0
    for path in cache_root.rglob("*.npz"):
        if path.is_symlink() or not path.is_file():
            raise LocalQ1CacheBuilderError("cache contains a non-regular artifact")
        count += 1
        byte_count += path.stat().st_size
    return count, byte_count


def prebuild_local_q1_fragment_reserve(
    *,
    records: Sequence[TrainingPairRecord],
    loader_factory: Callable[[], Callable[[MaskMemberRef], np.ndarray]],
    output_dir: Path,
    inventory_config: LocalCacheInventoryConfig,
    cache_limits: GeometryCacheLimits,
    producer_workers: int,
) -> Mapping[str, int]:
    """Build a fresh bounded record reserve without freezing population receipts.

    MM30K uses this once, applies the existing supervision-blind pair qualifier,
    deletes only unselected reserve entries from this newly created directory,
    and then calls :func:`build_local_q1_cache` in prebuilt mode.  No existing
    cache or user file can be adopted by this helper.
    """

    population = tuple(records)
    if not population or any(
        not isinstance(record, TrainingPairRecord) for record in population
    ):
        raise LocalQ1CacheBuilderError("fragment reserve records are invalid")
    if not callable(loader_factory):
        raise TypeError("loader_factory must be callable")
    if not isinstance(inventory_config, LocalCacheInventoryConfig):
        raise TypeError("inventory_config must be LocalCacheInventoryConfig")
    if not isinstance(cache_limits, GeometryCacheLimits):
        raise TypeError("cache_limits must be GeometryCacheLimits")
    target = Path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise LocalQ1CacheBuilderError(
            "fragment reserve output already exists; refusing overwrite"
        ) from exc
    cache_root = target / CACHE_DIRECTORY_NAME
    cache_root.mkdir()
    references = _unique_fragment_tasks(population)
    results = _produce_fragments(
        references=references,
        loader_factory=loader_factory,
        cache_root=cache_root,
        inventory_config=inventory_config,
        cache_limits=cache_limits,
        producer_workers=producer_workers,
    )
    if sum(item.task_count for item in results) != len(references):
        raise LocalQ1CacheBuilderError("reserve producers did not cover every fragment")
    return MappingProxyType(
        {
            "requested_worker_count": producer_workers,
            "active_worker_count": len(results),
            "physical_reference_task_count": sum(item.task_count for item in results),
            "producer_cache_hit_count": sum(item.cache_hit_count for item in results),
            "producer_cache_miss_count": sum(item.cache_miss_count for item in results),
            "producer_decoded_mask_pixels": sum(
                item.decoded_mask_pixels for item in results
            ),
            "pruned_artifact_count": 0,
        }
    )


def build_local_q1_cache(
    *,
    population: LocalQ1Population,
    loader_factory: Callable[[], Callable[[MaskMemberRef], np.ndarray]],
    output_dir: Path,
    inventory_config: Optional[LocalCacheInventoryConfig] = None,
    cache_limits: Optional[GeometryCacheLimits] = None,
    producer_workers: int = 1,
    fresh_prebuilt_producer_stats: Optional[Mapping[str, int]] = None,
) -> LocalQ1CacheBuildArtifacts:
    """Cold-build, eagerly verify, replay, and atomically freeze all receipts.

    ``fresh_prebuilt_producer_stats`` is the narrow MM30K research seam.  When
    supplied, ``output_dir/fragment_cache`` must already be a freshly claimed,
    receipt-free cache containing exactly the selected population's artifacts.
    The caller uses the same geometry producer to rank a bounded reserve, drops
    unused reserve artifacts, and passes the actual producer counters here.
    Existing Route-A callers retain the original cold-build path unchanged.
    """

    if not isinstance(population, LocalQ1Population):
        raise TypeError("population must be LocalQ1Population")
    if not callable(loader_factory):
        raise TypeError("loader_factory must be callable")
    settings = inventory_config or LocalCacheInventoryConfig()
    limits = cache_limits or GeometryCacheLimits()
    if not isinstance(settings, LocalCacheInventoryConfig):
        raise TypeError("inventory_config must be LocalCacheInventoryConfig")
    if not isinstance(limits, GeometryCacheLimits):
        raise TypeError("cache_limits must be GeometryCacheLimits")
    target = Path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    cache_root = target / CACHE_DIRECTORY_NAME
    started_utc = _utc_now()
    started = time.monotonic()
    references = _unique_fragment_tasks(population.records)
    prebuilt = fresh_prebuilt_producer_stats is not None
    if prebuilt:
        if (
            target.is_symlink()
            or not target.is_dir()
            or {item.name for item in target.iterdir()} != {CACHE_DIRECTORY_NAME}
            or cache_root.is_symlink()
            or not cache_root.is_dir()
        ):
            raise LocalQ1CacheBuilderError(
                "fresh prebuilt output must contain only fragment_cache"
            )
        required_stats = {
            "requested_worker_count",
            "active_worker_count",
            "physical_reference_task_count",
            "producer_cache_hit_count",
            "producer_cache_miss_count",
            "producer_decoded_mask_pixels",
            "pruned_artifact_count",
        }
        stats = dict(fresh_prebuilt_producer_stats or {})
        if set(stats) != required_stats or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in stats.values()
        ):
            raise LocalQ1CacheBuilderError("fresh prebuilt producer stats are invalid")
        if stats["requested_worker_count"] <= 0 or stats["active_worker_count"] <= 0:
            raise LocalQ1CacheBuilderError("fresh prebuilt worker counts are invalid")
        shard_results = ()
    else:
        try:
            target.mkdir()
        except FileExistsError as exc:
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 output already exists; refusing duplicate producer/overwrite"
            ) from exc
        cache_root.mkdir()
        shard_results = _produce_fragments(
            references=references,
            loader_factory=loader_factory,
            cache_root=cache_root,
            inventory_config=settings,
            cache_limits=limits,
            producer_workers=producer_workers,
        )
        if sum(result.task_count for result in shard_results) != len(references):
            raise LocalQ1CacheBuilderError(
                "producer shards did not cover every fragment"
            )

    writable_cache = GeometryArtifactCache(cache_root, limits=limits)
    inventory = _inventory_with_fresh_loader(
        population, loader_factory, writable_cache, settings
    )
    if inventory.receipt.get("status") != "qualified":
        raise LocalQ1CacheBuilderError("LOCAL-Q1 geometry inventory failed closed")
    cache_receipt = dict(inventory.cache_receipt)
    cache_receipt_file_sha = canonical_receipt_file_sha256(cache_receipt)
    cache_receipt_content_sha = _receipt_content_sha256(
        cache_receipt, "fragment cache receipt"
    )
    frozen_cache = GeometryArtifactCache.from_frozen_receipt(
        cache_root,
        cache_receipt,
        limits=limits,
        trust=FrozenReceiptTrust(
            expected_content_sha256=cache_receipt_content_sha,
            expected_file_sha256=cache_receipt_file_sha,
        ),
    )
    replay = _inventory_with_fresh_loader(
        population, loader_factory, frozen_cache, settings
    )
    replay_counts = replay.receipt["operational_counts"]
    if (
        replay.semantic_commitment_sha256 != inventory.semantic_commitment_sha256
        or replay_counts["cache_miss_count"] != 0
        or replay_counts["cache_hit_count"]
        != replay_counts["unique_canonical_fragment_count"]
    ):
        raise LocalQ1CacheBuilderError(
            "eager read-only LOCAL-Q1 replay was not exact and zero-miss"
        )

    artifact_file_count, artifact_file_bytes = _cache_disk_inventory(cache_root)
    artifact_count = int(cache_receipt["artifact_count"])
    if artifact_file_count != artifact_count:
        raise LocalQ1CacheBuilderError("cache disk/receipt artifact counts differ")
    if prebuilt:
        producer_misses = stats["producer_cache_miss_count"]
        producer_hits = stats["producer_cache_hit_count"]
        producer_task_count = stats["physical_reference_task_count"]
        decoded_mask_pixels = stats["producer_decoded_mask_pixels"]
        pruned_artifact_count = stats["pruned_artifact_count"]
        active_worker_count = stats["active_worker_count"]
        requested_worker_count = stats["requested_worker_count"]
    else:
        producer_misses = sum(result.cache_miss_count for result in shard_results)
        producer_hits = sum(result.cache_hit_count for result in shard_results)
        producer_task_count = sum(result.task_count for result in shard_results)
        decoded_mask_pixels = sum(
            result.decoded_mask_pixels for result in shard_results
        )
        pruned_artifact_count = 0
        active_worker_count = len(shard_results)
        requested_worker_count = producer_workers
    if producer_misses != artifact_count + pruned_artifact_count:
        raise LocalQ1CacheBuilderError(
            "process-local producers did not publish each canonical artifact once"
        )
    inventory_receipt = dict(inventory.receipt)
    inventory_file_sha = canonical_receipt_file_sha256(inventory_receipt)
    inventory_content_sha = _receipt_content_sha256(
        inventory_receipt, "local cache inventory receipt"
    )
    finished_utc = _utc_now()
    elapsed_seconds = time.monotonic() - started
    role_locks = {
        name: population.input_role_locks[name].portable_dict()
        for name in sorted(population.input_role_locks)
    }
    receipt_locks = population.freeze_receipt["locks"]
    if "historical_identity_index" in receipt_locks:
        identity_lock_name = "historical_identity_index"
    elif "population_identity" in receipt_locks:
        identity_lock_name = "population_identity"
    else:
        raise LocalQ1CacheBuilderError("population receipt lacks an identity lock")
    identity_lock = receipt_locks[identity_lock_name]
    population_lock = {
        "freeze_file_sha256": population.freeze_file_sha256,
        "freeze_content_sha256": population.freeze_content_sha256,
        "input_roles": role_locks,
        identity_lock_name: dict(identity_lock),
        "training_record_count": len(population.training_records),
        "validation_record_count": len(population.validation_records),
    }
    build_receipt: Dict[str, Any] = {
        "schema_version": LOCAL_Q1_CACHE_BUILD_RECEIPT_VERSION,
        "builder_version": LOCAL_Q1_CACHE_BUILDER_VERSION,
        "status": "complete_geometry_cache_frozen_zero_miss_no_model_no_test",
        "scope": {
            "experiment": "LOCAL-Q1",
            "input_modality": "canonical_bool_mask_only",
            "splits": ["train", "val"],
            "sealed_real_capability": False,
            "historical_test_capability": False,
            "model_backend_imported": False,
            "model_executed": False,
        },
        "population_lock": population_lock,
        "precache_authority": population.precache_authority.portable_dict(),
        "producer_contract": {
            "mode": (
                "fresh_mm30k_ranked_reserve_then_exact_selected_cache"
                if prebuilt
                else "single_writer_process"
                if len(shard_results) == 1
                else "process_local_deterministic_physical_reference_shards"
            ),
            "requested_worker_count": requested_worker_count,
            "active_worker_count": active_worker_count,
            "worker_loader_policy": "one_independent_loader_per_worker",
            "worker_cache_policy": "one_independent_cache_handle_per_worker",
            "shared_python_loader_or_cache_objects": False,
            "receipt_writer": "parent_only_after_eager_verification",
            "physical_reference_task_count": producer_task_count,
            "producer_cache_hit_count": producer_hits,
            "producer_cache_miss_count": producer_misses,
            "producer_decoded_mask_pixels": decoded_mask_pixels,
            "pruned_unselected_reserve_artifact_count": pruned_artifact_count,
        },
        "cache": {
            "artifact_count": artifact_count,
            "artifact_file_count": artifact_file_count,
            "artifact_file_bytes": artifact_file_bytes,
            "logical_array_bytes": sum(
                int(item["array_bytes"]) for item in cache_receipt["artifacts"]
            ),
            "receipt_file_sha256": cache_receipt_file_sha,
            "receipt_content_sha256": cache_receipt_content_sha,
        },
        "inventory": {
            "semantic_commitment_sha256": inventory.semantic_commitment_sha256,
            "receipt_file_sha256": inventory_file_sha,
            "receipt_content_sha256": inventory_content_sha,
            "record_count": inventory_receipt["operational_counts"]["record_count"],
            "unique_physical_reference_count": inventory_receipt["operational_counts"][
                "unique_physical_reference_count"
            ],
            "unique_canonical_fragment_count": inventory_receipt["operational_counts"][
                "unique_canonical_fragment_count"
            ],
            "read_only_replay_cache_miss_count": replay_counts["cache_miss_count"],
            "read_only_replay_cache_hit_count": replay_counts["cache_hit_count"],
        },
        "timing": {
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "elapsed_seconds": elapsed_seconds,
        },
        "portable_privacy": {
            "local_paths_present": False,
            "row_member_pair_fragment_or_component_ids_present": False,
            "secrets_present": False,
            "aggregate_counts_and_commitments_only": True,
        },
    }
    build_receipt["content_sha256"] = _sha256(_canonical_json(build_receipt))
    build_receipt = _portable(build_receipt, "build_receipt")

    cache_receipt_path = target / CACHE_RECEIPT_NAME
    inventory_receipt_path = target / INVENTORY_RECEIPT_NAME
    build_receipt_path = target / BUILD_RECEIPT_NAME
    observed_cache_file_sha = _write_new_canonical(cache_receipt_path, cache_receipt)
    observed_inventory_file_sha = _write_new_canonical(
        inventory_receipt_path, inventory_receipt
    )
    observed_build_file_sha = _write_new_canonical(build_receipt_path, build_receipt)
    if (
        observed_cache_file_sha != cache_receipt_file_sha
        or observed_inventory_file_sha != inventory_file_sha
    ):
        raise LocalQ1CacheBuilderError("canonical receipt serialization changed")
    return LocalQ1CacheBuildArtifacts(
        output_dir=target,
        cache_receipt_path=cache_receipt_path,
        inventory_receipt_path=inventory_receipt_path,
        build_receipt_path=build_receipt_path,
        cache_receipt=MappingProxyType(cache_receipt),
        inventory_receipt=MappingProxyType(inventory_receipt),
        build_receipt=MappingProxyType(dict(build_receipt)),
        cache_receipt_file_sha256=observed_cache_file_sha,
        inventory_receipt_file_sha256=observed_inventory_file_sha,
        build_receipt_file_sha256=observed_build_file_sha,
        read_only_cache=frozen_cache,
    )


def build_production_local_q1_cache(
    *,
    binding: LocalQ1ProductionBinding,
    output_dir: Path,
    producer_workers: int,
    inventory_config: Optional[LocalCacheInventoryConfig] = None,
    cache_limits: Optional[GeometryCacheLimits] = None,
) -> LocalQ1CacheBuildArtifacts:
    """Production convenience wrapper with the only three approved archives."""

    population = rebuild_production_local_q1_population(binding)
    factory = ProductionMaskLoaderFactory(
        mm_archive=binding.role_paths["mm_archive"],
        eccv_archive=binding.role_paths["eccv_archive"],
        synthetic_archive=binding.role_paths["synthetic_archive"],
        source_bundle=binding.source_bundle,
        expected_source_bundle_manifest_sha256=(
            population.precache_authority.source_bundle_manifest_sha256
        ),
    )
    return build_local_q1_cache(
        population=population,
        loader_factory=factory,
        output_dir=output_dir,
        inventory_config=inventory_config,
        cache_limits=cache_limits,
        producer_workers=producer_workers,
    )


def _load_canonical_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_content_sha256: str,
    name: str,
) -> Mapping[str, Any]:
    expected_file = _require_sha256(expected_file_sha256, name + " file sha256")
    expected_content = _require_sha256(
        expected_content_sha256, name + " content sha256"
    )
    try:
        if path.is_symlink() or not path.is_file():
            raise LocalQ1CacheBuilderError("{} is missing".format(name))
        payload = path.read_bytes()
    except OSError as exc:
        raise LocalQ1CacheBuilderError("cannot read {}".format(name)) from exc
    if not hmac.compare_digest(_sha256(payload), expected_file):
        raise LocalQ1CacheBuilderError("{} external file hash mismatch".format(name))
    receipt = _strict_json_bytes(payload, name)
    portable = _portable(dict(receipt), name)
    if payload != _canonical_json(portable):
        raise LocalQ1CacheBuilderError("{} is not canonical JSON".format(name))
    if not hmac.compare_digest(
        _receipt_content_sha256(portable, name), expected_content
    ):
        raise LocalQ1CacheBuilderError("{} external content hash mismatch".format(name))
    return MappingProxyType(dict(portable))


def _validate_output_tree(output_dir: Path) -> None:
    required_top = {
        CACHE_DIRECTORY_NAME,
        CACHE_RECEIPT_NAME,
        INVENTORY_RECEIPT_NAME,
        BUILD_RECEIPT_NAME,
    }
    # The research entrypoint freezes its explicit planning configuration next
    # to the completed cache receipts.  It is not part of the geometry cache
    # itself, but every plan/train/resume path reads and validates it
    # independently before reopening the cache.  Keep the generic cold-build
    # layout valid while accepting exactly this one known sidecar.
    optional_top = {PLANNING_CONFIG_NAME}
    try:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise LocalQ1CacheBuilderError("LOCAL-Q1 cache output is missing")
        observed_top = {item.name for item in output_dir.iterdir()}
        if not required_top.issubset(observed_top) or not observed_top.issubset(
            required_top | optional_top
        ):
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 cache output has missing or extra top-level entries"
            )
        planning_config = output_dir / PLANNING_CONFIG_NAME
        if PLANNING_CONFIG_NAME in observed_top and (
            planning_config.is_symlink() or not planning_config.is_file()
        ):
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 planning config sidecar is not a regular file"
            )
        cache_root = output_dir / CACHE_DIRECTORY_NAME
        for item in cache_root.rglob("*"):
            if item.is_symlink():
                raise LocalQ1CacheBuilderError("LOCAL-Q1 cache contains a symlink")
            relative = item.relative_to(cache_root)
            if item.is_dir():
                if relative == Path(".key-locks") or (
                    len(relative.parts) == 1
                    and re.fullmatch(r"[0-9a-f]{2}", relative.name)
                ):
                    continue
                raise LocalQ1CacheBuilderError(
                    "LOCAL-Q1 cache contains an extra directory"
                )
            text = relative.as_posix()
            if text in {".commit.lock", ".quota-ledger.json"}:
                continue
            if re.fullmatch(r"\.key-locks/[0-9a-f]{2}\.lock", text):
                continue
            if re.fullmatch(r"[0-9a-f]{2}/[0-9a-f]{64}\.npz", text):
                if relative.parts[0] != relative.stem[:2]:
                    raise LocalQ1CacheBuilderError(
                        "LOCAL-Q1 cache artifact path is non-canonical"
                    )
                continue
            raise LocalQ1CacheBuilderError("LOCAL-Q1 cache contains an extra file")
    except OSError as exc:
        raise LocalQ1CacheBuilderError("cannot inspect LOCAL-Q1 cache output") from exc


def reopen_local_q1_cache(
    *,
    population: LocalQ1Population,
    loader_factory: Callable[[], Callable[[MaskMemberRef], np.ndarray]],
    output_dir: Path,
    trust: LocalQ1CacheTrust,
    inventory_config: Optional[LocalCacheInventoryConfig] = None,
    cache_limits: Optional[GeometryCacheLimits] = None,
    replay_source_masks: bool = True,
) -> OpenedLocalQ1Cache:
    """Externally authorize and eagerly verify one completed cache.

    ``replay_source_masks=False`` is the research fast-resume path.  The
    frozen cache artifacts are still read and verified in full, but the
    already-frozen inventory receipt is reused instead of decoding every
    source mask and repeating the same census.  Cold/production callers keep
    the historical full source replay by default.
    """

    if not isinstance(population, LocalQ1Population):
        raise TypeError("population must be LocalQ1Population")
    if not callable(loader_factory):
        raise TypeError("loader_factory must be callable")
    if not isinstance(trust, LocalQ1CacheTrust):
        raise TypeError("trust must be LocalQ1CacheTrust")
    if type(replay_source_masks) is not bool:
        raise TypeError("replay_source_masks must be bool")
    if (
        trust.expected_freeze_file_sha256 != population.freeze_file_sha256
        or trust.expected_freeze_content_sha256 != population.freeze_content_sha256
        or trust.expected_run_plan_file_sha256
        != population.precache_authority.run_plan_file_sha256
        or trust.expected_run_plan_content_sha256
        != population.precache_authority.run_plan_content_sha256
        or trust.expected_source_bundle_manifest_sha256
        != population.precache_authority.source_bundle_manifest_sha256
    ):
        raise LocalQ1CacheBuilderError(
            "reopen plan/source/freeze anchors differ from rebuilt population lock"
        )
    settings = inventory_config or LocalCacheInventoryConfig()
    limits = cache_limits or GeometryCacheLimits()
    target = Path(output_dir)
    _validate_output_tree(target)
    cache_receipt = _load_canonical_receipt(
        target / CACHE_RECEIPT_NAME,
        expected_file_sha256=trust.expected_cache_receipt_file_sha256,
        expected_content_sha256=trust.expected_cache_receipt_content_sha256,
        name="fragment cache receipt",
    )
    inventory_receipt = _load_canonical_receipt(
        target / INVENTORY_RECEIPT_NAME,
        expected_file_sha256=trust.expected_inventory_receipt_file_sha256,
        expected_content_sha256=trust.expected_inventory_receipt_content_sha256,
        name="local cache inventory receipt",
    )
    build_receipt = _load_canonical_receipt(
        target / BUILD_RECEIPT_NAME,
        expected_file_sha256=trust.expected_build_receipt_file_sha256,
        expected_content_sha256=trust.expected_build_receipt_content_sha256,
        name="LOCAL-Q1 cache build receipt",
    )
    if (
        inventory_receipt.get("semantic_commitment_sha256")
        != trust.expected_inventory_semantic_sha256
        or inventory_receipt.get("cache_receipt_canonical_sha256")
        != trust.expected_cache_receipt_file_sha256
        or build_receipt.get("inventory", {}).get("semantic_commitment_sha256")
        != trust.expected_inventory_semantic_sha256
    ):
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 inventory semantic commitment differs from external lock"
        )
    if (
        build_receipt.get("schema_version") != LOCAL_Q1_CACHE_BUILD_RECEIPT_VERSION
        or build_receipt.get("status")
        != "complete_geometry_cache_frozen_zero_miss_no_model_no_test"
        or build_receipt.get("population_lock", {}).get("freeze_file_sha256")
        != trust.expected_freeze_file_sha256
        or build_receipt.get("population_lock", {}).get("freeze_content_sha256")
        != trust.expected_freeze_content_sha256
        or build_receipt.get("precache_authority")
        != {
            "run_plan_file_sha256": trust.expected_run_plan_file_sha256,
            "run_plan_content_sha256": trust.expected_run_plan_content_sha256,
            "source_bundle_manifest_sha256": (
                trust.expected_source_bundle_manifest_sha256
            ),
        }
        or build_receipt.get("cache", {}).get("receipt_file_sha256")
        != trust.expected_cache_receipt_file_sha256
        or build_receipt.get("cache", {}).get("receipt_content_sha256")
        != trust.expected_cache_receipt_content_sha256
        or build_receipt.get("inventory", {}).get("receipt_file_sha256")
        != trust.expected_inventory_receipt_file_sha256
        or build_receipt.get("inventory", {}).get("receipt_content_sha256")
        != trust.expected_inventory_receipt_content_sha256
        or build_receipt.get("cache", {}).get("artifact_count")
        != cache_receipt.get("artifact_count")
        or build_receipt.get("inventory", {}).get("unique_canonical_fragment_count")
        != cache_receipt.get("artifact_count")
        or build_receipt.get("scope")
        != {
            "experiment": "LOCAL-Q1",
            "input_modality": "canonical_bool_mask_only",
            "splits": ["train", "val"],
            "sealed_real_capability": False,
            "historical_test_capability": False,
            "model_backend_imported": False,
            "model_executed": False,
        }
    ):
        raise LocalQ1CacheBuilderError("LOCAL-Q1 build cross-locks are inconsistent")
    expected_roles = {
        name: population.input_role_locks[name].portable_dict()
        for name in sorted(population.input_role_locks)
    }
    if build_receipt["population_lock"].get("input_roles") != expected_roles:
        raise LocalQ1CacheBuilderError("LOCAL-Q1 build input-role locks changed")

    cache = GeometryArtifactCache.from_frozen_receipt(
        target / CACHE_DIRECTORY_NAME,
        cache_receipt,
        limits=limits,
        trust=FrozenReceiptTrust(
            expected_content_sha256=trust.expected_cache_receipt_content_sha256,
            expected_file_sha256=trust.expected_cache_receipt_file_sha256,
        ),
    )
    if replay_source_masks:
        replay = _inventory_with_fresh_loader(
            population, loader_factory, cache, settings
        )
        counts = replay.receipt["operational_counts"]
        if (
            replay.semantic_commitment_sha256
            != trust.expected_inventory_semantic_sha256
            or counts["cache_miss_count"] != 0
            or counts["cache_hit_count"] != counts["unique_canonical_fragment_count"]
        ):
            raise LocalQ1CacheBuilderError(
                "full production LOCAL-Q1 replay is not exact and zero-miss"
            )
    else:
        counts = inventory_receipt.get("operational_counts", {})
        build_inventory = build_receipt.get("inventory", {})
        if (
            counts.get("cache_miss_count") != 0
            or counts.get("cache_hit_count")
            != counts.get("unique_canonical_fragment_count")
            or build_inventory.get("read_only_replay_cache_miss_count") != 0
            or build_inventory.get("read_only_replay_cache_hit_count")
            != counts.get("unique_canonical_fragment_count")
        ):
            raise LocalQ1CacheBuilderError(
                "frozen LOCAL-Q1 inventory does not prove zero-miss replay"
            )
        replay = LocalCacheInventoryResult(
            receipt=inventory_receipt,
            cache_receipt=cache_receipt,
            semantic_commitment_sha256=(trust.expected_inventory_semantic_sha256),
            cache_receipt_canonical_sha256=(trust.expected_cache_receipt_file_sha256),
        )
    return OpenedLocalQ1Cache(
        population=population,
        cache=cache,
        inventory_replay=replay,
        build_receipt=build_receipt,
    )


def reopen_production_local_q1_cache(
    *,
    binding: LocalQ1ProductionBinding,
    output_dir: Path,
    trust: LocalQ1CacheTrust,
    inventory_config: Optional[LocalCacheInventoryConfig] = None,
    cache_limits: Optional[GeometryCacheLimits] = None,
) -> OpenedLocalQ1Cache:
    """Rebuild all production roles before delegating to strict cache reopen."""

    if not isinstance(trust, LocalQ1CacheTrust):
        raise TypeError("trust must be LocalQ1CacheTrust")
    if (
        trust.expected_freeze_file_sha256 != PRODUCTION_LOCAL_Q1_FREEZE_FILE_SHA256
        or trust.expected_freeze_content_sha256
        != PRODUCTION_LOCAL_Q1_FREEZE_CONTENT_SHA256
        or trust.expected_run_plan_file_sha256
        != PRODUCTION_LOCAL_Q1_RUN_PLAN_FILE_SHA256
        or trust.expected_run_plan_content_sha256
        != PRODUCTION_LOCAL_Q1_RUN_PLAN_CONTENT_SHA256
    ):
        raise LocalQ1CacheBuilderError(
            "reopen plan/freeze anchors differ from production experiment lock"
        )
    population = rebuild_production_local_q1_population(binding)
    factory = ProductionMaskLoaderFactory(
        mm_archive=binding.role_paths["mm_archive"],
        eccv_archive=binding.role_paths["eccv_archive"],
        synthetic_archive=binding.role_paths["synthetic_archive"],
        source_bundle=binding.source_bundle,
        expected_source_bundle_manifest_sha256=(
            population.precache_authority.source_bundle_manifest_sha256
        ),
    )
    return reopen_local_q1_cache(
        population=population,
        loader_factory=factory,
        output_dir=output_dir,
        trust=trust,
        inventory_config=inventory_config,
        cache_limits=cache_limits,
    )


def _parse_binding(value: str) -> Tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or role not in _ROLE_NAMES or not path:
        raise argparse.ArgumentTypeError("binding must be a canonical ROLE=PATH")
    return role, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bind",
        action="append",
        type=_parse_binding,
        required=True,
        metavar="ROLE=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producer-workers", type=int, default=4)
    parser.add_argument("--plan", type=Path, default=CANONICAL_LOCAL_Q1_RUN_PLAN_PATH)
    parser.add_argument("--source-bundle", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    pairs = list(args.bind)
    role_paths = dict(pairs)
    if len(role_paths) != len(pairs) or set(role_paths) != set(_ROLE_NAMES):
        raise LocalQ1CacheBuilderError(
            "CLI requires every canonical LOCAL-Q1 role exactly once"
        )
    artifacts = build_production_local_q1_cache(
        binding=LocalQ1ProductionBinding(
            plan_path=args.plan,
            source_bundle=args.source_bundle,
            role_paths=role_paths,
        ),
        output_dir=args.output_dir,
        producer_workers=args.producer_workers,
    )
    print(artifacts.build_receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUILD_RECEIPT_NAME",
    "CANONICAL_LOCAL_Q1_RUN_PLAN_PATH",
    "CACHE_DIRECTORY_NAME",
    "CACHE_RECEIPT_NAME",
    "INVENTORY_RECEIPT_NAME",
    "LOCAL_Q1_CACHE_BUILDER_VERSION",
    "LOCAL_Q1_CACHE_BUILD_RECEIPT_VERSION",
    "LOCAL_Q1_MM30K_POPULATION_SCHEMA_VERSION",
    "LOCAL_Q1_PLAN_ANCHOR_NORMALIZED_HASH_MODE",
    "LOCAL_Q1_RUN_PLAN_ROLE_SPECS",
    "LOCAL_Q1_RUN_PLAN_SCHEMA_VERSION",
    "LOCAL_Q1_SOURCE_BUNDLE_HASH_MODE",
    "PRODUCTION_LOCAL_Q1_FREEZE_CONTENT_SHA256",
    "PRODUCTION_LOCAL_Q1_FREEZE_FILE_SHA256",
    "PRODUCTION_LOCAL_Q1_IDENTITY_CONTENT_SHA256",
    "PRODUCTION_LOCAL_Q1_IDENTITY_MEMBER_COUNT",
    "PRODUCTION_LOCAL_Q1_RUN_PLAN_CONTENT_SHA256",
    "PRODUCTION_LOCAL_Q1_RUN_PLAN_FILE_SHA256",
    "PRODUCTION_LOCAL_Q1_ROLE_LOCKS",
    "LocalQ1CacheBuildArtifacts",
    "LocalQ1CacheBuilderError",
    "LocalQ1CacheTrust",
    "LocalQ1FileLock",
    "LocalQ1Population",
    "LocalQ1PrecacheAuthority",
    "LocalQ1ProductionBinding",
    "OpenedLocalQ1Cache",
    "ProductionMaskLoaderFactory",
    "attest_rebuilt_local_q1_population",
    "build_local_q1_cache",
    "build_production_local_q1_cache",
    "local_q1_plan_role_file_sha256",
    "local_q1_source_bundle_manifest",
    "prebuild_local_q1_fragment_reserve",
    "rebuild_production_local_q1_population",
    "reopen_local_q1_cache",
    "reopen_production_local_q1_cache",
    "verify_external_local_q1_run_plan",
    "verify_local_q1_production_plan",
]
