#!/usr/bin/env python3
"""Build the portable, byte-deterministic LOCAL-Q1 pre-cache run plan.

This is a byte/metadata preflight only.  Archive members are never opened,
masks are never decoded, and no provider, model, backend, or cache is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from staging.pairwise_v0_2.training.local_q1_cache_builder import (
    LOCAL_Q1_RUN_PLAN_ROLE_SPECS,
    LOCAL_Q1_RUN_PLAN_SCHEMA_VERSION,
    PRODUCTION_LOCAL_Q1_FREEZE_CONTENT_SHA256,
    PRODUCTION_LOCAL_Q1_FREEZE_FILE_SHA256,
    LocalQ1CacheBuilderError,
    local_q1_plan_role_file_sha256,
    local_q1_source_bundle_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "local_q1_run_plan.json"
DEFAULT_SOURCE_BUNDLE = REPOSITORY_ROOT / "staging"
DEFAULT_ROLE_PATHS: Mapping[str, Path] = {
    "freeze_receipt": Path(__file__).resolve().parent / "local_q1_freeze.json",
    "mm_archive": Path(
        "/Users/yuqingzhang/Desktop/dataset/dunhuang_augmented_data.zip"
    ),
    "eccv_archive": Path("/Users/yuqingzhang/Desktop/ECCV/code/1113data.tar.gz"),
    "mm_fingerprint_cache": (
        REPOSITORY_ROOT / "staging/pairwise_v0_1/cache/mm_group_fingerprints.json"
    ),
    "eccv_fingerprint_cache": (
        REPOSITORY_ROOT / "staging/pairwise_v0_1/cache/eccv_group_fingerprints.json"
    ),
    "historical_split": (
        REPOSITORY_ROOT
        / "staging/pairwise_v0_1/data_gate/split_candidate_70_15_15.json"
    ),
    "synthetic_manifest": (
        REPOSITORY_ROOT / "staging/pairwise_v0_2/manifests/"
        "dunhuang_pairwise_mask_subset_v0_2/manifest/synthetic_groups.jsonl"
    ),
    "synthetic_archive": (
        REPOSITORY_ROOT / "staging/pairwise_v0_2/manifests/"
        "dunhuang_pairwise_mask_subset_v0_2/pairwise_mask_subset.zip"
    ),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalQ1CacheBuilderError("LOCAL-Q1 plan input is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 plan input JSON root must be an object"
        )
    return value


def _lock_matches(path: Path, lock: Any, *, require_bytes: bool = True) -> bool:
    if not isinstance(lock, Mapping):
        return False
    if lock.get("sha256") != _sha256_file(path):
        return False
    return not require_bytes or lock.get("bytes") == path.stat().st_size


def _validate_freeze(
    freeze_path: Path,
    role_paths: Mapping[str, Path],
    *,
    expected_freeze_file_sha256: str = PRODUCTION_LOCAL_Q1_FREEZE_FILE_SHA256,
    expected_freeze_content_sha256: str = PRODUCTION_LOCAL_Q1_FREEZE_CONTENT_SHA256,
) -> Mapping[str, Any]:
    if _sha256_file(freeze_path) != expected_freeze_file_sha256:
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 plan builder requires canonical freeze bytes"
        )
    freeze = _load_json(freeze_path)
    unsigned = dict(freeze)
    stored = unsigned.pop("content_sha256", None)
    observed = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if stored != observed or observed != expected_freeze_content_sha256:
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 plan builder requires canonical freeze content"
        )
    locks = freeze.get("locks")
    if not isinstance(locks, Mapping):
        raise LocalQ1CacheBuilderError("canonical LOCAL-Q1 freeze lacks locks")
    archives = locks.get("historical_archive_containers")
    metadata = locks.get("historical_metadata_artifacts")
    if not isinstance(archives, Mapping) or not isinstance(metadata, Mapping):
        raise LocalQ1CacheBuilderError(
            "canonical LOCAL-Q1 freeze lacks historical input locks"
        )
    checks = {
        "mm_archive": archives.get("mm_augmented"),
        "eccv_archive": archives.get("eccv_1113data"),
        "mm_fingerprint_cache": metadata.get("mm_fingerprint_cache"),
        "eccv_fingerprint_cache": metadata.get("eccv_fingerprint_cache"),
        "historical_split": metadata.get("historical_split"),
        "synthetic_manifest": locks.get("synthetic_manifest"),
    }
    for role, lock in checks.items():
        if not _lock_matches(role_paths[role], lock):
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 plan input differs from freeze: " + role
            )
    if not _lock_matches(
        role_paths["synthetic_archive"],
        locks.get("synthetic_archive_reference"),
        require_bytes=False,
    ):
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 plan input differs from freeze: synthetic_archive"
        )
    identity = locks.get("historical_identity_index")
    if not isinstance(identity, Mapping) or set(identity) != {
        "member_count",
        "content_sha256",
    }:
        raise LocalQ1CacheBuilderError(
            "canonical LOCAL-Q1 freeze identity-index lock changed"
        )
    return freeze


def build_local_q1_run_plan(
    *,
    role_paths: Mapping[str, Path],
    source_bundle: Path,
    output: Path,
    expected_freeze_file_sha256: str = PRODUCTION_LOCAL_Q1_FREEZE_FILE_SHA256,
    expected_freeze_content_sha256: str = PRODUCTION_LOCAL_Q1_FREEZE_CONTENT_SHA256,
) -> Mapping[str, Any]:
    """Write one canonical portable plan from external path-only bindings."""

    normalized = {str(role): Path(path) for role, path in role_paths.items()}
    if set(normalized) != set(LOCAL_Q1_RUN_PLAN_ROLE_SPECS):
        raise LocalQ1CacheBuilderError("LOCAL-Q1 plan builder role set is not exact")
    for role, path in normalized.items():
        if path.is_symlink() or not path.is_file():
            raise LocalQ1CacheBuilderError(
                "LOCAL-Q1 plan input is not a regular file: " + role
            )
    source_bundle = Path(source_bundle)
    if source_bundle.is_symlink() or not source_bundle.is_dir():
        raise LocalQ1CacheBuilderError(
            "LOCAL-Q1 plan source bundle is not a regular directory"
        )

    for value, name in (
        (expected_freeze_file_sha256, "freeze file SHA-256"),
        (expected_freeze_content_sha256, "freeze content SHA-256"),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise LocalQ1CacheBuilderError(name + " must be lowercase SHA-256")
    if (
        expected_freeze_file_sha256 == PRODUCTION_LOCAL_Q1_FREEZE_FILE_SHA256
        and expected_freeze_content_sha256
        == PRODUCTION_LOCAL_Q1_FREEZE_CONTENT_SHA256
    ):
        # Preserve the two-argument seam used by the existing fixture builder.
        freeze = _validate_freeze(normalized["freeze_receipt"], normalized)
    else:
        freeze = _validate_freeze(
            normalized["freeze_receipt"],
            normalized,
            expected_freeze_file_sha256=expected_freeze_file_sha256,
            expected_freeze_content_sha256=expected_freeze_content_sha256,
        )
    source_manifest = local_q1_source_bundle_manifest(source_bundle)
    roles = []
    for role in sorted(LOCAL_Q1_RUN_PLAN_ROLE_SPECS):
        spec = LOCAL_Q1_RUN_PLAN_ROLE_SPECS[role]
        path = normalized[role]
        roles.append(
            {
                "role": role,
                "logical_id": spec["logical_id"],
                "kind": spec["kind"],
                "hash_mode": spec["hash_mode"],
                "bytes": path.stat().st_size,
                "sha256": local_q1_plan_role_file_sha256(path, spec["hash_mode"]),
            }
        )
    plan: Dict[str, Any] = {
        "schema_version": LOCAL_Q1_RUN_PLAN_SCHEMA_VERSION,
        "status": "frozen_no_execution",
        "experiment": "LOCAL-Q1-PRECACHE",
        "freeze_content_sha256": freeze["content_sha256"],
        "identity_index": dict(freeze["locks"]["historical_identity_index"]),
        "roles": roles,
        "source_bundle": source_manifest,
        "scope": {
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
        },
    }
    plan["content_sha256"] = hashlib.sha256(_canonical_json(plan)).hexdigest()
    encoded = (
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_SOURCE_BUNDLE)
    parser.add_argument(
        "--freeze-file-sha256",
        default=PRODUCTION_LOCAL_Q1_FREEZE_FILE_SHA256,
    )
    parser.add_argument(
        "--freeze-content-sha256",
        default=PRODUCTION_LOCAL_Q1_FREEZE_CONTENT_SHA256,
    )
    for role, path in DEFAULT_ROLE_PATHS.items():
        parser.add_argument("--" + role.replace("_", "-"), type=Path, default=path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {role: getattr(args, role) for role in LOCAL_Q1_RUN_PLAN_ROLE_SPECS}
    build_local_q1_run_plan(
        role_paths=paths,
        source_bundle=args.source_bundle,
        output=args.output,
        expected_freeze_file_sha256=args.freeze_file_sha256,
        expected_freeze_content_sha256=args.freeze_content_sha256,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_ROLE_PATHS",
    "DEFAULT_SOURCE_BUNDLE",
    "build_local_q1_run_plan",
]
