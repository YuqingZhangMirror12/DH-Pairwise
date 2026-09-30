#!/usr/bin/env python3
"""Build the portable, byte-deterministic C0-N-Q1 production run plan.

This is a metadata/byte preflight only.  Archives are hashed as files; no
archive member is opened, no mask is decoded, and no backend/model is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    HISTORICAL_TEST_ACCESS_EVIDENCE,
    HistoricalIdentityIndex,
    IDENTITY_INDEX_CONTENT_SCHEMA,
)
from staging.pairwise_v0_2.training.c0_runner import (
    C0RunnerContract,
    C0RunnerError,
    PRODUCTION_FREEZE_CONTENT_SHA256,
    PRODUCTION_FREEZE_FILE_SHA256,
    PRODUCTION_RUN_PLAN_ROLE_SPECS,
    RUN_PLAN_SCHEMA_VERSION,
    SOURCE_BUNDLE_HASH_MODE,
    c0_plan_role_file_sha256,
    c0_source_bundle_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "c0_run_plan.json"
DEFAULT_ROLE_PATHS: Mapping[str, Path] = {
    "freeze_receipt": Path(__file__).resolve().parent / "c0_n_q1_freeze.json",
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
    "source_bundle": REPOSITORY_ROOT / "staging",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise C0RunnerError("C0 plan input is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise C0RunnerError("C0 plan input JSON root must be an object")
    return value


def _validate_freeze(
    freeze_path: Path,
    role_paths: Mapping[str, Path],
) -> Mapping[str, Any]:
    if _sha256_file(freeze_path) != PRODUCTION_FREEZE_FILE_SHA256:
        raise C0RunnerError("C0 plan builder requires the canonical freeze bytes")
    freeze = _load_json(freeze_path)
    unsigned = dict(freeze)
    stored = unsigned.pop("content_sha256", None)
    observed = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if stored != observed or observed != PRODUCTION_FREEZE_CONTENT_SHA256:
        raise C0RunnerError("C0 plan builder requires canonical freeze content")

    locks = freeze.get("locks")
    archives = locks.get("archives") if isinstance(locks, Mapping) else None
    metadata = locks.get("metadata_artifacts") if isinstance(locks, Mapping) else None
    if not isinstance(archives, Mapping) or not isinstance(metadata, Mapping):
        raise C0RunnerError("canonical freeze lacks input locks")
    checks = {
        "mm_archive": archives.get("mm_augmented"),
        "eccv_archive": archives.get("eccv_1113data"),
        "mm_fingerprint_cache": metadata.get("mm_fingerprint_cache"),
        "eccv_fingerprint_cache": metadata.get("eccv_fingerprint_cache"),
        "historical_split": metadata.get("historical_split"),
    }
    for role, lock in checks.items():
        path = role_paths[role]
        if not isinstance(lock, Mapping) or (
            lock.get("sha256") != _sha256_file(path)
            or lock.get("bytes") != path.stat().st_size
        ):
            raise C0RunnerError("C0 plan input differs from freeze: " + role)
    return freeze


def build_c0_run_plan(
    *,
    role_paths: Mapping[str, Path],
    output: Path,
) -> Mapping[str, Any]:
    """Build and write one canonical portable plan from path-only bindings."""

    normalized = {str(role): Path(path) for role, path in role_paths.items()}
    if set(normalized) != set(PRODUCTION_RUN_PLAN_ROLE_SPECS):
        raise C0RunnerError("C0 plan builder role set is not exact")
    for role, path in normalized.items():
        if path.is_symlink():
            raise C0RunnerError("C0 plan input cannot be a symlink: " + role)
        is_bundle = role == "source_bundle"
        if (is_bundle and not path.is_dir()) or (not is_bundle and not path.is_file()):
            raise C0RunnerError("C0 plan input has wrong path kind: " + role)

    freeze = _validate_freeze(normalized["freeze_receipt"], normalized)
    index = HistoricalIdentityIndex.from_files(
        mm_cache_path=normalized["mm_fingerprint_cache"],
        eccv_cache_path=normalized["eccv_fingerprint_cache"],
        split_path=normalized["historical_split"],
    )
    source_bundle = c0_source_bundle_manifest(normalized["source_bundle"])
    roles = []
    for role in sorted(PRODUCTION_RUN_PLAN_ROLE_SPECS):
        spec = PRODUCTION_RUN_PLAN_ROLE_SPECS[role]
        path = normalized[role]
        is_bundle = spec["hash_mode"] == SOURCE_BUNDLE_HASH_MODE
        roles.append(
            {
                "role": role,
                "logical_id": spec["logical_id"],
                "kind": spec["kind"],
                "hash_mode": spec["hash_mode"],
                "bytes": (
                    int(source_bundle["total_bytes"])
                    if is_bundle
                    else path.stat().st_size
                ),
                "sha256": c0_plan_role_file_sha256(path, spec["hash_mode"]),
            }
        )

    plan: Dict[str, Any] = {
        "schema_version": RUN_PLAN_SCHEMA_VERSION,
        "status": "frozen_no_execution",
        "experiment": "C0-N-Q1",
        "contract": dict(C0RunnerContract().portable_dict()),
        "freeze_content_sha256": freeze["content_sha256"],
        "identity_index": {
            "schema_version": IDENTITY_INDEX_CONTENT_SCHEMA,
            "member_count": index.identity_count,
            "content_sha256": index.content_sha256,
        },
        "roles": roles,
        "source_bundle": source_bundle,
        "scope": {
            "archive_bytes_hashed": True,
            "archive_members_opened": False,
            "mask_pixels_decoded": False,
            "model_or_backend_created": False,
            "historical_test_access": dict(HISTORICAL_TEST_ACCESS_EVIDENCE),
            "sealed_real_read": False,
        },
    }
    plan["content_sha256"] = hashlib.sha256(_canonical_json(plan)).hexdigest()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    for role, path in DEFAULT_ROLE_PATHS.items():
        parser.add_argument("--" + role.replace("_", "-"), type=Path, default=path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {role: getattr(args, role) for role in PRODUCTION_RUN_PLAN_ROLE_SPECS}
    build_c0_run_plan(role_paths=paths, output=args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUTPUT", "DEFAULT_ROLE_PATHS", "build_c0_run_plan"]
