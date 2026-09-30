#!/usr/bin/env python3
"""Continue exact-six in a new root after a real-only authority correction.

The original launcher intentionally refuses to resume when any source hash
changes.  This utility preserves that fail-closed rule.  It verifies and
copies only the already-complete synthetic/corrosion prefix into a fresh
launcher root, records byte-level lineage, and then lets the unchanged
launcher resume the two real-data stages under the corrected source hashes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from experiments.rachel_n512_formal_30k import run_exact6_evaluation as launcher


SCHEMA_VERSION = "rachel-exact-six-prefix-continuation/1.0"
IMPORTED_STAGES = ("synthetic", "corrosion")
STALE_SELECTION_SHA256 = (
    "b79957e4b71c9da8d00ab1e9956af966399430b60e89fead5ab3d781056ca254"
)
FROZEN_SELECTION_SHA256 = (
    "ba469ad0bbc1f9f61c8e74bf6159b8846c1e37853724379e5cfaa1b42d60db6b"
)
REAL_EXTERNAL_SOURCE = (
    "staging/pairwise_v0_2/baselines/rachel_n512_real_external.py"
)
PAIRED_BOOTSTRAP_SOURCE = (
    "experiments/rachel_n512_formal_30k/paired_cluster_bootstrap.py"
)
LAUNCHER_SOURCE = "experiments/rachel_n512_formal_30k/run_exact6_evaluation.py"
EXPECTED_CHANGED_HASHED_SOURCES = {
    LAUNCHER_SOURCE,
    REAL_EXTERNAL_SOURCE,
    PAIRED_BOOTSTRAP_SOURCE,
}


def _verified_stamped_json(path: Path, description: str) -> Dict[str, object]:
    value = launcher._read_json(path, description)
    expected = value.get("content_sha256")
    body = dict(value)
    body.pop("content_sha256", None)
    if not isinstance(expected, str) or expected != launcher._canonical_sha256(body):
        raise launcher.ExactSixEvaluationGateError(
            description + " content SHA-256 differs"
        )
    return value


def _tree_manifest(path: Path) -> Tuple[Mapping[str, object], ...]:
    root = Path(path)
    if root.is_symlink():
        raise launcher.ExactSixEvaluationGateError(
            "prefix output must not be a symlink: " + str(root)
        )
    if root.is_file():
        return (
            {
                "path": root.name,
                "size": root.stat().st_size,
                "sha256": launcher._sha256_file(root),
            },
        )
    if not root.is_dir():
        raise launcher.ExactSixEvaluationGateError(
            "prefix output is not a file or directory: " + str(root)
        )
    rows = []
    for value in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if value.is_symlink():
            raise launcher.ExactSixEvaluationGateError(
                "prefix output contains a symlink: " + str(value)
            )
        if value.is_dir():
            continue
        if not value.is_file():
            raise launcher.ExactSixEvaluationGateError(
                "prefix output contains a special member: " + str(value)
            )
        rows.append(
            {
                "path": value.relative_to(root).as_posix(),
                "size": value.stat().st_size,
                "sha256": launcher._sha256_file(value),
            }
        )
    if not rows:
        raise launcher.ExactSixEvaluationGateError("prefix output tree is empty")
    return tuple(rows)


def _copy_output(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, copy_function=shutil.copy2)
    else:
        shutil.copy2(source, destination)


def _source_hash_map(root: Path, relative_paths: Sequence[str]) -> Dict[str, str]:
    result = {}
    for relative in relative_paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise launcher.ExactSixEvaluationGateError(
                "isolated source member is missing or symlinked: " + str(path)
            )
        result[relative] = launcher._sha256_file(path)
    return result


def _require_exact_constant_replacement(
    old_root: Path, new_root: Path, relative: str
) -> None:
    old_path = old_root / relative
    new_path = new_root / relative
    old_payload = old_path.read_bytes()
    new_payload = new_path.read_bytes()
    stale = STALE_SELECTION_SHA256.encode("ascii")
    frozen = FROZEN_SELECTION_SHA256.encode("ascii")
    if (
        old_payload.count(stale) != 1
        or frozen in old_payload
        or new_payload != old_payload.replace(stale, frozen)
    ):
        raise launcher.ExactSixEvaluationGateError(
            relative + " is not the reviewed one-token authority correction"
        )


def _recompute_prefix_authority(
    old_values: Mapping[str, object], expected_sha256: object
) -> Mapping[str, object]:
    source_root = old_values.get("source_root")
    if not isinstance(source_root, str) or not isinstance(expected_sha256, str):
        raise launcher.ExactSixEvaluationGateError(
            "prefix source/authority identity is missing"
        )
    code = """
import json
import os
from experiments.rachel_n512_formal_30k import run_exact6_evaluation as launcher
values = json.loads(os.environ["RACHEL_PREFIX_CONFIG"])
values["resume"] = False
values["preflight_only"] = True
config = launcher.ExactSixEvaluationConfig(**values)
print(json.dumps(launcher.freeze_all_training_authorities(config), sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = source_root
    environment["RACHEL_PREFIX_CONFIG"] = json.dumps(dict(old_values))
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=source_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise launcher.ExactSixEvaluationGateError(
            "cannot recompute prefix authority: " + completed.stderr[-1000:]
        )
    try:
        authority = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise launcher.ExactSixEvaluationGateError(
            "recomputed prefix authority is not JSON"
        ) from error
    if (
        not isinstance(authority, Mapping)
        or authority.get("content_sha256") != expected_sha256
    ):
        raise launcher.ExactSixEvaluationGateError(
            "current prefix source/training authority differs from its launcher config"
        )
    return authority


def _validate_balanced_authority(
    path: Path, manifest_path: Path
) -> Mapping[str, object]:
    authority = launcher._read_json(path, "frozen balanced authority")
    construction_receipt = authority.get("construction_receipt")
    construction = (
        construction_receipt.get("construction")
        if isinstance(construction_receipt, Mapping)
        else None
    )
    rows = (
        construction_receipt.get("constructed_pairs")
        if isinstance(construction_receipt, Mapping)
        else None
    )
    if (
        authority.get("status")
        != "complete_six_methods_balanced_with_constructed_cross_case_distractors_not_gt_negative"
        or authority.get("pair_count") != 1016
        or authority.get("positive_count") != 508
        or authority.get("negative_count") != 508
        or not isinstance(rows, list)
        or len(rows) != 469
        or construction.get("constructed_selection_sha256")
        != FROZEN_SELECTION_SHA256
        or launcher._canonical_sha256(rows) != FROZEN_SELECTION_SHA256
    ):
        raise launcher.ExactSixEvaluationGateError(
            "frozen balanced authority identity differs"
        )
    manifest = launcher._read_json(manifest_path, "real manifest")
    manifest_sha = manifest.get("manifest_sha256")
    if (
        not isinstance(manifest_sha, str)
        or authority.get("manifest_sha256") != manifest_sha
    ):
        raise launcher.ExactSixEvaluationGateError(
            "frozen balanced authority real manifest differs"
        )
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": launcher._sha256_file(path),
        "manifest_sha256": manifest_sha,
        "constructed_count": len(rows),
        "constructed_selection_sha256": FROZEN_SELECTION_SHA256,
        "constructed_rows_canonical_sha256": launcher._canonical_sha256(rows),
    }


def _config_from_prefix(
    old_config: Mapping[str, object], source_root: Path, output_root: Path
) -> launcher.ExactSixEvaluationConfig:
    raw = old_config.get("config")
    if not isinstance(raw, Mapping):
        raise launcher.ExactSixEvaluationGateError("prefix launcher config is malformed")
    values = dict(raw)
    values["source_root"] = source_root
    values["output_root"] = output_root
    values["resume"] = False
    values["preflight_only"] = False
    try:
        return launcher.ExactSixEvaluationConfig(**values)
    except (TypeError, ValueError) as error:
        raise launcher.ExactSixEvaluationGateError(
            "prefix launcher configuration cannot be restored: " + str(error)
        ) from error


def prepare_continuation(
    *,
    prefix_root: Path,
    source_root: Path,
    output_root: Path,
    balanced_authority_path: Path,
) -> launcher.ExactSixEvaluationConfig:
    prefix_root = prefix_root.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    output_root = Path(os.path.abspath(str(output_root)))
    if output_root.exists() or output_root.is_symlink():
        raise launcher.ExactSixEvaluationGateError(
            "continuation output root already exists"
        )
    old_config = _verified_stamped_json(
        prefix_root / "launcher_config.json", "prefix launcher config"
    )
    old_state = _verified_stamped_json(
        prefix_root / "launcher_state.json", "prefix launcher state"
    )
    old_stages = old_state.get("stages")
    if (
        old_state.get("schema_version") != launcher.STATE_SCHEMA_VERSION
        or old_state.get("config_sha256") != old_config.get("content_sha256")
        or not isinstance(old_stages, Mapping)
        or set(old_stages) != set(launcher.STAGES)
    ):
        raise launcher.ExactSixEvaluationGateError(
            "prefix launcher config/state linkage differs"
        )
    config = _config_from_prefix(old_config, source_root, output_root)
    launcher._validate_namespaces(config)
    authority = launcher.freeze_all_training_authorities(config)
    if (
        launcher.real_external.EXPECTED_CONSTRUCTED_SELECTION_SHA256
        != FROZEN_SELECTION_SHA256
    ):
        raise launcher.ExactSixEvaluationGateError(
            "new real evaluator does not contain the frozen selection authority"
        )
    config_record = launcher._config_record(config, authority)
    old_values = old_config.get("config")
    new_values = config_record.get("config")
    if not isinstance(old_values, Mapping) or not isinstance(new_values, Mapping):
        raise launcher.ExactSixEvaluationGateError("launcher config payload is malformed")
    ignored = {"source_root", "output_root"}
    old_comparable = {key: value for key, value in old_values.items() if key not in ignored}
    new_comparable = {key: value for key, value in new_values.items() if key not in ignored}
    if old_comparable != new_comparable:
        raise launcher.ExactSixEvaluationGateError(
            "continuation changes more than source/output roots"
        )

    old_authority = _recompute_prefix_authority(
        old_values, old_config.get("authority_content_sha256")
    )

    source_relatives = tuple(launcher._source_hashes())
    old_source_value = old_values.get("source_root")
    if not isinstance(old_source_value, str):
        raise launcher.ExactSixEvaluationGateError("prefix source root is missing")
    old_source_root = Path(old_source_value).resolve(strict=True)
    old_source_hashes = _source_hash_map(old_source_root, source_relatives)
    new_source_hashes = _source_hash_map(source_root, source_relatives)
    changed = tuple(
        relative
        for relative in source_relatives
        if old_source_hashes[relative] != new_source_hashes[relative]
    )
    if set(changed) != EXPECTED_CHANGED_HASHED_SOURCES:
        raise launcher.ExactSixEvaluationGateError(
            "hashed source changes do not match the reviewed authority fix: "
            + ",".join(changed)
        )
    for relative in (REAL_EXTERNAL_SOURCE, PAIRED_BOOTSTRAP_SOURCE):
        _require_exact_constant_replacement(old_source_root, source_root, relative)
    recorded_old_source_hashes = old_authority.get("source_sha256")
    if not isinstance(recorded_old_source_hashes, Mapping) or any(
        old_source_hashes.get(relative) != sha256
        for relative, sha256 in recorded_old_source_hashes.items()
    ):
        raise launcher.ExactSixEvaluationGateError(
            "prefix source bytes differ from its recomputed frozen authority"
        )

    stages = old_stages
    failed_real = stages.get("real_pair_only")
    if (
        not isinstance(failed_real, Mapping)
        or failed_real.get("status") != "failed"
        or FROZEN_SELECTION_SHA256 not in str(failed_real.get("last_error", ""))
    ):
        raise launcher.ExactSixEvaluationGateError(
            "prefix did not fail on the now-corrected frozen selection"
        )

    source_outputs: Dict[str, Path] = {}
    source_manifests: Dict[str, Tuple[Mapping[str, object], ...]] = {}
    for stage in IMPORTED_STAGES:
        row = stages.get(stage)
        if not isinstance(row, Mapping) or row.get("status") != "complete":
            raise launcher.ExactSixEvaluationGateError(
                stage + " is not complete in the prefix run"
            )
        source_outputs[stage] = launcher._validate_stage_output(
            prefix_root, stage, row.get("output")
        )
        source_manifests[stage] = _tree_manifest(source_outputs[stage])

    balanced_authority = _validate_balanced_authority(
        balanced_authority_path.resolve(strict=True), config.real_manifest_path
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix="." + output_root.name + ".prefix-import-",
            dir=str(output_root.parent),
        )
    )
    try:
        imported: Dict[str, Mapping[str, object]] = {}
        final_outputs: Dict[str, Path] = {}
        for stage in IMPORTED_STAGES:
            relative = Path(stage) / "imported-prefix-001" / source_outputs[stage].name
            temporary_output = temporary / relative
            _copy_output(source_outputs[stage], temporary_output)
            copied_manifest = _tree_manifest(temporary_output)
            if copied_manifest != source_manifests[stage]:
                raise launcher.ExactSixEvaluationGateError(
                    stage + " copied prefix bytes differ"
                )
            launcher._validate_stage_output(temporary, stage, str(temporary_output))
            final_output = output_root / relative
            final_outputs[stage] = final_output
            imported[stage] = {
                "source_output": str(source_outputs[stage]),
                "destination_output": str(final_output),
                "file_count": len(copied_manifest),
                "byte_count": sum(int(row["size"]) for row in copied_manifest),
                "tree_manifest_sha256": launcher._canonical_sha256(copied_manifest),
            }

        receipt: MutableMapping[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_verified_synthetic_corrosion_prefix_import",
            "source_prefix_root": str(prefix_root),
            "destination_root": str(output_root),
            "old_launcher_config_file_sha256": launcher._sha256_file(
                prefix_root / "launcher_config.json"
            ),
            "old_launcher_state_file_sha256": launcher._sha256_file(
                prefix_root / "launcher_state.json"
            ),
            "old_authority_content_sha256": old_config.get(
                "authority_content_sha256"
            ),
            "old_authority_recomputed_equal": True,
            "new_authority_content_sha256": authority.get("content_sha256"),
            "old_source_sha256": old_source_hashes,
            "new_source_sha256": new_source_hashes,
            "changed_hashed_sources": list(changed),
            "authority_fix": {
                "stale_unbacked_code_constant_sha256": STALE_SELECTION_SHA256,
                "corrected_frozen_selection_sha256": FROZEN_SELECTION_SHA256,
                "frozen_authority": balanced_authority,
            },
            "imported_stages": imported,
            "old_prefix_modified": False,
        }
        receipt["content_sha256"] = launcher._canonical_sha256(receipt)

        state = launcher._initial_state(str(config_record["content_sha256"]))
        for stage in IMPORTED_STAGES:
            old_row = stages[stage]
            row = state["stages"][stage]
            row.update(
                {
                    "status": "complete",
                    "attempts": int(old_row.get("attempts", 1)),
                    "output": str(final_outputs[stage]),
                    "imported_from": str(source_outputs[stage]),
                    "tree_manifest_sha256": imported[stage][
                        "tree_manifest_sha256"
                    ],
                }
            )
        state["prefix_import_receipt"] = str(output_root / "prefix_import_receipt.json")
        state["prefix_import_receipt_content_sha256"] = receipt["content_sha256"]
        launcher._stamp_state(state)
        launcher._write_json_new(temporary / "launcher_config.json", config_record)
        launcher._write_json_new(temporary / "launcher_state.json", state)
        launcher._write_json_new(temporary / "prefix_import_receipt.json", receipt)
        os.rename(temporary, output_root)
        launcher._fsync_directory(output_root.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    # Re-open through the normal resume gate before returning the configuration.
    launcher._prepare_output_root(replace(config, resume=True), config_record)
    for stage in IMPORTED_STAGES:
        launcher._validate_stage_output(output_root, stage, str(final_outputs[stage]))
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prefix-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--balanced-authority", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = prepare_continuation(
            prefix_root=arguments.prefix_root,
            source_root=arguments.source_root,
            output_root=arguments.output_root,
            balanced_authority_path=arguments.balanced_authority,
        )
        if arguments.prepare_only:
            result: Mapping[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "status": "continuation_prepared_no_real_stage_started",
                "output_root": str(config.output_root),
            }
        else:
            result = launcher.run_exact_six_evaluation(replace(config, resume=True))
    except (launcher.ExactSixEvaluationGateError, ValueError, OSError) as error:
        print("exact-six continuation refused: " + str(error), file=os.sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
