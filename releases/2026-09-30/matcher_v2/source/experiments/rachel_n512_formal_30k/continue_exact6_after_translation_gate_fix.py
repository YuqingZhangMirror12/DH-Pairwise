#!/usr/bin/env python3
"""Continue exact-six after the reviewed real-translation authority fix.

This utility is intentionally narrower than a normal resume.  It accepts only
the exact failed ``_003`` shape: synthetic, corrosion, and real pair-only are
complete, while real translation failed with the reviewed matched-authority
message.  It copies those three completed stages byte-for-byte into a fresh
root and leaves only real translation pending under the corrected source.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from experiments.rachel_n512_formal_30k import (
    continue_exact6_after_authority_fix as prefix_tools,
)
from experiments.rachel_n512_formal_30k import run_exact6_evaluation as launcher


SCHEMA_VERSION = "rachel-exact-six-translation-gate-continuation/1.0"
IMPORTED_STAGES = launcher.STAGES[:3]
TRANSLATION_STAGE = launcher.STAGES[3]
PREFIX_IMPORT_STATUS = (
    "complete_verified_synthetic_corrosion_real_pair_only_prefix_import"
)
EXPECTED_TRANSLATION_FAILURE = (
    "RachelRealTranslationGTError:"
    "strict/balanced/wrapper matched authority differs"
)
LAUNCHER_SOURCE = "experiments/rachel_n512_formal_30k/run_exact6_evaluation.py"
TRANSLATION_SOURCE = (
    "staging/pairwise_v0_2/baselines/rachel_n512_real_translation_gt.py"
)
EXPECTED_CHANGED_HASHED_SOURCES = {LAUNCHER_SOURCE, TRANSLATION_SOURCE}
EXPECTED_OLD_CHANGED_SHA256 = {
    LAUNCHER_SOURCE: "9d2afe762a9a1ae2937171f2c8f1549f2c73e47997d9492d470fbf98c987d790",
    TRANSLATION_SOURCE: "0dd7d123616869d052819ff63e7f9cdcc5caa385c15f6ae06b38abe3e290cc98",
}
EXPECTED_NEW_CHANGED_SHA256 = {
    LAUNCHER_SOURCE: "490584eec908fe2a3ea45790a42172e064e493a2d53c500d41af94e17db1226d",
    TRANSLATION_SOURCE: "2e11c6b56fb744fe8447bd66d0bb4b66cfd435c204aac2f855b8f32bd8cabe15",
}


def _without_source_hashes(authority: Mapping[str, object]) -> Mapping[str, object]:
    value = dict(authority)
    value.pop("content_sha256", None)
    value.pop("source_sha256", None)
    return value


def _require_exact_translation_failure(
    state: Mapping[str, object],
) -> Mapping[str, object]:
    stages = state.get("stages")
    if not isinstance(stages, Mapping):
        raise launcher.ExactSixEvaluationGateError(
            "prefix launcher stages are malformed"
        )
    translation = stages.get(TRANSLATION_STAGE)
    if (
        not isinstance(translation, Mapping)
        or translation.get("status") != "failed"
        or translation.get("attempts") != 1
        or translation.get("output") is not None
        or translation.get("last_error") != EXPECTED_TRANSLATION_FAILURE
    ):
        raise launcher.ExactSixEvaluationGateError(
            "prefix translation failure identity differs"
        )
    return translation


def _require_reviewed_source_transition(
    old_hashes: Mapping[str, str], new_hashes: Mapping[str, str]
) -> Tuple[str, ...]:
    if set(old_hashes) != set(new_hashes):
        raise launcher.ExactSixEvaluationGateError(
            "old/new hashed source inventories differ"
        )
    changed = tuple(
        relative
        for relative in old_hashes
        if old_hashes[relative] != new_hashes[relative]
    )
    if set(changed) != EXPECTED_CHANGED_HASHED_SOURCES:
        raise launcher.ExactSixEvaluationGateError(
            "hashed source changes do not match the reviewed translation gate fix: "
            + ",".join(changed)
        )
    for relative in sorted(EXPECTED_CHANGED_HASHED_SOURCES):
        if (
            old_hashes.get(relative) != EXPECTED_OLD_CHANGED_SHA256[relative]
            or new_hashes.get(relative) != EXPECTED_NEW_CHANGED_SHA256[relative]
        ):
            raise launcher.ExactSixEvaluationGateError(
                relative + " reviewed old/new byte identity differs"
            )
    return changed


def _source_file_identity(path: Path) -> Mapping[str, object]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise launcher.ExactSixEvaluationGateError(
            "continuation source must be a regular file"
        )
    return {
        "path": str(resolved),
        "file_sha256": launcher._sha256_file(resolved),
    }


def _validate_prefix_state(
    old_config: Mapping[str, object], old_state: Mapping[str, object]
) -> Mapping[str, object]:
    stages = old_state.get("stages")
    if (
        old_config.get("schema_version") != launcher.SCHEMA_VERSION
        or old_state.get("schema_version") != launcher.STATE_SCHEMA_VERSION
        or old_state.get("status") != "running"
        or old_state.get("config_sha256") != old_config.get("content_sha256")
        or not isinstance(stages, Mapping)
        or set(stages) != set(launcher.STAGES)
        or old_state.get("terminal_receipt") is not None
    ):
        raise launcher.ExactSixEvaluationGateError(
            "prefix launcher config/state linkage differs"
        )
    _require_exact_translation_failure(old_state)
    return stages


def _same_file_hash(path: Path, expected: str, description: str) -> None:
    if launcher._sha256_file(path) != expected:
        raise launcher.ExactSixEvaluationGateError(description + " changed during copy")


def prepare_continuation(
    *,
    prefix_root: Path,
    source_root: Path,
    output_root: Path,
) -> launcher.ExactSixEvaluationConfig:
    prefix_root = prefix_root.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    output_root = Path(os.path.abspath(str(output_root)))
    if output_root.exists() or output_root.is_symlink():
        raise launcher.ExactSixEvaluationGateError(
            "continuation output root already exists"
        )
    actual_source_root = Path(launcher.__file__).resolve().parents[2]
    if source_root != actual_source_root:
        raise launcher.ExactSixEvaluationGateError(
            "continuation is not executing from the configured corrected source root"
        )

    old_config_path = prefix_root / "launcher_config.json"
    old_state_path = prefix_root / "launcher_state.json"
    old_receipt_path = prefix_root / "prefix_import_receipt.json"
    old_config = prefix_tools._verified_stamped_json(
        old_config_path, "prefix launcher config"
    )
    old_state = prefix_tools._verified_stamped_json(
        old_state_path, "prefix launcher state"
    )
    stages = _validate_prefix_state(old_config, old_state)
    previous_prefix_import = launcher._validate_prefix_import(prefix_root, old_state)
    if previous_prefix_import is None:
        raise launcher.ExactSixEvaluationGateError(
            "prefix continuation receipt is missing"
        )
    old_receipt = prefix_tools._verified_stamped_json(
        old_receipt_path, "previous prefix import receipt"
    )
    if (
        old_state.get("prefix_import_receipt") != str(old_receipt_path)
        or old_state.get("prefix_import_receipt_content_sha256")
        != old_receipt.get("content_sha256")
    ):
        raise launcher.ExactSixEvaluationGateError(
            "previous prefix receipt linkage differs"
        )

    old_file_sha256 = {
        "launcher_config": launcher._sha256_file(old_config_path),
        "launcher_state": launcher._sha256_file(old_state_path),
        "prefix_import_receipt": launcher._sha256_file(old_receipt_path),
    }

    config = prefix_tools._config_from_prefix(old_config, source_root, output_root)
    launcher._validate_namespaces(config)
    authority = launcher.freeze_all_training_authorities(config)
    config_record = launcher._config_record(config, authority)
    old_values = old_config.get("config")
    new_values = config_record.get("config")
    if not isinstance(old_values, Mapping) or not isinstance(new_values, Mapping):
        raise launcher.ExactSixEvaluationGateError(
            "launcher config payload is malformed"
        )
    ignored = {"source_root", "output_root"}
    if (
        {key: value for key, value in old_values.items() if key not in ignored}
        != {key: value for key, value in new_values.items() if key not in ignored}
    ):
        raise launcher.ExactSixEvaluationGateError(
            "continuation changes more than source/output roots"
        )

    old_authority = prefix_tools._recompute_prefix_authority(
        old_values, old_config.get("authority_content_sha256")
    )
    if _without_source_hashes(old_authority) != _without_source_hashes(authority):
        raise launcher.ExactSixEvaluationGateError(
            "training authority changed beyond reviewed source bytes"
        )

    source_relatives = tuple(launcher._source_hashes())
    old_source_value = old_values.get("source_root")
    if not isinstance(old_source_value, str):
        raise launcher.ExactSixEvaluationGateError("prefix source root is missing")
    old_source_root = Path(old_source_value).resolve(strict=True)
    old_source_hashes = prefix_tools._source_hash_map(
        old_source_root, source_relatives
    )
    new_source_hashes = prefix_tools._source_hash_map(source_root, source_relatives)
    changed = _require_reviewed_source_transition(
        old_source_hashes, new_source_hashes
    )
    if old_authority.get("source_sha256") != old_source_hashes:
        raise launcher.ExactSixEvaluationGateError(
            "prefix source bytes differ from its recomputed frozen authority"
        )
    if authority.get("source_sha256") != new_source_hashes:
        raise launcher.ExactSixEvaluationGateError(
            "corrected source bytes differ from the new frozen authority"
        )

    source_outputs: Dict[str, Path] = {}
    source_manifests: Dict[str, Tuple[Mapping[str, object], ...]] = {}
    for stage in IMPORTED_STAGES:
        row = stages.get(stage)
        if (
            not isinstance(row, Mapping)
            or row.get("status") != "complete"
            or type(row.get("attempts")) is not int  # noqa: E721
            or int(row["attempts"]) <= 0
        ):
            raise launcher.ExactSixEvaluationGateError(
                stage + " is not complete in the prefix run"
            )
        source_outputs[stage] = launcher._validate_stage_output(
            prefix_root, stage, row.get("output")
        )
        source_manifests[stage] = prefix_tools._tree_manifest(
            source_outputs[stage]
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
        destination_parent = {
            "synthetic": "synthetic",
            "corrosion": "corrosion",
            "real_pair_only": "real",
        }
        for stage in IMPORTED_STAGES:
            relative = (
                Path(destination_parent[stage])
                / "imported-prefix-001"
                / source_outputs[stage].name
            )
            temporary_output = temporary / relative
            prefix_tools._copy_output(source_outputs[stage], temporary_output)
            copied_manifest = prefix_tools._tree_manifest(temporary_output)
            if copied_manifest != source_manifests[stage]:
                raise launcher.ExactSixEvaluationGateError(
                    stage + " copied prefix bytes differ"
                )
            launcher._validate_stage_output(
                temporary, stage, str(temporary_output)
            )
            final_output = output_root / relative
            final_outputs[stage] = final_output
            imported[stage] = {
                "source_output": str(source_outputs[stage]),
                "destination_output": str(final_output),
                "file_count": len(copied_manifest),
                "byte_count": sum(int(row["size"]) for row in copied_manifest),
                "tree_manifest_sha256": launcher._canonical_sha256(
                    copied_manifest
                ),
            }

        # Fail if the prior run or either source tree moved while bytes copied.
        for name, path in (
            ("launcher_config", old_config_path),
            ("launcher_state", old_state_path),
            ("prefix_import_receipt", old_receipt_path),
        ):
            _same_file_hash(path, old_file_sha256[name], "prefix " + name)
        for stage in IMPORTED_STAGES:
            if prefix_tools._tree_manifest(source_outputs[stage]) != source_manifests[
                stage
            ]:
                raise launcher.ExactSixEvaluationGateError(
                    stage + " source output changed during copy"
                )
        if (
            prefix_tools._source_hash_map(old_source_root, source_relatives)
            != old_source_hashes
            or prefix_tools._source_hash_map(source_root, source_relatives)
            != new_source_hashes
        ):
            raise launcher.ExactSixEvaluationGateError(
                "old/new isolated source bytes changed during copy"
            )

        receipt: MutableMapping[str, object] = {
            "schema_version": launcher.PREFIX_IMPORT_SCHEMA_VERSION,
            "continuation_schema_version": SCHEMA_VERSION,
            "status": PREFIX_IMPORT_STATUS,
            "source_prefix_root": str(prefix_root),
            "destination_root": str(output_root),
            "imported_stage_order": list(IMPORTED_STAGES),
            "source_prefix": {
                "launcher_config_file_sha256": old_file_sha256[
                    "launcher_config"
                ],
                "launcher_config_content_sha256": old_config.get(
                    "content_sha256"
                ),
                "launcher_state_file_sha256": old_file_sha256["launcher_state"],
                "launcher_state_content_sha256": old_state.get("content_sha256"),
                "previous_prefix_import_receipt_file_sha256": old_file_sha256[
                    "prefix_import_receipt"
                ],
                "previous_prefix_import_receipt_content_sha256": old_receipt.get(
                    "content_sha256"
                ),
                "previous_prefix_import": previous_prefix_import,
            },
            "translation_failure": {
                "stage": TRANSLATION_STAGE,
                "attempts": 1,
                "last_error": EXPECTED_TRANSLATION_FAILURE,
                "matched_exactly": True,
            },
            "old_authority_content_sha256": old_authority.get("content_sha256"),
            "old_authority_recomputed_equal": True,
            "new_authority_content_sha256": authority.get("content_sha256"),
            "authority_equal_except_source_sha256": True,
            "old_source_sha256": old_source_hashes,
            "new_source_sha256": new_source_hashes,
            "changed_hashed_sources": list(changed),
            "reviewed_changed_source_sha256": {
                relative: {
                    "old": old_source_hashes[relative],
                    "new": new_source_hashes[relative],
                }
                for relative in changed
            },
            "continuation_source": _source_file_identity(Path(__file__)),
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
                    "attempts": int(old_row["attempts"]),
                    "output": str(final_outputs[stage]),
                    "imported_from": str(source_outputs[stage]),
                    "tree_manifest_sha256": imported[stage][
                        "tree_manifest_sha256"
                    ],
                }
            )
        state["prefix_import_receipt"] = str(
            output_root / "prefix_import_receipt.json"
        )
        state["prefix_import_receipt_content_sha256"] = receipt["content_sha256"]
        launcher._stamp_state(state)

        launcher._write_json_new(temporary / "launcher_config.json", config_record)
        launcher._write_json_new(temporary / "launcher_state.json", state)
        launcher._write_json_new(
            temporary / "prefix_import_receipt.json", receipt
        )
        os.rename(temporary, output_root)
        launcher._fsync_directory(output_root.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    # Re-open every newly written identity through the normal resume gates.
    reopened_config = prefix_tools._verified_stamped_json(
        output_root / "launcher_config.json", "continuation launcher config"
    )
    reopened_state = prefix_tools._verified_stamped_json(
        output_root / "launcher_state.json", "continuation launcher state"
    )
    reopened_receipt = prefix_tools._verified_stamped_json(
        output_root / "prefix_import_receipt.json", "continuation prefix receipt"
    )
    if (
        reopened_config != config_record
        or reopened_state != state
        or reopened_receipt != receipt
    ):
        raise launcher.ExactSixEvaluationGateError(
            "continuation artifacts differ after atomic publish"
        )
    launcher._prepare_output_root(replace(config, resume=True), config_record)
    validated_import = launcher._validate_prefix_import(output_root, reopened_state)
    if (
        not isinstance(validated_import, Mapping)
        or validated_import.get("imported_stage_order") != list(IMPORTED_STAGES)
    ):
        raise launcher.ExactSixEvaluationGateError(
            "three-stage prefix import did not pass the normal resume gate"
        )
    for stage in IMPORTED_STAGES:
        launcher._validate_stage_output(
            output_root, stage, reopened_state["stages"][stage]["output"]
        )
    _require_exact_translation_failure(old_state)
    if reopened_state["stages"][TRANSLATION_STAGE] != {
        "status": "pending",
        "attempts": 0,
        "output": None,
    }:
        raise launcher.ExactSixEvaluationGateError(
            "new translation stage is not fresh and pending"
        )
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prefix-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = prepare_continuation(
            prefix_root=arguments.prefix_root,
            source_root=arguments.source_root,
            output_root=arguments.output_root,
        )
        if arguments.prepare_only:
            receipt_path = config.output_root / "prefix_import_receipt.json"
            result: Mapping[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "status": "continuation_prepared_no_translation_stage_started",
                "output_root": str(config.output_root),
                "prefix_import_receipt": str(receipt_path),
                "prefix_import_receipt_file_sha256": launcher._sha256_file(
                    receipt_path
                ),
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
