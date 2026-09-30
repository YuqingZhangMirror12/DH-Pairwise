from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.rachel_n512_formal_30k import (
    continue_exact6_after_translation_gate_fix as continuation,
)
from experiments.rachel_n512_formal_30k import run_exact6_evaluation as launcher


def _write_stamped(path: Path, value: dict) -> dict:
    result = dict(value)
    result["content_sha256"] = launcher._canonical_sha256(result)
    launcher._write_json_new(path, result)
    return result


def _prefix_fixture(root: Path, stages: tuple[str, ...], *, order: bool) -> dict:
    root.mkdir()
    imported = {}
    state = launcher._initial_state("c" * 64)
    for stage in stages:
        output = root / (stage + ".bin")
        output.write_bytes(stage.encode("ascii"))
        tree_sha = "a" * 64
        state["stages"][stage].update(
            {
                "status": "complete",
                "attempts": 1,
                "output": str(output),
                "tree_manifest_sha256": tree_sha,
            }
        )
        imported[stage] = {
            "destination_output": str(output),
            "tree_manifest_sha256": tree_sha,
        }
    status = next(
        status
        for status, expected in launcher.PREFIX_IMPORT_STAGES_BY_STATUS.items()
        if expected == stages
    )
    receipt = {
        "schema_version": launcher.PREFIX_IMPORT_SCHEMA_VERSION,
        "status": status,
        "destination_root": str(root),
        "source_prefix_root": "/source-prefix",
        "old_authority_content_sha256": "b" * 64,
        "imported_stages": imported,
    }
    if order:
        receipt["imported_stage_order"] = list(stages)
    receipt["content_sha256"] = launcher._canonical_sha256(receipt)
    receipt_path = root / "prefix_import_receipt.json"
    launcher._write_json_new(receipt_path, receipt)
    state["prefix_import_receipt"] = str(receipt_path)
    state["prefix_import_receipt_content_sha256"] = receipt["content_sha256"]
    return state


def test_prefix_gate_keeps_two_stage_receipts_and_accepts_exact_three_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        launcher, "_output_tree_manifest_sha256", lambda path: "a" * 64
    )
    two_root = tmp_path / "two"
    two_state = _prefix_fixture(two_root, launcher.STAGES[:2], order=False)
    three_root = tmp_path / "three"
    three_state = _prefix_fixture(three_root, launcher.STAGES[:3], order=True)

    assert launcher._validate_prefix_import(two_root, two_state)[
        "imported_stage_order"
    ] == list(launcher.STAGES[:2])
    assert launcher._validate_prefix_import(three_root, three_state)[
        "imported_stage_order"
    ] == list(launcher.STAGES[:3])


def test_three_stage_prefix_requires_explicit_exact_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        launcher, "_output_tree_manifest_sha256", lambda path: "a" * 64
    )
    root = tmp_path / "three"
    state = _prefix_fixture(root, launcher.STAGES[:3], order=False)
    with pytest.raises(
        launcher.ExactSixEvaluationGateError,
        match="stage order is missing",
    ):
        launcher._validate_prefix_import(root, state)


def test_translation_failure_must_match_byte_for_byte() -> None:
    state = {
        "stages": {
            continuation.TRANSLATION_STAGE: {
                "status": "failed",
                "attempts": 1,
                "output": None,
                "last_error": continuation.EXPECTED_TRANSLATION_FAILURE,
            }
        }
    }
    assert continuation._require_exact_translation_failure(state)[
        "last_error"
    ] == continuation.EXPECTED_TRANSLATION_FAILURE
    state["stages"][continuation.TRANSLATION_STAGE]["last_error"] += " "
    with pytest.raises(
        launcher.ExactSixEvaluationGateError,
        match="failure identity differs",
    ):
        continuation._require_exact_translation_failure(state)


def test_source_transition_accepts_only_reviewed_exact_bytes() -> None:
    stable = "staging/stable.py"
    old = dict(continuation.EXPECTED_OLD_CHANGED_SHA256)
    new = dict(continuation.EXPECTED_NEW_CHANGED_SHA256)
    old[stable] = new[stable] = "f" * 64
    assert set(continuation._require_reviewed_source_transition(old, new)) == (
        continuation.EXPECTED_CHANGED_HASHED_SOURCES
    )

    tampered = dict(new)
    tampered[continuation.TRANSLATION_SOURCE] = "e" * 64
    with pytest.raises(
        launcher.ExactSixEvaluationGateError,
        match="reviewed old/new byte identity differs",
    ):
        continuation._require_reviewed_source_transition(old, tampered)


def test_prepare_copies_exact_three_stage_prefix_and_leaves_translation_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix_root = tmp_path / "prefix-003"
    prefix_root.mkdir()
    old_source_root = tmp_path / "source-002"
    old_source_root.mkdir()
    source_root = Path(launcher.__file__).resolve().parents[2]
    output_root = tmp_path / "fresh-004"

    old_hashes = dict(continuation.EXPECTED_OLD_CHANGED_SHA256)
    new_hashes = dict(continuation.EXPECTED_NEW_CHANGED_SHA256)
    old_hashes["staging/stable.py"] = "f" * 64
    new_hashes["staging/stable.py"] = "f" * 64
    authority_common = {
        "schema_version": launcher.SCHEMA_VERSION,
        "status": "all_train_validation_authorities_frozen_before_sealed_open",
        "fixed_training_authority": True,
    }

    def authority(source_hashes: dict[str, str]) -> dict:
        result = dict(authority_common)
        result["source_sha256"] = source_hashes
        result["content_sha256"] = launcher._canonical_sha256(result)
        return result

    old_authority = authority(old_hashes)
    new_authority = authority(new_hashes)
    old_config = launcher.ExactSixEvaluationConfig(
        source_root=old_source_root,
        n512_run_directory=tmp_path / "n512",
        matched_mm_run_directory=tmp_path / "matched",
        pairingnet_run_directory=tmp_path / "pairing",
        shreddingnet_freeze_path=tmp_path / "shredding" / "freeze.json",
        dataset_root=tmp_path / "dataset",
        real_manifest_path=tmp_path / "real-control" / "manifest.json",
        real_local_receipt_path=tmp_path / "real-control" / "receipt.json",
        real_main_root=tmp_path / "real-main",
        real_supp_root=tmp_path / "real-supp",
        output_root=prefix_root,
    )
    old_config_record = launcher._config_record(old_config, old_authority)
    launcher._write_json_new(prefix_root / "launcher_config.json", old_config_record)

    previous_receipt = _write_stamped(
        prefix_root / "prefix_import_receipt.json",
        {
            "schema_version": launcher.PREFIX_IMPORT_SCHEMA_VERSION,
            "status": "complete_verified_synthetic_corrosion_prefix_import",
        },
    )
    state = launcher._initial_state(old_config_record["content_sha256"])
    for stage in continuation.IMPORTED_STAGES:
        output = prefix_root / "outputs" / (stage + ".bin")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((stage + "-frozen").encode("ascii"))
        state["stages"][stage].update(
            {"status": "complete", "attempts": 1, "output": str(output)}
        )
    state["stages"][continuation.TRANSLATION_STAGE].update(
        {
            "status": "failed",
            "attempts": 1,
            "output": None,
            "last_error": continuation.EXPECTED_TRANSLATION_FAILURE,
        }
    )
    state["prefix_import_receipt"] = str(
        prefix_root / "prefix_import_receipt.json"
    )
    state["prefix_import_receipt_content_sha256"] = previous_receipt[
        "content_sha256"
    ]
    launcher._stamp_state(state)
    launcher._write_json_new(prefix_root / "launcher_state.json", state)

    def source_hash_map(root: Path, relatives) -> dict[str, str]:
        del relatives
        return old_hashes if Path(root).resolve() == old_source_root else new_hashes

    def validate_prefix(root: Path, observed_state: dict) -> dict:
        del observed_state
        if Path(root).resolve() == prefix_root:
            return {"imported_stage_order": list(launcher.STAGES[:2])}
        return {"imported_stage_order": list(continuation.IMPORTED_STAGES)}

    monkeypatch.setattr(launcher, "freeze_all_training_authorities", lambda c: new_authority)
    monkeypatch.setattr(launcher, "_source_hashes", lambda: new_hashes)
    monkeypatch.setattr(continuation.prefix_tools, "_source_hash_map", source_hash_map)
    monkeypatch.setattr(
        continuation.prefix_tools,
        "_recompute_prefix_authority",
        lambda values, expected: old_authority,
    )
    monkeypatch.setattr(
        launcher,
        "_validate_stage_output",
        lambda root, stage, value: Path(value).resolve(strict=True),
    )
    monkeypatch.setattr(launcher, "_validate_prefix_import", validate_prefix)

    config = continuation.prepare_continuation(
        prefix_root=prefix_root,
        source_root=source_root,
        output_root=output_root,
    )

    assert config.output_root == output_root
    published_state = json.loads(
        (output_root / "launcher_state.json").read_text(encoding="utf-8")
    )
    assert published_state["stages"][continuation.TRANSLATION_STAGE] == {
        "status": "pending",
        "attempts": 0,
        "output": None,
    }
    receipt = json.loads(
        (output_root / "prefix_import_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["imported_stage_order"] == list(continuation.IMPORTED_STAGES)
    assert receipt["translation_failure"]["matched_exactly"] is True
    assert receipt["old_prefix_modified"] is False
    for stage in continuation.IMPORTED_STAGES:
        source = Path(receipt["imported_stages"][stage]["source_output"])
        destination = Path(receipt["imported_stages"][stage]["destination_output"])
        assert destination.read_bytes() == source.read_bytes()
