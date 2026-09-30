"""Read-only, isolated deep validators for the same-data benchmark queue.

This module deliberately exposes no training command.  The queue invokes it
under ``python -I`` to reuse the frozen upstream and benchmark loaders without
ever constructing a sealed-test or real-data dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
from typing import Any, Dict, Mapping, Optional, Sequence


UPSTREAM_SCHEMA = "rachel-benchmark-queue-upstream-verification/1.0"
PAIRING_SCHEMA = "rachel-benchmark-queue-pairingnet-verification/1.0"
SHREDDING_SCHEMA = "rachel-benchmark-queue-shreddingnet-verification/1.0"


class StrictVerifierError(RuntimeError):
    """A frozen artifact failed an independent loader or content check."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_directory(value: Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise StrictVerifierError(label + " must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise StrictVerifierError(label + " traverses a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise StrictVerifierError(label + " is not a directory")
    return resolved


def _regular(root: Path, relative: str, label: str) -> Path:
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise StrictVerifierError(label + " path is not confined")
    path = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise StrictVerifierError(label + " traverses a symlink")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise StrictVerifierError(label + " is not a regular file")
    return path


def _json(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StrictVerifierError(label + " is not readable JSON") from error
    if not isinstance(value, dict):
        raise StrictVerifierError(label + " is not one JSON object")
    return value


def _exact(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise StrictVerifierError(label + " fields differ")


def _content_sha(value: Mapping[str, object], label: str) -> str:
    declared = value.get("content_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise StrictVerifierError(label + " lacks content SHA-256")
    payload = dict(value)
    payload.pop("content_sha256", None)
    if hashlib.sha256(_canonical_bytes(payload)).hexdigest() != declared:
        raise StrictVerifierError(label + " content SHA-256 differs")
    return declared


def _finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def verify_upstreams(
    *, n512_run: Path, matched_run: Path, dataset_root: Path
) -> Dict[str, object]:
    """Strict-load both upstream winner sets and replay their shared authority."""

    from staging.pairwise_v0_2.baselines import rachel_matched_mm_evaluation as matched
    from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed

    nroot = _absolute_directory(n512_run, "N512 finalized run")
    mroot = _absolute_directory(matched_run, "matched-MM finalized run")
    expected_dataset = _absolute_directory(dataset_root, "Rachel dataset root")
    nreceipt, nsha, nwinners = sealed._freeze_completed_winners(nroot)
    sealed._require_formal_convergence(nreceipt, nwinners)
    mreceipt, msha, mwinners = matched.freeze_matched_mm_winners(
        mroot, device="cpu"
    )
    aligned = sealed._require_matched_training_alignment(nreceipt, mreceipt)
    if aligned != expected_dataset:
        raise StrictVerifierError("upstream aligned dataset root differs")
    evidence = sealed._matched_training_hash_evidence(
        nreceipt,
        mreceipt,
        n512_run_directory=nroot,
        matched_run_directory=mroot,
    )
    config_authority = sealed._canonical_config_authority(nwinners)
    nrows = {
        str(row["arm"]): row
        for row in nreceipt["arm_results"]
        if isinstance(row, Mapping)
    }
    n_winner_rows = {
        winner.arm: {
            "epoch": winner.epoch,
            "checkpoint_sha256": winner.checkpoint_sha256,
            "threshold_content_sha256": winner.threshold.content_sha256,
            "stop_reason": nrows[winner.arm]["stop_reason"],
            "convergence_claim": nrows[winner.arm]["convergence_claim"],
        }
        for winner in nwinners
    }
    m_winner_rows = {
        method: {
            "epoch": winner.epoch,
            "checkpoint_sha256": winner.checkpoint_sha256,
            "threshold_content_sha256": winner.threshold.content_sha256,
        }
        for method, winner in mwinners.items()
    }
    return {
        "schema_version": UPSTREAM_SCHEMA,
        "status": "verified_formal_upstream_winners_and_alignment",
        "n512": {
            "run_root": str(nroot),
            "receipt_sha256": nsha,
            "schema_version": nreceipt["schema_version"],
            "status": nreceipt["status"],
            "fingerprint_sha256": nreceipt["fingerprint_sha256"],
            "config": nreceipt["config"],
            "population": nreceipt["population"],
            "winners": n_winner_rows,
        },
        "matched_mm": {
            "run_root": str(mroot),
            "receipt_sha256": msha,
            "schema_version": mreceipt["schema_version"],
            "status": mreceipt["status"],
            "fingerprint_sha256": mreceipt["fingerprint_sha256"],
            "config": mreceipt["config"],
            "population": mreceipt["population"],
            "stop_reason": mreceipt["stop_reason"],
            "convergence_claim": mreceipt["convergence_claim"],
            "winners": m_winner_rows,
        },
        "canonical_config_authority": config_authority,
        "matched_training_hash_evidence": evidence,
        "dataset_root": str(aligned),
        "formal_convergence_verified": True,
        "winner_checkpoints_strict_loaded_cpu": True,
        "test_or_real_opened": False,
    }


def _read_jsonl(path: Path, label: str) -> list[Dict[str, object]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise StrictVerifierError(
                        label + " line " + str(line_number) + " is not an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StrictVerifierError(label + " is not readable JSONL") from error
    return rows


def verify_pairingnet(
    *,
    output_root: Path,
    dataset_root: Path,
    official_source_root: Path,
    device: str,
    precision: str,
    num_workers: int,
) -> Dict[str, object]:
    """Strict-load PairingNet winner/last and replay the frozen threshold."""

    import torch

    from staging.pairwise_v0_2.baselines import rachel_pairingnet_benchmark as pair

    root = _absolute_directory(output_root, "PairingNet output root")
    dataset = _absolute_directory(dataset_root, "PairingNet dataset root")
    official = _absolute_directory(official_source_root, "PairingNet official root")
    receipt_path = _regular(root, "completion_receipt.json", "PairingNet receipt")
    receipt = _json(receipt_path, "PairingNet receipt")
    _exact(
        receipt,
        {
            "schema_version", "status", "method_id", "official_commit",
            "adaptation_claim", "adaptation", "stop_reason",
            "convergence_demonstrated", "epochs_completed", "winner_epoch",
            "winner_checkpoint_kind", "adapter_source_sha256",
            "adaptation_contract_sha256", "selection_threshold_used",
            "selection_key", "resume_count", "resume_contract",
            "epoch_history_sha256", "last_checkpoint_sha256",
            "winner_checkpoint_sha256", "validation_threshold",
            "validation_threshold_file_sha256", "inference_contract_sha256",
            "winner_validation_report_sha256",
            "winner_validation_predictions_sha256", "population_audit",
            "official_source_audit", "sealed_synthetic_accessed",
            "real_data_accessed",
        },
        "PairingNet receipt",
    )
    epochs = receipt.get("epochs_completed")
    winner_epoch = receipt.get("winner_epoch")
    selection_key = receipt.get("selection_key")
    if (
        receipt.get("schema_version") != pair.SCHEMA_VERSION
        or receipt.get("status") != "train_validation_complete"
        or receipt.get("method_id") != pair.METHOD_ID
        or receipt.get("official_commit") != pair.OFFICIAL_COMMIT
        or receipt.get("stop_reason") != "validation_plateau"
        or receipt.get("convergence_demonstrated") is not True
        or type(epochs) is not int
        or not 20 <= epochs <= 128
        or type(winner_epoch) is not int
        or not 1 <= winner_epoch <= epochs
        or not isinstance(selection_key, list)
        or len(selection_key) != 7
        or not _finite_tree(selection_key)
        or receipt.get("selection_threshold_used") is not False
        or type(receipt.get("resume_count")) is not int
        or receipt["resume_count"] != 0
        or receipt.get("sealed_synthetic_accessed") is not False
        or receipt.get("real_data_accessed") is not False
    ):
        raise StrictVerifierError("PairingNet completion/convergence differs")
    if receipt.get("resume_contract") != {
        "checkpoint": "last.pt",
        "granularity": "completed_epoch",
        "partial_epoch_policy": "discard_and_replay_from_last_completed_epoch",
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
    }:
        raise StrictVerifierError("PairingNet resume contract differs")

    artifacts = {
        "winner": ("winner.pt", "winner_checkpoint_sha256"),
        "last": ("last.pt", "last_checkpoint_sha256"),
        "threshold": ("validation_threshold.json", "validation_threshold_file_sha256"),
        "inference_contract": ("inference_contract.json", "inference_contract_sha256"),
        "validation_report": ("winner_validation_report.json", "winner_validation_report_sha256"),
        "validation_predictions": ("winner_validation_predictions.jsonl", "winner_validation_predictions_sha256"),
    }
    paths: Dict[str, Path] = {}
    hashes = {}
    for label, (relative, field) in artifacts.items():
        path = _regular(root, relative, "PairingNet " + label)
        observed = _sha256_file(path)
        if observed != receipt.get(field):
            raise StrictVerifierError("PairingNet " + label + " hash differs")
        paths[label] = path
        hashes[label] = observed

    model_config = pair.PairingNetRachelModelConfig()
    run_config = pair.PairingNetRachelRunConfig(
        dataset_root=dataset,
        output_root=root,
        official_source_root=official,
        device=device,
        precision=precision,
        num_workers=num_workers,
    )
    pair._validate_formal_protocol(run_config, model_config)
    population = pair.audit_train_val_population(dataset)
    official_audit = pair.audit_official_source(official)
    if receipt.get("population_audit") != population:
        raise StrictVerifierError("PairingNet current population audit differs")
    if receipt.get("official_source_audit") != official_audit:
        raise StrictVerifierError("PairingNet current official audit differs")
    model = pair.load_frozen_pairingnet_checkpoint(paths["winner"], torch.device("cpu"))
    del model
    last = pair._load_last_checkpoint(
        paths["last"],
        device=torch.device("cpu"),
        model_config=model_config,
        run_config=run_config,
        population_audit=population,
        official_source_audit=official_audit,
        source_sha256=str(receipt["adapter_source_sha256"]),
        adaptation_contract_sha256=str(receipt["adaptation_contract_sha256"]),
    )
    if (
        last.get("completed_epoch") != epochs
        or last.get("best_epoch") != winner_epoch
        or last.get("best_key") != selection_key
        or last.get("epoch_history_sha256") != receipt.get("epoch_history_sha256")
        or int(last.get("plateau", -1)) < run_config.patience
    ):
        raise StrictVerifierError("PairingNet last/plateau binding differs")

    threshold = _json(paths["threshold"], "PairingNet threshold")
    threshold_value = pair.load_frozen_validation_threshold(
        paths["threshold"], paths["winner"]
    )
    if receipt.get("validation_threshold") != threshold:
        raise StrictVerifierError("PairingNet receipt threshold differs")
    predictions = _read_jsonl(
        paths["validation_predictions"], "PairingNet validation predictions"
    )
    if (
        len(predictions) != 3_000
        or len({row.get("pair_id") for row in predictions}) != 3_000
        or not _finite_tree(predictions)
    ):
        raise StrictVerifierError("PairingNet validation predictions differ")
    replayed = pair._fit_frozen_validation_threshold(
        predictions,
        winner_checkpoint_path=paths["winner"],
        model_config=model_config,
    )
    if replayed != threshold:
        raise StrictVerifierError("PairingNet validation threshold replay differs")
    inference_contract = _json(
        paths["inference_contract"], "PairingNet inference contract"
    )
    expected_contract = pair.frozen_inference_contract(
        threshold_value, str(threshold["artifact_content_sha256"])
    )
    if inference_contract != expected_contract:
        raise StrictVerifierError("PairingNet inference contract differs")
    report = _json(paths["validation_report"], "PairingNet validation report")
    if (
        set(report)
        != {
            "schema_version", "method_id", "selection_threshold_used",
            "selection_key", "primary_frozen_validation_threshold",
            "adapter_native_0_5_secondary", "validation_threshold",
            "sealed_synthetic_accessed", "real_data_accessed",
        }
        or report.get("schema_version")
        != "rachel-pairingnet-winner-validation/1.0"
        or report.get("method_id") != pair.METHOD_ID
        or report.get("selection_threshold_used") is not False
        or report.get("selection_key") != selection_key
        or report.get("validation_threshold") != threshold
        or report.get("sealed_synthetic_accessed") is not False
        or report.get("real_data_accessed") is not False
        or not _finite_tree(report)
    ):
        raise StrictVerifierError("PairingNet validation report differs")
    return {
        "schema_version": PAIRING_SCHEMA,
        "status": "verified_pairingnet_train_val_completion",
        "output_root": str(root),
        "completion_receipt_sha256": _sha256_file(receipt_path),
        "artifacts": hashes,
        "epochs_completed": epochs,
        "winner_epoch": winner_epoch,
        "plateau": last["plateau"],
        "stop_reason": receipt["stop_reason"],
        "validation_threshold": threshold_value,
        "manifest_content_sha256": {
            "train": population["manifests"]["train_sha256"],
            "val": population["manifests"]["val_sha256"],
        },
        "winner_and_last_strict_loaded_cpu": True,
        "threshold_replayed": True,
        "inference_contract_verified": True,
        "validation_report_verified": True,
        "test_or_real_opened": False,
    }


def verify_shreddingnet(
    *,
    output_root: Path,
    dataset_root: Path,
    official_source_root: Path,
) -> Dict[str, object]:
    """Strict-load the three-stage ShreddingNet bundle and its last progress."""

    import torch

    from staging.pairwise_v0_2.baselines import rachel_shreddingnet_benchmark as shred

    root = _absolute_directory(output_root, "ShreddingNet output root")
    dataset = _absolute_directory(dataset_root, "ShreddingNet dataset root")
    official = _absolute_directory(official_source_root, "ShreddingNet official root")
    freeze_path = _regular(root, "train_val_freeze.json", "ShreddingNet freeze")
    freeze = shred._read_verified_json(freeze_path, "ShreddingNet freeze")
    _exact(
        freeze,
        {
            "schema_version", "checkpoint_kind", "status", "method_id",
            "official_commit", "recipe", "runtime_batch_config",
            "adapter_identity", "dataset_binding", "train_val_provenance",
            "checkpoints", "threshold", "dataset_audit",
            "official_source_audit", "stage_completion_receipts",
            "inference_contract", "scope", "numeric_contract", "content_sha256",
        },
        "ShreddingNet freeze",
    )
    freeze_content = _content_sha(freeze, "ShreddingNet freeze")
    if (
        freeze.get("schema_version") != shred.FREEZE_SCHEMA_VERSION
        or freeze.get("checkpoint_kind") != shred.FREEZE_CHECKPOINT_KIND
        or freeze.get("status") != "complete_train_val_frozen_no_test_or_real"
        or freeze.get("method_id") != shred.METHOD_ID
        or freeze.get("official_commit") != shred.OFFICIAL_COMMIT
        or freeze.get("scope", {}).get("sealed_test_or_real_opened") is not False
    ):
        raise StrictVerifierError("ShreddingNet freeze identity differs")
    try:
        recipe = shred.ReleaseRecipe(**freeze["recipe"])
        runtime = shred.RuntimeBatchConfig(**freeze["runtime_batch_config"])
    except (KeyError, TypeError, ValueError) as error:
        raise StrictVerifierError("ShreddingNet recipe/runtime differs") from error
    shred._require_executable_release_recipe(recipe)
    current_audit, train_metadata, val_metadata = shred.audit_rachel_train_val(
        dataset
    )
    current_official = shred.audit_official_source(official)
    if freeze.get("dataset_audit") != current_audit:
        raise StrictVerifierError("ShreddingNet current dataset audit differs")
    if freeze.get("official_source_audit") != current_official:
        raise StrictVerifierError("ShreddingNet current official audit differs")
    inference = shred.load_frozen_inference(freeze_path, device="cpu")
    del inference

    run_contract_path = _regular(root, "run_contract.json", "ShreddingNet run contract")
    run_contract = shred._read_verified_json(run_contract_path, "run contract")
    _content_sha(run_contract, "ShreddingNet run contract")
    if (
        run_contract.get("schema_version") != shred.SCHEMA_VERSION
        or run_contract.get("checkpoint_kind") != "rachel_shreddingnet_run_contract"
        or run_contract.get("status") != "initialized_train_val_only"
        or run_contract.get("method_id") != shred.METHOD_ID
        or run_contract.get("official_commit") != shred.OFFICIAL_COMMIT
        or run_contract.get("output_root_lexical") != str(root)
        or run_contract.get("recipe") != asdict(recipe)
        or run_contract.get("runtime_batch_config") != asdict(runtime)
        or run_contract.get("adapter_identity") != freeze.get("adapter_identity")
        or run_contract.get("dataset_binding") != freeze.get("dataset_binding")
        or run_contract.get("train_val_provenance") != freeze.get("train_val_provenance")
        or run_contract.get("dataset_audit") != current_audit
        or run_contract.get("official_source_audit") != current_official
        or run_contract.get("scope")
        != {
            "sealed_test_or_real_opened": False,
            "rgb_opened": False,
            "fresh_no_clobber": True,
            "resume_requested": False,
            "root_state_before_invocation": "absent",
            "recovered_safe_empty_orphan": False,
        }
    ):
        raise StrictVerifierError("ShreddingNet run contract/provenance differs")

    stage_artifacts: Dict[str, object] = {}
    checkpoint_table = freeze.get("checkpoints")
    completion_table = freeze.get("stage_completion_receipts")
    if not isinstance(checkpoint_table, Mapping) or not isinstance(
        completion_table, Mapping
    ) or set(checkpoint_table) != {"coarse", "matching", "classify"} or set(
        completion_table
    ) != {"coarse", "matching", "classify"}:
        raise StrictVerifierError("ShreddingNet stage tables differ")
    matching_winner_sha: Optional[str] = None
    for stage in ("coarse", "matching", "classify"):
        completion_path = _regular(
            root, "stages/{}/completion.json".format(stage), stage + " completion"
        )
        completion = shred._read_verified_json(completion_path, stage + " completion")
        completion_content = _content_sha(completion, stage + " completion")
        if completion_table[stage] != {
            "path": "stages/{}/completion.json".format(stage),
            "content_sha256": completion_content,
        }:
            raise StrictVerifierError(stage + " freeze completion binding differs")
        settings = shred._stage_recipe(stage, recipe)
        history = completion.get("history")
        winner_epoch = completion.get("winner_epoch")
        train_indices = [
            index
            for index, row in enumerate(train_metadata)
            if row.label or stage == "classify"
        ]
        val_indices = [
            index
            for index, row in enumerate(val_metadata)
            if row.label or stage == "classify"
        ]
        if (
            completion.get("schema_version") != shred.SCHEMA_VERSION
            or completion.get("checkpoint_kind")
            != "rachel_shreddingnet_{}_completion".format(stage)
            or completion.get("status") != "complete_train_val_stage"
            or completion.get("method_id") != shred.METHOD_ID
            or completion.get("stage") != stage
            or completion.get("settings") != settings
            or completion.get("runtime_batch_config") != asdict(runtime)
            or completion.get("effective_batch_size") != settings["batch_size"]
            or completion.get("completed_epochs") != settings["epochs"]
            or not isinstance(history, list)
            or len(history) != settings["epochs"]
            or type(winner_epoch) is not int
            or not 0 <= winner_epoch < settings["epochs"]
            or not math.isfinite(float(completion.get("winner_value")))
            or completion.get("adapter_identity") != freeze.get("adapter_identity")
            or completion.get("dataset_binding") != freeze.get("dataset_binding")
            or completion.get("train_val_provenance") != freeze.get("train_val_provenance")
            or completion.get("sealed_test_or_real_opened") is not False
            or not _finite_tree(history)
        ):
            raise StrictVerifierError(stage + " completion/history differs")
        shred._validate_stage_history_orders(
            stage=stage,
            history=history,
            train_indices=train_indices,
            val_indices=val_indices,
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            seed=recipe.seed,
        )
        winner_row = checkpoint_table[stage]
        if not isinstance(winner_row, Mapping):
            raise StrictVerifierError(stage + " freeze winner row differs")
        winner_path = _regular(root, str(winner_row.get("path")), stage + " winner")
        winner_sha = _sha256_file(winner_path)
        if (
            winner_sha != winner_row.get("sha256")
            or winner_sha != completion.get("winner_sha256")
            or winner_row.get("winner_epoch_zero_based") != winner_epoch
            or winner_row.get("winner_epoch_one_based") != winner_epoch + 1
            or winner_row.get("checkpoint_kind")
            != shred._stage_checkpoint_kind(stage, "winner")
        ):
            raise StrictVerifierError(stage + " winner binding differs")
        progress_path = _regular(
            root, "stages/{}/progress.pt".format(stage), stage + " progress"
        )
        progress = shred._torch_load(progress_path, torch.device("cpu"))
        if (
            set(progress)
            != {
                "schema_version", "checkpoint_kind", "method_id", "stage",
                "epoch_completed", "best_epoch", "best_value", "history",
                "recipe", "matching_winner_sha256", "adapter_identity",
                "dataset_binding", "train_val_provenance",
                "validation_order_contract", "best_validation_order",
                "model_state_dict", "optimizer_state_dict",
                "scheduler_state_dict", "scaler_state_dict",
                "runtime_batch_config",
            }
            or progress.get("schema_version") != shred.SCHEMA_VERSION
            or progress.get("checkpoint_kind")
            != shred._stage_checkpoint_kind(stage, "progress")
            or progress.get("method_id") != shred.METHOD_ID
            or progress.get("stage") != stage
            or progress.get("epoch_completed") != settings["epochs"] - 1
            or progress.get("best_epoch") != winner_epoch
            or progress.get("best_value") != completion.get("winner_value")
            or progress.get("history") != history
            or progress.get("recipe") != asdict(recipe)
            or progress.get("runtime_batch_config") != asdict(runtime)
            or progress.get("adapter_identity") != freeze.get("adapter_identity")
            or progress.get("dataset_binding") != freeze.get("dataset_binding")
            or progress.get("train_val_provenance") != freeze.get("train_val_provenance")
            or progress.get("matching_winner_sha256")
            != (matching_winner_sha if stage == "classify" else None)
        ):
            raise StrictVerifierError(stage + " progress/last state differs")
        model = shred._stage_model(stage, recipe).to(torch.device("cpu"))
        model.load_state_dict(progress["model_state_dict"], strict=True)
        del model
        if stage == "matching":
            matching_winner_sha = winner_sha
        stage_artifacts[stage] = {
            "completion_file_sha256": _sha256_file(completion_path),
            "completion_content_sha256": completion_content,
            "winner_sha256": winner_sha,
            "progress_sha256": _sha256_file(progress_path),
            "completed_epochs": completion["completed_epochs"],
            "winner_epoch_zero_based": winner_epoch,
        }

    report_path = _regular(root, "validation_report.json", "ShreddingNet validation report")
    report = shred._read_verified_json(report_path, "validation report")
    report_content = _content_sha(report, "ShreddingNet validation report")
    if (
        report.get("schema_version")
        != "rachel-shreddingnet-validation-report/1.0"
        or report.get("status")
        != "complete_descriptive_validation_replay_no_test_or_real"
        or report.get("method_id") != shred.METHOD_ID
        or report.get("freeze_content_sha256") != freeze_content
        or report.get("adapter_identity") != freeze.get("adapter_identity")
        or report.get("dataset_binding") != freeze.get("dataset_binding")
        or report.get("train_val_provenance") != freeze.get("train_val_provenance")
        or report.get("validation_manifest_sha256")
        != freeze["dataset_audit"]["manifest_sha256"]["val"]
        or report.get("threshold_protocol", {}).get("primary_threshold")
        != freeze["threshold"]["threshold"]
        or report.get("scope", {}).get("validation_only") is not True
        or report.get("scope", {}).get("sealed_test_or_real_opened") is not False
        or not _finite_tree(report)
    ):
        raise StrictVerifierError("ShreddingNet validation report differs")
    return {
        "schema_version": SHREDDING_SCHEMA,
        "status": "verified_shreddingnet_train_val_completion",
        "output_root": str(root),
        "freeze_file_sha256": _sha256_file(freeze_path),
        "freeze_content_sha256": freeze_content,
        "run_contract_file_sha256": _sha256_file(run_contract_path),
        "validation_report_file_sha256": _sha256_file(report_path),
        "validation_report_content_sha256": report_content,
        "stage_artifacts": stage_artifacts,
        "manifest_content_sha256": freeze["dataset_binding"]["manifest_content_sha256"],
        "three_stage_winners_strict_loaded_cpu": True,
        "three_stage_progress_checkpoints_strict_loaded_cpu": True,
        "threshold_score_and_fit_replayed": True,
        "inference_contract_verified": True,
        "validation_report_verified": True,
        "test_or_real_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="read-only benchmark queue verifier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    upstream = subparsers.add_parser("upstreams")
    upstream.add_argument("--n512-run", type=Path, required=True)
    upstream.add_argument("--matched-run", type=Path, required=True)
    upstream.add_argument("--dataset-root", type=Path, required=True)
    pair = subparsers.add_parser("pairingnet")
    pair.add_argument("--output-root", type=Path, required=True)
    pair.add_argument("--dataset-root", type=Path, required=True)
    pair.add_argument("--official-source-root", type=Path, required=True)
    pair.add_argument("--device", required=True)
    pair.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    pair.add_argument("--num-workers", type=int, required=True)
    shred = subparsers.add_parser("shreddingnet")
    shred.add_argument("--output-root", type=Path, required=True)
    shred.add_argument("--dataset-root", type=Path, required=True)
    shred.add_argument("--official-source-root", type=Path, required=True)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        if parsed.command == "upstreams":
            result = verify_upstreams(
                n512_run=parsed.n512_run,
                matched_run=parsed.matched_run,
                dataset_root=parsed.dataset_root,
            )
        elif parsed.command == "pairingnet":
            result = verify_pairingnet(
                output_root=parsed.output_root,
                dataset_root=parsed.dataset_root,
                official_source_root=parsed.official_source_root,
                device=parsed.device,
                precision=parsed.precision,
                num_workers=parsed.num_workers,
            )
        else:
            result = verify_shreddingnet(
                output_root=parsed.output_root,
                dataset_root=parsed.dataset_root,
                official_source_root=parsed.official_source_root,
            )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
