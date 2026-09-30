"""Dedicated S3 VAL-only freeze: fixed epoch20 plus auxiliary winners13..20.

This never reads a live S0-S2 winner/budget freeze and never opens TEST/REAL/OOD.
An existing joint trajectory may supply its epoch13..20 checkpoints and VAL
rows, but its old [5,20] selected winner is deliberately not an input. Only
run-owned s3_freezes artifacts are written; original training files stay intact.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from experiments.rachel_n512_formal_30k.score_design_stages import (
    EVALUATION_SCHEMA as SCHEMA, phase, stage_protocol,
)
from experiments.rachel_n512_formal_30k.score_design_input_variants import full24_reference_config
from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
from experiments.rachel_n512_formal_30k.run_layout_decoder_experiment import classification, fit_threshold
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig

SEED, TRAIN_COUNT, VAL_COUNT, BUDGET = 260913, 24000, 3000, 20
ELIGIBLE = tuple(range(13, 21))
SELECTIONS = ("fixed_epoch", "max_f1", "recall95")
STAGED_CHECKPOINT_SCHEMA = "rachel-score-staged-checkpoint/1"
JOINT_CHECKPOINT_SCHEMA = "rachel-score-design-training/1"
BRANCHES = ("coarse", "local", "fused")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_new_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != canonical(document):
            raise ValueError("refusing to replace a different existing S3 freeze")
        return
    # A crash must not leave a half-written final freeze that blocks resume.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".s3-freeze-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(canonical(document), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)  # Same filesystem; atomic publication, no replacement.
        except FileExistsError:
            if json.loads(path.read_text()) != canonical(document):
                raise ValueError("another writer published a different S3 freeze")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_training_epoch(root, epoch, *, expected_sha256=None):
    """Read only run-owned recovery checkpoints, whose NumPy RNG needs pickle.

    Unlike a foreign/published weight file, these are explicitly trusted local
    trainer artifacts. They include optimizer and NumPy RNG state, unsupported
    by the old sealed-test weights-only loader. Restrict the filename and root;
    once frozen, verify its SHA before loading. Never accepts winner.pt.
    """
    if epoch not in ELIGIBLE:
        raise ValueError("only original epoch13..20 checkpoints may be imported")
    root = Path(root).resolve(strict=True)
    path = (root / ("epoch_%03d.pt" % epoch)).resolve(strict=True)
    if path.parent != root or path.name != "epoch_%03d.pt" % epoch:
        raise ValueError("checkpoint must be an original epoch file inside the run")
    if expected_sha256 is not None and sealed._sha256_file(path) != expected_sha256:
        raise ValueError("checkpoint differs from frozen SHA before load")
    result = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(result, dict):
        raise ValueError("trainer epoch checkpoint must be a dictionary")
    return result


def _model_design(payload, schedule):
    if schedule == "staged":
        if (payload.get("s3_checkpoint_schema") != STAGED_CHECKPOINT_SCHEMA
                or payload.get("s3_training_schema") != "rachel-score-staged-training/1"
                or "score_design_schema" in payload):
            raise ValueError("staged requires the dedicated S3 checkpoint schema")
        design = payload["s3_model_metadata"]
        expected_kind = "score_s3_" + design["architecture"]
    elif schedule == "joint":
        if payload.get("score_design_schema") != JOINT_CHECKPOINT_SCHEMA or "s3_checkpoint_schema" in payload:
            raise ValueError("joint import requires an original score-design epoch checkpoint")
        design = payload["score_design"]
        expected_kind = "score_design_" + design["architecture"]
        if design.get("training_mode") != "joint":
            raise ValueError("imported comparison trajectory must actually be joint")
    else:
        raise ValueError("schedule must be staged or joint")
    if design.get("architecture") not in ("candidate_pair", "candidate_dual") or payload.get("model_kind") != expected_kind:
        raise ValueError("S3 requires the same registered candidate P/R architecture")
    if canonical(design["model_config"]) != canonical(asdict(full24_reference_config())):
        raise ValueError("S3 requires unchanged Full24 model inputs and architecture")
    if canonical(payload["model_config"]) != canonical(design["model_config"]):
        raise ValueError("top-level model metadata differs from the candidate wrapper")
    return design


def checkpoint_contract(payload, *, schedule, epoch):
    """Small, explicit protocol audit; no predictions/targets are consulted."""
    if epoch not in ELIGIBLE:
        raise ValueError("S3 auxiliary checkpoint window is exactly13..20")
    if (payload.get("epoch") != epoch or payload.get("seed") != SEED
            or payload.get("global_exposure") != epoch * TRAIN_COUNT
            or payload.get("optimizer_updates") != epoch * TRAIN_COUNT // 16
            or payload.get("completed_segments") != epoch * 4
            or payload.get("formal_training_counted") is not True
            or payload.get("source_weights_loaded") is not False
            or payload.get("resample_contour_cap") is not None
            or payload.get("seam_loss_enabled") is not False):
        raise ValueError("checkpoint seed/exposure/update/input provenance differs from S3")
    design = _model_design(payload, schedule)
    identity = payload["resume_identity"]
    expected_identity_schema = "rachel-score-staged-training/1" if schedule == "staged" else JOINT_CHECKPOINT_SCHEMA
    if (identity.get("seed") != SEED or identity.get("train_count") != TRAIN_COUNT
            or identity.get("schema_version") != expected_identity_schema
            or identity.get("validation_count") != VAL_COUNT
            or identity.get("train_split") != "train" or identity.get("validation_split") != "val"
            or identity.get("score_design") != design["architecture"]
            or identity.get("held_out_used_for_training_or_selection") is not False
            or identity.get("microbatch") != 4 or identity.get("effective_batch") != 16
            or identity.get("precision") != "fp32"):
        raise ValueError("requires fixed Full24 TRAIN and cleanVAL3000 provenance")
    training = payload["training_data"]
    if training.get("kind") != "fixed_e1_materialized" or training.get("unique_count") != TRAIN_COUNT:
        raise ValueError("S3 dataset is not the fixed Full24 materialized TRAIN")
    expected_phase = asdict(phase(epoch, schedule=schedule, architecture=design["architecture"]))
    if schedule == "staged":
        if canonical(payload.get("phase")) != canonical(expected_phase):
            raise ValueError("staged checkpoint is not in the registered C phase")
        if not payload.get("pretraining_receipt"):
            raise ValueError("staged C checkpoint lacks its completed M12 receipt")
        if identity.get("max_epochs") != BUDGET:
            raise ValueError("staged budget must be exactly20 epochs")
    elif identity.get("training_mode") != "joint" or identity.get("max_epochs") != 50:
        raise ValueError("joint import must use the unchanged nested50 trajectory at epoch20")
    for name in ("train_manifest_sha256", "validation_manifest_sha256", "initial_weights_sha256",
                 "shared_base_initial_weights_sha256"):
        value = identity.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("missing frozen provenance hash: " + name)
    if canonical(payload["loss_config"]) != canonical(identity["loss_config"]):
        raise ValueError("checkpoint source loss differs from trajectory identity")
    common_fields = ("seed", "score_design", "train_count", "train_manifest_sha256",
                     "validation_count", "validation_manifest_sha256", "base_model_metadata",
                     "loss_config", "initial_weights_sha256", "shared_base_initial_weights_sha256",
                     "microbatch", "effective_batch", "optimizer", "weight_decay", "precision")
    common = {name: identity[name] for name in common_fields}
    common["candidate_config"] = design["candidate_config"]
    common["lr_by_epoch"] = identity["lr_by_epoch"][:BUDGET]
    if len(common["lr_by_epoch"]) != BUDGET:
        raise ValueError("learning-rate schedule lacks the complete20-epoch budget")
    if canonical(common["base_model_metadata"]) != canonical(dict(
            model_kind="full", model_config=asdict(full24_reference_config()), model_options={})):
        raise ValueError("reference model is not the unchanged common Full24 base")
    registered = stage_protocol(design["architecture"], schedule=schedule,
                                base=RachelN512LossConfig(**payload["loss_config"]))
    if schedule == "staged":
        if canonical(identity.get("phase_protocol")) != canonical(registered):
            raise ValueError("declared staged loss/freeze protocol differs from registered M12+C8")
        receipt = payload["pretraining_receipt"]
        expected_receipt = dict(schema_version=registered["schema_version"], architecture=design["architecture"],
            split="train", completed_phase="M", completed_epochs=12, pair_exposures=288000,
            optimizer_updates=18000, train_manifest_sha256=identity["train_manifest_sha256"],
            run_identity_sha256=digest(identity))
        if any(receipt.get(key) != value for key, value in expected_receipt.items()):
            raise ValueError("M12 receipt budget/data/run binding differs")
        state_digest = hashlib.sha256()
        for name, tensor in sorted(payload["model_state_dict"].items()):
            if not name.startswith("base_model.") or name.startswith(("base_model.fusion.", "base_model.local_head.")):
                continue
            value = tensor.detach().cpu().contiguous()
            state_digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
            state_digest.update(value.numpy().tobytes())
        if (receipt.get("trained_matcher_sha256") != state_digest.hexdigest()
                or receipt.get("initial_matcher_sha256") == receipt.get("trained_matcher_sha256")):
            raise ValueError("C checkpoint matcher differs from the frozen trained M12 state")
    return canonical(dict(schedule=schedule, architecture=design["architecture"], seed=SEED,
        common_training_contract=common, common_training_contract_sha256=digest(common),
        trajectory_identity_sha256=digest(identity), phase=expected_phase, phase_protocol=registered,
        model_design=design))


def validation_report(rows):
    """Recalibrate from only a checkpoint's saved, complete clean-VAL rows."""
    if len(rows) != VAL_COUNT or len({r["pair_id"] for r in rows}) != VAL_COUNT:
        raise ValueError("requires3000 unique saved VAL pair rows")
    if any(type(r["label"]) is not bool or type(r["decision_valid"]) is not bool for r in rows):
        raise ValueError("VAL labels/decision validity must be explicit booleans")
    labels = np.array([r["label"] for r in rows], bool)
    if int(labels.sum()) != 1500:
        raise ValueError("requires the unchanged balanced cleanVAL3000 population")
    thresholds, methods = {}, {}
    for branch in BRANCHES:
        scores = np.asarray([r["classification"][branch] for r in rows], float)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("VAL branch scores must be finite probabilities")
        thresholds[branch] = float(fit_threshold(labels, scores))
        methods[branch] = classification(labels, scores, thresholds[branch])
    report = dict(sample_count=VAL_COUNT, positive_count=1500, negative_count=1500,
        thresholds=thresholds, methods=methods,
        decision_coverage=float(np.mean([r["decision_valid"] for r in rows])),
        selection_key=[methods["fused"]["f1"], methods["fused"]["auprc"]],
        threshold_grain="equal pair rows; no cluster reweighting", pose_used_for_selection=False)
    points = fit_operating_points(labels, [r["classification"]["fused"] for r in rows])
    return canonical(report), canonical(points)


def read_epoch_record(root, epoch, *, schedule):
    root = Path(root).resolve(strict=True)
    checkpoint = root / ("epoch_%03d.pt" % epoch)
    rows_path = root / ("validation_%03d_rows.json" % epoch)
    validation_path = root / ("validation_%03d.json" % epoch)
    payload = load_training_epoch(root, epoch)
    contract = checkpoint_contract(payload, schedule=schedule, epoch=epoch)
    saved = json.loads(validation_path.read_text())
    if saved.get("epoch") != epoch or saved.get("global_exposure") != epoch * TRAIN_COUNT:
        raise ValueError("saved validation is not attached to the requested epoch")
    if schedule == "staged" and (saved.get("selection_eligible") is not True
                                 or canonical(saved.get("phase")) != contract["phase"]):
        raise ValueError("staged VAL rows were not emitted in the eligible C phase")
    rows = json.loads(rows_path.read_text())
    report, points = validation_report(rows)
    # The complete identical pair-ID/label order must hold throughout13..20.
    population_digest = digest([(r["pair_id"], r["label"]) for r in rows])
    return canonical(dict(epoch=epoch, global_exposure=epoch * TRAIN_COUNT,
        checkpoint=str(checkpoint), checkpoint_sha256=sealed._sha256_file(checkpoint),
        validation_predictions=str(rows_path), validation_predictions_sha256=sealed._sha256_file(rows_path),
        validation_record=str(validation_path), validation_record_sha256=sealed._sha256_file(validation_path),
        validation_population_sha256=population_digest, contract=contract,
        classifier_thresholds=report["thresholds"], operating_points=points, validation=report))


def select_records(records):
    if sorted(r["epoch"] for r in records) != list(ELIGIBLE):
        raise ValueError("requires every checkpoint13..20, not an old selected winner")
    ordered = sorted(records, key=lambda r: r["epoch"])
    if (len({r["contract"]["trajectory_identity_sha256"] for r in ordered}) != 1
            or len({r["validation_population_sha256"] for r in ordered}) != 1):
        raise ValueError("epoch13..20 trajectory/VAL population is inconsistent")
    eligible = [r for r in ordered if r["validation"]["decision_coverage"] == 1.0]
    if not eligible:
        raise ValueError("no eligible fully covered auxiliary VAL checkpoint")
    chosen = {"fixed_epoch": ordered[-1],
              "max_f1": max(eligible, key=lambda r: (tuple(r["validation"]["selection_key"]), -r["epoch"])),
              "recall95": max(eligible, key=lambda r: (tuple(r["operating_points"]["selection_key"]), -r["epoch"]))}
    result = {}
    for selection, row in chosen.items():
        result[selection] = dict(row, selection=selection, selected_epoch=row["epoch"],
            selected_global_exposure=row["global_exposure"],
            checkpoint_role="paired_epoch20_anchor_not_a_VAL_winner" if selection == "fixed_epoch" else "VAL_winner13..20",
            eligible_epoch_range=[20, 20] if selection == "fixed_epoch" else [13, 20],
            thresholds_fitted_on="cleanVAL3000 only; selected checkpoint's own rows",
            test_or_real_or_ood_used_for_fit=False)
    return canonical(result)


def publish_s3_freeze(root, *, schedule="staged", expected_identity=None):
    """Called only after committed epoch20 and all eligible saved VAL rows."""
    root = Path(root).resolve(strict=True)
    # epoch020 is written just before last.pt. Refuse a crash-window snapshot
    # whose epoch file exists but whose optimizer/RNG commit has not completed.
    last_path = (root / "last.pt").resolve(strict=True)
    if last_path.parent != root or last_path.name != "last.pt":
        raise ValueError("last.pt must be the run-owned recovery commit")
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    completed = last.get("completed_segments", -1)
    if (type(completed) is not int or completed < 80
            or schedule == "staged" and completed != 80
            or schedule == "joint" and completed > 200
            or last.get("global_exposure") != completed * 6000
            or last.get("optimizer_updates") != completed * 6000 // 16
            or "optimizer_state_dict" not in last or "rng_state" not in last):
        raise ValueError("epoch20 optimizer/RNG recovery has not been committed")
    records = [read_epoch_record(root, epoch, schedule=schedule) for epoch in ELIGIBLE]
    if any(r["contract"]["trajectory_identity_sha256"] != digest(last["resume_identity"]) for r in records):
        raise ValueError("epoch13..20 do not match the committed recovery trajectory")
    if expected_identity is not None and any(r["contract"]["trajectory_identity_sha256"] != digest(expected_identity) for r in records):
        raise ValueError("saved checkpoints differ from the caller's final trajectory identity")
    selected = select_records(records)
    freeze = canonical(dict(schema_version=SCHEMA, status="frozen_s3_epoch20", schedule=schedule,
        training_run=str(root), architecture=records[-1]["contract"]["architecture"], seed=SEED,
        budget_epochs=20, budget_exposures=480000, budget_optimizer_updates=30000,
        eligible_epoch_range=[13, 20], primary_selection="fixed_epoch", primary_epoch=20,
        selection_population="cleanVAL3000 only", held_out_used_for_fit=False,
        live_s0_s2_winner_imported=False, phase_protocol=records[-1]["contract"]["phase_protocol"],
        common_training_contract=records[-1]["contract"]["common_training_contract"],
        common_training_contract_sha256=records[-1]["contract"]["common_training_contract_sha256"],
        validation_population_sha256=records[-1]["validation_population_sha256"],
        source_epochs=[{k: r[k] for k in ("epoch", "checkpoint", "checkpoint_sha256",
                       "validation_predictions", "validation_predictions_sha256")} for r in records],
        selections=selected))
    write_new_json(root / "s3_freezes" / "freeze.json", freeze)
    return freeze


def validate_s3_freeze(freeze):
    if (freeze.get("schema_version") != SCHEMA or freeze.get("status") != "frozen_s3_epoch20"
            or freeze.get("schedule") not in ("staged", "joint") or freeze.get("seed") != SEED
            or freeze.get("budget_epochs") != 20 or freeze.get("budget_exposures") != 480000
            or freeze.get("budget_optimizer_updates") != 30000
            or freeze.get("eligible_epoch_range") != [13, 20]
            or freeze.get("primary_selection") != "fixed_epoch" or freeze.get("primary_epoch") != 20
            or freeze.get("selection_population") != "cleanVAL3000 only"
            or freeze.get("held_out_used_for_fit") is not False
            or freeze.get("live_s0_s2_winner_imported") is not False
            or set(freeze.get("selections", {})) != set(SELECTIONS)):
        raise ValueError("requires a dedicated S3 epoch20/[13,20] freeze; old winners are forbidden")
    if [record.get("epoch") for record in freeze.get("source_epochs", [])] != list(ELIGIBLE):
        raise ValueError("S3 freeze must preserve all eight source epochs13..20")
    if digest(freeze["common_training_contract"]) != freeze.get("common_training_contract_sha256"):
        raise ValueError("S3 common training contract digest differs")
    for selection, record in freeze["selections"].items():
        epoch = record.get("selected_epoch")
        interval = [20, 20] if selection == "fixed_epoch" else [13, 20]
        epoch_allowed = epoch == 20 if selection == "fixed_epoch" else epoch in ELIGIBLE
        if (record.get("selection") != selection or record.get("eligible_epoch_range") != interval
                or not epoch_allowed):
            raise ValueError("S3 selected epoch/window differs from registered rule")
        if record.get("test_or_real_or_ood_used_for_fit") is not False:
            raise ValueError("S3 selection must be held-out blind")
        if record.get("selected_global_exposure") != epoch * TRAIN_COUNT:
            raise ValueError("S3 selected checkpoint exposure differs")
        thresholds = list(record["classifier_thresholds"].values()) + list(record["operating_points"]["thresholds"].values())
        if set(record["classifier_thresholds"]) != set(BRANCHES) or not {"max_f1", "recall_95"} <= set(record["operating_points"]["thresholds"]):
            raise ValueError("S3 freeze lacks registered branch thresholds")
        if any(not np.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
            raise ValueError("S3 thresholds must be finite probabilities")
    return freeze


def publish_s3_comparison(joint_run, staged_run, output):
    joint = validate_s3_freeze(publish_s3_freeze(joint_run, schedule="joint"))
    staged = validate_s3_freeze(publish_s3_freeze(staged_run, schedule="staged"))
    if (joint["common_training_contract"] != staged["common_training_contract"]
            or joint["validation_population_sha256"] != staged["validation_population_sha256"]):
        raise ValueError("joint/staged Full24, seed, architecture, initialization, LR or VAL differ")
    comparison = dict(schema_version=SCHEMA, status="paired_frozen", budget_epochs=20,
        joint_freeze=str(Path(joint_run).resolve() / "s3_freezes/freeze.json"),
        staged_freeze=str(Path(staged_run).resolve() / "s3_freezes/freeze.json"),
        common_training_contract_sha256=joint["common_training_contract_sha256"],
        differences={"joint": joint["phase_protocol"], "staged": staged["phase_protocol"]},
        primary="fixed_epoch20", auxiliary_window=[13, 20],
        interpretation="equal480K total exposures; different phase losses/freeze and per-module update budgets",
        held_out_used_for_fit=False)
    write_new_json(output, comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-run", required=True)
    parser.add_argument("--staged-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    publish_s3_comparison(args.joint_run, args.staged_run, args.output)


if __name__ == "__main__":
    main()
