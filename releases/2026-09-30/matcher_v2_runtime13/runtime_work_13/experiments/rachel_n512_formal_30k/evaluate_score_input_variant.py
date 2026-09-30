"""Evaluate a typed single-input-axis model using SIM-VAL-only budget freezes.

The live score trainer/evaluator and their checkpoint schemas stay unchanged.
N512, joint training, physical window/grid metadata and the canonical Top2
decoder are fixed by the registered input-model factory. P/R and three branch
scores are saved before held-out layout GT is attached. No held-out threshold
selection, fallback or layout correction is performed here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import evaluate_score_design as core
from experiments.rachel_n512_formal_30k.score_design_checkpoint_io import load_owned_epoch_checkpoint
from experiments.rachel_n512_formal_30k.score_design_input_variants import InputVariantSpec, full24_reference_config
# This is only a pure clean-VAL row validator/calibrator, not S3 selection or
# its checkpoint loader. Recompute frozen thresholds as an integrity check.
from experiments.rachel_n512_formal_30k.freeze_score_staged_comparison import validation_report

SCHEMA = "rachel-score-input-evaluation/1"
TRAINING_SCHEMA = "rachel-score-input-training/1"
CHECKPOINT_SCHEMA = "rachel-score-input-checkpoint/1"
BUDGETS, SELECTIONS = (5, 10, 20, 30, 50), ("max_f1", "recall95")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _owned_path(root, value, expected_name):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if path.resolve(strict=True) != root / expected_name:
        raise ValueError("freeze must reference the exact run-owned " + expected_name)
    return path.resolve(strict=True)


def comparison_training_contract(identity):
    """Trajectory fields shared with existing live S0-S2 default-input runs.

Do not compare full initial-state digests: physical sampling buffers differ by
design. Compare the common random Full24 base digest, architecture/head and all
loss/data/update controls. Input-specific metadata is checked separately.
"""
    fields = ("seed", "score_design", "training_mode", "train_manifest_sha256",
        "validation_manifest_sha256", "train_count", "train_split", "validation_count",
        "validation_split", "loss_config", "shared_base_initial_weights_sha256",
        "candidate_config", "candidate_correctness_tolerance_px", "candidate_correctness_weight",
        "candidate_correctness_reduction", "microbatch", "effective_batch", "optimizer",
        "weight_decay", "lr_by_epoch", "grad_clip_norm", "precision", "max_epochs",
        "segment_pairs", "validation_every_epochs", "min_selection_epoch",
        "held_out_used_for_training_or_selection")
    missing = set(fields) - set(identity)
    if missing:
        raise ValueError("training identity lacks comparison fields: " + ", ".join(sorted(missing)))
    result = _json({key: identity[key] for key in fields})
    # Legacy original runs use{} while the typed model schema recordsnull.
    # Both explicitly mean no candidate head, not a tunable model difference.
    if result["score_design"] == "original" and result["candidate_config"] is None:
        result["candidate_config"] = {}
    if (result["training_mode"] != "joint" or result["seed"] != 260913
            or result["train_count"] != 24000 or result["train_split"] != "train"
            or result["validation_count"] != 3000 or result["validation_split"] != "val"
            or result["held_out_used_for_training_or_selection"] is not False):
        raise ValueError("input variants require shared Full24/joint/cleanVAL3000 training")
    return result


def load_frozen_input_model(training_run, budget, selection):
    if budget not in BUDGETS or selection not in SELECTIONS:
        raise ValueError("unregistered budget or selection")
    root = Path(training_run).resolve(strict=True)
    path = root / "budget_freezes" / ("%03d" % budget) / "freeze.json"
    freeze = json.loads(path.read_text())
    if (freeze.get("schema_version") != TRAINING_SCHEMA
            or freeze.get("status") != "frozen_at_budget"
            or freeze.get("selection_population") != "cleanVAL3000 only"
            or freeze.get("held_out_used_for_fit") is not False
            or freeze.get("budget_epochs") != budget
            or freeze.get("budget_exposures") != budget * 24000
            or freeze.get("eligible_epoch_range") != [5, budget]):
        raise ValueError("requires a complete new-schema SIM-VAL-only budget freeze")
    if set(freeze.get("winners", {})) != set(SELECTIONS):
        raise ValueError("freeze must contain both registered selections")
    selected = freeze["winners"][selection]
    epoch = selected.get("selected_epoch")
    if (type(epoch) is not int or not 5 <= epoch <= budget
            or selected.get("selection") != selection
            or selected.get("selected_global_exposure") != epoch * 24000
            or selected.get("test_or_real_or_ood_used_for_fit") is not False):
        raise ValueError("selected checkpoint is outside the SIM-VAL budget contract")
    checkpoint_path = _owned_path(root, selected["checkpoint"], "epoch_%03d.pt" % epoch)
    payload = load_owned_epoch_checkpoint(root, checkpoint_path, epoch, selected["checkpoint_sha256"])
    metadata = payload.get("score_input_metadata", {})
    resume = payload.get("resume_identity", {})
    if (payload.get("score_input_checkpoint_schema") != CHECKPOINT_SCHEMA
            or payload.get("score_input_training_schema") != TRAINING_SCHEMA
            or resume.get("schema_version") != TRAINING_SCHEMA
            or "score_design_schema" in payload or "s3_checkpoint_schema" in payload
            or payload.get("model_kind") != "score_input_" + str(metadata.get("architecture"))
            or payload.get("global_exposure") != epoch * 24000
            or payload.get("optimizer_updates") != epoch * 1500
            or payload.get("completed_segments") != epoch * 4
            or payload.get("resample_contour_cap") is not None
            or payload.get("source_weights_loaded") is not False
            or payload.get("formal_training_counted") is not True
            or metadata.get("training_mode") != "joint"
            or payload.get("seed") != metadata.get("seed")
            or resume.get("seed") != metadata.get("seed")):
        raise ValueError("checkpoint is not a complete typed joint input-variant epoch")
    if (freeze.get("resume_identity_sha256") != digest(resume)
            or freeze.get("resume_identity") != resume
            or freeze.get("score_input_metadata") != metadata):
        raise ValueError("frozen input model or training identity differs from checkpoint")
    variant = metadata["input_variant"]
    if (freeze.get("input_spec") != variant["spec"]
            or freeze.get("changed_axis") != variant["changed_axis"]):
        raise ValueError("frozen input spec/axis differs from checkpoint")
    comparison = comparison_training_contract(resume)
    if (resume["score_design"] != metadata["architecture"]
            or resume["loss_config"] != payload.get("loss_config")):
        raise ValueError("checkpoint architecture/loss differs from training identity")
    rows_path = _owned_path(root, selected["validation_predictions"], "validation_%03d_rows.json" % epoch)
    if core.sealed._sha256_file(rows_path) != selected.get("validation_predictions_sha256"):
        raise ValueError("frozen SIM-VAL calibration rows changed or lack SHA256")
    thresholds, operating = selected["classifier_thresholds"], selected["operating_points"]
    report_check, points_check = validation_report(json.loads(rows_path.read_text()))
    if (report_check["decision_coverage"] != 1.0 or thresholds != report_check["thresholds"]
            or operating != points_check):
        raise ValueError("frozen thresholds are not derived from the SHA-bound complete SIM-VAL rows")
    if set(thresholds) != set(core.BRANCHES) or not {"max_f1", "recall_95"} <= set(operating["thresholds"]):
        raise ValueError("freeze lacks branch and fused max-F1/recall95 thresholds")
    for value in list(thresholds.values()) + list(operating["thresholds"].values()):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("frozen thresholds must be finite probabilities")
    if (thresholds["fused"] != operating["thresholds"]["max_f1"]
            or selected.get("primary_pair_threshold") != operating["thresholds"]["max_f1" if selection == "max_f1" else "recall_95"]):
        raise ValueError("frozen primary/branch thresholds disagree")
    # The typed factory validates every early/post option and sampling grid,
    # then reconstructs the correct BASE before loading any wrapper weights.
    from experiments.rachel_n512_formal_30k.train_score_input_variant import load_score_input_checkpoint
    model = load_score_input_checkpoint(payload)
    if model.config.contour_cap != 512:
        raise ValueError("only original preparedN512 is registered")
    identity = dict(training_run=str(root), budget=budget, selection=selection,
        freeze_path=str(path), freeze_sha256=core.sealed._sha256_file(path),
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=selected["checkpoint_sha256"],
        epoch=epoch, seed=payload["seed"], architecture=metadata["architecture"],
        model_config=asdict(model.config), candidate_config=metadata["candidate_config"],
        score_input_metadata=metadata, input_spec=variant["spec"], changed_axis=variant["changed_axis"],
        comparison_training_contract=comparison, comparison_training_contract_sha256=digest(comparison),
        resample_contour_cap=None, classifier_thresholds=thresholds, operating_points=operating,
        winner_record=selected, test_or_real_used_for_fit=False, ood_used_for_fit=False)
    return model.eval().requires_grad_(False), identity


def validate_paired_baseline(identity, baseline_identity):
    """Single axis against default inputs, never another scoring head/stage."""
    for key in ("budget", "selection", "seed", "architecture", "comparison_training_contract"):
        if identity.get(key) != baseline_identity.get(key):
            raise ValueError("paired baseline differs in " + key)
    spec = InputVariantSpec(**identity["input_spec"])
    baseline_spec = InputVariantSpec(**baseline_identity["input_spec"])
    if baseline_spec.changed_axes() or len(spec.changed_axes()) != 1:
        raise ValueError("paired input comparison requires one changed axis versus the default baseline")
    return dict(changed_axis=spec.changed_axes()[0], same_budget=True, same_score_architecture=True,
        same_seed=True, same_data_loss_and_update_contract=True,
        default_baseline_spec=_json(asdict(baseline_spec)), variant_spec=_json(asdict(spec)),
        comparison_training_contract_sha256=digest(identity["comparison_training_contract"]))


def load_baseline_identity(baseline_evaluation, identity, split):
    root = Path(baseline_evaluation).resolve(strict=True)
    protocol = json.loads((root / "protocol.json").read_text())
    if (protocol.get("status") != "complete" or protocol.get("split") != split
            or protocol.get("decoder") != core.fixed.DECODER_NAME
            or protocol.get("decoder_config") != _json(asdict(core.fixed.TOP2_CONFIG))):
        raise ValueError("baseline must be a completed same-split canonical-layout evaluation")
    prior = protocol["model"]
    if protocol.get("schema_version") == SCHEMA:
        model, baseline = load_frozen_input_model(prior["training_run"], prior["budget"], prior["selection"])
        del model
    elif protocol.get("schema_version") == core.SCHEMA:
        # Reuse an already completed, default-input live joint run. This is
        # representation/schema reuse, not importing a staged winner.
        model, baseline = core.load_frozen_model(prior["training_run"], prior["budget"], prior["selection"])
        if (_json(asdict(model.config)) != _json(asdict(full24_reference_config()))
                or baseline.get("resample_contour_cap") is not None):
            raise ValueError("live baseline is not the default Full24 input configuration")
        payload = load_owned_epoch_checkpoint(baseline["training_run"], baseline["checkpoint_path"],
            baseline["epoch"], baseline["checkpoint_sha256"])
        baseline["comparison_training_contract"] = comparison_training_contract(payload["resume_identity"])
        baseline["input_spec"] = _json(asdict(InputVariantSpec()))
        del model, payload
    else:
        raise ValueError("baseline schema is neither typed input nor live joint S0-S2; staged is forbidden")
    for key in ("training_run", "budget", "selection", "checkpoint_sha256", "freeze_sha256",
                "epoch", "seed", "architecture", "classifier_thresholds", "operating_points"):
        if prior.get(key) != baseline.get(key):
            raise ValueError("baseline evaluation identity no longer matches its freeze: " + key)
    contract = validate_paired_baseline(identity, baseline)
    contract.update(source=str(root), baseline_checkpoint_sha256=baseline["checkpoint_sha256"],
        baseline_schema=protocol["schema_version"], baseline_retrained=False,
        selection_changed=False, fallback_applied=False)
    return protocol, contract


def compare_baseline(rows, baseline_evaluation, split, identity, current_protocol):
    previous, contract = load_baseline_identity(baseline_evaluation, identity, split)
    for field in ("manifest_sha256", "sample_count", "keep_ids_sha256"):
        if previous.get(field) != current_protocol.get(field):
            raise ValueError("baseline population/review export differs: " + field)
    # Core produces gained/lost IDs at10/20px without tuning a threshold or
    # applying baseline poses. OOD explicitly has no comparable pose truth.
    return dict(**core.compare_baseline(rows, baseline_evaluation, split), input_comparison_contract=contract)


def run(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers nonnegative")
    if args.split == "real" and (not args.keep_ids or not Path(args.keep_ids).is_file()):
        raise ValueError("REAL requires the existing keep-IDs export")
    torch.set_num_threads(1)
    model, identity = load_frozen_input_model(args.training_run, args.budget, args.selection)
    if args.baseline_evaluation:
        load_baseline_identity(args.baseline_evaluation, identity, args.split)
    core.sealed._set_determinism(identity["seed"])
    device = torch.device(args.device)
    model = model.to(device).eval().requires_grad_(False)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    targets, metadata, source_units = {}, None, None
    if args.split == "test":
        root = Path(args.dataset)
        manifest_path = root / "pairs/test.jsonl"
        manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
        if len(manifest) != 3000 or sum(r["label"] for r in manifest) != 1500:
            raise ValueError("requires complete balanced TEST3000")
        expected_ids = [r["pair_id"] for r in manifest]
        source_units = {r["pair_id"]: sorted({r["fragment_" + s]["split_unit_id"] for s in "ab"}) for r in manifest}
        dataset = core.RachelPairDataset(root, "test")
        batches = core.make_ablation_loader(dataset, tuple(range(len(dataset))), batch_size=args.batch_size,
            num_workers=args.workers, seed=identity["seed"], contour_cap=512)
        source = dict(dataset_root=str(root), manifest_sha256=core.sealed._sha256_file(manifest_path))
    else:
        cache = Path(args.prepared_cache if args.split == "real" else args.ood_prepared)
        if args.split == "real":
            metadata, arrays = core.load_prepared_cache(cache)
            if (len(metadata["pairs"]) != 1016 or sum(r["label"] for r in metadata["pairs"]) != 508
                    or sum(r["label"] and r["strict"] for r in metadata["pairs"]) != 508):
                raise ValueError("requires unchanged Dunhuang1016")
        else:
            metadata = json.loads((cache / "manifest.json").read_text())
            if (len(metadata["pairs"]) != 301 or len(metadata["fragment_ids"]) != 602
                    or metadata.get("layout_gt_provided") is not False
                    or metadata.get("negative_pairs_constructed") is not False):
                raise ValueError("requires original positive-only Turufan301")
            with np.load(cache / "inputs.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in ("packed_masks", "points", "valid")}
        expected_ids = [r["pair_id"] for r in metadata["pairs"]]
        batches = core.real.input_batches(metadata, arrays, args.batch_size)
        source = dict(prepared_cache=str(cache), manifest_sha256=core.sealed._sha256_file(cache / "manifest.json"))
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate evaluation pair ID")
    protocol = dict(schema_version=SCHEMA, status="running", split=args.split, model=identity, **source,
        sample_count=len(expected_ids), batch_size=args.batch_size, precision="fp32",
        decoder=core.fixed.DECODER_NAME, decoder_config=asdict(core.fixed.TOP2_CONFIG),
        decoder_design_unchanged=True, layout_predictions_may_change_with_trained_weights=True,
        resampling="original prepared512", model_input_fields=list(core.FIELDS),
        ground_truth_used_to_select_candidates=False, gt_attached_after_complete_prediction_freeze=True,
        thresholds_fitted=False, test_or_real_used_for_fit=False, ood_used_for_fit=False,
        overlap_definition="intersection/min(original areas), B placement=-t, nearest integer raster",
        invalid_decision_policy="reject; rank score below all valid probabilities",
        script_sha256=core.sealed._sha256_file(Path(__file__)))
    core.save(destination / "protocol.json", protocol)
    predictions, started = [], time.monotonic()
    try:
        with (destination / "pair_predictions.jsonl").open("x", encoding="utf-8") as stream:
            for batch in batches:
                batch_rows = core.predict_batch(model, batch, device)
                if args.split == "test":
                    for i, pair_id in enumerate(batch.pair_ids):
                        targets[pair_id] = dict(label=bool(batch.labels[i]),
                            translation_rc=core.common.clean(batch.translation_a_to_b_rc[i]) if batch.translation_valid[i] else None,
                            source_unit_ids=source_units[pair_id])
                predictions.extend(batch_rows)
                for row in batch_rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        if [r["pair_id"] for r in predictions] != expected_ids:
            raise ValueError("predictions do not match complete ordered evaluation population")
        core.save(destination / "prediction_complete.json", dict(status="all_predictions_frozen",
            sample_count=len(predictions), real_gt_opened=False, review_labels_opened=False,
            checkpoint_sha256=identity["checkpoint_sha256"]))
        if args.split == "real":
            rows = core.real.attach_ground_truth(predictions, metadata["pairs"], args.translation_gt_json)
            kept = set(json.loads(Path(args.keep_ids).read_text())["kept_positive_pair_ids"])
            if len(kept) != 295 or not kept <= {r["pair_id"] for r in rows if r["label"]}:
                raise ValueError("keep export differs from reviewed295-positive cohort")
            for row in rows:
                row["review_status"] = ("keep" if row["pair_id"] in kept else "exclude") if row["label"] else "not_reviewed_negative"
            protocol["keep_ids_sha256"] = core.sealed._sha256_file(Path(args.keep_ids))
        elif args.split == "test":
            rows = core.fixed.attach_test_targets(predictions, targets)
        else:
            rows = predictions
            for row in rows:
                row.update(label=True, target_translation_rc=None, layout_gt_available=False)
                row["layouts"][core.fixed.DECODER_NAME]["translation_l2_px"] = None
        core.fixed.write_rows(destination / "pair_results.jsonl", rows)
        summary = dict(status="complete", split=args.split, model=identity,
            groups=core.summarize_populations(rows, identity, args.split),
            selection_on_this_population=False, threshold_fitting_performed=False)
        if args.baseline_evaluation:
            summary["paired_baseline_layout"] = compare_baseline(rows, args.baseline_evaluation, args.split, identity, protocol)
        core.save(destination / "summary.json", summary)
        protocol.update(status="complete", elapsed_seconds=time.monotonic() - started)
        core.save(destination / "protocol.json", protocol)
        return summary
    except Exception as error:
        protocol.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - started)
        core.save(destination / "protocol.json", protocol)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run", required=True)
    p.add_argument("--budget", required=True, type=int, choices=BUDGETS)
    p.add_argument("--selection", required=True, choices=SELECTIONS)
    p.add_argument("--split", required=True, choices=("test", "real", "ood"))
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="/root/autodl-tmp/dataset_rachel_pairwise_n512_v1")
    p.add_argument("--prepared-cache", default=str(core.fixed.DEFAULT_PREPARED_CACHE))
    p.add_argument("--ood-prepared", default=str(core.DEFAULT_OOD))
    p.add_argument("--translation-gt-json", default=str(core.real.DEFAULT_TRANSLATION_GT))
    p.add_argument("--keep-ids")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--baseline-evaluation")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
