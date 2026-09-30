"""Prepare exactly one post-S3 input experiment; never launch a process.

Caller chooses one architecture, one changed input axis and one finite budget.
Completed default-input live runs are reused, never retrained. S3 completion is
a prerequisite, not a source of TEST/REAL/OOD-driven architecture selection.
Only JSON metadata and selected checkpoint metadata (map_location='meta') are
read. No network is constructed, no NPZ is opened and no predictions rescored.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

import torch

from experiments.rachel_n512_formal_30k.prepare_score_budget_queue import (
    ARMS, BUDGETS, SELECTIONS, SPLITS, PYTHON, DATASET, MATERIALIZED, METADATA_CHECKPOINT,
    evaluation_path as baseline_evaluation_path)
from experiments.rachel_n512_formal_30k.score_design_input_variants import InputVariantSpec, full24_reference_config
from experiments.rachel_n512_formal_30k.train_score_input_variant import input_spec
from experiments.rachel_n512_formal_30k.train_score_design import learning_rate
from experiments.rachel_n512_formal_30k.train_edge_weathering import _sha256
from experiments.rachel_n512_formal_30k.evaluate_score_input_variant import (
    comparison_training_contract, validate_paired_baseline, digest, core)
from experiments.rachel_n512_formal_30k.freeze_score_staged_comparison import (
    validate_s3_freeze, write_new_json, SCHEMA as S3_SCHEMA)
from staging.pairwise_v0_2.models.rachel_candidate_score import CandidateScoreConfig

SCHEMA = "rachel-score-input-queue-preparation/1"
COUNTS = dict(test=3000, real=1016, ood=301)


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def locations(root, architecture, budget, spec, run_name):
    root = Path(root)
    if not root.is_absolute() or architecture not in ARMS or budget not in BUDGETS:
        raise ValueError("absolute root, registered architecture and finite budget required")
    if not isinstance(spec, InputVariantSpec) or len(spec.changed_axes()) != 1:
        raise ValueError("exactly one nondefault input axis is required; baseline is reused")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("run-name must be a simple unique name, not a path")
    run = root / "input_variants" / run_name
    return dict(root=root, run=run, training=run / "training", smoke=run / "smoke32",
        queue=root / "queues" / ("input_" + run_name), baseline=root / architecture / "training")


def build_config(root, architecture, budget, spec, run_name, *, python=PYTHON,
                 dataset=DATASET, materialized=MATERIALIZED, metadata_checkpoint=METADATA_CHECKPOINT):
    paths = locations(root, architecture, budget, spec, run_name)
    options = ["--coarse-size", str(spec.coarse_size), "--transport-fusion", spec.transport_fusion]
    if len(spec.window_sizes_px) == 1:
        options += ["--single-window", str(int(spec.window_sizes_px[0]))]
    train = [python, "-m", "experiments.rachel_n512_formal_30k.train_score_input_variant",
        "--checkpoint", str(metadata_checkpoint), "--dataset", str(dataset),
        "--train-materialized-manifest", str(materialized), "--architecture", architecture,
        "--workers", "4", "--stop-after-epoch", str(budget)] + options
    stages = [dict(name="input_smoke32", command=train + ["--output", str(paths["smoke"]), "--smoke", "32"],
        marker=str(paths["smoke"]), completion_path=str(paths["smoke"] / "smoke.json"),
        completion_statuses=["smoke_complete"]),
        dict(name="input_random_train_%03d" % budget, command=train + ["--output", str(paths["training"])],
        marker=str(paths["training"]), completion_path=str(paths["training"] / "status.json"),
        completion_statuses=["complete" if budget == 50 else "budget_complete"], resume_arguments=["--resume"])]
    for selection in SELECTIONS:
        for split in SPLITS:
            output = paths["run"] / "evaluation" / ("budget_%03d" % budget) / selection / split
            command = [python, "-m", "experiments.rachel_n512_formal_30k.evaluate_score_input_variant",
                "--training-run", str(paths["training"]), "--budget", str(budget), "--selection", selection,
                "--split", split, "--dataset", str(dataset), "--output", str(output),
                "--batch-size", "4", "--workers", "4", "--baseline-evaluation",
                str(baseline_evaluation_path(paths["root"], architecture, budget, selection, split))]
            if split == "real":
                command += ["--keep-ids", str(paths["root"] / "keep_ids.json")]
            stages.append(dict(name="input_%03d_%s_%s" % (budget, selection, split), command=command,
                marker=str(output), completion_path=str(output / "protocol.json"), completion_statuses=["complete"]))
    return dict(root=str(paths["queue"]), source=str(paths["root"] / "input_source"), dependencies=[], stages=stages)


def owned(root, value, name):
    root = Path(root).resolve(strict=True)
    path = Path(value)
    path = path if path.is_absolute() else root / path
    resolved = path.resolve(strict=True)
    if resolved != root / name:
        raise ValueError("evidence must name the exact owned file: " + name)
    return resolved


def verify_evaluation(output, *, schema, split, expected_model, keep_sha):
    """Check completion/identity only; never inspect metric values or rows."""
    output = Path(output)
    protocol = read(output / "protocol.json")
    if (protocol.get("schema_version") != schema or protocol.get("status") != "complete"
            or protocol.get("split") != split or protocol.get("sample_count") != COUNTS[split]
            or protocol.get("decoder") != core.fixed.DECODER_NAME
            or protocol.get("decoder_config") != canonical(asdict(core.fixed.TOP2_CONFIG))
            or protocol.get("thresholds_fitted") is not False
            or protocol.get("test_or_real_used_for_fit") is not False
            or protocol.get("ood_used_for_fit") is not False):
        raise ValueError(str(output) + ": incomplete or non-frozen evaluation")
    model = protocol.get("model", {})
    if any(canonical(model.get(k)) != canonical(v) for k, v in expected_model.items()):
        raise ValueError(str(output) + ": evaluation model differs from frozen metadata")
    if split == "real" and protocol.get("keep_ids_sha256") != keep_sha:
        raise ValueError(str(output) + ": reviewed REAL population changed")
    if read(output / "summary.json").get("status") != "complete":
        raise ValueError(str(output) + ": summary has not completed")
    if not (output / "pair_results.jsonl").is_file() or (output / "pair_results.jsonl").stat().st_size == 0:
        raise ValueError(str(output) + ": missing saved predictions")
    return dict(path=str(output), protocol_sha256=_sha256(output / "protocol.json"), sample_count=COUNTS[split])


def verify_s3(root, comparison_path, architecture, keep_sha):
    path = Path(comparison_path).resolve(strict=True)
    comparison = read(path)
    if (comparison.get("schema_version") != S3_SCHEMA or comparison.get("status") != "paired_frozen"
            or comparison.get("budget_epochs") != 20 or comparison.get("primary") != "fixed_epoch20"
            or comparison.get("auxiliary_window") != [13, 20] or comparison.get("held_out_used_for_fit") is not False):
        raise ValueError("explicit complete S3 paired-comparison evidence is required")
    freezes, records = {}, []
    for schedule in ("joint", "staged"):
        freeze_path = Path(comparison[schedule + "_freeze"]).resolve(strict=True)
        freeze = validate_s3_freeze(read(freeze_path))
        run = Path(freeze["training_run"]).resolve(strict=True)
        s3_arch = freeze["architecture"]
        expected_run = root / s3_arch / "training" if schedule == "joint" else root / "s3" / s3_arch / "training"
        if (s3_arch not in ("candidate_pair", "candidate_dual") or run != expected_run
                or freeze_path != run / "s3_freezes/freeze.json" or freeze["schedule"] != schedule
                or architecture != "original" and s3_arch != architecture
                or freeze["common_training_contract_sha256"] != comparison["common_training_contract_sha256"]):
            raise ValueError("S3 evidence belongs to another architecture, schedule or experiment")
        for selection, winner in freeze["selections"].items():
            for split in SPLITS:
                output = run.parent / "evaluation/s3_epoch020" / selection / split
                expected = dict(training_run=str(run), budget=20, selection=selection, schedule=schedule,
                    architecture=s3_arch, seed=260913, epoch=winner["selected_epoch"],
                    checkpoint_sha256=winner["checkpoint_sha256"], freeze_sha256=_sha256(freeze_path),
                    classifier_thresholds=winner["classifier_thresholds"], operating_points=winner["operating_points"])
                records.append(verify_evaluation(output, schema="rachel-score-staged-evaluation/1",
                    split=split, expected_model=expected, keep_sha=keep_sha))
        freezes[schedule] = freeze
    if (freezes["joint"]["common_training_contract"] != freezes["staged"]["common_training_contract"]
            or freezes["joint"]["validation_population_sha256"] != freezes["staged"]["validation_population_sha256"]):
        raise ValueError("S3 paired runs have unequal training/validation contracts")
    return dict(comparison=str(path), comparison_sha256=_sha256(path), evaluations=records,
        architecture=freezes["joint"]["architecture"],
        role="stage-order evidence only; no original staged conclusion" if architecture == "original" else "same-architecture S3 prerequisite")


def verify_baseline(root, architecture, budget, spec, keep_sha):
    run = root / architecture / "training"
    freeze_path = run / "budget_freezes" / ("%03d" % budget) / "freeze.json"
    freeze = read(freeze_path)
    if (freeze.get("schema_version") != "rachel-score-design-training/1"
            or freeze.get("status") != "frozen_at_budget" or freeze.get("budget_epochs") != budget
            or freeze.get("budget_exposures") != budget * 24000 or freeze.get("eligible_epoch_range") != [5, budget]
            or freeze.get("selection_population") != "cleanVAL3000 only"
            or freeze.get("held_out_used_for_fit") is not False or set(freeze.get("winners", {})) != set(SELECTIONS)):
        raise ValueError("default live baseline lacks the requested completed SIM-VAL budget freeze")
    owned(run, "epoch_%03d.pt" % budget, "epoch_%03d.pt" % budget)
    identities, records, checked = [], [], {}
    for selection, winner in freeze["winners"].items():
        epoch = winner["selected_epoch"]
        if (not 5 <= epoch <= budget or winner.get("selection") != selection
                or winner.get("selected_global_exposure") != epoch * 24000
                or winner.get("test_or_real_or_ood_used_for_fit") is not False):
            raise ValueError("baseline winner is outside its SIM-VAL budget")
        checkpoint = owned(run, winner["checkpoint"], "epoch_%03d.pt" % epoch)
        if checkpoint not in checked:
            if _sha256(checkpoint) != winner["checkpoint_sha256"]:
                raise ValueError("baseline checkpoint changed since its freeze")
            # Trusted run-owned file: tensor results remain on meta. No network
            # is instantiated or trained parameter set retained for inference.
            saved = torch.load(checkpoint, map_location="meta", weights_only=False)
            identity = saved["resume_identity"]
            if (digest(identity) != freeze["resume_identity_sha256"]
                    or saved.get("score_design_schema") != "rachel-score-design-training/1"
                    or saved.get("source_weights_loaded") is not False or saved.get("formal_training_counted") is not True
                    or saved.get("epoch") != epoch or saved.get("completed_segments") != epoch * 4
                    or saved.get("global_exposure") != epoch * 24000
                    or saved.get("optimizer_updates") != epoch * 1500
                    or identity.get("schema_version") != "rachel-score-design-training/1"
                    or "optimizer_state_dict" not in saved or "rng_state" not in saved
                    or saved.get("resample_contour_cap") is not None
                    or canonical(saved["model_config"]) != canonical(asdict(full24_reference_config()))):
                raise ValueError("baseline checkpoint is not the original default-input joint trajectory")
            checked[checkpoint] = (identity, winner["checkpoint_sha256"])
            del saved
        identity, checkpoint_sha = checked[checkpoint]
        if winner["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("baseline selections disagree about the same checkpoint")
        contract = comparison_training_contract(identity)
        if contract["score_design"] != architecture or contract["max_epochs"] != 50:
            raise ValueError("baseline architecture/schedule differs")
        baseline = dict(budget=budget, selection=selection, seed=260913, architecture=architecture,
            comparison_training_contract=contract, input_spec=canonical(asdict(InputVariantSpec())))
        intended = dict(baseline, input_spec=canonical(asdict(spec)))
        validate_paired_baseline(intended, baseline)
        val = owned(run, winner["validation_predictions"], "validation_%03d_rows.json" % epoch)
        if val.stat().st_size == 0:
            raise ValueError("missing baseline validation predictions")
        expected = dict(training_run=str(run), budget=budget, selection=selection, architecture=architecture,
            seed=260913, epoch=epoch, checkpoint_sha256=winner["checkpoint_sha256"],
            freeze_sha256=_sha256(freeze_path), model_config=canonical(asdict(full24_reference_config())),
            classifier_thresholds=winner["classifier_thresholds"], operating_points=winner["operating_points"])
        for split in SPLITS:
            records.append(verify_evaluation(baseline_evaluation_path(root, architecture, budget, selection, split),
                schema="rachel-score-design-evaluation/1", split=split, expected_model=expected, keep_sha=keep_sha))
        identities.append(identity)
    if identities[0] != identities[1]:
        raise ValueError("baseline winners came from different training trajectories")
    return dict(training_run=str(run), freeze=str(freeze_path), freeze_sha256=_sha256(freeze_path),
        comparison_training_contract=comparison_training_contract(identities[0]), evaluations=records,
        baseline_retrained=False), identities[0]


def verify_training_inputs(identity, architecture, *, dataset, materialized, metadata_checkpoint):
    if (_sha256(materialized) != identity["train_manifest_sha256"]
            or _sha256(Path(dataset) / "pairs/val.jsonl") != identity["validation_manifest_sha256"]):
        raise ValueError("planned TRAIN/VAL manifests differ from the default baseline")
    source = torch.load(metadata_checkpoint, map_location="meta", weights_only=False)
    if (canonical(source.get("model_config")) != canonical(asdict(full24_reference_config()))
            or source.get("seam_loss_enabled", False) or source.get("loss_config") != identity["loss_config"]):
        raise ValueError("planned metadata source changes reference architecture/loss")
    expected = dict(seed=260913, train_count=24000, validation_count=3000, max_epochs=50,
        microbatch=4, effective_batch=16, segment_pairs=6000, optimizer="AdamW", weight_decay=1e-4,
        grad_clip_norm=5., precision="fp32", lr_by_epoch=[learning_rate(e) for e in range(1, 51)],
        training_mode="joint", min_selection_epoch=5, validation_every_epochs=1,
        candidate_correctness_weight=.5 if architecture == "candidate_dual" else 0.,
        candidate_correctness_tolerance_px=20.,
        candidate_config=asdict(CandidateScoreConfig()) if architecture != "original" else {})
    if any(canonical(identity.get(k)) != canonical(v) for k, v in expected.items()):
        raise ValueError("baseline does not share the registered input-trainer update contract")
    return dict(dataset=str(dataset), materialized=str(materialized), metadata_checkpoint=str(metadata_checkpoint),
        source_weights_loaded=False, metadata_checkpoint_sha256=_sha256(metadata_checkpoint))


def prepare(root, architecture, budget, spec, run_name, *, s3_comparison, python=PYTHON,
            dataset=DATASET, materialized=MATERIALIZED, metadata_checkpoint=METADATA_CHECKPOINT):
    root = Path(root).resolve(strict=True)
    paths = locations(root, architecture, budget, spec, run_name)
    if paths["queue"].exists() or paths["run"].exists():
        raise FileExistsError("input run/queue name already exists; no extension or duplicate preparation")
    if not (root / "input_source").is_dir():
        raise ValueError("existing independent input_source required; live source is not used")
    kept = read(root / "keep_ids.json")["kept_positive_pair_ids"]
    if len(kept) != 295 or len(set(kept)) != 295:
        raise ValueError("requires the unchanged295-positive review export")
    keep_sha = _sha256(root / "keep_ids.json")
    s3 = verify_s3(root, s3_comparison, architecture, keep_sha)
    baseline, identity = verify_baseline(root, architecture, budget, spec, keep_sha)
    inputs = verify_training_inputs(identity, architecture, dataset=dataset,
        materialized=materialized, metadata_checkpoint=metadata_checkpoint)
    config = build_config(root, architecture, budget, spec, run_name, python=python,
        dataset=dataset, materialized=materialized, metadata_checkpoint=metadata_checkpoint)
    paths["run"].mkdir(parents=True, exist_ok=False)
    paths["queue"].mkdir(parents=True, exist_ok=False)
    preparation = dict(schema_version=SCHEMA, status="prepared_not_launched", architecture=architecture,
        budget=budget, input_spec=canonical(asdict(spec)), changed_axis=spec.changed_axes()[0],
        s3_prerequisite=s3, baseline=baseline, training_inputs=inputs, run_name=run_name,
        smoke_pairs=32, formal_pair_exposures=budget * 24000, formal_optimizer_updates=budget * 1500,
        evaluation_count=6, only_requested_budget_evaluated=True, default_baseline_retrained=False,
        architecture_budget_choice="caller-specified before preparation; no held-out metric values inspected",
        held_out_used_for_fit=False, held_out_metrics_used_to_choose_next_action=False,
        checkpoint_inspection="selected owned metadata on meta device; no model constructed",
        launches_processes=False)
    write_new_json(paths["run"] / "preparation.json", preparation)
    write_new_json(paths["queue"] / "preparation.json", preparation)
    write_new_json(paths["queue"] / "config.json", config)
    return dict(status="prepared_not_launched", config=str(paths["queue"] / "config.json"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--architecture", required=True, choices=ARMS)
    p.add_argument("--budget", required=True, type=int, choices=BUDGETS)
    p.add_argument("--run-name", required=True)
    p.add_argument("--s3-comparison", required=True)
    p.add_argument("--coarse-size", type=int, choices=(128, 256, 512), default=128)
    p.add_argument("--single-window", type=int, choices=(7, 16, 32, 64))
    p.add_argument("--transport-fusion", choices=("early", "post"), default="early")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--materialized", default=MATERIALIZED)
    p.add_argument("--metadata-checkpoint", default=METADATA_CHECKPOINT)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(prepare(args.root, args.architecture, args.budget, input_spec(args), args.run_name,
        s3_comparison=args.s3_comparison, dataset=args.dataset, materialized=args.materialized,
        metadata_checkpoint=args.metadata_checkpoint)))
