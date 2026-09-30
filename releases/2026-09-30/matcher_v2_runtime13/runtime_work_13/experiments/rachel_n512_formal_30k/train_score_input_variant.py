"""Independent, one-input-axis joint trainer; live and S3 trainers stay intact.

Example (remote Linux CUDA only):
  python -m experiments.rachel_n512_formal_30k.train_score_input_variant \
    --checkpoint /path/to/Full24.pt --dataset /path/to/release \
    --train-materialized-manifest /path/to/train_e1_24k.json \
    --architecture candidate_pair --coarse-size 256 --output /path/to/new_run \
    --stop-after-epoch 5
Resume with the same identity and --resume --stop-after-epoch 10 (then20/30/50).

The baseline is coarse128/four windows7,16,32,64/early fusion/N512. Select at
most one of coarse256/512, --single-window7/16/32/64, or --transport-fusion post.
Source checkpoint supplies only the unchanged Full24 loss/reference metadata.
Every build shares the reference random base AND candidate-head initialization;
actual base kind/options and regenerated sampling grids use the typed factory.
The old joint training segment, VAL evaluator, schedule and winner ordering are
reused directly. No staged/input cross experiment, new targets, or layout change.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.rachel_n512_formal_30k.score_design_input_variants import (
    FOUR_WINDOWS, InputVariantSpec, full24_reference_config)
from experiments.rachel_n512_formal_30k.score_design_variant_model import (
    build_score_input_model, restore_score_input_model)
from experiments.rachel_n512_formal_30k.train_score_design import (
    SEED, TRAIN_COUNT, VAL_COUNT, SEGMENT_SIZE, MAX_EPOCHS, MICROBATCH, EFFECTIVE_BATCH,
    MIN_SELECTION_EPOCH, BUDGETS, SELECTIONS, learning_rate, segment_plan, canonical_digest,
    run_lock, event, update_winners, validate_resume as validate_joint_progress,
    train_score_segment as train_input_segment)
from experiments.rachel_n512_formal_30k.train_joint_damage import build_random_model, state_digest
from experiments.rachel_n512_formal_30k.train_edge_weathering import (
    _cpu_model_state, _sha256, capture_rng_state, restore_rng_state)
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import evaluate_pair_validation, save_json
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from staging.pairwise_v0_2.models.rachel_candidate_score import RachelCandidateScore, CandidateScoreConfig
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedWeatheredDataset
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader

SCHEMA = "rachel-score-input-training/1"
CHECKPOINT_SCHEMA = "rachel-score-input-checkpoint/1"


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def input_spec(args):
    windows = FOUR_WINDOWS if args.single_window is None else (float(args.single_window),)
    return InputVariantSpec(coarse_size=args.coarse_size, window_sizes_px=windows,
                            transport_fusion=args.transport_fusion, contour_cap=512)


def build_training_model(source, spec, architecture):
    """Typed model and canonical post-construction RNG, with no pretrained load."""
    reference_config = asdict(full24_reference_config())
    if (canonical(source.get("model_config")) != canonical(reference_config) or
            source.get("seam_loss_enabled", False)):
        raise ValueError("metadata source must be the unchanged Full24 reference configuration")
    built = build_score_input_model(spec, architecture=architecture, seed=SEED)
    # The typed factory preserves caller RNG. Advance the global generators to
    # the live joint reference's post-base/head point, independent of input axis.
    metadata = dict(model_kind="full", model_options={}, model_config=reference_config,
                    loss_config=source["loss_config"], seam_loss_enabled=False)
    reference, _, loss_config, shared_digest = build_random_model(metadata, seed=SEED)
    if architecture != "original":
        reference = RachelCandidateScore(reference, CandidateScoreConfig(), architecture)
    reference_score_digest = state_digest(reference)
    del reference
    if built.metadata["input_variant"]["initialization"]["reference_state_sha256"] != shared_digest:
        raise ValueError("input factory and live joint reference initializations differ")
    digests = dict(shared_base_initial_weights_sha256=shared_digest,
        initial_weights_sha256=state_digest(built.model),
        reference_score_initial_weights_sha256=reference_score_digest,
        initial_torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    return built, loss_config, digests


def create_optimizer(model, architecture):
    model.train()
    if architecture == "original":
        model.requires_grad_(True)
    else:
        model.set_training_mode("joint")
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=learning_rate(1), weight_decay=1e-4)


def experiment_identity(args, *, source_path, metadata, loss_config, training, digests):
    val_manifest = Path(args.dataset).resolve() / "pairs" / "val.jsonl"
    return dict(schema_version=SCHEMA, seed=SEED, training_mode="joint", score_design=args.architecture,
        score_input_metadata=metadata, input_spec=metadata["input_variant"]["spec"],
        changed_axis=metadata["input_variant"]["changed_axis"],
        base_model_metadata=metadata["base_model_metadata"],
        reference_base_model_metadata=metadata["input_variant"]["reference_model_metadata"],
        candidate_config=metadata["candidate_config"], loss_config=asdict(loss_config), **digests,
        metadata_source_checkpoint_sha256=_sha256(source_path), source_weights_loaded=False,
        train_count=TRAIN_COUNT, train_split="train", train_manifest=str(training.manifest_path),
        train_manifest_sha256=_sha256(training.manifest_path), validation_count=VAL_COUNT,
        validation_split="val", validation_manifest=str(val_manifest), validation_manifest_sha256=_sha256(val_manifest),
        max_epochs=MAX_EPOCHS, segment_pairs=SEGMENT_SIZE, microbatch=MICROBATCH,
        effective_batch=EFFECTIVE_BATCH, optimizer="AdamW", weight_decay=1e-4,
        lr_by_epoch=[learning_rate(e) for e in range(1, MAX_EPOCHS + 1)],
        grad_clip_norm=5.0, precision="fp32", workers=args.workers,
        candidate_correctness_weight=.5 if args.architecture == "candidate_dual" else 0.,
        candidate_correctness_tolerance_px=20.,
        candidate_correctness_reduction="mean valid candidates per pair, then mean supervised pairs",
        validation_every_epochs=1, min_selection_epoch=MIN_SELECTION_EPOCH, budgets=list(BUDGETS),
        selection_rules={"max_f1": "VAL fused F1, then AP, then earliest epoch",
                         "recall95": "VAL precision at empirical95% recall, then AP, then earliest epoch"},
        thresholds="each selected checkpoint cleanVAL only; accept score>=threshold",
        optimizer_reset_at_epoch_or_budget=False, staged_training=False, contour_resampling=False,
        held_out_used_for_training_or_selection=False)


def checkpoint_metadata(metadata):
    base = metadata["base_model_metadata"]
    return dict(score_input_checkpoint_schema=CHECKPOINT_SCHEMA, score_input_training_schema=SCHEMA,
        score_input_metadata=metadata, model_kind="score_input_" + metadata["architecture"],
        model_config=base["model_config"], base_model_kind=base["model_kind"],
        base_model_options=base["model_options"])


def validate_input_checkpoint(checkpoint, identity):
    if (checkpoint.get("score_input_checkpoint_schema") != CHECKPOINT_SCHEMA or
            checkpoint.get("score_input_training_schema") != SCHEMA or
            "score_design_schema" in checkpoint or "s3_checkpoint_schema" in checkpoint):
        raise ValueError("not an independent score-input checkpoint")
    if identity.get("schema_version") != SCHEMA:
        raise ValueError("not a score-input training identity")
    validate_joint_progress(checkpoint, identity)
    metadata = checkpoint.get("score_input_metadata")
    if canonical(metadata) != canonical(identity["score_input_metadata"]):
        raise ValueError("checkpoint input/model metadata differs from identity")
    aliases = checkpoint_metadata(metadata)
    if any(canonical(checkpoint.get(k)) != canonical(v) for k, v in aliases.items()):
        raise ValueError("checkpoint base kind/config/options or model kind differs")
    if checkpoint.get("epoch") != (checkpoint["completed_segments"] + 3) // 4:
        raise ValueError("checkpoint epoch differs from committed segment count")


def load_score_input_checkpoint(checkpoint):
    """The only supported loader for these early/post typed model checkpoints."""
    identity = checkpoint.get("resume_identity", {})
    validate_input_checkpoint(checkpoint, identity)
    return restore_score_input_model(checkpoint["score_input_metadata"], checkpoint["model_state_dict"]).model


def restore_training_state(model, optimizer, checkpoint, identity):
    validate_input_checkpoint(checkpoint, identity)
    # Typed restoration also validates the actual stored physical sampling grid.
    validated = restore_score_input_model(checkpoint["score_input_metadata"], checkpoint["model_state_dict"])
    model.load_state_dict(validated.model.state_dict(), strict=True)
    del validated
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint["completed_segments"], checkpoint["winners"]


def payload(model, optimizer, *, metadata, identity, loss_config, training_data, completed, winners, role):
    return dict(**checkpoint_metadata(metadata), model_state_dict=_cpu_model_state(model),
        optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state(),
        resume_identity=identity, loss_config=asdict(loss_config), completed_segments=completed,
        epoch=(completed + 3) // 4, global_exposure=completed * SEGMENT_SIZE,
        optimizer_updates=completed * SEGMENT_SIZE // EFFECTIVE_BATCH, winners=winners,
        checkpoint_role=role, training_data=training_data, seed=SEED, initialization="random",
        source_weights_loaded=False, initial_weights_sha256=identity["initial_weights_sha256"],
        shared_base_initial_weights_sha256=identity["shared_base_initial_weights_sha256"],
        formal_training_counted=True, precision="fp32", seam_loss_enabled=False, resample_contour_cap=None)


def publish_freezes(root, *, epoch, winners, identity):
    if epoch < MIN_SELECTION_EPOCH:
        return
    resolved = {}
    for name, record in winners.items():
        record = dict(record)
        record["checkpoint_sha256"] = _sha256(record["checkpoint"])
        record["validation_predictions_sha256"] = _sha256(record["validation_predictions"])
        resolved[name] = record
    if set(resolved) != set(SELECTIONS):
        raise RuntimeError("no fully covered eligible VAL checkpoint")
    freeze = dict(schema_version=SCHEMA, status="frozen_at_budget" if epoch in BUDGETS else "provisional",
        budget_epochs=epoch, budget_exposures=epoch * TRAIN_COUNT, eligible_epoch_range=[5, epoch],
        score_input_metadata=identity["score_input_metadata"], input_spec=identity["input_spec"],
        changed_axis=identity["changed_axis"], winners=resolved, resume_identity=identity,
        resume_identity_sha256=canonical_digest(identity), selection_population="cleanVAL3000 only",
        held_out_used_for_fit=False, legacy_score_checkpoint_compatible=False, staged_training=False)
    save_json(root / "current_winners.json", freeze)
    if epoch in BUDGETS:
        save_json(root / "budget_freezes" / ("%03d" % epoch) / "freeze.json", freeze)


def run(args):
    spec = input_spec(args)  # Reject multi-axis/N1024 experiments before I/O.
    stop = args.stop_after_epoch
    if not 1 <= stop <= MAX_EPOCHS or args.workers < 0 or args.log_every <= 0:
        raise ValueError("invalid stop/workers/log interval")
    if args.resume and args.smoke:
        raise ValueError("discard-only smoke cannot resume formal training")
    if platform.system() != "Linux" or not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("input-variant training requires the remote Linux CUDA server")
    torch.set_num_threads(1)
    source_path = Path(args.checkpoint).resolve(strict=True)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    built, loss_config, digests = build_training_model(source, spec, args.architecture)
    del source
    training = MaterializedWeatheredDataset(args.train_materialized_manifest)
    validation = RachelPairDataset(args.dataset, "val")
    if len(training) != TRAIN_COUNT or len(validation) != VAL_COUNT:
        raise ValueError("requires unchanged materialized Full24 TRAIN24000 and cleanVAL3000")
    identity = experiment_identity(args, source_path=source_path, metadata=built.metadata,
        loss_config=loss_config, training=training, digests=digests)
    data_record = dict(kind="fixed_e1_materialized", unique_count=len(training),
        manifest=str(training.manifest_path), stats=training.stats, protocol=training.protocol)
    root = Path(args.output).resolve()
    if args.resume:
        if not root.is_dir():
            raise ValueError("resume directory does not exist")
    else:
        root.mkdir(parents=True, exist_ok=False)
    args.output = str(root)
    with run_lock(root):
        return _run_locked(args, root, built, training, validation, loss_config, identity, data_record)


def _run_locked(args, root, built, training, validation, loss_config, identity, data_record):
    device = torch.device(args.device)
    model, metadata = built.model.to(device), built.metadata
    optimizer = create_optimizer(model, args.architecture)
    completed, winners = 0, {}
    if args.resume:
        saved = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        completed, winners = restore_training_state(model, optimizer, saved, identity)
        del saved
        if completed > args.stop_after_epoch * 4:
            raise ValueError("requested stop precedes committed progress")
        if completed and completed % 4 == 0:
            publish_freezes(root, epoch=completed // 4, winners=winners, identity=identity)
    protocol = dict(**identity, status="running", arguments=vars(args), training_data=data_record,
        plan=segment_plan(), requested_stop_after_epoch=args.stop_after_epoch,
        smoke=bool(args.smoke), formal_training_counted=not bool(args.smoke),
        implementation_sha256=_sha256(__file__),
        runtime=dict(torch=torch.__version__, python=platform.python_version(), cuda=torch.version.cuda))
    save_json(root / "protocol.json", protocol)
    if not args.resume and not args.smoke:
        runner._atomic_torch_save(root / "last.pt", payload(model, optimizer, metadata=metadata,
            identity=identity, loss_config=loss_config, training_data=data_record,
            completed=0, winners={}, role="initial_recovery"))
    event(root, event="resume" if args.resume else "start", completed_segments=completed,
          stop_after_epoch=args.stop_after_epoch, input_spec=identity["input_spec"], schema=SCHEMA)
    started = time.monotonic()
    try:
        for segment in segment_plan():
            number, epoch = segment["number"], segment["epoch"]
            if number <= completed:
                continue
            if epoch > args.stop_after_epoch:
                break
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch)
            count = args.smoke or SEGMENT_SIZE
            order = runner.epoch_indices(TRAIN_COUNT, seed=SEED, epoch=epoch, limit=None)
            order = order[segment["offset"]:segment["offset"] + count]
            loader = make_weathering_loader(training, order, batch_size=MICROBATCH,
                num_workers=args.workers, seed=SEED + number, contour_cap=512)
            save_json(root / "status.json", dict(status="running", phase="train", pid=os.getpid(),
                epoch=epoch, segment=number, global_exposure=completed * SEGMENT_SIZE,
                optimizer_updates=completed * SEGMENT_SIZE // EFFECTIVE_BATCH))
            torch.cuda.reset_peak_memory_stats(device)
            report = train_input_segment(model, loader, optimizer, loss_config, device, args, epoch)
            if report["samples"] != count or report["optimizer_updates"] != count // EFFECTIVE_BATCH:
                raise RuntimeError("input-variant segment exposure/update count differs")
            save_json(root / ("segment_%03d.json" % number), dict(segment=segment, training=report))
            if args.smoke:
                result = dict(status="smoke_complete", smoke=True, formal_training_counted=False,
                    weights_discarded=True, training=report)
                save_json(root / "smoke.json", result)
                save_json(root / "status.json", result)
                protocol.update(result)
                save_json(root / "protocol.json", protocol)
                return result
            if segment["epoch_complete"]:
                from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
                save_json(root / "status.json", dict(status="running", phase="validation", epoch=epoch,
                    global_exposure=number * SEGMENT_SIZE, optimizer_updates=number * SEGMENT_SIZE // EFFECTIVE_BATCH))
                val_loader = make_ablation_loader(validation, list(range(VAL_COUNT)), batch_size=8,
                    num_workers=args.workers, seed=SEED, contour_cap=512)
                val_report, rows = evaluate_pair_validation(model, val_loader, device)
                points = fit_operating_points([r["label"] for r in rows], [r["classification"]["fused"] for r in rows])
                save_json(root / ("validation_%03d_rows.json" % epoch), rows)
                save_json(root / ("validation_%03d.json" % epoch), dict(epoch=epoch,
                    global_exposure=number * SEGMENT_SIZE, validation=val_report,
                    operating_points=points, selection_eligible=epoch >= MIN_SELECTION_EPOCH))
                winners = update_winners(winners, root=root, epoch=epoch, report=val_report, points=points)
                event(root, event="validation_complete", epoch=epoch, validation=val_report, operating_points=points)
            recovery = payload(model, optimizer, metadata=metadata, identity=identity,
                loss_config=loss_config, training_data=data_record, completed=number, winners=winners,
                role="epoch_anchor" if segment["epoch_complete"] else "recovery")
            if segment["epoch_complete"]:
                runner._atomic_torch_save(root / ("epoch_%03d.pt" % epoch), recovery)
            runner._atomic_torch_save(root / "last.pt", recovery)
            completed = number
            del recovery
            if segment["epoch_complete"]:
                publish_freezes(root, epoch=epoch, winners=winners, identity=identity)
            event(root, event="segment_committed", segment=number, epoch=epoch,
                  global_exposure=completed * SEGMENT_SIZE)
        if completed != args.stop_after_epoch * 4:
            raise RuntimeError("requested complete-epoch budget was not reached")
        status = "complete" if args.stop_after_epoch == MAX_EPOCHS else "budget_complete"
        result = dict(status=status, phase="train_val_complete", pid=os.getpid(), epoch=args.stop_after_epoch,
            completed_segments=completed, global_exposure=completed * SEGMENT_SIZE,
            optimizer_updates=completed * SEGMENT_SIZE // EFFECTIVE_BATCH,
            can_resume_to_epoch=MAX_EPOCHS if args.stop_after_epoch < MAX_EPOCHS else None,
            elapsed_s=time.monotonic() - started)
        protocol.update(result)
        save_json(root / "protocol.json", protocol)
        save_json(root / "status.json", result)
        event(root, event=status, **{k: v for k, v in result.items() if k != "status"})
        return result
    except BaseException as error:
        save_json(root / "status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error), completed_segments=completed, resumable_exposure=completed * SEGMENT_SIZE))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--train-materialized-manifest", required=True)
    p.add_argument("--architecture", choices=("original", "candidate_pair", "candidate_dual"), required=True)
    p.add_argument("--coarse-size", type=int, choices=(128, 256, 512), default=128)
    p.add_argument("--single-window", type=int, choices=(7, 16, 32, 64))
    p.add_argument("--transport-fusion", choices=("early", "post"), default="early")
    p.add_argument("--output", required=True)
    p.add_argument("--stop-after-epoch", type=int, default=MAX_EPOCHS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--smoke", type=int, choices=(32, 64, 128), nargs="?", const=32)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
