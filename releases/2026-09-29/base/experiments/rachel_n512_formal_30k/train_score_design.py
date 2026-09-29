"""Nested-budget score-design training on fixed materialized Full24 TRAIN.

CUDA server only. --checkpoint contributes architecture/loss metadata, NEVER
weights. One seed260913 trajectory supplies 5/10/20/30/50-epoch comparisons.
The complete plan is always50 epochs; --stop-after-epoch only pauses it and is
not part of resume identity. Every6000 pairs commits optimizer+RNG to last.pt;
every epoch also retains its complete state and clean-VAL3000 predictions.
Selection starts at epoch5 and uses VAL alone. TEST/REAL/OOD are never opened.
All training progress goes to progress.jsonl, not repetitive stdout messages.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
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

from experiments.rachel_n512_formal_30k.train_joint_damage import build_random_model, state_digest
from experiments.rachel_n512_formal_30k.train_edge_weathering import (
    _cpu_model_state, _sha256, capture_rng_state, restore_rng_state)
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import (
    evaluate_pair_validation, save_json)
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from staging.pairwise_v0_2.models.rachel_model_factory import (
    model_metadata, load_rachel_checkpoint)
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedWeatheredDataset
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_weathering_training import (
    make_weathering_loader, compute_weathering_loss, WeatheringStatistics)

SCHEMA = "rachel-score-design-training/1"
SEED, TRAIN_COUNT, VAL_COUNT = 260913, 24000, 3000
SEGMENT_SIZE, MAX_EPOCHS = 6000, 50
MICROBATCH, EFFECTIVE_BATCH, MIN_SELECTION_EPOCH = 4, 16, 5
BUDGETS = (5, 10, 20, 30, 50)
SELECTIONS = ("max_f1", "recall95")


def learning_rate(epoch):
    if type(epoch) is not int or not 1 <= epoch <= MAX_EPOCHS:
        raise ValueError("epoch must be in1..50")
    return 1e-4 if epoch <= 3 else 2e-5


def segment_plan():
    return [dict(number=(epoch - 1) * 4 + part + 1, epoch=epoch,
                 offset=part * SEGMENT_SIZE, count=SEGMENT_SIZE,
                 global_start=(epoch - 1) * TRAIN_COUNT + part * SEGMENT_SIZE,
                 global_stop=(epoch - 1) * TRAIN_COUNT + (part + 1) * SEGMENT_SIZE,
                 epoch_complete=part == 3, learning_rate=learning_rate(epoch))
            for epoch in range(1, MAX_EPOCHS + 1) for part in range(4)]


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def experiment_identity(args, *, source_path, initial_digest, architecture,
                        loss_config, training, base_metadata, shared_base_digest=None,
                        candidate_config=None):
    """Operational stop/logging/output controls cannot alter the trajectory."""
    val_manifest = Path(args.dataset).resolve() / "pairs" / "val.jsonl"
    return dict(schema_version=SCHEMA, seed=SEED,
        score_design=args.architecture, training_mode=args.training_mode,
        base_model_metadata=base_metadata, architecture=architecture,
        loss_config=asdict(loss_config), initial_weights_sha256=initial_digest,
        shared_base_initial_weights_sha256=shared_base_digest or initial_digest,
        candidate_config=candidate_config or {}, candidate_correctness_tolerance_px=20.0,
        candidate_correctness_weight=0.5 if args.architecture == "candidate_dual" else 0.0,
        candidate_correctness_reduction="mean valid candidates per pair, then mean supervised pairs",
        metadata_source_checkpoint_sha256=_sha256(source_path),
        train_manifest=str(training.manifest_path),
        train_manifest_sha256=_sha256(training.manifest_path),
        train_count=len(training), train_split="train",
        validation_manifest=str(val_manifest), validation_manifest_sha256=_sha256(val_manifest),
        validation_count=VAL_COUNT, validation_split="val",
        max_epochs=MAX_EPOCHS, segment_pairs=SEGMENT_SIZE, microbatch=MICROBATCH,
        effective_batch=EFFECTIVE_BATCH, optimizer="AdamW", weight_decay=1e-4,
        lr_by_epoch=[learning_rate(e) for e in range(1, MAX_EPOCHS + 1)],
        grad_clip_norm=5.0, precision="fp32", workers=args.workers,
        validation_every_epochs=1, min_selection_epoch=MIN_SELECTION_EPOCH,
        budgets=list(BUDGETS), optimizer_reset_at_epoch_or_budget=False,
        selection_rules={"max_f1": "VAL fused F1, then AP, then earliest epoch",
                         "recall95": "VAL precision at empirical95% recall, then AP, then earliest epoch"},
        thresholds="independently fit on each selected checkpoint VAL; accept score>=threshold",
        held_out_used_for_training_or_selection=False)


def validate_resume(checkpoint, identity):
    if checkpoint.get("resume_identity") != identity:
        previous = checkpoint.get("resume_identity", {})
        different = sorted(k for k in set(previous) | set(identity) if previous.get(k) != identity.get(k))
        raise ValueError("resume identity differs: " + ", ".join(different))
    completed = checkpoint.get("completed_segments")
    if type(completed) is not int or not 0 <= completed <= MAX_EPOCHS * 4:
        raise ValueError("invalid recovery segment count")
    if (checkpoint.get("global_exposure") != completed * SEGMENT_SIZE or
            checkpoint.get("optimizer_updates") != completed * SEGMENT_SIZE // EFFECTIVE_BATCH):
        raise ValueError("recovery exposure/update counts differ")
    if "optimizer_state_dict" not in checkpoint or "rng_state" not in checkpoint:
        raise ValueError("resume requires optimizer and RNG, not only model weights")


def score_model_metadata(model, architecture="original"):
    if architecture == "original":
        return dict(**model_metadata(model), score_design_schema=SCHEMA,
                    score_design=dict(architecture="original", base_model_metadata=model_metadata(model)))
    return dict(model_kind="score_design_" + architecture,
        model_config=asdict(model.config), model_options=asdict(model.candidate_config),
        score_design_schema=SCHEMA, score_design=model.metadata())


def load_score_checkpoint(checkpoint):
    """Explicit evaluation entry; candidate checkpoints cannot masquerade as Full."""
    if checkpoint.get("score_design_schema") != SCHEMA:
        raise ValueError("not a score-design checkpoint")
    design = checkpoint.get("score_design", {})
    if design.get("architecture") == "original":
        if checkpoint.get("model_kind") != "full":
            raise ValueError("original architecture metadata mismatch")
        return load_rachel_checkpoint(checkpoint)
    from staging.pairwise_v0_2.models.rachel_candidate_score import build_score_model
    if (design.get("architecture") not in ("candidate_pair", "candidate_dual") or
            checkpoint.get("model_kind") != "score_design_" + design["architecture"]):
        raise ValueError("unknown candidate checkpoint architecture")
    model = build_score_model(design["model_config"], design["architecture"], design["candidate_config"])
    model.set_training_mode(design["training_mode"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def train_score_segment(model, loader, optimizer, loss_config, device, args, epoch):
    """Original Full24 loss/FP32 accumulation plus ONLY the registered R term."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = EFFECTIVE_BATCH // MICROBATCH
    samples, updates, statistics = 0, 0, WeatheringStatistics()
    loss_names = ("fused_pair_bce", "coarse_pair_bce", "local_pair_bce", "assignment_nll",
                  "translation_smooth_l1", "sinkhorn_residual")
    loss_sums = {name: 0.0 for name in ("total", "original_total", "candidate_correctness_bce") + loss_names}
    candidate_counts = dict(valid=0, positive=0, negative=0, supervised_pairs=0)
    initial_status = json.loads((Path(args.output) / "status.json").read_text())
    started = time.perf_counter()
    for step, wrapped in enumerate(loader):
        group_start = (step // accumulation) * accumulation
        group_samples = min(EFFECTIVE_BATCH, len(loader.dataset) - group_start * MICROBATCH)
        inputs, targets = runner._full_batch(wrapped.batch, device)
        output = model(*inputs)
        pose = torch.as_tensor(wrapped.pose_supervision_enabled, dtype=torch.bool, device=device)
        original = compute_weathering_loss(output, *targets, config=loss_config,
                                           pose_supervision_enabled=pose)
        correctness = original.total.new_zeros(())
        if args.architecture != "original":
            from staging.pairwise_v0_2.models.rachel_candidate_score import candidate_correctness_loss
            # All positive GT remains valid for R, INCLUDING changed/corroded positives.
            # targets[-1] is true GT availability; pose is only the old shift-loss gate.
            correctness, rstats = candidate_correctness_loss(
                output, targets[0], targets[-2], targets[-1], tolerance_px=20.0)
            valid_count = int(rstats["candidate_supervised"].sum().item())
            positive_count = int(rstats["candidate_positive_count"].item())
            candidate_counts["valid"] += valid_count
            candidate_counts["positive"] += positive_count
            candidate_counts["negative"] += valid_count - positive_count
            candidate_counts["supervised_pairs"] += int(rstats["candidate_pair_count"].item())
        total = original.total + 0.5 * correctness if args.architecture == "candidate_dual" else original.total
        count = len(wrapped.batch.pair_ids)
        (total * (count / group_samples)).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        samples += count
        values = torch.stack((total.detach(), original.total.detach(), correctness.detach()) +
                             tuple(getattr(original, name).detach() for name in loss_names)).cpu().tolist()
        for name, value in zip(loss_sums, values):
            loss_sums[name] += float(value) * count
        statistics.add(wrapped)
        if (step + 1) % args.log_every == 0:
            progress = dict(event="train_progress", variant=args.architecture, epoch=epoch,
                samples=samples, updates=updates, mean_loss=loss_sums["total"] / samples,
                loss_components={k: v / samples for k, v in loss_sums.items()},
                candidate_correctness_counts=candidate_counts,
                seconds=time.perf_counter() - started)
            event(Path(args.output), **progress)
            save_json(Path(args.output) / "status.json", dict(status="running", phase="train",
                epoch=epoch, global_exposure=initial_status["global_exposure"] + samples,
                optimizer_updates=initial_status["optimizer_updates"] + updates, progress=progress))
    return dict(samples=samples, optimizer_updates=updates, mean_loss=loss_sums["total"] / max(1, samples),
        loss_components={k: v / max(1, samples) for k, v in loss_sums.items()},
        candidate_correctness_counts=candidate_counts, seconds=time.perf_counter() - started,
        peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device), weathering=statistics.report())


@contextmanager
def run_lock(root):
    with (root / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another trainer owns this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def event(root, **value):
    value.setdefault("unix_time", time.time())
    with (root / "progress.jsonl").open("a", buffering=1) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def winner_record(root, epoch, report, points, selection):
    checkpoint = root / ("epoch_%03d.pt" % epoch)
    return dict(selection=selection, selected_epoch=epoch,
        selected_global_exposure=epoch * TRAIN_COUNT,
        checkpoint=str(checkpoint), checkpoint_sha256=None,
        validation_predictions=str(root / ("validation_%03d_rows.json" % epoch)),
        classifier_thresholds=report["thresholds"], operating_points=points,
        primary_pair_threshold=points["thresholds"]["max_f1" if selection == "max_f1" else "recall_95"],
        validation=report, selection_key=(report["selection_key"] if selection == "max_f1" else points["selection_key"]),
        test_or_real_or_ood_used_for_fit=False)


def update_winners(winners, *, root, epoch, report, points):
    """Return a new state; exact ties preserve the earlier checkpoint."""
    result = dict(winners)
    if epoch < MIN_SELECTION_EPOCH or report["decision_coverage"] != 1.0:
        return result
    for selection in SELECTIONS:
        candidate = winner_record(root, epoch, report, points, selection)
        previous = result.get(selection)
        if previous is None or tuple(candidate["selection_key"]) > tuple(previous["selection_key"]):
            result[selection] = candidate
    return result


def publish_freezes(root, *, epoch, winners, identity):
    """Only called after last.pt committed; files derive from committed state."""
    if epoch < MIN_SELECTION_EPOCH:
        return
    resolved = {}
    for name, record in winners.items():
        record = dict(record)
        record["checkpoint_sha256"] = _sha256(record["checkpoint"])
        resolved[name] = record
    if set(resolved) != set(SELECTIONS):
        raise RuntimeError("no eligible fully-covered VAL checkpoint")
    freeze = dict(schema_version=SCHEMA, status="frozen_at_budget" if epoch in BUDGETS else "provisional",
        budget_epochs=epoch, budget_exposures=epoch * TRAIN_COUNT,
        eligible_epoch_range=[MIN_SELECTION_EPOCH, epoch], winners=resolved,
        resume_identity_sha256=canonical_digest(identity),
        selection_population="cleanVAL3000 only", held_out_used_for_fit=False)
    save_json(root / "current_winners.json", freeze)
    if epoch in BUDGETS:
        save_json(root / "budget_freezes" / ("%03d" % epoch) / "freeze.json", freeze)


def _payload(model, *, args, identity, loss_config, data_record, epoch,
             samples, updates, completed, winners, role):
    return dict(model_state_dict=_cpu_model_state(model),
        **score_model_metadata(model, args.architecture), loss_config=asdict(loss_config),
        epoch=epoch, seed=SEED, global_exposure=samples, optimizer_updates=updates,
        completed_segments=completed, winners=winners, checkpoint_role=role,
        resume_identity=identity, initialization="random", source_weights_loaded=False,
        initial_weights_sha256=identity["initial_weights_sha256"], training_data=data_record,
        shared_base_initial_weights_sha256=identity["shared_base_initial_weights_sha256"],
        seam_loss_enabled=False, resample_contour_cap=None,
        precision="fp32", formal_training_counted=True)


def run(args):
    if args.training_mode != "joint":
        raise NotImplementedError("stage requires an explicitly selected trained matcher; no hidden schedule")
    if args.max_epochs != MAX_EPOCHS:
        raise ValueError("max_epochs must remain50; use --stop-after-epoch for nested budgets")
    stop = MAX_EPOCHS if args.stop_after_epoch is None else args.stop_after_epoch
    if not 1 <= stop <= MAX_EPOCHS or args.workers < 0 or args.log_every <= 0:
        raise ValueError("invalid stop epoch, workers, or log interval")
    if args.resume and args.smoke:
        raise ValueError("smoke is discard-only and cannot resume formal training")
    if not args.device.startswith("cuda") or not torch.cuda.is_available() or platform.system() != "Linux":
        raise RuntimeError("training must run on the remote Linux CUDA server")
    torch.set_num_threads(1)
    source_path = Path(args.checkpoint).resolve(strict=True)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    model, architecture, loss_config, initial_digest = build_random_model(source, seed=SEED)
    del source
    base_metadata = model_metadata(model)
    shared_base_digest = initial_digest
    candidate_config = {}
    if args.architecture != "original":
        from staging.pairwise_v0_2.models.rachel_candidate_score import RachelCandidateScore, CandidateScoreConfig
        # Construct head AFTER the exactly common random base; no seed reset/weights load.
        model = RachelCandidateScore(model, CandidateScoreConfig(), args.architecture)
        candidate_config = asdict(model.candidate_config)
        initial_digest = state_digest(model)
    training = MaterializedWeatheredDataset(args.train_materialized_manifest)
    if len(training) != TRAIN_COUNT:
        raise ValueError("formal and smoke both require the fixed Full24 materialized24000 manifest")
    validation = RachelPairDataset(args.dataset, "val")
    if len(validation) != VAL_COUNT:
        raise ValueError("requires unchanged cleanVAL3000")
    data_record = dict(kind="fixed_e1_materialized", unique_count=len(training),
                       manifest=str(training.manifest_path), stats=training.stats, protocol=training.protocol)
    identity = experiment_identity(args, source_path=source_path, initial_digest=initial_digest,
        architecture=architecture, loss_config=loss_config, training=training, base_metadata=base_metadata,
        shared_base_digest=shared_base_digest, candidate_config=candidate_config)
    root = Path(args.output).resolve()
    if args.resume:
        if not root.is_dir():
            raise ValueError("resume output directory does not exist")
    else:
        root.mkdir(parents=True, exist_ok=False)
    args.output = str(root)
    args.batch_size, args.effective_batch_size = MICROBATCH, EFFECTIVE_BATCH
    args.enable_seam_loss, args.seam_weight = False, 0.0
    args.status_path = str(root / "status.json")
    with run_lock(root):
        return _run_locked(args, root, model, training, validation, identity,
                           loss_config, data_record, stop)


def _run_locked(args, root, model, training, validation, identity, loss_config, data_record, stop):
    device = torch.device(args.device)
    model = model.to(device).train()
    if args.architecture == "original":
        model.requires_grad_(True)
    else:
        model.set_training_mode("joint")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=learning_rate(1), weight_decay=1e-4)
    winners, completed, samples, updates = {}, 0, 0, 0
    if args.resume:
        recovery = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        validate_resume(recovery, identity)
        model.load_state_dict(recovery["model_state_dict"], strict=True)
        optimizer.load_state_dict(recovery["optimizer_state_dict"])
        winners, completed = recovery["winners"], recovery["completed_segments"]
        samples, updates = recovery["global_exposure"], recovery["optimizer_updates"]
        restore_rng_state(recovery["rng_state"])
        if completed > stop * 4:
            raise ValueError("requested stop precedes the committed checkpoint")
        # Repair a budget pointer if interrupted after last.pt but before publication.
        if completed and completed % 4 == 0:
            publish_freezes(root, epoch=completed // 4, winners=winners, identity=identity)
        del recovery
    protocol = dict(**identity, status="running", arguments=vars(args), training_data=data_record,
        source_weights_loaded=False, initialization="random", requested_stop_after_epoch=stop,
        plan=segment_plan(), implementation_sha256=_sha256(__file__),
        runtime=dict(torch=torch.__version__, python=platform.python_version(), cuda=torch.version.cuda),
        retention="last.pt at every6000 pairs; full optimizer/RNG in every epoch checkpoint; immutable budget pointers",
        smoke=bool(args.smoke), formal_training_counted=not bool(args.smoke))
    save_json(root / "protocol.json", protocol)
    if not args.resume and not args.smoke:
        # Make even a failure before the first6000-pair boundary resumable.
        initial = _payload(model, args=args, identity=identity, loss_config=loss_config,
            data_record=data_record, epoch=0, samples=0, updates=0, completed=0,
            winners={}, role="initial_recovery")
        initial.update(optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state())
        runner._atomic_torch_save(root / "last.pt", initial)
        del initial
    event(root, event="resume" if args.resume else "start", completed_segments=completed,
          stop_after_epoch=stop, architecture=args.architecture, smoke=bool(args.smoke))
    started = time.monotonic()
    try:
        for segment in segment_plan():
            number, epoch = segment["number"], segment["epoch"]
            if number <= completed:
                continue
            if epoch > stop:
                break
            order = runner.epoch_indices(TRAIN_COUNT, seed=SEED, epoch=epoch, limit=None)
            count = args.smoke if args.smoke else SEGMENT_SIZE
            order = order[segment["offset"]:segment["offset"] + count]
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch)
            loader = make_weathering_loader(training, order, batch_size=MICROBATCH,
                num_workers=args.workers, seed=SEED + number, contour_cap=512)
            save_json(root / "status.json", dict(status="running", pid=os.getpid(), phase="train",
                epoch=epoch, segment=number, global_exposure=samples, optimizer_updates=updates))
            torch.cuda.reset_peak_memory_stats(device)
            # Reuse the exact original Full24/E1 loss, accumulation and clipping.
            report = train_score_segment(model, loader, optimizer, loss_config, device, args, epoch)
            if report["samples"] != count or report["optimizer_updates"] != count // EFFECTIVE_BATCH:
                raise RuntimeError("segment exposure/update count differs")
            samples += count
            updates += report["optimizer_updates"]
            save_json(root / ("segment_%03d.json" % number), dict(segment=segment, training=report))
            if args.smoke:
                result = dict(status="complete", smoke=True, formal_training_counted=False,
                    weights_discarded=True, samples=samples, updates=updates, training=report)
                save_json(root / "smoke.json", result)
                save_json(root / "status.json", result)
                protocol.update(status="smoke_complete", elapsed_s=time.monotonic() - started)
                save_json(root / "protocol.json", protocol)
                return result
            if segment["epoch_complete"]:
                from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
                save_json(root / "status.json", dict(status="running", pid=os.getpid(), phase="validation",
                    epoch=epoch, global_exposure=samples, optimizer_updates=updates))
                val_loader = make_ablation_loader(validation, list(range(VAL_COUNT)), batch_size=8,
                    num_workers=args.workers, seed=SEED, contour_cap=512)
                val_report, rows = evaluate_pair_validation(model, val_loader, device)
                points = fit_operating_points([r["label"] for r in rows],
                    [r["classification"]["fused"] for r in rows])
                save_json(root / ("validation_%03d_rows.json" % epoch), rows)
                save_json(root / ("validation_%03d.json" % epoch), dict(epoch=epoch,
                    global_exposure=samples, validation=val_report, operating_points=points,
                    selection_eligible=epoch >= MIN_SELECTION_EPOCH))
                winners = update_winners(winners, root=root, epoch=epoch, report=val_report, points=points)
                event(root, event="validation_complete", epoch=epoch, global_exposure=samples,
                      validation=val_report, operating_points=points)
            recovery = _payload(model, args=args, identity=identity, loss_config=loss_config,
                data_record=data_record, epoch=epoch, samples=samples, updates=updates,
                completed=number, winners=winners, role="recovery")
            recovery.update(optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state())
            if segment["epoch_complete"]:
                runner._atomic_torch_save(root / ("epoch_%03d.pt" % epoch), recovery)
            runner._atomic_torch_save(root / "last.pt", recovery)
            completed = number
            del recovery
            if segment["epoch_complete"]:
                publish_freezes(root, epoch=epoch, winners=winners, identity=identity)
            event(root, event="segment_committed", segment=number, epoch=epoch,
                  global_exposure=samples, optimizer_updates=updates)
        if samples != stop * TRAIN_COUNT or updates != samples // EFFECTIVE_BATCH:
            raise RuntimeError("incomplete requested budget")
        status = "complete" if stop == MAX_EPOCHS else "budget_complete"
        result = dict(status=status, pid=os.getpid(), phase="train_val_complete", epoch=stop,
            completed_segments=completed, global_exposure=samples, optimizer_updates=updates,
            can_resume_to_epoch=MAX_EPOCHS if stop < MAX_EPOCHS else None,
            elapsed_s=time.monotonic() - started)
        protocol.update(result)
        save_json(root / "protocol.json", protocol)
        save_json(root / "status.json", result)
        event(root, event=status, **{k: v for k, v in result.items() if k != "status"})
        return result
    except BaseException as error:
        save_json(root / "status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            pid=os.getpid(), error=repr(error), completed_segments=completed,
            resumable_exposure=completed * SEGMENT_SIZE, weights_after_boundary_discarded=True))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="architecture/loss metadata only; weights never loaded")
    p.add_argument("--dataset", required=True, help="release root; only pairs/val.jsonl is read")
    p.add_argument("--train-materialized-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--architecture", choices=("original", "candidate_pair", "candidate_dual"), default="original")
    p.add_argument("--training-mode", choices=("joint", "stage"), default="joint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--stop-after-epoch", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", type=int, choices=(32, 64, 128), nargs="?", const=128,
                   help="separate output, discard-only samples; no formal checkpoint/selection")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
