"""Independent S3 random-init M12+C8 trainer; never modifies the live queue.

Example (remote Linux CUDA only; source checkpoint supplies metadata only):
  python -m experiments.rachel_n512_formal_30k.train_score_staged \
    --checkpoint /path/to/full24-or-candidate.pt --dataset /path/to/release \
    --train-materialized-manifest /path/to/train_e1_24k.json \
    --architecture candidate_pair --output /path/to/new_s3 --stop-after-epoch 12
Continue with the same arguments and --resume --stop-after-epoch 20.

Total budget is always20x24000 pairs. M1..12 and C13..20 actually use the
registered phase forward/loss and two lifetime optimizer groups. Checkpoints
have a dedicated S3 schema and must use load_staged_checkpoint, NOT the live
score loader. The paired epoch20 anchor and auxiliary13..20 VAL winners are
published through the independent S3 freeze module after the last commit.
Existing joint20 runs can be imported by that module without retraining.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.rachel_n512_formal_30k import score_design_stages as stages
from experiments.rachel_n512_formal_30k.train_joint_damage import build_random_model, state_digest
from experiments.rachel_n512_formal_30k.train_edge_weathering import (
    _cpu_model_state, _sha256, capture_rng_state, restore_rng_state)
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import (
    evaluate_pair_validation, save_json)
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from experiments.rachel_n512_formal_30k.train_score_design import (
    canonical_digest, learning_rate, run_lock, event)
from staging.pairwise_v0_2.models.rachel_candidate_score import (
    CandidateScoreConfig, RachelCandidateScore, build_score_model)
from staging.pairwise_v0_2.models.rachel_model_factory import model_metadata
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedWeatheredDataset
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader, WeatheringStatistics

SCHEMA = "rachel-score-staged-training/1"
CHECKPOINT_SCHEMA = "rachel-score-staged-checkpoint/1"
SEED, TRAIN_COUNT, VAL_COUNT = 260913, 24000, 3000
SEGMENT_SIZE, MICROBATCH, EFFECTIVE_BATCH, TOTAL_EPOCHS = 6000, 4, 16, 20


def segment_plan(architecture):
    return [dict(number=(epoch - 1) * 4 + part + 1, epoch=epoch,
        phase=asdict(stages.phase(epoch, architecture=architecture)), offset=part * SEGMENT_SIZE,
        count=SEGMENT_SIZE, global_start=(epoch - 1) * TRAIN_COUNT + part * SEGMENT_SIZE,
        global_stop=(epoch - 1) * TRAIN_COUNT + (part + 1) * SEGMENT_SIZE,
        epoch_complete=part == 3, learning_rate=learning_rate(epoch))
        for epoch in range(1, TOTAL_EPOCHS + 1) for part in range(4)]


def build_random_candidate(source, architecture):
    """Same base-then-head initialization as joint; no source weight consumption."""
    if architecture not in ("candidate_pair", "candidate_dual"):
        raise ValueError("S3 requires candidate_pair or candidate_dual")
    design = source.get("s3_model_metadata", source.get("score_design", {}))
    config = design.get("model_config", source.get("model_config"))
    candidate_config = design.get("candidate_config", source.get("candidate_config", {}))
    metadata = dict(model_kind="full", model_options={}, model_config=config,
                    loss_config=source["loss_config"], seam_loss_enabled=source.get("seam_loss_enabled", False))
    base, checked, loss_config, shared_digest = build_random_model(metadata, seed=SEED)
    model = RachelCandidateScore(base, CandidateScoreConfig(**candidate_config), architecture)
    return model, checked, loss_config, dict(initial_weights_sha256=state_digest(model),
        shared_base_initial_weights_sha256=shared_digest,
        initial_matcher_state_sha256=stages.matcher_state_digest(model))


def experiment_identity(args, source_path, model, loss_config, training, digests):
    val_manifest = Path(args.dataset).resolve() / "pairs" / "val.jsonl"
    return dict(schema_version=SCHEMA, schedule="staged", seed=SEED,
        score_design=args.architecture, training_mode="M12_C8",
        base_model_metadata=model_metadata(model.base_model), candidate_config=asdict(model.candidate_config),
        loss_config=asdict(loss_config), **digests,
        metadata_source_checkpoint_sha256=_sha256(source_path), source_weights_loaded=False,
        train_count=TRAIN_COUNT, train_split="train", train_manifest=str(training.manifest_path),
        train_manifest_sha256=_sha256(training.manifest_path), validation_count=VAL_COUNT,
        validation_manifest=str(val_manifest), validation_manifest_sha256=_sha256(val_manifest),
        validation_split="val", max_epochs=TOTAL_EPOCHS, segment_pairs=SEGMENT_SIZE,
        microbatch=MICROBATCH, effective_batch=EFFECTIVE_BATCH, optimizer="AdamW", weight_decay=1e-4,
        lr_by_epoch=[learning_rate(e) for e in range(1, TOTAL_EPOCHS + 1)],
        grad_clip_norm=5.0, precision="fp32", workers=args.workers,
        validation_every_epochs=1, candidate_correctness_weight=.5 if args.architecture == "candidate_dual" else 0.,
        candidate_correctness_tolerance_px=20., phase_protocol=stages.stage_protocol(args.architecture, base=loss_config),
        lifetime_optimizer_groups=["matcher", "classifier"], optimizer_reset_at_phase_transition=False,
        held_out_used_for_training_or_selection=False)


def checkpoint_metadata(model):
    return dict(s3_checkpoint_schema=CHECKPOINT_SCHEMA, s3_training_schema=SCHEMA,
        model_kind="score_s3_" + model.architecture, s3_model_metadata=model.metadata(),
        model_config=asdict(model.config), candidate_config=asdict(model.candidate_config))


def _receipt_context(identity):
    return dict(expected_train_manifest_sha256=identity["train_manifest_sha256"],
                expected_run_identity_sha256=canonical_digest(identity))


def validate_checkpoint_progress(checkpoint, identity):
    if (checkpoint.get("s3_checkpoint_schema") != CHECKPOINT_SCHEMA or
            checkpoint.get("s3_training_schema") != SCHEMA or "score_design_schema" in checkpoint):
        raise ValueError("not an S3 checkpoint; live checkpoint is not resumable here")
    if checkpoint.get("resume_identity") != identity:
        old = checkpoint.get("resume_identity", {})
        changed = sorted(k for k in set(old) | set(identity) if old.get(k) != identity.get(k))
        raise ValueError("S3 resume identity differs: " + ", ".join(changed))
    completed = checkpoint.get("completed_segments")
    if type(completed) is not int or not 0 <= completed <= TOTAL_EPOCHS * 4:
        raise ValueError("invalid S3 committed segment")
    epoch = (completed + 3) // 4
    expected = dict(epoch=epoch, global_exposure=completed * SEGMENT_SIZE,
                    optimizer_updates=completed * SEGMENT_SIZE // EFFECTIVE_BATCH)
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError("S3 checkpoint exposure/update/epoch counts differ")
    spec = stages.phase(max(epoch, 1), architecture=identity["score_design"])
    if checkpoint.get("phase") != asdict(spec):
        raise ValueError("S3 checkpoint phase differs from its budget")
    if (completed >= stages.MATCHER_EPOCHS * 4) != bool(checkpoint.get("pretraining_receipt")):
        raise ValueError("committed M12 receipt presence differs from training progress")
    return spec


def load_staged_checkpoint(checkpoint):
    """Dedicated S3 inference loader; never mislabels these weights as live Full."""
    identity = checkpoint.get("resume_identity", {})
    spec = validate_checkpoint_progress(checkpoint, identity)
    metadata = checkpoint.get("s3_model_metadata", {})
    architecture = metadata.get("architecture")
    if (architecture != identity.get("score_design") or
            checkpoint.get("model_kind") != "score_s3_" + str(architecture)):
        raise ValueError("S3 architecture metadata differs")
    if (metadata.get("model_config") != identity["base_model_metadata"]["model_config"] or
            metadata.get("candidate_config") != identity["candidate_config"] or
            checkpoint.get("model_config") != metadata["model_config"] or
            checkpoint.get("candidate_config") != metadata["candidate_config"] or
            metadata.get("training_mode") != ("head_only" if spec.name == "C" else "joint")):
        raise ValueError("S3 model configuration/mode differs from its frozen identity")
    model = build_score_model(metadata["model_config"], architecture, metadata["candidate_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    receipt = checkpoint.get("pretraining_receipt")
    if receipt:
        stages.verify_pretraining_receipt(model, receipt, **_receipt_context(identity))
    stages.configure_phase(model, spec, training=False, receipt=receipt, **_receipt_context(identity))
    return model


def restore_training_state(model, optimizer, checkpoint, identity):
    spec = validate_checkpoint_progress(checkpoint, identity)
    if "optimizer_state_dict" not in checkpoint or "rng_state" not in checkpoint:
        raise ValueError("S3 resume requires optimizer and RNG states")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if [g.get("phase_family") for g in optimizer.param_groups] != ["matcher", "classifier"]:
        raise ValueError("S3 optimizer must retain both lifetime groups")
    receipt = checkpoint.get("pretraining_receipt")
    if receipt:
        stages.verify_pretraining_receipt(model, receipt, **_receipt_context(identity))
    stages.configure_phase(model, spec, receipt=receipt, **_receipt_context(identity))
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint["completed_segments"], receipt


def payload(model, optimizer, *, identity, loss_config, data_record, completed, receipt, role):
    epoch = (completed + 3) // 4
    return dict(**checkpoint_metadata(model), model_state_dict=_cpu_model_state(model),
        optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state(),
        resume_identity=identity, loss_config=asdict(loss_config),
        completed_segments=completed, epoch=epoch, phase=asdict(stages.phase(max(epoch, 1), architecture=model.architecture)),
        global_exposure=completed * SEGMENT_SIZE, optimizer_updates=completed * SEGMENT_SIZE // EFFECTIVE_BATCH,
        pretraining_receipt=receipt, training_data=data_record, checkpoint_role=role,
        seed=SEED, initialization="random", source_weights_loaded=False, formal_training_counted=True,
        initial_weights_sha256=identity["initial_weights_sha256"],
        shared_base_initial_weights_sha256=identity["shared_base_initial_weights_sha256"],
        seam_loss_enabled=False, resample_contour_cap=None, precision="fp32")


def train_segment(model, loader, optimizer, loss_config, device, args, spec, *, global_exposure=0,
                  optimizer_updates=0):
    """Actual phase forward/loss; CPU is only usable by tiny unit fixtures."""
    optimizer.zero_grad(set_to_none=True)
    accumulation, samples, updates = EFFECTIVE_BATCH // MICROBATCH, 0, 0
    statistics, started = WeatheringStatistics(), time.perf_counter()
    names = ("fused_pair_bce", "coarse_pair_bce", "local_pair_bce", "assignment_nll",
             "translation_smooth_l1", "sinkhorn_residual")
    loss_sums = {key: 0.0 for key in ("total", "phase_base_total", "candidate_correctness_bce") + names}
    counters = {}
    profile = stages.phase_loss_profile(spec, loss_config)[1]
    for step, wrapped in enumerate(loader):
        group_start = step // accumulation * accumulation
        group_samples = min(EFFECTIVE_BATCH, len(loader.dataset) - group_start * MICROBATCH)
        inputs, targets = runner._full_batch(wrapped.batch, device)
        output = stages.forward_for_phase(model, spec, *inputs)
        pose = torch.as_tensor(wrapped.pose_supervision_enabled, dtype=torch.bool, device=device)
        result = stages.compute_phase_loss(spec, output, *targets, pose_supervision_enabled=pose,
                                           base=loss_config, split="train")
        count = len(wrapped.batch.pair_ids)
        (result.total * (count / group_samples)).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        samples += count
        values = torch.stack((result.total.detach(), result.base_loss.total.detach(), result.candidate_correctness_bce.detach())
            + tuple(getattr(result.base_loss, key).detach() for key in names)).cpu().tolist()
        for key, value in zip(loss_sums, values):
            loss_sums[key] += float(value) * count
        for key, value in result.counts.items():
            counters[key] = counters.get(key, 0) + int(value.item())
        statistics.add(wrapped)
        if (step + 1) % args.log_every == 0:
            progress = dict(event="train_progress", phase=spec.name, epoch=spec.epoch,
                samples=samples, updates=updates, loss_components={k: v / samples for k, v in loss_sums.items()},
                supervision_counts=counters, elapsed_s=time.perf_counter() - started)
            event(Path(args.output), **progress)
            save_json(Path(args.output) / "status.json", dict(status="running", phase="train_" + spec.name,
                epoch=spec.epoch, global_exposure=global_exposure + samples,
                optimizer_updates=optimizer_updates + updates, progress=progress))
    return dict(samples=samples, optimizer_updates=updates, phase_profile=profile,
        loss_components={k: v / max(1, samples) for k, v in loss_sums.items()},
        supervision_counts=counters, weathering=statistics.report(), elapsed_s=time.perf_counter() - started,
        peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)


def _publish_completed_freeze(root, identity):
    from experiments.rachel_n512_formal_30k.freeze_score_staged_comparison import publish_s3_freeze
    return publish_s3_freeze(root, schedule="staged", expected_identity=identity)


def run(args):
    stop = args.stop_after_epoch
    if not 1 <= stop <= TOTAL_EPOCHS or args.workers < 0 or args.log_every <= 0:
        raise ValueError("invalid operational stop/workers/log interval")
    if args.resume and args.smoke:
        raise ValueError("smoke is discard-only and cannot resume a formal run")
    if platform.system() != "Linux" or not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("S3 training requires the remote Linux CUDA server")
    torch.set_num_threads(1)
    source_path = Path(args.checkpoint).resolve(strict=True)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    model, _, loss_config, digests = build_random_candidate(source, args.architecture)
    del source
    training = MaterializedWeatheredDataset(args.train_materialized_manifest)
    validation = RachelPairDataset(args.dataset, "val")
    if len(training) != TRAIN_COUNT or len(validation) != VAL_COUNT:
        raise ValueError("S3 requires fixed materialized TRAIN24000 and cleanVAL3000")
    identity = experiment_identity(args, source_path, model, loss_config, training, digests)
    data_record = dict(kind="fixed_e1_materialized", unique_count=len(training),
        manifest=str(training.manifest_path), stats=training.stats, protocol=training.protocol)
    root = Path(args.output).resolve()
    if args.resume:
        if not root.is_dir():
            raise ValueError("resume output directory does not exist")
    else:
        root.mkdir(parents=True, exist_ok=False)
    args.output = str(root)
    with run_lock(root):
        return _run_locked(args, root, model, training, validation, loss_config, identity, data_record)


def _run_locked(args, root, model, training, validation, loss_config, identity, data_record):
    device = torch.device(args.device)
    model = model.to(device)
    stages.configure_phase(model, stages.phase(1, architecture=args.architecture))
    optimizer = torch.optim.AdamW(stages.optimizer_parameter_groups(model), lr=learning_rate(1), weight_decay=1e-4)
    completed, receipt = 0, None
    if args.resume:
        checkpoint = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        completed, receipt = restore_training_state(model, optimizer, checkpoint, identity)
        del checkpoint
        if completed > args.stop_after_epoch * 4:
            raise ValueError("requested stop precedes committed progress")
    protocol = dict(**identity, status="running", arguments=vars(args),
        requested_stop_after_epoch=args.stop_after_epoch, training_data=data_record,
        plan=segment_plan(args.architecture), smoke=bool(args.smoke),
        formal_training_counted=not bool(args.smoke), implementation_sha256=_sha256(__file__),
        runtime=dict(torch=torch.__version__, python=platform.python_version(), cuda=torch.version.cuda))
    save_json(root / "protocol.json", protocol)
    if not args.resume and not args.smoke:
        runner._atomic_torch_save(root / "last.pt", payload(model, optimizer, identity=identity,
            loss_config=loss_config, data_record=data_record, completed=0, receipt=None, role="initial_recovery"))
    if receipt:
        save_json(root / "matcher_pretraining_receipt.json", receipt)  # Repair interrupted pointer publication.
    event(root, event="resume" if args.resume else "start", completed_segments=completed,
          stop_after_epoch=args.stop_after_epoch, schema=SCHEMA)
    started = time.monotonic()
    try:
        for segment in segment_plan(args.architecture):
            number, epoch = segment["number"], segment["epoch"]
            if number <= completed:
                continue
            if epoch > args.stop_after_epoch:
                break
            spec = stages.phase(epoch, architecture=args.architecture)
            report = stages.configure_phase(model, spec, receipt=receipt, **_receipt_context(identity))
            if number % 4 == 1:
                event(root, event="epoch_phase_configuration", **report)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch)
            order = runner.epoch_indices(TRAIN_COUNT, seed=SEED, epoch=epoch, limit=None)
            count = args.smoke or SEGMENT_SIZE
            order = order[segment["offset"]:segment["offset"] + count]
            loader = make_weathering_loader(training, order, batch_size=MICROBATCH,
                num_workers=args.workers, seed=SEED + number, contour_cap=512)
            save_json(root / "status.json", dict(status="running", pid=os.getpid(), phase="train_" + spec.name,
                epoch=epoch, segment=number, global_exposure=completed * SEGMENT_SIZE,
                optimizer_updates=completed * SEGMENT_SIZE // EFFECTIVE_BATCH))
            torch.cuda.reset_peak_memory_stats(device)
            training_report = train_segment(model, loader, optimizer, loss_config, device, args, spec,
                global_exposure=completed * SEGMENT_SIZE, optimizer_updates=completed * SEGMENT_SIZE // EFFECTIVE_BATCH)
            if training_report["samples"] != count or training_report["optimizer_updates"] != count // EFFECTIVE_BATCH:
                raise RuntimeError("S3 segment exposure/update count differs")
            save_json(root / ("segment_%03d.json" % number), dict(segment=segment, training=training_report))
            if args.smoke:
                result = dict(status="smoke_complete", smoke=True, formal_training_counted=False,
                    weights_discarded=True, training=training_report)
                save_json(root / "smoke.json", result)
                save_json(root / "status.json", result)
                protocol.update(result)
                save_json(root / "protocol.json", protocol)
                return result
            if segment["epoch_complete"]:
                from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
                save_json(root / "status.json", dict(status="running", phase="validation", epoch=epoch,
                    training_phase=spec.name, global_exposure=number * SEGMENT_SIZE))
                val_loader = make_ablation_loader(validation, list(range(VAL_COUNT)), batch_size=8,
                    num_workers=args.workers, seed=SEED, contour_cap=512)
                # Target-blind pair inference; M scores are explicitly diagnostic.
                val_report, rows = evaluate_pair_validation(model, val_loader, device)
                points = fit_operating_points([r["label"] for r in rows], [r["classification"]["fused"] for r in rows])
                save_json(root / ("validation_%03d_rows.json" % epoch), rows)
                save_json(root / ("validation_%03d.json" % epoch), dict(epoch=epoch, phase=asdict(spec),
                    global_exposure=number * SEGMENT_SIZE, validation=val_report, operating_points=points,
                    selection_eligible=epoch >= 13, diagnostic_only=epoch <= 12))
                event(root, event="validation_complete", epoch=epoch, training_phase=spec.name,
                      selection_eligible=epoch >= 13, validation=val_report)
                if epoch == stages.MATCHER_EPOCHS:
                    receipt = stages.matcher_pretraining_receipt(model,
                        initial_matcher_sha256=identity["initial_matcher_state_sha256"],
                        completed_epochs=epoch, pair_exposures=number * SEGMENT_SIZE,
                        optimizer_updates=number * SEGMENT_SIZE // EFFECTIVE_BATCH,
                        train_manifest_sha256=identity["train_manifest_sha256"],
                        run_identity_sha256=canonical_digest(identity))
            if spec.name == "C":
                stages.verify_pretraining_receipt(model, receipt, **_receipt_context(identity))
            checkpoint = payload(model, optimizer, identity=identity, loss_config=loss_config,
                data_record=data_record, completed=number, receipt=receipt,
                role="epoch_anchor" if segment["epoch_complete"] else "recovery")
            if segment["epoch_complete"]:
                runner._atomic_torch_save(root / ("epoch_%03d.pt" % epoch), checkpoint)
            runner._atomic_torch_save(root / "last.pt", checkpoint)
            completed = number
            del checkpoint
            if receipt:
                save_json(root / "matcher_pretraining_receipt.json", receipt)
            event(root, event="segment_committed", segment=number, epoch=epoch, training_phase=spec.name,
                  global_exposure=completed * SEGMENT_SIZE)
        if completed != args.stop_after_epoch * 4:
            raise RuntimeError("S3 stopped before the requested complete-epoch budget")
        if completed == TOTAL_EPOCHS * 4:
            _publish_completed_freeze(root, identity)
        status = "complete" if args.stop_after_epoch == TOTAL_EPOCHS else "budget_complete"
        result = dict(status=status, pid=os.getpid(), epoch=args.stop_after_epoch,
            completed_segments=completed, global_exposure=completed * SEGMENT_SIZE,
            optimizer_updates=completed * SEGMENT_SIZE // EFFECTIVE_BATCH,
            pretraining_complete=receipt is not None, phase="train_val_complete",
            can_resume_to_epoch=TOTAL_EPOCHS if args.stop_after_epoch < TOTAL_EPOCHS else None,
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
    p.add_argument("--checkpoint", required=True, help="architecture/loss metadata only; never source weights")
    p.add_argument("--dataset", required=True)
    p.add_argument("--train-materialized-manifest", required=True)
    p.add_argument("--architecture", required=True, choices=("candidate_pair", "candidate_dual"))
    p.add_argument("--output", required=True)
    p.add_argument("--stop-after-epoch", type=int, default=TOTAL_EPOCHS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--smoke", type=int, nargs="?", const=32, choices=(32, 64, 128))
    return p


if __name__ == "__main__":
    run(parser().parse_args())
