"""Scratch Full matcher: staged versus joint order of matched damage exposures.

Six TRAIN24k epochs, 144k exposures, 9000 AdamW updates, six clean VAL3000
events. Each pair sees clean three times plus fixed E1 weather epochs 1/2/3.
Both arms share random initialization and epoch-index order. Their only data
difference is temporal order. LR is 1e-4 for epochs 1..3, then 2e-5 for 4..6.
--checkpoint supplies architecture/loss metadata ONLY: no source weights load.
--resume restores current model/optimizer/RNG at a completed VAL boundary.
--smoke=128/8 discards weights and never produces a held-out-evaluable freeze.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from experiments.rachel_n512_formal_30k.train_realism_data_ablation import (
    MICROBATCH, EFFECTIVE_BATCH, check_fixed_architecture, emit, evaluate_pair_validation,
    exposure_plan, make_training_dataset, save_json)
from experiments.rachel_n512_formal_30k.train_edge_weathering import (
    _sha256, _cpu_model_state, capture_rng_state, restore_rng_state, validate_resume)
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from staging.pairwise_v0_2.models.rachel_model_factory import build_rachel_model, model_metadata
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.pairwise_data.rachel_staged_damage_dataset import (
    StagedDamageDataset, schedule_audit)
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from staging.pairwise_v0_2.training.rachel_weathering_training import (
    make_weathering_loader, train_weathering_epoch)

SCHEMA = "rachel-joint-damage-training/1"
SEED = 260910
TRAIN_COUNT, VAL_COUNT, EPOCHS = 24000, 3000, 6
TOTAL_EXPOSURES = TRAIN_COUNT * EPOCHS
LR_SCHEDULE = {1: 1e-4, 2: 1e-4, 3: 1e-4, 4: 2e-5, 5: 2e-5, 6: 2e-5}
SELECTION_RULE = "maximize clean VAL fused equal-row F1; AP breaks ties; earliest exposure breaks exact ties; require full decision coverage"


def learning_rate(epoch):
    if type(epoch) is not int or epoch not in LR_SCHEDULE:
        raise ValueError("learning-rate epoch must be in 1..6")
    return LR_SCHEDULE[epoch]


def state_digest(model):
    """Canonical state tensor digest, independent of torch.save container IDs."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_random_model(source, *, seed=SEED):
    """Read metadata only; deliberately never consume source model_state_dict."""
    runner._set_determinism(seed)
    metadata = dict(model_kind=source.get("model_kind", "full"),
                    model_config=source["model_config"], model_options=source.get("model_options", {}))
    model = build_rachel_model(**metadata)
    architecture = check_fixed_architecture(model, source)
    loss = RachelN512LossConfig(**source["loss_config"])
    return model, architecture, loss, state_digest(model)


def parse_smoke(value):
    try:
        samples, updates = str(value).split("/") if "/" in str(value) else (str(value), None)
        samples = int(samples)
        updates = samples // EFFECTIVE_BATCH if updates is None else int(updates)
        if samples <= 0 or samples > TRAIN_COUNT or samples % EFFECTIVE_BATCH or updates != samples // EFFECTIVE_BATCH:
            raise ValueError
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("smoke must be samples/updates, e.g. 128/8, on full effective16 batches")
    return samples, updates


def checkpoint_payload(model, loss_config, data_record, protocol, epoch, samples, updates, events):
    return dict(model_state_dict=_cpu_model_state(model), **model_metadata(model),
        loss_config=asdict(loss_config), source_loss_config=protocol["source_loss_config"],
        epoch=epoch, global_exposure=samples, optimizer_updates=updates, validation_event=events,
        variant=protocol["arm"], initialization="random", initial_checkpoint=None, initial_epoch=None,
        initial_weights_sha256=protocol["initial_weights_sha256"],
        metadata_source_checkpoint=protocol["metadata_source_checkpoint"],
        metadata_source_checkpoint_sha256=protocol["metadata_source_checkpoint_sha256"],
        source_weights_loaded=False, seed=SEED, precision="fp32", resample_contour_cap=None,
        seam_loss_enabled=False, seam_loss_weight=0., unique_count=data_record["unique_count"],
        training_data=data_record, selection_rule=SELECTION_RULE,
        weathering_policy=protocol["weathering_policy"], shared_learning_rate_schedule=LR_SCHEDULE,
        optimizer_state_reset_at_phase_boundary=False)


def run(args, *, dataset_factory=None, extra_policy=None):
    """Optional new-arm adapter supplies factory(base, arm, seed, cache_dir)."""
    if args.workers < 0 or args.log_every <= 0:
        raise ValueError("workers must be nonnegative and log_every positive")
    if args.resume and args.smoke:
        raise ValueError("discard-only smoke cannot resume a formal run")
    if args.arm.startswith("partial_") and dataset_factory is None:
        from experiments.rachel_n512_formal_30k.train_partial_seam import read_pair_metadata
        from staging.pairwise_v0_2.pairwise_data.rachel_staged_damage_dataset import GuidedStagedDamageDataset
        if not args.outline_bank:
            raise ValueError("partial controls require a TRAIN-only outline bank")
        probability = .2 if args.arm == "partial_low" else 1.
        dataset_factory = partial(GuidedStagedDamageDataset,
            pair_metadata=read_pair_metadata(args.train_manifest), bank=args.outline_bank,
            probability=probability)
        extra_policy = dict(partial_seam_enabled=True, partial_probability_within_weather_slots=probability,
            partial_max_attempts=12, partial_geometry="oblique TRAIN contour profile; source-seam quantile offset",
            partial_clean_slots_untouched=True, partial_geometry_epochs=[1, 2, 3],
            partial_topology_connectivity=8,
            positive_negative_joint_acceptance=True, hard_negative_overlay_used=False,
            partial_actual_coverage_reported_from_training=True,
            original_gt_translation_preserved=True,
            artificial_cut_edge_exclusion_px=8., minimum_supervised_seam_px=32.,
            minimum_matched_tokens=4, physical_source_seam_retention=[.25, .75],
            outline_bank=str(Path(args.outline_bank).resolve()))
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("full-network training/smoke must run explicitly on the remote CUDA server")
    torch.set_num_threads(1)
    source_path, manifest_path, root = map(lambda p: Path(p).resolve(), (args.checkpoint, args.train_manifest, args.dataset))
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    model, architecture, loss_config, initial_digest = build_random_model(source)
    # Do not retain warm weights, optimizer, epoch, or RNG in any new artifact.
    source_loss_config = dict(source["loss_config"])
    del source
    base, data_record = make_training_dataset(root, manifest_path)
    validation = RachelPairDataset(root, "val")
    if (len(base), data_record["unique_count"], len(validation)) != (TRAIN_COUNT, TRAIN_COUNT, VAL_COUNT):
        raise ValueError("requires unchanged unique matched TRAIN24k and clean VAL3000")
    manifest = json.loads(manifest_path.read_text())
    pair_ids = [entry["row"]["pair_id"] for entry in manifest["entries"]]
    schedule_arm = "joint" if args.arm.startswith("partial_") else args.arm
    audit = schedule_audit(pair_ids, arm=schedule_arm, seed=SEED)
    if audit["total_slots"] != {"0": 72000, "1": 24000, "2": 24000, "3": 24000}:
        raise RuntimeError("six-exposure multiset budget differs")
    policy = dict(schedule=audit, requested_endpoint_tier_probabilities_within_weather_slot=dict(clean=.70, mild=.25, moderate=.05),
        weather_variant_epochs=[1, 2, 3], weather_variant_seed=SEED, weather_variants_independent_of_training_epoch=True,
        correspondence_policy="E1 source-arc inherited reciprocal assignments; no geometric relabeling",
        translation_loss_policy="original pose preserved; E1 translation auxiliary only on unchanged positives",
        pair_label_policy="unchanged source label with guarded E1 fallbacks",
        clean_slots_are_exact_original_samples=True, label_used_for_schedule=False, validation_weathered=False,
        augmentation_caveat="E1 weather slots include requested clean endpoints and guarded fallbacks; 3 weather exposures do not mean 3 applied corrosions")
    if extra_policy:
        policy.update(json.loads(json.dumps(extra_policy, allow_nan=False)))
    identity = dict(schema_version=SCHEMA, arm=args.arm, seed=SEED, initialization="random",
        initial_weights_sha256=initial_digest, metadata_source_checkpoint=str(source_path),
        metadata_source_checkpoint_sha256=_sha256(source_path), train_manifest=str(manifest_path),
        train_manifest_sha256=_sha256(manifest_path), dataset_root=str(root), architecture=architecture,
        source_loss_config=source_loss_config, train_count=TRAIN_COUNT, total_exposures=TOTAL_EXPOSURES,
        validation_interval=TRAIN_COUNT, microbatch=MICROBATCH, effective_batch=EFFECTIVE_BATCH,
        shared_learning_rate_schedule=LR_SCHEDULE, weight_decay=1e-4, precision="fp32", policy=policy)
    plan = exposure_plan(TRAIN_COUNT, total=TOTAL_EXPOSURES, validation_interval=TRAIN_COUNT)
    destination = Path(args.output).resolve()
    if args.resume:
        last = torch.load(destination / "last.pt", map_location="cpu", weights_only=False)
        validate_resume(last, identity, plan)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        last = None
    args.batch_size, args.effective_batch_size = MICROBATCH, EFFECTIVE_BATCH
    args.enable_seam_loss, args.seam_weight = False, 0.
    args.output, args.status_path = str(destination), str(destination / "status.json")
    device = torch.device(args.device)
    model = model.to(device).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(1), weight_decay=1e-4)
    factory = dataset_factory or StagedDamageDataset
    training = factory(base, arm=schedule_arm, seed=SEED, cache_dir=args.cache_dir)
    protocol = dict(schema_version=SCHEMA, status="running", arm=args.arm, initialization="random",
        initial_checkpoint=None, initial_epoch=None, initial_weights_sha256=initial_digest,
        metadata_source_checkpoint=str(source_path), metadata_source_checkpoint_sha256=identity["metadata_source_checkpoint_sha256"],
        source_weights_loaded=False, initial_model_metadata=architecture, source_loss_config=source_loss_config,
        effective_loss_config=asdict(loss_config), full_network_trainable=True, optimizer="AdamW",
        shared_learning_rate_schedule=LR_SCHEDULE, optimizer_state_reset_at_phase_boundary=False,
        weight_decay=1e-4, gradient_clip_norm=5., precision="fp32", seed=SEED,
        microbatch_size=MICROBATCH, effective_batch_size=EFFECTIVE_BATCH,
        training_data=data_record, unique_count=TRAIN_COUNT, weathering_policy=policy,
        planned_global_exposures=TOTAL_EXPOSURES, planned_optimizer_updates=9000,
        planned_dataset_epochs=EPOCHS, validation_root=str(root), validation_split="val", validation_sample_count=VAL_COUNT,
        validation_exposure_anchors=[TRAIN_COUNT * e for e in range(1, 7)],
        selection_rule=SELECTION_RULE, threshold_rule="equal-row clean validation F1; largest threshold among ties",
        no_pose_composite_selection=True, test_used_for_selection=False, real_used_for_selection=False,
        exposure_plan=plan, arguments=vars(args), resume_identity=identity,
        recovery_rule="current model+optimizer+RNG from last.pt completed VAL only; never restore winner to continue training",
        accumulation_caveat="micro4/effective16; auxiliary target-normalized losses may differ from a single batch16 loss",
        historical_comparison_caveat="old warm-start E1 is historical only, not equal total-training-budget evidence")
    if args.smoke:
        protocol.update(smoke=True, planned_global_exposures=args.smoke[0], planned_optimizer_updates=args.smoke[1],
            planned_dataset_epochs=0, validation_exposure_anchors=[], weights_discarded=True)
    save_json(destination / "protocol.json", protocol)
    save_json(destination / "architecture.json", architecture)
    save_json(destination / "schedule_audit.json", audit)
    samples = updates = events = completed = 0
    best_key = best_checkpoint = best_rows = freeze = None
    if last is not None:
        model.load_state_dict(last["model_state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state_dict"])
        samples, updates, events, completed = (last[k] for k in ("global_exposure", "optimizer_updates", "validation_event", "completed_segments"))
        best_key = tuple(last["best_key"]) if last["best_key"] is not None else None
        best_checkpoint, best_rows, freeze = (last[k] for k in ("winner_checkpoint", "winner_validation_rows", "winner_freeze"))
        if best_checkpoint is not None:
            runner._atomic_torch_save(destination / "winner.pt", best_checkpoint)
            save_json(destination / "winner_validation.json", best_rows)
            save_json(destination / "train_val_freeze.json", freeze)
        restore_rng_state(last["rng_state"])
    started, epoch = time.perf_counter(), None
    try:
        if args.smoke:
            epoch = args.smoke_epoch
            training.set_epoch(epoch)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch)
            order = runner.epoch_indices(len(training), seed=SEED, epoch=epoch, limit=args.smoke[0])
            loader = make_weathering_loader(training, order, batch_size=MICROBATCH, num_workers=args.workers,
                seed=SEED + epoch, contour_cap=model.config.contour_cap)
            report = train_weathering_epoch(model, loader, optimizer, loss_config, device, args, epoch, args.arm)
            if (report["samples"], report["optimizer_updates"]) != args.smoke:
                raise RuntimeError("smoke exposure/update budget mismatch")
            result = dict(status="complete", smoke=True, weights_discarded=True, formal_training_budget_counted=False,
                epoch=epoch, training=report, initial_weights_sha256=initial_digest)
            save_json(destination / "smoke.json", result)
            save_json(destination / "status.json", result)
            protocol.update(status="complete", smoke_result=result)
            save_json(destination / "protocol.json", protocol)
            return
        for segment_index, segment in enumerate(plan, 1):
            if segment_index <= completed:
                continue
            epoch = segment["epoch"]
            # Preserve the same optimizer object and moments across epoch 3->4.
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch)
            training.set_epoch(epoch)
            order = runner.epoch_indices(len(training), seed=SEED, epoch=epoch, limit=None)
            loader = make_weathering_loader(training, order, batch_size=MICROBATCH, num_workers=args.workers,
                seed=SEED + segment_index, contour_cap=model.config.contour_cap)
            save_json(destination / "status.json", dict(status="running", phase="train", epoch=epoch,
                global_exposure=samples, optimizer_updates=updates, learning_rate=learning_rate(epoch)))
            torch.cuda.reset_peak_memory_stats(device)
            report = train_weathering_epoch(model, loader, optimizer, loss_config, device, args, epoch, args.arm)
            if (report["samples"], report["optimizer_updates"]) != (TRAIN_COUNT, 1500) or not isinstance(report.get("weathering"), dict):
                raise RuntimeError("observed epoch budget/statistics mismatch")
            samples += report["samples"]
            updates += report["optimizer_updates"]
            result = dict(segment=segment_index, epoch=epoch, epoch_complete=True, epoch_slice=[0, TRAIN_COUNT],
                global_exposure=samples, optimizer_updates=updates, training=report,
                learning_rate=learning_rate(epoch), schedule=audit["per_epoch"][epoch - 1],
                pair_order_sha256=hashlib.sha256(json.dumps([pair_ids[i] for i in order], separators=(",", ":")).encode()).hexdigest())
            save_json(destination / ("segment_%02d.json" % segment_index), result)
            emit(dict(event="training_segment_complete", **result))
            save_json(destination / "status.json", dict(status="running", phase="validation", epoch=epoch,
                global_exposure=samples, optimizer_updates=updates))
            val_loader = make_ablation_loader(validation, tuple(range(len(validation))), batch_size=MICROBATCH,
                num_workers=args.workers, seed=SEED, contour_cap=model.config.contour_cap)
            val_report, rows = evaluate_pair_validation(model, val_loader, device)
            if tuple(val_report[k] for k in ("sample_count", "positive_count", "negative_count")) != (VAL_COUNT, 1500, 1500):
                raise RuntimeError("validation must cover unchanged balanced clean VAL3000")
            events += 1
            result = dict(validation_event=events, epoch=epoch, global_exposure=samples,
                optimizer_updates=updates, validation=val_report)
            save_json(destination / ("validation_%06d.json" % samples), result)
            emit(dict(event="validation_complete", **result))
            key = tuple(val_report["selection_key"])
            if val_report["decision_coverage"] == 1. and (best_key is None or key > best_key):
                best_key = key
                best_checkpoint = checkpoint_payload(model, loss_config, data_record, protocol, epoch, samples, updates, events)
                best_checkpoint["checkpoint_role"] = "val_selected_winner"
                best_rows = rows
                runner._atomic_torch_save(destination / "winner.pt", best_checkpoint)
                save_json(destination / "winner_validation.json", rows)
                freeze = dict(schema_version=SCHEMA, status="provisional", selected_epoch=epoch,
                    selected_global_exposure=samples, selected_optimizer_updates=updates, selected_validation_event=events,
                    unique_count=TRAIN_COUNT, checkpoint=str(destination / "winner.pt"), validation=val_report,
                    classifier_thresholds=val_report["thresholds"], selection_rule=SELECTION_RULE,
                    initialization="random", initial_checkpoint=None, initial_weights_sha256=initial_digest,
                    metadata_source_checkpoint=str(source_path), metadata_source_checkpoint_sha256=identity["metadata_source_checkpoint_sha256"],
                    seed=SEED, precision="fp32", test_or_real_used_for_fit=False, original_validation_unchanged=True,
                    weathering_policy=policy, shared_learning_rate_schedule=LR_SCHEDULE,
                    last_checkpoint_is_selection_candidate=False)
                save_json(destination / "train_val_freeze.json", freeze)
            recovery = checkpoint_payload(model, loss_config, data_record, protocol, epoch, samples, updates, events)
            recovery.update(checkpoint_role="last_recovery_only", completed_segments=segment_index,
                optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state(), resume_identity=identity,
                best_key=best_key, winner_checkpoint=best_checkpoint, winner_validation_rows=best_rows, winner_freeze=freeze)
            runner._atomic_torch_save(destination / "last.pt", recovery)
            completed = segment_index
        if (samples, updates, events) != (TOTAL_EXPOSURES, 9000, 6) or best_key is None:
            raise RuntimeError("formal run must complete 144k/9000/6 clean VAL with a full-coverage winner")
        totals = dict(completed_global_exposures=samples, completed_optimizer_updates=updates,
            completed_dataset_epochs=EPOCHS, completed_validation_events=events)
        freeze.update(status="complete", checkpoint_sha256=_sha256(destination / "winner.pt"), **totals)
        save_json(destination / "train_val_freeze.json", freeze)
        protocol.update(status="complete", selected_epoch=freeze["selected_epoch"],
            selected_global_exposure=freeze["selected_global_exposure"], elapsed_seconds_this_process=time.perf_counter() - started, **totals)
        save_json(destination / "protocol.json", protocol)
        save_json(destination / "status.json", dict(status="complete", epoch=6, global_exposure=samples,
            optimizer_updates=updates, selected_global_exposure=freeze["selected_global_exposure"]))
    except Exception as error:
        failure = dict(status="failed", epoch=epoch, global_exposure=samples, optimizer_updates=updates,
            completed_validation_events=events, resumable_completed_segments=completed,
            resumable_global_exposure=completed * TRAIN_COUNT, partial_segment_budget_counted=False,
            recovery_checkpoint=str(destination / "last.pt") if completed else None, error=repr(error))
        save_json(destination / "failure_state.json", failure)
        save_json(destination / "status.json", failure)
        protocol.update(status="failed", failure=failure)
        save_json(destination / "protocol.json", protocol)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=("staged", "joint", "partial_low", "partial_high"))
    p.add_argument("--outline-bank")
    p.add_argument("--checkpoint", required=True, help="architecture/loss metadata source only; never loads its weights")
    p.add_argument("--dataset", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", nargs="?", const=(128, 8), type=parse_smoke, help="discard-only samples/updates (default128/8)")
    p.add_argument("--smoke-epoch", type=int, choices=range(1, 7), default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    return p


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
