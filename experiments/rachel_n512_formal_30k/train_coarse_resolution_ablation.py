"""Single-factor coarse256/512 continuation of the four-scale N512 candidate.

TRAIN24k x 5 epochs, seed260909, FP32 microbatch4/effective16, AdamW2e-5,
and the original loss exactly match Stage2 original24k. Each run starts from
the pinned coarse128 candidate, never from a Stage2 winner. Only coarse_size
changes; all parameter tensors are loaded strictly without resetting a head.
Selection is five original VAL3000 fused row-F1 events (AP tie-break), no pose S.
No TEST/REAL inputs are opened by this command. This module does not launch
jobs automatically. Use evaluate_coarse_resolution_ablation only after both
resolution arms have completed their independent validation freezes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from experiments.rachel_n512_formal_30k import train_realism_data_ablation as base
from staging.pairwise_v0_2.models.rachel_model_factory import (
    build_rachel_model, load_rachel_checkpoint, model_metadata,
)
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


SOURCE_CHECKPOINT_SHA256 = "350bf95413f828698f3396294a317569b013891d0906e2d4212d046fba059233"
SCHEMA = "rachel-four-scale-coarse-resolution/1"
SELECTION_RULE = "maximize VAL fused equal-row F1; AP breaks ties; earliest exposure breaks exact ties; require full decision coverage"


def same_json(left, right):
    return json.dumps(base.clean(left), sort_keys=True) == json.dumps(base.clean(right), sort_keys=True)


def build_resolution_model(checkpoint, coarse_size):
    if coarse_size not in (256, 512):
        raise ValueError("only coarse256/coarse512 are authorized; reuse Stage2 original24k for128")
    source = load_rachel_checkpoint(checkpoint)  # strict, including every source tensor
    initial = base.check_fixed_architecture(source, checkpoint)
    config = dict(initial["model_config"], coarse_size=coarse_size)
    model = build_rachel_model("full", config)
    model.load_state_dict(source.state_dict(), strict=True)
    return model, dict(initial_model_metadata=initial,
                       effective_model_metadata=model_metadata(model),
                       changed_config={"coarse_size": {"before": 128, "after": coarse_size}},
                       parameter_loading="strict source and strict resized model; no tensors reset or filtered")


def check_resolution_architecture(model, checkpoint):
    """Restricted external-evaluation validator; never relax the data-arm check."""
    metadata = model_metadata(model)
    if model.config.coarse_size not in (256, 512):
        raise ValueError("resolution winner must use coarse256 or coarse512")
    initial = checkpoint.get("initial_model_metadata", {})
    expected = dict(initial.get("model_config", {}), coarse_size=model.config.coarse_size)
    if (initial.get("model_kind") != "full" or initial.get("model_options") != {}
            or initial.get("model_config", {}).get("coarse_size") != 128
            or metadata["model_kind"] != "full" or metadata["model_options"]
            or not same_json(metadata["model_config"], expected)):
        raise ValueError("winner must differ from its original Full architecture only in coarse_size")
    c = model.config
    if ((c.canvas_size, c.contour_cap, c.patch_size) != (800, 512, 16)
            or tuple(c.window_sizes_px) != (7.0, 16.0, 32.0, 64.0)):
        raise ValueError("fixed four-scale/N512/800canvas/patch16 architecture required")
    if (checkpoint.get("initial_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or checkpoint.get("seam_loss_enabled", False)
            or "loss_config" not in checkpoint
            or not same_json(checkpoint["loss_config"], checkpoint.get("source_loss_config"))):
        raise ValueError("winner must retain pinned initialization and unchanged original loss")
    return metadata


def check_control(control_run, source_path, original_root, initial, loss_config):
    """Read the already completed original24k receipt, not its external results."""
    root = Path(control_run).resolve(strict=True)
    protocol = json.loads((root / "protocol.json").read_text())
    freeze = json.loads((root / "train_val_freeze.json").read_text())
    data = protocol.get("training_data", {})
    if (protocol.get("status") != "complete" or freeze.get("status") != "complete"
            or protocol.get("arm") != "original24k"
            or data.get("kind") != "original_released_train" or data.get("sample_count") != 24000
            or Path(data.get("root", "")).resolve() != original_root
            or Path(protocol.get("initial_checkpoint", "")).resolve() != source_path
            or not same_json(protocol.get("initial_model_metadata"), initial)
            or not same_json(protocol.get("effective_loss_config"), loss_config)
            or protocol.get("seed") != base.SEED or protocol.get("learning_rate") != base.LEARNING_RATE
            or protocol.get("microbatch_size") != base.MICROBATCH
            or protocol.get("effective_batch_size") != base.EFFECTIVE_BATCH
            or protocol.get("completed_global_exposures") != 120000
            or protocol.get("completed_optimizer_updates") != 7500
            or protocol.get("completed_validation_events") != 5
            or protocol.get("selection_rule") != SELECTION_RULE):
        raise ValueError("original24k control receipt does not match the registered initialization/data/budget")
    return str(root)


def run(args):
    if args.workers < 0 or args.log_every <= 0:
        raise ValueError("workers must be nonnegative and log_every positive")
    if not base.torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("training may only be explicitly launched on the remote CUDA server")
    source_path, original_root = Path(args.checkpoint).resolve(), Path(args.dataset).resolve()
    if sealed._sha256_file(source_path) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("source checkpoint is not the pinned 350bf9 four-scale candidate")
    checkpoint = sealed._torch_load_checkpoint(source_path)
    model, change = build_resolution_model(checkpoint, args.coarse_size)
    loss_config = base.RachelN512LossConfig(**checkpoint["loss_config"])
    control = check_control(args.control_training_run, source_path, original_root,
                            change["initial_model_metadata"], asdict(loss_config))
    training, data_record = base.make_training_dataset(original_root)
    validation = base.RachelPairDataset(original_root, "val")
    if (len(training), len(validation)) != (24000, 3000):
        raise ValueError("resolution experiment requires original TRAIN24000 and VAL3000")
    plan = base.exposure_plan(len(training))
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    device = base.torch.device(args.device)
    base.torch.set_num_threads(1)
    base.runner._set_determinism(base.SEED)
    model = model.to(device).train().requires_grad_(True)
    optimizer = base.torch.optim.AdamW(model.parameters(), lr=base.LEARNING_RATE, weight_decay=1e-4)
    args.batch_size, args.effective_batch_size = base.MICROBATCH, base.EFFECTIVE_BATCH
    args.enable_seam_loss, args.seam_weight = False, 0.0
    arm = "coarse%d" % args.coarse_size
    provenance = dict(**change, initial_checkpoint=str(source_path),
        initial_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256, initial_epoch=checkpoint.get("epoch"),
        control_training_run=control, source_loss_config=checkpoint["loss_config"],
        effective_loss_config=asdict(loss_config), initialization="warm-start", full_network_trainable=True,
        precision="fp32", seed=base.SEED, learning_rate=base.LEARNING_RATE,
        seam_loss_enabled=False, seam_loss_weight=0.0,
        microbatch_size=base.MICROBATCH, effective_batch_size=base.EFFECTIVE_BATCH,
        optimizer="AdamW", weight_decay=1e-4, gradient_clip_norm=5.0,
        training_data=data_record, unique_count=data_record["unique_count"],
        selection_rule=SELECTION_RULE, no_pose_composite_selection=True,
        threshold_rule="equal-row validation F1; largest threshold among ties",
        validation_root=str(original_root), validation_split="val", validation_sample_count=3000,
        original_validation_unchanged=True, test_or_real_used_for_fit=False)
    protocol = dict(schema_version=SCHEMA, status="running", arm=arm, **provenance,
        planned_global_exposures=120000, planned_optimizer_updates=7500, planned_dataset_epochs=5,
        validation_exposure_anchors=[24000, 48000, 72000, 96000, 120000], exposure_plan=plan,
        heldout_gate="both coarse256 and coarse512 must finish five VAL events before either external evaluation",
        arguments=vars(args))
    base.save_json(destination / "protocol.json", protocol)
    base.save_json(destination / "architecture.json", change["effective_model_metadata"])
    best_key, samples, updates, events, epoch = None, 0, 0, 0, 0
    started = time.perf_counter()
    try:
        for index, segment in enumerate(plan, 1):
            epoch = segment["epoch"]
            order = base.runner.epoch_indices(len(training), seed=base.SEED, epoch=epoch, limit=None)
            loader = base.make_ablation_loader(training, order, batch_size=base.MICROBATCH,
                num_workers=args.workers, seed=base.SEED + index, contour_cap=512)
            base.save_json(destination / "status.json", dict(status="running", phase="train", epoch=epoch,
                global_exposure=samples, optimizer_updates=updates))
            base.torch.cuda.reset_peak_memory_stats(device)
            trained = base.train_epoch(model, loader, optimizer, loss_config, device, args, epoch, arm)
            if (trained["samples"], trained["optimizer_updates"]) != (24000, 1500):
                raise RuntimeError("observed epoch differs from the matched exposure/update budget")
            samples += trained["samples"]
            updates += trained["optimizer_updates"]
            base.save_json(destination / ("segment_%02d.json" % index), dict(segment=index, epoch=epoch,
                global_exposure=samples, optimizer_updates=updates, training=trained))
            base.save_json(destination / "status.json", dict(status="running", phase="validation", epoch=epoch,
                global_exposure=samples, optimizer_updates=updates))
            val_loader = base.make_ablation_loader(validation, tuple(range(3000)), batch_size=base.MICROBATCH,
                num_workers=args.workers, seed=base.SEED, contour_cap=512)
            report, rows = base.evaluate_pair_validation(model, val_loader, device)
            if (report["sample_count"], report["positive_count"], report["negative_count"]) != (3000, 1500, 1500):
                raise RuntimeError("validation must cover every original balanced VAL row")
            events += 1
            result = dict(validation_event=events, epoch=epoch, global_exposure=samples,
                          optimizer_updates=updates, validation=report)
            base.save_json(destination / ("validation_%06d.json" % samples), result)
            base.emit(dict(event="validation_complete", arm=arm, **result))
            key = tuple(report["selection_key"])
            if report["decision_coverage"] == 1.0 and (best_key is None or key > best_key):
                best_key = key
                saved = dict(**provenance, **model_metadata(model), model_state_dict=model.state_dict(),
                    loss_config=asdict(loss_config), epoch=epoch, global_exposure=samples,
                    optimizer_updates=updates, validation_event=events, variant=arm,
                    resample_contour_cap=None)
                base.runner._atomic_torch_save(destination / "winner.pt", saved)
                base.save_json(destination / "winner_validation.json", rows)
                freeze = dict(status="provisional", selected_epoch=epoch, selected_global_exposure=samples,
                    selected_optimizer_updates=updates, selected_validation_event=events,
                    checkpoint=str(destination / "winner.pt"), unique_count=24000,
                    checkpoint_sha256=sealed._sha256_file(destination / "winner.pt"),
                    validation=report, classifier_thresholds=report["thresholds"],
                    selection_rule=SELECTION_RULE, initial_checkpoint=str(source_path),
                    initial_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256, seed=base.SEED, precision="fp32",
                    test_or_real_used_for_fit=False, original_validation_unchanged=True)
                base.save_json(destination / "train_val_freeze.json", freeze)
        if (samples, updates, events) != (120000, 7500, 5) or best_key is None:
            raise RuntimeError("no complete matched-budget validation winner")
        completion = dict(completed_global_exposures=samples, completed_optimizer_updates=updates,
                          completed_validation_events=events, completed_dataset_epochs=5)
        freeze.update(status="complete", **completion)
        base.save_json(destination / "train_val_freeze.json", freeze)
        protocol.update(status="complete", **completion, selected_epoch=freeze["selected_epoch"],
                        selected_global_exposure=freeze["selected_global_exposure"], elapsed_seconds=time.perf_counter()-started)
        base.save_json(destination / "protocol.json", protocol)
        base.save_json(destination / "status.json", dict(status="complete", **completion))
    except Exception as error:
        base.save_json(destination / "status.json", dict(status="failed", epoch=epoch,
            global_exposure=samples, optimizer_updates=updates, error=repr(error)))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="pinned 350bf9 original four-scale/coarse128 candidate")
    p.add_argument("--dataset", required=True, help="original released TRAIN24k and VAL3k; no composite manifest")
    p.add_argument("--control-training-run", required=True, help="completed Stage2 original24k training directory")
    p.add_argument("--coarse-size", type=int, choices=(256, 512), required=True)
    p.add_argument("--output", required=True, help="new coarse256 or coarse512 directory under a shared training root")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
