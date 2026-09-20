"""Equal-exposure data ablation, full-network warm start, TRAIN/VAL only.

The fixed protocol is 120,000 training pair exposures and 7,500 updates:
24k unique rows x 5 epochs versus 60k rows x 2 epochs. Both arms have exactly
five validation/selection opportunities, at 24k/48k/72k/96k/120k exposures.
Thus a 60k dataset epoch may be interrupted for validation. Each training
segment ends on a complete effective batch; epoch permutations are never
regenerated at those interruptions. Dataset epoch count is not the budget.

Example (run each arm with a separate, initially nonexistent output directory):
  python -m experiments.rachel_n512_formal_30k.train_realism_data_ablation \
      --checkpoint candidate/winner.pt --dataset /path/to/original_release \
      --train-manifest /path/to/composite_60000.json --output runs/realism60k

Omit --train-manifest for the original released 24k control. Composite manifests
use CompositeRachelPairDataset and retain its original RachelPairSample targets.
VAL always comes from --dataset/val, never from a composite source. No TEST/REAL
is opened here. Subsequent existing evaluation runners load winner.pt normally.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint, model_metadata
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from experiments.rachel_n512_formal_30k.run_architecture_ablation import train_epoch
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from experiments.rachel_n512_formal_30k.run_layout_decoder_experiment import clean, classification, fit_threshold


TOTAL_EXPOSURES = 120_000
VALIDATION_INTERVAL = 24_000
MICROBATCH = 4
EFFECTIVE_BATCH = 16
LEARNING_RATE = 2e-5
SEED = 260909


def emit(value):
    print(json.dumps(clean(value), ensure_ascii=False, allow_nan=False), flush=True)


def save_json(path, value):
    runner._atomic_json(Path(path), clean(value))


def exposure_plan(dataset_size, *, total=TOTAL_EXPOSURES,
                  validation_interval=VALIDATION_INTERVAL, effective_batch=EFFECTIVE_BATCH):
    """Generate disjoint within-epoch slices, with a shared global VAL clock."""
    if dataset_size <= 0 or total <= 0 or validation_interval <= 0 or effective_batch <= 0:
        raise ValueError("budget values must be positive")
    if total % dataset_size or total % validation_interval:
        raise ValueError("total exposure must comprise complete epochs and validation intervals")
    if any(n % effective_batch for n in (dataset_size, total, validation_interval)):
        raise ValueError("epoch, exposure budget, and VAL interval must end on complete effective batches")
    result = []
    global_exposure = 0
    for epoch in range(1, total // dataset_size + 1):
        offset = 0
        while offset < dataset_size:
            next_validation = (global_exposure // validation_interval + 1) * validation_interval
            length = min(dataset_size - offset, next_validation - global_exposure)
            end = global_exposure + length
            result.append(dict(epoch=epoch, epoch_start=offset, epoch_stop=offset + length,
                               sample_count=length, global_start=global_exposure, global_stop=end,
                               optimizer_updates=length // effective_batch,
                               validate_after=end % validation_interval == 0,
                               epoch_complete=offset + length == dataset_size))
            offset += length
            global_exposure = end
    return result


def check_fixed_architecture(model, checkpoint):
    metadata = model_metadata(model)
    c = model.config
    if metadata["model_kind"] != "full" or metadata["model_options"]:
        raise ValueError("data ablation requires unchanged Full descriptor-fusion candidate")
    if (c.canvas_size, c.coarse_size, c.contour_cap, c.patch_size) != (800, 128, 512, 16):
        raise ValueError("fixed candidate requires canvas800/coarse128/N512/patch16")
    if tuple(c.window_sizes_px) != (7.0, 16.0, 32.0, 64.0):
        raise ValueError("fixed candidate must use 7/16/32/64 descriptor-fused windows")
    if checkpoint.get("seam_loss_enabled", False):
        raise ValueError("this data-only ablation does not add/remove a seam training objective")
    if "loss_config" not in checkpoint:
        raise ValueError("source checkpoint must record the original loss config")
    return metadata


def make_training_dataset(original_root, manifest_path=None):
    """Use standard sample targets unchanged; read only lightweight provenance."""
    if manifest_path is None:
        dataset = RachelPairDataset(original_root, "train")
        return dataset, dict(kind="original_released_train", root=str(Path(original_root).resolve()),
                             sample_count=len(dataset), unique_count=len(dataset),
                             unique_count_definition="unique original released TRAIN pair IDs",
                             target_policy="original RachelPairDataset targets unchanged")
    from staging.pairwise_v0_2.pairwise_data.rachel_composite_training import CompositeRachelPairDataset
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != "rachel-composite-training/1" or manifest.get("split") != "train":
        raise ValueError("unsupported composite training manifest schema")
    entries = manifest["entries"]
    if any(entry["row"].get("split") != "train" for entry in entries):
        raise ValueError("composite training sources must all be TRAIN splits")
    if any(not Path(entry["source_root"]).is_absolute() for entry in entries):
        raise ValueError("composite source roots must be absolute")
    identities = {(str(Path(entry["source_root"]).resolve()),
                   tuple(sorted(entry["row"]["fragment_" + side]["fragment_token"] for side in "ab")))
                  for entry in entries}
    pair_ids = {entry["row"]["pair_id"] for entry in entries}
    if len(identities) != len(entries) or len(pair_ids) != len(entries):
        raise ValueError("data-volume arms require unique manifest pair IDs and source samples, not repeated-row upsampling")
    dataset = CompositeRachelPairDataset(manifest_path)
    if len(dataset) != len(entries):
        raise ValueError("composite loader size differs from manifest rows")
    return dataset, dict(kind="composite_training", manifest=str(manifest_path),
                         sample_count=len(dataset), unique_count=len(identities),
                         unique_pair_id_count=len(pair_ids),
                         unique_count_definition="unique (source_root, unordered fragment-token pair) TRAIN entries",
                         manifest_stats=manifest.get("stats", {}),
                         target_policy="original source RachelPairDataset targets unchanged")


def evaluate_pair_validation(model, loader, device):
    """Row-F1 calibration without cluster weights, pose decoding, or composite S."""
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            inputs, _ = runner._full_batch(batch, device)
            output = model(*inputs)
            scores = {name: getattr(output, name + "_probability").detach().cpu().numpy()
                      for name in ("coarse", "local", "fused")}
            if any(not np.isfinite(values).all() for values in scores.values()):
                raise ValueError("nonfinite validation pair scores")
            for index, pair_id in enumerate(batch.pair_ids):
                rows.append(dict(pair_id=pair_id, label=bool(batch.labels[index]),
                                 classification={name: float(values[index]) for name, values in scores.items()},
                                 decision_valid=bool(output.decision_valid[index].item())))
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("validation pair IDs are not unique")
    labels = np.asarray([row["label"] for row in rows], bool)
    if not labels.any() or labels.all():
        raise ValueError("VAL must contain both classes")
    thresholds, methods = {}, {}
    for branch in ("coarse", "local", "fused"):
        scores = np.asarray([row["classification"][branch] for row in rows])
        thresholds[branch] = fit_threshold(labels, scores)
        methods[branch] = classification(labels, scores, thresholds[branch])
    return dict(sample_count=len(rows), positive_count=int(labels.sum()), negative_count=int((~labels).sum()),
                methods=methods, thresholds=thresholds,
                decision_coverage=float(np.mean([row["decision_valid"] for row in rows])),
                selection_key=list(validation_selection_key(methods["fused"])),
                threshold_grain="equal pair rows; no cluster reweighting",
                pose_used_for_selection=False), rows


def validation_selection_key(fused_metrics):
    key = (float(fused_metrics["f1"]), float(fused_metrics["auprc"]))
    if not all(math.isfinite(value) for value in key):
        raise ValueError("selection F1/AP must be finite")
    return key


def run(args):
    if args.workers < 0 or args.log_every <= 0:
        raise ValueError("workers must be nonnegative and log_every positive")
    if args.arm == "seam_loss":
        raise ValueError("arm label seam_loss is reserved by the reused trainer; this is a data-only ablation")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("full-network training must be explicitly run on the remote CUDA server")
    original_root = Path(args.dataset).resolve()
    source_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    # No filtered/non-strict loading, sampler regeneration, or config override.
    model = load_rachel_checkpoint(checkpoint)
    architecture = check_fixed_architecture(model, checkpoint)
    loss_config = RachelN512LossConfig(**checkpoint["loss_config"])
    training, data_record = make_training_dataset(original_root, args.train_manifest)
    validation = RachelPairDataset(original_root, "val")
    if len(training) not in (24_000, 60_000) or len(validation) != 3_000:
        raise ValueError("formal data ablation requires TRAIN24k/60k and unchanged VAL3000")
    plan = exposure_plan(len(training))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(1)
    runner._set_determinism(SEED)
    model = model.to(device).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # These names are the reused train_epoch interface, not adjustable arms.
    args.batch_size, args.effective_batch_size = MICROBATCH, EFFECTIVE_BATCH
    args.enable_seam_loss, args.seam_weight = False, 0.0
    protocol = dict(schema_version="rachel-realism-data-ablation/1", status="running",
                    arm=args.arm, initialization="warm-start", initial_checkpoint=str(source_path),
                    initial_epoch=checkpoint.get("epoch"), initial_model_metadata=architecture,
                    source_loss_config=checkpoint["loss_config"], effective_loss_config=asdict(loss_config),
                    full_network_trainable=True, optimizer="AdamW", learning_rate=LEARNING_RATE,
                    weight_decay=1e-4, gradient_clip_norm=5.0, precision="fp32", seed=SEED,
                    microbatch_size=MICROBATCH, effective_batch_size=EFFECTIVE_BATCH,
                    training_data=data_record, unique_count=data_record["unique_count"],
                    planned_global_exposures=TOTAL_EXPOSURES, planned_optimizer_updates=TOTAL_EXPOSURES // EFFECTIVE_BATCH,
                    planned_dataset_epochs=TOTAL_EXPOSURES // len(training),
                    validation_root=str(original_root), validation_split="val", validation_sample_count=len(validation),
                    validation_exposure_anchors=list(range(VALIDATION_INTERVAL, TOTAL_EXPOSURES + 1, VALIDATION_INTERVAL)),
                    selection_rule="maximize VAL fused equal-row F1; AP breaks ties; earliest exposure breaks exact ties; require full decision coverage",
                    threshold_rule="equal-row validation F1; largest threshold among ties",
                    no_pose_composite_selection=True, target_rules_changed=False,
                    test_used_for_selection=False, real_used_for_selection=False,
                    exposure_plan=plan, arguments=vars(args),
                    budget_caveat="Equal pair exposures and updates, not equal dataset epoch count. 24k reshuffles five times; 60k twice.",
                    accumulation_caveat="Same microbatch4/effective16 and original loss; auxiliary target-normalized losses need not equal a single batch16 loss.")
    save_json(destination / "protocol.json", protocol)
    save_json(destination / "architecture.json", architecture)
    total_samples = total_updates = 0
    best_key = None
    current_epoch = None
    epoch_order = None
    validation_events = 0
    started = time.perf_counter()
    try:
        for segment_index, segment in enumerate(plan, 1):
            epoch = segment["epoch"]
            if epoch != current_epoch:
                # Exactly one full permutation per dataset epoch, reused across
                # validation interruptions instead of reshuffling each slice.
                epoch_order = runner.epoch_indices(len(training), seed=SEED, epoch=epoch, limit=None)
                current_epoch = epoch
            indices = epoch_order[segment["epoch_start"]:segment["epoch_stop"]]
            loader = make_ablation_loader(training, indices, batch_size=MICROBATCH,
                                           num_workers=args.workers, seed=SEED + segment_index,
                                           contour_cap=model.config.contour_cap)
            save_json(destination / "status.json", dict(status="running", phase="train", epoch=epoch,
                                                       global_exposure=total_samples, optimizer_updates=total_updates,
                                                       segment=segment_index))
            torch.cuda.reset_peak_memory_stats(device)
            train_report = train_epoch(model, loader, optimizer, loss_config, device, args, epoch, args.arm)
            if train_report["samples"] != segment["sample_count"] or train_report["optimizer_updates"] != segment["optimizer_updates"]:
                raise RuntimeError("observed segment exposures/updates differ from matched budget")
            total_samples += train_report["samples"]
            total_updates += train_report["optimizer_updates"]
            if total_samples != segment["global_stop"]:
                raise RuntimeError("training exposure sequence skipped or duplicated a segment")
            segment_report = dict(segment=segment_index, epoch=epoch, epoch_complete=segment["epoch_complete"],
                                  epoch_slice=[segment["epoch_start"], segment["epoch_stop"]],
                                  global_exposure=total_samples, optimizer_updates=total_updates, training=train_report)
            save_json(destination / ("segment_%02d.json" % segment_index), segment_report)
            emit(dict(event="training_segment_complete", **segment_report))
            if not segment["validate_after"]:
                continue
            validation_events += 1
            save_json(destination / "status.json", dict(status="running", phase="validation", epoch=epoch,
                                                       global_exposure=total_samples, optimizer_updates=total_updates))
            val_loader = make_ablation_loader(validation, tuple(range(len(validation))), batch_size=MICROBATCH,
                                               num_workers=args.workers, seed=SEED, contour_cap=model.config.contour_cap)
            report, rows = evaluate_pair_validation(model, val_loader, device)
            if report["sample_count"] != 3000:
                raise RuntimeError("validation did not cover all original 3000 pair rows")
            result = dict(validation_event=validation_events, epoch=epoch, global_exposure=total_samples,
                          optimizer_updates=total_updates, validation=report)
            save_json(destination / ("validation_%06d.json" % total_samples), result)
            emit(dict(event="validation_complete", **result))
            key = tuple(report["selection_key"])
            if report["decision_coverage"] == 1.0 and (best_key is None or key > best_key):
                best_key = key
                saved = dict(model_state_dict=model.state_dict(), **model_metadata(model),
                             loss_config=asdict(loss_config), source_loss_config=checkpoint["loss_config"],
                             epoch=epoch, global_exposure=total_samples, optimizer_updates=total_updates,
                             validation_event=validation_events, variant=args.arm, initialization="warm-start",
                             initial_checkpoint=str(source_path), initial_epoch=checkpoint.get("epoch"),
                             seed=SEED, precision="fp32", resample_contour_cap=None,
                             seam_loss_enabled=False, seam_loss_weight=0.0, unique_count=data_record["unique_count"],
                             training_data=data_record, selection_rule=protocol["selection_rule"])
                runner._atomic_torch_save(destination / "winner.pt", saved)
                save_json(destination / "winner_validation.json", rows)
                freeze = dict(status="provisional", selected_epoch=epoch,
                              selected_global_exposure=total_samples, selected_optimizer_updates=total_updates,
                              selected_validation_event=validation_events, unique_count=data_record["unique_count"],
                              checkpoint=str(destination / "winner.pt"), validation=report,
                              classifier_thresholds=report["thresholds"], selection_rule=protocol["selection_rule"],
                              initial_checkpoint=str(source_path), seed=SEED, precision="fp32",
                              test_or_real_used_for_fit=False, original_validation_unchanged=True)
                save_json(destination / "train_val_freeze.json", freeze)
        if (total_samples, total_updates, validation_events) != (120_000, 7_500, 5):
            raise RuntimeError("completed training did not meet the fixed matched-exposure protocol")
        if best_key is None:
            raise RuntimeError("no full-decision-coverage validation winner; no held-out evaluation authorized")
        with (destination / "train_val_freeze.json").open(encoding="utf-8") as stream:
            freeze = json.load(stream)
        freeze.update(status="complete", completed_global_exposures=total_samples,
                      completed_optimizer_updates=total_updates, completed_dataset_epochs=TOTAL_EXPOSURES // len(training),
                      completed_validation_events=validation_events)
        save_json(destination / "train_val_freeze.json", freeze)
        protocol.update(status="complete", completed_global_exposures=total_samples,
                        completed_optimizer_updates=total_updates, completed_validation_events=validation_events,
                        selected_epoch=freeze["selected_epoch"], selected_global_exposure=freeze["selected_global_exposure"],
                        elapsed_seconds=time.perf_counter() - started)
        save_json(destination / "protocol.json", protocol)
        save_json(destination / "status.json", dict(status="complete", epoch=TOTAL_EXPOSURES // len(training),
                                                   global_exposure=total_samples, optimizer_updates=total_updates,
                                                   selected_global_exposure=freeze["selected_global_exposure"]))
    except Exception as error:
        save_json(destination / "status.json", dict(status="failed", epoch=current_epoch,
                                                   global_exposure=total_samples, optimizer_updates=total_updates,
                                                   error=repr(error)))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="original released dataset root; supplies unchanged VAL")
    p.add_argument("--train-manifest", help="optional 24k/60k composite TRAIN manifest")
    p.add_argument("--output", required=True, help="new arm directory containing winner.pt and train_val_freeze.json")
    p.add_argument("--arm", default="realism_data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    return p


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
