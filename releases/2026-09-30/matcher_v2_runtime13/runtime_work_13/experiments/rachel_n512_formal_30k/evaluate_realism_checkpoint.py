"""One-pass TEST/REAL evaluation of a completed equal-exposure data arm.

The trained Full model supplies its unchanged coarse/local/fused probabilities.
All pairs use the previously fixed Top2 mode translation decoder, independently
of pair scores. No thresholds, decoder, or model are selected here. In
particular, a matrix head trained on the old frozen matcher is not transferable
without refitting and is neither loaded nor applied by this evaluator.

FP32 and batch8 are fixed. REAL uses the existing target-blind prepared inputs;
the translation GT is opened only after every prediction is closed and fsynced.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from experiments.rachel_n512_formal_30k import run_layout_decoder_experiment as common
from experiments.rachel_n512_formal_30k import run_real_layout_decoder_experiment as real
from experiments.rachel_n512_formal_30k.run_real_contiguous_seam_ablation import (
    DEFAULT_PREPARED_CACHE, load_prepared_cache,
)
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from experiments.rachel_n512_formal_30k.fragment_size_strata import pair_size_metadata, summarize_size_strata
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import check_fixed_architecture, save_json
from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint
from staging.pairwise_v0_2.models.translation_layout import TranslationLayoutConfig, estimate_translation_layout
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


BATCH_SIZE = 8
DECODER_NAME = "full_top2_mode"
TOP2_CONFIG = TranslationLayoutConfig(correspondence_mode="topk_union", top_k=2,
    max_candidates=512, min_inliers=3, inlier_radius_px=10.0)


def load_training_winner(training_run, *, architecture_validator=check_fixed_architecture):
    """Require a complete data-arm receipt before accessing held-out inputs."""
    root = Path(training_run).resolve(strict=True)
    freeze_path, checkpoint_path = root / "train_val_freeze.json", root / "winner.pt"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (freeze.get("status") != "complete" or freeze.get("test_or_real_used_for_fit") is not False
            or freeze.get("original_validation_unchanged") is not True
            or freeze.get("completed_global_exposures") != 120000
            or freeze.get("completed_optimizer_updates") != 7500
            or freeze.get("completed_validation_events") != 5):
        raise ValueError("evaluation requires the complete validation-only 120k-exposure data-arm freeze")
    validation = freeze["validation"]
    if (validation.get("sample_count"), validation.get("positive_count"),
            validation.get("negative_count")) != (3000, 1500, 1500):
        raise ValueError("winner must have been selected on the original balanced VAL3000")
    if validation.get("pose_used_for_selection") is not False:
        raise ValueError("this evaluator requires the data-arm row-F1/AP selection receipt, not pose-composite selection")
    thresholds = freeze["classifier_thresholds"]
    if set(thresholds) != {"coarse", "local", "fused"} or any(
            not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError("the frozen three-branch probability thresholds must be finite and in [0,1]")
    if thresholds != validation["thresholds"]:
        raise ValueError("classifier thresholds differ from the selected validation event")
    checkpoint = sealed._torch_load_checkpoint(checkpoint_path)
    for saved, selected in (("epoch", "selected_epoch"),
                            ("global_exposure", "selected_global_exposure"),
                            ("optimizer_updates", "selected_optimizer_updates"),
                            ("validation_event", "selected_validation_event"),
                            ("unique_count", "unique_count")):
        if saved not in checkpoint or checkpoint[saved] != freeze.get(selected):
            raise ValueError("winner checkpoint differs from its training freeze: " + saved)
    if checkpoint.get("precision") != "fp32" or freeze.get("precision") != "fp32":
        raise ValueError("the registered data experiment is FP32")
    model = load_rachel_checkpoint(checkpoint)
    architecture = architecture_validator(model, checkpoint)
    checkpoint_sha = sealed._sha256_file(checkpoint_path)
    if freeze.get("checkpoint_sha256", checkpoint_sha) != checkpoint_sha:
        raise ValueError("winner checkpoint SHA differs from its training freeze")
    identity = dict(training_run=str(root), training_freeze=str(freeze_path),
        training_freeze_sha256=sealed._sha256_file(freeze_path),
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
        checkpoint_epoch=checkpoint["epoch"], global_exposure=checkpoint["global_exposure"],
        optimizer_updates=checkpoint["optimizer_updates"], validation_event=checkpoint["validation_event"],
        unique_count=checkpoint["unique_count"], seed=int(checkpoint["seed"]), precision="fp32",
        model_metadata=architecture, loss_config=checkpoint["loss_config"],
        original_fused_threshold=float(thresholds["fused"]), branch_validation_thresholds=thresholds,
        model_selection_rule=freeze["selection_rule"])
    return model.eval().requires_grad_(False), identity, thresholds


def predict_batch(model, batch, device):
    """One unchanged Full forward, followed by target-blind fixed geometry."""
    tensors = [sealed._tensor(getattr(batch, name), device, dtype) for name, dtype in (
        ("mask_a", torch.float32), ("mask_b", torch.float32), ("points_rc_a", torch.float32),
        ("points_rc_b", torch.float32), ("contour_valid_a", torch.bool), ("contour_valid_b", torch.bool))]
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = model(*tensors)
    scores = {name: getattr(output, name + "_probability").detach().float().cpu().numpy()
              for name in ("coarse", "local", "fused")}
    if any(values.shape != (len(batch.pair_ids),) or not np.isfinite(values).all()
           for values in scores.values()):
        raise ValueError("model returned nonfinite or misaligned pair probabilities")
    assignment = output.assignment.detach().float().cpu().numpy()
    rows = []
    for i, pair_id in enumerate(batch.pair_ids):
        estimate = estimate_translation_layout(batch.points_rc_a[i], batch.points_rc_b[i], assignment[i],
            batch.contour_valid_a[i], batch.contour_valid_b[i], config=TOP2_CONFIG)
        diagnostics = asdict(estimate)
        for key in ("t_a_to_b_rc", "candidate_indices", "inlier_mask"):
            diagnostics.pop(key, None)
        rows.append(common.clean(dict(pair_id=pair_id, fragment_a=batch.fragment_a_tokens[i],
            fragment_b=batch.fragment_b_tokens[i], decision_valid=bool(output.decision_valid[i].item()),
            classification={name: float(values[i]) for name, values in scores.items()},
            layouts={DECODER_NAME: dict(translation_rc=estimate.t_a_to_b_rc,
                offset_b_in_a_rc=-estimate.t_a_to_b_rc, valid=bool(estimate.valid), diagnostics=diagnostics)},
            **pair_size_metadata(batch.mask_a[i], batch.mask_b[i]))))
    return rows


def attach_test_targets(predictions, targets):
    for row in predictions:
        target = targets[row["pair_id"]]
        row.update(label=target["label"], target_translation_rc=target["translation_rc"],
                   source_unit_ids=target["source_unit_ids"])
        layout = row["layouts"][DECODER_NAME]
        layout["translation_l2_px"] = (float(np.linalg.norm(
            np.asarray(layout["translation_rc"], float) - np.asarray(target["translation_rc"], float)))
            if layout["valid"] and target["translation_rc"] is not None else None)
    return predictions


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(common.clean(row), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(args, *, winner_loader=None):
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    torch.set_num_threads(1)
    model, identity, thresholds = (winner_loader or load_training_winner)(args.training_run)
    sealed._set_determinism(identity["seed"])
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model = model.to(device).eval()
    targets, source_units = {}, {}
    if args.split == "real":
        metadata, arrays = load_prepared_cache(args.prepared_cache)
        expected_ids = [row["pair_id"] for row in metadata["pairs"]]
        if sum(row["label"] and row["strict"] for row in metadata["pairs"]) != 508:
            raise ValueError("strict547 must retain exactly the balanced population's 508 positives")
        batches = real.input_batches(metadata, arrays, BATCH_SIZE)
        source = dict(prepared_cache=str(Path(args.prepared_cache).resolve()),
            prepared_manifest_sha256=metadata["manifest_sha256"],
            prepared_cache_manifest_sha256=sealed._sha256_file(Path(args.prepared_cache) / "manifest.json"))
    else:
        root = Path(args.dataset).resolve()
        dataset = RachelPairDataset(root, "test")
        manifest_path = root / "pairs" / "test.jsonl"
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = [json.loads(line) for line in stream if line.strip()]
        if len(dataset) != 3000 or len(manifest) != 3000 or sum(row["label"] for row in manifest) != 1500:
            raise ValueError("TEST requires the original balanced 3000 pairs")
        expected_ids = [row["pair_id"] for row in manifest]
        source_units = {row["pair_id"]: sorted({row["fragment_" + side]["split_unit_id"] for side in "ab"})
                        for row in manifest}
        batches = make_ablation_loader(dataset, tuple(range(len(dataset))), batch_size=BATCH_SIZE,
            num_workers=args.workers, seed=identity["seed"], contour_cap=512)
        source = dict(dataset_root=str(root), test_manifest_sha256=sealed._sha256_file(manifest_path))
    protocol = dict(identity, **source, schema_version="realism-checkpoint-evaluation/1", status="running",
        split=args.split, sample_count=len(expected_ids), batch_size=BATCH_SIZE, autocast_enabled=False,
        selected_full_decoder=DECODER_NAME, decoder_config=asdict(TOP2_CONFIG),
        decoder_policy="fixed existing Top2 mode; no held-out or new validation decoder selection",
        all_pairs_decoded=True, single_model_forward_per_batch=True, classifier_modified=False,
        matrix_head_used=False, rotation_estimated=False, routing_used=False,
        translation_convention="t_a_to_b_rc=b-a; B placement in A frame=-t_a_to_b_rc; canvas800 pixels",
        test_or_real_used_for_fit=False, target_gt_evaluation_after_prediction_freeze=args.split == "real",
        seam_quality_evaluated=False,
        seam_quality_unavailable="no_dataset_correspondence_reference" if args.split == "real" else "not_requested_in_fixed_top2_evaluation")
    common.write_json(destination / "protocol.json", protocol)
    started, predictions = time.perf_counter(), []
    try:
        with (destination / "pair_predictions.jsonl").open("x", encoding="utf-8") as stream:
            for batch in batches:
                batch_predictions = predict_batch(model, batch, device)
                # Synthetic targets are loader-provided, but are never passed
                # into predict_batch or used until its fixed decoding finishes.
                if args.split == "test":
                    for i, pair_id in enumerate(batch.pair_ids):
                        targets[pair_id] = dict(label=bool(batch.labels[i]),
                            translation_rc=common.clean(batch.translation_a_to_b_rc[i]) if batch.translation_valid[i] else None,
                            source_unit_ids=source_units[pair_id])
                predictions.extend(batch_predictions)
                for row in batch_predictions:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if len(predictions) % 64 == 0 or len(predictions) == len(expected_ids):
                    print(json.dumps(dict(processed=len(predictions), total=len(expected_ids),
                                          elapsed_s=round(time.perf_counter() - started, 2))), flush=True)
            os.fsync(stream.fileno())
        if [row["pair_id"] for row in predictions] != expected_ids:
            raise ValueError("predictions did not cover every population row in the frozen order")
        common.write_json(destination / "prediction_complete.json", dict(status="all_predictions_frozen",
            sample_count=len(predictions), checkpoint_sha256=identity["checkpoint_sha256"],
            translation_gt_json_opened=False,
            synthetic_targets_loader_provided=args.split == "test"))
        # REAL GT is first accessed here, after the complete target-blind file
        # was closed/fsynced and the completion receipt was written.
        rows = (real.attach_ground_truth(predictions, metadata["pairs"], args.translation_gt_json)
                if args.split == "real" else attach_test_targets(predictions, targets))
        write_rows(destination / "pair_results.jsonl", rows)
        summary = common.summarize(rows, thresholds["fused"], thresholds)
        summary["size_strata"] = summarize_size_strata(rows, thresholds["fused"], thresholds)
        if args.split == "real":
            strict_rows = [row for row in rows if row["strict_member"]]
            strict = common.summarize(strict_rows, thresholds["fused"], thresholds)
            strict["size_strata"] = summarize_size_strata(strict_rows, thresholds["fused"], thresholds)
            summary["strict_summary"] = strict
            common.write_json(destination / "strict547_summary.json", strict)
        summary.update(status="complete", split=args.split,
            population="real_balanced1016" if args.split == "real" else "synthetic_test3000",
            selected_full_decoder=DECODER_NAME, selection_source="fixed_prior_top2_not_selected_on_this_evaluation",
            checkpoint_sha256=identity["checkpoint_sha256"], precision="fp32", batch_size=BATCH_SIZE,
            elapsed_s=time.perf_counter() - started, original_classification_preserved=True,
            branch_validation_thresholds=thresholds, original_fused_threshold=thresholds["fused"],
            decision_coverage=float(np.mean([row["decision_valid"] for row in rows])),
            matrix_head_used=False, rotation_estimated=False, routing_used=False,
            test_or_real_used_for_fit=False, target_gt_evaluation_after_prediction_freeze=args.split == "real",
            seam_quality_evaluated=False, seam_quality_unavailable=protocol["seam_quality_unavailable"])
        protocol.update(status="complete", elapsed_s=summary["elapsed_s"])
        save_json(destination / "protocol.json", protocol)
        common.write_json(destination / "receipt.json", dict(protocol,
            pair_predictions_sha256=sealed._sha256_file(destination / "pair_predictions.jsonl"),
            pair_results_sha256=sealed._sha256_file(destination / "pair_results.jsonl")))
        common.write_json(destination / "summary.json", summary)
        print(json.dumps(common.clean(summary), ensure_ascii=False, allow_nan=False), flush=True)
    except Exception as error:
        protocol.update(status="failed", error=repr(error), completed_predictions=len(predictions))
        save_json(destination / "protocol.json", protocol)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run", required=True, help="completed data arm with winner.pt and train_val_freeze.json")
    p.add_argument("--split", required=True, choices=("test", "real"))
    p.add_argument("--dataset", required=True, help="unchanged original release; TEST only, not composite TRAIN")
    p.add_argument("--output", required=True, help="new result directory; existing output is never overwritten")
    p.add_argument("--prepared-cache", type=Path, default=DEFAULT_PREPARED_CACHE)
    p.add_argument("--translation-gt-json", type=Path, default=real.DEFAULT_TRANSLATION_GT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    return p


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
