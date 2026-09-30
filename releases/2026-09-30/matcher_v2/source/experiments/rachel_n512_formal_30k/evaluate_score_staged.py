"""Evaluate only dedicated S3 fixed20/[13,20] freezes, without held-out fitting.

Pure prediction/summary helpers are shared with evaluate_score_design; its
loader, run function, globals and live checkpoint schema are not modified.
Persist P/R, coarse/local/fused and canonical Top2 predictions before REAL GT
or review labels are attached. Turufan is positive-only and has no layout GT.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import evaluate_score_design as core
from experiments.rachel_n512_formal_30k.freeze_score_staged_comparison import (
    SCHEMA, SELECTIONS, checkpoint_contract, validate_s3_freeze, load_training_epoch,
)

EVALUATION_SCHEMA = "rachel-score-staged-evaluation/1"


def load_s3_frozen_model(training_run, selection="fixed_epoch"):
    root = Path(training_run).resolve(strict=True)
    path = root / "s3_freezes" / "freeze.json"
    freeze = validate_s3_freeze(json.loads(path.read_text()))
    if selection not in SELECTIONS or Path(freeze["training_run"]).resolve() != root:
        raise ValueError("S3 freeze belongs to a different selection/run")
    selected = freeze["selections"][selection]
    epoch = selected["selected_epoch"]
    checkpoint_path = root / ("epoch_%03d.pt" % epoch)
    if Path(selected["checkpoint"]).resolve() != checkpoint_path:
        raise ValueError("S3 requires an original epoch checkpoint, never winner.pt")
    checkpoint_digest = core.sealed._sha256_file(checkpoint_path)
    if checkpoint_digest != selected["checkpoint_sha256"]:
        raise ValueError("S3 selected checkpoint differs from its freeze")
    rows_path = root / ("validation_%03d_rows.json" % epoch)
    if (Path(selected["validation_predictions"]).resolve() != rows_path
            or core.sealed._sha256_file(rows_path) != selected["validation_predictions_sha256"]):
        raise ValueError("S3 selected validation calibration rows changed")
    payload = load_training_epoch(root, epoch, expected_sha256=checkpoint_digest)
    contract = checkpoint_contract(payload, schedule=freeze["schedule"], epoch=epoch)
    if (contract != selected["contract"]
            or contract["common_training_contract_sha256"] != freeze["common_training_contract_sha256"]):
        raise ValueError("S3 checkpoint training/phase contract differs from freeze")
    if freeze["schedule"] == "staged":
        from experiments.rachel_n512_formal_30k.train_score_staged import load_staged_checkpoint
        model = load_staged_checkpoint(payload)
    else:
        from experiments.rachel_n512_formal_30k.train_score_design import load_score_checkpoint
        model = load_score_checkpoint(payload)
    identity = dict(training_run=str(root), budget=20, selection=selection,
        s3_comparison_schema=SCHEMA, schedule=freeze["schedule"],
        primary_selection="fixed_epoch", auxiliary_eligible_epoch_range=[13, 20],
        freeze_path=str(path), freeze_sha256=core.sealed._sha256_file(path),
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_digest,
        epoch=epoch, seed=payload["seed"], architecture=contract["architecture"],
        model_config=asdict(model.config), candidate_config=asdict(model.candidate_config),
        score_design_metadata=contract["model_design"], phase_protocol=freeze["phase_protocol"],
        resample_contour_cap=payload.get("resample_contour_cap"),
        classifier_thresholds=selected["classifier_thresholds"], operating_points=selected["operating_points"],
        winner_record=selected, test_or_real_used_for_fit=False, ood_used_for_fit=False)
    return model.eval().requires_grad_(False), identity


def run(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers nonnegative")
    if args.split == "real" and (not args.keep_ids or not Path(args.keep_ids).is_file()):
        raise ValueError("REAL evaluation requires the existing keep-IDs export")
    torch.set_num_threads(1)
    model, identity = load_s3_frozen_model(args.training_run, args.selection)
    core.sealed._set_determinism(identity["seed"])
    device = torch.device(args.device)
    model = model.to(device).eval().requires_grad_(False)
    if identity["model_config"]["contour_cap"] != 512 or identity["resample_contour_cap"] is not None:
        raise ValueError("S3 uses untouched original preparedN512 inputs")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    targets, metadata, source_units = {}, None, None
    if args.split == "test":
        root = Path(args.dataset)
        manifest_path = root / "pairs/test.jsonl"
        manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
        if len(manifest) != 3000 or sum(r["label"] for r in manifest) != 1500:
            raise ValueError("requires the complete balanced TEST3000")
        expected_ids = [r["pair_id"] for r in manifest]
        source_units = {r["pair_id"]: sorted({r["fragment_" + side]["split_unit_id"] for side in "ab"}) for r in manifest}
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
    protocol = dict(schema_version=EVALUATION_SCHEMA, s3_comparison_schema=SCHEMA,
        status="running", split=args.split, model=identity, **source,
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
                # Reused target-blind forward stores named P/R and three branch scores.
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
            raise ValueError("predictions do not match the complete ordered evaluation population")
        core.save(destination / "prediction_complete.json", dict(status="all_predictions_frozen",
            sample_count=len(predictions), real_gt_opened=False, review_labels_opened=False,
            checkpoint_sha256=identity["checkpoint_sha256"]))
        if args.split == "real":
            rows = core.real.attach_ground_truth(predictions, metadata["pairs"], args.translation_gt_json)
            kept = set(json.loads(Path(args.keep_ids).read_text())["kept_positive_pair_ids"])
            positives = {r["pair_id"] for r in rows if r["label"]}
            if len(kept) != 295 or not kept <= positives:
                raise ValueError("keep export differs from the reviewed295-positive cohort")
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
            summary["paired_baseline_layout"] = core.compare_baseline(rows, args.baseline_evaluation, args.split)
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
    p.add_argument("--selection", choices=SELECTIONS, default="fixed_epoch")
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
