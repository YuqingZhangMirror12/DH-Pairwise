"""Evaluate separately frozen, newly sampled N512/N1024 source-density runs.

SIM TEST uses a COMPLETE cap-specific CleanSourceDensityDataset cache with real
source-derived correspondence labels. REAL/OOD use mask-only re-extraction at
BOTH caps, preserving the original mask/scale and ignoring correspondence GT.
Only another dedicated new-density N512 run can be the paired N1024 baseline.
Legacy prepared512, input-axis and staged runs are never accepted as controls.
The score heads, P/R diagnostics, canonical layout decoder and held-out metric
definitions are reused unchanged. This entry never fits a held-out threshold.
Source-density v1/v2/v3/v4 is selected only by the frozen training identity;
the shared typed consumer rejects a missing reader or a different cache version.
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
from experiments.rachel_n512_formal_30k import resampled_input_support as resampling
from experiments.rachel_n512_formal_30k.score_design_checkpoint_io import load_owned_epoch_checkpoint
from experiments.rachel_n512_formal_30k.freeze_score_staged_comparison import validation_report
from experiments.rachel_n512_formal_30k.train_score_density import (
    identity_source_density_version, make_clean_density_dataset, validate_source_density_protocols)
from staging.pairwise_v0_2.pairwise_data import rachel_preprocess

SCHEMA = "rachel-score-density-evaluation/1"
TRAINING_SCHEMA = "rachel-score-density-training/1"
CHECKPOINT_SCHEMA = "rachel-score-density-checkpoint/1"
BUDGETS, SELECTIONS = (5, 10, 20, 30, 50), ("max_f1", "recall95")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def cap_common_protocol(protocol):
    """Only cap and its derived identity can differ, not labels/source/code."""
    return {key: value for key, value in protocol.items() if key not in ("contour_cap", "identity_sha256")}


def comparison_training_contract(identity):
    source_version = identity_source_density_version(identity)
    fields = ("seed", "score_design", "training_mode", "train_count", "train_split", "validation_count",
        "validation_split", "loss_config", "initial_weights_sha256", "candidate_config",
        "candidate_correctness_tolerance_px", "candidate_correctness_weight", "candidate_correctness_reduction",
        "train_source_selection_sha256", "train_source_selection_file_sha256", "train_source_pipeline",
        "train_original_manifest_sha256", "validation_manifest_sha256", "validation_common_pipeline_sha256",
        "microbatch", "effective_batch", "optimizer", "weight_decay", "lr_by_epoch", "grad_clip_norm",
        "precision", "max_epochs", "segment_pairs", "validation_every_epochs", "min_selection_epoch",
        "held_out_used_for_training_or_selection")
    if set(fields) - set(identity):
        raise ValueError("density identity lacks common source/pipeline/trajectory fields")
    common = canonical({key: identity[key] for key in fields})
    if (identity.get("schema_version") != TRAINING_SCHEMA or common["training_mode"] != "joint"
            or common["seed"] != 260913 or common["train_count"] != 24000
            or common["train_split"] != "train" or common["validation_count"] != 3000
            or common["validation_split"] != "val" or common["held_out_used_for_training_or_selection"] is not False):
        raise ValueError("requires new-density Full24/joint/complete cleanVAL3000")
    val = identity["validation_density_protocol"]
    validate_source_density_protocols(source_version,
        train_pipeline=common["train_source_pipeline"], clean_protocol=val)
    if (digest(cap_common_protocol(val)) != common["validation_common_pipeline_sha256"]
            or val["identity_sha256"] != identity["validation_density_identity_sha256"]
            or val["source_manifest_sha256"] != common["validation_manifest_sha256"]
            or val.get("source_pair_count") != 3000 or val.get("split") != "val"
            or val.get("pair_labels") != "unchanged original split labels"
            or val.get("assignment_targets") != "fresh exact clean source-cell ancestry, not old512 token indices"
            or val.get("weathering_applied") is not False or val.get("test_used_for_selection") is not False):
        raise ValueError("VAL does not have the registered genuine cap-specific source targets")
    # Preserve the historical v1 comparison identity byte-for-byte. Explicit
    # Each explicit newer consumer forms a separate comparison family, even
    # with the same cap. Existing v1/v2/v3 comparison identities are unchanged.
    if source_version != "v1":
        common.update(source_density_version=source_version,
            source_density_consumer=canonical(identity["source_density_consumer"]))
    return common


def owned(root, value, expected_name):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if path.resolve(strict=True) != root / expected_name:
        raise ValueError("requires exact run-owned " + expected_name)
    return path.resolve(strict=True)


def load_frozen_density_model(training_run, budget, selection):
    if budget not in BUDGETS or selection not in SELECTIONS:
        raise ValueError("unregistered density budget/selection")
    root = Path(training_run).resolve(strict=True)
    freeze_path = root / "budget_freezes" / ("%03d" % budget) / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if (freeze.get("schema_version") != TRAINING_SCHEMA or freeze.get("status") != "frozen_at_budget"
            or freeze.get("budget_epochs") != budget or freeze.get("budget_exposures") != budget * 24000
            or freeze.get("eligible_epoch_range") != [5, budget]
            or freeze.get("selection_population") != "source-rebuilt cleanVAL3000 only"
            or freeze.get("held_out_used_for_fit") is not False
            or set(freeze.get("winners", {})) != set(SELECTIONS)):
        raise ValueError("requires complete dedicated density SIM-VAL-only freeze")
    selected = freeze["winners"][selection]
    epoch = selected.get("selected_epoch")
    if (type(epoch) is not int or not 5 <= epoch <= budget or selected.get("selection") != selection
            or selected.get("selected_global_exposure") != epoch * 24000
            or selected.get("test_or_real_or_ood_used_for_fit") is not False):
        raise ValueError("density selected epoch violates budget/VAL contract")
    checkpoint = owned(root, selected["checkpoint"], "epoch_%03d.pt" % epoch)
    payload = load_owned_epoch_checkpoint(root, checkpoint, epoch, selected["checkpoint_sha256"])
    resume = payload.get("resume_identity", {})
    cap = payload.get("contour_cap")
    if (payload.get("score_density_checkpoint_schema") != CHECKPOINT_SCHEMA
            or payload.get("score_density_training_schema") != TRAINING_SCHEMA
            or any(name in payload for name in ("score_design_schema", "score_input_checkpoint_schema", "s3_checkpoint_schema"))
            or cap not in (512, 1024) or freeze.get("contour_cap") != cap
            or payload.get("global_exposure") != epoch * 24000
            or payload.get("optimizer_updates") != epoch * 1500 or payload.get("completed_segments") != epoch * 4
            or payload.get("source_weights_loaded") is not False or payload.get("formal_training_counted") is not True
            or payload.get("resample_contour_cap") != cap or payload.get("density_targets_rebuilt_from_source") is not True
            or payload.get("seed") != resume.get("seed")
            or freeze.get("score_density_metadata") != payload.get("score_density_metadata")
            or freeze.get("resume_identity") != resume or freeze.get("resume_identity_sha256") != digest(resume)):
        raise ValueError("not a complete matching typed density epoch")
    common = comparison_training_contract(resume)
    if (resume["validation_density_protocol"]["contour_cap"] != cap
            or resume["loss_config"] != payload.get("loss_config")
            or payload.get("training_data", {}).get("kind") != "paired_source_density_materialized"
            or payload["training_data"].get("unique_count") != 24000
            or payload["training_data"].get("paired_identity_sha256") != resume["train_source_selection_sha256"]):
        raise ValueError("density cap/loss differs from training identity")
    val_path = owned(root, selected["validation_predictions"], "validation_%03d_rows.json" % epoch)
    if core.sealed._sha256_file(val_path) != selected.get("validation_predictions_sha256"):
        raise ValueError("frozen complete VAL predictions changed or lack SHA256")
    report, operating = validation_report(json.loads(val_path.read_text()))
    thresholds = selected["classifier_thresholds"]
    if (report["decision_coverage"] != 1.0 or thresholds != report["thresholds"]
            or selected["operating_points"] != operating
            or selected["primary_pair_threshold"] != operating["thresholds"]["max_f1" if selection == "max_f1" else "recall_95"]):
        raise ValueError("density thresholds do not match saved complete SIM-VAL rows")
    from experiments.rachel_n512_formal_30k.train_score_density import load_score_density_checkpoint
    model = load_score_density_checkpoint(payload)
    if model.config.contour_cap != cap:
        raise ValueError("restored density model cap differs from freeze")
    metadata = payload["score_density_metadata"]
    identity = dict(training_run=str(root), budget=budget, selection=selection, epoch=epoch, contour_cap=cap,
        freeze_path=str(freeze_path), freeze_sha256=core.sealed._sha256_file(freeze_path),
        checkpoint_path=str(checkpoint), checkpoint_sha256=selected["checkpoint_sha256"], seed=payload["seed"],
        architecture=resume["score_design"], model_config=asdict(model.config),
        candidate_config=resume["candidate_config"], score_density_metadata=metadata,
        comparison_training_contract=common, comparison_training_contract_sha256=digest(common),
        validation_density_protocol=resume["validation_density_protocol"],
        classifier_thresholds=thresholds, operating_points=operating, winner_record=selected,
        resample_contour_cap=cap, original_prepared512_control=False,
        test_or_real_used_for_fit=False, ood_used_for_fit=False)
    source_version = identity_source_density_version(resume)
    if source_version != "v1":
        identity.update(source_density_version=source_version,
            source_density_consumer=canonical(resume["source_density_consumer"]))
    return model.eval().requires_grad_(False), identity


def validate_clean_preparation(receipt, rows, source_density_version="v1"):
    validate_source_density_protocols(source_density_version, preparation=receipt)
    ids = [row["pair_id"] for row in rows]
    if (len(rows) != 3000 or len(set(ids)) != 3000 or sum(bool(r["label"]) for r in rows) != 1500
            or any(r["split"] != "test" for r in rows)
            or receipt.get("schema_version") != "clean-source-density-preparation/1"
            or receipt.get("status") != "complete" or receipt.get("split") != "test"
            or receipt.get("full_split") is not True or receipt.get("original_split_count") != 3000
            or receipt.get("selected_count") != 3000 or receipt.get("cached_pair_count") != 3000
            or receipt.get("pair_ids") != ids or receipt.get("failures") != []
            or receipt.get("model_inference") is not False or receipt.get("model_or_threshold_selected") is not False):
        raise ValueError("TEST requires complete genuine paired-density cache of all original3000 pairs")
    records = receipt.get("records", [])
    if len(records) != 3000 or [row.get("pair_id") for row in records] != ids:
        raise ValueError("TEST preparation records omit/reorder original pairs")
    for source, prepared in zip(rows, records):
        if bool(prepared["label"]) != bool(source["label"]) or set(prepared["caps"]) != {"512", "1024"}:
            raise ValueError("TEST preparation must preserve labels and include both caps")
        for cap in (512, 1024):
            record = prepared["caps"][str(cap)]
            if (record["source_split"] != "test" or bool(record["pose_supervision_enabled"]) != bool(source["label"])
                    or len(record["real_points"]) != 2 or not all(4 <= n <= cap for n in record["real_points"])
                    or (not 0 < record["positive_matches"] <= min(record["real_points"])
                        if source["label"] else record["positive_matches"] != 0)):
                raise ValueError("TEST cache has invalid density or all-ignore/copied positive targets")


def load_clean_test_dataset(root, cache_root, cap, *, source_density_version="v1"):
    root, cache_root = Path(root).resolve(strict=True), Path(cache_root).resolve(strict=True)
    manifest_path = root / "pairs/test.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    receipt_path = cache_root / "clean_test_preparation.json"
    receipt = json.loads(receipt_path.read_text())
    validate_clean_preparation(receipt, rows, source_density_version)
    cap_cache = cache_root / ("test_n%d" % cap)
    if not (cap_cache / "cache_identity.json").is_file():
        raise ValueError("cap-specific TEST cache identity missing; no on-demand regeneration in evaluator")
    if any(not (cap_cache / (hashlib.sha256(row["pair_id"].encode()).hexdigest() + ".npz")).is_file() for row in rows):
        raise ValueError("TEST cache is incomplete; prepare before model evaluation")
    # Consumer version comes from the frozen training identity, never a CLI
    # override or fallback when a cache uses a different producer protocol.
    dataset = make_clean_density_dataset(root, "test", cap, cap_cache,
        source_density_version=source_density_version)
    source = dict(dataset_root=str(root), manifest_sha256=core.sealed._sha256_file(manifest_path),
        clean_density_cache=str(cap_cache), density_preparation_sha256=core.sealed._sha256_file(receipt_path),
        evaluation_input_contract=dataset.protocol,
        evaluation_input_common_sha256=digest(cap_common_protocol(dataset.protocol)),
        test_target_origin="fresh source-cell ancestry at each cap; no512 reindex or all-ignore positive labels")
    return dataset, rows, source


def real_input_contract(cap):
    return dict(schema_version="rachel-density-mask-only-evaluation-input/1", contour_cap=cap,
        original_masks_and_scale_unchanged=True, original_prepared512_used_as_control=False,
        correspondence_targets="ignored for inference only; no loss or correspondence evaluation on REAL/OOD",
        extraction="InputContourResampler from original binary mask for BOTH caps", smoothing_sigma=3.,
        code_sha256={"resampler": core.sealed._sha256_file(Path(resampling.__file__)),
                     "contour_extractor": core.sealed._sha256_file(Path(rachel_preprocess.__file__))})


def validate_paired_density(identity, baseline):
    if identity["contour_cap"] != 1024 or baseline["contour_cap"] != 512:
        raise ValueError("density comparison requires new1024 versus new512, not legacy prepared512")
    for key in ("architecture", "budget", "selection", "seed", "comparison_training_contract"):
        if identity[key] != baseline[key]:
            raise ValueError("density baseline differs in " + key)
    return dict(single_changed_axis="contour_cap", baseline_cap=512, variant_cap=1024,
        same_source_selection_and_pipeline=True, same_validation_source_and_labels=True,
        same_data_loss_init_update_contract=True, original_prepared512_used=False,
        comparison_training_contract_sha256=digest(identity["comparison_training_contract"]))


def baseline_identity(directory, identity, split):
    directory = Path(directory).resolve(strict=True)
    protocol = json.loads((directory / "protocol.json").read_text())
    if (protocol.get("schema_version") != SCHEMA or protocol.get("status") != "complete"
            or protocol.get("split") != split or protocol.get("decoder") != core.fixed.DECODER_NAME
            or protocol.get("decoder_config") != canonical(asdict(core.fixed.TOP2_CONFIG))):
        raise ValueError("baseline must be a completed dedicated density evaluation; old/input/S3 forbidden")
    prior = protocol["model"]
    model, frozen = load_frozen_density_model(prior["training_run"], prior["budget"], prior["selection"])
    del model
    for key in ("checkpoint_sha256", "freeze_sha256", "epoch", "seed", "architecture", "contour_cap",
                "classifier_thresholds", "operating_points", "score_density_metadata"):
        if prior[key] != frozen[key]:
            raise ValueError("baseline evaluation differs from its density freeze: " + key)
    return protocol, validate_paired_density(identity, frozen)


def compare_baseline(rows, directory, split, identity, protocol):
    previous, contract = baseline_identity(directory, identity, split)
    for field in ("manifest_sha256", "sample_count", "keep_ids_sha256", "evaluation_input_common_sha256"):
        if previous.get(field) != protocol.get(field):
            raise ValueError("density evaluation populations/pipelines differ: " + field)
    return dict(**core.compare_baseline(rows, directory, split), density_comparison_contract=contract)


def run(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers nonnegative")
    if args.split == "real" and (not args.keep_ids or not Path(args.keep_ids).is_file()):
        raise ValueError("REAL requires the existing keep-IDs export")
    if args.split == "test" and not args.clean_density_cache_root:
        raise ValueError("TEST requires --clean-density-cache-root; old512 loader is forbidden")
    torch.set_num_threads(1)
    model, identity = load_frozen_density_model(args.training_run, args.budget, args.selection)
    if args.baseline_evaluation:
        baseline_identity(args.baseline_evaluation, identity, args.split)
    core.sealed._set_determinism(identity["seed"])
    cap, device = identity["contour_cap"], torch.device(args.device)
    model = model.to(device).eval().requires_grad_(False)
    targets, metadata, source_units, resampler = {}, None, None, None
    if args.split == "test":
        dataset, manifest, source = load_clean_test_dataset(args.dataset, args.clean_density_cache_root, cap,
            source_density_version=identity.get("source_density_version", "v1"))
        expected_ids = [row["pair_id"] for row in manifest]
        source_units = {r["pair_id"]: sorted({r["fragment_" + s]["split_unit_id"] for s in "ab"}) for r in manifest}
        batches = resampling.make_ablation_loader(dataset, tuple(range(len(dataset))), batch_size=args.batch_size,
            num_workers=args.workers, seed=identity["seed"], contour_cap=cap)
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
        # Deliberately unconditional for512 as well as1024. Old prepared
        # contour tokens are not reused as the paired density control.
        resampler = resampling.InputContourResampler(cap)
        contract = real_input_contract(cap)
        source = dict(prepared_cache=str(cache), manifest_sha256=core.sealed._sha256_file(cache / "manifest.json"),
            evaluation_input_contract=contract, evaluation_input_common_sha256=digest(cap_common_protocol(contract)))
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate evaluation pair ID")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema_version=SCHEMA, status="running", split=args.split, model=identity, **source,
        sample_count=len(expected_ids), batch_size=args.batch_size, precision="fp32",
        decoder=core.fixed.DECODER_NAME, decoder_config=asdict(core.fixed.TOP2_CONFIG),
        decoder_design_unchanged=True, layout_predictions_may_change_with_trained_weights=True,
        resampling="genuine clean source-density cache" if args.split == "test" else "mask-only original scale; Gaussian sigma3; BOTH caps freshly extracted",
        model_input_fields=list(core.FIELDS), original_prepared512_control=False,
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
                if resampler is not None:
                    batch = resampler.resample_batch(batch)
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
        if resampler is not None:
            protocol["inference_resampler_cache"] = resampler.cache_info()
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
    p.add_argument("--clean-density-cache-root")
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
