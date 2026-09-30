"""Frozen E1: clean/damaged pairability heads and separate pose reliability.

Pclean/Pdamage retain every original binary pair label, share initialization,
normalization and order, and differ only in their cached TRAIN input features.
Rdamage trains on damaged TRAIN positives only: valid Top2 translation <=10px.
It never gates pairability. All selection uses original clean synthetic VAL;
TEST/REAL predictions are frozen before their targets are attached. One E1
forward supplies the identical unmodified raw layout for all three heads.
"""
from __future__ import annotations

import argparse
import copy
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

from experiments.rachel_n512_formal_30k import run_seam_geometry_head as base
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import RachelWeatheredDataset
from staging.pairwise_v0_2.training.rachel_weathering_training import (
    WeatheringStatistics, make_weathering_loader,
)

common, fixed, real, sealed = base.common, base.fixed, base.real, base.sealed
FEATURE_NAMES, GEOMETRY_NAMES, DECODER = base.FEATURE_NAMES, base.GEOMETRY_NAMES, base.DECODER
SEED, EPOCHS, TRAIN_COUNT, VAL_COUNT = base.SEED, base.EPOCHS, base.TRAIN_COUNT, base.VAL_COUNT
HEAD_BATCH, MATCHER_BATCH = base.HEAD_BATCH, base.MATCHER_BATCH
SCHEMA = "rachel-damage-separate-heads/1"
PAIR_HEADS, RELIABILITY_HEAD = ("pclean", "pdamage"), "rdamage"
HEADS = PAIR_HEADS + (RELIABILITY_HEAD,)
DAMAGE_EPOCH = 1
DAMAGE_POLICY = dict(seed=SEED, epoch=DAMAGE_EPOCH, clean_probability=.70,
    mild_probability=.25, moderate_probability=.05, mild_depth_px=2., moderate_depth_px=4.,
    realization="one fixed E1 TRAIN feature table, reused in all ten head epochs",
    original_labels_preserved=True, original_gt_translation_preserved=True,
    higher_coverage_data_used=False)


def load_e1_winner(checkpoint):
    path = Path(checkpoint).resolve(strict=True)
    root = path if path.is_dir() else path.parent
    if path.is_file() and path.name != "winner.pt":
        raise ValueError("--checkpoint must be the frozen winner.pt or its training directory")
    model, identity, thresholds = fixed.load_training_winner(root)
    saved = sealed._torch_load_checkpoint(Path(identity["checkpoint_path"]))
    if saved.get("variant") != "edge_weathering_e1":
        raise ValueError("separate heads require the fixed E1 winner, not E0/E2 or another arm")
    identity.update(matcher_variant=saved["variant"], matcher_training_data=saved["training_data"])
    return model.eval().requires_grad_(False), identity, thresholds


def pose_targets(rows):
    """Evaluation/target helper only. Invalid predicted poses are failures, not missing rows."""
    result = []
    for row in rows:
        pose = row["layouts"][DECODER]
        if row["label"]:
            gt = np.asarray(row.get("target_translation_rc"), float)
            if gt.shape != (2,) or not np.isfinite(gt).all():
                raise ValueError("every true pair requires finite translation GT for pose targets")
            error = pose.get("translation_l2_px")
            if pose["valid"] and (error is None or not np.isfinite(error) or error < 0):
                raise ValueError("a valid positive layout requires a finite nonnegative error")
            result.append(bool(pose["valid"] and error <= 10.0))
        else:
            result.append(False)
    return np.asarray(result, bool)


def align_training(clean, damage, val):
    if not clean or len(clean) != len(damage) or not val:
        raise ValueError("clean/damaged TRAIN must be nonempty and equally sized; VAL required")
    for left, right in zip(clean, damage):
        for key in ("pair_id", "fragment_a", "fragment_b", "label", "target_translation_rc"):
            if left[key] != right[key]:
                raise ValueError("damage changed original TRAIN identity/label/GT: " + key)
    for population in (clean, val):
        ids = [row["pair_id"] for row in population]
        if len(set(ids)) != len(ids) or any(row["label"] not in (0, 1) for row in population):
            raise ValueError("head populations require unique pair IDs and original binary labels")
        if len({bool(row["label"]) for row in population}) != 2:
            raise ValueError("pairability TRAIN/VAL require both original pair classes")
    if {row["pair_id"] for row in clean} & {row["pair_id"] for row in val}:
        raise ValueError("TRAIN and VAL pair IDs overlap")


def reliability_metrics(rows, scores, threshold):
    """Condition on actual pairability, never predicted P or R. Fixed 10 equal-width bins."""
    scores = np.asarray(scores, float)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("reliability scores must be finite probabilities aligned to all rows")
    mask = np.asarray([r["label"] for r in rows], bool)
    positives = [r for r in rows if r["label"]]
    y, s = pose_targets(positives), scores[mask]
    result = dict(sample_count=len(y), target_positive_count=int(y.sum()),
        target_negative_count=int((~y).sum()), threshold=float(threshold),
        population="actual true pairs only, independent of pairability acceptance",
        auroc=None, auprc=None, brier=None, ece10=None, calibration_bins=[],
        accepted_count=0, accepted_coverage=None, accepted_pose_success_count=0,
        conditional_pose_success=None, raw_pose_success_rate=None)
    if not len(y):
        result["metrics_status"] = "no_true_pairs"
        return result
    result.update(common.classification(y, s, threshold))
    # AP/ROC are explicitly unavailable for single-target-class populations.
    if len(np.unique(y)) != 2:
        result.update(auroc=None, auprc=None, ranking_status="single_target_class")
    else:
        result["ranking_status"] = "available"
    accepted = s >= threshold
    result.update(brier=float(np.mean((s - y) ** 2)), accepted_count=int(accepted.sum()),
        accepted_coverage=float(accepted.mean()), accepted_pose_success_count=int((accepted & y).sum()),
        conditional_pose_success=float(y[accepted].mean()) if accepted.any() else None,
        raw_pose_success_rate=float(y.mean()))
    ece = 0.
    for index in range(10):
        in_bin = (s >= index / 10) & ((s < (index + 1) / 10) if index < 9 else (s <= 1.))
        count = int(in_bin.sum())
        confidence, success = (float(s[in_bin].mean()), float(y[in_bin].mean())) if count else (None, None)
        result["calibration_bins"].append(dict(lower=index / 10, upper=(index + 1) / 10,
            upper_inclusive=index == 9, count=count, mean_probability=confidence, pose_success_rate=success))
        if count:
            ece += count / len(y) * abs(confidence - success)
    result["ece10"] = float(ece)
    return result


def initialize_heads(clean_features, damage_positive_features):
    torch.manual_seed(SEED)
    reference = base.SeamGeometryPairHead()
    reference.set_training_normalization(clean_features)
    heads = {name: copy.deepcopy(reference) for name in HEADS}
    heads[RELIABILITY_HEAD].set_training_normalization(damage_positive_features)
    return heads


def train_heads(clean, damage, val, *, output, identity, original_thresholds,
                input_identity=None, callback=None, epochs=EPOCHS, batch_size=HEAD_BATCH):
    align_training(clean, damage, val)
    positive_damage = [r for r in damage if r["label"]]
    positive_val = [r for r in val if r["label"]]
    x = dict(pclean=base.feature_matrix(clean), pdamage=base.feature_matrix(damage),
        rdamage=base.feature_matrix(positive_damage))
    y = {name: torch.tensor([r["label"] for r in clean], dtype=torch.float32) for name in PAIR_HEADS}
    y[RELIABILITY_HEAD] = torch.tensor(pose_targets(positive_damage), dtype=torch.float32)
    vx, rvx = base.feature_matrix(val), base.feature_matrix(positive_val)
    vy = np.asarray([r["label"] for r in val], bool)
    rvy = pose_targets(positive_val)
    heads = initialize_heads(x["pclean"], x["rdamage"])
    optimizers = {n: torch.optim.AdamW(h.parameters(), lr=base.LEARNING_RATE,
        weight_decay=base.WEIGHT_DECAY) for n, h in heads.items()}
    selected, states, best = {}, {}, {}
    steps = {n: 0 for n in HEADS}
    for epoch in range(1, epochs + 1):
        record = dict(epoch=epoch, heads={})
        for name, head in heads.items():
            # Same P permutations; R shuffles only its own true-pair population.
            order = np.random.default_rng(SEED + 1000003 * epoch).permutation(len(x[name]))
            total = 0.
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                head.train()
                optimizers[name].zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x[name][idx]), y[name][idx])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite independent head loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
                optimizers[name].step()
                total += float(loss.detach()) * len(idx)
                steps[name] += 1
            with torch.inference_mode():
                scores = torch.sigmoid(head.eval()(rvx if name == RELIABILITY_HEAD else vx)).numpy()
            targets = rvy if name == RELIABILITY_HEAD else vy
            threshold = common.fit_threshold(targets, scores)
            metrics = (reliability_metrics(positive_val, scores, threshold) if name == RELIABILITY_HEAD
                else common.classification(targets, scores, threshold))
            ap = metrics.get("auprc")
            key = ((-metrics["brier"], ap if ap is not None else -1.) if name == RELIABILITY_HEAD
                else (metrics["f1"], ap))
            record["heads"][name] = dict(train_loss=total / len(order), pair_exposures=epoch * len(order),
                optimizer_updates=steps[name], validation_metrics=metrics)
            if name not in best or key > best[name]:
                best[name], states[name] = key, copy.deepcopy(head.state_dict())
                selected[name] = dict(epoch=epoch, threshold=threshold, validation_metrics=metrics)
        base.save_json(Path(output) / ("epoch_%02d.json" % epoch), record)
        if callback:
            callback(record)
    for name, head in heads.items():
        head.load_state_dict(states[name], strict=True)
        head.eval().requires_grad_(False)
    budgets = {n: dict(train_unique_count=len(x[n]), completed_epochs=epochs,
        completed_pair_exposures=epochs * len(x[n]), completed_optimizer_updates=steps[n]) for n in HEADS}
    freeze = dict(schema_version=SCHEMA, status="complete", selected=selected,
        matcher_checkpoint_id=identity["checkpoint_sha256"], matcher_identity=identity,
        input_identity=input_identity, feature_names=list(FEATURE_NAMES), geometry_names=list(GEOMETRY_NAMES),
        original_branch_thresholds=original_thresholds, original_fused_threshold=original_thresholds["fused"],
        precision="fp32", head_batch_size=batch_size, budgets=budgets, seed=SEED,
        validation_sample_count=len(val), validation_positive_count=len(positive_val),
        selection_rules=dict(pairability="maximum clean VAL equal-row F1, then AP, then earliest epoch",
            reliability="minimum clean VAL-positive Brier, then AP if defined, then earliest epoch"),
        threshold_rule="each selected head: clean VAL target F1, largest threshold among ties",
        pairability_targets="original binary pair labels, unchanged by layout accuracy or damage",
        reliability_target="TRAIN true pair AND valid predicted Top2 translation error <=10 canvas pixels",
        reliability_train_target_positive_count=int(y[RELIABILITY_HEAD].sum()),
        reliability_validation_target_positive_count=int(rvy.sum()),
        damage_policy=DAMAGE_POLICY, pairability_shared_initialization=True,
        pairability_shared_normalization="clean TRAIN features only; identical buffers in both P heads",
        reliability_normalization="damaged TRAIN true-pair features only",
        pairability_shared_order=True, test_or_real_used_for_fit=False,
        reliability_gates_pairability=False, layout_modified=False, matcher_frozen=True,
        geometry_target_blind=True, selected_full_decoder=DECODER, decoder_config=asdict(base.TOP2_CONFIG),
        head_training_device="cpu", matcher_inference_batch_size=MATCHER_BATCH,
        optimizer="AdamW", learning_rate=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY,
        heads={n: h.metadata() for n, h in heads.items()})
    checkpoint = dict(schema_version=SCHEMA, matcher_checkpoint_id=identity["checkpoint_sha256"],
        feature_names=list(FEATURE_NAMES), selected=selected,
        heads={n: dict(config=asdict(h.config), state_dict=states[n]) for n, h in heads.items()})
    torch.save(checkpoint, Path(output) / "heads.pt")
    freeze["head_checkpoint_sha256"] = sealed._sha256_file(Path(output) / "heads.pt")
    common.write_json(Path(output) / "validation_freeze.json", freeze)
    return heads, freeze


def load_frozen_heads(output, matcher_checkpoint_id, input_identity):
    root = Path(output)
    freeze = json.loads((root / "validation_freeze.json").read_text())
    if (freeze.get("schema_version") != SCHEMA or freeze.get("status") != "complete"
            or freeze.get("matcher_checkpoint_id") != matcher_checkpoint_id
            or freeze.get("input_identity") != input_identity
            or freeze.get("test_or_real_used_for_fit") is not False
            or freeze.get("reliability_gates_pairability") is not False
            or freeze.get("feature_names") != list(FEATURE_NAMES)
            or freeze.get("head_batch_size") != HEAD_BATCH
            or freeze.get("validation_sample_count") != VAL_COUNT
            or freeze.get("validation_positive_count") != VAL_COUNT // 2
            or freeze.get("damage_policy") != DAMAGE_POLICY or freeze.get("precision") != "fp32"):
        raise ValueError("incomplete or mismatched separate-head validation freeze")
    for name in HEADS:
        count = TRAIN_COUNT // 2 if name == RELIABILITY_HEAD else TRAIN_COUNT
        expected = dict(train_unique_count=count, completed_epochs=EPOCHS,
            completed_pair_exposures=EPOCHS * count,
            completed_optimizer_updates=EPOCHS * math.ceil(count / HEAD_BATCH))
        if freeze.get("budgets", {}).get(name) != expected:
            raise ValueError("wrong registered head training budget: " + name)
    if sealed._sha256_file(root / "heads.pt") != freeze["head_checkpoint_sha256"]:
        raise ValueError("serialized heads differ from their freeze")
    saved = sealed._torch_load_checkpoint(root / "heads.pt")
    if (saved.get("schema_version") != SCHEMA or saved.get("matcher_checkpoint_id") != matcher_checkpoint_id
            or saved.get("feature_names") != list(FEATURE_NAMES) or saved.get("selected") != freeze["selected"]
            or set(saved["heads"]) != set(HEADS)):
        raise ValueError("head checkpoint and freeze identities differ")
    heads = {}
    for name in HEADS:
        config = base.SeamGeometryHeadConfig(**saved["heads"][name]["config"])
        if config != base.SeamGeometryHeadConfig():
            raise ValueError("all separate heads must use the same registered 24-input MLP")
        head = base.SeamGeometryPairHead(config)
        head.load_state_dict(saved["heads"][name]["state_dict"], strict=True)
        heads[name] = head.eval().requires_grad_(False)
    return heads, freeze


def score_heads(heads, rows):
    base.score_heads(heads, rows)
    for row in rows:
        row["pose_reliability"] = dict(probability=row["head_scores"].pop(RELIABILITY_HEAD),
            head=RELIABILITY_HEAD, gates_pairability=False)
    return rows


def update_status(args, stage, **values):
    state = dict(schema_version=SCHEMA, status="running", stage=stage, pid=os.getpid(), **values)
    base.save_json(Path(args.output) / "status.json", state)
    base.emit(state)


def extract_population(args, split, destination, model, identity, *, dataset=None, damaged=False, heads=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    metadata, targets, stats = None, {}, WeatheringStatistics() if damaged else None
    if split == "real":
        if damaged:
            raise ValueError("damage is TRAIN-only")
        metadata, arrays = base.load_prepared_cache(args.prepared_cache)
        expected_ids = [r["pair_id"] for r in metadata["pairs"]]
        batches = real.input_batches(metadata, arrays, MATCHER_BATCH)
        source = dict(prepared_cache=str(Path(args.prepared_cache).resolve()),
            prepared_manifest_sha256=metadata["manifest_sha256"],
            prepared_cache_manifest_sha256=sealed._sha256_file(Path(args.prepared_cache) / "manifest.json"))
        expected_n = 1016
    else:
        dataset = dataset if dataset is not None else base.RachelPairDataset(Path(args.dataset), split)
        expected_n = TRAIN_COUNT if split == "train" else VAL_COUNT
        if len(dataset) != expected_n or (damaged and split != "train"):
            raise ValueError("wrong frozen population size or non-TRAIN weathering")
        expected_ids = None
        loader = make_weathering_loader if damaged else base.make_ablation_loader
        batches = loader(dataset, tuple(range(len(dataset))), batch_size=MATCHER_BATCH,
            num_workers=args.workers, seed=SEED, contour_cap=512)
        source = dict(dataset_root=str(Path(args.dataset).resolve()),
            input_manifest_sha256=sealed._sha256_file(Path(args.train_manifest) if split == "train"
                else Path(args.dataset) / "pairs" / (split + ".jsonl")))
    protocol = dict(schema_version=SCHEMA, status="running", split=split,
        matcher_checkpoint_id=identity["checkpoint_sha256"], precision="fp32", batch_size=MATCHER_BATCH,
        frozen_matcher=True, feature_names=list(FEATURE_NAMES), geometry_target_blind=True,
        layout_modified=False, selected_full_decoder=DECODER, decoder_config=asdict(base.TOP2_CONFIG),
        damage_policy=DAMAGE_POLICY if damaged else None, reliability_gates_pairability=False,
        target_gt_evaluation_after_prediction_freeze=split == "real", **source)
    common.write_json(destination / "protocol.json", protocol)
    started, predictions = time.perf_counter(), []
    with (destination / "pair_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for item in batches:
            batch = item.batch if damaged else item
            predicted = base.predict_batch(model, batch, torch.device(args.device))
            if heads is not None:
                score_heads(heads, predicted)
            if damaged:
                stats.add(item)
                for row, report in zip(predicted, item.reports):
                    row["damage"] = {k: report[k] for k in ("changed_pair", "changed_a", "changed_b",
                        "tier", "fallback_reason", "original_gt_translation_preserved")}
            predictions.extend(predicted)
            for row in predicted:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            # Neither these targets nor sidecar damage reports enter predict_batch/heads.
            if split != "real":
                for i, pair_id in enumerate(batch.pair_ids):
                    targets[pair_id] = dict(label=bool(batch.labels[i]), source_unit_ids=[],
                        translation_rc=common.clean(batch.translation_a_to_b_rc[i]) if batch.translation_valid[i] else None)
            stream.flush()
            if len(predictions) % 64 == 0:
                update_status(args, "cache_" + split + ("_damage" if damaged else ""),
                    processed=len(predictions), total=expected_n, elapsed_s=time.perf_counter() - started)
        os.fsync(stream.fileno())
    ids = [r["pair_id"] for r in predictions]
    if len(ids) != expected_n or len(set(ids)) != expected_n or (expected_ids is not None and ids != expected_ids):
        raise ValueError("prediction population incomplete, duplicated or reordered")
    common.write_json(destination / "prediction_complete.json", dict(status="all_predictions_frozen",
        sample_count=expected_n, translation_gt_json_opened=False, head_scores_frozen=heads is not None,
        pair_predictions_sha256=sealed._sha256_file(destination / "pair_predictions.jsonl")))
    rows = (real.attach_ground_truth(predictions, metadata["pairs"], args.translation_gt_json) if split == "real"
        else fixed.attach_test_targets(predictions, targets))
    if sum(r["label"] for r in rows) != expected_n // 2:
        raise ValueError("registered populations must remain balanced")
    pose_targets(rows)  # fail on missing positive targets, but retain invalid-pose failures
    fixed.write_rows(destination / "pair_results.jsonl", rows)
    protocol.update(status="complete", sample_count=len(rows), positive_count=expected_n // 2,
        elapsed_s=time.perf_counter() - started,
        pair_results_sha256=sealed._sha256_file(destination / "pair_results.jsonl"))
    if damaged:
        protocol["actual_damage_statistics"] = stats.report()
    if split == "real":
        protocol["translation_gt_json_sha256"] = sealed._sha256_file(args.translation_gt_json)
    base.save_json(destination / "protocol.json", protocol)
    return rows, protocol


def summarize_methods(rows, freeze, *, include_strata=True):
    labels = np.asarray([r["label"] for r in rows], bool)
    result = dict(sample_count=len(rows), positive_count=int(labels.sum()), negative_count=int((~labels).sum()),
        methods={}, layout={}, reliability_gates_pairability=False, raw_layout_shared_by_all_heads=True)
    if not rows:
        result["metrics_status"] = "empty"
        return result
    original = common.summarize(rows, freeze["original_fused_threshold"], freeze["original_branch_thresholds"])
    pose = copy.deepcopy(original["layout"][DECODER])
    pose.pop("assembly", None)
    correct = pose_targets(rows)
    pose.update(positive_count=int(labels.sum()), raw_layout10_success_count=int(correct.sum()))
    result["layout"][DECODER], result["original_classification"] = pose, original["classification"]
    valid = np.asarray([r["layouts"][DECODER]["valid"] for r in rows], bool)
    errors = np.asarray([r["layouts"][DECODER]["translation_l2_px"]
        if r["layouts"][DECODER]["translation_l2_px"] is not None else np.inf for r in rows])
    thresholds = {n: freeze["selected"][n]["threshold"] for n in PAIR_HEADS}
    thresholds["original_fused"] = freeze["original_fused_threshold"]
    for name, threshold in thresholds.items():
        scores = np.asarray([r["head_scores"][name] for r in rows])
        accepted = scores >= threshold  # R is deliberately absent from this decision.
        metric = common.classification(labels, scores, threshold)
        if len(np.unique(labels)) != 2:
            metric.update(auroc=None, auprc=None)
        joint = {}
        for tolerance in (2, 5, 8, 10):
            tp = int((accepted & labels & valid & (errors <= tolerance)).sum())
            fp, fn = int(accepted.sum()) - tp, int(labels.sum()) - tp
            joint[str(tolerance)] = dict(tp=tp, fp=fp, fn=fn, precision=tp / max(1, tp + fp),
                recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn))
        result["methods"][name] = dict(classification=metric, joint=joint, threshold=threshold)
    result["reliability"] = {RELIABILITY_HEAD: reliability_metrics(rows,
        [r["pose_reliability"]["probability"] for r in rows], freeze["selected"][RELIABILITY_HEAD]["threshold"])}
    if include_strata:
        result["size_strata"] = dict(fixed_bin_edges=[.25, .5], thresholds_refit=False,
            groups={key: summarize_methods([r for r in rows if r["size_ratio_stratum"] == key],
                freeze, include_strata=False) for key in base.SIZE_RATIO_STRATA + (base.INVALID_SIZE_STRATUM,)})
    return result


def write_evaluation(destination, rows, split, freeze, identity, freeze_path, protocol=None):
    destination = Path(destination)
    if protocol is None:
        destination.mkdir(parents=True, exist_ok=False)
        fixed.write_rows(destination / "pair_results.jsonl", rows)
    summary = summarize_methods(rows, freeze)
    if split == "real":
        strict = [r for r in rows if r["strict_member"]]
        if len(strict) != 547 or sum(r["label"] for r in strict) != 508:
            raise ValueError("strict REAL must retain 508 positive and 39 negative pairs")
        summary["strict_summary"] = summarize_methods(strict, freeze)
        common.write_json(destination / "strict547_summary.json", summary["strict_summary"])
    provenance = dict(schema_version=SCHEMA, status="complete", split=split,
        matcher_checkpoint_id=identity["checkpoint_sha256"], matcher_identity=identity,
        validation_freeze_sha256=sealed._sha256_file(freeze_path),
        head_checkpoint_sha256=freeze["head_checkpoint_sha256"], feature_names=list(FEATURE_NAMES),
        layout_modified=False, raw_layout_shared_by_all_heads=True, matcher_frozen=True,
        reliability_gates_pairability=False, test_or_real_used_for_fit=False,
        target_gt_evaluation_after_prediction_freeze=split == "real", selected_full_decoder=DECODER,
        reliability_caveat="R predicts current translation reliability conditional on a true pair; low R is not nonpairability",
        calibration_policy="raw sigmoid; fixed ten equal-width probability bins; no TEST/REAL recalibration",
        geometry_caveat="predicted contour evidence, not annotated seam coverage or seam quality")
    summary.update(provenance)
    common.write_json(destination / "summary.json", summary)
    base.save_json(destination / "protocol.json", dict(protocol or {}, **provenance))
    common.write_json(destination / "receipt.json", dict(provenance,
        pair_results_sha256=sealed._sha256_file(destination / "pair_results.jsonl")))
    return summary


def run(args):
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    torch.set_num_threads(1)
    sealed._set_determinism(SEED)
    root = Path(args.output).resolve()
    if args.evaluate_only:
        if not root.is_dir():
            raise ValueError("--evaluate-only needs an existing completed head fit")
    else:
        root.mkdir(parents=True, exist_ok=False)
    try:
        update_status(args, "load_frozen_e1")
        model, identity, thresholds = load_e1_winner(args.checkpoint)
        model.to(torch.device(args.device)).eval().requires_grad_(False)
        inputs = {key: str(Path(getattr(args, key)).resolve()) for key in
            ("dataset", "train_manifest", "prepared_cache", "translation_gt_json")}
        inputs["train_manifest_sha256"] = sealed._sha256_file(Path(args.train_manifest))
        if args.evaluate_only:
            heads, freeze = load_frozen_heads(root, identity["checkpoint_sha256"], inputs)
        else:
            training, data_record = base.make_training_dataset(Path(args.dataset), Path(args.train_manifest))
            if (len(training) != TRAIN_COUNT or data_record["unique_count"] != TRAIN_COUNT
                    or Path(identity["matcher_training_data"]["manifest"]).resolve() != Path(args.train_manifest).resolve()):
                raise ValueError("head TRAIN must be the fixed original E1 matched24k manifest")
            protocol = dict(schema_version=SCHEMA, status="running", matcher_identity=identity,
                input_identity=inputs, training_data=data_record, damage_policy=DAMAGE_POLICY,
                heads=list(HEADS), feature_names=list(FEATURE_NAMES), planned_epochs=EPOCHS,
                pairability_exposures_per_head=EPOCHS * TRAIN_COUNT,
                reliability_positive_exposures=EPOCHS * TRAIN_COUNT // 2,
                head_batch_size=HEAD_BATCH, matcher_frozen=True, layout_modified=False,
                reliability_gates_pairability=False, test_or_real_used_for_fit=False,
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
            common.write_json(root / "protocol.json", protocol)
            clean, _ = extract_population(args, "train", root / "cache" / "train_clean", model, identity, dataset=training)
            weathered = RachelWeatheredDataset(training, seed=SEED, epoch=DAMAGE_EPOCH, cache_dir=args.cache_dir)
            damage, _ = extract_population(args, "train", root / "cache" / "train_damage", model, identity,
                dataset=weathered, damaged=True)
            del training, weathered
            val, _ = extract_population(args, "val", root / "cache" / "val_clean", model, identity)
            heads, freeze = train_heads(clean, damage, val, output=root, identity=identity,
                original_thresholds=thresholds, input_identity=inputs,
                callback=lambda r: update_status(args, "fit_heads", **r))
            heads, freeze = load_frozen_heads(root, identity["checkpoint_sha256"], inputs)
            score_heads(heads, val)
            write_evaluation(root / "val", val, "val", freeze, identity, root / "validation_freeze.json")
            del clean, damage, val
            protocol.update(status="fit_complete")
            base.save_json(root / "protocol.json", protocol)
        if not args.fit_only:
            for split in ("test", "real"):
                destination = root / split
                if (destination / "receipt.json").exists():
                    receipt = json.loads((destination / "receipt.json").read_text())
                    if (receipt.get("schema_version") != SCHEMA or receipt.get("status") != "complete"
                            or receipt.get("validation_freeze_sha256") != sealed._sha256_file(root / "validation_freeze.json")
                            or receipt.get("pair_results_sha256") != sealed._sha256_file(destination / "pair_results.jsonl")):
                        raise ValueError("existing held-out evaluation is incomplete or mismatched")
                    continue
                rows, protocol = extract_population(args, split, destination, model, identity, heads=heads)
                write_evaluation(destination, rows, split, freeze, identity, root / "validation_freeze.json", protocol)
        protocol = json.loads((root / "protocol.json").read_text())
        protocol["status"] = "fit_complete" if args.fit_only else "complete"
        base.save_json(root / "protocol.json", protocol)
        base.save_json(root / "status.json", dict(schema_version=SCHEMA, status=protocol["status"],
            stage=None, matcher_checkpoint_id=identity["checkpoint_sha256"], heads=list(HEADS)))
        base.emit(dict(status=protocol["status"], output=str(root)))
    except Exception as error:
        base.save_json(root / "status.json", dict(schema_version=SCHEMA, status="failed", error=repr(error), pid=os.getpid()))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="fixed E1 winner.pt or training directory containing its freeze")
    p.add_argument("--dataset", required=True, help="original release with unchanged clean VAL/TEST")
    p.add_argument("--train-manifest", required=True, help="same fixed matched24k manifest as E1")
    p.add_argument("--prepared-cache", type=Path, default=base.DEFAULT_PREPARED_CACHE)
    p.add_argument("--translation-gt-json", type=Path, default=real.DEFAULT_TRANSLATION_GT)
    p.add_argument("--output", required=True, help="new output; existing only with --evaluate-only")
    p.add_argument("--device", default="cuda:0", help="FP32 frozen E1 feature extraction; all head fitting uses CPU")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", type=Path, help="optional E1 TRAIN weathered-fragment disk cache")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--fit-only", action="store_true")
    group.add_argument("--evaluate-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
