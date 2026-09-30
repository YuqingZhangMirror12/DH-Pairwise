"""E3: frozen E2 matcher, matched shallow pair heads, unchanged Top2 layout.

Order is TRAIN/clean VAL target-blind feature extraction, TRAIN-only head fitting
and normalization, clean VAL epoch/threshold freeze, then TEST and REAL. REAL
translation GT is first read after every prediction (including both head scores)
has been closed/fsynced. The predeclared primary method is score_geometry;
score_only is a same-budget attribution control, never an alternative chosen on
held-out results. No matcher or geometry decoder is trained or selected here.

Use --fit-only to stop after the freeze and --evaluate-only with that same
--output to evaluate later. Incomplete stages are not overwritten or restarted.
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

from experiments.rachel_n512_formal_30k import evaluate_realism_checkpoint as fixed
from experiments.rachel_n512_formal_30k import run_layout_decoder_experiment as common
from experiments.rachel_n512_formal_30k import run_real_layout_decoder_experiment as real
from experiments.rachel_n512_formal_30k.run_real_contiguous_seam_ablation import (
    DEFAULT_PREPARED_CACHE, load_prepared_cache,
)
from experiments.rachel_n512_formal_30k.fragment_size_strata import (
    pair_size_metadata, SIZE_RATIO_STRATA, INVALID_SIZE_STRATUM,
)
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import make_training_dataset, save_json
from staging.pairwise_v0_2.models.seam_geometry_pair_head import (
    FEATURE_NAMES, GEOMETRY_NAMES, TOP2_CONFIG, SeamGeometryHeadConfig,
    SeamGeometryPairHead, geometry_evidence,
)
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


SCHEMA = "rachel-seam-geometry-pair-head/1"
METHODS = ("score_only", "score_geometry")
SEED, EPOCHS, TRAIN_COUNT, VAL_COUNT = 260909, 10, 24000, 3000
HEAD_BATCH, MATCHER_BATCH, LEARNING_RATE, WEIGHT_DECAY = 256, 8, 1e-3, 1e-4
DECODER = "full_top2_mode"


def emit(value):
    print(json.dumps(common.clean(value), ensure_ascii=False, allow_nan=False), flush=True)


def load_e2_winner(training_run):
    model, identity, thresholds = fixed.load_training_winner(training_run)
    checkpoint = sealed._torch_load_checkpoint(Path(identity["checkpoint_path"]))
    if checkpoint.get("variant") != "edge_consistency_e2":
        raise ValueError("E3 requires the predeclared E2 clean-VAL-selected winner, not another arm")
    identity["matcher_variant"] = checkpoint["variant"]
    identity["matcher_training_data"] = checkpoint["training_data"]
    return model.eval().requires_grad_(False), identity, thresholds


def feature_matrix(rows):
    values = np.asarray([row["features"] for row in rows], dtype=np.float32)
    if values.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(values).all():
        raise ValueError("cached features must be finite and use the frozen feature schema")
    return torch.from_numpy(values)


def score_heads(heads, rows):
    x = feature_matrix(rows)
    with torch.inference_mode():
        scores = {name: torch.sigmoid(head.eval()(x)).cpu().numpy() for name, head in heads.items()}
    for i, row in enumerate(rows):
        row["head_scores"] = {name: float(values[i]) for name, values in scores.items()}
        row["head_scores"]["original_fused"] = row["classification"]["fused"]
    return rows


def predict_batch(model, batch, device, heads=None):
    """Only six input tensors enter the frozen matcher; no labels/GT geometry."""
    tensors = [sealed._tensor(getattr(batch, name), device, dtype) for name, dtype in (
        ("mask_a", torch.float32), ("mask_b", torch.float32), ("points_rc_a", torch.float32),
        ("points_rc_b", torch.float32), ("contour_valid_a", torch.bool), ("contour_valid_b", torch.bool))]
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = model(*tensors)
    probabilities = {name: getattr(output, name + "_probability").detach().float().cpu().numpy()
                     for name in ("coarse", "local", "fused")}
    logits = {name: getattr(output, name + "_logit").detach().float().cpu().numpy()
              for name in ("coarse", "local")}
    assignment = output.assignment.detach().float().cpu().numpy()
    if any(value.shape != (len(batch.pair_ids),) or not np.isfinite(value).all()
           for value in list(probabilities.values()) + list(logits.values())):
        raise ValueError("nonfinite or misaligned frozen matcher scores")
    rows = []
    for i, pair_id in enumerate(batch.pair_ids):
        geometry, estimate = geometry_evidence(batch.points_rc_a[i], batch.points_rc_b[i], assignment[i],
            batch.contour_valid_a[i], batch.contour_valid_b[i], batch.mask_a[i], batch.mask_b[i])
        diagnostics = asdict(estimate)
        for key in ("t_a_to_b_rc", "candidate_indices", "inlier_mask"):
            diagnostics.pop(key, None)
        rows.append(common.clean(dict(pair_id=pair_id,
            fragment_a=batch.fragment_a_tokens[i], fragment_b=batch.fragment_b_tokens[i],
            decision_valid=bool(output.decision_valid[i].item()),
            classification={name: float(value[i]) for name, value in probabilities.items()},
            features=[float(logits["coarse"][i]), float(logits["local"][i])] + geometry.tolist(),
            layouts={DECODER: dict(valid=bool(estimate.valid), translation_rc=estimate.t_a_to_b_rc,
                offset_b_in_a_rc=-estimate.t_a_to_b_rc, diagnostics=diagnostics)},
            **pair_size_metadata(batch.mask_a[i], batch.mask_b[i]))))
    return score_heads(heads, rows) if heads is not None else rows


def extract_population(args, split, destination, model, identity, *, dataset=None, heads=None):
    """One frozen forward per batch; serialize all predictions before REAL GT."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    targets, metadata = {}, None
    if split == "real":
        metadata, arrays = load_prepared_cache(args.prepared_cache)
        expected_ids = [row["pair_id"] for row in metadata["pairs"]]
        if sum(row["strict"] and row["label"] for row in metadata["pairs"]) != 508:
            raise ValueError("strict547 must retain all 508 balanced REAL positives")
        batches = real.input_batches(metadata, arrays, MATCHER_BATCH)
        source = dict(prepared_cache=str(Path(args.prepared_cache).resolve()),
            prepared_manifest_sha256=metadata["manifest_sha256"])
    else:
        if dataset is None:
            dataset = RachelPairDataset(Path(args.dataset), split)
        expected_count = TRAIN_COUNT if split == "train" else VAL_COUNT
        if len(dataset) != expected_count:
            raise ValueError("wrong population size for " + split)
        expected_ids = None
        batches = make_ablation_loader(dataset, tuple(range(len(dataset))), batch_size=MATCHER_BATCH,
            num_workers=args.workers, seed=SEED, contour_cap=512)
        source = dict(dataset_root=str(Path(args.dataset).resolve()),
            train_manifest=str(Path(args.train_manifest).resolve()) if split == "train" else None)
    started, predictions = time.perf_counter(), []
    protocol = dict(schema_version=SCHEMA, status="running", split=split,
        matcher_checkpoint_id=identity["checkpoint_sha256"], checkpoint_sha256=identity["checkpoint_sha256"],
        precision="fp32", batch_size=MATCHER_BATCH, frozen_matcher=True,
        feature_names=list(FEATURE_NAMES), geometry_target_blind=True, layout_modified=False,
        selected_full_decoder=DECODER, decoder_config=asdict(TOP2_CONFIG),
        rotation_estimated=False, routing_used=False, seam_quality_evaluated=False,
        geometry_coverage_definition="unique predicted contour arc cells, not ground-truth seam coverage",
        translation_convention="t_a_to_b_rc=b-a; B placement=-t; 800-pixel model canvas",
        target_gt_evaluation_after_prediction_freeze=split == "real", **source)
    common.write_json(destination / "protocol.json", protocol)
    with (destination / "pair_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for batch in batches:
            predicted = predict_batch(model, batch, torch.device(args.device), heads)
            predictions.extend(predicted)
            for row in predicted:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if split != "real":
                for i, pair_id in enumerate(batch.pair_ids):
                    targets[pair_id] = dict(label=bool(batch.labels[i]), source_unit_ids=[],
                        translation_rc=common.clean(batch.translation_a_to_b_rc[i]) if batch.translation_valid[i] else None)
            stream.flush()
            if len(predictions) % 64 == 0:
                update_status(args, "cache_" + split, processed=len(predictions),
                    total=1016 if split == "real" else len(dataset), elapsed_s=time.perf_counter() - started)
        os.fsync(stream.fileno())
    pair_ids = [row["pair_id"] for row in predictions]
    if len(set(pair_ids)) != len(pair_ids) or (expected_ids is not None and pair_ids != expected_ids):
        raise ValueError("frozen predictions must cover the unique population in order")
    expected_n = 1016 if split == "real" else len(dataset)
    if len(predictions) != expected_n:
        raise ValueError("incomplete frozen prediction population")
    common.write_json(destination / "prediction_complete.json", dict(status="all_predictions_frozen",
        sample_count=len(predictions), matcher_checkpoint_id=identity["checkpoint_sha256"],
        translation_gt_json_opened=False, head_scores_frozen=heads is not None))
    rows = (real.attach_ground_truth(predictions, metadata["pairs"], args.translation_gt_json)
            if split == "real" else fixed.attach_test_targets(predictions, targets))
    positive = sum(row["label"] for row in rows)
    if positive != (508 if split == "real" else expected_n // 2):
        raise ValueError("population must be the registered balanced TRAIN/VAL/TEST/REAL pairs")
    fixed.write_rows(destination / "pair_results.jsonl", rows)
    protocol.update(status="complete", sample_count=len(rows), positive_count=positive,
        elapsed_s=time.perf_counter() - started)
    save_json(destination / "protocol.json", protocol)
    return rows, protocol


def train_heads(train_rows, val_rows, *, output, identity, original_thresholds, callback=None,
                epochs=EPOCHS, batch_size=HEAD_BATCH):
    """Fit only TRAIN. Select each head on clean VAL row-F1, then AP, then early epoch."""
    if set(row["pair_id"] for row in train_rows) & set(row["pair_id"] for row in val_rows):
        raise ValueError("TRAIN and VAL pair identities overlap")
    x, vx = feature_matrix(train_rows), feature_matrix(val_rows)
    y = torch.tensor([row["label"] for row in train_rows], dtype=torch.float32)
    vy = np.asarray([row["label"] for row in val_rows], bool)
    if len(np.unique(y.numpy())) != 2 or len(np.unique(vy)) != 2:
        raise ValueError("head TRAIN and VAL must both contain positive and negative rows")
    torch.manual_seed(SEED)
    reference = SeamGeometryPairHead()
    reference.set_training_normalization(x)
    heads = {name: SeamGeometryPairHead(SeamGeometryHeadConfig(use_geometry=name == "score_geometry"))
             for name in METHODS}
    for head in heads.values():
        head.load_state_dict(reference.state_dict(), strict=True)
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                  for name, head in heads.items()}
    best, states, selected, history = {}, {}, {}, []
    steps = 0
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(SEED + 1000003 * epoch).permutation(len(train_rows))
        totals = {name: 0.0 for name in METHODS}
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            for name, head in heads.items():
                head.train()
                optimizers[name].zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x[indices]), y[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite head loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
                optimizers[name].step()
                totals[name] += float(loss.detach()) * len(indices)
            steps += 1
        record = dict(epoch=epoch, pair_exposures_per_head=epoch * len(train_rows),
            optimizer_updates_per_head=steps, methods={})
        for name, head in heads.items():
            with torch.inference_mode():
                score = torch.sigmoid(head.eval()(vx)).numpy()
            threshold = common.fit_threshold(vy, score)
            metrics = common.classification(vy, score, threshold)
            key = (metrics["f1"], metrics["auprc"])
            record["methods"][name] = dict(train_loss=totals[name] / len(train_rows), **metrics)
            if name not in best or key > best[name]:
                best[name] = key
                states[name] = copy.deepcopy(head.state_dict())
                selected[name] = dict(epoch=epoch, threshold=threshold, val_f1=metrics["f1"],
                    val_auprc=metrics["auprc"], validation_metrics=metrics)
        history.append(record)
        save_json(Path(output) / ("epoch_%02d.json" % epoch), record)
        if callback:
            callback(record)
    for name, head in heads.items():
        head.load_state_dict(states[name], strict=True)
        head.eval().requires_grad_(False)
    freeze = dict(schema_version=SCHEMA, status="complete", primary_method="score_geometry",
        attribution_control="score_only", selected=selected,
        matcher_checkpoint_id=identity["checkpoint_sha256"], checkpoint_sha256=identity["checkpoint_sha256"],
        matcher_identity=identity, feature_names=list(FEATURE_NAMES), precision="fp32",
        original_branch_thresholds=original_thresholds, original_fused_threshold=original_thresholds["fused"],
        selection_rule="independently maximize clean VAL equal-row F1; AP breaks ties; earliest epoch breaks exact ties",
        threshold_rule="clean VAL equal-row F1; largest threshold among ties",
        primary_method_predeclared=True, pose_used_for_selection=False, test_or_real_used_for_fit=False,
        train_unique_count=len(train_rows), validation_sample_count=len(val_rows),
        validation_positive_count=int(vy.sum()), planned_epochs=epochs, completed_epochs=epochs,
        completed_pair_exposures_per_head=epochs * len(train_rows),
        completed_optimizer_updates_per_head=steps, head_batch_size=batch_size,
        head_training_device="cpu", matcher_inference_batch_size=MATCHER_BATCH,
        optimizer="AdamW", learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, seed=SEED,
        shared_initialization=True, shared_training_order=True, shared_training_budget=True,
        normalization_fit_split="train", heads={name: head.metadata() for name, head in heads.items()},
        selected_full_decoder=DECODER, decoder_config=asdict(TOP2_CONFIG), layout_modified=False,
        geometry_target_blind=True, hard_gate=False,
        budget_caveat="head exposures/updates only; matcher is frozen and its earlier training is separate")
    checkpoint = dict(schema_version=SCHEMA, matcher_checkpoint_id=identity["checkpoint_sha256"],
        feature_names=list(FEATURE_NAMES), selected=selected,
        heads={name: dict(config=asdict(head.config), state_dict=states[name]) for name, head in heads.items()})
    torch.save(checkpoint, Path(output) / "head.pt")
    freeze["head_checkpoint_sha256"] = sealed._sha256_file(Path(output) / "head.pt")
    freeze["head_checkpoint_path"] = str((Path(output) / "head.pt").resolve())
    freeze["head_checkpoint_paths"] = {name: freeze["head_checkpoint_path"] for name in METHODS}
    common.write_json(Path(output) / "validation_freeze.json", freeze)
    return heads, freeze


def load_frozen_heads(output, matcher_checkpoint_id):
    root = Path(output)
    freeze = json.loads((root / "validation_freeze.json").read_text(encoding="utf-8"))
    if (freeze.get("status") != "complete" or freeze.get("test_or_real_used_for_fit") is not False
            or freeze.get("matcher_checkpoint_id") != matcher_checkpoint_id
            or freeze.get("primary_method") != "score_geometry"
            or freeze.get("feature_names") != list(FEATURE_NAMES)
            or freeze.get("completed_epochs") != EPOCHS
            or freeze.get("completed_pair_exposures_per_head") != EPOCHS * TRAIN_COUNT
            or freeze.get("completed_optimizer_updates_per_head") != EPOCHS * math.ceil(TRAIN_COUNT / HEAD_BATCH)
            or freeze.get("head_batch_size") != HEAD_BATCH
            or freeze.get("precision") != "fp32"
            or freeze.get("train_unique_count") != TRAIN_COUNT
            or freeze.get("validation_sample_count") != VAL_COUNT
            or freeze.get("validation_positive_count") != 1500):
        raise ValueError("held-out E3 requires the complete matching TRAIN24k/VAL3000 head freeze")
    if sealed._sha256_file(root / "head.pt") != freeze["head_checkpoint_sha256"]:
        raise ValueError("head checkpoint differs from frozen head")
    checkpoint = sealed._torch_load_checkpoint(root / "head.pt")
    if (checkpoint.get("matcher_checkpoint_id") != matcher_checkpoint_id
            or checkpoint.get("selected") != freeze["selected"]
            or checkpoint.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("head checkpoint and freeze identities differ")
    heads = {}
    for name in METHODS:
        saved = checkpoint["heads"][name]
        head = SeamGeometryPairHead(SeamGeometryHeadConfig(**saved["config"]))
        if head.config.use_geometry != (name == "score_geometry"):
            raise ValueError("head geometry/control configuration differs")
        head.load_state_dict(saved["state_dict"], strict=True)
        heads[name] = head.eval().requires_grad_(False)
    return heads, freeze


def recall_at_fpr(labels, scores):
    """Empirical ROC diagnostic, whole tied-score groups; not a fitted policy."""
    y, s = np.asarray(labels, bool), np.asarray(scores, float)
    positive, negative = int(y.sum()), int((~y).sum())
    if not positive or not negative:
        return {str(value): None for value in (.01, .05, .1)}
    order = np.argsort(-s, kind="stable")
    ends = np.r_[np.flatnonzero(s[order][:-1] != s[order][1:]), len(s) - 1]
    tp, fp = np.cumsum(y[order])[ends], np.cumsum(~y[order])[ends]
    return {str(value): float(np.max(np.r_[0, tp[fp <= value * negative]]) / positive)
            for value in (.01, .05, .1)}


def summarize_methods(rows, freeze, *, include_strata=True):
    labels = np.asarray([r["label"] for r in rows], bool)
    thresholds = {name: freeze["selected"][name]["threshold"] for name in METHODS}
    thresholds["original_fused"] = freeze["original_fused_threshold"]
    result = dict(sample_count=len(rows), positive_count=int(labels.sum()),
        negative_count=int((~labels).sum()), methods={}, layout={})
    if not rows:
        result["metrics_status"] = "empty"
        return result
    original = common.summarize(rows, thresholds["original_fused"], freeze["original_branch_thresholds"])
    # Pose itself is shared. Remove classifier-dependent assembly from this
    # shared section; each head's joint is reported separately below.
    pose = copy.deepcopy(original["layout"][DECODER])
    pose.pop("assembly", None)
    result["layout"][DECODER] = pose
    result["original_classification"] = original["classification"]
    valid = np.array([r["layouts"][DECODER]["valid"] for r in rows], bool)
    errors = np.array([r["layouts"][DECODER]["translation_l2_px"]
        if r["layouts"][DECODER]["translation_l2_px"] is not None else np.inf for r in rows])
    for name, threshold in thresholds.items():
        score = np.asarray([r["head_scores"][name] for r in rows], float)
        accepted = score >= threshold
        metric = common.classification(labels, score, threshold)
        if len(np.unique(labels)) != 2:
            metric.update(auroc=None, auprc=1.0 if labels.all() else None)
        joint = {}
        for tolerance in (2, 5, 8, 10):
            tp = int((accepted & labels & valid & (errors <= tolerance)).sum())
            fp, fn = int(accepted.sum()) - tp, int(labels.sum()) - tp
            joint[str(tolerance)] = dict(tp=tp, fp=fp, fn=fn,
                precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                f1=2 * tp / max(1, 2 * tp + fp + fn))
        result["methods"][name] = dict(threshold=threshold, classification=metric,
            recall_at_fpr=recall_at_fpr(labels, score), joint=joint)
    if include_strata:
        result["size_strata"] = dict(fixed_bin_edges=[.25, .5], thresholds_refit=False,
            area_ratio_definition="smaller/larger nonzero filled-mask pixel count on model canvas800",
            groups={key: summarize_methods([r for r in rows if r["size_ratio_stratum"] == key],
                freeze, include_strata=False) for key in SIZE_RATIO_STRATA + (INVALID_SIZE_STRATUM,)})
    return result


def write_evaluation(destination, rows, split, freeze, identity, freeze_path, protocol=None):
    destination = Path(destination)
    if protocol is None:  # VAL reuses cached features, not a new matcher pass.
        destination.mkdir(parents=True, exist_ok=False)
        fixed.write_rows(destination / "pair_results.jsonl", rows)
    summary = summarize_methods(rows, freeze)
    if split == "real":
        strict_rows = [r for r in rows if r["strict_member"]]
        if len(strict_rows) != 547 or sum(r["label"] for r in strict_rows) != 508:
            raise ValueError("wrong strict REAL population")
        summary["strict_summary"] = summarize_methods(strict_rows, freeze)
        common.write_json(destination / "strict547_summary.json", summary["strict_summary"])
    provenance = dict(schema_version=SCHEMA, status="complete", split=split,
        population="real_balanced1016" if split == "real" else "synthetic_" + split + "3000",
        matcher_checkpoint_id=identity["checkpoint_sha256"], checkpoint_sha256=identity["checkpoint_sha256"],
        matcher_identity=identity, precision="fp32", validation_freeze=str(Path(freeze_path).resolve()),
        validation_freeze_sha256=sealed._sha256_file(freeze_path),
        head_checkpoint_sha256=freeze["head_checkpoint_sha256"], primary_method="score_geometry",
        head_checkpoint_path=str(Path(freeze_path).resolve().parent / "head.pt"),
        selected_full_decoder=DECODER, layout_modified=False, matcher_frozen=True,
        test_or_real_used_for_fit=False, target_gt_evaluation_after_prediction_freeze=split == "real",
        seam_quality_evaluated=False,
        seam_quality_unavailable="geometry features are predicted evidence, not annotated reference seam metrics",
        recall_at_fpr_policy="ranking diagnostic using whole tied-score groups; no threshold deployment/refitting")
    summary.update(provenance)
    common.write_json(destination / "summary.json", summary)
    save_json(destination / "protocol.json", dict(protocol or {}, **provenance))
    common.write_json(destination / "receipt.json", dict(provenance,
        pair_results_sha256=sealed._sha256_file(destination / "pair_results.jsonl")))
    return summary


def update_status(args, stage, **values):
    state = dict(schema_version=SCHEMA, status="running", stage=stage, pid=os.getpid(), **values)
    save_json(Path(args.output) / "status.json", state)
    emit(state)


def run(args):
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    torch.set_num_threads(1)
    sealed._set_determinism(SEED)
    root = Path(args.output).resolve()
    if args.evaluate_only:
        if not root.is_dir():
            raise ValueError("--evaluate-only requires an existing completed head fit")
    else:
        root.mkdir(parents=True, exist_ok=False)
    try:
        update_status(args, "load_frozen_e2")
        model, identity, thresholds = load_e2_winner(args.training_run)
        model.to(torch.device(args.device)).eval().requires_grad_(False)
        if args.evaluate_only:
            heads, freeze = load_frozen_heads(root, identity["checkpoint_sha256"])
        else:
            training, data_record = make_training_dataset(Path(args.dataset), Path(args.train_manifest))
            if len(training) != TRAIN_COUNT or data_record["unique_count"] != TRAIN_COUNT:
                raise ValueError("E3 requires the fixed unique matched TRAIN24k")
            if (Path(identity["matcher_training_data"]["manifest"]).resolve()
                    != Path(args.train_manifest).resolve()):
                raise ValueError("E3 must fit heads on the same matched TRAIN24k manifest as E2")
            protocol = dict(schema_version=SCHEMA, status="running", matcher_identity=identity,
                training_data=data_record,
                arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                primary_method="score_geometry",
                feature_names=list(FEATURE_NAMES), geometry_names=list(GEOMETRY_NAMES),
                planned_epochs=EPOCHS, head_batch_size=HEAD_BATCH,
                planned_pair_exposures_per_head=EPOCHS * TRAIN_COUNT,
                planned_optimizer_updates_per_head=EPOCHS * math.ceil(TRAIN_COUNT / HEAD_BATCH),
                target_blind_features=True, layout_modified=False, matcher_frozen=True,
                test_or_real_used_for_fit=False)
            common.write_json(root / "protocol.json", protocol)
            train_rows, _ = extract_population(args, "train", root / "cache" / "train", model, identity, dataset=training)
            del training
            val_rows, _ = extract_population(args, "val", root / "cache" / "val", model, identity)
            update_status(args, "fit_heads")
            heads, freeze = train_heads(train_rows, val_rows, output=root, identity=identity,
                original_thresholds=thresholds,
                callback=lambda record: update_status(args, "fit_heads", **record))
            # Exercise the exact serialized deployment interface before any
            # held-out dataset is constructed or translation GT is opened.
            heads, freeze = load_frozen_heads(root, identity["checkpoint_sha256"])
            score_heads(heads, val_rows)
            write_evaluation(root / "val", val_rows, "val", freeze, identity, root / "validation_freeze.json")
            del train_rows, val_rows
            protocol.update(status="fit_complete", head_freeze=str(root / "validation_freeze.json"))
            save_json(root / "protocol.json", protocol)
        if not args.fit_only:
            for split in ("test", "real"):
                destination = root / split
                if (destination / "summary.json").exists():
                    summary = json.loads((destination / "summary.json").read_text())
                    if (summary.get("status") != "complete"
                            or summary.get("matcher_checkpoint_id") != identity["checkpoint_sha256"]
                            or summary.get("validation_freeze_sha256") != sealed._sha256_file(root / "validation_freeze.json")):
                        raise ValueError("existing held-out summary is incomplete or belongs to another freeze")
                    continue
                update_status(args, "cache_" + split)
                rows, protocol = extract_population(args, split, destination, model, identity, heads=heads)
                write_evaluation(destination, rows, split, freeze, identity,
                    root / "validation_freeze.json", protocol)
        final_protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        final_protocol.update(status="fit_complete" if args.fit_only else "complete")
        save_json(root / "protocol.json", final_protocol)
        save_json(root / "status.json", dict(status="fit_complete" if args.fit_only else "complete",
            stage=None, matcher_checkpoint_id=identity["checkpoint_sha256"], primary_method="score_geometry"))
        emit(dict(status="fit_complete" if args.fit_only else "complete", output=str(root)))
    except Exception as error:
        save_json(root / "status.json", dict(status="failed", error=repr(error), pid=os.getpid()))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run", required=True, help="complete E2 winner and train_val_freeze")
    p.add_argument("--dataset", required=True, help="original release; clean VAL/TEST remain unchanged")
    p.add_argument("--train-manifest", required=True, help="fixed matched24k TRAIN manifest")
    p.add_argument("--prepared-cache", type=Path, default=DEFAULT_PREPARED_CACHE)
    p.add_argument("--translation-gt-json", type=Path, default=real.DEFAULT_TRANSLATION_GT)
    p.add_argument("--output", required=True, help="new E3 output; existing only with --evaluate-only")
    p.add_argument("--device", default="cuda:0", help="FP32 frozen matcher device; tiny heads fit on CPU")
    p.add_argument("--workers", type=int, default=4)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--fit-only", action="store_true")
    group.add_argument("--evaluate-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
