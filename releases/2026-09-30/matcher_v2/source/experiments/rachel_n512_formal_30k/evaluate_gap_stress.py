"""Frozen TEST-only gap stress: one model and one inward-erosion condition.

E0/E1/E2 retain their own completed clean-VAL winner and fused threshold. E3's
fixed geometry and score-only heads, when requested for E2, share its one
forward and unchanged Top2 layout. No fitting, routing, or decoder selection.
Requested 0/2/4 pixels is a maximum inward-erosion depth, not actual gap width.
Every original TEST row is retained, including skips, unchanged rows and failed
poses. --limit is a deterministic prefix probe, never a full-test result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
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
from experiments.rachel_n512_formal_30k import run_seam_geometry_head as geometry_head
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import save_json
from staging.pairwise_v0_2.models.physical_translation_layout import translated_intersection_area
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset, collate_rachel_pairs
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


SCHEMA = "rachel-gap-stress-evaluation/1"
SEED, BATCH_SIZE, TEST_COUNT, POSITIVE_COUNT = 260910, 8, 3000, 1500
DECODER = "full_top2_mode"
ARM_VARIANTS = dict(e0="matched24k", e1="edge_weathering_e1", e2="edge_consistency_e2")


def _stress_interfaces():
    from staging.pairwise_v0_2.pairwise_data.rachel_gap_stress import (
        RachelGapStressDataset, closing_direction_bias,
    )
    return RachelGapStressDataset, closing_direction_bias


@dataclass(frozen=True)
class GapStressBatch:
    batch: object
    reports: tuple
    metrics: tuple


def collate_gap_stress(samples):
    """Only student samples enter the ordinary batch; sidecars stay separate."""
    batch = collate_rachel_pairs([item.student for item in samples], contour_cap=512)
    return GapStressBatch(batch, tuple(item.report for item in samples), tuple(item.metrics for item in samples))


def load_frozen_arm(training_run, model_label, e3_head_root=None):
    if model_label not in ARM_VARIANTS:
        raise ValueError("model_label must identify E0, E1 or E2")
    if e3_head_root is not None and model_label != "e2":
        raise ValueError("E3 heads can only accompany their fixed E2 matcher")
    model, identity, thresholds = fixed.load_training_winner(training_run)
    checkpoint = sealed._torch_load_checkpoint(Path(identity["checkpoint_path"]))
    if checkpoint.get("variant") != ARM_VARIANTS[model_label]:
        raise ValueError("model_label differs from the actual completed training arm")
    identity["matcher_variant"] = checkpoint["variant"]
    heads, head_freeze = None, None
    if e3_head_root is not None:
        heads, head_freeze = geometry_head.load_frozen_heads(e3_head_root, identity["checkpoint_sha256"])
        if head_freeze["original_branch_thresholds"] != thresholds:
            raise ValueError("E3 freeze native thresholds differ from the fixed E2 winner")
        identity.update(e3_head_root=str(Path(e3_head_root).resolve()),
            e3_validation_freeze=str(Path(e3_head_root).resolve() / "validation_freeze.json"),
            e3_validation_freeze_sha256=sealed._sha256_file(Path(e3_head_root) / "validation_freeze.json"),
            head_checkpoint_sha256=head_freeze["head_checkpoint_sha256"])
    return model.eval().requires_grad_(False), identity, thresholds, heads, head_freeze


def predict_student_batch(model, batch, device, heads=None):
    """The same existing prediction paths; each calls the matcher exactly once."""
    if heads is not None:
        return geometry_head.predict_batch(model, batch, device, heads)
    return fixed.predict_batch(model, batch, device)


def mask_overlap_metrics(mask_a, mask_b, prediction_rc, prediction_valid, gt_rc=None):
    """Metric-only full-plane intersections, using B placement=-t in both cases.

    Ratios use the smaller *current stress-input* filled-mask area, not original
    scan area. GT overlap is undefined for negatives; invalid pose has no
    predicted overlap measurement, not a fabricated zero intersection.
    """
    ma, mb = np.asarray(mask_a).squeeze().astype(bool), np.asarray(mask_b).squeeze().astype(bool)
    area_a, area_b = int(ma.sum()), int(mb.sum())
    denominator = min(area_a, area_b)
    result = dict(area_a_px=area_a, area_b_px=area_b, denominator_px=denominator,
        denominator="smaller current stress filled-mask area on canvas800",
        gt_intersection_px=None, gt_overlap_ratio=None,
        predicted_intersection_px=None, predicted_overlap_ratio=None)
    if denominator <= 0:
        return result
    for name, translation, valid in (("gt", gt_rc, gt_rc is not None),
                                     ("predicted", prediction_rc, prediction_valid)):
        if valid and translation is not None:
            translation = np.asarray(translation, float)
            if translation.shape == (2,) and np.isfinite(translation).all():
                intersection = translated_intersection_area(ma, mb, translation)
                result[name + "_intersection_px"] = intersection
                result[name + "_overlap_ratio"] = intersection / denominator
    return result


def evaluate_batch_predictions(wrapped, predictions, closing_metric):
    """GT/normal sidecars first affect metrics after the student's prediction."""
    batch = wrapped.batch
    if len(predictions) != len(batch.pair_ids):
        raise ValueError("prediction batch does not match student rows")
    rows = []
    for i, prediction in enumerate(predictions):
        if prediction["pair_id"] != batch.pair_ids[i]:
            raise ValueError("prediction row order changed")
        report, sidecar = wrapped.reports[i], wrapped.metrics[i]
        if report["pair_id"] != prediction["pair_id"] or sidecar.pair_id != prediction["pair_id"]:
            raise ValueError("metric/weathering sidecars are not aligned with predictions")
        label = bool(batch.labels[i])
        gt = np.asarray(batch.translation_a_to_b_rc[i], float) if batch.translation_valid[i] else None
        layout = dict(prediction["layouts"][DECODER])
        layout["translation_l2_px"] = (float(np.linalg.norm(np.asarray(layout["translation_rc"], float) - gt))
            if layout["valid"] and gt is not None else None)
        closing = closing_metric(layout["translation_rc"] if layout["valid"] else None, sidecar)
        row = dict(prediction, label=label, target_translation_rc=common.clean(gt),
            layouts={DECODER: layout}, stress_report=common.clean(report),
            actual_changed=bool(report["changed_pair"]), closing_direction_bias=common.clean(closing),
            mask_overlap=mask_overlap_metrics(batch.mask_a[i], batch.mask_b[i],
                layout["translation_rc"], layout["valid"], gt))
        rows.append(common.clean(row))
    return rows


def _distribution(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
        median=float(np.median(values)) if len(values) else None,
        p10=float(np.quantile(values, .1)) if len(values) else None,
        p90=float(np.quantile(values, .9)) if len(values) else None)


def summarize_metric_sidecars(rows):
    positive = [row for row in rows if row["label"]]
    closing = [row["closing_direction_bias"].get("signed_closing_bias_px") for row in positive]
    valid = [value for value in closing if value is not None and np.isfinite(value)]
    support = [row["closing_direction_bias"].get("support", {}).get("coverage_fraction") for row in positive]
    invalid_reasons = Counter(str(row["closing_direction_bias"].get("invalid_reason") or "unspecified_no_measurement")
        for row in positive if row["closing_direction_bias"].get("signed_closing_bias_px") is None)
    close_summary = dict(_distribution(valid), positive_count=len(positive),
        positive_measurement_coverage=len(valid) / len(positive) if positive else None,
        positive_closing_count=sum(value > 0 for value in valid),
        negative_opening_count=sum(value < 0 for value in valid),
        zero_bias_count=sum(value == 0 for value in valid),
        invalid_reason_counts=dict(invalid_reasons),
        seam_normal_support_coverage=_distribution(support),
        definition="(predicted_t - original_gt_t) projected on clean A outward seam normals toward B; positive closes gap",
        sidecar_only=True, not_actual_gap_width=True)
    for key in ("positive_closing_component_px", "negative_opening_component_px",
                "closing_arc_fraction", "opening_arc_fraction", "neutral_arc_fraction"):
        close_summary[key] = _distribution([row["closing_direction_bias"].get(key) for row in positive])
    overlap = {name: _distribution([row["mask_overlap"].get(name) for row in positive])
               for name in ("gt_intersection_px", "gt_overlap_ratio", "predicted_intersection_px", "predicted_overlap_ratio")}
    overlap.update(positive_count=len(positive),
        denominator="smaller current stress filled-mask area; 800 model pixels",
        invalid_predictions_excluded_from_overlap_measurement_but_fail_pose_and_joint=True)
    return dict(closing_direction_bias=close_summary, mask_overlap=overlap)


def summarize_stress(rows):
    """Pre-outcome subset membership; unchanged/skipped pairs are not dropped."""
    changed = [row for row in rows if row["actual_changed"]]
    sides = [row["stress_report"]["side_" + side] for row in rows for side in "ab"]
    applied = [side for side in sides if side["applied"]]
    skipped = Counter(str(side.get("skip_reason")) for side in sides if side.get("skipped"))
    return dict(sample_count=len(rows), changed_pair_count=len(changed),
        unchanged_pair_count=len(rows) - len(changed),
        changed_positive_count=sum(row["label"] for row in changed),
        changed_negative_count=sum(not row["label"] for row in changed),
        actual_changed_pair_rate=len(changed) / len(rows) if rows else None,
        endpoint_count=len(sides), applied_endpoint_count=len(applied),
        actual_applied_endpoint_rate=len(applied) / len(sides) if sides else None,
        skipped_endpoint_count=sum(skipped.values()), skipped_endpoint_reasons=dict(skipped),
        applied_removed_area_px=_distribution([side.get("removed_area_px") for side in applied]),
        applied_removed_fraction=_distribution([side.get("removed_fraction") for side in applied]),
        subset_rule="changed_pair from deterministic mask augmentation, determined before any model outcome",
        requested_depth_is_actual_gap_width=False)


def summarize_rows(rows, thresholds, head_freeze=None, *, include_changed=True):
    positive = sum(row["label"] for row in rows)
    if not rows:
        return dict(sample_count=0, positive_count=0, negative_count=0, metrics_status="empty",
            classification=None, layout=None, methods={}, **summarize_metric_sidecars(rows))
    result = common.summarize(rows, thresholds["fused"], thresholds)
    result["negative_count"] = len(rows) - positive
    result["methods"] = {}
    methods = dict(native_fused=(thresholds["fused"], [r["classification"]["fused"] for r in rows]))
    if head_freeze is not None:
        for name in geometry_head.METHODS:
            methods[name] = (head_freeze["selected"][name]["threshold"], [r["head_scores"][name] for r in rows])
    labels = np.asarray([r["label"] for r in rows], bool)
    errors = np.asarray([r["layouts"][DECODER]["translation_l2_px"]
        if r["layouts"][DECODER]["translation_l2_px"] is not None else np.inf for r in rows])
    valid = np.asarray([r["layouts"][DECODER]["valid"] for r in rows], bool)
    for name, (threshold, values) in methods.items():
        score = np.asarray(values, float)
        metrics = common.classification(labels, score, threshold)
        if len(np.unique(labels)) != 2:
            metrics.update(auroc=None, auprc=1.0 if labels.all() else None)
        accepted = score >= threshold
        joint = {}
        for tolerance in (2, 5, 8, 10):
            tp = int((accepted & labels & valid & (errors <= tolerance)).sum())
            fp, fn = int(accepted.sum()) - tp, positive - tp
            joint[str(tolerance)] = dict(tp=tp, fp=fp, fn=fn,
                precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                f1=2 * tp / max(1, 2 * tp + fp + fn))
        result["methods"][name] = dict(threshold=threshold, classification=metrics, joint=joint,
            recall_at_fpr=geometry_head.recall_at_fpr(labels, score))
    result["stress"] = summarize_stress(rows)
    result.update(summarize_metric_sidecars(rows))
    if include_changed:
        changed = [row for row in rows if row["actual_changed"]]
        result["actual_changed_subset"] = summarize_rows(changed, thresholds, head_freeze, include_changed=False)
        result["actual_changed_subset"].update(
            subset_rule="stress_report.changed_pair before model prediction; not successful cases",
            all_pair_coverage=len(changed) / len(rows),
            all_positive_coverage=sum(r["label"] for r in changed) / positive if positive else None)
    return result


def evaluation_scope(limit, total=TEST_COUNT):
    if type(limit) is not int or limit < 0 or limit > total:
        raise ValueError("limit must be zero (full) or a positive deterministic prefix <= TEST3000")
    return dict(full_test=limit == 0, probe_only=limit != 0,
        evaluated_pair_count=limit or total, total_test_pair_count=total,
        population="synthetic_test3000" if limit == 0 else "synthetic_test_prefix_probe")


def run(args):
    scope = evaluation_scope(args.limit)
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    rows, protocol = [], None
    try:
        torch.set_num_threads(1)
        sealed._set_determinism(SEED)
        model, identity, thresholds, heads, head_freeze = load_frozen_arm(
            args.training_run, args.model_label, args.e3_head_root)
        model.to(torch.device(args.device)).eval().requires_grad_(False)
        dataset_root = Path(args.dataset).resolve()
        base = RachelPairDataset(dataset_root, "test")
        manifest_path = dataset_root / "pairs" / "test.jsonl"
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = [json.loads(line) for line in stream if line.strip()]
        if len(base) != TEST_COUNT or len(manifest) != TEST_COUNT or sum(row["label"] for row in manifest) != POSITIVE_COUNT:
            raise ValueError("gap stress requires the unchanged balanced original TEST3000 population")
        expected = manifest[:scope["evaluated_pair_count"]]
        stress_dataset_type, closing_metric = _stress_interfaces()
        stress_dataset = stress_dataset_type(base, max_depth_px=args.depth, seed=SEED, cache_dir=args.cache_dir)
        loader = runner._loader(stress_dataset, tuple(range(len(expected))), batch_size=BATCH_SIZE,
            num_workers=args.workers, seed=SEED)
        loader.collate_fn = collate_gap_stress
        protocol = dict(identity, **scope, schema_version=SCHEMA, status="running", split="test",
            model_label=args.model_label, depths=[args.depth], requested_max_inward_erosion_depth_px=args.depth,
            requested_depth_is_actual_gap_width=False, seed=SEED, precision="fp32", batch_size=BATCH_SIZE,
            dataset_root=str(dataset_root), test_manifest_sha256=sealed._sha256_file(manifest_path),
            augmentation_cache=str(Path(args.cache_dir).resolve()) if args.cache_dir else None,
            original_fused_threshold=thresholds["fused"], branch_validation_thresholds=thresholds,
            selected_full_decoder=DECODER, decoder_config=asdict(fixed.TOP2_CONFIG),
            matcher_frozen=True, all_rows_retained=scope["full_test"],
            all_declared_population_rows_retained=True, all_pairs_decoded=True,
            single_model_forward_per_batch=True, e3_shares_e2_forward=heads is not None,
            e3_head_methods=list(geometry_head.METHODS) if heads is not None else [],
            e3_thresholds={name: head_freeze["selected"][name]["threshold"] for name in geometry_head.METHODS} if head_freeze else None,
            test_or_real_used_for_fit=False, layout_modified=False, rotation_estimated=False, routing_used=False,
            student_forward_fields=["mask_a", "mask_b", "points_rc_a", "points_rc_b", "contour_valid_a", "contour_valid_b"],
            gt_used_in_prediction=False, metric_sidecar_used_in_prediction=False,
            gt_metric_timing="original TEST targets are loader-provided; metrics run only after each batch prediction",
            translation_convention="t_a_to_b_rc=b-a; B placement=-t; unchanged800 model frame and original GT",
            changed_subset_rule="actual changed_pair mask report before model outcomes",
            no_fresh_holdout_claim="TEST was used in previous experiments; this is a frozen stress diagnostic")
        common.write_json(destination / "protocol.json", protocol)
        started = time.perf_counter()
        with (destination / "pair_predictions.jsonl").open("x", encoding="utf-8") as stream:
            for wrapped in loader:
                predictions = predict_student_batch(model, wrapped.batch, torch.device(args.device), heads)
                for prediction in predictions:
                    stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                # This function cannot alter a prediction or a model input.
                rows.extend(evaluate_batch_predictions(wrapped, predictions, closing_metric))
                stream.flush()
                if len(rows) % 64 == 0 or len(rows) == len(expected):
                    state = dict(status="running", processed=len(rows), total=len(expected),
                        model_label=args.model_label, depth=args.depth, elapsed_s=time.perf_counter() - started,
                        **scope)
                    save_json(destination / "status.json", state)
                    print(json.dumps(state), flush=True)
            os.fsync(stream.fileno())
        if [row["pair_id"] for row in rows] != [row["pair_id"] for row in expected]:
            raise ValueError("stress evaluation changed original TEST row order or count")
        if [row["label"] for row in rows] != [bool(row["label"]) for row in expected]:
            raise ValueError("stress evaluation changed original TEST pair labels")
        common.write_json(destination / "prediction_complete.json", dict(status="all_predictions_frozen",
            checkpoint_sha256=identity["checkpoint_sha256"], **scope))
        fixed.write_rows(destination / "pair_results.jsonl", rows)
        summary = summarize_rows(rows, thresholds, head_freeze)
        protocol.update(status="complete", elapsed_s=time.perf_counter() - started,
            actual_changed_pair_count=summary["stress"]["changed_pair_count"])
        summary.update(protocol)
        save_json(destination / "protocol.json", protocol)
        common.write_json(destination / "receipt.json", dict(protocol,
            pair_predictions_sha256=sealed._sha256_file(destination / "pair_predictions.jsonl"),
            pair_results_sha256=sealed._sha256_file(destination / "pair_results.jsonl")))
        common.write_json(destination / "summary.json", summary)
        save_json(destination / "status.json", dict(status="complete", **scope,
            model_label=args.model_label, depths=[args.depth], elapsed_s=protocol["elapsed_s"]))
        print(json.dumps(dict(status="complete", output=str(destination), **scope)), flush=True)
    except Exception as error:
        state = dict(status="failed", error=repr(error), processed=len(rows), **scope)
        save_json(destination / "status.json", state)
        if protocol is not None:
            save_json(destination / "protocol.json", dict(protocol, status="failed", error=repr(error)))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run", required=True)
    p.add_argument("--model-label", required=True, choices=tuple(ARM_VARIANTS))
    p.add_argument("--dataset", required=True, help="original released TEST3000, never a composite TRAIN manifest")
    p.add_argument("--depth", required=True, type=int, choices=(0, 2, 4), help="max inward erosion depth, NOT gap width")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--e3-head-root", type=Path, help="complete E3 head freeze; E2 only; shared E2 forward")
    p.add_argument("--output", required=True, help="new destination, one arm and one depth")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="deterministic first N rows, marked probe even when N=3000")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
