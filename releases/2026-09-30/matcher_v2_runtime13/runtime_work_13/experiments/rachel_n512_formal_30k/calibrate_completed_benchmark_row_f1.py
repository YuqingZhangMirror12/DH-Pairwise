"""CPU-only row-F1 calibration of completed PairingNet/ShreddingNet predictions.

Validation predictions alone choose thresholds, with the exact decision rule of
run_layout_decoder_experiment.fit_threshold (used by the architecture runner):
score >= threshold, maximum equal-row F1, ties prefer the higher threshold.
The validation freeze is written before opening any test or real input. No model
is imported, no prediction is regenerated, and historical artifacts are untouched.

Example (the defaults point at this experiment's existing local inputs):
    python3 -m experiments.rachel_n512_formal_30k.calibrate_completed_benchmark_row_f1
    python3 -m experiments.rachel_n512_formal_30k.calibrate_completed_benchmark_row_f1 --full-top2-only

Each JSON result is created exclusively. An existing complete result is reused;
a partial output is not overwritten. Use a new --output-root for another run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/pairwise_ablation_v3_20260907/benchmark_row_f1"
METHODS = ("pairingnet_adapted", "shreddingnet_adapted")
RULE = "maximize_equal_row_f1_then_higher_threshold; accept_score_ge_threshold"
RULE_SOURCE = "experiments/rachel_n512_formal_30k/run_layout_decoder_experiment.py:fit_threshold"
VALIDATION_REMOTE_SOURCES = {
    "pairingnet_adapted": "/root/autodl-tmp/rachel_same_data_benchmark_direct_20260904_001/pairingnet_train/winner_validation_predictions.jsonl",
    "shreddingnet_adapted": "/root/autodl-tmp/rachel_same_data_benchmark_direct_20260904_001/shreddingnet_train/validation_threshold_scores.jsonl",
}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def check_pair_grain(rows, expected_count, positive_count):
    """Only pair-grain/population checks, not a general artifact audit."""
    ids = [row["pair_id"] for row in rows]
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        raise ValueError("Expected %d unique pair rows" % expected_count)
    if any(not isinstance(pair_id, str) or not pair_id for pair_id in ids):
        raise ValueError("Each row needs a nonempty pair_id")
    if any(type(row["label"]) is not bool for row in rows):
        raise ValueError("Expected boolean pair labels")
    if sum(row["label"] for row in rows) != positive_count:
        raise ValueError("Unexpected positive pair count")
    for row in rows:
        score = row["score"]
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Expected finite pair probabilities in [0,1]")
    return {row["pair_id"]: row["label"] for row in rows}


def score_groups(rows):
    groups = {}
    for row in rows:
        positive, negative = groups.get(row["score"], (0, 0))
        groups[row["score"]] = (positive + int(row["label"]), negative + int(not row["label"]))
    return sorted(groups.items(), reverse=True)


def fit_validation_threshold(validation_rows):
    """Same grouped descending sweep and first-maximum tie rule as the runner."""
    positives = sum(row["label"] for row in validation_rows)
    tp = predicted = 0
    best_f1, threshold = -1.0, None
    for score, (pos, neg) in score_groups(validation_rows):
        tp += pos
        predicted += pos + neg
        f1 = 2 * tp / max(1, predicted + positives)
        if f1 > best_f1:
            best_f1, threshold = f1, float(score)
    if threshold is None:
        raise ValueError("Empty validation predictions")
    return threshold


def classification(rows, threshold):
    positives = sum(row["label"] for row in rows)
    negatives = len(rows) - positives
    tp = sum(row["label"] and row["score"] >= threshold for row in rows)
    fp = sum(not row["label"] and row["score"] >= threshold for row in rows)
    fn, tn = positives - tp, negatives - fp
    result = dict(threshold=threshold, auroc=None, auprc=None, f1=2 * tp / max(1, 2 * tp + fp + fn),
                  recall=tp / max(1, positives), precision=tp / max(1, tp + fp),
                  accuracy=(tp + tn) / len(rows), tp=tp, fp=fp, fn=fn, tn=tn)
    # Tied-score trapezoidal AUROC and grouped average precision, matching the
    # runner's training.metrics._ranking_metrics without importing torch.
    if positives and negatives:
        tp = fp = 0
        previous_tpr = previous_fpr = auroc = auprc = 0.0
        for _, (pos, neg) in score_groups(rows):
            tp += pos
            fp += neg
            tpr, fpr = tp / positives, fp / negatives
            auroc += (fpr - previous_fpr) * (tpr + previous_tpr) * 0.5
            auprc += (tpr - previous_tpr) * tp / (tp + fp)
            previous_tpr, previous_fpr = tpr, fpr
        result.update(auroc=auroc, auprc=auprc)
    return result


def assembly_at_10px(rows, threshold):
    if any("pose_valid" not in row or "translation_error_px" not in row for row in rows if row["label"]):
        return dict(status="missing_positive_geometry")
    positives = sum(row["label"] for row in rows)
    selected = [row for row in rows if row["score"] >= threshold]
    tp = sum(row["label"] and row["pose_valid"] and row["translation_error_px"] is not None
             and row["translation_error_px"] <= 10 for row in selected)
    fp, fn = len(selected) - tp, positives - tp
    return dict(status="complete", threshold=threshold, tolerance_px=10, tp=tp, fp=fp, fn=fn,
                f1=2 * tp / max(1, 2 * tp + fp + fn), precision=tp / max(1, tp + fp),
                recall=tp / max(1, positives),
                definition="accepted positive with valid pose <=10px; wrong/invalid accepted pose is FP+FN")


def population_result(rows, original_threshold, row_threshold):
    return dict(sample_count=len(rows), positive_count=sum(row["label"] for row in rows),
                classification={
                    "at_historical_cluster_balanced_f1_threshold": classification(rows, original_threshold),
                    "at_validation_row_f1_threshold": classification(rows, row_threshold)},
                assembly_f1_at_10px={
                    "at_historical_cluster_balanced_f1_threshold": assembly_at_10px(rows, original_threshold),
                    "at_validation_row_f1_threshold": assembly_at_10px(rows, row_threshold)})


def load_validation(path, method):
    field = "pair_probability" if method == "pairingnet_adapted" else "pair_score"
    rows = [dict(pair_id=row["pair_id"], label=row["label"], score=row[field]) for row in read_jsonl(path)]
    check_pair_grain(rows, 3000, 1500)
    return rows


def load_test(path):
    rows = []
    for raw in read_jsonl(path):
        score = raw["scores"][raw["main_score"]]
        if score["valid"] is not True or raw["decision"]["valid"] is not True:
            raise ValueError("This completed benchmark population must have valid pair scores")
        geometry = raw.get("geometry", {})
        rows.append(dict(pair_id=raw["pair_id"], label=raw["label"], score=score["probability"],
                         pose_valid=geometry.get("translation_prediction_valid", False),
                         translation_error_px=geometry.get("translation_l2_px")))
    check_pair_grain(rows, 3000, 1500)
    return rows


def load_real(document, method, geometry_document):
    geometry = None if geometry_document is None else {row["pair_id"]: row[method] for row in geometry_document["positive_pairs"]}
    strict_labels = {row["pair_id"]: row["label"] for row in document["strict_547"]["pairs"]}
    if len(strict_labels) != 547 or len(document["strict_547"]["pairs"]) != 547:
        raise ValueError("Expected 547 unique strict pair IDs")
    rows = []
    for raw in document["balanced_1016"]["pairs"]:
        value = raw["methods"][method]
        if value["valid"] is not True:
            raise ValueError("This completed benchmark population must have valid pair scores")
        row = dict(pair_id=raw["pair_id"], label=raw["label"], score=value["probability"])
        if raw["label"] and geometry is not None:
            item = geometry[row["pair_id"]]
            # pair-ID join of already evaluated geometry; never rerun a decoder.
            row.update(pose_valid=item["translation_prediction_valid"],
                       translation_error_px=item["translation_l2_error_px"])
        rows.append(row)
    labels = check_pair_grain(rows, 1016, 508)
    if any(labels.get(pair_id) != label for pair_id, label in strict_labels.items()):
        raise ValueError("Strict pairs must be an identically labelled subset of balanced predictions")
    strict = [row for row in rows if row["pair_id"] in strict_labels]
    check_pair_grain(strict, 547, 508)
    if geometry is not None and set(geometry) != {row["pair_id"] for row in strict if row["label"]}:
        raise ValueError("Geometry must cover exactly the 508 strict positive pairs")
    return rows, strict


def load_full_cached(directory, split):
    """Stream only Full fused probability and cached Top2 layout from each pair."""
    summary = read_json(directory / "summary.json")
    if summary.get("status") != "complete" or summary.get("split") != split:
        raise ValueError("Expected completed Full cached " + split + " predictions")
    rows = []
    with (directory / "pair_results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            layout = raw["layouts"]["full_top2_mode"]
            row = dict(pair_id=raw["pair_id"], label=raw["label"], score=raw["classification"]["fused"],
                       pose_valid=layout["valid"], translation_error_px=layout["translation_l2_px"])
            if split == "real":
                if type(raw.get("strict_member")) is not bool:
                    raise ValueError("Real cache needs explicit boolean strict_member")
                row["strict_member"] = raw["strict_member"]
            rows.append(row)
    check_pair_grain(rows, 1016 if split == "real" else 3000, 508 if split == "real" else 1500)
    return rows, summary


def run_full_top2(args):
    """Separate Full anchor; never reads or changes the Pairing/Shred outputs."""
    sources = {split: str(path) for split, path in (("validation", args.full_validation_dir),
               ("test", args.full_test_dir), ("real", args.full_real_dir))}
    output = args.output_root / "full_top2_row_f1.json"
    if output.exists():
        existing = read_json(output)
        if existing.get("status") == "complete" and existing.get("rule") == RULE and existing.get("sources") == sources:
            return existing
        raise ValueError("Existing Full output is incomplete or belongs to different inputs")
    args.output_root.mkdir(parents=True, exist_ok=True)
    freeze_path = args.output_root / "full_top2_validation_row_f1_freeze.json"
    if freeze_path.exists():
        raise ValueError("Partial Full output; use a new output root instead of overwriting")
    # Validation-only load and threshold fit. Test/real are opened only below the
    # exclusive freeze write, not for choosing a threshold or a layout decoder.
    validation, val_summary = load_full_cached(args.full_validation_dir, "val")
    threshold = fit_validation_threshold(validation)
    authority = read_json(args.full_validation_dir / "validation_freeze.json")
    original_threshold = authority["original_fused_threshold"]
    write_new(freeze_path, dict(status="complete_validation_only", source_split="validation",
              sample_count=3000, positive_count=1500, rule=RULE, rule_source=RULE_SOURCE,
              threshold=threshold, original_threshold=original_threshold,
              validation_metrics=classification(validation, threshold), sources=sources,
              test_or_real_used_for_fit=False, test_or_real_opened_before_threshold_freeze=False,
              created_at=datetime.now(timezone.utc).isoformat()))
    test, test_summary = load_full_cached(args.full_test_dir, "test")
    real, real_summary = load_full_cached(args.full_real_dir, "real")
    strict = [row for row in real if row["strict_member"]]
    check_pair_grain(strict, 547, 508)
    result = dict(status="complete", method="full_n512_with_full_top2_mode", rule=RULE,
                  rule_source=RULE_SOURCE, row_f1_threshold=threshold, original_threshold=original_threshold,
                  sources=sources, validation_freeze=str(freeze_path),
                  populations={name: population_result(rows, original_threshold, threshold)
                               for name, rows in (("test3000", test), ("balanced1016", real), ("strict547", strict))},
                  cache_precision={name: summary.get("precision") for name, summary in
                                   (("validation", val_summary), ("test", test_summary), ("real", real_summary))},
                  test_or_real_used_for_fit=False, predictions_or_geometry_modified=False,
                  historical_artifacts_overwritten=False, model_or_gpu_executed=False,
                  cached_pose_semantics="original cached full_top2_mode layouts unchanged; assembly recomputed from each pair's score, valid flag and error, not copied from original-threshold summary; invalid poses remain failures",
                  assembly_population_note="strict547 primary real joint metric; balanced1016 is selected-list diagnostic with constructed distractors",
                  created_at=datetime.now(timezone.utc).isoformat())
    write_new(output, result)
    return result


def run(args):
    sources = dict(pairing_validation=str(args.pairing_validation), shredding_validation=str(args.shredding_validation),
                   exact6_root=str(args.exact6_root))
    summary_path = args.output_root / "summary.json"
    if summary_path.exists():
        existing = read_json(summary_path)
        if existing.get("status") == "complete" and existing.get("rule") == RULE and existing.get("sources") == sources:
            return existing  # Reuse, including original timestamps; no rewrites.
        raise ValueError("Existing output is incomplete or belongs to different inputs")
    args.output_root.mkdir(parents=True, exist_ok=True)
    freeze_path = args.output_root / "validation_row_f1_freeze.json"
    if freeze_path.exists():
        raise ValueError("Partial previous run; use a new output root rather than overwrite its freeze")

    # No test/real file is opened before both validation-only thresholds freeze.
    validation = {method: load_validation(path, method) for method, path in zip(
        METHODS, (args.pairing_validation, args.shredding_validation))}
    if {row["pair_id"]: row["label"] for row in validation[METHODS[0]]} != {
            row["pair_id"]: row["label"] for row in validation[METHODS[1]]}:
        raise ValueError("Benchmark methods must share the same validation pair population")
    thresholds = {method: fit_validation_threshold(rows) for method, rows in validation.items()}
    freeze = dict(status="complete_validation_only", rule=RULE, rule_source=RULE_SOURCE,
                  created_at=datetime.now(timezone.utc).isoformat(), source_split="validation", sample_count=3000,
                  positive_count=1500, test_or_real_used_for_fit=False,
                  test_or_real_opened_before_threshold_freeze=False, sources=sources,
                  validation_input_remote_origins=VALIDATION_REMOTE_SOURCES,
                  methods={method: dict(threshold=thresholds[method],
                                        validation_metrics=classification(validation[method], thresholds[method]))
                           for method in METHODS})
    write_new(freeze_path, freeze)

    # Downstream data can only evaluate the frozen decisions.
    historical = read_json(args.exact6_root / "synthetic_summary.json")
    real = read_json(args.exact6_root / "real_pair_only.json")
    real_geometry_path = args.exact6_root / "real_translation_gt.json"
    real_geometry = read_json(real_geometry_path) if real_geometry_path.exists() else None
    methods = {}
    for method in METHODS:
        original_threshold = historical["methods"][method]["threshold"]
        test = load_test(args.exact6_root / "synthetic_stage" / method / "pair_scores.jsonl")
        real_rows, strict = load_real(real, method, real_geometry)
        methods[method] = dict(original_threshold=original_threshold, row_f1_threshold=thresholds[method],
                              populations={name: population_result(rows, original_threshold, thresholds[method])
                                           for name, rows in (("test3000", test), ("balanced1016", real_rows), ("strict547", strict))})
    summary = dict(status="complete", created_at=datetime.now(timezone.utc).isoformat(), rule=RULE,
                   rule_source=RULE_SOURCE, sources=sources, validation_freeze=str(freeze_path), methods=methods,
                   test_or_real_used_for_fit=False, predictions_or_geometry_modified=False,
                   historical_artifacts_overwritten=False, model_or_gpu_executed=False,
                   population_notes={"test3000": "1500 positive + 1500 negative, equal-row metrics",
                                     "balanced1016": "508 positive + 508 negative; includes constructed distractors, not all GT negatives",
                                     "strict547": "508 identical positive pairs + 39 authoritative GT negatives; descriptive classification",
                                     "assembly": "strict547 primary real assembly; balanced1016 assembly is selected-list diagnostic only"},
                   geometry_source=str(real_geometry_path) if real_geometry is not None else None,
                   cached_pose_semantics="decision-threshold postprocessing with original cached layouts unchanged; no pose recomputed for newly accepted pairs; missing/invalid cached poses remain failures",
                   limitations=["same-data mask-only N512 upright adaptations, not exact paper reproduction",
                                "new calibration only; no new training or model predictions",
                                "historical predictions retain their original per-model inference precision",
                                "joint metrics are reference values under original cached pose availability, which may retain adapter-side gates"])
    write_new(summary_path, summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing-validation", type=Path, default=DEFAULT_OUTPUT / "inputs/winner_validation_predictions.jsonl")
    parser.add_argument("--shredding-validation", type=Path, default=DEFAULT_OUTPUT / "inputs/validation_threshold_scores.jsonl")
    parser.add_argument("--exact6-root", type=Path, default=ROOT / "reports/pairwise_exact6_20260906")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--full-top2-only", action="store_true", help="Write a separate Full+Top2 calibrated anchor; do not touch Pairing/Shred outputs")
    base = ROOT / "reports/pairwise_ablation_v3_20260907"
    parser.add_argument("--full-validation-dir", type=Path, default=base / "frozen_seam_val")
    parser.add_argument("--full-test-dir", type=Path, default=base / "frozen_seam_test")
    parser.add_argument("--full-real-dir", type=Path, default=base / "frozen_seam_real")
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


if __name__ == "__main__":
    args = parse_args()
    result = run_full_top2(args) if args.full_top2_only else run(args)
    filename = "full_top2_row_f1.json" if args.full_top2_only else "summary.json"
    thresholds = {"full_top2_mode": result["row_f1_threshold"]} if args.full_top2_only else {
        key: value["row_f1_threshold"] for key, value in result["methods"].items()}
    print(json.dumps(dict(summary=str(args.output_root / filename), thresholds=thresholds)))
