"""Saved-output C1/C2 residual intervention; no model loading, fit, or inference.

python decompose.py --endpoint C1:real=/path/to/complete/real --output /new/dir
Repeat --endpoint for other models/splits. Thresholds are the endpoint's own
frozen SIMVAL max_f1 and recall_95; neither branch is recalibrated here.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "endpoint_compare_v1/compare.py"
spec = importlib.util.spec_from_file_location("frozen_endpoint_compare_v1", SHARED)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
ATOL = 2e-6  # Float32 logit addition + sigmoid replayed with Python float64.
CAVEATS = [
    "The global branch was jointly updated during C1/C2 training; it is not C0.",
    "Removing the residual at the output is an inference-output intervention, not an independently trained ablation.",
    "Same own frozen SIMVAL threshold for both outputs; no REAL/OOD threshold fitting or separate global calibration.",
    "decision_valid is held fixed; no Sinkhorn, candidate selection, Matcher, or layout is rerun.",
    "C1/C2 global and residual branches can co-adapt; the flip counts do not identify a standalone causal training benefit.",
    "OOD has301 positives only and no layout GT; no F1, binary accuracy, or layout accuracy is reported."]


def sigmoid(value):
    if value >= 0:
        return 1. / (1. + math.exp(-value))
    z = math.exp(value)
    return z / (1. + z)


def reconstruct(rows, atol=ATOL):
    """Require actual exported branch outputs for every row; never invent zero."""
    issues, values, errors = defaultdict(list), {}, []
    for row in rows:
        pair_id = row["pair_id"]
        d = row.get("candidate_details")
        if not isinstance(d, dict):
            issues["candidate_details_missing"].append(pair_id)
            continue
        missing = [k for k in ("global_logit", "local_residual_logit", "local_eligible") if k not in d]
        if missing:
            issues["missing_fields:"+",".join(missing)].append(pair_id)
            continue
        g, delta, eligible = d["global_logit"], d["local_residual_logit"], d["local_eligible"]
        if not common.finite(g) or not common.finite(delta) or not isinstance(eligible, bool):
            issues["nonfinite_or_wrong_type_components"].append(pair_id)
            continue
        if not eligible and delta != 0:
            issues["ineligible_with_nonzero_applied_residual"].append(pair_id)
            continue
        total = g + delta
        if not math.isfinite(total):
            issues["nonfinite_sum"].append(pair_id)
            continue
        pg, pf = sigmoid(g), sigmoid(total)
        err = abs(pf-row["classification"]["fused"])
        errors.append((err, pair_id))
        if err > atol:
            issues["fused_probability_reconstruction_mismatch"].append(pair_id)
        values[pair_id] = {"global_probability": pg, "reconstructed_fused_probability": pf,
                           "local_eligible": eligible, "probability_absolute_error": err}
    diagnostics = {"absolute_tolerance": atol, "relative_tolerance": 0,
        "row_count": len(rows), "reconstructed_count": len(values),
        "max_probability_absolute_error": max((v[0] for v in errors), default=None),
        "mean_probability_absolute_error": sum(v[0] for v in errors)/len(errors) if errors else None,
        "worst_pair_id": max(errors)[1] if errors else None,
        "issues": {k: {"count": len(ids), "pair_ids": ids} for k, ids in issues.items()}}
    return values, diagnostics


def flips(rows, values, threshold, split):
    def accept(row, name):
        return row["decision_valid"] and values[row["pair_id"]][name] >= threshold
    # A tolerance-sized probability replay difference must not silently change a
    # threshold decision; preserve the ambiguity rather than fudge the score.
    disagreements = [r["pair_id"] for r in rows if accept(r, "reconstructed_fused_probability")
                     != common.accepted(r, threshold)]
    if disagreements:
        return {"status": "unavailable", "reason": "reconstruction_crosses_frozen_threshold",
                "pair_ids": disagreements, "threshold": threshold}
    rescued = lambda r: not accept(r, "global_probability") and accept(r, "reconstructed_fused_probability")
    lost = lambda r: accept(r, "global_probability") and not accept(r, "reconstructed_fused_probability")
    pack = lambda ids: {"count": len(ids), "pair_ids": ids}
    positive = [r for r in rows if r["label"]]
    result = {"status": "complete", "threshold": threshold, "sample_count": len(rows),
        "positive_count": len(positive), "negative_count": len(rows)-len(positive),
        "local_eligible_count": sum(values[r["pair_id"]]["local_eligible"] for r in rows),
        "decision_invalid_count": sum(not r["decision_valid"] for r in rows),
        "positive_rescued": pack([r["pair_id"] for r in positive if rescued(r)]),
        "positive_lost": pack([r["pair_id"] for r in positive if lost(r)]),
        "global_accepted_positive": sum(accept(r, "global_probability") for r in positive),
        "fused_accepted_positive": sum(accept(r, "reconstructed_fused_probability") for r in positive)}
    result["global_positive_recall"] = result["global_accepted_positive"]/len(positive) if positive else None
    result["fused_positive_recall"] = result["fused_accepted_positive"]/len(positive) if positive else None
    if split == "ood":
        result["unavailable_metrics"] = "positive-only OOD; no binary F1/Accuracy/AUROC/AP or GT layout metrics"
        return result
    negative = [r for r in rows if not r["label"]]
    result.update(negative_new_false_positive=pack([r["pair_id"] for r in negative if rescued(r)]),
                  negative_corrected_false_positive=pack([r["pair_id"] for r in negative if lost(r)]),
                  global_false_positive_count=sum(accept(r, "global_probability") for r in negative),
                  fused_false_positive_count=sum(accept(r, "reconstructed_fused_probability") for r in negative))
    missing = [r["pair_id"] for r in positive if common.layout_state(r) is None]
    good = [r for r in positive if common.layout_state(r) is True]
    result["correct_layout20"] = {"status": "partial" if missing else "complete",
        "missing_positive_pair_ids": missing, "known_correct_count": len(good),
        "raw_correct_count": None if missing else len(good),
        "known_positive_rescued": pack([r["pair_id"] for r in good if rescued(r)]),
        "known_positive_lost": pack([r["pair_id"] for r in good if lost(r)]),
        "known_global_rejected_correct": sum(not accept(r, "global_probability") for r in good),
        "known_fused_rejected_correct": sum(not accept(r, "reconstructed_fused_probability") for r in good)}
    return result


def decompose_endpoint(path, split):
    endpoint = common.load_endpoint(path, split)
    result = {k: v for k, v in endpoint.items() if k not in ("rows", "summary")}
    if endpoint["status"] != "complete":
        return result
    rows = list(endpoint["rows"].values())
    values, diagnostics = reconstruct(rows)
    result["reconstruction"] = diagnostics
    if diagnostics["issues"]:
        result.update(status="unavailable", reason="missing or inconsistent exported components; no rows dropped")
        return result
    result["populations"] = {}
    for group, selected in common.populations(rows, split).items():
        counts = (len(selected), sum(r["label"] for r in selected), sum(not r["label"] for r in selected))
        expected = common.REAL_COUNTS[group] if split == "real" else common.EXPECTED[split]
        if counts != expected:
            result["populations"][group] = {"status": "unavailable", "reason": "cohort_count_mismatch",
                                            "observed_counts": counts, "expected_counts": expected}
            result["status"] = "needs_attention"
            continue
        result["populations"][group] = {op: flips(selected, values, endpoint["thresholds"][op], split) for op in common.OPS}
        if any(v["status"] != "complete" for v in result["populations"][group].values()):
            result["status"] = "needs_attention"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", required=True, metavar="NAME:SPLIT=PATH")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = {}
    for item in args.endpoint:
        name_split, path = item.split("=", 1); name, split = name_split.rsplit(":", 1)
        if not name or split not in common.EXPECTED or name_split in inputs:
            parser.error("invalid/duplicate endpoint " + name_split)
        inputs[name_split] = (Path(path).expanduser().resolve(), split)
    if any(args.output.resolve() == p or p in args.output.resolve().parents for p, split in inputs.values()):
        parser.error("output must be separate from all source endpoints")
    results = {key: decompose_endpoint(path, split) for key, (path, split) in inputs.items()}
    result = {"schema": "candidate-output-residual-decomposition/1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if all(r["status"] == "complete" for r in results.values()) else "needs_attention",
        "reused_endpoint_comparator": {"path": str(SHARED), "sha256": hashlib.sha256(SHARED.read_bytes()).hexdigest()},
        "intervention": "sigmoid(global_logit) versus sigmoid(global_logit+local_residual_logit)",
        "threshold_rule": "endpoint own frozen SIMVAL threshold, unchanged across both outputs",
        "caveats": CAVEATS, "endpoints": results, "inference_performed": False, "threshold_fitting_performed": False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "endpoints": {k: {"status": v["status"], "reason": v.get("reason")}
                                                                   for k, v in results.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
