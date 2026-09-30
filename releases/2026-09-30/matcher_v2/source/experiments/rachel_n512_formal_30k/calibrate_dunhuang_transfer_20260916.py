"""Post-hoc REAL calibration -> OOD transfer; never edits benchmark or training.

Only fixed epoch20 scores are used. Thresholds are fitted before OOD evaluation.
REAL is an in-sample calibration population in this experiment, not a test set.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/autodl-tmp/rachel_score_design_20260913_001"
MODEL_ROOTS = {
    "s4_depth1": "new_s345_20260914/s4_cross_attention",
    "s6_depth2": "attention_depth_20260915/s4_cross_attention_depth2",
    "s6_depth4": "attention_depth_20260915/s4_cross_attention_depth4",
    "s7_augmented_full24": "s6_s7_20260915/priority_after_s5/s7_augmented_full24",
}


def collect():
    # Read-only remote extraction: do not copy large candidate/contour arrays.
    source = '''import json, pathlib
root = pathlib.Path(ROOT)
models = []
for name, relative in MODEL_ROOTS.items():
    base = root / relative / "evaluation" / "fixed_epoch"
    item = {"name": name, "sources": {}}
    for split in ("real", "ood"):
        path = base / split / "pair_results.jsonl"
        item[split + "_rows"] = rows = []
        item["sources"][split] = str(path)
        for line in path.open():
            r = json.loads(line)
            if split == "real" and r["label"] and r.get("review_status") != "keep":
                continue
            pose = r["layouts"]["full_top2_mode"]
            rows.append({"pair_id": r["pair_id"], "label": r["label"],
                "score": r["classification"]["fused"],
                "decision_valid": r["decision_valid"],
                "case_cluster": r.get("case_cluster"),
                "strict_member": r.get("strict_member"),
                "review_status": r.get("review_status"),
                "layout_valid": pose["valid"],
                "translation_l2_px": pose.get("translation_l2_px")})
    models.append(item)
print(json.dumps({"models": models}, ensure_ascii=False))
'''
    source = "ROOT = " + repr(REMOTE_ROOT) + "\nMODEL_ROOTS = " + repr(MODEL_ROOTS) + "\n" + source
    command = ["ssh", "-S", "none", "-i", str(Path.home() / ".ssh/id_ed25519_rachel_benchmark_20260904"),
        "-p", "42993", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=2",
        "root@connect.westb.seetacloud.com",
        "/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python -"]
    result = subprocess.run(command, input=source, text=True, capture_output=True, check=True, timeout=120)
    data = json.loads(result.stdout)
    data["captured_at"] = datetime.now(timezone.utc).isoformat()
    data["schema"] = "dunhuang-calibration-inputs/1"
    return data


def metrics(rows, threshold):
    tp = fp = fn = tn = 0
    raw_correct = accepted_correct = 0
    for r in rows:
        accepted = r["score"] >= threshold
        positive = bool(r["label"])
        tp += positive and accepted
        fp += not positive and accepted
        fn += positive and not accepted
        tn += not positive and not accepted
        correct = positive and r.get("layout_valid", False) and r.get("translation_l2_px") is not None and r["translation_l2_px"] <= 20
        raw_correct += correct
        accepted_correct += correct and accepted
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=(tp + tn) / len(rows), precision=tp / max(1, tp + fp),
        recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn),
        fpr=fp / max(1, fp + tn), raw_layout20_correct=raw_correct,
        accepted_layout20_correct=accepted_correct)


def fit_thresholds(real_rows, target=.95):
    """Ties stay together. Higher threshold breaks metric ties.

    The high-recall rule maximizes precision over ALL feasible thresholds,
    not simply the largest threshold attaining the desired recall.
    """
    if not real_rows or not 0 < target <= 1:
        raise ValueError("need rows and valid recall target")
    positives = sum(bool(r["label"]) for r in real_rows)
    if positives in (0, len(real_rows)):
        raise ValueError("calibration requires both classes")
    candidates = [metrics(real_rows, t) for t in sorted({r["score"] for r in real_rows}, reverse=True)]
    max_f1 = max(candidates, key=lambda m: (m["f1"], m["threshold"]))
    high_recall = max((m for m in candidates if m["recall"] >= target),
        key=lambda m: (m["precision"], m["threshold"]))
    return {"dunhuang_max_f1": max_f1["threshold"],
        "dunhuang_recall95_max_precision": high_recall["threshold"]}


def auc(rows):
    positive = [r["score"] for r in rows if r["label"]]
    negative = [r["score"] for r in rows if not r["label"]]
    return sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def reference_receipts():
    items = {}
    for path in (HERE / "S6_ALL_DEPTHS_COMPLETED_COMPARISON_20260916.json", HERE / "S7_COMPLETED_COMPARISON_20260916.json"):
        for e in json.loads(path.read_text())["evaluations"]:
            if e["selection"] == "fixed_epoch" and e["split"] in ("real", "ood"):
                items[e["model"], e["split"]] = e
    return items


def validate_inputs(data):
    reference_ids = {}
    for model in data["models"]:
        for split, count, positive_count in (("real", 803, 295), ("ood", 301, 301)):
            rows = model[split + "_rows"]
            assert len(rows) == count and sum(bool(r["label"]) for r in rows) == positive_count
            ids = {r["pair_id"]: bool(r["label"]) for r in rows}
            assert len(ids) == count, "duplicate pair IDs"
            assert ids == reference_ids.setdefault(split, ids), "models differ in evaluated pairs"
            assert all(r["decision_valid"] and math.isfinite(r["score"]) for r in rows)
        negatives = [r for r in model["real_rows"] if not r["label"]]
        assert sum(bool(r["strict_member"]) for r in negatives) == 39


def run(data, output):
    validate_inputs(data)
    refs = reference_receipts()
    freeze = {"schema": "dunhuang-threshold-freeze/1", "fit_population": "295 reviewed-kept positives + 508 negatives (39 strict, 469 constructed)",
        "checkpoint": "fixed_epoch20_all_models", "fit_rule": "score >= threshold; maxF1 or maxPrecision subject to Recall>=95%; higher threshold breaks ties",
        "ood_used_for_fit": False, "models": []}
    for model in data["models"]:
        real = model["real_rows"]
        freeze["models"].append(dict(name=model["name"], thresholds=fit_thresholds(real)))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError("completed result already exists; use a new output directory")
    if (output / "threshold_freeze.json").exists():
        assert json.loads((output / "threshold_freeze.json").read_text()) == freeze
    else:
        (output / "threshold_freeze.json").write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n")
    frozen = json.loads((output / "threshold_freeze.json").read_text())
    result = {"schema": "dunhuang-threshold-transfer/1", "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": ["Post-hoc analysis requested by user, not a replacement for the original SIMVAL benchmark.",
            "Dunhuang is used for calibration; its recalibrated metrics are in-sample, not independent test performance.",
            "Turufan thresholds are applied unchanged, but this dataset has been observed previously in this project.",
            "Turufan contains only 301 positives: only positive recall/miss counts are reported. No layout GT.",
            "All four models fixed at epoch20. No weight, layout, data-label, checkpoint or original protocol modification.",
            "All 803 REAL and 301 OOD decisions valid and finite. Pair IDs and labels aligned across models."], "models": []}
    for model, fm in zip(data["models"], frozen["models"]):
        name = model["name"]
        real, ood = model["real_rows"], model["ood_rows"]
        original = refs[name, "real"]["classification"]["max_f1"]
        original_t = original["threshold"]
        reproduction = metrics(real, original_t)
        assert all(reproduction[k] == original[k] for k in ("tp", "fp", "fn", "tn"))
        ood_original = metrics(ood, original_t)
        assert ood_original["tp"] == refs[name, "ood"]["classification"]["max_f1"]["accepted_positive_count"]
        assert reproduction["raw_layout20_correct"] == refs[name, "real"]["layout20"]["max_f1"]["raw_correct"]
        area = auc(real)
        assert abs(area - original["auroc"]) < 1e-10
        out = dict(name=name, real_auroc=area, sources=model["sources"], baseline_reproduced=True, operating_points={})
        thresholds = {"simval_max_f1": original_t, **fm["thresholds"]}
        for policy, t in thresholds.items():
            real_metric = metrics(real, t)
            ood_metric = metrics(ood, t)
            strata = {}
            for label, strict in (("strict_negative", True), ("constructed_negative", False)):
                group = [r for r in real if not r["label"] and bool(r["strict_member"]) == strict]
                strata[label] = {"n": len(group), "fp": sum(r["score"] >= t for r in group)}
            rescued = sum(r["score"] >= t and r["score"] < original_t for r in ood)
            lost = sum(r["score"] < t and r["score"] >= original_t for r in ood)
            out["operating_points"][policy] = dict(real_calibration=real_metric, real_negative_strata=strata,
                ood_positive_only={"n": len(ood), "tp": ood_metric["tp"], "fn": ood_metric["fn"],
                    "recall": ood_metric["recall"], "rescued_vs_original": rescued, "lost_vs_original": lost})
        result["models"].append(out)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Dunhuang calibration → Turufan transfer", "", "Fixed epoch20; Dunhuang 295 positives + 508 negatives; Turufan 301 positives only.",
        "Dunhuang values below are in-sample calibration results. Threshold-only change; no retraining.", "",
        "|Model|Threshold rule|Threshold|DH Accuracy|DH Precision|DH Recall|DH F1|DH FP / 508|Turufan Recall|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in result["models"]:
        for policy, op in m["operating_points"].items():
            r, o = op["real_calibration"], op["ood_positive_only"]
            lines.append(f'|{m["name"]}|{policy}|{r["threshold"]:.8g}|{r["accuracy"]:.2%}|{r["precision"]:.2%}|{r["recall"]:.2%}|{r["f1"]:.2%}|{r["fp"]}|{o["tp"]}/301 ({o["recall"]:.2%})|')
    lines += ["", *result["notes"], ""]
    (output / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=HERE / "REAL_CALIBRATION_SCORE_INPUTS_20260916.json")
    p.add_argument("--output", type=Path, default=HERE / "dunhuang_threshold_transfer_20260916")
    args = p.parse_args()
    if not args.inputs.exists():
        args.inputs.write_text(json.dumps(collect(), indent=2, ensure_ascii=False) + "\n")
    run(json.loads(args.inputs.read_text()), args.output)
