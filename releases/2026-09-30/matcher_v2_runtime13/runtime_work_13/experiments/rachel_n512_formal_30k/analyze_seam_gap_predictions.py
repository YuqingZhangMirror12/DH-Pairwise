"""CPU-only join of GT-seam damage diagnostics and frozen stress predictions.

No model, image, training data, GPU, or threshold fitting is used. Positive-only
damage groups report recall/counts, not misleading subgroup precision. Overall
P/F1 remain the original evaluator's values. Completed probes stay probes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "rachel-seam-gap-prediction-analysis/1"
MANIFEST_SCHEMA = "rachel-seam-gap-manifest/1"
SEED, TEST_COUNT, POSITIVE_COUNT = 260910, 3000, 1500
DECODER = "full_top2_mode"
RADII = ("2", "5", "10")
GROUPS = ("all_positive", "GT_seam_damaged_positive", "measured_below_criterion_positive", "unmeasured_positive")
HEADS = ("score_geometry", "score_only")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scope(value, name):
    if value.get("status") != "complete":
        raise ValueError(name + " is not complete")
    count = value.get("evaluated_pair_count")
    if type(count) is not int or not 1 <= count <= TEST_COUNT or value.get("total_test_pair_count") != TEST_COUNT:
        raise ValueError(name + " has an invalid declared TEST population")
    if type(value.get("full_test")) is not bool or type(value.get("probe_only")) is not bool:
        raise ValueError(name + " must explicitly mark full_test and probe_only")
    if value["full_test"] == value["probe_only"] or (value["full_test"] and count != TEST_COUNT):
        raise ValueError(name + " full/probe scope is inconsistent")
    return count


def _identity(row):
    if type(row.get("label")) is not bool:
        raise ValueError("pair label must be an explicit boolean")
    values = tuple(row.get(key) for key in ("pair_id", "fragment_a", "fragment_b"))
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("pair ID and ordered endpoint identities are required")
    return values + (row["label"],)


def _probability(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + " must be a finite probability in [0,1]")
    return float(value)


def bind_inputs(diagnostics_path, evaluation_root):
    diagnostics_path, root = Path(diagnostics_path).resolve(), Path(evaluation_root).resolve()
    sources = dict(diagnostics=str(diagnostics_path), receipt=str(root / "receipt.json"),
                   original_summary=str(root / "summary.json"), pair_results=str(root / "pair_results.jsonl"))
    manifest, receipt, summary = (_read(sources[key]) for key in ("diagnostics", "receipt", "original_summary"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported GT-seam diagnostics schema")
    manifest_count, evaluated_count = _scope(manifest, "diagnostics"), _scope(receipt, "evaluation receipt")
    if summary.get("status") != "complete":
        raise ValueError("evaluation summary is not complete")
    if manifest.get("seed") != SEED or receipt.get("seed") != SEED:
        raise ValueError("diagnostics and evaluation must use fixed seed260910")
    depth = manifest.get("max_depth_px")
    if depth not in (0, 2, 4) or receipt.get("depths") != [depth]:
        raise ValueError("diagnostics and evaluation erosion depths differ")
    source_sha = manifest.get("test_manifest_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64 or receipt.get("test_manifest_sha256") != source_sha:
        raise ValueError("diagnostics and evaluation TEST manifest identities differ")
    for key in ("seed", "depths", "test_manifest_sha256", "full_test", "probe_only",
                "evaluated_pair_count", "total_test_pair_count", "checkpoint_sha256", "model_label"):
        if key not in receipt or summary.get(key) != receipt[key]:
            raise ValueError("evaluation summary/receipt source binding differs: " + key)
    if receipt.get("selected_full_decoder") != DECODER or receipt.get("test_or_real_used_for_fit") is not False:
        raise ValueError("evaluation must retain fixed Top2 and frozen VAL-only decisions")
    diagnostic_rows = manifest.get("rows")
    if not isinstance(diagnostic_rows, list) or len(diagnostic_rows) != manifest_count:
        raise ValueError("diagnostics rows do not match its completion count")
    diagnostic_ids = [_identity(row) for row in diagnostic_rows]
    if len({row[0] for row in diagnostic_ids}) != len(diagnostic_ids):
        raise ValueError("duplicate diagnostic pair IDs")
    if manifest["full_test"] and sum(row[3] for row in diagnostic_ids) != POSITIVE_COUNT:
        raise ValueError("full diagnostic manifest must retain 1500 TEST positives")
    pair_path = Path(sources["pair_results"])
    with pair_path.open(encoding="utf-8") as stream:
        predictions = [json.loads(line) for line in stream if line.strip()]
    prediction_ids = [_identity(row) for row in predictions]
    if len(predictions) != evaluated_count or len({row[0] for row in prediction_ids}) != evaluated_count:
        raise ValueError("prediction rows are partial or contain duplicate pair IDs")
    if receipt.get("pair_results_sha256") != _sha(pair_path):
        raise ValueError("pair results do not match the completed evaluation receipt")
    if evaluated_count > manifest_count or prediction_ids != diagnostic_ids[:evaluated_count]:
        raise ValueError("prediction IDs/labels/ordered endpoints are not the same diagnostic prefix")
    positives = sum(row[3] for row in prediction_ids)
    if (summary.get("sample_count") != evaluated_count or summary.get("positive_count") != positives
            or summary.get("negative_count") != evaluated_count - positives):
        raise ValueError("original evaluation summary population differs from predictions")
    if receipt["full_test"] and positives != POSITIVE_COUNT:
        raise ValueError("full evaluation must retain 1500 TEST positives")
    matched = diagnostic_rows[:evaluated_count]
    for prediction, diagnostic in zip(predictions, matched):
        if type(diagnostic.get("actual_changed")) is not bool or prediction.get("actual_changed") is not diagnostic["actual_changed"]:
            raise ValueError("prediction and diagnostics actual changed-mask identities differ")
    thresholds = {"native_fused": _probability(receipt.get("original_fused_threshold"), "frozen fused threshold")}
    has_heads = any("head_scores" in row for row in predictions)
    head_thresholds = receipt.get("e3_thresholds")
    if has_heads or head_thresholds:
        if receipt.get("model_label") != "e2" or not isinstance(head_thresholds, dict) or set(head_thresholds) != set(HEADS):
            raise ValueError("E3 head scores require this E2 receipt's two frozen head thresholds")
        thresholds.update({head: _probability(head_thresholds[head], "frozen " + head + " threshold") for head in HEADS})
    for method, threshold in thresholds.items():
        saved_method = summary.get("methods", {}).get(method)
        if not isinstance(saved_method, dict) or saved_method.get("threshold") != threshold:
            raise ValueError("original summary method threshold differs from receipt: " + method)
        if not isinstance(saved_method.get("classification"), dict):
            raise ValueError("original overall classification is missing: " + method)
        for row in predictions:
            score = (row.get("classification", {}).get("fused") if method == "native_fused"
                     else row.get("head_scores", {}).get(method))
            _probability(score, method + " prediction score")
    return manifest, receipt, summary, predictions, matched, thresholds, sources


def positive_groups(diagnostics):
    groups = {name: [] for name in GROUPS}
    unknown_reasons = Counter()
    for index, row in enumerate(diagnostics):
        if not row["label"]:
            continue
        groups["all_positive"].append(index)
        detail = row.get("diagnostics")
        if not isinstance(detail, dict):
            raise ValueError("positive row lacks GT-seam diagnostic measurement")
        if detail.get("valid") is True:
            decision = detail.get("actual_seam_damaged")
            if type(decision) is not bool:
                raise ValueError("measured positive requires explicit true/false seam damage")
            groups["GT_seam_damaged_positive" if decision else "measured_below_criterion_positive"].append(index)
        else:
            groups["unmeasured_positive"].append(index)
            unknown_reasons[str(detail.get("invalid_reason", "unspecified"))] += 1
    return groups, dict(unknown_reasons)


def _pose_correct(row, tolerance):
    layouts = row.get("layouts")
    pose = layouts.get(DECODER) if isinstance(layouts, dict) else None
    if not isinstance(pose, dict):
        return False
    error = pose.get("translation_l2_px")
    return (pose.get("valid") is True and type(error) in (int, float)
            and math.isfinite(error) and 0 <= error <= tolerance)


def group_metrics(indices, predictions, accepted):
    count = len(indices)
    raw = {radius: sum(_pose_correct(predictions[index], int(radius)) for index in indices) for radius in RADII}
    joint = {radius: sum(accepted[index] and _pose_correct(predictions[index], int(radius)) for index in indices) for radius in RADII}
    return dict(metrics_status="measured" if count else "empty", positive_count=count,
                accepted_tp=sum(accepted[index] for index in indices) if count else None,
                pair_recall=sum(accepted[index] for index in indices) / count if count else None,
                raw_layout={radius: dict(correct_count=raw[radius] if count else None,
                                       recall=raw[radius] / count if count else None) for radius in RADII},
                joint={radius: dict(tp=joint[radius] if count else None,
                                   recall=joint[radius] / count if count else None) for radius in RADII},
                precision_not_reported=True)


def analyze(diagnostics_path, evaluation_root):
    manifest, receipt, original, predictions, diagnostics, thresholds, sources = bind_inputs(diagnostics_path, evaluation_root)
    groups, unknown_reasons = positive_groups(diagnostics)
    negative = [index for index, row in enumerate(predictions) if not row["label"]]
    methods = {}
    for method, threshold in thresholds.items():
        scores = [row["classification"]["fused"] if method == "native_fused" else row["head_scores"][method] for row in predictions]
        accepted = [score >= threshold for score in scores]
        fp = sum(accepted[index] for index in negative)
        methods[method] = dict(threshold=threshold, threshold_source="receipt.original_fused_threshold" if method == "native_fused" else "receipt.e3_thresholds." + method,
            groups={group: group_metrics(indices, predictions, accepted) for group, indices in groups.items()},
            shared_negative_cohort=dict(negative_count=len(negative), fp=fp if negative else None,
                                        fpr=fp / len(negative) if negative else None),
            original_overall_classification=original["methods"][method]["classification"],
            original_overall_metrics_source=sources["original_summary"] + " :: methods." + method + ".classification")
    full = receipt["full_test"] and manifest["full_test"]
    measured = len(groups["GT_seam_damaged_positive"]) + len(groups["measured_below_criterion_positive"])
    positive_count = len(groups["all_positive"])
    return dict(schema_version=SCHEMA, status="complete", full_test=full, probe_only=not full,
        population="synthetic_test3000" if full else "synthetic_test_prefix_probe",
        completed_at=datetime.now(timezone.utc).isoformat(), cpu_only=True,
        sample_count=len(predictions), positive_count=positive_count, negative_count=len(negative),
        evaluated_pair_count=len(predictions), total_test_pair_count=TEST_COUNT,
        diagnostics_manifest_pair_count=len(manifest["rows"]), diagnostics_manifest_full_test=manifest["full_test"],
        evaluation_full_test=receipt["full_test"], seed=SEED, max_depth_px=manifest["max_depth_px"],
        test_manifest_sha256=receipt["test_manifest_sha256"], model_label=receipt["model_label"],
        checkpoint_sha256=receipt["checkpoint_sha256"],
        group_counts={key: len(value) for key, value in groups.items()},
        measurement=dict(measured_positive_count=measured, unmeasured_positive_count=len(groups["unmeasured_positive"]),
            measured_positive_fraction=measured / positive_count if positive_count else None,
            unmeasured_reasons=unknown_reasons,
            any_edge_changed_positive_count=sum(row["actual_changed"] for row in diagnostics if row["label"])),
        methods=methods, sources=sources, source_sha256={name: _sha(path) for name, path in sources.items()},
        interpretation=["Positive groups depend only on precomputed GT-seam damage geometry, not model scores, acceptance, or pose success.",
            "Unmeasured seams remain unknown; measured_below_criterion does not prove physically undamaged.",
            "Positive-only groups have no meaningful standalone Precision/F1; original overall P/F1 are copied, not redefined.",
            "All methods use the same complete evaluated negative cohort for FP/FPR, without damage-conditioned negative selection.",
            "Raw layout recall uses every positive in the group; invalid/missing/nonfinite pose error is failure. Joint additionally requires frozen-threshold acceptance.",
            "GT-supported raster seam diagnostics on synthetic TEST are not measured real-world physical gap-width ground truth.",
            "No thresholds are fitted, no method is selected, no prediction is changed, and a completed prefix probe is not a full-test result."])


def _rate(value):
    return "—" if value is None else "%.2f%%" % (value * 100)


def _count(value):
    return "—" if value is None else str(value)


def markdown(result):
    lines = ["# GT 接缝损伤分组：已有冻结预测", "",
             ("范围：完整 TEST。" if result["full_test"] else "范围：prefix probe（即使执行 complete，也不是完整 TEST 结论）。") +
             " %d 对；正例 %d；负例 %d；请求最大向内腐蚀 %s px。" %
             (result["sample_count"], result["positive_count"], result["negative_count"], result["max_depth_px"]), "",
             "分组由预测之前的 GT 接缝损伤几何确定，不按模型成败筛选。未测量单列；以下全是正例组，因此不报告子集 Precision/F1。"]
    for method, record in result["methods"].items():
        lines += ["", "## " + method, "", "冻结阈值 %.8g。" % record["threshold"], "",
                  "| GT 正例组 | 正例数 | 接受 TP | Pair Recall | Raw R2 | Raw R5 | Raw R10 | Joint TP2 / R2 | Joint TP5 / R5 | Joint TP10 / R10 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
        for name, group in record["groups"].items():
            values = [name, str(group["positive_count"]), _count(group["accepted_tp"]), _rate(group["pair_recall"])]
            values += [_rate(group["raw_layout"][radius]["recall"]) for radius in RADII]
            values += [_count(group["joint"][radius]["tp"]) + " / " + _rate(group["joint"][radius]["recall"]) for radius in RADII]
            lines.append("| " + " | ".join(values) + " |")
        negative, overall = record["shared_negative_cohort"], record["original_overall_classification"]
        lines += ["", "同一全部负例 cohort：N=%s，FP=%s，FPR=%s。" %
                  (negative["negative_count"], _count(negative["fp"]), _rate(negative["fpr"])), "",
                  "原 summary 总体指标（原样引用）：Precision=%s，F1=%s。" % (_rate(overall.get("precision")), _rate(overall.get("f1")))]
    lines += ["", "## 边界与来源", ""] + ["- " + text for text in result["interpretation"]]
    lines += ["", "测量覆盖：%s；未测量正例 %s。" % (_rate(result["measurement"]["measured_positive_fraction"]), result["measurement"]["unmeasured_positive_count"]), ""]
    lines += ["- [" + name + "](<" + path + ">)" for name, path in result["sources"].items()]
    return "\n".join(lines) + "\n"


def run(args):
    result = analyze(args.diagnostics, args.evaluation)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    (output / "summary.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps(dict(status="complete", output=str(output), full_test=result["full_test"],
                         probe_only=result["probe_only"], sample_count=result["sample_count"], group_counts=result["group_counts"])))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics", required=True, help="complete prediction-independent diagnostics.json")
    p.add_argument("--evaluation", required=True, help="completed evaluate_gap_stress output directory")
    p.add_argument("--output", required=True, help="new directory; existing reports never overwritten")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
