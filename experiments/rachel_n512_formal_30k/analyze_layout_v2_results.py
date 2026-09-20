"""Summarize same-run layout predictions and paired geometry uncertainty.

Requires completed prediction directories; it never waits for jobs, opens model
weights, refits a threshold, or substitutes historical real predictions.  The
Exact-6 summary contributes explicitly historical point estimates only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


TOLERANCES = (2, 5, 8, 10)
ORIGINAL = "full_original"
SHRED = "full_with_shred_matching_layout"
REPLICATES = 20_000
DISPLAY = {
    ORIGINAL: "Full original",
    "full_sparse_cauchy": "Full + sparse Cauchy",
    "full_reciprocal_mode": "Full + reciprocal mode",
    "full_top2_mode": "Full + top-2 mode",
    "full_affinity_mode": "Full + affinity mode",
    "full_reciprocal_mode_r5": "Full + reciprocal mode (5 px)",
    SHRED: "Full + Shred correspondence",
}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_population(directory, kind, selected_decoder):
    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    if summary.get("status") != "complete":
        raise ValueError("Prediction summary is not complete: " + str(directory))
    with (directory / "pair_results.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    expected_count, expected_positive = (1016, 508) if kind == "real" else (3000, 1500)
    labels = np.asarray([row["label"] for row in rows], dtype=bool)
    if (len(rows), int(labels.sum())) != (expected_count, expected_positive):
        raise ValueError("{} needs {} rows / {} GT positives; got {} / {}".format(
            kind, expected_count, expected_positive, len(rows), int(labels.sum())))
    if summary.get("sample_count") != len(rows) or summary.get("positive_count") != int(labels.sum()):
        raise ValueError(kind + " summary/GT population counts differ")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError(kind + " duplicate pair IDs")
    if kind == "test" and not all(
        isinstance(row.get("source_unit_ids"), list)
        and 1 <= len(row["source_unit_ids"]) <= 2
        and all(isinstance(unit, str) and unit for unit in row["source_unit_ids"])
        and len(set(row["source_unit_ids"])) == len(row["source_unit_ids"])
        for row in rows
    ):
        raise ValueError("test requires manuscript source_unit_ids, never fragment-token substitutes")
    if summary.get("selected_full_decoder") != selected_decoder:
        raise ValueError(kind + " differs from the validation-selected decoder")
    variants = tuple(rows[0]["layouts"])
    if ORIGINAL not in variants or selected_decoder not in variants:
        raise ValueError(kind + " lacks original or selected decoder predictions")
    errors = {name: np.full(len(rows), np.inf) for name in variants}
    pose_valid = {name: np.zeros(len(rows), bool) for name in variants}
    for index, row in enumerate(rows):
        if set(row["layouts"]) != set(variants):
            raise ValueError(kind + " has incomplete layout predictions")
        target = row.get("target_translation_rc")
        if labels[index]:
            target = np.asarray(target, dtype=float)
            if target.shape != (2,) or not np.isfinite(target).all():
                raise ValueError(kind + " positive pair lacks 2-D translation GT")
        elif target is not None:
            raise ValueError(kind + " negative pair unexpectedly contains translation GT")
        for name, layout in row["layouts"].items():
            translation = np.asarray(layout["translation_rc"], dtype=float)
            valid = bool(layout["valid"]) and translation.shape == (2,) and np.isfinite(translation).all()
            pose_valid[name][index] = valid
            if labels[index] and valid:
                errors[name][index] = np.linalg.norm(translation - target)
    strict = np.ones(len(rows), dtype=bool)
    if kind == "real":
        # Explicit membership avoids relying on output row ordering.
        if not all("strict_member" in row for row in rows):
            raise ValueError("real rows need explicit strict_member membership")
        strict = np.asarray([row["strict_member"] for row in rows], dtype=bool)
        if int(strict.sum()) != 547 or int(labels[strict].sum()) != 508:
            raise ValueError("real strict population must contain 547 pairs / 508 positives")
        if not all(row.get("case_id") or row.get("cluster_id") for row in rows if row["strict_member"]):
            raise ValueError("real strict pairs lack case_id for whole-case bootstrap")
    return dict(directory=directory, summary=summary, rows=rows, labels=labels,
                variants=variants, errors=errors, pose_valid=pose_valid, strict=strict)


def classification(labels, scores, threshold):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("classification scores must be finite")
    labels = np.asarray(labels, dtype=bool)
    predicted = scores >= threshold
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    tn = int((~predicted & ~labels).sum())
    result = dict(threshold=float(threshold), accuracy=(tp + tn) / len(labels),
                  precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                  f1=2 * tp / max(2 * tp + fp + fn, 1), tp=tp, fp=fp, fn=fn, tn=tn)
    # Tied-score AUROC and average precision, without a sklearn dependency.
    order = np.argsort(-scores, kind="stable")
    sorted_score, sorted_label = scores[order], labels[order]
    last = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(scores) - 1]
    tps = np.cumsum(sorted_label)[last].astype(float)
    fps = (last + 1) - tps
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives and negatives:
        recall = np.r_[0.0, tps / positives]
        fpr = np.r_[0.0, fps / negatives]
        result["auroc"] = float(np.trapz(recall, fpr))
        result["auprc"] = float(np.sum(np.diff(recall) * tps / (last + 1)))
    return result


def layout_points(population, original_threshold, strict=False):
    include = population["strict"] if strict else np.ones(len(population["rows"]), bool)
    labels = population["labels"][include]
    predicted = np.asarray([row["classification"]["fused"] for row in population["rows"]])[include] >= original_threshold
    target_count, predicted_count = int(labels.sum()), int(predicted.sum())
    result = {}
    for name in population["variants"]:
        errors = population["errors"][name][include]
        valid = population["pose_valid"][name][include]
        finite_positive = errors[labels & np.isfinite(errors)]
        metrics = dict(positive_count=target_count, predicted_edge_count=predicted_count,
                       positive_pose_coverage=float((labels & valid).sum() / target_count),
                       median_px_conditional=float(np.median(finite_positive)) if len(finite_positive) else None,
                       p90_px_conditional=float(np.quantile(finite_positive, .9)) if len(finite_positive) else None,
                       recall={}, assembly={})
        for tolerance in TOLERANCES:
            correct = labels & valid & (errors <= tolerance)
            correct_count = int(correct.sum())
            tp = int((correct & predicted).sum())
            metrics["recall"][str(tolerance)] = correct_count / target_count
            metrics["assembly"][str(tolerance)] = dict(
                precision=tp / max(predicted_count, 1), recall=tp / target_count,
                f1=2 * tp / (predicted_count + target_count), tp=tp,
                fp=predicted_count - tp, fn=target_count - tp)
        result[name] = metrics
    return result


def interval(samples, point, delta=False):
    finite = np.asarray(samples, float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("All paired-bootstrap draws lack positive targets")
    result = dict(point_estimate=float(point), percentile_95_ci=np.quantile(finite, [.025, .975]).tolist(),
                  valid_replicates=len(finite))
    if delta:
        result["probability_delta_gt_zero"] = float(np.mean(finite > 0))
    return result


def paired_outcome_counts(population, selected, threshold):
    """Expose which positives were rescued or lost, using existing predictions."""

    labels = population["labels"]
    predicted = np.asarray([row["classification"]["fused"] >= threshold for row in population["rows"]])
    result = {}
    for comparator in (ORIGINAL, SHRED):
        if comparator not in population["variants"] or comparator == selected:
            continue
        newer, older = population["errors"][selected], population["errors"][comparator]
        metrics = {}
        for tolerance in TOLERANCES:
            new_ok, old_ok = newer <= tolerance, older <= tolerance
            metrics[str(tolerance)] = dict(
                both_correct=int((labels & new_ok & old_ok).sum()),
                selected_only_correct=int((labels & new_ok & ~old_ok).sum()),
                comparator_only_correct=int((labels & ~new_ok & old_ok).sum()),
                both_fail=int((labels & ~new_ok & ~old_ok).sum()),
                accepted_positive_rescued=int((labels & predicted & new_ok & ~old_ok).sum()),
                accepted_positive_lost=int((labels & predicted & ~new_ok & old_ok).sum()),
            )
        both_valid = labels & np.isfinite(newer) & np.isfinite(older)
        difference = newer[both_valid] - older[both_valid]
        result[selected + "_versus_" + comparator] = dict(
            by_tolerance=metrics, jointly_valid_positive_count=int(both_valid.sum()),
            selected_minus_comparator_error_quantiles_px=(
                dict(zip(("p10", "median", "p90"), np.quantile(difference, [.1, .5, .9]).tolist()))
                if len(difference) else None
            ),
        )
    return result


def selected_layout_diagnostics(population, selected):
    """Describe geometric ambiguity without fitting a post-hoc acceptance rule."""

    labels, errors = population["labels"], population["errors"][selected]
    result = {"role": "postprediction_description_only_no_threshold_fit", "groups": {}}
    for name, mask in (("correct_at_10px", labels & (errors <= 10)),
                       ("failed_at_10px", labels & (errors > 10))):
        rows = [population["rows"][index] for index in np.flatnonzero(mask)]
        group = {"positive_count": len(rows), "diagnostics": {}}
        for key in ("inlier_count", "weighted_inlier_fraction", "runner_up_support_ratio", "residual_px"):
            values = [row["layouts"][selected].get("diagnostics", {}).get(key) for row in rows]
            values = [float(value) for value in values if value is not None and np.isfinite(value)]
            if values:
                group["diagnostics"][key] = dict(zip(("p10", "median", "p90"), np.quantile(values, [.1, .5, .9]).tolist()))
        result["groups"][name] = group
    result["positive_error_tail_counts_including_invalid"] = {
        method: {"above_{}_px".format(tolerance): int((labels & (population["errors"][method] > tolerance)).sum())
                 for tolerance in (10, 50, 100, 300)}
        for method in (ORIGINAL, selected, SHRED) if method in population["variants"]
    }
    return result


def paired_bootstrap(population, selected, threshold, kind, replicates=REPLICATES, seed=260908):
    include = population["strict"] if kind == "real" else np.ones(len(population["rows"]), bool)
    rows = [row for index, row in enumerate(population["rows"]) if include[index]]
    labels = population["labels"][include]
    predicted = np.asarray([row["classification"]["fused"] >= threshold for row in rows], bool)
    methods = [ORIGINAL, selected] + ([SHRED] if SHRED in population["variants"] else [])
    methods = list(dict.fromkeys(methods))
    keys = ["r_at_{}".format(tolerance) for tolerance in TOLERANCES] + ["assembly_f1_at_10"]
    evidence = []
    for method in methods:
        error = population["errors"][method][include]
        valid = population["pose_valid"][method][include]
        for tolerance in TOLERANCES:
            evidence.append(labels & valid & (error <= tolerance))
        evidence.append(labels & predicted & valid & (error <= 10))
    evidence = np.asarray(evidence, dtype=float).T
    if kind == "real":
        units = sorted({str(row.get("case_id") or row["cluster_id"]) for row in rows})
        index = {unit: i for i, unit in enumerate(units)}
        first = np.asarray([index[str(row.get("case_id") or row["cluster_id"])] for row in rows])
        second = first
        sampling = "whole_real_case_multinomial; all strict547 pairs in a case move together"
    else:
        units = sorted({unit for row in rows for unit in row["source_unit_ids"]})
        index = {unit: i for i, unit in enumerate(units)}
        first = np.asarray([index[row["source_unit_ids"][0]] for row in rows])
        second = np.asarray([index[row["source_unit_ids"][-1]] for row in rows])
        sampling = "manuscript_lineage_endpoint_pigeonhole; same-unit count once, distinct-unit counts multiplied"
    rng = np.random.default_rng(seed)
    draws = np.full((replicates, len(methods), len(keys)), np.nan)
    for start in range(0, replicates, 128):
        count = min(128, replicates - start)
        multiplicity = rng.multinomial(len(units), np.full(len(units), 1.0 / len(units)), size=count)
        weights = multiplicity[:, first]
        if kind != "real":
            weights = weights * np.where(first == second, 1, multiplicity[:, second])
        weights = weights.astype(float)
        positives = weights @ labels.astype(float)
        predicted_count = weights @ predicted.astype(float)
        numerators = (weights @ evidence).reshape(count, len(methods), len(keys))
        with np.errstate(divide="ignore", invalid="ignore"):
            values = numerators / positives[:, None, None]
            values[:, :, -1] = 2 * numerators[:, :, -1] / (positives + predicted_count)[:, None]
        values[positives <= 0] = np.nan
        draws[start:start + count] = values
    points = layout_points(population, threshold, strict=kind == "real")
    point = np.asarray([[points[method]["recall"][str(tolerance)] for tolerance in TOLERANCES]
                        + [points[method]["assembly"]["10"]["f1"]] for method in methods])
    comparisons = [(method, ORIGINAL) for method in methods if method != ORIGINAL]
    if SHRED in methods and selected != SHRED:
        comparisons.append((selected, SHRED))
    return dict(
        population="strict547" if kind == "real" else "test3000",
        sample_count=len(rows), positive_count=int(labels.sum()), sampling_unit=sampling,
        sampling_unit_count=len(units), seed=seed, requested_replicates=replicates,
        skipped_no_positive_replicates=int(np.isnan(draws[:, 0, 0]).sum()),
        methods={method: {key: interval(draws[:, mi, ki], point[mi, ki])
                          for ki, key in enumerate(keys)} for mi, method in enumerate(methods)},
        paired_deltas={a + "_minus_" + b: {
            key: interval(draws[:, methods.index(a), ki] - draws[:, methods.index(b), ki],
                          point[methods.index(a), ki] - point[methods.index(b), ki], delta=True)
            for ki, key in enumerate(keys)} for a, b in comparisons},
    )


def historical_points(document):
    result = {"role": "historical_point_estimates_only_not_same_run_comparators", "test": {}, "real": {}}
    for method in ("pairingnet_adapted", "shreddingnet_adapted"):
        synthetic = document["pose"]["synthetic"][method]
        real = document["pose"]["real_strict547_positive_translation_gt"]["methods"][method]
        result["test"][method] = dict(
            classification=document["classification"]["synthetic_native_all_valid"][method],
            recall={str(t): synthetic["unconditional_positive_translation_recall"]["at_{}px".format(t)] for t in TOLERANCES},
            assembly_f1_at_10=synthetic["assembly_edge"]["at_10px"]["f1"],
            median_px_conditional=synthetic["median_l2_px_conditioned_on_valid_pose"],
        )
        result["real"][method] = dict(
            classification=document["classification"]["real_balanced1016_inferential"]["methods"][method],
            recall={str(t): real["case_bootstrap"]["recall_at_{}px".format(t)]["point_estimate"] for t in TOLERANCES},
            assembly_f1_at_10=real["assembly_edge_strict547"]["by_tolerance"]["at_10px"]["f1"],
            median_px_conditional=real["case_bootstrap"]["median_l2_px"]["point_estimate"],
        )
    return result


def make_plots(populations, selected, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"test": "Synthetic test", "real": "Real, authoritative positives"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    columns = len(populations)
    fig, axes = plt.subplots(1, columns, figsize=(6 * columns, 4), squeeze=False)
    fig_bars, bar_axes = plt.subplots(1, columns, figsize=(6 * columns, 4), squeeze=False)
    colors = ["#787878", "#167D9A", "#CA5B35"]
    for position, (kind, population) in enumerate(populations.items()):
        methods = [ORIGINAL, selected] + ([SHRED] if SHRED in population["variants"] else [])
        methods = list(dict.fromkeys(methods))
        ax, bars = axes[0, position], bar_axes[0, position]
        for mi, method in enumerate(methods):
            errors = np.sort(population["errors"][method][population["labels"]])
            visible = errors[np.isfinite(errors) & (errors <= 50)]
            xx = np.r_[0.0, visible, 50.0]
            yy = np.r_[0.0, np.arange(1, len(visible) + 1) / len(errors), len(visible) / len(errors)]
            # Errors above 50 px and invalid poses remain in the denominator.
            ax.step(xx, yy, where="post", color=colors[mi], label=DISPLAY.get(method, method), linewidth=2)
            recall = [float(np.mean(errors <= tolerance)) for tolerance in TOLERANCES]
            width = .8 / len(methods)
            bars.bar(np.arange(4) - .4 + width * (mi + .5), recall, width=width,
                     color=colors[mi], label=DISPLAY.get(method, method))
        ax.set(xlim=(0, 50), ylim=(0, 1.02), xlabel="Translation error (px)",
               ylabel="Fraction of all GT-positive pairs", title=titles[kind])
        ax.axvline(10, color="#555555", linestyle=":", linewidth=1)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8, loc="lower right")
        bars.set(ylim=(0, 1.02), xticks=np.arange(4), xticklabels=["R@{}".format(t) for t in TOLERANCES],
                 ylabel="Recall over all GT-positive pairs", title=titles[kind])
        bars.grid(axis="y", alpha=.2)
        bars.set_axisbelow(True)
    handles, labels = bar_axes[0, 0].get_legend_handles_labels()
    fig_bars.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=9,
                    frameon=False, bbox_to_anchor=(.5, .01))
    files = []
    for figure, stem in ((fig, "translation_error_cdf"), (fig_bars, "translation_recall")):
        figure.tight_layout(rect=(0, .1, 1, 1) if figure is fig_bars else None)
        for suffix in ("png", "pdf"):
            path = output / (stem + "." + suffix)
            figure.savefig(path, dpi=200, bbox_inches="tight")
            files.append(path.name)
        plt.close(figure)
    return files


def format_number(value):
    return "—" if value is None else "{:.4f}".format(value)


def markdown_report(result):
    selected = result["selected_full_decoder"]
    lines = ["本轮保留 Full 的整体 coarse、Sliding Window、Sinkhorn 配对分支；位姿使用独立的无旋转解码。",
             "", "验证集选择：`{}`。分类阈值仅来自验证集；所有本轮质量指标均从本轮逐对预测重新计算。".format(selected), ""]
    for kind, section in result["populations"].items():
        lines += ["**{}：{} 对，{} 个正样本。**".format(
            {"validation": "验证集", "test": "合成测试集", "real": "真实集"}[kind], section["sample_count"], section["positive_count"]), "",
            "| 配对分支 | AUROC | AUPRC | Accuracy | F1 | Recall |", "|---|---:|---:|---:|---:|---:|"]
        for branch, metric in section["classification"].items():
            lines.append("| {} | {} | {} | {} | {} | {} |".format(branch, *[format_number(metric.get(k)) for k in ("auroc", "auprc", "accuracy", "f1", "recall")]))
        lines += ["", "coarse/local 使用各自在验证集上选定的阈值；fused 使用原 Full 冻结阈值。", "",
                  "| Layout | R@2 | R@5 | R@8 | R@10 | Assembly F1@10 | 有效位姿中位误差 px |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, metric in section["layout"].items():
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                DISPLAY.get(name, name) + ("（验证集所选）" if name == selected else ""),
                *[format_number(metric["recall"][str(t)]) for t in TOLERANCES],
                format_number(metric["assembly"]["10"]["f1"]), format_number(metric["median_px_conditional"])))
        lines += [""]
        if kind == "real":
            lines += ["真实集分类表使用 balanced1016；layout 的 Assembly F1 及置信区间使用 strict547（508 个真实正样本、39 个真实负样本）。", ""]
        if "bootstrap" in section:
            bootstrap = section["bootstrap"]
            lines += ["20,000 次配对 bootstrap；95% 区间如下（新方法减比较方法）：", "",
                      "| 比较 | ΔR@10 [95% CI] | ΔAssembly F1@10 [95% CI] |", "|---|---:|---:|"]
            for name, comparison in bootstrap["paired_deltas"].items():
                values = []
                for key in ("r_at_10", "assembly_f1_at_10"):
                    value = comparison[key]
                    values.append("{:+.4f} [{:+.4f}, {:+.4f}]".format(value["point_estimate"], *value["percentile_95_ci"]))
                lines.append("| {} | {} | {} |".format(name, *values))
            lines += [""]
    lines += ["历史 Exact-6 基准仅作点估计参考；没有用旧 real 位姿替代本轮 real，也没有把历史结果纳入同次配对 bootstrap。", "",
              "| 历史数据 | 方法 | R@10 | Assembly F1@10 |", "|---|---|---:|---:|"]
    for kind in ("test", "real"):
        for method, point in result["historical_exact6"][kind].items():
            lines.append("| {} | {} | {} | {} |".format(kind, method, format_number(point["recall"]["10"]), format_number(point["assembly_f1_at_10"])))
    lines += ["", "R@k 的分母为所有 GT 正样本；无效位姿记失败。Assembly 的错误位姿同时形成错误预测边与漏掉的正确边。",
              "合成测试按 manuscript lineage（source_unit_ids）重采样：同一 lineage 只乘一次计数，跨 lineage 对使用两端 multinomial 计数乘积；真实集按 case 整组重采样。既有测试集曾在此前实验中查看过，本轮比较属于后续实验，不是全新未见测试。", "",
              "![Translation error CDF](translation_error_cdf.png)", "", "![Translation recall](translation_recall.png)", ""]
    return "\n".join(lines)


def analyze(validation_dir, test_dir, exact6_summary, output_dir, real_dir=None):
    output = Path(output_dir)
    freeze = read_json(Path(validation_dir) / "validation_freeze.json")
    if freeze.get("source_split") != "validation" or freeze.get("probe_only"):
        raise ValueError("Requires full validation selection, not a probe")
    selected = freeze["selected_full_decoder"]
    threshold = float(freeze["original_fused_threshold"])
    directories = {"validation": validation_dir, "test": test_dir}
    if real_dir is not None:
        directories["real"] = real_dir
    populations = {kind: read_population(directory, kind, selected) for kind, directory in directories.items()}
    historical = historical_points(read_json(exact6_summary))
    result: Dict = dict(schema_version="rachel-layout-v2-analysis/1.0", status="complete",
                        selected_full_decoder=selected, original_full_classification_threshold=threshold,
                        same_run_quality_only=True, rotation_estimated=False, routing_used=False,
                        historical_exact6=historical, populations={},
                        sources={**{kind: str(Path(value).resolve()) for kind, value in directories.items()},
                                 "exact6_summary": str(Path(exact6_summary).resolve())})
    for kind, population in populations.items():
        section = dict(sample_count=len(population["rows"]), positive_count=int(population["labels"].sum()),
                       classification={}, classification_at_validation_row_f1_threshold={},
                       layout=layout_points(population, threshold, strict=kind == "real"),
                       paired_outcome_counts=paired_outcome_counts(population, selected, threshold),
                       selected_layout_diagnostics=selected_layout_diagnostics(population, selected))
        for branch in ("coarse", "local", "fused"):
            scores = [row["classification"][branch] for row in population["rows"]]
            branch_threshold = threshold if branch == "fused" else float(freeze["branch_validation_thresholds"][branch])
            section["classification"][branch] = classification(population["labels"], scores, branch_threshold)
            section["classification_at_validation_row_f1_threshold"][branch] = classification(
                population["labels"], scores, float(freeze["branch_validation_thresholds"][branch]))
        if kind != "validation":
            section["bootstrap"] = paired_bootstrap(population, selected, threshold, kind,
                                                      seed=260908 if kind == "test" else 260909)
        if kind == "real":
            section["strict_sample_count"] = 547
            section["layout_balanced1016_diagnostic"] = layout_points(population, threshold)
        result["populations"][kind] = section
    output.mkdir(parents=True, exist_ok=True)
    result["figures"] = make_plots({kind: population for kind, population in populations.items() if kind != "validation"}, selected, output)
    with (output / "comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    (output / "comparison.md").write_text(markdown_report(result), encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--exact6-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.validation_dir, args.test_dir, args.exact6_summary, args.output_dir, args.real_dir)
    print(json.dumps({"status": result["status"], "selected_full_decoder": result["selected_full_decoder"],
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
