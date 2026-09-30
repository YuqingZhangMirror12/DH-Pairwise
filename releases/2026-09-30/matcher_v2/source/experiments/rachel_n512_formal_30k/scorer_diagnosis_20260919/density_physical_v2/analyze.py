"""Stdlib-only, offline pair-level analysis of completed density_physical_v2.

No checkpoint loading, inference, remote access, GT metrics, or threshold fitting.
Example: python3 analyze.py --results results.json --output analysis_v1
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
EXPECTED_CHECKPOINT = "7c1212e2d9d62c25954457add3f9319b03dfe8fbc42aa96116e40caae0dc2c37"
EXPECTED_CASES = "7bab8e29a348e1ea62607bcf45376e6fe6e8dac2f2b7802d6b7415463d8544cf"
EPS = 1e-12


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest())


def finite(value, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("missing/nonfinite numeric evidence")
    return float(value)


def stats(values):
    values = list(values)
    present = [finite(v) for v in values if v is not None]
    return dict(pair_count=len(values), observed_pair_count=len(present), missing_pair_count=len(values)-len(present),
        mean=statistics.fmean(present) if present else None, median=statistics.median(present) if present else None,
        min=min(present) if present else None, max=max(present) if present else None)


def recovery(b, c):
    if b is None or c is None:
        return dict(B=b, C=c, reduction=None, recovery_fraction=None, direction="unavailable")
    b, c = finite(b), finite(c)
    if b < 0 or c < 0:
        raise ValueError("drifts must be nonnegative")
    difference = b-c
    return dict(B=b, C=c, reduction=difference,
        recovery_fraction=difference/b if b > EPS else None,
        direction="closer" if difference > EPS else "farther" if difference < -EPS else "equal")


def summarize_recovery(records):
    records = list(records)
    b, c = stats(r["B"] for r in records), stats(r["C"] for r in records)
    # The paired intersection is explicit if a zero-norm feature side is unavailable.
    paired = [r for r in records if r["B"] is not None and r["C"] is not None]
    denominator = math.fsum(r["B"] for r in paired)
    return dict(B=b, C=c, absolute_reduction=stats(r["reduction"] for r in records),
        per_pair_recovery_fraction=stats(r["recovery_fraction"] for r in records),
        ratio_of_paired_mean_drift_reduction=(denominator-math.fsum(r["C"] for r in paired))/denominator if denominator > EPS else None,
        ratio_paired_count=len(paired), directions=dict(Counter(r["direction"] for r in records)))


def layout_change(a, b):
    for v in (a, b):
        if type(v["valid"]) is not bool:
            raise ValueError("layout valid must be bool")
    displacement = None
    delta_rc = None
    if a["valid"] and b["valid"]:
        av, bv = a["translation_a_to_b_rc"], b["translation_a_to_b_rc"]
        if len(av) != 2 or len(bv) != 2:
            raise ValueError("invalid translation dimensions")
        delta_rc = [finite(y)-finite(x) for x, y in zip(av, bv)]
        displacement = math.hypot(*delta_rc)
    ar, br = finite(a["residual_px"], True), finite(b["residual_px"], True)
    return dict(valid_transition=f"{int(a['valid'])}->{int(b['valid'])}",
        translation_delta_rc=delta_rc, translation_change_px=displacement,
        inlier_delta=finite(b["inlier_count"])-finite(a["inlier_count"]),
        residual_delta_px=br-ar if ar is not None and br is not None else None)


def pair_record(row):
    branch = row["branches"]
    if set(branch) != {"R", "A", "B", "C"}:
        raise ValueError("all four branches are required")
    a = finite(branch["A"]["full"]["logit"])
    full = {k: {"logit": finite(v["full"]["logit"]), "probability": finite(v["full"]["probability"]),
                "valid_counts": v["full"]["valid_counts"]} for k, v in branch.items()}
    bridge = dict(signed_logit=full["A"]["logit"]-full["R"]["logit"],
        absolute_logit=abs(full["A"]["logit"]-full["R"]["logit"]),
        signed_probability=full["A"]["probability"]-full["R"]["probability"],
        original_has_padding=any(n < 512 for n in full["R"]["valid_counts"]),
        layout_change=layout_change(branch["R"]["layout"], branch["A"]["layout"]))
    for key, expected in (("logit", bridge["signed_logit"]), ("probability", bridge["signed_probability"])):
        if not math.isclose(row["bridge_A_minus_R"][key], expected, rel_tol=0, abs_tol=EPS):
            raise ValueError("saved/recomputed R-to-A bridge differs")
    full_drift = recovery(abs(full["B"]["logit"]-a), abs(full["C"]["logit"]-a))
    anchors = {k: {f: finite(branch[k]["anchors512"]["score"][f]) for f in ("logit", "probability")} for k in "BC"}
    anchor_drift = recovery(abs(anchors["B"]["logit"]-a), abs(anchors["C"]["logit"]-a))
    feature = {}
    for stage in ("patches", "encoded", "context"):
        sides = {k: {s: finite(branch[k]["anchors512"]["errors_to_A"][s][stage]["relative_l2"], True)
                     for s in "ab"} for k in "BC"}
        means = {k: statistics.fmean(sides[k].values()) if all(v is not None for v in sides[k].values()) else None for k in "BC"}
        feature[stage] = dict(side_relative_l2=sides,
            pair_mean_relative_l2=recovery(means["B"], means["C"]),
            max_abs={k: max(finite(branch[k]["anchors512"]["errors_to_A"][s][stage]["max_abs"]) for s in "ab") for k in "BC"})
    return dict(pair_id=row["pair_id"], dataset=row["dataset"], name=row.get("name"), stratum=row.get("stratum"),
        fragment_a=row["fragment_a"], fragment_b=row["fragment_b"], arrays_sha256=row["arrays_sha256"],
        full_scores=full, anchor_scores=anchors, bridge_R_to_A=bridge,
        feature_errors=feature, full_logit_absolute_drift_from_A=full_drift,
        anchor_logit_absolute_drift_from_A=anchor_drift,
        layout={k: branch[k]["layout"] for k in "RABC"},
        layout_change_from_A={k: layout_change(branch["A"]["layout"], branch[k]["layout"]) for k in "BC"},
        sinkhorn_converged={k: branch[k]["sinkhorn_converged"] for k in "RABC"})


def summarize_layout_changes(changes):
    return dict(valid_transitions=dict(Counter(r["valid_transition"] for r in changes)),
        translation_change_px=stats(r["translation_change_px"] for r in changes),
        inlier_delta=stats(r["inlier_delta"] for r in changes),
        residual_delta_px=stats(r["residual_delta_px"] for r in changes))


def summarize(rows):
    return dict(pair_count=len(rows), dataset_counts=dict(Counter(r["dataset"] for r in rows)),
        bridge_R_to_A=dict(signed_logit=stats(r["bridge_R_to_A"]["signed_logit"] for r in rows),
            absolute_logit=stats(r["bridge_R_to_A"]["absolute_logit"] for r in rows),
            signed_probability=stats(r["bridge_R_to_A"]["signed_probability"] for r in rows),
            pairs_with_original_padding=sum(r["bridge_R_to_A"]["original_has_padding"] for r in rows),
            layout_change=summarize_layout_changes([r["bridge_R_to_A"]["layout_change"] for r in rows])),
        anchor_context_relative_l2=summarize_recovery(r["feature_errors"]["context"]["pair_mean_relative_l2"] for r in rows),
        full_logit_absolute_drift_from_A=summarize_recovery(r["full_logit_absolute_drift_from_A"] for r in rows),
        anchor_logit_absolute_drift_from_A=summarize_recovery(r["anchor_logit_absolute_drift_from_A"] for r in rows),
        patch_and_encoded_max_abs={stage: {k: max(r["feature_errors"][stage]["max_abs"][k] for r in rows) for k in "BC"} for stage in ("patches", "encoded")},
        layouts={k: dict(valid_pairs=sum(r["layout"][k]["valid"] for r in rows),
            reasons=dict(Counter(r["layout"][k]["reason"] for r in rows)),
            inliers=stats(r["layout"][k]["inlier_count"] for r in rows),
            candidates=stats(r["layout"][k]["candidate_count"] for r in rows),
            residual_px=stats(r["layout"][k]["residual_px"] for r in rows),
            sinkhorn_converged_pairs=sum(r["sinkhorn_converged"][k] for r in rows)) for k in "RABC"},
        layout_change_from_A={k: summarize_layout_changes([r["layout_change_from_A"][k] for r in rows]) for k in "BC"})


def analyze(result, cases):
    protocol = result["protocol"]
    if (protocol.get("schema") != "density-physical-context/2" or protocol.get("status") != "complete"
            or protocol.get("completed_cases") != 40 or protocol.get("case_count") != 40
            or protocol.get("parameters_unchanged") is not True
            or protocol.get("original_conv_configuration_unchanged") is not True
            or protocol["model"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT):
        raise ValueError("requires complete exact40 S7 run with unchanged parameters/conv configuration")
    if any(protocol.get(k) is not False for k in ("GPU_used", "training", "threshold_fit", "GT_used", "physical_masks_resized", "coarse_size_changed")):
        raise ValueError("unexpected experiment scope")
    rows = result["rows"]
    reference = {r["pair_id"]: r for r in cases}
    ids = [r["pair_id"] for r in rows]
    if len(rows) != 40 or len(set(ids)) != 40 or len(cases) != 40 or set(ids) != set(reference):
        raise ValueError("requires all exact40 unique pair IDs, never an intersection/subset")
    for row in rows:
        ref = reference[row["pair_id"]]
        if any(row[k] != ref[k] for k in ("dataset", "fragment_a", "fragment_b", "arrays_sha256")):
            raise ValueError("case identity or source arrays differ")
    paired = [pair_record(r) for r in rows]
    if Counter(r["dataset"] for r in paired) != {"real": 20, "ood": 12, "test": 8}:
        raise ValueError("predeclared dataset counts differ")
    groups = {"all": summarize(paired)}
    groups.update({d: summarize([r for r in paired if r["dataset"] == d]) for d in ("real", "ood", "test")})
    return dict(schema="density-physical-offline-analysis/1", status="complete",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        checks=dict(exact40_pair_ids=True, all_fragment_endpoints_and_arrays_equal=True,
            no_pair_removed=True, original_parameters_unchanged=True, original_conv_config_unchanged=True),
        definitions=dict(unit="one pair; each context relativeL2 value is mean(A-side,B-side), then equally weighted across pairs; not80 independent fragments",
            reference="A uniform512 bridge baseline, separately report R(original saved512) to A",
            recovery_fraction="(B_error-C_error)/B_error; no clipping; undefined if B<=1e-12",
            direction="C closer/farther/equal to A using absolute1e-12 tolerance; not correctness",
            aggregate_recovery="1-sum(C_error)/sum(B_error) over same available pairs; separate from mean/median per-pair recovery",
            translation_change="Euclidean distance between predicted layout translations, only where BOTH decoder outputs are valid; not GT error",
            missing="null stays missing; every summary records observed pair count"),
        protocol_evidence={k: protocol[k] for k in ("parameter_sha256", "script_sha256", "cases_sha256", "selection_json_sha256", "parameters_unchanged", "original_conv_configuration_unchanged")},
        groups=groups, pairs=paired, limitations=protocol["limitations"],
        no_GT_accuracy_metrics=True, no_threshold_fit=True, no_model_execution=True)


def findings(result):
    def number(x):
        return "NA" if x is None else f"{x:.6f}"
    lines = ["# S7 density physical v2：40对离线数值整理", "",
        "完整40对及端点/数组身份与既定清单一致；源protocol complete，记录参数与卷积配置未变。以下以pair为单位，context每对先取两侧相对L2均值，未把80侧当80个独立样本。", "",
        "| 数据域 | pairs | R→A绝对logit漂移均值 | B context L2 | C context L2 | full logit漂移 B/C | anchor logit漂移 B/C |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, g in result["groups"].items():
        ctx, full, anchor = (g[k] for k in ("anchor_context_relative_l2", "full_logit_absolute_drift_from_A", "anchor_logit_absolute_drift_from_A"))
        lines.append(f"| {name} | {g['pair_count']} | {number(g['bridge_R_to_A']['absolute_logit']['mean'])} | {number(ctx['B']['mean'])} | {number(ctx['C']['mean'])} | {number(full['B']['mean'])}/{number(full['C']['mean'])} | {number(anchor['B']['mean'])}/{number(anchor['C']['mean'])} |")
    lines += ["", "表内为均值；中位数、逐pair值及分母保存在analysis.json。R→A是原采样到强制uniform512的桥接，不并入B/C密度解释。", "",
        "| 数据域 | context恢复比例 | full恢复比例 | anchor恢复比例 | C更接近A的对数 context/full/anchor |",
        "|---|---:|---:|---:|---|"]
    for name, g in result["groups"].items():
        records = [g[k] for k in ("anchor_context_relative_l2", "full_logit_absolute_drift_from_A", "anchor_logit_absolute_drift_from_A")]
        ratios = [number(r["ratio_of_paired_mean_drift_reduction"]) for r in records]
        closer = "/".join(str(r["directions"].get("closer", 0)) for r in records)
        lines.append(f"| {name} | {' | '.join(ratios)} | {closer} |")
    lines += ["", "恢复比例=1−C漂移/B漂移（配对均值之比，不截断）；负值表示偏离A更大，不是准确率下降。完整/anchor差异同时含Scorer attention和pooling变化。", "",
        "| 数据域 | decoder valid R/A/B/C | A→B位移变化中位数(px;n) | A→C位移变化中位数(px;n) |",
        "|---|---|---:|---:|"]
    for name, g in result["groups"].items():
        b, c = (g["layout_change_from_A"][k]["translation_change_px"] for k in "BC")
        valid = "/".join(str(g["layouts"][k]["valid_pairs"]) for k in "RABC")
        lines.append(f"| {name} | {valid} | {number(b['median'])};{b['observed_pair_count']} | {number(c['median'])};{c['observed_pair_count']} |")
    lines += ["", "位移变化只比较两个有效decoder输出，不是GT误差，也不代表摆对率。valid转换、inlier/candidate数、残差及逐例translation全部保留在analysis.json。", "",
        "限制：40对原本按score strata选取；S7冻结权重下的描述性干预不能证明总体准确率或收敛。仅固定每片相对512的卷积物理邻域；GN和landmark仍用1024，非跨片统一3px，也未改变mask/128 coarse/patch窗口。没有GT准确率、Recall、阈值拟合或新推理。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results.json")
    parser.add_argument("--cases", type=Path, default=HERE.parent / "heatmaps_v1/s7/cases.json")
    parser.add_argument("--output", type=Path, default=HERE / "analysis_v1")
    args = parser.parse_args()
    result, result_source = read(args.results)
    cases, cases_source = read(args.cases)
    if cases_source["sha256"] != EXPECTED_CASES or result["protocol"]["cases_sha256"] != cases_source["sha256"]:
        raise ValueError("source cases SHA does not match pinned existing40")
    analysis = analyze(result, cases)
    analysis["sources"] = dict(results=result_source, cases=cases_source,
        analyzer=dict(path=str(Path(__file__)), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (args.output / "FINDINGS.md").write_text(findings(analysis))
    print(json.dumps(dict(status="complete", pair_count=40, datasets=analysis["groups"]["all"]["dataset_counts"], output=str(args.output))))


if __name__ == "__main__":
    main()
