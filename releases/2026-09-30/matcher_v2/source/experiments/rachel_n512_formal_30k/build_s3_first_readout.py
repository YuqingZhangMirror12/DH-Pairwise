"""Merge existing frozen S3 / first-five / historical readouts, without inference.

No threshold fitting, model selection, or training. Standard-library only.
All measurement values are read from artifacts; only wording is authored here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "reports/rachel_score_design_20260913_001"
SCHEMA = "rachel-s3-first-comparison/1"


def read(path):
    return json.loads(Path(path).read_text())


def unique(rows, **filters):
    found = [row for row in rows if all(row.get(k) == v for k, v in filters.items())]
    if len(found) != 1:
        raise ValueError("expected exactly one frozen row: %r; got%d" % (filters, len(found)))
    return found[0]


def classification(row, counts=None):
    counts = row if counts is None else counts
    result = {key: row.get(key) for key in ("threshold", "precision", "recall", "f1", "accuracy", "auroc", "tp", "fp", "fn", "tn")}
    result["ap"] = row.get("ap", row.get("auprc"))
    result.update(positive_count=counts["positive_count"], negative_count=counts["negative_count"])
    result["accepted_positive_count"] = result["tp"]
    result["fpr"] = row.get("fpr", row.get("false_positive_rate"))
    if result["fpr"] is None and result["fp"] is not None and result["negative_count"]:
        result["fpr"] = result["fp"] / result["negative_count"]
    if result["recall"] is None:
        result["recall"] = row.get("positive_recall")
    if not result["positive_count"] or not result["negative_count"]:
        for key in ("precision", "f1", "accuracy", "ap", "auroc"):
            if result[key] is not None:
                raise ValueError("single-class artifact claims a full-binary metric")
    if result["positive_count"]:
        if result["tp"] + result["fn"] != result["positive_count"]:
            raise ValueError("positive confusion counts differ from cohort")
    if result["negative_count"]:
        if result["fp"] + result["tn"] != result["negative_count"]:
            raise ValueError("negative confusion counts differ from cohort")
    return result


def layout(row):
    result = dict(positive_count=row["positive_count"],
        raw_correct=row.get("raw_correct", row.get("raw_good_count")),
        accepted_correct=row.get("accepted_correct", row.get("accepted_good_count")),
        fn_good=row.get("classification_FN_but_layout_correct", row.get("fn_raw_good_count")))
    if result["raw_correct"] != result["accepted_correct"] + result["fn_good"]:
        raise ValueError("raw layout does not partition into accepted and FN-good")
    result["raw_recall"] = result["raw_correct"] / result["positive_count"]
    result["gated_recall"] = result["accepted_correct"] / result["positive_count"]
    return result


def with_layout(record, by_split):
    record["layout20"] = by_split
    record.update(raw_layout20=by_split["real"]["raw_correct"],
        gated_layout20=by_split["real"]["accepted_correct"], fn_good20=by_split["real"]["fn_good"])
    return record


def first_five(source):
    data = read(source)
    if data["status"] != "complete":
        raise ValueError("first-five comparison incomplete")
    result, evidence = [], []
    for name in ("S0", "S1", "S2"):
        reference = unique(data["sources"], model=name, population="SIM TEST")
        summary_path = Path(reference["summary"])
        summary = read(summary_path)
        meta = summary["model"]
        if summary["status"] != "complete" or meta["selection"] != "max_f1" or meta["epoch"] != 5:
            raise ValueError("not the frozen first-five maxF1 checkpoint")
        training = dict(selected_epoch=meta["epoch"], budget_epochs=meta["budget"], seed=meta["seed"],
            selected_pair_exposures=meta["winner_record"]["selected_global_exposure"],
            initialization="random initialization (documented first-five protocol)",
            checkpoint_selection_rule="SIM VAL max-F1 at first-five budget",
            checkpoint_sha256=meta["checkpoint_sha256"], equal_budget_to_S3=False,
            normalization_note="historical logical micro4/effective16; not S3 samplewise micro1 objective")
        entry = dict(model=name, display_label=name + " first5", training=training)
        poses = {}
        for split, population in (("test", "SIM TEST"), ("real", "敦煌保留"), ("ood", "Turufan")):
            raw = unique(data["classification"], model=name, population=population, operating_point="max_f1")
            entry[split] = classification(raw)
            if raw["threshold"] != meta["operating_points"]["thresholds"]["max_f1"]:
                raise ValueError("first-five threshold identity differs")
            if split != "ood":
                poses[split] = layout(unique(data["layout"], model=name, population=population,
                    operating_point="max_f1", tolerance_px=20))
        result.append(with_layout(entry, poses))
        evidence.append(str(summary_path.resolve()))
    return result, evidence


def historical(source):
    data = read(source)
    if data["status"] != "complete" or data["thresholds_refitted"] or data["new_inference"]:
        raise ValueError("historical reference contract differs")
    result = []
    for key, name in (("historical_e1", "E1"), ("full_e1_24k_winner", "Full-E1-24K")):
        old = data["models"][key]
        training = deepcopy(old["training"])
        training.update(initialization="warm start" if training["inherited_pretraining"] else "random initialization",
            checkpoint_sha256=old["checkpoint_sha256"], equal_budget_to_S3=False,
            normalization_note="historical micro4/effective16; not S3 samplewise micro1 objective")
        entry = dict(model=name, display_label=name + "（历史）", training=training)
        poses = {}
        measurements = old["measurements"]["max_f1"]
        for split, population in (("test", "test_full"), ("real", "real_keep_plus_all_negative"), ("ood", "ood_positive_only")):
            group = measurements[population]
            entry[split] = classification(group["classification"], group)
            if entry[split]["threshold"] != old["threshold_policies"]["max_f1"]:
                raise ValueError("historical maxF1 threshold differs")
            if split != "ood":
                poses[split] = layout(group["positive_only_layout"]["20"])
        result.append(with_layout(entry, poses))
    return result


def s3(source):
    data = read(source)
    if data["threshold_fitting_performed"] or data["held_out_model_selection_performed"] or data["smoke_counted"]:
        raise ValueError("S3 readout must contain frozen formal evaluation only")
    finished = [e for e in data["evaluations"] if e["arm"] == "s3_matrix" and e["status"] == "complete"]
    if len(finished) != 9 or {(e["selection"], e["split"]) for e in finished} != {
            (selection, split) for selection in ("fixed_epoch", "max_f1", "recall95") for split in ("test", "real", "ood")}:
        raise ValueError("S3 requires all nine completed frozen evaluation identities")
    if any(e["selected_epoch"] != 20 for e in finished) or len({e["checkpoint_sha256"] for e in finished}) != 1:
        raise ValueError("this first readout expects all three independently frozen selections at epoch20")
    selected = {split:unique(finished, selection="fixed_epoch", split=split) for split in ("test", "real", "ood")}
    first = selected["test"]
    if first["model_design"]["matrix_head_revision"] != "bn_relu_pool_v2":
        raise ValueError("legacy defective S3 head is not reportable as repaired S3")
    ident = first["training_identity"]
    training = dict(first["training_budget"], seed=ident["seed"], head_seed=ident["head_seed"],
        classifier_phase_seed=ident["classifier_phase_seed"], initialization="random M12 base; fresh repaired head; exact M12 reused",
        checkpoint_selection_rule="fixed epoch20 primary; auxiliary SIM VAL max-F1 and P@95R independently also selected epoch20",
        matrix_head_revision=ident["matrix_head_revision"], logical_microbatch=ident["microbatch"],
        effective_batch=ident["effective_batch"], checkpoint_sha256=first["checkpoint_sha256"],
        normalization_note="samplewise logical micro1; not equivalent to historical micro4 normalization")
    operating_points = {}
    for op in ("max_f1", "recall_95"):
        record, poses = {}, {}
        for split, population in (("test", "all"), ("real", "kept_plus_all_negative"), ("ood", "all")):
            evaluation = selected[split]
            raw = unique(evaluation["classification_rows"], population=population, branch="fused", operating_point=op)
            record[split] = classification(raw)
            if raw["threshold"] != first["operating_points"]["thresholds"][op]:
                raise ValueError("S3 split threshold differs from fixed checkpoint VAL")
            if split != "ood":
                poses[split] = layout(unique(evaluation["layout_rows"], population=population, operating_point=op, tolerance_px=20))
        record["negative_strata"] = {name:classification(unique(selected["real"]["classification_rows"],
            population="negative_" + name, branch="fused", operating_point=op)) for name in ("strict", "constructed")}
        operating_points[op] = with_layout(record, poses)
    record = dict(model="S3", display_label="S3 修复版 M12+C8", training=training, **deepcopy(operating_points["max_f1"]))
    costs = {}
    for split in ("test", "real", "ood"):
        low, high = operating_points["max_f1"][split], operating_points["recall_95"][split]
        costs[split] = dict(additional_accepted_positives=high["tp"]-low["tp"],
            additional_false_positives=None if low["fp"] is None else high["fp"]-low["fp"],
            recall_change=high["recall"]-low["recall"],
            additional_gated_good_layout=None if split == "ood" else
                operating_points["recall_95"]["layout20"][split]["accepted_correct"]-operating_points["max_f1"]["layout20"][split]["accepted_correct"])
    evidence = [str(Path(e["evaluation"]) / name) for e in selected.values()
        for name in ("protocol.json", "summary.json", "prediction_complete.json", "pair_results.jsonl")]
    evidence += [first["provenance"]["local_freeze_path"]]
    return record, operating_points, costs, evidence, data["counts"]


def build(root):
    root = Path(root)
    old_path = root / "s2_first5_readout_20260914/comparison.json"
    hist_path = root / "historical_references_20260914/references.json"
    s3_path = root / "s3_complete_20260914/frozen_readout/results.json"
    early, early_evidence = first_five(old_path)
    current, ops, costs, sources, counts = s3(s3_path)
    diagnostic_path = root / "s3_complete_20260914/train8_normalization_diagnostic.json"
    diagnostic = read(diagnostic_path)
    if (diagnostic["status"] != "complete" or diagnostic["held_out_read"] or diagnostic["weights_updated"] or
            diagnostic["thresholds_fitted"] or not diagnostic["identical_cached_q_for_both_modes"] or
            diagnostic["checkpoint_sha256"] != current["training"]["checkpoint_sha256"]):
        raise ValueError("normalization diagnostic must bind unchanged S3 weights and identical TRAIN-only Q")
    diagnosis = {key:diagnostic[key] for key in ("schema_version", "checkpoint_sha256", "sample_count",
        "population", "modes", "identical_cached_q_for_both_modes", "original_model_unchanged",
        "weights_updated", "held_out_read", "thresholds_fitted", "caveat")}
    current.update(display_label="S3 v2（实现诊断）", interpretation_status="implementation_diagnostic_not_architecture_comparison")
    models = historical(hist_path) + early + [current]
    for model in models:
        for split, expected in (("test",(1500,1500)), ("real",(295,508)), ("ood",(301,0))):
            if (model[split]["positive_count"],model[split]["negative_count"]) != expected:
                raise ValueError("comparison population changed")
    files = [old_path, hist_path, s3_path, diagnostic_path, root / "NEW_S345_PLAN_20260914.md"]
    return dict(schema_version=SCHEMA, status="complete", models=models,
        selected_operating_point="each frozen checkpoint SIM VAL max_f1", s3_operating_points=ops,
        s3_normalization_diagnostic=diagnosis,
        s3_interpretation="S3 v2：原样保留的实现诊断结果，尚非可信的架构胜负证据",
        s3_r95_cost=costs, collector_scope=counts, s3_complete_evaluations=9,
        pending_evaluations_not_S3="collector pending27 belongs to subsequent S4/S5; not missing S3 results",
        threshold_fitting_performed=False, held_out_model_selection_performed=False, new_inference=False,
        evidence=dict(files=[str(p.resolve()) for p in files]+early_evidence+sources,
            source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            current_fields="classification[]/layout[]; model+population+max_f1; layout tolerance20",
            historical_fields="models.*.measurements.max_f1; training checkpoint selection preserved",
            s3_fields="evaluations[arm=s3_matrix,selection=fixed_epoch,status=complete].classification_rows/layout_rows"),
        caveats=[
            "预算/起点不等：S0–S2为random5/120K；历史E1最后5轮120K另有未量化继承；Full为random6/144K；S3为M12+C8/480K。",
            "S3修复运行执行192K C曝光并继承288K M曝光；不把已继承M12称为本次重新训练。",
            "Full checkpoint按VAL P@95R→AP选择，此处仅采用该checkpoint的VAL max-F1阈值；其余选模规则分别保留。",
            "S3逻辑micro1/samplewise与旧micro4的监督归一化不完全等价；同有效batch16不能消除此差异。",
            "人工保留集为回顾性筛选；所有模型共同295正+508负，raw/gated layout分母均为295。",
            "Turufan只有301正例，无负例和layout GT；只能报告正例接受/召回，不能报告全二分类或layout准确率。",
            "同一TRAIN输入与Q已证实S3 v2存在严重train/eval归一化前向错位；这8例不衡量泛化。本汇总不据此否定分阶段或CNN方法。",
        ])


def pct(value):
    return "—" if value is None else "%.2f" % (100*value)


def metrics(row):
    return " / ".join(pct(row[key]) for key in ("precision","recall","f1","ap"))


def render(data):
    lines = ["# S3 首次完整冻结结果与既有参考", "",
        "**"+data["s3_interpretation"]+"。**", "",
        "S3 v2的M12+C8及9项三域评测已完成；三种事前登记选择均落在epoch20。下表原样保留固定epoch20的结果，并使用各checkpoint自己的SIM VAL max-F1阈值。没有在REAL/OOD选模型或拟合新阈值。", "",
        "百分比单位为%。SIM TEST为1500正+1500负；敦煌为人工保留295正+全部508负。", "",
        "| 模型 | TEST P / R / F1 / AP | 敦煌 P / R / F1 / AP | Raw布局20 | 分类通过且摆对 | FN但摆对 | OOD接受正例 / 召回 |",
        "|---|---|---|---:|---:|---:|---:|"]
    for model in data["models"]:
        lines.append("| %s | %s | %s | %d/295 | %d/295 | %d | %d/301 / %s%% |" % (
            model["display_label"], metrics(model["test"]), metrics(model["real"]),
            model["raw_layout20"], model["gated_layout20"], model["fn_good20"], model["ood"]["tp"], pct(model["ood"]["recall"])))
    diagnosis = data["s3_normalization_diagnostic"]
    lines += ["", "## 已确认的训练／推理模式错位", "",
        "固定TRAIN前4正+前4负、同一Q、同一权重，仅切换分类头的统计模式：", "",
        "| 统计模式 | BCE | 正例平均分 | 负例平均分 |", "|---|---:|---:|---:|"]
    for mode, label in (("train_per_pair_stats", "训练：逐pair统计"), ("eval_running_stats", "推理：running统计")):
        values = diagnosis["modes"][mode]
        lines.append("| %s | %.5f | %.5f | %.5f |" % (label, values["mean_bce"],
            values["positive_mean_score"], values["negative_mean_score"]))
    lines += ["", "该TRAIN-only检查确认了前向模式错位；未更新原权重、未读held-out或拟合阈值，不能用这8例声明泛化改善。后续须以明确版本解决训练／推理归一化一致性，再仅用SIM VAL选模型和阈值；原S3 v2指标不覆盖。", ""]
    lines += ["", "## S3提高召回的误报代价", "",
        "R95也是同一checkpoint在SIM VAL冻结的阈值，不保证其他域达到95%召回。", "",
        "| 工作点 / 阈值 | TEST P / R；FP | 敦煌 P / R；FP | 敦煌通过且摆对 / FN但摆对 | strict FP/39 | constructed FP/469 | OOD接受/301 |",
        "|---|---|---|---|---:|---:|---:|"]
    for op, row in data["s3_operating_points"].items():
        t,r,o = row["test"],row["real"],row["ood"]
        lines.append("| %s / %.6f | %s / %s；%d | %s / %s；%d | %d / %d | %d | %d | %d |" % (
            op,t["threshold"],pct(t["precision"]),pct(t["recall"]),t["fp"],pct(r["precision"]),pct(r["recall"]),r["fp"],
            row["gated_layout20"],row["fn_good20"],row["negative_strata"]["strict"]["fp"],row["negative_strata"]["constructed"]["fp"],o["tp"]))
    cost = data["s3_r95_cost"]["real"]
    lines += ["", "在敦煌保留集，R95相对max-F1额外接受%d个正例，同时增加%d个负例误报；分类通过且摆对增加%d例。" % (
        cost["additional_accepted_positives"],cost["additional_false_positives"],cost["additional_gated_good_layout"]), "",
        "## 比较限制", ""]
    lines += ["- "+note for note in data["caveats"]]
    lines += ["", "## 可复算来源", "", "完整精度、混淆矩阵、阈值、训练身份与来源文件在 `comparison.json`。核心输入：", ""]
    lines += ["- `"+path+"`" for path in data["evidence"]["files"][:4]]
    lines += ["", "重跑：`python -m experiments.rachel_n512_formal_30k.build_s3_first_readout --root <本轮reports目录>`。仅重写本汇总输出，不改原始结果。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build(args.root)
    output = args.output or args.root / "s3_complete_20260914"
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (output / "READOUT.md").write_text(render(result))
    print(json.dumps(dict(status="complete", model_count=len(result["models"]), output=str(output.resolve())), ensure_ascii=False))


if __name__ == "__main__":
    main()
