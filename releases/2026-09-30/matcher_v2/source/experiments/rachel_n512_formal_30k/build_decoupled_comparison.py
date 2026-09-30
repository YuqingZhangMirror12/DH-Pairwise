"""Compare completed frozen decoupled arms to existing references, without inference.

The registered fixed-epoch20 checkpoint is primary. All independently frozen
SIM-VAL selections remain inspectable; this script never ranks held-out winners.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k.build_s3_first_readout import (
    DEFAULT_ROOT, classification, first_five, historical, layout, metrics, pct,
    read, unique, with_layout,
)

ARMS = {
    "s3_matrix": "S3 v2（实现诊断）",
    "s3_matrix_per_pair_norm_v3": "S3 v3（逐对归一化修复）",
    "s4_cross_attention": "S4 Cross-attention",
    "s5_control512": "S5 paired512 对照",
    "s5_step3_cap2048": "S5 等弧长步长／cap2048",
}
SELECTIONS = ("fixed_epoch", "max_f1", "recall95")
SPLITS = ("test", "real", "ood")
POPULATIONS = {"test": "all", "real": "kept_plus_all_negative", "ood": "all"}


def selection_record(evaluations, arm, selection):
    chosen = {split: unique(evaluations, arm=arm, selection=selection, split=split)
        for split in SPLITS}
    if any(row["status"] != "complete" for row in chosen.values()):
        raise ValueError("cannot compare unfinished evaluation: " + arm + "/" + selection)
    first = chosen["test"]
    for row in chosen.values():
        for key in ("checkpoint_sha256", "selected_epoch", "operating_points", "training_identity"):
            if row[key] != first[key]:
                raise ValueError("cross-domain frozen identity differs: " + key)
    identity = first["training_identity"]
    training = dict(first["training_budget"], seed=identity["seed"],
        head_seed=identity.get("head_seed"), classifier_phase_seed=identity.get("classifier_phase_seed"),
        matrix_head_revision=identity.get("matrix_head_revision"),
        logical_microbatch=identity["microbatch"], effective_batch=identity["effective_batch"],
        checkpoint_sha256=first["checkpoint_sha256"],
        checkpoint_selection_rule=selection + "; SIM VAL only; fixed_epoch20 is primary",
        equal_budget_to_first5_or_history=False,
        normalization_note="samplewise logical micro1; not identical to historical micro4 normalization")
    operating_points = {}
    for op in ("max_f1", "recall_95"):
        record, poses = {}, {}
        for split in SPLITS:
            raw = unique(chosen[split]["classification_rows"], population=POPULATIONS[split],
                branch="fused", operating_point=op)
            record[split] = classification(raw)
            if raw["threshold"] != first["operating_points"]["thresholds"][op]:
                raise ValueError("threshold not frozen on the selected checkpoint")
            if split != "ood":
                poses[split] = layout(unique(chosen[split]["layout_rows"],
                    population=POPULATIONS[split], operating_point=op, tolerance_px=20))
        record["negative_strata"] = {name: classification(unique(chosen["real"]["classification_rows"],
            population="negative_" + name, branch="fused", operating_point=op))
            for name in ("strict", "constructed")}
        operating_points[op] = with_layout(record, poses)
    return dict(training=training, operating_points=operating_points,
        paired_baseline_layout={split: row.get("paired_baseline_layout") for split, row in chosen.items()},
        sources=[str(Path(row["evaluation"]) / filename) for row in chosen.values()
            for filename in ("protocol.json", "summary.json", "prediction_complete.json", "pair_results.jsonl")])


def current_arm(evaluations, arm):
    selected = [row for row in evaluations if row["arm"] == arm]
    expected = {(s, p) for s in SELECTIONS for p in SPLITS}
    if len(selected) != 9 or {(row["selection"], row["split"]) for row in selected} != expected:
        raise ValueError("arm requires nine distinct registered evaluations: " + arm)
    selections = {key: selection_record(selected, arm, key) for key in SELECTIONS}
    primary = selections["fixed_epoch"]
    if primary["training"]["selected_epoch"] != 20:
        raise ValueError("primary checkpoint must be fixed epoch20")
    return dict(model=arm, display_label=ARMS[arm], training=primary["training"],
        **deepcopy(primary["operating_points"]["max_f1"])), selections


def build(root, frozen_readout, required_arms):
    root, frozen_readout = Path(root), Path(frozen_readout)
    data = read(frozen_readout)
    if any(data[key] for key in ("threshold_fitting_performed", "held_out_model_selection_performed", "smoke_counted")):
        raise ValueError("only frozen, formal, non-fitting evidence may enter comparison")
    old_path = root / "s2_first5_readout_20260914/comparison.json"
    hist_path = root / "historical_references_20260914/references.json"
    early, early_sources = first_five(old_path)
    models = historical(hist_path) + early
    selections, pending = {}, {}
    for arm in ARMS:
        rows = [r for r in data["evaluations"] if r["arm"] == arm]
        if len(rows) != 9 or any(r["status"] != "complete" for r in rows):
            if arm in required_arms:
                raise ValueError("required arm not fully evaluated: " + arm)
            pending[arm] = [{k: r.get(k) for k in ("selection", "split", "status", "reason")} for r in rows]
            continue
        model, selections[arm] = current_arm(rows, arm)
        models.append(model)
    for model in models:
        for split, expected in (("test", (1500, 1500)), ("real", (295, 508)), ("ood", (301, 0))):
            if (model[split]["positive_count"], model[split]["negative_count"]) != expected:
                raise ValueError("comparison population changed: " + model["model"])
    files = [old_path, hist_path, frozen_readout]
    return dict(schema_version="rachel-decoupled-comparison/1", status="complete_available_arms",
        models=models, selections=selections, pending_arms=pending,
        primary="registered fixed_epoch20 for new arms; each frozen checkpoint SIM VAL max_f1 threshold",
        new_inference=False, threshold_fitting_performed=False, held_out_model_selection_performed=False,
        evidence=dict(files=[str(p.resolve()) for p in files] + early_sources,
            source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),
        caveats=[
            "S0–S2为random5/120K；Full为random6/144K；历史E1最后5轮120K另有未量化的继承训练；新S3/S4为M12+C8/480K逻辑预算，不能据此单独归因架构。",
            "S4复用确切M12，只新增192K分类训练；S3 v3是原权重的显式推理归一化修复，新增优化为0，另做SIM VAL选模及阈值冻结。",
            "Full历史checkpoint按VAL P@95R→AP选取；此处使用其已有max-F1阈值，并未重新选模。",
            "S3 v2仅保留为已知train/eval归一化错位的实现诊断，不能视为二维CNN或分阶段训练失败的有效证据。",
            "敦煌保留集为回顾性人工筛选，统一295正+508负（39 strict+469 constructed）；不等于盲测总体。",
            "20px raw布局为不经过分类筛选的诊断；FN但摆对没有自动改为分类正确，也没有实施GT救回。",
            "Turufan仅301正例且无layout GT，只报告正例接受/召回，不报告Accuracy、Precision、F1、AP、AUROC或摆放准确率。",
            "辅助max_f1/recall95是独立SIM VAL checkpoint选择；JSON保留全部选择，不按REAL/OOD结果挑选赢家。",
        ])


def render(data):
    lines = ["# 分阶段分类实验与既有参考", "", data["primary"], "",
        "百分比单位%。新实验主表固定epoch20；JSON同时保留三种事前登记选模及两种阈值。", "",
        "| 模型 | TEST P / R / F1 / AP | 敦煌保留 P / R / F1 / AP | Raw摆对20px | 通过且摆对 | FN但摆对 | OOD接受 / 召回 |",
        "|---|---|---|---:|---:|---:|---:|"]
    for m in data["models"]:
        lines.append("| %s | %s | %s | %d/295 | %d/295 | %d | %d/301 / %s%% |" % (
            m["display_label"], metrics(m["test"]), metrics(m["real"]), m["raw_layout20"],
            m["gated_layout20"], m["fn_good20"], m["ood"]["tp"], pct(m["ood"]["recall"])))
    lines += ["", "## 新实验的独立冻结选择", "",
        "| 模型 · 选择 · epoch | 阈值工作点 | TEST P / R / F1 | 敦煌 P / R / F1 | 敦煌FP/508 | OOD接受/301 |",
        "|---|---|---|---|---:|---:|"]
    for arm, selections in data["selections"].items():
        for name, selection in selections.items():
            for op, row in selection["operating_points"].items():
                lines.append("| %s · %s · %d | %s · %.6g | %s | %s | %d | %d |" % (
                    ARMS[arm], name, selection["training"]["selected_epoch"], op, row["test"]["threshold"],
                    " / ".join(pct(row["test"][k]) for k in ("precision", "recall", "f1")),
                    " / ".join(pct(row["real"][k]) for k in ("precision", "recall", "f1")),
                    row["real"]["fp"], row["ood"]["tp"]))
    lines += ["", "## 解释边界", ""] + ["- " + c for c in data["caveats"]]
    lines += ["", "## 尚未完成或未纳入的实验", "", ", ".join(data["pending_arms"]) or "无", "",
        "## 来源", ""] + ["- [%s](%s)" % (Path(p).name, p) for p in data["evidence"]["files"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--frozen-readout", type=Path, required=True)
    parser.add_argument("--required-arm", choices=tuple(ARMS), action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = build(args.root, args.frozen_readout, args.required_arm)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "READOUT.md").write_text(render(data))
    print(json.dumps(dict(status=data["status"], models=[m["model"] for m in data["models"]], output=str(args.output.resolve()))))


if __name__ == "__main__":
    main()
