"""Offline S7 C8/C16 fixed endpoint comparison; consume COMPLETE endpoints only.

Missing endpoint protocols produce a clearly partial result. No fetching,
threshold fitting, inference, queue mutation, or model selection occurs here.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
R = "/root/autodl-tmp/rachel_score_design_20260913_001/"
PREFIX = {"C8": "s6_s7_20260915/priority_after_s5/s7_augmented_full24",
          "C16": "scorer_diagnosis_20260919/continuation_v1/s7_c16"}
SOURCES = {}
METRICS_SOURCE = HERE.parent / "continuation_endpoint_v1/analyze.py"
spec = importlib.util.spec_from_file_location("completed_s6_endpoint_metrics", METRICS_SOURCE)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def read(relative, lines=False):
    path = RAW / relative
    data = path.read_bytes()
    SOURCES[relative] = {"remote_path":R+relative,"local_path":str(path),
                         "sha256":hashlib.sha256(data).hexdigest()}
    return [json.loads(line) for line in data.splitlines()] if lines else json.loads(data)


def accepted(row, threshold):
    return bool(row["decision_valid"] and row["classification"]["fused"] >= threshold)


def good_layout(row):
    layout = row["layouts"]["full_top2_mode"]
    return bool(row["label"] and layout["valid"] and layout.get("translation_l2_px") is not None
                and layout["translation_l2_px"] <= 20)


def run():
    queue = read("scorer_diagnosis_20260919/continuation_v1/queue_status.json")
    train = read(PREFIX["C16"] + "/protocol.json")
    status = read(PREFIX["C16"] + "/status.json")
    freeze = read(PREFIX["C16"] + "/classifier_freezes/freeze.json")
    assert status["status"] == "complete" and status["classifier_epochs"] == 16
    comparisons, populations, gates, models, missing = [], {}, [], {}, []
    for split in ("test", "real", "ood"):
        new_protocol = RAW / PREFIX["C16"] / "evaluation/fixed_epoch" / split / "protocol.json"
        if not new_protocol.exists():
            missing.append({"split":split,"reason":"no collected COMPLETE endpoint protocol"})
            continue
        # Read readiness receipt before any possibly partial rows/summary.
        ready = read(PREFIX["C16"] + "/evaluation/fixed_epoch/" + split + "/protocol.json")
        if ready.get("status") != "complete":
            missing.append({"split":split,"reason":"endpoint protocol not complete"})
            continue
        summaries, selected, all_rows = {}, {}, {}
        for arm in ("C8", "C16"):
            root = PREFIX[arm] + "/evaluation/fixed_epoch/" + split
            assert read(root + "/protocol.json")["status"] == "complete"
            summaries[arm] = read(root + "/summary.json")
            assert summaries[arm]["status"] == "complete"
            rows = read(root + "/pair_results.jsonl", lines=True)
            all_rows[arm] = rows
            assert len({r["pair_id"] for r in rows}) == len(rows)
            selected[arm] = [r for r in rows if split != "real" or not r["label"] or r["review_status"] == "keep"]
            identity = summaries[arm]["model"]
            models[arm] = {k:identity[k] for k in ("checkpoint_path","checkpoint_sha256","epoch","selection")}
        assert [r["pair_id"] for r in all_rows["C8"]] == [r["pair_id"] for r in all_rows["C16"]]
        for a,b in zip(all_rows["C8"],all_rows["C16"]):
            for key in ("label","fragment_a","fragment_b","target_translation_rc","review_status"):
                assert a.get(key) == b.get(key), (split,key,a["pair_id"])
        group = "kept_plus_all_negative" if split == "real" else "all"
        assert len(selected["C16"]) == {"test":3000,"real":803,"ood":301}[split]
        populations[split] = {"ordered_pair_ids_equal":True,"labels_endpoints_GT_review_equal":True,
            "all_count":len(all_rows["C16"]),"analyzed_group":group,"analyzed_count":len(selected["C16"]),
            "positive_count":sum(r["label"] for r in selected["C16"]),
            "exact_full_layout_objects_equal":all(a["layouts"]==b["layouts"] for a,b in zip(all_rows["C8"],all_rows["C16"]))}
        for op in ("max_f1", "recall_95"):
            old_t = summaries["C8"]["model"]["operating_points"]["thresholds"][op]
            new_t = summaries["C16"]["model"]["operating_points"]["thresholds"][op]
            for arm,t,source in (("C8",old_t,"C8_SIMVAL_frozen"),("C16",old_t,"C8_SIMVAL_frozen"),
                                 ("C16",new_t,"C16_SIMVAL_frozen")):
                measured = shared.metrics(selected[arm],t,split)
                if arm == "C8" or source == "C16_SIMVAL_frozen":
                    official = summaries[arm]["groups"][group]["classification"]["fused"][op]
                    for key,value in official.items():
                        assert math.isclose(measured[key],value,rel_tol=1e-10,abs_tol=1e-10), (split,arm,key)
                comparisons.append({"split":split,"group":group,"checkpoint":arm,"operating_point":op,
                                    "threshold_source":source,"metrics":measured})
            if split == "real":
                good = [(a,b) for a,b in zip(selected["C8"],selected["C16"]) if good_layout(a) and good_layout(b)]
                for mode,t in (("same_old_threshold",old_t),("own_SIMVAL_threshold",new_t)):
                    old_accept = sum(accepted(a,old_t) for a,b in good)
                    new_accept = sum(accepted(b,t) for a,b in good)
                    lost = [a["pair_id"] for a,b in good if accepted(a,old_t) and not accepted(b,t)]
                    rescued = [a["pair_id"] for a,b in good if not accepted(a,old_t) and accepted(b,t)]
                    gates.append({"operating_point":op,"mode":mode,"C8_threshold":old_t,"C16_threshold":t,
                        "common_correct_layout_positive_count":len(good),"C8_accepted_correct":old_accept,
                        "C16_accepted_correct":new_accept,"C8_rejected_correct":len(good)-old_accept,
                        "C16_rejected_correct":len(good)-new_accept,"lost_count":len(lost),"rescued_count":len(rescued),
                        "lost_pair_ids":lost,"rescued_pair_ids":rescued})
    history = []
    for epoch in (26,27,28):
        path = PREFIX["C16"] + f"/validation_{epoch:03d}.json"
        if (RAW/path).exists():
            v=read(path); history.append({"classifier_epoch":epoch-12,
                "SIMVAL_max_f1":v["operating_points"]["validation"]["max_f1"],
                "SIMVAL_recall95":v["operating_points"]["validation"]["recall_95"]})
    endpoint_stages = [{"name":s["name"],"status":s.get("status","not_started")}
                       for s in queue["stages"] if s["name"].startswith("s7_")
                       and s["kind"] == "endpoint_evaluation"]
    endpoint_complete = sum(s["status"] == "complete" for s in endpoint_stages)
    output = {"schema_version":"S7-C8-C16-fixed-endpoint-comparison/1", "status":"partial" if missing else "complete",
        "status_scope":"fixed_epoch TEST/REAL/OOD comparison only; not all nine queued evaluations",
        "collected_at_utc":datetime.now(timezone.utc).isoformat(),"sources":SOURCES,
        "shared_metrics_source":str(METRICS_SOURCE),"shared_metrics_sha256":hashlib.sha256(METRICS_SOURCE.read_bytes()).hexdigest(),
        "queue_snapshot":{k:queue.get(k) for k in ("status","active_stage","active_name","active_pid")},
        "completed_s7_stages":sum(s.get("status")=="complete" for s in queue["stages"] if s["name"].startswith("s7_")),
        "all_nine_evaluation_queue_snapshot":{"complete":endpoint_complete,"total":len(endpoint_stages),
            "all_complete":endpoint_complete == len(endpoint_stages),"stages":endpoint_stages},
        "missing_endpoints":missing,"models":models,"training_status":status,
        "continuation_identity":train["continuation_identity"],
        "frozen_selection_epochs":{k:v["selected_epoch"] for k,v in freeze["selections"].items()},
        "populations":populations,"comparisons":comparisons,"correct_layout_gate_transitions":gates,
        "SIMVAL_C14_C16":history,"thresholds_fitted_here":False,"new_inference_performed":False,
        "caveats":["S7 original C8 and C16 fixed endpoints; neither selected by REAL/OOD.",
                   "REAL uses kept295 positive plus unchanged508 negative.",
                   "OOD positive-only301: no binary Accuracy/Precision/F1/AUC estimate.",
                   "Only source endpoints with complete protocol were opened."]}
    filename = "comparison_partial.json" if missing else "comparison.json"
    (HERE/filename).write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    lines = ["# S7 C8→C16 终点对照", "", "固定epoch三域主比较状态："+output["status"]+"。固定旧C8阈值与新C16 SIMVAL阈值分开；不在真实域拟合。", "",
        f"训练C16已完成。采集时全部评估队列完成{endpoint_complete}/{len(endpoint_stages)}，当前为{queue.get('active_name')}；本报告不宣称9个评估全部完成。", "",
        "C8=epoch20，C16=epoch28；原S7 Matcher冻结，原数据与PairBCE不变，追加8轮/192K pair曝光；LR2e-5，physical/effective batch16，恢复AdamW与RNG。", ""]
    if missing: lines += ["尚缺："+"、".join(r["split"] for r in missing)+"；没有读取这些未完成终点的数据。", ""]
    lines += ["| 数据 | 权重／阈值 | 阈值 | Accuracy | Recall | F1 | TP／FP | AUROC／AUPRC |",
              "|---|---|---:|---:|---:|---:|---|---|"]
    for r in comparisons:
        if r["operating_point"]!="max_f1":continue
        m=r["metrics"]; tag=r["checkpoint"]+"／"+("旧" if r["threshold_source"].startswith("C8") else "新")
        if r["split"]=="ood":
            lines.append(f"| OOD301正例 | {tag} | {m['threshold']:.9f} | — | {m['positive_recall']:.2%} | — | {m['accepted_positive_count']}／无负例 | — |")
        else:
            lines.append(f"| {r['split']} | {tag} | {m['threshold']:.9f} | {m['accuracy']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} | {m['tp']}／{m['fp']} | {m['auroc']:.6f}／{m['auprc']:.6f} |")
    lines += ["", "## 正确layout被分类拒绝", ""]
    for g in gates:
        if g["operating_point"]=="max_f1":
            lines.append(f"- {g['mode']}：共同layout≤20正例{g['common_correct_layout_positive_count']}；拒绝{g['C8_rejected_correct']}→{g['C16_rejected_correct']}，获准{g['C8_accepted_correct']}→{g['C16_accepted_correct']}；续训丢失{g['lost_count']}、救回{g['rescued_count']}。")
    if not missing:
        lines += ["", "## 解释与边界", "",
            "- TEST排序及F1改善，但真实域没有同步获益：敦煌AUROC 0.720019→0.720646几乎不变，AUPRC 0.617488→0.607153。不能从单次实验称其显著改善或显著退化。",
            "- 与S6不同，S7固定旧阈值时REAL Recall略升、F1近乎不变，OOD多找回4对；采用自身SIMVAL新阈值后召回明显下降。因此这次召回下降主要伴随SIMVAL工作点升高，不应归结为所有真实样本分数都退化。",
            "- TEST3000、REAL全部1016、OOD301的pair顺序/标签/GT/review一致，完整layout对象也逐例相同。保留敦煌原始摆对216/295未改变；门控变化不是摆放变化。",
            "- OOD只有301正例，无layout GT；只报告正例召回，不报告二分类Accuracy/Precision/F1或摆放准确率。",
            "- 所有模型/阈值在SIMVAL冻结；本分析不依据REAL/OOD选模型，不启动新训练或推理。"]
    lines += ["", "JSON包含Recall95两种阈值、精确数值、case ID与源文件哈希。所有原始数据保留在raw/，不覆盖旧训练或评价。", ""]
    (HERE/"FINDINGS.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"status":output["status"],"compared_splits":list(populations),"missing":missing},ensure_ascii=False))


if __name__ == "__main__":
    run()
