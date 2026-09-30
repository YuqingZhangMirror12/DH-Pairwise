"""Read collected C8/C16 endpoints; no fitting, inference, or remote writes."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
REMOTE = "/root/autodl-tmp/rachel_score_design_20260913_001/"
PREFIX = {"C8": "attention_depth_20260915/s4_cross_attention_depth2",
          "C16": "scorer_diagnosis_20260919/continuation_v1/s6_d2_c16"}
SOURCES = {}


def read(relative, lines=False):
    path = RAW / relative
    data = path.read_bytes()
    SOURCES[str(relative)] = {"remote_path": REMOTE + str(relative),
                             "local_path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    return [json.loads(x) for x in data.splitlines()] if lines else json.loads(data)


def ranking(rows):
    # Same tied-score ROC trapezoids/AP step definition as training/metrics.py.
    grouped = defaultdict(lambda: [0, 0])
    for r in rows:
        score = r["classification"]["fused"] if r["decision_valid"] else -1.
        grouped[score][0 if r["label"] else 1] += 1
    positive = sum(r["label"] for r in rows); negative = len(rows) - positive
    tp = fp = 0; old_tpr = old_fpr = auc = ap = 0.
    for score in sorted(grouped, reverse=True):
        p, n = grouped[score]; tp += p; fp += n
        tpr, fpr = tp / positive, fp / negative
        auc += (fpr - old_fpr) * (tpr + old_tpr) / 2
        ap += (tpr - old_tpr) * tp / (tp + fp)
        old_tpr, old_fpr = tpr, fpr
    return {"auroc": auc, "auprc": ap}


def metrics(rows, threshold, split):
    accepted = [r["decision_valid"] and r["classification"]["fused"] >= threshold for r in rows]
    tp = sum(a and r["label"] for a, r in zip(accepted, rows))
    fp = sum(a and not r["label"] for a, r in zip(accepted, rows))
    positive = sum(r["label"] for r in rows); negative = len(rows) - positive
    fn, tn = positive - tp, negative - fp
    out = {"threshold": threshold, "sample_count": len(rows), "positive_count": positive,
           "negative_count": negative, "decision_valid_count": sum(r["decision_valid"] for r in rows)}
    if not negative:
        out.update(accepted_positive_count=tp, false_negative_count=fn, positive_recall=tp/positive,
            binary_metrics_unavailable="Positive-only OOD: no binary Accuracy/Precision/F1/AUROC/AUPRC claim",
            layout_unavailable="No OOD layout GT")
        return out
    out.update(tp=tp, fp=fp, tn=tn, fn=fn, accuracy=(tp+tn)/len(rows),
        precision=tp/max(1,tp+fp), recall=tp/positive, f1=2*tp/max(1,2*tp+fp+fn), **ranking(rows))
    good = [r["label"] and r["layouts"]["full_top2_mode"]["valid"]
            and r["layouts"]["full_top2_mode"].get("translation_l2_px") is not None
            and r["layouts"]["full_top2_mode"]["translation_l2_px"] <= 20 for r in rows]
    out["layout20"] = {"raw_correct": sum(good), "raw_positive_recall": sum(good)/positive,
        "accepted_correct": sum(a and g for a,g in zip(accepted,good)),
        "classification_FN_but_layout_correct": sum(not a and g for a,g in zip(accepted,good)),
        "accepted_positive_bad_layout": sum(a and r["label"] and not g for a,r,g in zip(accepted,rows,good))}
    return out


def run():
    queue = read("scorer_diagnosis_20260919/continuation_v1/queue_status.json")
    training = read(PREFIX["C16"] + "/protocol.json")
    status = read(PREFIX["C16"] + "/status.json")
    freeze = read(PREFIX["C16"] + "/classifier_freezes/freeze.json")
    assert status["status"] == "complete" and status["classifier_epochs"] == 16
    comparisons, populations, models = [], {}, {}
    for split in ("test", "real", "ood"):
        summaries, selected = {}, {}
        for arm in ("C8", "C16"):
            root = PREFIX[arm] + "/evaluation/fixed_epoch/" + split
            summaries[arm] = read(root + "/summary.json")
            protocol = read(root + "/protocol.json")
            assert summaries[arm]["status"] == protocol["status"] == "complete"
            rows = read(root + "/pair_results.jsonl", lines=True)
            if arm == "C8": old_rows = rows
            else:
                assert [r["pair_id"] for r in old_rows] == [r["pair_id"] for r in rows]
                assert len({r["pair_id"] for r in rows}) == len(rows)
                for a,b in zip(old_rows,rows):
                    for key in ("label", "fragment_a", "fragment_b", "target_translation_rc", "review_status"):
                        assert a.get(key) == b.get(key), (split,key,a["pair_id"])
                populations[split] = {
                    "all_ordered_pair_ids_equal": True, "labels_endpoints_GT_review_status_equal": True,
                    "all_count": len(rows),
                    "exact_full_layout_objects_equal": all(a["layouts"] == b["layouts"] for a,b in zip(old_rows,rows)),
                    "ordered_pair_ids_sha256": hashlib.sha256(json.dumps([r["pair_id"] for r in rows]).encode()).hexdigest()}
            selected[arm] = [r for r in rows if split != "real" or not r["label"] or r["review_status"] == "keep"]
            identity = summaries[arm]["model"]
            models[arm] = {key: identity[key] for key in ("checkpoint_path", "checkpoint_sha256", "epoch", "selection")}
        assert len(selected["C16"]) == {"test":3000,"real":803,"ood":301}[split]
        group = "kept_plus_all_negative" if split == "real" else "all"
        populations[split].update(analyzed_group=group, analyzed_count=len(selected["C16"]),
            positive_count=sum(r["label"] for r in selected["C16"]))
        for op in ("max_f1", "recall_95"):
            old_t = summaries["C8"]["model"]["operating_points"]["thresholds"][op]
            new_t = summaries["C16"]["model"]["operating_points"]["thresholds"][op]
            for arm, threshold, rule in (("C8",old_t,"C8_SIMVAL_frozen"),
                                         ("C16",old_t,"C8_SIMVAL_frozen"),
                                         ("C16",new_t,"C16_SIMVAL_frozen")):
                measured = metrics(selected[arm], threshold, split)
                if (arm == "C8") or rule == "C16_SIMVAL_frozen":
                    official = summaries[arm]["groups"][group]["classification"]["fused"][op]
                    for key,value in official.items():
                        assert math.isclose(measured[key],value,rel_tol=1e-10,abs_tol=1e-10), (split,arm,key)
                comparisons.append({"split":split,"group":group,"checkpoint":arm,
                    "operating_point":op,"threshold_source":rule,"metrics":measured})
    history = []
    for epoch in (20,26,27,28):
        source = (PREFIX["C8"] + "/training" if epoch == 20 else PREFIX["C16"])
        v = read(source + f"/validation_{epoch:03d}.json")
        item = {"epoch":epoch, "classifier_epoch":epoch-12,
                "simval_max_f1":v["operating_points"]["validation"]["max_f1"],
                "simval_recall95":v["operating_points"]["validation"]["recall_95"]}
        if epoch == 20:
            existing = HERE.parent / "convergence/continuation_snapshot_v1/epoch_metrics.json"
            data = existing.read_bytes()
            previous = next(r for r in json.loads(data)["epochs"] if r["absolute_epoch"] == 20)
            assert math.isclose(previous["val_max_f1"], item["simval_max_f1"]["f1"], abs_tol=1e-12)
            item.update(train_samples=previous["train_samples"], train_bce=previous["train_pair_bce"],
                        train_bce_source=str(existing))
            SOURCES["existing_C8_training_metrics"] = {"local_path":str(existing),
                "sha256":hashlib.sha256(data).hexdigest(), "source_kind":"existing verified convergence receipt"}
        else:
            segments = [read(PREFIX["C16"] + f"/segment_{n:03d}.json")["training"] for n in range((epoch-1)*4+1,epoch*4+1)]
            n = sum(s["samples"] for s in segments)
            item.update(train_samples=n, train_bce=sum(s["samples"]*s["loss_components"]["fused_pair_bce"] for s in segments)/n,
                        training_seconds=sum(s["elapsed_s"] for s in segments))
        history.append(item)
    output = {"schema_version":"S6-C8-C16-fixed-endpoint-comparison/1", "status":"complete",
        "collected_at_utc":datetime.now(timezone.utc).isoformat(), "sources":SOURCES,
        "queue_snapshot":{k:queue[k] for k in ("status","active_stage","active_name","active_pid")},
        "s6_completed_stage_count":sum(s.get("status")=="complete" for s in queue["stages"] if s["name"].startswith("s6_d2")),
        "models":models,"training_status":status,"continuation_identity":training["continuation_identity"],
        "frozen_selection_epochs":{k:v["selected_epoch"] for k,v in freeze["selections"].items()},
        "populations":populations,"comparisons":comparisons,"training_and_SIMVAL_history":history,
        "thresholds_fitted_here":False,"held_out_used_to_choose_model_or_threshold":False,
        "notes":["Primary endpoints are C8 epoch20 and C16 epoch28, not REAL-selected checkpoints.",
                 "REAL is reviewed295 positive plus unchanged508 negative, total803.",
                 "AUROC and AUPRC do not depend on the reporting threshold; invalid decisions ranked below valid.",
                 "OOD has301 positive pairs only; report recall/counts, not binary precision/F1/Accuracy.",
                 "C16 auxiliary max_f1 and recall95 selected epoch28 as well; this report extracts fixed_epoch only.",
                 "No inference, training, restart, remote writes, or source-output modifications performed."]}
    (HERE / "comparison.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    lines = ["# S6-D2 C8→C16：固定终点与独立SIMVAL阈值", "",
        "原C8=epoch20；追加8轮、192K pair曝光后C16=epoch28。Matcher冻结、PairBCE不变、LR2e-5、physical/effective batch16。S6训练与9个终点评估均完成；采集时队列已进入S7。", "",
        "## max-F1工作点：旧阈值固定与新SIMVAL阈值分开", "",
        "| 数据 | 权重 / 阈值来源 | 阈值 | Accuracy | Recall | F1 | TP / FP | AUROC / AUPRC |",
        "|---|---|---:|---:|---:|---:|---|---|"]
    for r in comparisons:
        if r["operating_point"] != "max_f1": continue
        m=r["metrics"]; title=r['checkpoint']+" / "+("旧阈值" if r['threshold_source'].startswith('C8') else "新阈值")
        if r["split"]=="ood":
            lines.append(f"| OOD正例301 | {title} | {m['threshold']:.9f} | — | {m['positive_recall']:.2%} | — | {m['accepted_positive_count']} / 无负例 | — |")
        else:
            lines.append(f"| {'TEST3000' if r['split']=='test' else '保留敦煌803'} | {title} | {m['threshold']:.9f} | {m['accuracy']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} | {m['tp']} / {m['fp']} | {m['auroc']:.6f} / {m['auprc']:.6f} |")
    lines += ["", "完整精确小数、Precision、混淆计数、Recall95工作点、原始/通过分类后的layout20及来源路径见 comparison.json。", "",
        "## 收敛证据", "", "| 分类轮次 | TRAIN BCE | SIMVAL F1 | SIMVAL AUPRC | SIMVAL maxF1阈值 |", "|---|---:|---:|---:|---:|"]
    for r in history:
        v=r['simval_max_f1'];loss=f"{r['train_bce']:.6f}" if 'train_bce' in r else '未提取'
        lines.append(f"| C{r['classifier_epoch']} | {loss} | {v['f1']:.4%} | {v['auprc']:.6f} | {v['threshold']:.9f} |")
    lines += ["", "## 结论与边界", "",
        "- 追加训练改善SIMVAL/TEST，但没有同步改善真实域。即固定旧阈值，敦煌Recall仍从60.68%降至55.25%、F1从60.58%降至57.60%；OOD从65/301降至58/301。因此真实退化不只是新阈值造成。",
        "- 使用各自SIMVAL max-F1阈值时，TEST F1从94.28%升至95.28%，保留敦煌F1从60.58%降至57.54%，OOD从65/301降至60/301。敦煌AUROC/AUPRC也下降，不能只靠一个阈值变化解释。",
        "- C14→C16 TRAIN BCE继续下降，SIMVAL F1小幅上升而AUPRC约0.986附近波动；不证明完全收敛，但结果不支持‘只要同配方继续训练就能解决真实域错位’。",
        "- 两终点三域的完整layout对象逐例相同，REAL保留正例原始layout≤20为212/295。旧阈值下正确layout获准147→135，新SIMVAL阈值下147→138；变动来自分类门控，不是摆放退化。",
        "- 两终点全部pair ID顺序、标签、碎片端点、GT与人工保留状态一致。未在REAL/OOD调阈值或挑选checkpoint；本结果不触发额外训练。",
        "- raw/保存原远端JSON/JSONL副本；唯一首次下载缺失项为旧validation路径，已按实际training/路径补取，未改任何历史结果。", ""]
    (HERE / "FINDINGS.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"comparisons":len(comparisons),"populations":populations,"source_files":len(SOURCES)},ensure_ascii=False))


if __name__ == "__main__":
    run()
