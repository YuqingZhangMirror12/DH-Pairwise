"""Project completed C8--C11 fixed-threshold evidence; no model/threshold fit."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "epoch_metrics.json").read_text())
    records = data["epochs"]
    assert [r["classifier_epoch"] for r in records] == [8, 9, 10, 11]
    thresholds = {r["fixed_c8_r95_threshold"]["threshold"] for r in records}
    assert len(thresholds) == 1
    threshold = thresholds.pop()
    chart_rows, evidence_rows = [], []
    for row in records:
        assert (row["val_samples"], row["val_positive_count"], row["val_negative_count"]) == (3000, 1500, 1500)
        counts = row["fixed_c8_r95_threshold"]
        original = json.loads((ROOT / row["saved_probability_source"]).read_text())
        recomputed = dict(tp=0, fp=0, fn=0, tn=0)
        for sample in original:
            y = bool(sample["label"])
            pred = sample["classification"]["fused"] >= threshold
            key = "tp" if y and pred else "fn" if y else "fp" if pred else "tn"
            recomputed[key] += 1
        assert all(recomputed[k] == counts[k] for k in recomputed)
        epoch = "C" + str(row["classifier_epoch"])
        chart_rows.extend([dict(epoch=epoch, error="误报 FP", pairs=counts["fp"]),
                           dict(epoch=epoch, error="漏判 FN", pairs=counts["fn"])])
        evidence_rows.append(dict(epoch=epoch, tp=counts["tp"], fp=counts["fp"],
            fn=counts["fn"], tn=counts["tn"], recall_percent=100 * counts["recall"], threshold=threshold))
    source = dict(label="S6-D2 仿真验证续训记录", files=[dict(label="epoch_metrics.json"),
        dict(label="validation_020_rows.json"), dict(label="validation_021_rows.json"),
        dict(label="validation_022_rows.json"), dict(label="validation_023_rows.json")],
        metricDefinitions=[dict(name="误判数", definition="在同一组3000对仿真验证样本上，沿用C8高召回阈值0.4257810116，统计误报与漏判的配对数量。")],
        filters=["模型：S6-D2，冻结 Matcher", "验证集：1500正例、1500负例", "只包含已完成的C8至C11"],
        caveats=["C8到C11的端点改善不是逐轮单调改善。",
                 "单次训练的仿真验证结果，不能推出敦煌或Turufan性能提高。",
                 "C16完整预算尚未完成；未据此选择新模型或重新拟合阈值。"])
    chart = dict(schemaVersion=1, id="s6-fixed-recall-threshold-continuation", queryId="s6-fixed-threshold-errors",
        title="S6续训：固定原高召回阈值下的误判数", chart=dict(type="line", x="epoch", y="pairs", series="error",
            showXAxisLabel=True, xLabel="Scorer累计训练轮次", yLabel="误判配对数（对）", startAtZero=True),
        rows=chart_rows, source=source, theme="codex-classic", height=300)
    receipt = dict(schemaVersion=1, items=[dict(id="s6-continuation-fixed-recall", title="续训后，同等召回的误报减少",
        queries=[dict(id="s6-fixed-threshold-errors", source=source,
            reportingPeriod="同一SIMVAL上的C8、C9、C10、C11",
            columns=[dict(field=k, label=v) for k, v in [("epoch", "Scorer轮次"), ("tp", "真阳性"), ("fp", "误报"),
                ("fn", "漏判"), ("tn", "真阴性"), ("recall_percent", "召回率（%）"), ("threshold", "固定阈值")]],
            rows=evidence_rows, preview=dict(kind="aggregate", note="四个完整轮次的同一3000对验证总体汇总。", totalRows=4),
            methods=[dict(language="calculation", code="pred = score >= 0.4257810115814209; Recall = TP / 1500; FP decrease from C8 to C11 = 149 - 110 = 39 pairs.")])])])
    for name, payload in (("fixed-threshold-chart.json", chart), ("fixed-threshold-sources.json", receipt)):
        (ROOT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status="projected", completed_epochs=[8, 9, 10, 11],
        fixed_threshold=threshold, original_predictions_recounted=12000)))


if __name__ == "__main__":
    main()
