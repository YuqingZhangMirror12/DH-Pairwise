"""Project reviewed threshold-transfer results for the shared inline renderer."""
import inspect
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k.calibrate_dunhuang_transfer_20260916 import fit_thresholds, metrics

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "dunhuang_threshold_transfer_20260916"
result = json.loads((OUTPUT / "results.json").read_text())
names = {"s4_depth1": "S4 单层", "s6_depth2": "S6 双层", "s6_depth4": "S6 四层", "s7_augmented_full24": "S7 增强"}
policies = {"simval_max_f1": "原仿真阈值", "dunhuang_max_f1": "敦煌最大 F1", "dunhuang_recall95_max_precision": "敦煌召回≥95%"}
rows, preview = [], []
for model in result["models"]:
    for rule, op in model["operating_points"].items():
        r, o = op["real_calibration"], op["ood_positive_only"]
        row = dict(model=names[model["name"]], rule=policies[rule], recall_pct=100 * o["recall"])
        rows.append(row)
        preview.append(dict(**row, threshold=r["threshold"], turufan_tp=o["tp"], turufan_n=o["n"],
            dunhuang_precision_pct=100*r["precision"], dunhuang_recall_pct=100*r["recall"],
            dunhuang_f1_pct=100*r["f1"], dunhuang_fp=r["fp"], dunhuang_negative_n=508))

source = dict(label="固定 epoch20 的逐对预测与敦煌阈值校准",
    files=[{"label": "REAL_CALIBRATION_SCORE_INPUTS_20260916.json"}, {"label": "threshold_freeze.json"}, {"label": "results.json"}],
    filters=["四个模型均固定 epoch20", "敦煌正例仅保留人工 keep 的295对", "敦煌保留全部508对负例", "Turufan保留全部301对正例"],
    metricDefinitions=[{"label": "Turufan Recall", "definition": "Turufan召回率是301组真实可拼对中，分数达到该模型固定阈值的比例。"}],
    caveats=["敦煌被用于阈值校准，其指标为拟合内结果，不再是独立测试指标。",
        "Turufan只有正例，不能判断误报或完整分类准确性，也没有布局GT。该数据集此前已被观察，并非全新封存测试集。",
        "召回≥95%规则在敦煌全部满足条件的阈值中选择Precision最高者；不保证Turufan同样达到95%。",
        "降低阈值同时增加敦煌误报。S6双层：原阈值117/508误报，最大F1阈值196/508，95%召回阈值368/508。"],
    evidenceFlow=[{"kind": "calculation", "title": "只拟合阈值", "detail": "仅使用敦煌803对枚举观测分数阈值；同分样本一起处理。先写入冻结阈值，再原样应用于Turufan。未修改模型、布局、标签和原始评估。"},
        {"kind": "validation", "title": "原始指标复现", "detail": "四个模型的原阈值敦煌混淆矩阵和Turufan通过数均与已完成的评估记录一致；四个模型的样本ID及标签一致，803/301条分数均有效。"}])
chart = dict(schemaVersion=1, id="threshold-transfer", queryId="threshold-transfer",
    title="同一模型换阈值后，Turufan 召回如何变化",
    chart=dict(type="bar", x="model", y="recall_pct", series="rule", xLabel="模型", yLabel="Turufan Recall (%)", startAtZero=True, showValues=True),
    rows=rows, source=source, generatedAt=result["created_at"], height=390, theme="codex-classic")
receipt = dict(schemaVersion=1, items=[dict(id="threshold-transfer", title="敦煌选阈值后原样迁移到Turufan", queries=[dict(
    id="threshold-transfer", source=source, rows=preview, columns=list(preview[0]), capturedAt=result["created_at"],
    preview={"kind": "aggregate", "note": "4个固定模型 × 3种阈值规则，完整12个汇总结果。", "totalRows": 12},
    methods=[{"language": "python", "code": inspect.getsource(metrics) + "\n" + inspect.getsource(fit_thresholds)}])])])
for name, payload in (("chart-input.json", chart), ("sources-input.json", receipt)):
    (OUTPUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print("Projected 12 reviewed model/threshold results; no case identifiers or connection details.")
