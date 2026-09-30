"""Build a compact technical report from completed, same-run layout analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ORIGINAL = "full_original"
SHRED = "full_with_shred_matching_layout"


def build(root):
    analysis = json.loads((root / "comparison.json").read_text())
    chosen = analysis["selected_full_decoder"]
    names = {ORIGINAL: "原 Full", chosen: "Full + 新几何解算", SHRED: "Full + Shred 精匹配"}
    title = "Pairwise Matching and Translation Layout"
    datasets, charts, tables, blocks = {}, [], [], []
    sources = [dict(id="new", label="本轮冻结预测及配对统计", path="reports/pairwise_layout_v2_20260906/comparison.json",
        query=dict(description="逐对预测重算配对指标、平移误差，以及按稿件/真实案例分组的20,000次配对bootstrap。",
        tables_used=["validation_3000/pair_results.jsonl", "test_3000/pair_results.jsonl", "real_1016/pair_results.jsonl"],
        metric_definitions=dict(recall="GT正例中有效且平移L2误差不超过指定像素的比例；无效位姿计失败。",
            assembly_f1="2*配对且位移正确的数量/(预测配对数量+GT正例数量)。"))),
        dict(id="historical", label="历史 Exact-6 同数据适配 benchmark", path="reports/pairwise_exact6_20260906/final_summary.json")]
    def md(identifier, heading, body, source=None):
        block = dict(id=identifier, type="markdown", body=heading+"\n\n"+body if body else heading)
        if source:
            block["sourceId"] = source
        blocks.append(block)
    def pct(value):
        return "{:.2f}%".format(100*value)
    def delta_text(population, comparator):
        item = analysis["populations"][population]["bootstrap"]["paired_deltas"][chosen+"_minus_"+comparator]["r_at_10"]
        return "{:+.2f} 个百分点（95% CI {:+.2f} 至 {:+.2f}）".format(item["point_estimate"]*100, *[v*100 for v in item["percentile_95_ci"]])
    test, real = (analysis["populations"][k] for k in ("test", "real"))
    md("title", "# "+title, "")
    md("summary", "## 保留配对网络，独立解算相对位置", 
       "已实现两种零旋转 layout：从我们自己的 Sinkhorn 对应矩阵做稳健平移解算；或固定接入 ShreddingNet 的精匹配模块。两者均保留原 coarse、Sliding Window、Sinkhorn 和融合配对分数，没有按分数切换模型。\n\n"
       "新 Full 解算的 R@10：合成测试 **{} → {}**，真实正例 **{} → {}**。本轮是已训练权重上的几何模块实验，不是新增训练。".format(
           pct(test["layout"][ORIGINAL]["recall"]["10"]),pct(test["layout"][chosen]["recall"]["10"]),
           pct(real["layout"][ORIGINAL]["recall"]["10"]),pct(real["layout"][chosen]["recall"]["10"])), "new")
    md("scope", "## 配对、摆放和联合成功分别计量", 
       "验证与合成测试各 3,000 对、1,500 个正例。真实配对评估为 balanced1016（508 正例）；真实联合摆放评估为 strict547（508 正例、39 个真实负例），不把补造负例当作真实案例。\n\n"
       "R@k 是全部 GT 正例中平移误差 ≤k 像素的比例，与配对阈值无关；Assembly F1 同时要求配对正确和摆放正确。无效位姿计失败，中位误差仅在有效位姿上统计。AUROC/AUPRC 衡量配对排序，F1/Recall 依赖阈值。", "new")
    branch_rows, layout_rows = [], []
    for kind, label in (("test", "合成测试"), ("real", "真实数据")):
        section = analysis["populations"][kind]
        for method, display in names.items():
            metric = section["layout"][method]
            layout_rows.append(dict(population=label, method=display, r2=metric["recall"]["2"], r10=metric["recall"]["10"],
                assembly_f1=metric["assembly"]["10"]["f1"], median_px=metric["median_px_conditional"],
                origin="本轮同次预测", positive_count=section["positive_count"]))
        for method, historical in analysis["historical_exact6"][kind].items():
            historic_classification = historical["classification"]["row"] if kind == "test" else historical["classification"]["native"]["row"]
            branch_rows.append(dict(population=label, branch=("PairingNet" if method.startswith("pairing") else "ShreddingNet")+"（历史）",
                **{k:historic_classification.get(k) for k in ("auroc","auprc","accuracy","precision","recall","f1","threshold")}))
            layout_rows.append(dict(population=label, method="PairingNet" if method.startswith("pairing") else "ShreddingNet",
                r2=historical["recall"]["2"], r10=historical["recall"]["10"], assembly_f1=historical["assembly_f1_at_10"],
                median_px=historical["median_px_conditional"], origin="历史同数据适配", positive_count=section["positive_count"]))
        chart_rows=[]
        for method, display in names.items():
            metric=section["layout"][method]
            for tolerance in (2,5,8,10):
                chart_rows.append(dict(tolerance="R@{}".format(tolerance), tolerance_px=tolerance, method=display,
                    recall=metric["recall"][str(tolerance)], positive_count=section["positive_count"],
                    correct_count=round(metric["recall"][str(tolerance)]*section["positive_count"]),
                    assembly_f1=metric["assembly"][str(tolerance)]["f1"], population=label))
        datasets[kind+"_recall"] = chart_rows
        md(kind+"_finding", "## "+label+"：比较相同配对输出下的摆放", 
           "新 Full 相对原 Full 的 R@10 变化为 **{}**。Shred 精匹配的 R@2 为 {}，新 Full 为 {}；两种几何方案的精细位置准确性仍需分开看，不能只看 10 像素成功率。图中每组共享相同的正样本分母。".format(
               delta_text(kind, ORIGINAL), pct(section["layout"][SHRED]["recall"]["2"]),pct(section["layout"][chosen]["recall"]["2"])), "new")
        chart_id=kind+"_layout_recall"
        charts.append(dict(id=chart_id,title=label+"的平移成功率",dataset=kind+"_recall",type="bar", sourceId="new",
            encodings=dict(x=dict(field="tolerance"),y=dict(field="recall"),color=dict(field="method")),
            options=dict(grouping="grouped",valueFormat="percent")))
        blocks.append(dict(id=kind+"_chart",type="chart",chartId=chart_id))
        for branch, metric in section["classification"].items():
            branch_rows.append(dict(population=label,branch=branch,**{k:metric[k] for k in ("auroc","auprc","accuracy","precision","recall","f1","threshold")}))
    md("benchmarks", "## 与两个 benchmark 比较时保留实验边界", 
       "下面同时列出历史 PairingNet/ShreddingNet 同数据、mask-only、N512、已知朝向适配结果。它们不是原论文完整复现，也不参与本轮配对 bootstrap。本轮真实 Full 的实际计算精度为 FP32，与历史真实评估一致；原 Full 与新解算共享同一次前向输出，改善量使用这个同次对照。")
    datasets["layout_comparison"]=layout_rows
    tables.append(dict(id="layout_table",title="平移与联合摆放",dataset="layout_comparison",sourceId="new",
        columns=[dict(field=f,label=l) for f,l in (("population","数据"),("method","方法"),("r2","R@2"),("r10","R@10"),("assembly_f1","Assembly F1@10"),("median_px","中位误差 px"),("origin","结果来源"))],
        defaultSort=dict(field="population",direction="asc")))
    # The table joins new and historical sources; make both explicit.
    tables[-1]["source"] = dict(label="本轮与历史点估计",query=dict(description="新方法使用comparison.json；benchmark使用Exact-6历史点估计，不混合作配对区间。",tables_used=["reports/pairwise_layout_v2_20260906/comparison.json","reports/pairwise_exact6_20260906/final_summary.json"]))
    tables[-1].pop("sourceId")
    blocks.append(dict(id="layout_table_block",type="table",tableId="layout_table"))
    md("pairing", "## 整体 coarse 与局部 Sliding Window 都保留", 
       "下表把三条配对输出分开。coarse/local 使用各自在验证集拟合的 F1 阈值，fused 沿用原保守冻结阈值，因此不能仅凭各行 F1 就判断融合是否有用。比较排序能力应看 AUROC/AUPRC；全部分支均使用各自验证 F1 阈值的敏感性表保存在配套统计中。新几何模块不会反馈修改这些分数。历史摘要未提供的 Accuracy/Precision/阈值留空，不反推填补。")
    datasets["classification"]=branch_rows
    tables.append(dict(id="classification_table",title="配对分支指标",dataset="classification",sourceId="new",
        columns=[dict(field=f,label=l) for f,l in (("population","数据"),("branch","分支"),("auroc","AUROC"),("auprc","AUPRC"),("accuracy","Accuracy"),("precision","Precision"),("recall","Recall"),("f1","F1"),("threshold","阈值"))],
        defaultSort=dict(field="population",direction="asc")))
    blocks.append(dict(id="classification_block",type="table",tableId="classification_table"))
    md("method", "## 借用稳健配准思想，不把两个模型的分数混合", 
       "第一版并非独立 MLP 直接预测 dx/dy：它将所有候选位移加权平均，再做两轮 Cauchy 修正，位移损失则训练对应关系。多峰对应可能把均值拉离正确位移。\n\n"
       "第二版从同一个 Sinkhorn 矩阵取双向 Top-2 候选、最多 512 个，寻找 10 像素内支持最强的位移，再做内点加权精修，至少要求 3 个内点。配置只由完整验证集选择。原分类头仍使用旧 dispersion，以隔离几何变更。\n\n"
       "PairingNet/ShreddingNet 官方算法把描述子对应与 RANSAC 几何配准分开。这里固定旋转为零：移植 Shred 精匹配对照只运行 matcher 和稳健平移求解，不运行 Shred coarse/分类器。它对全部输入使用固定模块，不做置信度路由。\n\n"
       "坐标约定：`point_b = point_a + t_a_to_b_rc`；把 A 固定后，B 的实际摆放偏移为 `-t_a_to_b_rc`。这解决两两相对位置，不等同于已完成多碎片全局装配。")
    md("limits", "## 提升已经测到，但真实泛化仍有边界", 
       "置信区间来自 20,000 次配对重采样：合成集按 manuscript lineage，真实 strict547 按完整 case。所有无效位姿计失败；这些测试集以前已被查看，本轮不是全新盲测。\n\n"
       "本轮证明了已有对应关系可支持比原稠密均值更高的布局成功率，不证明两个 benchmark 在所有情形都更差，也不证明真实碎片已经达到合成数据的准确率。真实新 Full 虽然中位误差从 173.26 降到 32.00 像素，但 P90 从 412.14 升到 540.79 像素；选错主峰时仍可能严重错位。两种 layout 的差异、错误主峰和配对漏检应分别处理。", "new")
    md("next", "## 下一版以准确性为主线", 
       "已经提供可调用的 Full + 稳健几何接口，以及 Full + Shred fine 固定组合接口。建议保留两种位姿输出作准确性对照，不再以吞吐选择路径。后续训练若继续开展，重点应是独立真实验证数据上的 coarse/local 融合、跨域对应点质量，以及用位移主峰与次峰的支持差异识别不可靠布局；本轮不使用真实测试集拟合拒绝阈值。\n\n"
       "尚需回答：实际摆放需要 2 像素还是 10 像素容差？多块全局装配时，怎样利用两两位姿的内点支持和闭环一致性排除错误边？这些不影响本轮两两位姿接口的使用。")
    tables[-1]["source"] = tables[0]["source"]
    tables[-1].pop("sourceId")
    for table in tables:
        for column in table["columns"]:
            if column["field"] in ("r2","r10","assembly_f1","accuracy","precision","recall","f1"):
                column["format"] = "percent"
    return dict(surface="report",manifest=dict(version=1,surface="report",title=title,blocks=blocks,charts=charts,tables=tables,sources=sources),
                snapshot=dict(version=1,status="ready",datasets=datasets),sources=sources)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    args=parser.parse_args()
    artifact=build(args.results)
    with (args.results/"artifact.json").open("w",encoding="utf-8") as stream:
        json.dump(artifact,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write("\n")
