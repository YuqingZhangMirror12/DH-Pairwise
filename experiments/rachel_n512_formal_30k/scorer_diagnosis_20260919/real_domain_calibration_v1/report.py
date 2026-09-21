"""Write inspectable Markdown and future-only threshold artifacts, no HTML."""
import argparse
from pathlib import Path
from .prepare import ARMS, read, save

LABELS={"reference512_h4":"512／4 heads对照","cap128_h4":"128候选点对",
    "cap256_h4":"256候选点对","cap512_h8":"512／8 heads",
    "gcn_pairing_h4":"PairingNet式GCN","gcn_shredding_h4":"ShreddingNet式GCN",
    "joint_D_h4":"联合D","stable_h4":"稳定输入"}


def pct(x):return "—" if x is None else f"{100*x:.2f}%"


def threshold(x):return f"{x:.4g}" if abs(x)<0.001 else f"{x:.4f}"


def write(root):
    root=Path(root);source=read(root/"results.json")
    assert source["status"]=="complete"
    lines=["# 真实域五折阈值校准：冻结模型，独立折测试", "",
        "## 口径", "",
        "本轮只校准阈值，没有更新Matcher、Scorer或融合权重，也没有挑选新检查点。",
        "每个数据集按已知来源分成5折：4折校准阈值，剩余1折测试，轮换全部5折。",
        "以下为所有未参与阈值校准的测试预测合并后的指标，每一对只计数一次。",
        "每轮的校准／测试之间，来源组和碎片ID交集均为0。", "",
        "敦煌保留295个人工保留正例、39个原有同来源负例；另在折内重建469个跨来源负例。",
        "Turufan保留301个正例，按用户定义构造301个跨来源负例。负例不是逐对人工核验的GT。",
        "两域先分来源、再在折内采负例；同一写本正反面以及现有重复mask记录合并来源组。",
        "原有单碎片预处理不变；不同来源本身没有共同物理尺度，结论适用于该构造基准。",
        "因为负例集合变了，旧SIM阈值也在同一新集合重新计算，不能直接拿上轮F1作增益基线。",
        "下列GCN两组是我们Scorer使用相应GCN聚合的变体，不是PairingNet／ShreddingNet原模型。", "",
        "## 关键发现", "",
        "- 敦煌存在明显阈值迁移问题：512／4 heads对照的F1由60.19%升至76.78%，召回由43.05%升至67.80%；原本正确且通过分类的Layout由126增至188（共216个正确Layout）。模型和Layout均未改变。",
        "- 真实域校准后，敦煌多数模型F1接近75%–77%。8 heads的77.25%仅比对照高0.47个百分点，不能凭这一次五折认定结构优势。",
        "- Turufan并非普遍改善：512／4 heads对照虽然召回升高，但Accuracy从70.27%降至64.29%，新增134个误报。两组GCN的独立折结果更均衡：Accuracy74.92%、Precision82.89%、Recall62.79%、F1 71.46%。这只是该构造负例基准上的结果。",
        "- Turufan正负1:1，全部判可拼也有F1 66.67%。所以部分模型校准后F1约66%–67%，不应解读为已解决分类。",
        "- 强求校准召回95%会带来大量误报：对照在敦煌误报345/508，在Turufan误报263/301。阈值能补救一部分漏判，但不能替代正负样本分离能力。",
        "- 因此当前支持真实域阈值校准，但不支持仅凭阈值高就判定仿真过拟合；也不建议不设误报／Precision限制地强推95%召回。", "",
        "## 主策略：校准折最大F1", "",
        "阈值显示中位数及5折范围。最大F1完全在校准折计算，未使用测试折标签挑阈值。", ""]
    future={}
    for split,title in (("real","敦煌"),("ood","Turufan")):
        data=source["results"][split];future[split]={}
        lines += [f"### {title}：{data['positive']}正＋{data['negative']}负", "",
            "| 模型 | 旧SIM阈值 | 旧阈值Accuracy | 旧Recall | 旧F1 | CV阈值中位数[范围] | CV Accuracy | CV Precision | CV Recall | CV F1 | CV误报 | AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            model=data["models"][arm];base=model["baseline_on_same_new_negative_population"]["simval_max_f1"]
            cv=model["cv"]["max_f1"];m=cv["pooled_out_of_fold"]
            lines.append(f"| {LABELS[arm]} | {threshold(base['threshold'])} | {pct(base['accuracy'])} | {pct(base['recall'])} | {pct(base['f1'])} | {threshold(cv['threshold_median'])} [{threshold(cv['threshold_min'])}, {threshold(cv['threshold_max'])}] | {pct(m['accuracy'])} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {m['fp']} | {m['auroc']:.4f} |")
            future[split][arm]=dict(checkpoint_sha256=model["checkpoint_sha256"],
                max_f1=model["cv"]["max_f1"]["all_data_refit_for_future_only"],
                recall_95=model["cv"]["recall_95"]["all_data_refit_for_future_only"])
        lines += ["", "AUROC使用同一份原始分数计算，调阈值不会改变AUROC；表中的CV改善是决策改善，不是重新学习后的排序改善。", ""]
    lines += ["## 召回优先策略：校准折召回至少95%", "",
        "选择满足条件的最高阈值。95%是校准折目标，不保证独立测试折也达到95%。", ""]
    for split,title in (("real","敦煌"),("ood","Turufan")):
        lines += [f"### {title}", "",
            "| 模型 | 阈值中位数[范围] | CV Accuracy | CV Precision | CV Recall | CV F1 | TP | FP | FN | TN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            cv=source["results"][split]["models"][arm]["cv"]["recall_95"];m=cv["pooled_out_of_fold"]
            lines.append(f"| {LABELS[arm]} | {threshold(cv['threshold_median'])} [{threshold(cv['threshold_min'])}, {threshold(cv['threshold_max'])}] | {pct(m['accuracy'])} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |")
        lines += [""]
    lines += ["## 敦煌正确Layout是否被救回", "",
        "原始Layout没有重算或改变；共有216/295个正例在20px误差内。", "",
        "| 模型 | 旧SIM阈值放行正确Layout/216 | CV最大F1放行/216 | CV召回95放行/216 |",
        "|---|---:|---:|---:|"]
    for arm in ARMS:
        model=source["results"]["real"]["models"][arm]
        vals=[model["baseline_on_same_new_negative_population"]["simval_max_f1"]["layout_correct_accepted"],
            model["cv"]["max_f1"]["pooled_out_of_fold"]["layout_correct_accepted"],
            model["cv"]["recall_95"]["pooled_out_of_fold"]["layout_correct_accepted"]]
        lines.append(f"| {LABELS[arm]} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines += ["", "Turufan没有Layout GT，本轮新增负例不改变这一限制。", "",
        "## 分折记录", ""]
    for split,title in (("real","敦煌"),("ood","Turufan")):
        data=source["results"][split]
        lines += [f"### {title}：{data['source_group_count']}个来源组", "",
            "| 测试折 | 校准样本 | 测试样本 | 校准正例 | 测试正例 | 共享来源 | 共享碎片 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for f in data["fold_checks"]:
            lines.append(f"| {f['fold']+1} | {f['calibration_count']} | {f['test_count']} | {f['calibration_positive']} | {f['test_positive']} | {f['shared_sources']} | {f['shared_fragments']} |")
        lines += [""]
    lines += ["## 解释边界与可交接文件", "",
        "- 阈值接近0.9本身不是过拟合证明。阈值交叉验证改善F1/召回，支持存在真实域操作点不匹配；模型排序、困难正负样本覆盖问题仍需看AUROC和误报。",
        "- 这些真实数据已参与前期人工筛选、失败分析和架构探索。本轮避免新的阈值拟合泄漏，但不能声称恢复了完全未见过的盲测集。",
        "- 一次完整五折；不是五个互相独立的数据集，也没有将折间波动当作独立样本置信区间。",
        "- 跨来源负例可能比同来源困难非邻接更容易，也可能包含未被来源元数据识别的关联；不能推广为任意真实检索流量上的性能。",
        "- 若根据这些CV指标继续选网络，再报告该网络泛化，需要额外外层选择/新留出测试；本轮没有自动部署或选择模型。",
        "- `thresholds_refit_for_future.json`是完成CV后用全部标注数据拟合的未来使用阈值，不能拿其在当前全量数据上的分数当独立测试结果。原SIMVAL冻结文件未改。",
        "- 精确指标与每折阈值在`results.json`；逐例独立折预测在`oof/<split>/<arm>.json`；来源与负例清单在`<split>/manifest.json`。",
        "- 源模型与旧阈值在`model_freezes.json`；新负例预测在`predictions/`；推理任务记录在`inference_queue.json`。", "",
        "方法参考：[scikit-learn阈值调优](https://scikit-learn.org/stable/modules/classification_threshold.html)，",
        "[按组交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)。", ""]
    (root/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    save(root/"thresholds_refit_for_future.json",dict(status="future_use_only_not_deployed",fitted_on_all_real_labels_after_cv=True,
        evaluation_rule="Do not use full-data-fit predictions as independent test metrics; use the reported OOF metrics.",thresholds=future))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();write(a.root)
