# DH-Pairwise

最新实现更新（2026-09-25）：[S7-Consensus 同一摆放多组支持归并修复](docs/S7_CONSENSUS_MERGE_REPAIR.md)。修复过严的归并判定，使共同证据同时用于 Layout 精修和 Scorer；训练后收益尚待冻结评估。下方历史结果仍保留原实验口径。

敦煌／Turufan 写卷碎片的两两可拼分类与无旋转平面摆放。项目代码与研究交接快照：**2026-09-21**。

本仓库包含当前模型、训练、仿真增强、评估、消融实现及必要的历史 Python 依赖；不是整个私人实验工作目录的备份。模型权重、真实数据、仿真样本、逐例人工标注和同事未授权公开的代码包不在仓库内。

## 从哪里开始

- **[完整交接文档](docs/HANDOFF.md)**：任务、输入输出、网络、代码位置、实验结论及尚未验证的方向。
- [实验结果与口径](docs/RESULTS.md)：固定预算下的仿真、人工筛选敦煌及 Turufan 结果。
- [运行与外部文件](docs/RUNNING.md)：环境、训练入口、检查点和数据依赖。
- [代码导航](docs/CODE_MAP.md)：Matcher、Scorer、数据与评估入口。
- [D／ResGCN 等八组新实验](docs/LOCAL_EVIDENCE_V2.md)：实际网络改动、训练配方、源码与真实域校准入口。
- [第三方来源与适配限制](THIRD_PARTY_NOTICES.md)。

## 当前主线

```text
两块材料 Mask + 有序外轮廓
  → 7/16/32/64px 多尺度局部窗口 → Patch CNN → Context
  → primal/dual 互补描述子 → 带 dustbin 的 Sinkhorn 对应矩阵
       ├─ 对应位移共识 → 无旋转 xy Layout
       └─ 选中候选的局部特征 → 独立 Cross-Attention Scorer → 可拼概率
```

最新已完成：**S7 Matcher M12 冻结 + fresh Scorer C16（2 层、96D，4/8 heads 对照）**。新增 `local_evidence_v2` 包含 512 对照、128/256 对应点对上限、8 heads、两种 14 层 ResGCN、联合 D 和稳定局部证据输入，共八组。它们不更新 Matcher，也不改变原 Layout。两层是本轮实际配置，早期四层主版本计划已被后续用户安排替代。历史 E1、Full-E1、S 系列的相关实现仍保留，不能把不同训练配方混称为同一个模型。

此前相同 S7 数据、Matcher 和分类预算下，全量轮廓 token 改为最终匹配端点，人工保留敦煌集 F1 58.04%→66.67%、AUROC 0.7384→0.8646，误报 133→7；主阈值召回却为 59.32%→51.19%。这支持“局部证据选择有价值”，**并不表示召回问题已经解决**。端点指匹配 Patch 中心，不是接缝起止点。新的八组使用对应点对＋Q＋残差、batch48；与历史去重端点 batch16 不能视为仅改变一个因素的对照。

## 最小环境与验证

建议 Python 3.10/3.11；原远端训练使用 PyTorch 2.5.1/CUDA 12.4。先安装适合本机的 PyTorch，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
python -m unittest experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.test_model
```

CPU 单测使用合成张量，不需下载数据或权重。完整训练／真实数据推理需要另行提供数据、checkpoint 和冻结协议文件；不能仅凭 clone 本仓库重现所有历史数字。[详细说明](docs/RUNNING.md)

## 评估边界

- 原实验的分类模型和阈值依据仿真 VAL 冻结；新增真实域校准只改工作阈值，不重新拟合权重。按来源分五折、四折选阈值、一折独立测试，汇总 out-of-fold 指标。
- 敦煌人工筛选集是 **295 正例 + 508 负例／构造干扰对**；属于经过人工与模型结果审阅的诊断集。真实域交叉验证重建了其中 469 个跨来源负例，先分来源、后在折内配负例；不能与原负例逐条等同。
- 原 Turufan 测试只有 **301 个已知正例，且无 Layout GT**，仅能报告正例接受率。新增校准队列另构造 301 个跨来源负例，才能在该合成负例口径下计算分类指标；仍不能报告摆放正确率。负例依赖“不同来源不可拼”的假设。
- 校准 v1 不限制阈值范围；v2 按用户约束使用 0.20–0.80 网格，并另报固定 0.30。真实域已用于多轮诊断，这些分折结果不等同于重新获得未接触的盲测集；两个阈值策略不混报。
- PairingNet/ShreddingNet 是项目 mask-only 适配基线，**不是原论文原生设置的等价复现**。
- Layout 数值有效不等于真实可拼；本研究不估计旋转。

本仓库保留研究代码的原始包路径，以兼容已有导入与 checkpoint。上述 09-21 发布快照没有启动新训练；09-25 修复版的重训与迁移边界见页首更新。新增实验应创建新的输出目录，不能覆盖历史实验。
