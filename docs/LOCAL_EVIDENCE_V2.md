# D、ResGCN 与局部证据 Scorer：2026-09-21 代码交接

本页对应已完成的八组 `local_evidence_v2` 实验，不是待执行设计。
训练、SIMVAL、SIMTEST、敦煌与 Turufan 评估全部完成于 2026-09-21 05:07（北京时间）。
此次发布只同步源码与文档，不重启训练，不改动服务器上的实验。

## 共用输入与训练边界

冻结 S7 M12 Matcher、original512 轮廓采样、Patch Encoder、Sinkhorn 和原 Layout 解码。
Scorer 使用预测对应点对两侧的 96D contextual features、原始 Sinkhorn `Qij`、相对预测最终位移的残差。
它不是重新读取原图的 coarse CNN，也不是将整圈 512 个点直接池化为分数。
GCN 组可先在完整有效轮廓上聚合上下文，但最终分类仍选取局部对应记录。

统一 2 层 Cross-Attention、96 维、默认 4 heads；独立从头训练 Scorer，Matcher 不反传。
PairBCE 是主分类监督；D 另加候选辅助监督。候选 GT 仅用于 TRAIN 数据采样和 loss，不输入推理网络。
最终分类是单独的 sigmoid score；生产 Layout 在这些组间完全相同。

| arm / 命令名称 | 相对于新 512 对照的改动 |
|---|---|
| `reference512_h4` | 至多 512 条匹配对应记录；2 层、4 heads，无新 GCN |
| `cap128_h4` | 按 Q 保留至多 128 条对应记录 |
| `cap256_h4` | 按 Q 保留至多 256 条对应记录 |
| `cap512_h8` | 只把注意力头数改为 8，总维度仍为 96 |
| `gcn_pairing_h4` | Scorer 分支增加 PairingNet 式 14 层残差图聚合，轮廓索引邻域 radius=8 |
| `gcn_shredding_h4` | Scorer 分支增加 ShreddingNet 适配代码的 14 层 ResGCN，有效轮廓图 radius=8 |
| `joint_D_h4` | S7 条件重采样＋候选正确性辅助 BCE，网络另加候选质量头 |
| `stable_h4` | 围绕原最终位移的软支持＋容忍间隙的残差＋局部坐标规范化 |

128/256 限制的是 **Scorer 对应记录数**，不是 Matcher 轮廓点数，也不是每侧去重后的中心数。
同一个中心可能参加多条对应记录；旧 `matched_tokens` 头则只使用去重端点。
两种 GCN 是本项目的 Scorer 聚合变体，不是原论文完整网络的复现，也不改变 Matcher context。

## 联合 D 具体做什么

从已有 S7 TRAIN24K 与冻结 Matcher 的 TRAIN 诊断中建立目标池，不生成新 mask：

- 可信正例要求原预测最终 Layout 的 TRAIN GT 位移误差 ≤20px。
- 少支持：两侧去重匹配端点数的较小值 ≤32。
- 部分接缝：`partial_curve` 样本，或较少侧匹配覆盖率 ≤0.15。
- 高残差可信正例：上述可信正例中残差位于上四分位。
- 困难负例：负 pair 中预测内点 Q 质量位于上四分位。

每轮 6,000 普通正例＋6,000 普通负例＋6,000 定向正例＋6,000 困难负例，
仍为 12K 正／12K 负；同一 pair 每轮最多重复 4 次，不足时有显式填充记录。
这是条件分布调整，不是把训练集扩为 48K，也不是改变正负 pair 标签。

候选辅助标签：正 pair 且候选误差 ≤20px 为 1；负 pair 或错误候选为 0；
缺失位姿监督的样本忽略辅助 loss。总损失为 `PairBCE + 0.5 * CandidateBCE`。
候选质量头用于辅助训练与诊断，不在推理时新增硬否决，也不替换原 Layout。

## 稳定局部证据具体做什么

保留原预测最终位移，放宽到该位移周围的原始候选记录，不限于原 10px 硬内点。
残差尺度由双方平均轮廓采样间距确定：`clip(3 * max(step_a, step_b), 10, 40)` px。
输入 `asinh(residual / scale)`，支持权重为相对 Q 除以 `1 + (residual / scale)^2`；
注意力、加权池化和最大值分支均考虑软支持，避免弱外点通过 max 绕过权重。
另加入中心化、规范化后的局部坐标；原始绝对 Q 仍保留。
此组 **没有加入 dustbin 概率硬拒绝**，也没有重训物理尺度不变的 Matcher。

## 相同预算与结果解释

每组固定 C16，S7 TRAIN24K、SIMVAL3K；实际 batch48、梯度累积1，有效 batch48，FP32。
每组 384,000 次样本曝光、8,000 次优化器更新。
AdamW 前 3 轮 `1e-4`、后 13 轮 `2e-5`，weight decay `1e-4`，梯度裁剪 5。
每轮仅做 SIMVAL；固定第16轮之后做 SIMTEST／真实域测试，不按真实域结果挑 epoch。

[最终汇总](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/FINAL_RESULTS_20260921.md)
保留原 SIMVAL 阈值结果与限制。新组之间预算一致；历史端点头为 batch16，不能称作同 batch 对照。
本轮没有跨域、跨指标全面胜出的新模型；GCN 和 D 的部分召回收益伴随误报或排序方面的代价。

## 源码位置

相对仓库根的实验包：`experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/`。

| 文件 | 职责 |
|---|---|
| [local_evidence_v2/model.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/model.py) | 八组注册、局部输入、两种 GCN、CA、D 辅助头、稳定输入 |
| [local_evidence_v2/data.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/data.py) | D 条件样本池、逐轮重采样与候选标签 |
| [local_evidence_v2/train.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/train.py) | 固定预算、Pair/候选 loss、恢复与冻结阈值 |
| [local_evidence_v2/evaluate.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/evaluate.py) | 固定 C16 权重加载、原 Layout 不变的推理适配 |
| [local_evidence_v2/queue.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/queue.py) | 历史四卡有限队列；每组训练＋测试后才接下一组 |
| [local_evidence_v2/test_model.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/test_model.py) | 合成张量上的前向／梯度、padding、空证据与输入限制测试 |

运行方式与外部数据要求见 [RUNNING.md](RUNNING.md)。不公开模型权重、缓存、逐例人工标注、真实样本或服务器凭据。

## 真实域分组校准：与训练实验分开

- [real_domain_calibration_v1/PROTOCOL.md](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/real_domain_calibration_v1/PROTOCOL.md)：八组固定模型、五折来源分组；四折选阈值、一折测，重复五次。此版允许较低阈值。
- [bounded_real_calibration_v2/PROTOCOL.md](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/bounded_real_calibration_v2/PROTOCOL.md)：相同折与构造负例，扩展历史模型；使用 0.20–0.80、步长 0.01 的阈值网格，并另报固定 0.30。

两版均不训练权重，也不把“各折阈值中位数”重新作用于全量数据后当作独立测试成绩。
指标来自合并的独立测试折；中位阈值只是摘要。
敦煌校准口径为 295 正＋39 严格负＋469 重新构造的跨来源负；Turufan 为 301 正＋301 构造负。
先按写本来源及已知重复／别名分折，再在折内构造负例；不得先随意组合负例后拆 pair。
来源未知的同源关系无法完全排除，且此前已经看过真实域诊断，所以不能称作全新盲测。
输出 manifests、registry 与逐例预测是外部研究文件，不随源码发布。
