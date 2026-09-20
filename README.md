# DH-Pairwise

敦煌／Turufan 写卷碎片的两两可拼分类与无旋转平面摆放。项目代码与研究交接快照：**2026-09-21**。

本仓库包含当前模型、训练、仿真增强、评估、消融实现及必要的历史 Python 依赖；不是整个私人实验工作目录的备份。模型权重、真实数据、仿真样本、逐例人工标注和同事未授权公开的代码包不在仓库内。

## 从哪里开始

- **[完整交接文档](docs/HANDOFF.md)**：任务、输入输出、网络、代码位置、实验结论及尚未验证的方向。
- [实验结果与口径](docs/RESULTS.md)：固定预算下的仿真、人工筛选敦煌及 Turufan 结果。
- [运行与外部文件](docs/RUNNING.md)：环境、训练入口、检查点和数据依赖。
- [代码导航](docs/CODE_MAP.md)：Matcher、Scorer、数据与评估入口。
- [第三方来源与适配限制](THIRD_PARTY_NOTICES.md)。

## 当前主线

```text
两块材料 Mask + 有序外轮廓
  → 7/16/32/64px 多尺度局部窗口 → Patch CNN → Context
  → primal/dual 互补描述子 → 带 dustbin 的 Sinkhorn 对应矩阵
       ├─ 对应位移共识 → 无旋转 xy Layout
       └─ 选中候选的局部特征 → 独立 Cross-Attention Scorer → 可拼概率
```

当前已完成的局部分类对照：**S7 Matcher M12 + fresh Scorer C16（2 层、96D、4 heads）**。后续 4 层与 4/8 heads 是研究计划，不是本快照中已验证的最佳配置。历史 E1、Full-E1、S 系列的相关实现仍保留，不能把不同训练配方混称为同一个模型。

相同 S7 数据、Matcher 和分类预算下，全量轮廓 token 改为最终匹配端点，人工保留敦煌集 F1 58.04%→66.67%、AUROC 0.7384→0.8646，误报 133→7；主阈值召回却为 59.32%→51.19%。这支持“局部证据选择有价值”，**并不表示召回问题已经解决**。端点指匹配 Patch 中心，不是接缝起止点。

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

- 分类模型和阈值依据仿真 VAL 冻结；不在同一真实测试集上反复选阈值再声称无偏测试表现。
- 敦煌人工筛选集是 **295 正例 + 508 负例／构造干扰对**；属于经过人工与模型结果审阅的诊断集。
- Turufan 是 **301 个已知正例，且无 Layout GT**；只能报告正例接受率，不能由此得出完整二分类 Accuracy/F1/AUROC 或摆放成功率。
- PairingNet/ShreddingNet 是项目 mask-only 适配基线，**不是原论文原生设置的等价复现**。
- Layout 数值有效不等于真实可拼；本研究不估计旋转。

本仓库保留研究代码的原始包路径，以兼容已有导入与 checkpoint。没有启动新训练，也没有重写现有服务器实验。新增实验应创建新的输出目录。

