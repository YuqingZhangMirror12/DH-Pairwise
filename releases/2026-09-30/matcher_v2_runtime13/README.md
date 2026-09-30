# Matcher V2：B3 正式训练源码及配套评价代码

这是 2026-09-30 的 **source13 代码快照**，补充而不覆盖先前的
[双卡可运行性验证版](../matcher_v2/README.md)。正式模型仍在训练，
本包不代表 B3 已完成、收敛，或已证明分类准确率提高。

## 当前实验与本包范围

截至最近一次全体实查（2026-09-30 18:29:47，北京时间），B3 在 GPU0/5
正式训练至 11,425 / 31,667 更新、365,600 次样本曝光。它不是 12 步短测。
这条时间戳是发布依据，不是实时监控。两个新轻量 Scorer 必须等 B3 Matcher
训练及必需终评完成后，再使用本分支的 SIM 选定 Matcher，各自从新头开始训练。
后续 B1/B2、两头及完整对比结果仍待完成。

| 目录或入口 | 内容 |
|---|---|
| [runtime_work_13/](runtime_work_13/) | 实际 B3 网络、训练、数据准入、T16、两种轻头、冻结终评代码 |
| [diagnostics_source_03/](diagnostics_source_03/) | 原始 Q、增益前上下文相似度、D17 脊形诊断，J/R/曲线分层 |
| [cal_budget_source_01/](cal_budget_source_01/) | 原生 Matcher Q-sum/Q×arc 的 CAL 负例误报预算复算 |
| [scorer_cal_budget_source_03/](scorer_cal_budget_source_03/) | 两种 Scorer 的原阈值与 CAL-only 1%/2%/5% 预算、独立计数审计 |
| [reference_select_source_01/](reference_select_source_01/) | 未重采样的外部 v4.2 SELECT 准入与原生 Matcher 评价 |
| [reference_scorer_source_01/](reference_scorer_source_01/) | 同一外部 SELECT 上的完整轻头评价及实际 Q 证据 |
| [control_queue_source_01/](control_queue_source_01/) | B1/B2 在已验证释放的资源上接续 Matcher、各自两个新头 |
| [supplemental_queue_source_01/](supplemental_queue_source_01/) | B1–B3 必需终评完成后的 15 项独立 CPU 伴随评价 |
| [continue_b3_heads.py](continue_b3_heads.py) | 已部署 B3 两头接续器；服务器专用，不是通用快速启动入口 |

源码沿用原包名：`experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919`。
网络在该包的 `matcher_v2_v1/network.py`、`adapter.py`；正式流程为
`compile_execution.py` → `launcher.py` / `pipeline.py` → `evaluation_controller.py`。

所有纳入的生产 Python 与 source13 逐文件字节一致；659 个原始基线文件保持不变。
所有伴随脚本也与对应已测试、部署版本逐字节一致。
唯一测试改写沿用早先已发布的合成 fixture：移除对私有案例清单的依赖，
不改生产逻辑。公开包省略私有资产，因此其整体 binding **不等于**远端研究源 binding。
详见 [packaging.json](packaging.json) 和 [code_binding.json](code_binding.json)。

## 模型与预算

- 保留原 CNN、landmark 上下文、一个 FP32 Sinkhorn 和 T16 精确并集。
- 新增全轮廓 self-attention、环形弧长 RoPE、同步双向点级 cross-attention，
  concat 残差、四尺度晚融合，以及初值 1/上限 5 的可学习增益。
- v2 默认关闭；主实验启用明确的序列化配置。抗滑动损失和 matchability 不加入主实验。
- Patch/Context 与 Q/几何轻头冻结各分支自己的选定 Matcher；不是沿用 E32 或旧头。

| 分支 | Matcher | 每个模块实际更新预算 |
|---|---|---:|
| B0 | 原始 | 24,000，使用已完成基线 |
| B1 | 原始 + 新直缝数据 | 31,667 |
| B2 | V2 + 原数据 | 24,000 |
| B3 | V2 + 新直缝数据 | 31,667 |

B1/B3 保留原 768,000 次曝光及原有顺序，再加 7,667 次优化更新、245,344 次直缝曝光。
新增批次从训练起点插入；有效 batch32，每批 16 正/16 负。
原 LR 时钟按原始更新前进，恢复位置按全部实际更新计数。
B1/B3 的曝光序列相同，B2 与 B0 的原计划相同。
因此 B1/B0、B3/B2 同时改变数据与训练量，不能称为纯数据单因素对比。
B3 已验证双卡 micro16/累积1/FP32；Scorer 计划各单卡 micro32/累积1。
不因发布代码而改变任何正在训练的计划、优化器或权重。

## 数据和评价边界

严格准入数据为 TRAIN6,000 / SELECT900 / TEST900。TRAIN M/J/R 为
1,200/2,400/2,400，SELECT、TEST 各为 200/400/300；各类正负各半。
直接素材为已审计的仿真碎片和程序条带，不用真实敦煌/Turufan 评估碎片增强。
几何超限样本被定向重采样，损伤段排除对应监督，原始拼接对身份去重。
既有家族、文件、解码像素与裁剪归一化像素隔离已检查；
完整写卷级跨库别名与未知来源关系仍不能宣称全部证明。

真实开发仅使用敦煌 fold1 CAL、fold2/3/4 SELECT，fold0 不选模型或阈值。
真实样本不反传；Turufan 无 Layout GT，Layout/Joint 必须为 null。
CAL 负例目标误报率不保证 SELECT 的实际误报率，两者分别报告。
阈值复算不改变赢家、Q、候选、Layout 或网络权重。
外部原版 v4.2 SELECT 与新 SELECT 有来源池重叠，单独列出，不能合成独立的 1,800 例成绩。

## 离线验证

依赖参考仓库根 `requirements.txt`。验证环境使用 Python3.11 / PyTorch2.2.2 CPU；
研究服务器使用 PyTorch2.5.1 / CUDA12.4。先安装对应环境依赖，然后在本目录运行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  python verify_public.py --output-new /your/new/outside-directory/public-cpu-check
```

该入口仅运行合成 CPU fixture，不启动远端队列，不打开实际样本或训练权重，
也不连接 SSH。独立进程检查运行时 194 项和伴随代码 183 项，合计 377 项；
任何失败、跳过、源码变化或意外 CUDA 使用都拒绝通过。
本次实际公开包结果见 [public_verification.json](public_verification.json)。

生产数据生成另需用户提供的三份外部 v4.2 生成器源码及仿真源；
完整训练还需私有数据、来源划分、曝光账本、几何协议、案例计划和新编译的执行配置。
这些文件、模型权重、原始图片、逐例结果、人工标注以及第三方私有 vendor 代码均未打包。
因此本包提供代码与离线验证复现，不承诺仅靠公开包重现私有数据上的所有历史数字。

**不要直接运行 `dispatch*` 或 `continue_b3_heads.py`。** 它们记录本次服务器调度，
绑定原路径、任务、哈希、卡号和已批准队列；不是可移植的训练配置。
公开包省略私有资产，不能替换远端运行源码或伪造原有准备回执。
在新环境部署需重新准入数据、编译计划、执行 CPU/GPU 门控并选用全新输出目录。
12 更新恢复一致性和 CPU 测试证明可运行性，不证明最终准确率、收敛或长期稳定性。
