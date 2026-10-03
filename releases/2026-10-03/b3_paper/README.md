# B3 模型架构、训练设计与实验结果：论文写作交接

版本：2026-10-03。用途：交给下一模型撰写 Methods、Experiments、Ablation 和 Limitations；不是把历史实验逐项堆进论文。

**本文件包含已经完成的新 CAL＋B3 终点四方向 TEST 结果。** 本轮只读取冻结代码、训练回执和逐对预测，重算同口径统计；没有新训练、重新生成数据、重新推理历史开发集、在 TEST 上调阈值或改动原文件。终点推理原有任务于 **2026-10-03 11:08:59 CST** 完成，实际退出码 0；全部 1,587 个 CAL 与 283 个真实 TEST 的四方向预测、模型不变、清单及摘要 SHA、先封存阈值再启动 TEST 的顺序已核验。


2026-10-03 15:58补充：[新SELECT完整扫描](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_endpoint_heads/docs/SELECT_SCAN.md)已核验，冻结规则选出U29667。[原终点U31667配套双卡Scorer源码与安排](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_endpoint_heads/README.md)已独立发布，Patch使用GPU0/1、Stats使用GPU2/3；15:48已确认正式训练进程启动，尚无新头最终结果。**下文真实TEST仍属于“终点Matcher＋旧Patch”，不能误写成新头重训结果。**

## 阅读入口与本次新增内容

本公开版供无 SSH 的论文写作对话阅读。**代码链接固定到 `9dff51b32dc73b6a915c6954f6ecd40e48708d37`**；旧训练镜像原样保留，最新推理、生成发布和标签修复另列，互不冒充。代码与机器可读汇总可直接从 GitHub 下载；私有图像/权重/逐对预测不公开，已有本地案例包继续可用。

| 需要的材料 | 本文位置 |
|---|---|
| B3 最终 Matcher V2 架构 | §2–4：输入、层次、维度、部分OT、候选几何 |
| 课程与直缝何时加入 | §6.1、§6.5：原时钟与实际时钟、三阶段插入数量 |
| 最终 Scorer | §5、§7：已有 Patch U29667；Stats 仅对照；四方向候选合池B |
| 仿真、敦煌800、吐鲁番602和留出折 | §10.3–10.5、§11.2–11.6：历史/终点、全量/TEST分别报告 |
| v17＋Patch 消融读数 | §10.1–10.5：明确这是版本级比较，不冒充单模块消融 |
| 样本量、比例、有无直缝的数据消融 | §12.1–12.3：B0/B2/B3实测与预算混杂 |

附录： [代码地图](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_paper/CODE_MAP.md)、[数据方法](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_paper/docs/SIMULATION_METHODS.md)、[SELECT/CAL](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_paper/docs/SELECT_CAL.md)、[待人审标签覆盖层](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_paper/docs/RELABEL.md)。

## 0. 给论文写作者的核心结论

1. **B3 是 mask-only 的双碎片匹配与平移拼接系统。** 主链为多尺度轮廓特征 → 全轮廓自注意力与跨片注意力 → primal/dual 相似度＋带 dustbin 的部分最优传输 → 确定性几何候选与 T16 合并 → 轻量候选评分头。不是 RGB 大模型，不是端到端多片全局拼图，也不是任意旋转 SE(2) 恢复。
2. **论文中必须给“B3”加权重/推理版本后缀。** 历史 B3 使用旧仿真 SELECT 选出的 Matcher U9667；最新推理使用用户指定的终点 Matcher U31667，仍接在 U9667 Matcher 下训练的已有 Patch U29667 头。不能称“终点 Matcher 和头重新配套训练”。
3. 最新四方向方案在吐鲁番 TEST 的 Pair-F1 为 **0.9391**，同终点同头单方向为 **0.8269**；敦煌候选覆盖 **53/59→57/59**、最终布局 **49/59→52/59**，但预登记主阈值下误报 **1→12**，Pair-F1 **0.8785→0.8455**。**多方向改善候选与域外召回，不等于在每个运行点都全面提高分类。**
4. 基线比较有明确缺口：PairingNet/ShreddingNet 与新版敦煌清单共同覆盖 **331 对（292 正、39 负）**，不是完整 800 对；吐鲁番共同覆盖 **301 个正例**，不能据此报告这两基线在 602 对上的 F1 或误报率。本文给出诚实的交集对比及 v17/B3 的完整集与保留 TEST 表。
5. 数据增强、课程、架构、训练预算、选模和推理方式在多个历史版本间同时改变，不能把全部提升归因于某一模块。新标签修复覆盖层尚未用于这些模型训练；新混合SELECT的后续完整扫描已选出U29667，见顶部补充；下文真实TEST表仍对应扫描之前的用户指定终点与旧Patch，不是新配套头结果。

数据章节已有材料，不在本文件重写：[仿真数据方法与案例](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_paper/docs/SIMULATION_METHODS.md)、[新 SELECT/CAL 构造、目录与实际统计](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_paper/docs/SELECT_CAL.md)。

## 1. 模型身份：避免把三件不同的事叫作“最新 B3”

| 本文简称 | Matcher | 评分头 | 推理与阈值 | 能否称已完成 |
|---|---|---|---|---|
| B3-H/Patch | Matcher V2，旧 SIM 选 U9667 | Patch U29667；在 U9667 下训练 | 单方向；旧冻结 SIM-CAL，τ=0.76 | 原预定训练和必需终评已完成 |
| B3-H/Stats | 同上 | Stats U31667；在 U9667 下训练 | 单方向；旧冻结 SIM-CAL，τ=0.45 | 已完成 |
| B3-E/1view | **终点 U31667** | 沿用 Patch U29667 | 0°；新混合 CAL，主 τ=0.78 | 本轮 CAL＋TEST 已完成 |
| B3-E/4view | **终点 U31667** | 沿用 Patch U29667 | 共同 0/90/180/270°，候选合池 B；新 CAL，主 τ=0.80 | 本轮 CAL＋TEST 已完成 |
| v17/Patch | 旧架构，v17 从零训练；选 E16 | Patch E28 | 单方向；旧 SIM-CAL，τ=0.72 | 已完成的历史对照；不是 B0 |

U 表示优化器 update，E 表示 epoch。B3-E 终点不是新 SELECT 自动评出的“最优 checkpoint”；它是用户根据此前观察指定的主模型。四方向默认方案先在敦煌开发折2–4比较并锁定，再在独立新 CAL 标定；真实 TEST 没有参与本轮方案选择。已有真实数据长期用于研发，不应写成“首次完全盲测”。

## 2. 任务定义、输入输出与系统结构

### 2.1 任务边界

给定方向已校正的两张碎片，判断是否真实相邻，并估计把 A 对齐到 B 坐标系的二维平移 **t = (Δrow, Δcol)**。模型对 A/B 使用共享编码器；展示“把 B 放到 A 旁边”的偏移时应取反，不能混用向量符号。

一次前向只读取以下六个字段：

| 输入 | 批张量形状 | 含义 |
|---|---|---|
| mask_a、mask_b | B×1×800×800 | 二值碎片支撑区域，不读 RGB 纹理 |
| points_rc_a、points_rc_b | B×512×2 | 沿边界有序排列的 row/col 轮廓点 |
| contour_valid_a、contour_valid_b | B×512 | 有效点掩码；512 是容量，不保证每片都满512个有效点 |

GT 位姿、点对应标签、数据来源、腐蚀前轮廓、损伤 mask 不进入模型前向。输出包括局部传输 Q、未匹配质量、最多8个几何候选、候选分数及最终选中的平移。无候选或数值无效时不能作为有效接受。

### 2.2 可据此绘制论文架构图

```text
二值 mask A/B + 有序轮廓（共享权重）
  │ 每点 4 个物理窗口：7 / 16 / 32 / 64 px，分别采样为 16×16
  ▼
共享 Patch CNN → 每尺度 96D
  │ softmax 尺度门控 + 四尺度拼接的残差投影
  ▼
局部循环卷积 / 32-landmark 上下文
  ▼
2 轮 [全轮廓 Self-Attention（环形弧长 RoPE）
       + 双向 Cross-Attention + FFN] → 每点 96D
  ▼
对称 primal/dual cosine + 可学习 sharpness + 多尺度后期 logit
  ▼
一次 FP32 dustbin Sinkhorn → Q 与 unmatched
  ▼
平移投票 / 损伤容差鲁棒拟合 / 物理重叠检查
  ▼
T16 完全链接候选合并，精确对应边去重并重拟合，最多 8 簇
  ├── Patch 头：局部/上下文特征 + Q/几何 → 候选分数
  └── Stats 头：仅 Q/几何统计 → 候选分数（独立对照头）
  ▼
最高候选分数 → CAL 冻结阈值 → 接受/拒绝 + 平移
```

四方向推理是在上述**整条前向链外面**做四次共同旋转、逆变换后合并候选，不是另一个训练网络。两评分头是分别训练、分别汇报的替代分支，**不是同时加权融合的双头系统**。

## 3. Matcher V2 的真实实现

本节按 B3 冻结训练镜像 `runtime_work_13` 和实际训练 binding，而不是按当前类默认值推断。尤其 `RachelN512Config` 默认只有32/64两个窗口，但 B3 的实际 binding 是 **7/16/32/64 四窗口**；旧全局/coarse分类模块虽然仍在兼容性 checkpoint 中，**B3 adapter 不执行它们**。

### 3.1 多尺度局部表征

每个物理窗口用最近邻采样得到16×16单通道 patch。共享 CNN 为三层3×3卷积，通道 **1→24→48→64**，每层 GroupNorm(1,C)、SiLU、2×2 MaxPool；空间均值后 Linear64→96＋LayerNorm。四尺度和两碎片共享该编码器。

基础融合是每尺度96维特征经 Linear96→1 得 softmax 权重，再加权求和。V2 同时把四个尺度拼为384维，通过 Linear384→96 添加残差；该新增投影零初始化，避免初始化时突然改变旧主干输出。它不是四个独立 CNN 或四个独立 Matcher。

代码：[采样器](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/staging/pairwise_v0_2/models/rachel_n512.py#L187)、[共享 Patch 编码器](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/staging/pairwise_v0_2/models/local_matcher.py#L78)、[V2 特征层](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/network.py#L188)。

### 3.2 局部顺序上下文与全轮廓交互

保留的基础上下文包含两层循环 Conv1d 残差块、归一化坐标嵌入和32个沿轮廓分箱汇聚的 landmark，用4头注意力交换上下文。随后 V2 增加 **2轮全轮廓自注意力＋双向跨片注意力**：特征96维、4头、每头24维，FFN 96→192→96，pre-LayerNorm、残差连接。双向 cross 使用本轮更新前的 A/B 表征，避免先更新A再把更新后的A用于B造成顺序不对称。

Self-attention 使用按有效轮廓弧长归一化的环形位置旋转编码：设累计弧长 s、闭环周长 L，采用整数谐波相位 **2πh·s/L**，最大谐波256。Cross-attention 不把两条不同轮廓的任意起点当作共同位置编码。

新增 attention 输出层和 FFN 末层零初始化，使新模块先以残差恒等附近起步。目的在于兼顾局部接缝纹理形状、长距离轮廓上下文和跨碎片互证；这是设计动机，不能把它直接写成已有单模块消融证明。环形编码也**不等于整套 CNN＋几何系统严格旋转等变**，四方向结果的差异正说明仍存在朝向敏感性。

代码：[基础循环/landmark上下文](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/staging/pairwise_v0_2/models/rachel_n512.py#L258)、[弧长与整数谐波](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/network.py#L66)、[Self/Cross blocks](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/network.py#L132)。

### 3.3 点相似度与部分最优传输

对上下文特征分别学习 primal 与 dual 投影，并归一化为单位向量。主相似度是交换两侧投影后的对称平均：

\[
C_{ij}=\tfrac12\big(\langle p(h_i^A),d(h_j^B)\rangle+\langle d(h_i^A),p(h_j^B)\rangle\big).
\]

V2 还将最终96维上下文与各尺度原始96维特征拼接，以192→48的 primal/dual 投影形成各尺度相似度 C⁽ˢ⁾。最终 logit 为

\[
S_{ij}=gC_{ij}+\sum_s\alpha_s C^{(s)}_{ij},\qquad g=\exp(\min(\log g,\log5)).
\]

g 初值1，新增尺度权重 α 初值0；最大 sharpness 为5。**本轮 use_matchability=false**，不能在论文图中加一个实际未开启的可匹配性门控。

S 经温度 **0.25** 的单次部分最优传输求解：在行列中加入 dustbin，用 FP32 log-space Sinkhorn 迭代100次，记录边际残差，容差1e-3。有效实点传输 Q 为至多512×512，另有 A/B unmatched 概率质量。Q 表示局部点匹配证据，**不是整对碎片可拼概率**。评分头不再运行第二个 Sinkhorn。

代码：[真实 B3 adapter 前向](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/adapter.py#L11)、[部分 OT](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/staging/pairwise_v0_2/models/optimal_transport.py#L148)。

## 4. 从 Q 到可解释拼接候选

### 4.1 原生候选、损伤容差与物理检查

取行/列各 top-2 的对应边，形成位移票 **pⱼᴮ−pᵢᴬ**，使用 Q 与观测弧长作为权重。配置初始16个种子、Q绝对门槛1e-4，迭代鲁棒拟合；最终候选预算8。接缝损伤引起的法向缺口不是强行当作切向位移：几何模型法向容许区间 **[0,13]px**、切向额外偏移0、可靠法向下限0.5、sigma下限1px。投票使用法向区间内0/.25/.5/.75/1五档假设。

采用基于轮廓采样间距的各向异性残差：normal/tangent sigma 系数分别约 **0.24575 / 0.32288**；证据切向系数0.61740，fallback系数0.29422。物理检查拒绝超过配置阈值的材料重叠（0.10）。这些都是**推理几何参数**，不是训练数据中腐蚀强度的上限；不能用13px容差宣称已解决30px深腐蚀标注。

### 4.2 T16 完全链接合并

对原始拟合位姿做完全链接聚类：同一簇中**任意两个原始位移**的距离都必须≤16px。不是单链接链式传递，也不是仅要求距移动中心≤16px。合并后取精确(i,j)对应边并集，重复边只计一次，重新读取原始在线 Q，对整个并集拟合共同平移，并限制共同平移相对原始假设的偏移；没有平均若干错误摆放来制造新证据。

这样设计主要针对同一接缝被多个相近种子拆开、以及链式归并吞并不同布局的问题。仍须报告未覆盖、选错和误报，不能把“有候选”写成“摆放正确”。

代码：[冻结运行时实际采用 T16 的别名](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/pose_consensus.py#L10)、[完全链接与并集重拟合](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/threshold_builder.py#L88)、[原生候选](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/legacy_pose_consensus.py)。**不要引用仍指向旧 builder 的其他工作副本来描述 B3。**

## 5. 轻量候选评分头

### 5.1 Patch/Context 主头

对于候选中的每条精确对应边，组合两类输入：

- 局部特征与上下文特征：每类对两碎片取均值和绝对差，共 **4×96=384维**。
- 8个 Q/几何量：log(1+100Q)、Q、残差距离/20、其平方、两端 unmatched 均值、unmatched 绝对差、候选外Q质量均值、log(1+弧长)/4。

392维边特征经 **392→64→32** MLP（GELU）。按 Q×弧长归一化加权均值池化和 max 池化，得到64维候选描述，再拼16维显式统计，经 **80→64→32→1** MLP，sigmoid输出候选分数。

16个统计量为：对应边数、Q总和、Q×弧长总质量的log值；Q均值/最大值；有效样本比例；两片轮廓覆盖率均值/最小值；残差加权均值/RMS/最大值；未匹配质量、候选外质量、按较小片面积归一的重叠率、原始位姿直径/16、合并成员数log。条件归一化池化之外保留绝对证据质量，避免极少边也因归一化获得虚高置信。

### 5.2 Stats 对照头

仅输入上述16维 Q/几何统计，MLP **16→64→32→1**；完全不读取 learned Patch/Context 特征，也不通过另一个神经模块间接读取。它是同一候选系统上的轻量评分对照，不是另一种 Matcher。

| 可学习部分 | 实际参数量 | 口径 |
|---|---:|---|
| B3 Matcher V2 | 696,879 | Matcher 训练回执登记的活跃训练参数 |
| Patch 头 | 34,529 | 冻结 Matcher 后独立训练 |
| Stats 头 | 3,201 | 冻结 Matcher 后独立训练 |
| Matcher＋Patch | 731,408 | 上述两项之和；不含兼容性 checkpoint 内未执行旧模块 |
| Matcher＋Stats | 700,080 | 同上 |

该计数不是 FLOPs、显存或序列化文件大小；没有额外大尺度全图 CNN 分类器、局部冲突分类器、learned pose refinement 或评分头 attention。候选 winner 为最高 logit，再比较标定阈值。可解释的是中间证据和几何量，不意味着概率已经跨域完美校准。

代码：[逐边特征与全部统计定义](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/binary_scorer_v1/head.py#L48)、[两个 MLP](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/binary_scorer_v1/head.py#L96)。参数量与实际配置来自 [B3 Patch training_complete/binding](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/training_design.json)。

## 6. 训练设计、监督与选模

### 6.1 三段独立训练，不是三网联合优化

先从随机初始化训练 Matcher；按当时冻结 SIM 规则选 U9667，冻结其参数和 eval 状态；然后分别新初始化 Patch / Stats 头训练，各自不回传更新 Matcher。最新 B3-E 仅替换推理用 Matcher 为终点U31667，未重新训练两个头。另一个 E32 小LR联合实验不属于 B3，不混入本论文主结果。

| 项目 | B3 实际值 |
|---|---|
| 训练库 | v17_filtered 12,179＋v17.5 6,000＋v18 3,000＋严格直缝6,000 = **27,179对** |
| 原课程更新 | 15,000 / 6,000 / 3,000，共24,000 |
| 新增直缝更新 | **7,667**，插入原课程，保留原数据曝光顺序 |
| 实际三阶段更新 | **19,667 / 8,000 / 4,000** |
| 每个模块预算 | **31,667更新，1,013,344 pair exposures**；Matcher、Patch、Stats各自达到该预算 |
| 批量 | effective batch32，正/负16/16；不能把重复曝光数当作独立样本数 |
| 优化器 | AdamW，weight decay1e-4，gradient clip norm5 |
| 学习率 | update0:1e-4；19667:5e-5；27667:2.5e-5；阶段切换不重置优化器 |
| 精度 | FP32；activation checkpointing用于显存节省；不把它写成模型模块 |
| 随机种子 | Matcher26092407，头26092406，数据顺序26092408 |
| 终止 | 固定共享更新预算；不是证明充分收敛后的停止 |

正负均衡采样与原始训练库的正负数量不同：v17_filtered有4,679正/7,500负；后续6K/3K/直缝6K库使用各自既定样本。论文要分别写“唯一数据量”和“训练呈现次数”。

### 6.2 Matcher loss

监督点对应以非负索引表示匹配、-1表示dustbin、-2表示忽略。实际损失为：

\[
\mathcal L_M=.5(\mathcal L_{match}+\mathcal L_{dustbin})
 +.5\mathcal L_{clean\text{-}pose}+.05\mathcal L_{marginal}.
\]

match项是已标定对应上的负对数似然；dustbin对两侧明确未对应点计算。clean-pose为历史whole-Q平移辅助项，只在 `pose_enabled` 且位姿有效的干净样本上使用，预测/目标除32后Smooth-L1；**不是要求腐蚀后轮廓点全部零间距**，也不是最终候选decoder。边际项为超过1e-3的最大行/列残差惩罚。

代码：[Matcher 损失及实际mask逻辑](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/scratch_matcher.py#L31)。历史点标签确有后来发现的漏配/深腐蚀继承问题；这里描述“当时实际训练了什么”，不表示监督语义已无误。

### 6.3 Scorer loss

候选级 BCE，加 **0.2×softplus(z_bad−z_good+1)** 的同对候选排序项。正片对的候选平移与GT误差≤20px才是正确候选；负片对候选为负。没有正确候选时，不把某个错误候选强行改成正例；无候选时不捏造可训练候选。先在一对已知候选内平均，再对batch的全部pair平均。无旧复杂头的局部冲突/extension/额外定位损失。

代码：[BCE与排序损失](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/binary_scorer_v1/loss.py#L23)。

### 6.4 SELECT、CAL、TEST职责

- SELECT用于checkpoint或设置选择，CAL用于已冻结模型的阈值标定，TEST只评估，三者不能混写。
- 历史B3-H仍按原仿真规则选出 Matcher9667、Patch29667、Stats31667；旧SELECT覆盖不足是本次改进的动因，但不能据此把所有旧测量删掉或声称终点在所有指标必优。
- 新混合数据正式清单：**SELECT1,596（796正800负），CAL1,587（793正794负）**；Gen2/3/4/5、v17/v17.5/v18、严格直缝均覆盖。正式版本为 `published_03`，实际来源、缺额/去重沿革及清单 SHA 见前述公开数据交接；数据 payload 不在 GitHub。
- 本轮B3-E只使用新CAL做阈值标定，没有做新SELECT checkpoint扫描。课程部分真实母图族为SELECT6、CAL11；增强views不是独立母图。
- Task3点标签FULL/TIGHT覆盖层已构建，但训练准入未开放，**当前B3和上述SELECT/CAL没有偷偷换成修正标签**。[标签修复交接](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_paper/docs/RELABEL.md)。

### 6.5 直缝从何时加入：按原课程时钟穿插

**直缝不是只在最后阶段加入。** 从第一阶段的热身窗口就加入；保留原课程 batch 顺序，再插入独立直缝 batch。窗口为原课程 update 的左闭右开区间，不是新增后的实际 update 编号。

| 原课程区间 | 课程来源 | 原更新 | 新增直缝更新 | 合计更新 | 该窗口直缝占比 |
| --- | --- | --- | --- | --- | --- |
| [0,1500) | v17_filtered 热身 | 1500 | 167 | 1667 | 10.02% |
| [1500,15000) | v17_filtered 后续 | 13500 | 4500 | 18000 | 25.00% |
| [15000,21000) | v17.5 | 6000 | 2000 | 8000 | 25.00% |
| [21000,24000) | v18 | 3000 | 1000 | 4000 | 25.00% |

三阶段分别 **15,000＋4,667＝19,667；6,000＋2,000＝8,000；3,000＋1,000＝4,000**。全部31,667更新中直缝7,667（24.21%），原课程24,000（75.79%）。每更新32对，新增直缝曝光245,344次；这是重复训练呈现次数，不是新增245,344个独立样本。各batch正/负16/16，不能拿目录行比例替代训练采样比例。

冻结实现：[WINDOWS与B2/B3分支](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/runtime_inputs.py#L14)，[原曝光保留的交错编排](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/additive_exposure.py)，[实际更新时钟与LR编译](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/runtime_schedule.py)；实际binding：[training_design.json](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/training_design.json)。

## 7. 四方向集成：最新推理设计与成本

两张碎片同时旋转同一角度 **0/90/180/270°**；不是A、B各选四个角度的16组合，也不是额外搜索相对旋转。mask用无插值90°旋转，轮廓点同步变换，保留索引；每个方向跑相同 Matcher＋T16＋Patch 头，然后把平移向量逆旋回原坐标。

**方案B：**汇集所有方向的真实候选；在原坐标按16px完全链接距离去近重复，每组保留实际最高Patch分候选；从合并池中取最高Patch分者作为最终布局与分数。重复候选不会因多方向重复出现而把Q或概率累加。没有平均位姿、学习新的融合权重或先挑最大Q方向。

开发比较还检查过方案A（各方向先由头选候选，再按Q证据挑方向）与方案C（B布局＋各方向top分数平均），以及Q-only诊断。锁定B只依据敦煌开发折2–4，既有头方案共60个，Q诊断另15个；吐鲁番与保留TEST不能反过来重选默认。

几何候选覆盖与布局不受最终阈值变化影响；误报、接受正确数和Pair-F1会变化。四次前向大约带来4倍视图计算量，0°基线直接复用四视图中的0°结果，没有额外重跑。旧开发计时单向/四向：敦煌约0.392/1.584秒每对、吐鲁番0.406/1.621秒每对；仅为该批运行记录，不是跨硬件优化后的部署吞吐benchmark。

代码：[旋转、逆变换、候选合池](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/rotation_ensemble_v1/core.py#L24)、[新CAL/TEST独立入口](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/rotation_ensemble_v1/endpoint_calibration.py)。实际部署源身份保留于 [本轮完成核验](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/endpoint_aggregates.json)；公开版保留模型/源码 SHA 和标定统计，旧 run01 已退役，不与它的部分旧CAL预测混用。

## 8. 对照方法：实验中的模型到底是什么

### 8.1 PairingNet 与 ShreddingNet

文献身份：**PairingNet: A Learning-based Pair-searching and -matching Network for Image Fragments**（ECCV2024，[作者代码](https://github.com/zhourixin/PairingNet)）；**ShreddingNet: Coarse-to-Fine Restoration for Multi-Source Shredded Manuscripts**（[作者代码](https://github.com/tqychy/shreddingnet)）。本文表格使用的是本项目在共同任务限制下的适配重训版本，**不是原论文表格、官方预训练成绩或完整多源多片系统复现**。

| 比较项 | PairingNet适配版 | ShreddingNet适配版 | B3 |
|---|---|---|---|
| 输入限制 | mask-only；支撑mask重复3通道替代RGB；N512，800画布 | 同左，保留轮廓分支 | 单通道mask；N512，800画布 |
| 表征 | 7×7局部patch、两路64D/14层ResGCN，轮廓序列±8邻居，学习融合 | 轮廓/伪纹理两路ResGCN，跨模态及两层跨片注意力 | 四物理尺度96D CNN＋局部上下文＋全轮廓Self/Cross |
| 局部匹配 | dual-softmax | dual-softmax；非Sinkhorn | 对称primal/dual＋dustbin Sinkhorn |
| 可拼评分 | 本项目双片判别MLP适配；不等同原pair-search检索分数 | fine对应矩阵形态学处理后3层2D-CNN分类器 | 几何候选级Patch或Stats MLP |
| 位姿 | 本项目已知方向translation consensus | 本项目translation consensus；非原多片MST结果 | 损伤容差拟合＋T16候选合并 |
| 实际训练数据 | E1物化24K，12K正12K负；VAL3K | 同一E1/VAL | v17→v17.5→v18＋严格直缝 |
| 实际预算 | 24 epochs，选E24 | coarse/matching/classify各24 epochs；选E20/E24/E17 | Matcher和每个头各31,667 updates |
| checkpoint选择 | VAL precision at recall95 | coarse/fine按本阶段规则；classify采用VAL precision at recall95 | 历史SIM选择；终点由用户指定 |
| 本次精度 | FP32 | AMP | FP32 |

两基线的E1训练集不是B3完整增强课程；既往审计记有6,968/24,000增强损伤样本，但这不是同强度、同组合分布的证明。不能将本表差异解释成“只更换架构”的公平因果实验，也不能把24轮预算写成原作者完整128轮recipe。

本轮采用逐对冻结阈值：PairingNet **0.80384767 / 0.08464187**、ShreddingNet **0.47705078 / 0.08361816**，分别是 `max_f1 / recall_first`。后者为预登记 `min(VAL max-F1, VAL recall99)`，不保证真实域99%召回。**ShreddingNet早期文档中的训练侧cluster-balanced阈值0.97802734，不是这些逐行预测表使用的阈值。**

代码与版本：[Pairing适配说明](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/staging/pairwise_v0_2/baselines/PAIRINGNET_RACHEL_ADAPTATION.md)、[Shredding适配说明](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/staging/pairwise_v0_2/baselines/RACHEL_SHREDDINGNET_BENCHMARK.md)、[实际Shredding阈值说明](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/comparison_aggregates.json)。官方checkout分别 `e878b781b2b2065a4b7da09d2f639e8f0a35e97a` / `0ae3b544ca4e910732f3f459b39aa15cdc62dbcb`；实际权重身份另见证据JSON，不能用当前类默认设置覆盖训练回执。

### 8.2 v17历史对照

选择已完成的 **纯v17、无课程、旧架构 Matcher E16＋Patch E28**。它与B3共享轻量Patch头类型，但Matcher架构、数据顺序和训练预算不同；没有已完成的“纯v17 Stats”可用，不能拿B0或v14补成该分支。此处采用它原冻结SIM-CAL阈值0.72，不从真实TEST中另挑阈值或epoch。

## 9. 指标、分母与对齐规则

| 指标 | 定义 |
|---|---|
| TP / FP / FN / TN | 接受真配对 / 接受负例 / 拒绝真配对 / 拒绝负例 |
| Pair-F1 | 2TP/(2TP+FP+FN)；不要求该TP的摆放正确 |
| Accuracy | (TP+TN)/总对数，强依赖正负比例 |
| Layout@20 | 正例中最终winner平移距GT≤20px；无布局计失败；不受接受阈值影响 |
| Candidate coverage@20 | 正例中至少一个候选距GT≤20px；不是最终winner正确率 |
| Joint-TP（J） | 既被接受、最终摆放又正确的正例数 |
| Wrong accepted（W） | 被接受但摆放错误/无效的正例数 |
| Joint-F1 | 2J/(2J+FP+W+P−J)，P为有GT的全部正例数；摆错同时不是正确找回 |
| AUC | 冻结分数排序统计；无候选/无效按运行实现记0，不是阈值优化 |

20px在统一800画布坐标下解释，不是原始文物物理尺寸。吐鲁番没有数值layout GT，因此其 Layout/Joint/候选覆盖指标均为 **null**，不填0，也不把视觉看起来对当成数值GT。

| 结果群体 | 总对数 | 正 / 负 | 本文用途 |
|---|---:|---:|---|
| 敦煌历史baseline原始记录 | 1,016 | 508 / 508 | 只作原始来源，不直接与800表比较 |
| 敦煌新版清单 | 原803，剔除3个已登记错误GT后800 | 292 / 508 | v17/B3-H完整历史读数 |
| 敦煌两基线与新版清单精确交集 | **331** | **292 / 39** | 四模型同样本比较；只有39负例，不能当完整真实域误报评测 |
| 敦煌保留TEST fold0 | **161** | **59 / 102** | v17/B3-H及最新B3-E；不是全新盲测 |
| 吐鲁番历史共同正例集 | **301** | **301 / 0** | 四模型的正例召回；无Precision/F1/FPR结论 |
| 吐鲁番新版完整集 | **602** | **301 / 301** | v17/B3-H；两基线尚无全部负例预测 |
| 吐鲁番保留TEST fold0 | **122** | **61 / 61** | v17/B3-H及最新B3-E |

交集由pair_id精确匹配，并核对有序fragment ID、标签及平移GT；不是按分数、正确与否或后验可拼性筛样。新版469个跨来源负例没有两基线的对应预测，本文保留此缺口，未补跑。数据与统计的复核脚本：[extract_evidence.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/analysis/extract_evidence_private_inputs.py)；公开完整计数与原证据SHA（不含私有交集ID）：[evidence.json](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/comparison_aggregates.json)。

## 10. 已完成的历史同口径对比

### 10.1 敦煌共同331对：同样本、不同训练recipe的系统比较

所有行均292正/39负；Layout分母292。基线两个冻结阈值都列出，不能只挑较差阈值来放大差距。由于正例占88.2%，不以此表Accuracy做主叙事；这里的F1也不能与下方800对的F1混用。

| 模型 / 冻结策略 | τ | TP / 292 | FP / 39 | Pair-F1 | Layout / 292 | Joint-TP | 错位接受 | Joint-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairingnet / max_f1 | 0.8038 | 35 | 0 | 0.2141 | 130 | 24 | 11 | 0.1468 |
| pairingnet / recall_first | 0.0846 | 136 | 1 | 0.6340 | 130 | 76 | 60 | 0.3543 |
| shreddingnet / max_f1 | 0.4771 | 137 | 7 | 0.6284 | 192 | 120 | 17 | 0.5505 |
| shreddingnet / recall_first | 0.0836 | 222 | 22 | 0.8284 | 192 | 173 | 49 | 0.6455 |
| v17 · Patch / frozen SIM-CAL | 0.7200 | 196 | 4 | 0.7967 | 241 | 191 | 5 | 0.7764 |
| B3-H · Patch / frozen SIM-CAL | 0.7600 | 182 | 1 | 0.7663 | 233 | 180 | 2 | 0.7579 |
| B3-H · Stats / frozen SIM-CAL | 0.4500 | 198 | 1 | 0.8065 | 238 | 195 | 3 | 0.7943 |

可以支持的结论：在这些冻结运行点，B3-H系统出现更少的“接受但摆错”，但不能单独归功于评分头；v17在这一批正例的最终winner布局 **241/292** 高于B3-H Patch **233/292**、Stats **238/292**，因此不能写“B3在敦煌所有指标均优于v17”。Shredding的recall-first增加真对召回也增加负例/错位接受；这正是仅报告Pair-F1不够的原因。

### 10.2 吐鲁番共同301个正例：仅正例召回

| 模型 / 冻结策略 | 接受 / 301 | 正例Recall |
| --- | --- | --- |
| pairingnet / max_f1 | 101 | 33.55% |
| pairingnet / recall_first | 185 | 61.46% |
| shreddingnet / max_f1 | 116 | 38.54% |
| shreddingnet / recall_first | 172 | 57.14% |
| v17 · Patch / frozen SIM-CAL | 208 | 69.10% |
| B3-H · Patch / frozen SIM-CAL | 248 | 82.39% |
| B3-H · Stats / frozen SIM-CAL | 259 | 86.05% |

这些数值回答“301个真实正配对中接受了多少”。没有为两基线生成负例分数，不报告它们在吐鲁番的准确率、F1、负例FP或布局成功率。v17/B3正例ID与两基线完全一致；这证明比较的是同一物理配对，不代表已证明两实现的预处理tensor逐位相同。输入适配/训练预算仍有差异，不能称官方基线等预算完整复现。

### 10.3 v17 与 B3-H：完整真实清单读数

这些完整集包含开发使用数据，仅作历史全量描述，不当作保留TEST。各模型采用其原SIM选模与冻结阈值。

| 数据集 / N | 模型 | Accuracy | Pair-F1 | TP / FP | Layout@20 | Joint-TP / 错位接受 | Joint-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 敦煌 / 800 | v17 · Patch | 86.62% | 0.7856 | 196 / 11 | 241/292 | 191 / 5 | 0.7655 |
| 敦煌 / 800 | B3-H · Patch | 86.12% | 0.7663 | 182 / 1 | 233/292 | 180 / 2 | 0.7579 |
| 敦煌 / 800 | B3-H · Stats | 87.75% | 0.8016 | 198 / 4 | 238/292 | 195 / 3 | 0.7895 |
| 吐鲁番 / 602 | v17 · Patch | 84.39% | 0.8157 | 208 / 1 | — | — | — |
| 吐鲁番 / 602 | B3-H · Patch | 91.03% | 0.9018 | 248 / 1 | — | — | — |
| 吐鲁番 / 602 | B3-H · Stats | 93.02% | 0.9250 | 259 / 0 | — | — | — |

### 10.4 保留TEST：v17 与 B3-H

| 数据集 / N | 模型 | Accuracy | Pair-F1 | TP / FP | Layout@20 | Joint-TP / 错位接受 | Joint-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 敦煌 / 161 | v17 · Patch | 88.82% | 0.8302 | 44 / 3 | 53/59 | 44 / 0 | 0.8302 |
| 敦煌 / 161 | B3-H · Patch | 91.30% | 0.8654 | 45 / 0 | 51/59 | 45 / 0 | 0.8654 |
| 敦煌 / 161 | B3-H · Stats | 91.30% | 0.8654 | 45 / 0 | 51/59 | 44 / 1 | 0.8462 |
| 吐鲁番 / 122 | v17 · Patch | 84.43% | 0.8155 | 42 / 0 | — | — | — |
| 吐鲁番 / 122 | B3-H · Patch | 88.52% | 0.8727 | 48 / 1 | — | — | — |
| 吐鲁番 / 122 | B3-H · Stats | 93.44% | 0.9298 | 53 / 0 | — | — | — |

B3-H在吐鲁番有清楚的Pair-F1改善；敦煌保留TEST分类改善，但Layout51/59低于v17的53/59。Stats在此吐鲁番TEST更好，不证明“去掉Patch特征必然更好”：它们训练/选中的头checkpoint不同，且只是一批数据、单次已完成实验。

### 10.5 仿真 TEST 与 v17＋Patch 读数

普通仿真TEST为3,000对（1,500正/1,500负），严格直缝TEST为900对（450正/450负）。这是已完成的冻结仿真TEST，不是新SELECT/CAL，也不是新CAL的训练拟合分数。下表均为各自SIM选模＋旧冻结SIM-CAL阈值、单方向；B3-H与最新终点B3-E不能混名。

| 模型/头 | 仿真TEST | N | τ | Accuracy | Pair-F1 | TP/FP | Layout@20 | Joint-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0/patch | 普通 | 3000 | 0.4600 | 96.87% | 0.9683 | 1437/31 | 1479/1500 | 0.9677 |
| B0/stats | 普通 | 3000 | 0.8000 | 97.53% | 0.9749 | 1440/14 | 1478/1500 | 0.9743 |
| B2/patch | 普通 | 3000 | 0.8000 | 97.53% | 0.9750 | 1442/16 | 1474/1500 | 0.9743 |
| B2/patch | 严格直缝 | 900 | 0.8000 | 66.78% | 0.5490 | 182/31 | 235/450 | 0.5279 |
| B2/stats | 普通 | 3000 | 0.7700 | 97.93% | 0.9791 | 1455/17 | 1478/1500 | 0.9785 |
| B2/stats | 严格直缝 | 900 | 0.7700 | 68.78% | 0.5957 | 207/38 | 229/450 | 0.5295 |
| B3-H/patch | 普通 | 3000 | 0.7600 | 97.27% | 0.9724 | 1443/25 | 1483/1500 | 0.9717 |
| B3-H/patch | 严格直缝 | 900 | 0.7600 | 96.44% | 0.9640 | 428/10 | 442/450 | 0.9617 |
| B3-H/stats | 普通 | 3000 | 0.4500 | 98.13% | 0.9812 | 1459/15 | 1484/1500 | 0.9805 |
| B3-H/stats | 严格直缝 | 900 | 0.4500 | 93.89% | 0.9411 | 439/44 | 444/450 | 0.9411 |
| v17/patch | 普通 | 3000 | 0.7200 | 98.03% | 0.9802 | 1464/23 | 1484/1500 | 0.9796 |

[表格全精度计数、源摘要及checkpoint SHA](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/simulation_data_ablation.json)。v17＋Patch的普通仿真Pair-F1为0.9802；完整敦煌/吐鲁番为0.7856/0.8157，留出TEST为0.8302/0.8155。它不是“只去掉某个B3模块”的结果，也没有纯v17的已完成Stats、Q-only或直缝TEST结果可补成空白分支。最新B3-E/4view本轮没有跑这两个仿真TEST，故不借用B3-H数值代填。

## 11. 最新结果：B3终点＋新CAL＋四方向

### 11.1 先冻结CAL阈值，再一次TEST

本轮两方案使用**同一个终点Matcher U31667、同一个已有Patch U29667头**。新CAL1,587对跑四视图；0°作为同模型同头baseline。主阈值按预登记 `[0.20,0.80]`、步长0.01、最大CAL Joint-F1、并列时最接近0.30选取。另预登记CAL负例FPR≤1/2/5%的运行点；它们不是TEST上反求的阈值。

| 预登记策略 | 单向τ | 单向CAL FP | 单向CAL Joint-F1 | 四向τ | 四向CAL FP | 四向CAL Joint-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 主网格 | 0.7800000000 | 25/794 | 0.9756 | 0.8000000000 | 76/794 | 0.9475 |
| CAL-FPR≤1.00% | 0.9754643440 | 7/794 | 0.9816 | 0.9912522435 | 7/794 | 0.9861 |
| CAL-FPR≤2.00% | 0.9461641312 | 15/794 | 0.9785 | 0.9798729420 | 15/794 | 0.9818 |
| CAL-FPR≤5.00% | 0.5390173197 | 39/794 | 0.9672 | 0.9416043162 | 39/794 | 0.9685 |

**重要限制：四方向主阈值0.80触及预登记搜索上界。** 它仅是该受限网格内的最优点，不是任意实数阈值下的全局最优。其CAL FP=76/794（9.57%），单方向主阈值CAL FP=25/794（3.15%）；四方向较高分数尾部需要重新标定。预先另列的1%点CAL FP均7/794。本文保留全部既定策略，不在看到TEST后扩网格、改主方案，或把某个TEST表现好的阈值改称预先选定主结果。

### 11.2 主运行点：最新与同终点单方向对照

| TEST / N | 方案 | τ | Accuracy | Pair-F1 | TP / FP | Layout@20 | Joint-TP / 错位接受 | Joint-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 敦煌 / 161 | 单方向0° | 0.7800 | 91.93% | 0.8785 | 47 / 1 | 49/59 | 46 / 1 | 0.8598 |
| 敦煌 / 161 | 四方向B | 0.8000 | 88.20% | 0.8455 | 52 / 12 | 52/59 | 50 / 2 | 0.8130 |
| 吐鲁番 / 122 | 单方向0° | 0.7800 | 85.25% | 0.8269 | 43 / 0 | — | — | — |
| 吐鲁番 / 122 | 四方向B | 0.8000 | 94.26% | 0.9391 | 54 / 0 | — | — | — |

阈值无关的几何/排序诊断：敦煌单→四候选覆盖 **53/59→57/59**，最终Layout **49/59→52/59**，AUC **0.9468→0.9497**；吐鲁番AUC **0.9073→0.9616**。四方向主运行点吐鲁番TP **43→54** 且FP仍0；敦煌TP **47→52**，但FP **1→12**，所以F1/Joint-F1反而下降。结论应同时交代收益与误报代价。

### 11.3 全部预登记低FPR运行点

“1%/2%/5%”是**CAL负例的预算**，不是保证真实TEST的FPR；同一策略的阈值用于敦煌和吐鲁番，不按域分别选择有利阈值。

| CAL-FPR预算 | 方案 | τ | 敦煌Pair-F1 | 敦煌TP / FP | 敦煌Joint-F1 | 吐鲁番Pair-F1 | 吐鲁番TP / FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.00% | 单方向0° | 0.9754643440 | 0.8571 | 45 / 1 | 0.8571 | 0.7921 | 40 / 0 |
| 1.00% | 四方向B | 0.9912522435 | 0.8762 | 46 / 0 | 0.8762 | 0.8807 | 48 / 0 |
| 2.00% | 单方向0° | 0.9461641312 | 0.8679 | 46 / 1 | 0.8679 | 0.7921 | 40 / 0 |
| 2.00% | 四方向B | 0.9798729420 | 0.8440 | 46 / 4 | 0.8440 | 0.8807 | 48 / 0 |
| 5.00% | 单方向0° | 0.5390173197 | 0.8785 | 47 / 1 | 0.8598 | 0.8190 | 43 / 1 |
| 5.00% | 四方向B | 0.9416043162 | 0.8673 | 49 / 5 | 0.8673 | 0.9107 | 51 / 0 |

四方向CAL-1%运行点在敦煌获得 **46个正确接受、负例FP0、错位接受0，Joint-F1=0.8762**；同策略单方向为45个正确接受、FP1、Joint-F1=0.8571。它在吐鲁番的Pair-F1为0.8807，低于四方向主阈值的0.9391，体现精度—召回取舍。不能各数据域从本表挑一行拼成一个不存在的“最佳统一设置”。

### 11.4 与历史B3-H的关系

四方向主运行点吐鲁番Pair-F1 **0.9391** 高于历史B3-H Patch **0.8727** 和Stats **0.9298**；但同时改变了Matcher checkpoint、CAL和视图数，不能归因于单一因素。相反，B3-E/1view与B3-E/4view是同权重的推理策略比较，更适合放进“多方向集成消融”。其阈值分别按相同CAL规则拟合，故是完整推理＋标定策略的消融，不是单纯在固定数值阈值下做4倍前向。

终点并非所有情形都优于历史选模：其单方向在吐鲁番主点F1=0.8269，低于历史B3-H Patch0.8727；两者阈值和CAL又不同，因此不能仅用这一个对比断言哪个checkpoint普遍更好。

### 11.5 开发集选择证据不混成TEST

已完成的终点开发比较使用敦煌639对和吐鲁番480对；敦煌仅折2–4选方案。先前四方向B在敦煌开发集coverage204→228/233、Layout192→207/233；同人口经验FP8/406下正确接受139→158。吐鲁番开发AUC约0.9410→0.9780。这里的“同人口经验FP”是开发ROC诊断，不是新CAL冻结阈值的TEST成绩。本文件主结果一律以上方新CAL/TEST表为准。

开发报告 SHA256：`b429d75133a1c2b7f0b66d55fd8f30675a841a143b44daba297ff2c8e689108b`；原始私有报告未公开。旧开发GPU总job被用户在旧CAL中途停止，无整job最终return0；只按完整per-dataset组件接纳已生成开发结果，再由独立CPU汇总成功。不要把这段历史与本轮实际完整退出0的endpoint_cal_run_02混为一谈。

### 11.6 最新终点 B3 的完整敦煌800／吐鲁番602（描述性）

复用先前完成的开发639/480对与本轮TEST161/122对，核对两部分ID不交叠、合并后与原完整清单严格相等、标签/折/GT/模型/头/策略一致。只按已封存的新CAL阈值重算计数，**没有重新前向、拟合阈值或选择设置**。

| 完整群体 | 方向 | τ | Accuracy | Pair-F1 | TP/FP | Layout@20 | Joint-TP/错位接受 | Joint-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 敦煌/800 | 0° | 0.7800 | 86.50% | 0.7857 | 198/14 | 241/292 | 189/9 | 0.7500 |
| 敦煌/800 | 四方向B | 0.8000 | 88.50% | 0.8441 | 249/49 | 259/292 | 236/13 | 0.8000 |
| 吐鲁番/602 | 0° | 0.7800 | 85.22% | 0.8272 | 213/1 | — | — | — |
| 吐鲁番/602 | 四方向B | 0.8000 | 94.02% | 0.9373 | 269/4 | — | — | — |

**这不是新的独立TEST。** 开发部分参与过设置选择；全量结果只回答“固定方案在完整清单上怎样”。旧开发任务在旧CAL中途主动停止，虽有逐dataset落盘完整组件和CPU汇总，未捕获全job最终内存model-after；不能把新TEST的完整退出证据追溯套给旧开发。保留TEST结论仍以§11.2–11.3为准。[全精度计数与合并证据边界](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/endpoint_full_descriptive.json)同时列出全部预登记低FPR策略，不从全量结果重选主阈值。

## 12. 已有消融、可写的主张与未做的证明

| 对比 | 当前证据 | 可写范围 / 不可写范围 |
|---|---|---|
| Patch vs Stats | 同B3-H Matcher、候选器和训练预算，各自独立训练/选模；§10.3–10.4 | 候选评分设计比较；不能宣称每个隐特征单独贡献已被证实 |
| 0° vs 四方向B | 同终点Matcher、同Patch权重、同CAL来源；§11 | 推理集成＋重标定的受控比较；有收益也有FP代价 |
| 旧v17 vs B3-H | 同真实清单的冻结结果 | 版本级系统比较；数据、课程、架构和预算混合变化 |
| PairingNet/ShreddingNet vs本方法 | 敦煌331交集、吐鲁番301正例 | mask-only适配系统的已有测量；非官方SOTA/完整公平复现 |
| Self-attention、Cross、RoPE、尺度logit、sharpness逐个移除 | 本文未接纳独立受控训练结果 | 架构中已实现，不等于逐项ablation成立 |
| 修正点标签训练的收益 | 仅生成覆盖层，未训练 | 不能写修标签后性能提升、解决深腐蚀问题 |
| 新SELECT重选最优checkpoint | 新数据已发布，本轮未扫描 | 不能把endpoint称为新SELECT验证的最优权重 |

建议论文主张保持为：**在仅使用碎片形状的双片任务下，以多尺度轮廓匹配、未对应建模、损伤容差候选和轻量候选评分构成可检查的拼接流程；训练采用逐级增强课程；多方向推理在固定权重下扩充候选并改善吐鲁番召回，但必须重新标定并报告误报。** “首次”“最优”“全面优于”“完全解决腐蚀”“严格旋转不变”等措辞目前没有充分支持。

如果论文必须有两基线在完整新真实清单上的主表，需要另行授权并冻结协议后补齐它们的469个敦煌新负例和吐鲁番负例预测；本任务没有自动启动这些实验。也尚无多随机种子均值/方差、按母图聚类的置信区间或显著性检验，不能凭单次百分点差写统计显著。

### 12.1 唯一样本量与比例（不是曝光量）

| 数据组成 | B0/B2无直缝行数 | B0/B2占比 | B3行数 | B3占比 | B3正/负 |
| --- | --- | --- | --- | --- | --- |
| v17_filtered | 12179 | 57.51% | 12179 | 44.81% | 4679/7500 |
| v17.5 | 6000 | 28.33% | 6000 | 22.08% | 3000/3000 |
| v18 | 3000 | 14.16% | 3000 | 11.04% | 1500/1500 |
| strict_straight | 0 | 0.00% | 6000 | 22.08% | 3000/3000 |

B0/B2为21,179对（9,179正/12,000负）；B3为27,179对（12,179正/15,000负）。v17_filtered有6,208行新增强、5,971行v14 fallback，不能把12,179都记成全新腐蚀。B0/B2原课程曝光阶段比例62.5%/25%/12.5%；B3新增直缝后的实际曝光见§6.5。

### 12.2 有无直缝的已完成实验与结果

| 分支 | Matcher架构 | 课程 | 严格直缝 | 每模块更新 / 曝光 |
|---|---|---|---|---|
| B0 | 旧架构 | v17_filtered→v17.5→v18 | 无 | 24,000 / 768,000 |
| B2 | Matcher V2 | 同三段课程 | 无 | 24,000 / 768,000 |
| B3-H | Matcher V2 | 同三段课程＋穿插直缝 | 6,000独立对 | 31,667 / 1,013,344 |

下表采用各分支各头原SIM选择/冻结阈值。普通仿真、直缝、完整真实集与保留折分列，不拿不同人口的F1相减。

| 分支/头 | 仿真3000 F1 | 直缝900 F1 | 敦煌800 F1 | 吐鲁番602 F1 | 敦煌TEST161 F1 | 吐鲁番TEST122 F1 |
| --- | --- | --- | --- | --- | --- | --- |
| B0/patch | 0.9683 | —（未纳入结果） | 0.7340 | 0.7798 | 0.7736 | 0.7677 |
| B0/stats | 0.9749 | —（未纳入结果） | 0.7112 | 0.7789 | 0.7800 | 0.7800 |
| B2/patch | 0.9750 | 0.5490 | 0.7838 | 0.8407 | 0.8545 | 0.8039 |
| B2/stats | 0.9791 | 0.5957 | 0.7717 | 0.8856 | 0.8302 | 0.8381 |
| B3-H/patch | 0.9724 | 0.9640 | 0.7663 | 0.9018 | 0.8654 | 0.8727 |
| B3-H/stats | 0.9812 | 0.9411 | 0.8016 | 0.9250 | 0.8654 | 0.9298 |

B2→B3的直缝TEST Patch-F1为0.5490→0.9640、Stats为0.5957→0.9411；但Patch普通仿真F1为0.9750→0.9724、敦煌800为0.7838→0.7663，不能只保留改善项。两头吐鲁番均改善。B2/B3使用的测试清单source身份与真实fold绑定已核对相同；[全部计数与源SHA](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/simulation_data_ablation.json)可核对阈值、FP、布局和Joint-F1。

### 12.3 数据消融能支持到什么程度

**B2 vs B3不是等算力、只改有无直缝的纯因果消融。** 同时增加7,667更新（31.95%）和245,344曝光，各自选中的Matcher/Scorer checkpoint也不同。因此可称“直缝扩充＋追加预算方案对照”，不可断言增益全由直缝数据造成。B0 vs B2预算/课程更接近，仍是整个Matcher实现包的比较，不是Self/Cross/RoPE逐项移除。

当前没有已完成的等预算v17-only→v17.5→v18逐层数据消融、不同数据量/采样比例扫描、多seed重复、修复标签重训或新SELECT重选效果。本文给出**实际训练样本量/比例**，不伪装成已经做过比例消融实验；缺项保留为后续实验，不自动开训。

## 13. 写作组织建议与可追溯文件

### 13.1 下一模型可直接采用的章节结构

1. Problem formulation：mask-only、方向已知、pair decision＋平移；输入与GT严格分离。
2. Multi-scale contour matcher：Patch CNN、局部与长程上下文、primal/dual、部分OT与dustbin。
3. Geometry-aware hypotheses：损伤容差、物理可行性、T16并集重拟合。
4. Lightweight candidate scoring：Patch主头、Stats对照，绝对质量与条件池化。
5. Curriculum training：引用已完成数据章节，再写预算、损失和选模边界。
6. Multi-orientation inference：共同四旋转、逆变换、候选池B和独立CAL。
7. Experiments：先说明baseline适配与可比子集，再列保留TEST；最后给同权重1view/4view消融与限制。

不要把“Methods提出的新版本”写成已经用修复标签或新SELECT重训；本次最新系统是 **旧已完成B3权重＋终点替换＋新CAL＋四方向**。为论文匿名化/命名时可另取正式方法名，但保留这些内部身份映射供审稿复查。

### 13.2 权重与完成回执

| 身份 | 文件 SHA256（不是tensor state hash） |
| --- | --- |
| B3终点 Matcher U31667 | `01c7070242c943861c1d44a6c5df4a5b463bb369a26439f9fbb053ea1faf6ac1` |
| B3历史 Matcher U9667 | `cdf5f03a9317b52926fbd757c85c481551edac333aa49df0e833383923ba0fbd` |
| B3 Patch U29667 | `099b4ae61c364268a8409689620eb5aa02cbd6c13fe936f56fa9d7545971b548` |
| B3 Stats U31667 | `feff10a3a97c5607045040864487959afa07cec5b5f00062106234b6d1eb3c0d` |
| v17 Matcher E16 | `6fe370c17a4e2cabebe327fed915ac173cf94ec379aea987896af6a19c746056` |
| v17 Patch E28 | `88e5ba288438933696393552cee0264ca0602e7c1004f8d38cb60b6adc67849c` |
| pairingnet / winner | `c81a3907e26f81b50c96bf7ab96aad9ce7783b1156f023597a638d2663af790d` |
| shreddingnet / classify | `f2efd55dcbb3edaa15e5ee40066a63f5c9b1f16372b376f153cc9a54cccbad6a` |
| shreddingnet / coarse | `5c8bc8d655e9a6a6573aadd51dbc01e01ffb79f7d94c56a30107e40a53e98bc3` |
| shreddingnet / matching | `65ce83b270662fb141be721a86ec1828b4e47f508561786b5788a0575f664637` |
| 本轮 protocol.json | `3165d7771ab9764734f96f1a6a533122aa799eed2867897e702a4c385ce574a3` |
| 本轮 calibration.json | `2e815c1d37dc0f21e496e071b62d27b28ff1ec1e1770e6ba126227c1b58d7245` |
| 本轮 test_report.json | `a05f2955cae8e42554a40136cb5286e3a01789f4f3d59d94e77268f8aa063fd6` |
| 本轮 complete.json | `abc43ba5c3d33e7a6993a23f34e6d04cfb567e71581673ab2e56e377881a702d` |
| 本轮 actual_return.json | `be89f729f48ceccb3cd1a1955bb1e4eb943c9b72b91d9e662b0b6d43ca27da79` |

公开阅读与复核入口（无需 SSH）：

- [历史基线、v17、B3 同口径完整计数](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/comparison_aggregates.json)
- [最新 CAL 封存、模型身份与 TEST 汇总](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/endpoint_aggregates.json)
- [终点完整真实集的描述性汇总](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/endpoint_full_descriptive.json)
- [仿真与数据消融的全精度计数](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/simulation_data_ablation.json)
- [实际训练配置与预算](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/training_design.json)
- [SELECT/CAL 实际交叉统计](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/evidence/select_cal_aggregate_statistics.json)
- [代码文件逐项 SHA256](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code_manifest.json)；[无需私有数据的校验脚本](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/analysis/verify_public.py)。
- 私有逐对预测、共同案例 ID、权重与人工标注不发布；公开汇总保留其来源 SHA。现有本地完整材料仍保留。
- [原始统计提取逻辑](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/analysis/extract_evidence_private_inputs.py)仅供阅读：依赖原私有输入目录，不能在公开包内凭空重建实验。

架构依据优先级：**冻结runtime_work_13＋实际training binding > 当前工作区同名文件 > 早期协议/说明中的默认值**。表格依据优先级：逐对预测＋固定人群/GT/阈值 > 原始摘要 > UI文字。本文不覆盖原始结果，不删除已暴露的问题或不利结果。

### 13.3 核验范围与未解决限制

本地重算验证了保存预测的指标算术、同集身份与GT，不证明所有原始人工GT永远正确。历史3例错误GT按既有登记剔除，未根据本轮模型结果再删案例。当前点对应修复待人审；新混合CAL母图族少、增强view相关，均应进入论文限制。无新训练、多seed验证、baseline补推理或人工标注变更。

本轮报告采用了来源可追溯与比较口径分离的工作方式，因而保留了“基线覆盖不足”“多方向主阈值误报增加”“v17部分布局指标更好”等结果，而不是只保留支持新方法的数字。
