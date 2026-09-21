# 敦煌／Turufan 碎片拼接：Scorer 根因与下一阶段交接

更新日期：2026-09-21（北京时间）。对象：接手本项目的另一个对话或 Agent。

GitHub 公开版：保留研究结论和代码导航；机器登录信息、原始样本与权重不公开。

这是一份项目状态与研究方向快照，不是新训练已启动的通知。此前七路实验、42 个分类头评估端点，以及随后 09-21 05:07 完成的 D／ResGCN 等八组均已完成。新增两层局部证据实验与真实域分折校准见 [LOCAL_EVIDENCE_V2.md](LOCAL_EVIDENCE_V2.md)。本页保留早期结果的历史口径，不表示当前 GPU 实时状态；后续成组训练不在此次发布范围内。

## 1. 先读这一页：当前共识

**下一阶段主线：让 Scorer 阅读 Matcher 找到的候选接缝局部特征，通过 Cross-Attention 判断这组局部证据是否支持可拼。全轮廓 token 打分保留为对照，不再作为主要改进方向。**

已经找到重要的失败机制与有效改进方向，但不能写成“唯一根因已经证明”或“Layout 正确但 Scorer 低分已经解决”。证据分为三层：

1. **训练对照支持：选取匹配区域有价值。** 固定 S7 Matcher、训练数据和分类头预算，全量 token 改为最终位移内点的去重端点，人工保留敦煌集 F1 从 58.04% 到 66.67%，AUROC 从 0.7384 到 0.8646；负例误报 133→7。C1/C2 等容量对照也支持局部证据选择，而不只是增加参数。
2. **还没有解决召回。** 上述主阈值下，正例召回 59.32%→51.19%，并非全面提高；216 个原本摆放正确的正例，通过分类仅 145→148，仍有 68 个被拒绝。不能只展示 F1 上升就称“救回 Layout 漏判问题已解决”。
3. **剩余机制有证据，但修复尚未完成。** 少支持真接缝在训练输出中不足；正 pair 可能被选到错误候选，却仍以正 pair 标签训练局部头；几何残差偏好会误伤真实接缝；采样密度、尺度和硬候选选择会改变局部输入。

“全轮廓汇聚会稀释或干扰接缝证据”是合理解释，但尚未证明所有低分都来自同一种信息损失。S4/S6/S7 的 CA Scorer 直接读描述子，并非先把 Sinkhorn 矩阵压成一个标量。局部化也不能被解释为完全不要全局上下文：目前选中的描述子已经过 Matcher 的 context 网络。

证据主入口：[最终根因综合][synthesis]、[完成范围][completion]、[固定预算完整汇总][summary]。

## 2. 任务目标、边界与用户方向

目标按优先级：

1. 判断两块真实写卷碎片是否可以拼接，重点减少漏判，同时控制伪接缝误报。
2. 从对应轮廓／Patch 计算平面相对 xy 位移。**不估计旋转。**
3. 使仿真训练分布覆盖真实数据：部分接缝、材料缺失／腐蚀、大小不等、多碎片合并或丢失、直边干扰。

当前用户方向：

- 主分类输入改为预测匹配区域；保留对应关系与匹配强度的方式继续对照，不预设越多元数据越好。
- 后续采用 S7 难化训练数据；按用户后续修订，新八组 Scorer 实际使用 **2 层、96 维**，已经完成 4／8 heads 对照。四层主版本是被替代的早期计划，不是本轮设置。
- 利用 dustbin 判断证据可信度值得试，但不能直接将当前 dustbin 概率设为硬否决。
- 实验优先，不再投入大量 HTML 美化、反复文件校验或逐分钟进度播报；汇报完成、失败、阻塞和改变结论的发现。
- 主要使用模型输入输出、实际对应点、候选位移和受控干预诊断；不要仅靠肉眼看热图归因。
- 吞吐不是首要研究指标，但训练应合理利用 GPU；不要再次无依据地把 microbatch 降为 1。

本次 GitHub 请求仅同步已完成实验的代码与交接，不启动或中断服务器训练。

## 3. 输入、输出与关键概念

### 输入

- 两张保持相互像素比例的二值材料 mask。正式 S7 模型输入画布为 800×800；mask 及 contour 坐标必须来自同一套预处理。
- 每块外轮廓的有序点 `points_rc_a/b`，形状 `[B,N,2]`，以及 `contour_valid_a/b`。
- 模型实际六输入：`mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b`。
- 当前研究主网络为 mask-only；RGB 主要用于人工检查。不要把论文原生 RGB 方法与项目 mask-only 适配结果混称为原论文复现。

### 输出

- Pair：`logit`、`sigmoid(logit)`，与该实验约定的阈值比较；原训练采用仿真验证阈值，后续真实域校准采用独立折阈值。须保留既有 `decision_valid` 语义。
- Matcher：非 dustbin 的匹配质量矩阵 `Q/assignment`、两侧未匹配质量 `unmatched_a/b`、双方 96 维 contextual token 特征。
- Layout：候选对应边、内点、位移、数值有效标记及候选分组信息。
- 位移约定：`t_a_to_b_rc` 满足 `pB ≈ pA + t`；把 B 放到 A 画布时使用反方向位移。不要混淆 row/column、xy 与摆放位移符号。

### 不要混淆的概念

- “端点”是对应边 `(A_i,B_j)` 两侧的 Patch 中心，不是接缝起止点。
- “少支持”口径：最终内点对应的两侧去重中心数取最小值；≤32 是诊断分组，不是物理接缝长度，也不是面积较小的那一块。
- 几何残差：`||pB_j - pA_i - t_pred||`；点对头当前输入除以 10px。不是 GT 位移误差，不是神经网络残差连接，也不是 Sinkhorn 行列平衡残差。
- `layout.valid` 仅表示解算满足数值／最低支持条件，**不代表 Pair 邻接或真实摆放正确**。
- GT Layout 正确也不能证明全部预测内点都是真接缝点。

## 4. 当前网络结构与训练方式

```text
两个二值 Mask + 有序 contour
  → 每个采样中心取 7/16/32/64px 视野，分别采样到 16×16
  → 共享 Patch CNN → 每尺度 96D → learned gate 融合为每中心一个 96D token
  → Context：2 个环形卷积残差块 + 坐标编码
               + 面向另一碎片 32 个汇聚节点的 Cross-Attention
  → primal/dual 互补描述子 → affinity → 带 dustbin 的 partial Sinkhorn
       ├─ Q → Top-2 union → 位移共识峰 → 内点／精修 → 无旋转 Layout
       └─ 选中区域的 contextual features → 独立 Cross-Attention Scorer → Pair score
```

### 4.1 Matcher 和采样

- 当前主对照为 **S7 M12、original512**。大轮廓按弧长重采样到上限 512；较短轮廓保留较少有效点。因此不是全数据固定像素步长，也不是每块强制 512 个有效点。
- 512 与旧 2048 版本不仅数量不同。S5/S8 的成对近似共同步长为 `max(3px, LA/2048, LB/2048)`，小碎片有效点更少。不能把它们当纯 token 数量消融。
- 四尺度先融合描述子，再执行一次 Sinkhorn；不是四张独立矩阵。
- Sinkhorn 温度 0.25、100 次迭代。输出已缩放到每个有效实点目标质量约 1：`sum_j Qij + unmatched_a[i] ≈ 1`，B 同理。
- Decoder 为行列 Top-2 **并集**、最多 512 条候选边、10px 内点半径、至少 3 内点、3 次加权均值精修。极小正质量也可入选；权重按该对最大值归一，未使用 dustbin 绝对拒绝。
- Matcher 阶段不使用 PairBCE；现有入口配置为 assignment NLL 权重 0.5、合适样本的软位移辅助权重 0.5、Sinkhorn 数值平衡辅助 0.05。assignment NLL 包括 matched 与 unmatched/dustbin 监督。受损样本位移监督按既有 eligibility 处理，不能强制所有腐蚀边严丝合缝。

实现：[Matcher][matcher]、[Patch CNN][patch]、[Sinkhorn][sinkhorn]、[Layout][layout]、[匹配损失][loss]、[分阶段训练入口][staged]。

### 4.2 当前新 Scorer，不是旧 Coarse CNN

本轮五种局部输入对照均使用冻结 S7 M12，fresh **2 层、96 维、4 heads** 的独立 CA 分类头。旧原版 S7 分类头为 1 层；另有 S6 系列 1/2/4 层对照，不能混成“新端点头已经是 4 层”。

- 双向 Cross-Attention；每层 A←B 与 B←A 共享参数，不同层参数独立；FFN 为 96→192→96。
- 最后才进行 attentive pooling 与 max pooling；双方汇聚特征的对称平均、绝对差进入 MLP，输出一个 logit。
- `all_tokens`：全部有效 contextual tokens。
- `matched_tokens`：最终 Layout 内点对应的去重端点特征；不显式保留一条边的对应身份，不直接输入 Q／残差标量。
- `matched_edges`：显式绑定对应双方特征，附原始 Qij 和归一化几何残差。
- `edge_seed`：使用初始主位移峰对应边。
- `edge_multi`：最多 5 组分离位移候选，共享点对 Scorer；取最高 logit 做 PairBCE。不是每行 Top-5，也不是全矩阵仅保留 5 条匹配。

Scorer 单独训练 PairBCE，Matcher 冻结；不反传修改其 Layout。当前 fresh 单卡头为 microbatch16、累积1、有效 batch16，AdamW；正式对照保留 C8/C16。多卡或其它分支须读各自 protocol，不推定所有实验 batch 相同。

`FrozenMatchedInference` 仍计算 base 历史 coarse 字段以兼容输出，但**新头不读取 coarse 特征或分数**。它替换 fused/local 分类输出，不改变生产 Layout；多候选最高分位置目前只是诊断输出，不是已上线重排。

实现：[CA Head][ca-head]、[fresh 输入变体][fresh-head]、[训练][fresh-train]、[推理组合][fresh-infer]、[多候选生成][candidates]。

### 4.3 参数量与同事方案的启发

- 我们 Patch Encoder 实际参数 44,936；完整 Matcher context 为 223,200，其中两个环形卷积块 148,224。
- 同事角度序列 Patch Encoder 为 241,312；PairingNet/ShreddingNet 发布结构对应项目实现的两路有效 Patch CNN 合计各 20,192，未含 GCN。不能据此说我们比两篇网络小。
- 两篇相关配置的 14 层 ResGCN 主要对应轮廓内特征传播；我们的这一部分更浅，但还另有跨碎片 context。参数量、深度与有效感受范围不是一回事。
- 同事最终方案：轮廓约每 2px 取点；窗口覆盖自身周长约 10%，步长约 5%，每窗重采样为 90 个序列位置，通常约 20–21 窗；1 层 self+cross attention、64D、4 heads。90 个位置不是 90px。
- 可借鉴其保留窗口形状信息、尺度处理和局部比较，但不必把图像 Patch 强制换为角度序列。
- 同事历史分数包含排序指标且模型曾在 Turfan 数据上训练，不能直接当我们的 OOD 分类 Accuracy 基线；已有 G0/G1 是启发式适配，非同事原生模型复现。

## 5. 数据与评估口径

### 5.1 当前 S7 数据

固定 TRAIN24K：12,000 正 pair＋12,000 负 pair；不是每个 epoch 重新生成一套 24K。继承旧数据中的普通正负、大小差距与多碎片组合，另按以下配方分配增强槽位：

| 配方 | 计划槽位占比 | 含义 |
|---|---:|---|
| reference_e1 | 30% | 沿用既有 E1 风格 materialized 数据，不等于全部干净 |
| partial_curve | 20% | 曲线部分接缝，非整段完美对应 |
| wave | 15% | 连续起伏内缩腐蚀 |
| seam_gaps | 15% | 接缝多处材料缺失 |
| local | 10% | 局部深腐蚀 |
| gen5_partition | 10% | Gen5 组合分区 |

强腐蚀请求深度 10–30px；接缝缺口 1–5 个，支持范围 15–50px。上述是配方／请求范围，**不能冒充实际施加成功比例或每例测得深度**；实际失败、回退与侵蚀量由生成 metadata 记录。人工新切边／缺口不作为新增正对应；原接缝标签按来源继承。

来源：[数据配方][s7-data]、[materialize][s7-materialize]、[强腐蚀实现][weathering]。配置与实际 TRAIN 来源也保存在 [matched_tokens 评估 protocol][endpoint-protocol] 的 `model.training_identity.cache_bindings.train.population`。

### 5.2 原始三域结果不能互相代替

本小节及第6节描述原 SIMVAL 阈值实验。后续真实域校准另构造 Turufan 301 负例，并重建敦煌跨来源负例；不能用新口径改写这些历史结果。

- 仿真 VAL3K / TEST3K 各 1,500 正＋1,500 负。阈值与模型选择只能使用约定的仿真验证，不能在测试集上反复选最优。
- 敦煌完整评估 1,016 对；当前人工保留口径为 **295 正＋508 负＝803 对**。负例含 39 个原始严格负例与 469 个构造干扰。人工看过模型表现后筛选，因此它是诊断口径，不是新的盲测集。
- Turufan 为仅有 frag1/frag2 的 **301 个已知正 pair**，没有负例与 Layout GT。只能报接受数／正例召回，不能把它报成二分类 Accuracy/F1/AUROC 或摆放准确率。同前缀两片保持同一原图像素比例，不同前缀没有统一尺度。
- Layout 正确：正例上解算有效且相对位移 L2 误差≤20px。另报“正确 Layout 且分类接受”的数量，避免 Pair 与 Layout 指标混淆。
- R99 是仿真验证校准到 99% recall 的工作点，不保证真实域 99%。一些头 max-F1 的仿真 recall 已超过 95%，所以 R95 阈值反而可能更高，不能把 R95 当作必然比 max-F1 更宽松。
- 困难验证 6,000 行来自原 3,000 SIMVAL 的 clean/requested 两个视图，不是 6,000 独立来源；无实际损伤的回退样本分列。

## 6. 本轮关键结果：足够支撑方向，不能夸成完全解决

以下均为 S7 M12＋fresh C16，各自 SIMVAL max-F1 冻结阈值；敦煌口径 295 正＋508 负。五种头的生产 Layout 完全相同，原始正确 216/295。

| Scorer 输入 | 阈值 | 敦煌 Recall | 敦煌 F1 | 负例误报 | 正确 Layout 被接受 /216 | Turufan 接受 /301 |
|---|---:|---:|---:|---:|---:|---:|
| all_tokens | 0.673569 | 59.32% | 58.04% | 133 | 145 | 52 |
| matched_tokens | 0.834463 | 51.19% | 66.67% | 7 | 148 | 128 |
| edge_seed | 0.921604 | 48.14% | 64.55% | 3 | 140 | 130 |
| matched_edges | 0.920265 | 49.15% | 65.46% | 3 | 143 | 134 |
| edge_multi | 0.924698 | 48.81% | 64.72% | 6 | 142 | 133 |

matched_tokens 的 R99 阈值为 **0.3293727934360504**：敦煌 Recall 72.20%、F1 76.62%，正确 Layout 接受 192/216，但负例误报 48。放宽阈值有明确代价，不能把全部提升归于网络改进。

### 已完成的其他主要探索

- **C1/C2 等容量残差分类：** C1 选局部端点，C2 看全量 token。敦煌 F1 63.60% vs 57.60%，支持局部选择。C1 仍保留原全量 CA 分数，不是纯局部分类。
- **保留点对与强度：** matched_edges/edge_seed 没有超过 matched_tokens 的敦煌 F1；它们同时改变对应绑定、重复端点、元数据和参数量，不是纯 Q 强度增益实验。
- **多候选：** 295 正例中，候选集合有正确位移的 250 例；取 Scorer 最高分位置后仅 217 例正确。分解为：45 无正确备选、33 有正确备选但选错、74 选对但拒绝、143 选对且接受。250 是 GT 辅助的覆盖上限，不是已实现的成功数。
- **少支持条件偏差：** TRAIN 中较少侧≤32端点的正例 606、负例 11,885；68 个“端点头拒绝但 Layout 正确”的真实正例中，52 属于少支持。606 个训练正 pair 中有 316 个选中 Layout 错误，不能不加区分地重加权后称作“补真接缝”。
- **残差干预：** 对选出的 14 个点对头漏判但摆放正确真实例，仅将直接残差输入减半／归零分别救回 12／14；归零也新增误报。证明敏感性，不证明删除残差后重训一定更好。
- **密度／尺度：** 固定物理 Patch 和锚点，改变周围采样仍使 context 特征变化；只改坐标也能改变分数。尚无已训练的尺度／间距归一化方案被证明最优。
- **Matcher M12→M16→M20：** 困难受损正例的 Layout20 正确数 847→848→844；各自等预算 fresh 全量 Scorer 也无单调真实域增益。不继续盲目补到 32/48 轮。
- **CA 深度、谱摘要、G0/G1：** 均已完成既定对照，未稳定解决主要缺口。已有四层实验不等于新局部头四层已训练，也不证明四层普遍最优。

完整指标、逐例输出和限制见 [最终综合][synthesis] 与 [汇总][summary]；不要用本页替代具体实验 protocol。

## 7. dustbin：已完成诊断与待验证方向

已有 Matcher 训练包含 dustbin NLL，问题不是“以前根本没有未匹配监督”。当前解码从非零 Q 中找相对一致的峰；负例也能产生数值有效候选，所以不能从 `valid` 推导可拼。

09-21 在现有 S7 40 例矩阵上完成固定解码诊断，不重训、不重算 Scorer、不拟合阈值。其中 12 个有 GT 的敦煌正例：原解码正确 8 个；要求单条 Q 大于两侧 dustbin 后 0 个；要求两侧总匹配质量占比>0.5 后 4 个；软加权后 7 个。没有展示收益。

含义：简单硬拒绝会丢掉低绝对质量、但含有有用几何信息的真接缝。**不能据此说所有 dustbin 方法无效，也不能认为直接套 0.5 已合理校准。** 更有意义的后续是给局部候选 Scorer 增加两侧未匹配／匹配强度信息，学习如何联合判断；避免整圈未匹配比例惩罚短接缝。

这组 dustbin-aware Scorer 训练尚未开展。[诊断说明与代码入口][dustbin-probe]（逐例结果不公开）。

## 8. 新一轮已经执行的改动与尚未解决的方向

`local_evidence_v2` 八组已实现并训练：512／4 heads 对照、128/256 对应记录上限、512／8 heads、PairingNet 式与 ShreddingNet 式 14 层 Scorer ResGCN、联合 D、稳定局部证据输入。
全部冻结 S7 M12 Matcher 和 Layout；新 Scorer 为两层96D、实际／有效 batch48、固定 C16。
精确网络和代码见 [本轮交接](LOCAL_EVIDENCE_V2.md)，结果见 [八组汇总](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/FINAL_RESULTS_20260921.md)。

D 把可信少支持／部分接缝／高残差正例与困难负例的条件重采样、候选正确性辅助 BCE 合并训练；不生成新 mask、不改 Pair 标签。
stable 合并软支持、间隙容忍残差和局部坐标规范化，结果未胜出。不能将两组联合改动解释为每个单独成分均有效或均无效。
GCN 有召回／误报权衡，未全面胜出；没有更改生产 Layout。候选排序、dustbin-aware Scorer 和 Matcher 物理尺度不变性仍未解决。

后续真实域校准已采用来源分五折、四折选阈值、一折测；v1 无阈值范围限制，v2 限制 0.20–0.80 并另报固定0.30。
保持权重冻结，并按来源折构造负例。该设计隔离每次阈值选择与测试，但多轮真实域诊断经历不能被消除，不能称作新盲测。
成组正负训练属于另一轮，不在本轮八组中，不应重启已经完成的队列。

## 9. GitHub 代码位置与外部运行文件

本文件是可公开交接版本。仓库：<https://github.com/YuqingZhangMirror12/DH-Pairwise>。
所有模型代码保留原始相对路径；新的源码根是 clone 后的仓库根，不是旧的本机工作目录。

请按 [代码导航](CODE_MAP.md) 找到每个模块；完整训练与 checkpoint 的外部文件需求见
[运行说明](RUNNING.md)。原服务器登录信息和本机凭据路径不在公开版中。

复现控制版本是 **S7 M12 + matched_tokens C16**，不是重新宣布一个生产冠军。
M12+C16 有时记录为绝对 epoch 28，绝不代表 Matcher 训练了 28 轮。
缓存只保存 contextual features、选中边和原始 Qij；新增 dustbin 通道需要新缓存。

本次发布的八组及上述历史实验均已完成；本次上传没有启动新训练，不代表后续另行登记的训练已经完成。
接手时先读 [实验状态](EXPERIMENTS.md) 和 [定量结果](RESULTS.md)，再选下一项独立实验，
不要照旧脚本重启已经完成的队列。新数据、模型或采样配置使用新的输出目录与验证阈值。

原始逐例诊断、checkpoint、freeze/protocol 和人工标注保留于所有者的私人研究环境，
并未随代码公开。公开表格是汇总快照，不宣称 clone 后无需外部文件即可独立重算全部数字。

## 10. 结论用语边界

推荐：**我们已验证 Scorer 应围绕预测接缝证据打分，并定位到训练局部证据分布和候选监督的错位；下一步让“可拼”与“候选是否可信”更一致。**

不要写：全轮廓特征完全无用；Cross-Attention 已证明唯一正确；Q/dustbin 一加就好；有 Layout 候选必能拼；四层已是本轮头；训练轮数已充分到不需再验证；人工筛选后成绩代表无偏真实泛化；原仅正例 Turufan 有二分类 F1；或新增构造负例后就能验证其 Layout 正确率。

[synthesis]: EXPERIMENTS.md
[completion]: EXPERIMENTS.md
[summary]: RESULTS.md
[matcher]: ../staging/pairwise_v0_2/models/rachel_n512.py
[patch]: ../staging/pairwise_v0_2/models/local_matcher.py
[sinkhorn]: ../staging/pairwise_v0_2/models/optimal_transport.py
[layout]: ../staging/pairwise_v0_2/models/translation_layout.py
[loss]: ../staging/pairwise_v0_2/training/rachel_n512_loss.py
[staged]: ../experiments/rachel_n512_formal_30k/train_score_decoupled.py
[ca-head]: ../staging/pairwise_v0_2/models/rachel_decoupled_score.py
[fresh-head]: ../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/model.py
[fresh-train]: ../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/train.py
[fresh-infer]: ../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/inference.py
[candidates]: ../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/candidate_groups.py
[s7-data]: ../staging/pairwise_v0_2/pairwise_data/rachel_s7_dataset.py
[s7-materialize]: ../experiments/rachel_n512_formal_30k/materialize_s7_training.py
[weathering]: ../staging/pairwise_v0_2/pairwise_data/rachel_strong_weathering.py
[endpoint-protocol]: RUNNING.md
[dustbin-probe]: DUSTBIN_DIAGNOSTIC.md
