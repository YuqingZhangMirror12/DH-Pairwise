# 仿真撕碎与课程增强：v17 / v17.5 / v18 论文写作交接

日期：2026-10-03（Asia/Shanghai）
范围：B3 实际准入的 v17_filtered、v17.5、v18；代码、参数、实际案例。
本件不重训、不重新生成数据、不修改标注；不是网络架构、结果表或消融实验的完整论文稿。

## 0. 本公开版的范围

本页保留方法、参数、配方组合与实际统计，代码链接均为固定GitHub版本，不依赖SSH。111例真实归档示例、原始NPZ/proof和画廊已单独交给数据所有者；**不把私有图像、案例ID或逐例元数据上传公开仓库**。需要图版时，可把已有本地 `Paper_simulation_methods_20261003` 资料包交给写作模型。图例是数据增强展示，不是模型预测成功案例。

上游基础撕碎器和私有v4.2生成器没有明确公开再分发许可，本仓库不冒充包含它们；以下Gen参数是方法记录，完整上游代码在所有者已有本地资料包中。项目自有增强与课程代码已公开。

## 1. 数据范围、版本名称与 B3 的关系

“v17”有两个不能混用的口径：完整生成档案有 TRAIN 24,000 对、CAL 1,500 对、SELECT 1,500 对、TEST 3,000 对；B3 使用的是进一步准入／去重后的 **v17_filtered 12,179 对**。后者并非正负各半。本文核心案例一律来自 B3 的最终准入目录，而非早期小试或正在构造的新 SELECT/CAL。

<!-- BEGIN POPULATION_TABLE -->
| B3 准入阶段 | 总 pair 行数 | 正例 | 负例 | 本版本新增强行 | v14 fallback 行 |
|---|---|---|---|---|---|
| v17_filtered | 12179 | 4679 | 7500 | 6208 | 5971 |
| v17.5 | 6000 | 3000 | 3000 | 6000 | 0 |
| v18 | 3000 | 1500 | 1500 | 3000 | 0 |
<!-- END POPULATION_TABLE -->

B3 最终准入目录还含 **6,000 对 strict straight-seam**，因此完整目录合计 27,179 对，而本件详细解释的三段课程合计 21,179 对。**不能在论文中写“B3 只用本文三段数据训练”。** 按本次要求，直缝包装器、B3 架构、训练结果与 ablation 见主模型交接；也不把数据目录行数等同于多轮训练曝光次数。

最重要的实际执行差异：

- B3 的 v17_filtered 中 **6,208 行实际采用 v17 新增强，5,971 行为明确登记的 v14 fallback**。fallback 是原样保留的旧版本样本，不等于没有任何增强，更不能归入“采用了 v17 新深度／新裁切”的统计。
- v17.5/v18 的端裁是**受条件约束的尝试**，不是每对都执行。满足原始接缝资格但找不到合格裁线时，代码保留该阶段输入并记录 skip；后面的主腐蚀、轻退化仍按配方执行。
- `partial` 是计划配方名。v17.5/v18 还会在 legacy Partial 破坏裁切阶段 20% 底线时跳过该次 Partial。图库中的 Partial 核心示例专门取 `partial_crop_applied=true`，并未因此改动数据。

<!-- BEGIN OPERATION_TABLE -->
| 版本 | 正例组 | 端裁已施加 | 端裁跳过 | 实际单端 | 实际双端 | Partial 配方组 | Partial 实做 | Partial 跳过 |
|---|---|---|---|---|---|---|---|---|
| v17.5 | 3000 | 1278 (42.6%) | 1722 | 1035 | 243 | 750 | 521 | 229 |
| v18 | 1500 | 603 (40.2%) | 897 | 507 | 96 | 375 | 241 | 134 |
<!-- END OPERATION_TABLE -->

这两种跳过不同：**endpoint trim** 是新加的单端／双端结构裁切；**legacy Partial** 是原配方中的局部断缺裁切。分别统计，不相加当作不同样本，也不把跳过视为一次成功裁切。计数证据见 operation_statistics.json（原本地资料包；未公开逐例记录），分母为正例组；每个正例均配一个同配方负例。

## 2. 代码入口与来源

本包按方法组织代码，保留文件内容，**不宣称这些扁平归档目录可以直接作为完整 Python 环境运行**。模块使用原项目的相对 import；正式复现应恢复对应冻结源树、配套依赖和数据清单，而不是在当前训练目录中热改。

| 层次 | 主要代码 | 用途与证据边界 |
|---|---|---|
| v14 前置基底、尺度与镜像 | [materialize.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v14_prerequisite/materialize.py)、[scale.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v14_prerequisite/scale.py)、[paired_augmentation_frozen.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v14_prerequisite/paired_augmentation_frozen.py)、[distribution_profile_v14.json](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v14_prerequisite/distribution_profile_v14.json) | v17 三版共同继承的 pre-weather 基底、尺度／镜像、原配方分配；不是说最终数据只用 v14 腐蚀 |
| v17 全量 | [pipeline.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v17_full/pipeline.py)、[generate.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v17_full/generate.py)、[source.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v17_full/source.py)、[geometry.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v17_full/geometry.py)、[fallback.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v17_full/fallback.py) | 实际 source_03：固定原始正片，有限尝试，不可行时按登记规则保留 v14 |
| v17 连续深度场与继承的轻退化 | [weather.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v17_runtime/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/aggressive_data_v17/weather.py)、[background_recession.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v14_prerequisite/background_recession.py) | v17 主场辅助模块与沿用的 s7_balanced_v2 轻退化实现；不是独立的 v17/light.py；参数为 3–8、5–15、1–3 px，分母定义见第 5 节 |
| v17.5/v18 配额和生成 | [full.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/full.py)、[full_plan.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/full_plan.py)、[generate.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/generate.py)、[spec.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/spec.py) | 实际 source_10；历史目录名含 v18_v19，但正式启用的只有 v17.5 / v18 |
| 新课程几何 | [geometry.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/geometry.py)、[trim.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/trim.py)、[primary.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/primary.py)、[weather.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/weather.py)、[light.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/light.py) | 条件端裁、连续主损伤场、whole-contour 末层轻退化 |
| 接缝保护与监督 | [crop_floor.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/crop_floor.py)、[seam_contract.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/seam_contract.py)、[audit.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/curriculum/audit.py)、[v17 supervision.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v17_full/supervision.py) | 原始／裁后 20%、最终可连接弧 30%、监督继承与归档一致性 |
| 历史 Partial | [partial_v14.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/v14_prerequisite/partial_v14.py) | 单端或开放中段断缺，原始底样保留的曲线定义 |
| 案例读取 | [rachel_materialized_dataset.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/reference_dependencies/staging/pairwise_v0_2/pairwise_data/rachel_materialized_dataset.py) | mask 解包、N512 点序列、继承 target；参考依赖不是新的生成源 |

### 2.1 原始撕碎：Gen2 / Gen3 / Gen4 / Gen5 到底是什么

代码名包含 `voronoi`，但这里不应笼统写成“调用标准欧氏 Voronoi 分区”：保留实现主要是**抽取自然轮廓弧，将弧段平移、旋转并组合成切割线，再提取连通子区域**。Gen 数字表示目标基础碎片数；后续选取两片或合并若干碎片得到 pair，并不是输入模型有 Gen 个分支。

| 来源名 | 保留代码中的几何组织 | 主要可核对参数（上游快照） |
|---|---|---|
| `gen2voronoi_1` | 两段自然弧拼成近相对方向的分界，目标 2 片 | 800×800；中心 x∈[200,600]、y∈[400,430]；第一角 0–40°，第二角相对加 155–195° |
| `gen2voronoi_2` | 单条自然弧，随机保持或旋转 90°，目标 2 片 | 800×800；两个方向离散选择 |
| `gen3voronoi` | 3 条弧在公共中心附近放射，目标 3 片 | 中心 x∈[200,600]、y∈[400,560]；角区间 0–40°、120–160°、260–300° |
| `gen4voronoi` | 4 条弧组合，目标 4 片 | 中心与 Gen3 同范围；角区间 0–25°、120–135°、180–200°、280–300° |
| `gen4voronoi_1_3` | 先一条分割弧，再组合 3 条放射弧；以实际连通区／数量验收 | 中心 x/y∈[200,600]；放射角区间与 Gen3 相同 |
| `gen5voronoi_1_1_3` | 保留函数实际实现为单中心 5 臂，并非按函数名机械理解为“先切 1、再切 1、再切 3” | 中心 x∈[250,550]、y∈[400,430]；基角 0–35°，其余相对角 65–80°、135–155°、208–228°、278–305° |

单位为上游 800 像素坐标。随机角与中心先采样，再经有效碎片数量、面积和连通条件接受／拒绝；**最终被接受分布不保证仍均匀**。源 mask 的 `no_erode` 不等于最终课程样本无腐蚀，腐蚀在后续独立阶段施加。

母图方面，保留 `final_dataset_generator.py` 会匹配候选母图内容纵横比，容差 0.35，可将 mask 旋转 90°匹配后再旋回输出；以母图 alpha 保留材料区域，padding 不作为材料。`label.csv` 保留 `image_name`、fragment ID、mask ID、中心位置与邻接关系。其邻接记录还会经过后续 pairwise 几何与对应检查，**不能只凭 CSV neighbor 就宣称某段腐蚀后点对仍有效**。

这里有一条复现边界：上游 snapshot 的函数、参数可以精确列出；但它不是每个早期 raw group 的逐次执行日志。原始组的实际来源以 `source_row`、fragment token、`label.csv` 和母图路径为准。本包没有重新生成原始撕碎来证明“每组都由这一版源码逐位复现”。原始 README 甚至有 Gen3–5 碎片数的过时表述，**请以实际函数／有效连通片数为准，不照抄上游 README 表格**。

### 2.2 合并基底与负例，不要误当另一种腐蚀

上游还包含 TRAIN 同母图 union 的小片组合，以及 Gen5 partition 构造。Gen5 partition 的登记模式有 `(1,2,2)`、`(2,1,2)`、`(3,1,1)`；它们描述 5 个基础碎片如何合并成连通子集合，不是腐蚀深度或训练 stage。代码见 [rachel_gen5_partition.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/reference_dependencies/staging/pairwise_v0_2/pairwise_data/rachel_gen5_partition.py) 与 [build_s7_gen5_pool.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/data_methods/reference_dependencies/experiments/rachel_n512_formal_30k/build_s7_gen5_pool.py)。交换角色的模式不能当成完全独立的几何分区重复计算。

负例来源包括同母图不相邻、同 Gen 不同母图、跨 Gen、Gen5 partition 的非相邻／跨 lineage 等。负例仍施加相同类别的形态变化，但独立在其自身 mask 上定位，**没有公共接缝 GT，也没有真实拼接位姿可画**。附图负例只是分开放置以便看形状。

## 3. 增强执行次序与几何单位

```text
Gen2–5 原始碎片 / 合法 union 或 partition pair
    ↓ pairwise 规范化、原始来源与对应关系
两片共享尺度 + 可选成对镜像       ← v14 前增强基底已归档
    ↓ 独立进入 v17 / v17.5 / v18，不串行腐蚀
条件性单端／双端自然曲线裁切
    ↓
11 类互斥配方之一：clean / Partial / weak / 一种主腐蚀 / 一种主腐蚀+weak
    ↓
除 clean 外：最终轻退化
    ↓
重建可见轮廓、继承／屏蔽监督、几何验收、记录实际参数与阶段像素
```

- 最终材料 mask 为 **800×800**；模型轮廓采样上限 **512 点／片**，粗 mask 为 **128×128**。深度的 px 是 800 画布像素，不是 512 点的序号或毫米。沿弧覆盖按几何弧长加权，不是简单数 contour token。
- 尺度：两片使用同一个比例，原请求限制 0.55–1.8；受 780 px 可容纳范围和拓扑回退约束，实际值写入 `pair_shared_scale.common_scale`。不能写成 A/B 分别随机缩放。
- 尺度同步变换坐标与位移：`p'=s·p+d_side`，`t'=s·t+d_B-d_A`，并保持点身份／对应；必要时向 identity 退回，不填洞或删掉小块来强行通过。
- 镜像：原 v14 计划 15% 成对反射，水平／垂直各半目标；两片一起反射，位移与坐标同时转换。后续筛选使课程实际比率可偏离 15%，**不是每个新版本重新独立抽 15%**。v17.5 正例组实际 435/3000，v18 为 207/1500。
- 本节成对镜像、上游母图匹配时的旋转、后来模型的 0/90/180/270° 多方向推理是**三件不同的事**。多方向推理不是本文腐蚀数据增强的一部分。

## 4. 增强方式：单独、组合与禁用组合

下表中的“单独”指**主损伤类型只有一种**；仍可能包含公共的尺度、镜像、端裁和末层 light。`clean` 表示没有主腐蚀和末层轻退化，不保证没有结构裁切。

| 配方键 | 中文含义 | 主损伤内容 | 最终 light |
|---|---|---|---|
| `clean` | 无腐蚀对照 | 无；保留端裁是否执行的记录 | 无 |
| `partial` | 局部断缺／局部接缝保留 | 结构裁切，不叠加另一种主腐蚀；可因 crop floor 跳过 | 有 |
| `mild` | 连续弱腐蚀 | 一条平滑渐进或平滑内凹深度场 | 有 |
| `wave` | 起伏退蚀 | 一种有平滑肩部的起伏深度场 | 有 |
| `wave_weak` | 起伏＋弱腐蚀 | wave 与一个连续弱场叠加，限幅 | 有 |
| `local_abrupt` | 局部突变腐蚀 | 有限弧区内的近平台深度 | 有 |
| `local_abrupt_weak` | 局部突变＋弱腐蚀 | abrupt 与连续弱场叠加，限幅 | 有 |
| `local_gradual` | 局部渐进腐蚀 | 从肩部约 1 px 平滑增加至峰附近，可叠平滑起伏 | 有 |
| `local_gradual_weak` | 局部渐进＋弱腐蚀 | gradual 与连续弱场叠加，限幅 | 有 |
| `gaps` | 离散开放缺口 | K 个沿外轮廓的缺口；不是内部随机挖洞 | 有 |
| `gaps_weak` | 缺口＋弱腐蚀 | K 个缺口＋其中一个区域的弱场，限幅 | 有 |

**没有**把 wave + abrupt + gradual + gaps 全部同时施加的配方，也没有 Partial＋另一主腐蚀的配方。`*_weak` 是确实存在的组合；末层 light 则是另一个基于后续轮廓的操作，不能与这里的 weak 混为同一层。

### 4.1 深度场的实现，而非像素噪声

在有序原轮廓上，以中心弧位置和支持长度定义局部坐标 `u∈[-1,1]`，构造连续深度 `D(u)`。缺口／弱渐进使用平滑的 `sin²` 包络；突变型在局部区间近似平台；wave 使用肩部与低频起伏；gradual 从肩部到峰值连续变化。精确公式见两个版本的 `weather.smooth_profile`。

材料是否移除依据到原边界的内向距离，代码使用 **EDT − 0.5 px** 的像素中心约定。组合场为 `min(15, D_major + D_weak)`。由于栅格化、肩部函数、两片轮廓以及裁切／末层 light 的影响：

1. 请求峰值不一定等于 `applied_max_depth_px`；表中范围是请求峰值范围，不是每个被移除像素都至少那么深。
2. 双侧拼接间隙不等于某一侧的腐蚀深度，更不等于简单把两个请求峰值相加。
3. weak 叠加须产生可测的独立移除（至少 4 个像素），主区域须有独立效果；不把“字段写了 weak、像素其实没变”当作组合案例。
4. 保持材料连通，不允许借操作新造封闭孔洞；失败候选按该版本既有规则拒绝／跳过／回退，而不是调低阈值凑配额。

## 5. 参数表：版本差异与正确分母

| 参数 | v17 新增强（不含 fallback） | v17.5 | v18 |
|---|---|---|---|
| 新端裁目标 | 原公共弧缩短 20%，像素容差 ±1 个百分点 | 若实际施加，缩短 25–40%，与预采样目标偏差≤1 个百分点 | 同 v17.5 |
| 请求裁侧 | 小面积片 70%／大面积片 30% 的组级计划 | 同；先固定侧类再尝试 | 同 |
| 单端／双端 | 原公共弧的一端或两端；不设强制 50/50 的接受比例 | 两种模式均有界尝试；可记录 skip | 同 |
| 该端裁新增面积损失 | 被裁片≤20% | ≤20% | ≤20% |
| 连续 weak 请求峰 | 3–8 px | 4–8 px | 5–8 px |
| wave / abrupt / gradual 请求峰 | 5–15 px | 7–15 px | 10–15 px |
| gaps 请求峰 | 5–15 px | 5–15 px | 5–15 px |
| gaps 数量 | K=1,2,3,4，先登记再验收 | K=1,2,3,4 | K=1,2,3,4 |
| 同侧主场＋weak 上限 | 15 px | 15 px | 15 px |
| 主损伤弧覆盖（含 weak） | 作用的可用原始支持弧≤50%；支持预算留有 44% 栅格余量 | 35–50%；场覆盖目标抽 40–46% | 40–60%；场覆盖目标抽 46–56% |
| 主区域双侧间隙峰验收 | 5–35 px；不是处处≥5 px | 5–35 px | 5–35 px |
| 最终 light 请求峰 | 1–3 px | 1–4 px | 1–4 px |
| 最终 light 覆盖 | 70%±2 个百分点；**剩余未腐蚀原轮廓**为分母，排除新裁边／主损伤弧 | 70%±2 个百分点；**主损伤后的完整现轮廓**，包括新裁边与已腐蚀边 | 同 v17.5 |
| 原始正例接缝资格 | 继承 v14 底样与 v17 约束 | 原公共弧／原较小面积片完整周长≥20% | 同 |
| 裁切阶段接缝资格 | Partial 保留弧／较小片周长≥15% | 所有结构裁切后、腐蚀前，公共弧／当时较小片完整周长≥20% | 同 |
| 最终可连接弧 | 继承对应、拓扑、区域／gap 验收 | 占**原始公共弧**≥30%；剔除结构与主腐蚀段，允许末层 light | 同 |
| 强制完全 pristine 配额 | 不在此概括为统一配额 | **关闭**；不是保留 25% 完全不动的旧方案 | 同 |
| 继承互相对应点 | 正例至少 4 个 | 至少 4 个 | 至少 4 个 |

上述连续均匀请求通常由 `rng.uniform(a,b)` 产生，上界按实现为开区间；表内连字符是范围简写。实际接受参数还受几何拒绝采样影响，不能宣称全量结果服从未经条件化的均匀分布。`clean`／`partial` 不应套用不存在的主腐蚀间隙峰；负例的公共接缝／Layout 指标应为不适用，不编造数值。

### 5.1 端裁和 Partial 的额外实现信息

端裁曲线来自 TRAIN 母图轮廓 bank 的 129 点弦法向 profile，允许翻转、反向、旋转定位。不是为每个样本画一条直线来替代；实际可见新裁边还需满足非直线性。只在原公共接缝端部截短，不把内部中断算成端裁。

v17.5/v18 同一目标／裁侧下最多尝试 96 个端裁候选，前后覆盖两种端部模式；找不到合格曲线时 `trim.applied=false`，保留原因。该 skip 不代表整个样本被丢弃，实际计数已在第 1 节列出。

legacy Partial 继承原 v14 的 end / open-middle 类型配置（原组级计划各半），中段裁切须保留两端接缝翼和材料连通；它可与新增端裁共同构成结构缺损，但不混入第二主腐蚀。后续筛选、裁切底线检查后，**不能假设最终课程 Partial 仍严格 50/50**。

### 5.2 当前深度与典型支持域的细节

v17 非缺口通常为一个主区域；local 支持长度额外受≤90 px 约束；缺口单区域支持上限受 35 px 与可用预算共同限制。v17.5/v18 普通主腐蚀允许 1–3 个区域，缺口仍按 K=1–4；区域长度由覆盖预算、可用弧和区域数共同决定，不能给所有类型写一个固定窗口长度。

末层 light 是连续弧段上的平滑场，不是“每个像素 70% 概率被腐蚀”。新课程不再保护 GT 接触附近或新裁边；`pristine_protection_enabled=false`。正例保留的最终可连接弧只允许最后 light 造成的小变化，主腐蚀（包括连续 weak）仍排除。

## 6. 采样比例：计划和实际必须分列

原 v14 的 11 类计划权重由三版继承：clean 15%、Partial 25%、mild 15%、wave 7.5%、wave+weak 7.5%、四种 local 各 3.75%、gaps 7.5%、gaps+weak 7.5%。v17.5/v18 用组级整数配额分配，所以四舍五入会出现 112/113 等轻微差异；v17_filtered 还受准入筛选与 fallback 影响。

下表单位为**pair 行数（正负合计）**，不是母图数、Gen 组数、独立原始碎片数或训练曝光。v17 的“实际新增强”只统计非 fallback 行；其余列同时保留配方标签数量，避免遗漏原样保留的旧样本。

<!-- BEGIN RECIPE_TABLE -->
| 配方 | 原计划权重 | v17_filtered 全部 | 其中新 v17 | 其中 fallback | v17.5 | v18 |
|---|---|---|---|---|---|---|
| `clean` 无腐蚀对照 | 15% | 1861 | 910 | 951 | 900 | 450 |
| `partial` 局部断缺 Partial | 25% | 2948 | 1866 | 1082 | 1500 | 750 |
| `mild` 连续弱腐蚀 | 15% | 1902 | 885 | 1017 | 900 | 450 |
| `wave` 起伏退蚀 | 7.5% | 882 | 409 | 473 | 450 | 226 |
| `wave_weak` 起伏＋弱腐蚀 | 7.5% | 920 | 402 | 518 | 450 | 224 |
| `local_abrupt` 局部突变腐蚀 | 3.75% | 454 | 208 | 246 | 226 | 112 |
| `local_abrupt_weak` 局部突变＋弱腐蚀 | 3.75% | 464 | 233 | 231 | 226 | 112 |
| `local_gradual` 局部渐进腐蚀 | 3.75% | 463 | 194 | 269 | 224 | 112 |
| `local_gradual_weak` 局部渐进＋弱腐蚀 | 3.75% | 439 | 214 | 225 | 224 | 112 |
| `gaps` 开放缺口 | 7.5% | 928 | 459 | 469 | 450 | 226 |
| `gaps_weak` 开放缺口＋弱腐蚀 | 7.5% | 918 | 428 | 490 | 450 | 226 |
| 合计 | 100% | 12179 | 6208 | 5971 | 6000 | 3000 |
<!-- END RECIPE_TABLE -->

源类别、正负细分和 generator-pair 原始计数均随 case_catalog.json（原本地资料包；未公开逐例记录） 提供。跨 Gen 负例单列为组合，不强行归到一个 Gen。union 行若未在该行直接标出单一 Gen，不凭文件名猜测；这些不影响本件所有核心图片的已记录来源。

## 7. 监督、来源和已知限制

基础 pair 带有材料坐标、相对位移与轮廓对应。增强过程中重建可见轮廓，按来源继承互相对应关系；被删掉的结构不能凭几何靠近“长出”新正匹配，新裁边也不能当成原接缝。目标数组使用非负对端索引及负值标记（无对应／ignore），具体传播逻辑见上述 frozen loader、`inherit_pair_targets`、`known_gap_links` 和版本监督入口。

**当前必须保留的限制：**已有用户审查指出 v17 / v17.5 / v18 某些腐蚀后局部对应点对标注有误。本轮工作只忠实整理已经使用过的数据和代码，没有修复这些标注，也没有全量重新认证。归档检查“文件一致、点对互反、处理流程可重放”不等价于“每个局部点对语义正确”。投稿前应补充独立逐例错误清单、修订规则、修订版本及其对实验的影响。

来源隔离的陈述也应分层：这几版新增端裁／增强使用登记的 TRAIN 基底与 donor，不引入真实评测碎片作为新增像素或裁切模板；但历史几何／分布设计参考过真实数据统计，并有真实开发集使用经历，不能因此称整个开发过程“从未看过真实数据”或把已有真实 TEST 视作全新盲测。本文没有读取真实 TEST 结果选择案例。

图库选择规则是：从 B3 最终准入 ID 中，按固定 SHA 次序、尽量覆盖不同登记 generator／母图，为每版每配方取 3 个正例；排除 v17 fallback 进入新配方核心图；Partial 必须实做。没有用模型预测、Q 分数、准确率或测试集来挑“成功拼接”图片。**这是机制示例，不是代表性随机性能样本，也不是同一个基底跨版本的控制变量实验。**

## 10. 可以写进论文的表述与必须避免的表述

可以据本文组织 Methods：自然边界驱动的基础撕碎；pair-shared 几何变化；条件端部裁切；沿弧连续、可组合且限幅的退蚀场；分阶段接缝保留条件；正负形态控制与可追溯归档。数值优先采用本表与 frozen 源码，并在首次出现时写明分母。

不要写成：

- “v17.5 是在 v17 上加一次腐蚀，v18 再加一次”；三者独立从前增强基底出发。
- “所有样本都裁掉 25–40% 接缝”；实际存在明确的 skip。
- “B3 的 v17 都采用新 3–8/5–15 px 强度”；其中存在 5,971 行 v14 fallback。
- “所有 Partial 都执行了局部裁切”；用实际 applied 字段区分。
- “70% 概率腐蚀任一像素”、“35 px 为每侧腐蚀量”、“≥30% 完全不变的接触”；均误读了定义。
- “每例有任意多种主腐蚀叠加”；实际主类型互斥，只允许指定 weak 组合与末层 light。
- “本文图是 B3 预测成功的案例／消融结果”；它们是归档仿真几何展示，不能证明模型收益。
- “这些 inherited 点对都已修复并经过独立语义认证”；独立修复覆盖层已构建但待人审、未用于B3重训。

下一步如要扩成论文，需要另行整理 B3 网络与训练协议、冻结结果口径以及真正的控制变量 ablation；不要凭此文件补造数值或因果结论。

最新模型/结果与实际直缝训练安排见 [主README](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_paper/README.md)。
