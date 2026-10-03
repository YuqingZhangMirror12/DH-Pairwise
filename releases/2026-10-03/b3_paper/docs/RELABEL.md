# B3 正例对应标签修复 v2：审核交接

2026-10-03。**独立全量标签构建及完整性核验完成；待 Yuqing 可视化审核，不允许直接用于训练。**

## 1. 交付范围

- 精确覆盖旧 B3 TRAIN 的27,179对：12,179正例另存FULL/TIGHT标签候选，15,000负例引用原文件、无覆盖层。
- 仅改变正例 target_a/target_b；旧sample、mask、轮廓、GT、原始Rachel/基础库、源文件与清单均不覆盖。E32当前训练、新SELECT/CAL和旧B3权重/结果未改。
- 本次为6个单线程nice10 CPU worker，GPU使用0、模型前向0、优化器更新0；全量子进程真实return0。
- 502例真实正例画廊已交给所有者本地审核；图像与逐例身份不在公开仓库。
- 全量阶段/配方统计（所有者已有本地证据，未公开逐例身份）；完成与完整性回执（所有者已有本地证据，未公开逐例身份）；低支持正例清单（所有者已有本地证据，未公开逐例身份）。

“构建完成”不等于全部标签已经人审，更不等于重训性能改善已被证明。

## 2. 根因与旧标签证据

**遗漏的对应被误当成强负监督**：rachel_training_dataset.py:401–405先将padding设−2，再把所有有效token设−1，最后写入稀疏对应。上游rachel_preprocess的_accepted_continuous_seam和_token_correspondences有意保守；缺少对应不能自动解释为确定无对应。

**来源继承没有最终跨片间距限制**：inherit_pair_targets继承旧匹配；rachel_s7_dataset.py:89–91中的source_arc_ancestry(...,30.)是同片新旧边界的来源搜索半径，不是GT位置下A–B距离。wave允许继承，却没有新增5/8px及破坏区过滤。

按清单登记fragment身份抽146对基础样本，A距B token≤3px的19,801点中，已有对应9,518点（48.07%）；未对应10,283点，其中8,398点（81.67%）有≤3px互近配对的几何可能。这与任务MD的“约44%”不是完全相同的数字/抽样：本次没有从目录中推断“最近的两片”，而是使用声明的fragment IDs、correspondence_path。互近可能仅用于诊断，不能直接据此填满标签。146对身份/SHA及读数（所有者已有本地证据，未公开逐例身份）。

全量旧B3正例含 **11,614对GT间距>8px的旧对应**。FULL、TIGHT均清为0；不改基础库原始对应文件。

## 3. 实现规则与对MD的必要澄清

### FULL：来源锚定重建

1. 默认−2。仅来源可靠、非共享接缝、非破坏区且距另一片最终密集轮廓>15px的无关外边界为−1。
2. 从未腐蚀proof/preweather mask找原始共享边界：源距离≤3px、源弧连续≥8px；不依赖旧稀疏点标签，不跨不相连源弧补桥。此3px是源几何判断，不是旧strict的最终标签门槛。
3. 完好段用最终token互近；后退段沿原始mask内法向找保留像素，再在原始来源坐标一对一配对，不以最终轮廓最近点沿缝滑移补标。近似追溯残差、分支歧义不可靠则忽略。
4. 最终≤5px为主体；5–8px仅源锚定的平滑后退、两侧相邻候选gap变化≤1.5px且±2token无重叠。缺少邻居不猜测，>8px一律−2。
5. local/notch/seam_gaps、partial新切边、trim及重叠按源记录排除；local/gaps肩部至少8px，裁切及strict区间至少3px。破坏保护传播到同一原始接缝两侧。
6. 每段原始接缝两端各4个实际最终token保护，除非≤3px且两侧完整。4个token不恒等于15px，原MD“约15px”仅是示意。
7. strict浅单侧缺口用明确保守参数：记录峰值≤5px、无双侧区间重叠，肩部仍排除；仍需最终5/8px条件。它不恢复旧strict最终3px规则。
8. 一对一、互反、源弧与最终轮廓顺序单调，并显式排除交叉线段；密度不一致的剩余点、端点歧义点都是−2。

### TIGHT：不新增匹配的对照

不改旧伙伴身份，仅删除>8px、破坏区及无法可靠追溯来源的旧匹配；不能证明为无关外轮廓的旧−1改−2，不新增匹配。较MD的极简“只收紧”多了来源不确定保护，未来ablation必须如实说明。

TIGHT不等同FULL：保留的5–8px旧匹配不被声称全部通过FULL的新平滑/重配对规则。两版本均未降低5–8px损失权重，旧loss未改。

### 旁路字段与坐标

- latent是已有源接缝的诊断记录，覆盖受旧稀疏锚点限制，不是完整点级真值。新标签使用原mask/proof来源，latent用于核对GT、gap、recession、保留像素；缺失信息不猜对应。
- strict的final_parent在旧v4.2最大4连通分量清理之前保存。只读复现该适配后，必须与归档最终mask逐像素相等，未改mask。
- 新版纠正了v1的过度忽略：不能把平滑token进入对片mask、或原始亚像素GT的窄栅格交叠当作“增广重叠损伤”。v2保留所有显式strict重叠区，额外几何保护只检查相对原始基底新增的实际mask交集并扩±2token。非strict已验证只减材料、GT不变，因此不会新增重叠。已把原mask完全不变但有亚像素栅格接触的clean情形纳入新增回归；历史v1输出保留、不推荐训练。
- 两个pilot文件在float32半像素位置有读回舍入分歧；验证落点只允许±0.0002px浮点包络，不放宽为1px几何容差。
- 来源缺失/矛盾正例保守隔离为全−2并单列，不默默回退旧匹配。来源隔离不是训练准入。

冻结参数：[relabel.py / Policy](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/correspondence_relabel_v2/relabel.py)。

## 4. 全量修改前后

“对应数”按A→B点对只计一次，不是图像对数。

|阶段|正例数|旧对应|FULL|TIGHT|旧>8px|FULL>8px|FULL 5–8px|FULL<4对正例|
|---|---|---|---|---|---|---|---|---|
|v17_filtered|4679|133643|259338|131116|2093|0|1622|42|
|v17.5|3000|112662|194931|106337|5894|0|3231|29|
|v18|1500|53710|92574|49864|3627|0|1739|16|
|straight_seam|3000|174950|216294|169984|0|0|10004|1|

FULL共763,137对：287,051对保留旧训练版本同一伙伴，476,086对为新来源锚定标签。“旧/新”指旧训练标签版本，不声称旧标签全是人工真值。

**不可直接通过原训练准入的问题**：FULL有88正例不足4对，其中27例0对；TIGHT有9例不足4对，其中0例0对。没有为凑门槛伪造标签，二分类label和GT不变。原“正例至少4对”训练准入门槛没有放宽。是否过滤这些样本、怎样保持对照样本和预算可比，待人审后另定；本轮不改训练。

低支持并不只出现在损伤配方：全量v18/clean仍有1例、strict有1例不足4对，均保留待审。固定抽样clean没有低支持，不能外推为全量clean全部通过。画廊是502例分层抽样，不是12,179例逐一人工验收。

来源证据隔离正例：0例，原因/ID见完整回执。全−2不代表已获人审认可。

## 5. 验证和证据边界

- 17项合成CPU回归本地/远端通过（本修订新增2项：基底亚像素接触不误作损伤、新增真实重叠仍保护）。深腐蚀、平滑5–8px、肩部、无损侧字段缺失、错误GT/latent、负例、reciprocity与坐标适配均有测试。
- 固定514实样本（502正+12负），正例每阶段至少120并加22个v14 fallback控制；非label数组逐一不变，负例全数组不变。
- 两候选版本各514例经实际load_sample、ReviewOverlayDataset、collate、check_batch，共1,028次；全量正例也逐条做同一CPU标签适配检查，无网络前向。
- 真实return0之后核对12,179份overlay及audit实际SHA；清单、统计、协议SHA均匹配，15,000负例无overlay。
- FULL所有对应≤8px；5–8px记录为源锚定平滑后退；已识别破坏/肩部区无对应；每例一对一互反与单调检查通过；A距另一片密集轮廓≤3px的点中−1=0。这不等于所有近点都补成对应，未知点仍为−2。
- 502例画廊筛选、搜索、图片渲染通过浏览器验证，已抽看真实案例。尚未经Yuqing最终人审、尚无新标签重训效果。


## 6. 公开实现

- [relabel.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/correspondence_relabel_v2/relabel.py)
- [overlay_dataset.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/correspondence_relabel_v2/overlay_dataset.py)
- [catalog.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/correspondence_relabel_v2/catalog.py)
- [build_full.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/correspondence_relabel_v2/build_full.py)
- [test_relabel.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/correspondence_relabel_v2/test_relabel.py)

**training_admitted=false**。本次只发布代码和方法，不授权过滤低支持、改预算或重训。
