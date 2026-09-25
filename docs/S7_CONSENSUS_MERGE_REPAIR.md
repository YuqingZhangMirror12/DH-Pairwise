# S7-Consensus：同一摆放的多组支持归并修复

2026-09-25。代码版本标识：`common-pose-merge-repair/1`。

## 本次修什么

修复的是 **MD 指南新算法首版实现中的候选归并过严**，不是退回历史 S7 完整算法，也不是重新扩大所有腐蚀容差。

目标：Matcher 找到的多组对应点，如果支持同一相近 Layout，应形成一个共同位姿及真实对应并集；这些证据一起参与注意力、摆放精修和最终 Scorer。不能只选最强小接缝，不能仅对若干小接缝的分类分数做 max/mean，也不能仅删除重复候选而没有共同证据。

旧归并代码会因为极少量软对应尾部或同一端点的替代对应而否决整个归并。诊断中，甚至重复输入同一候选，也可能被判定为互相冲突。因此“已经实现 union”并不意味着 union 在实际输入上能够生效。

## 修改后的处理路径

```text
一次 Matcher / partial Sinkhorn → 对应矩阵 Q（保留绝对质量和未匹配概率）
  → 多个位移初始假设，各自拟合
  → 检查共同位姿、对应冲突、材料重叠
  → 同位姿支持组的去重真实并集 + 共同拟合
  → 在共同位姿下回读完整 Q，召回几何兼容的支持
  → 2 层 / 96 维 / 4 heads 共享证据头
  → 所有组共同精修位姿
  → 在最终位姿重新构造证据、重新编码、重新评分
  → 同一个候选对象输出 Layout 和可拼分数
```

不同的位姿仍然分别评分。缺失的接缝区间不会被补成观测证据；相隔较远的两段真实支持也不要求中间必须存在完美接缝。

### 归并判定具体变化

| 环节 | 修复后的规则 |
|---|---|
| 位姿范围 | 检查各原始假设**拟合后的中心**能否被同一个有限范围的共同中心解释，限制链式归并漂移。不是无限累积“相邻就合并”。 |
| 极少量不一致软对应 | 不再由任何一条尾部对应一票否决。共同位姿须保留每个输入假设至少 95% 的原可解释绝对支持质量；超过 5% 才拒绝。 |
| 一对多对应 | 比较同一端点在两个假设下的条件位移分布，并保留其绝对质量。相同的歧义分布不再被误认为互相冲突。 |
| 真正冲突 | 若矛盾端点的质量比例超过 5%，仍不合并；不同对应模式不能靠均值强行变成一个接缝。 |
| 重复证据 | 同一 `(i,j)` 对应只保留一次，不因重复候选增加置信度；完全相同的候选不额外重复优化。 |
| 材料重叠 | 保留原有重叠约束；去重不会使本来有重叠的候选“变成正确”。 |

这里的两个 5% 是本次显式登记的归并策略，不是数学等价补丁，也没有用真实测试集调优。绝对匹配质量、dustbin 信息、原 TRAIN 标定的法向损伤范围和精确定位模型保持不变。

## 与 Layout、Scorer 的连接

`original_union_edge_ids` 保存原支持组的真实并集，用于诊断归并是否发生。`edge_ids` 表示当前共同位姿兼容的稀疏提案支持；两者不是同一个概念。

Scorer **不只读取这份稀疏 ID 清单**。它在共同位姿下从完整 Q 重新召回支持，保留对应特征、匹配权重、其他位姿质量、未匹配质量和几何残差。证据头对这份共同证据进行联合计算，定位分支与兼容分支分别赋予权重，再共同求解位移。

精修以后，完整 Q 不变，但所有依赖位姿的证据重新计算；最终分类分数与实际输出的位置一致。没有新增第二次 Sinkhorn，也没有将各组独立分数拼接成最终分数。

## 代码位置

所有核心文件位于 [s7_consensus_v1](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1)：

- [pose_consensus_repair.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/pose_consensus_repair.py)：修复后的归并规则、真实对应并集、共同拟合和诊断记录。
- [pose_consensus.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/pose_consensus.py)：训练和推理统一入口及版本标识。
- [legacy_pose_consensus.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/legacy_pose_consensus.py)：保留首版实现，供追溯；它不是当前入口。
- [evidence.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/evidence.py)、[consensus_head.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/consensus_head.py)：完整 Q 证据召回及共同注意力计算。
- [pose_refinement.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/pose_refinement.py)、[model.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/model.py)：共同摆放精修，以及最终位置重编码、重评分。
- [migration.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/migration.py)、[train.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/train.py)：显式断点迁移、重新仿真选模与训练。

与正在修复的首版相比，Patch Encoder、Context、Matcher、Sinkhorn、Scorer 容量、loss 系数和数据没有一起更改。这个仓库此前尚未包含 Consensus 主线，因此本次提交也补入运行和测试所需的模型源码及依赖；“新增文件多”不表示本次又改了所有模块。

## 验证了什么

本地与远端 CPU 套件共 154 项测试。关键行为测试是 [test_merge_joint_path.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/test_merge_joint_path.py)：

1. 四组断开的局部支持共同归并，第五组约 200px 以外的竞争位姿仍独立。
2. 四组分别出现在初始注意力证据、共同精修权重和最终位姿证据中。
3. 最终分数对每一组的局部特征均存在非零梯度。
4. 在相同初始位姿下移除任意一组，会改变共同精修结果。
5. 最终分数等于在最终摆放位置重新构造证据后的分数，而不是旧位置的分数。

其他测试覆盖重复对应不增加质量、一对多歧义、冲突位姿、材料重叠、链式漂移、A/B 互换、padding、梯度及迁移预算约束。运行：

```bash
python -m unittest discover \
  -s experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1 \
  -t . -p 'test_*.py'
```

这些是实现正确性证据，不是训练后准确率提升的证明。新模型的真实数据收益须等待冻结评估。

## 重训及选模边界

- 历史 M12 分支：冻结历史 S7 M12 Matcher，新共识头按原初始种子重新训练，旧头结果保留但不继续使用。
- 从零分支：保留随机初始化后已训练到第 26 轮的 Matcher、优化器、RNG、19,500 次更新及 624,000 对曝光；不把它换成 M12。用修复后的解码器重新评估实际保存的 E2/E4/E6/E22/E26，之后继续原预算；仿真选中的 Matcher 冻结后训练同样的新头。没有保存的历史轮次不伪造重建。
- 旧候选影响过仿真选模与学习率调度，因此不会静默沿用旧选择记录。迁移保留现有学习率和已经消耗的两次降学习率；在新解码器的当前基线重新开始无改善观察，Matcher 总上限仍为 48 轮。新头仍按原 16–48 轮平台规则训练。
- 每支两张 GPU，每卡 microbatch 8、累积 2、有效 batch 32；FP32。30K 数据不重新生成：TRAIN 24K，CAL 1.5K，SELECT 1.5K，TEST 3K。
- CAL 校准阈值，SELECT 选择权重；TEST 仅冻结后使用，真实数据不挑 epoch。新旧缓存、源代码绑定和选择记录不混用。

旧头已消耗的计算量不算作新头训练曝光，因此两次实验不能声称总计算成本完全相等。源码发布不包含数据、权重、人工标注、私人文档或服务器凭据。完整远端启动还需要外部检查点、数据契约、几何标定和显式迁移清单；仅 clone 仓库不能复现所有私人数据实验。

## 尚未证明的事项

这次取消不合理的严格归并拒绝，不保证所有短接缝都应合并。端点分布的一、二阶矩冲突检查仍是近似，不是对所有多峰形状的严格判定；长直边歧义、上游 Matcher 漏提案、错误的高质量对应仍可能失败。需要在新训练后分别检查归并覆盖、误合并、正确候选排序、分类漏判及最终 Layout，而不能只用候选数减少作为成功标准。
