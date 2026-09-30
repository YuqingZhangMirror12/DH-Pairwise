# Matcher v2：实现、验证与运行入口

这版只扩展 Matcher 的特征和 affinity 计算；一个 FP32 Sinkhorn、T16 complete-link、
精确点对并集及两个轻量 Scorer 的接口保持不变。默认关闭 v2。原659个正式基线 Python
文件按原 SHA 保留，不覆盖正在运行的任何版本。

## 与提供的 in-tree 版本比较

两版都有 U1 两层环形弧长 RoPE 自注意力、U3 同步双向点级交叉注意力、
U4 初值1/上限5的增益、U2 concat残差与四尺度晚融合；新增参数均为410,213。
我们的实现使用独立 adapter/config，而参考版使用旧网络配置的子类及树内扩展。
这属于代码组织差异，不能据此声称我们的架构更好。

借鉴参考版的完整尺寸回归方法，新增同输入/同权重逐项对照：10组800画布、N512样本，
两版零初始化都与旧 E32 完全一致；映射非零新分支后，输出最大差2.861e-6、
梯度最大差1.043e-7。真实6例只做前向；反向只用4例仿真。
这证明实现接近，不证明任务准确率提高。参考版的300步 CPU pilot 训练预算、可训练参数
和学习率不同，不并入 B1–B3 成绩。

保留我们的额外保护：无效/NaN padding 清零、有效点环绕而非512存储环绕、
空轮廓有限输出、下游新头随机流不被 v2 构造改变、完整模型/AdamW/各rank RNG恢复验证。
CUDA实测还修复了严格确定性模式下的前缀求和兼容问题，以及逐rank梯度审计目录创建；
没有关闭确定性要求。183项CPU测试与实际双卡12更新/断点恢复验证均已通过。
抗滑动损失仍不进入主实验，避免把结构实验与损失变化混在一起；matchability默认关闭。

## 文件导航

| 部分 | 文件 |
|---|---|
| 网络与六字段接口 | `network.py`, `adapter.py`, `spec.py` |
| 原始曝光全部保留、另加直缝更新 | `additive_exposure.py`, `runtime_schedule.py` |
| 数据准入与不可变执行计划 | `prepare_data.py`, `data_runtime.py`, `compile_execution.py` |
| 原生损失、DDP、断点、选模 | `model_runtime.py`, `execution.py`, `runtime_io.py`, `validation.py` |
| GPU短测及完整实验 | `launcher.py`, `pipeline.py` |
| 冻结终评与候选诊断 | `population.py`, `evaluate.py`, `evaluation_controller.py`, `terminal.py` |
| 对照外部参考 | `compare_intree_reference.py` |
| 经用户单独授权的旧任务暂让双卡 | `joint_priority.py` |

## 安全执行顺序

1. 从原正式 B0 源组成独立源码快照，核对659个基线文件与资产 SHA。
2. 运行 `verify_runtime_cpu`。CPU成功不替代 CUDA 验证。
3. `compile_execution` 生成新目录的计划。B2使用原已准入数据；B1/B3必须先通过新数据准入。
4. GPU入口默认只接受明确分配的空闲卡。先用 `launcher --gate-only`，双卡各micro8、
   累积2、有效batch32，真实训练12次更新，另做update1到12恢复。
   比较完整模型、AdamW、采样位置与各rank RNG；逐rank记录新模块梯度及显存。
5. 正式训练使用丢弃短测权重后的全新种子起点，最后必须成功退出并完成冻结评价。
   不能把launch、进程消失或短测完成当作正式训练完成。

命令包前缀：
`experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_v2_v1`。
例如在已组成的源码快照根目录：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m \
  experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_v2_v1.verify_runtime_cpu \
  --source-root . --output-new ../cpu_preparation.json
```

训练需另行提供数据、原来源划分/几何协议和架构参考checkpoint；仓库不发布这些样本、
权重或人工标注。路径与 SHA 必须据实际文件编译，不能把原服务器绑定改字串后直接运行。

## 预算和结论边界

B0/B2原24,000更新；B1/B3保留全部原曝光并另加7,667直缝更新，共31,667。
新增数据不会挤掉原数据。每个 Matcher 的两个新 Scorer 独立训练，冻结该分支自己选出的
Matcher。不同预算的比较不能被写成单独的数据效应。

新分支仅敦煌开发折可用于真实选模：fold1 CAL，fold2/3/4 SELECT；fold0和Turufan只做
预注册终评。真实数据不反传，Turufan Layout/Joint为null。B4–B6不自动启动。

最新实际验证状态见 `IMPLEMENTATION_STATUS.md` 及发布包中的带 SHA 回执；本 README
不是“训练已收敛”“准确率已提升”或“新数据已准入”的声明。
