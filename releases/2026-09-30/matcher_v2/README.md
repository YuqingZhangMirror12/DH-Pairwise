# Matcher v2：代码与实际双卡验证（2026-09-30）

已完成网络改造、训练/评价接入、与用户提供的 in-tree 参考实现对照，以及实际双卡短测。
**这是一份可运行性验证后的代码发布，不是正式训练结果或准确率提升报告。**
没有改写现有训练源码，也没有把短测权重作为正式模型。

## 比对后保留、吸收了什么

| 项目 | 处理与原因 |
|---|---|
| 整轮廓 self-attention、环形弧长 RoPE、同步双向 cross-attention | 两版核心设计一致；保留独立 adapter/config，旧配置默认不启用v2 |
| concat残差、四尺度晚融合、增益初值1/上限5 | 与参考逐项核对；新增410,213参数、68个张量；没有重复堆两套相同模块 |
| 完整800画布/N512对照 | 吸收参考的回归方法，增加同输入/同权重的数值与梯度比较；不仅检查能否导入 |
| 严格确定性CUDA与完整断点恢复 | 实测发现并修复CUDA前缀求和兼容及梯度记录目录问题；不关闭确定性要求 |
| padding、空轮廓、随机流 | 保留无效点/NaN保护、有效轮廓环绕、空轮廓有限输出和新头初始化随机流隔离 |
| 参考的可选抗滑动损失 | 暂不并入B1–B3；原损失不变，避免同时改变结构和损失而无法归因 |

两版用10组完整尺寸输入比对：6组真实输入只前向，4组仿真用于反向。
零初始化都与旧E32逐值一致；映射相同非零新增权重后，输出最大绝对差
`2.8610e-6`，梯度最大绝对差`1.0431e-7`。这是实现一致性证据，**不是性能提升**。
参考的短CPU训练配方不同，没有当作B1–B3的对照成绩。

## 验证结果

- 服务器研究源码：183项CPU测试，本地/远端均通过，无跳过。
- 发布包：另有独立CPU测试回执，见[公开包验证](cpu_verification.json)。测试使用合成数据。
- 双卡真实训练：FP32，800×800/N512，每卡micro8、累积2、有效batch32。
- 连续12更新/384曝光，与“保存第1更新，再恢复到第12更新”完全一致；模型、AdamW、两rank随机状态均核验。
- 每rank所有111个可训练参数张量都有有限梯度；从第2更新起各新增分支均有非零总梯度；未训练Scorer保持不变。
- 峰值allocated约6.61GiB，reserved约8.44GiB，卡容量约23.52GiB；未OOM。
- 短测权重未进入正式训练，新的正式更新数仍为0。原联合训练由保存的完整checkpoint接回。

完整摘要和原始回执SHA见[verification.json](verification.json)。两个失败前测保留在服务器：
第一个是严格确定性CUDA不支持原cumsum路径，第二个是审计目录缺失；均已修复，未计为成功。
短测不是长时间稳定性/收敛证明，也不是其他分支或两个新Scorer的GPU准入。

## 代码入口

源码在独立的[source/](source/)目录，避免把历史模型绑定到错误版本。公共包中保留的
**网络、训练和评价实现与GPU测试版字节一致**；仅一个单测fixture改为纯合成案例标识，
不再依赖私有人工审核案例。原659个生产基线文件仍按SHA逐字节保留。
公开包和服务器包含文件集合不同，不能混用两者的source-binding SHA。

主要包前缀：`experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919`。

| 文件/包 | 用途 |
|---|---|
| [matcher_v2_v1/network.py](source/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/network.py) | self/cross注意力、RoPE、多尺度残差、增益、确定性前缀求和 |
| [matcher_v2_v1/adapter.py](source/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/adapter.py) | 接入旧Matcher；单次FP32 Sinkhorn、六输入接口不变 |
| [matcher_v2_v1/README.md](source/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/README.md) | 模块说明、全部入口与预算 |
| [matcher_v2_v1/PROTOCOL.md](source/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/PROTOCOL.md) | B0–B3设计与选模边界 |
| `matcher_v2_v1/compile_execution.py`、`execution.py`、`launcher.py` | 不可变计划、训练、12更新/恢复门控；`--gate-only`不启动正式训练 |
| `matcher_v2_v1/evaluation_controller.py`、`terminal.py` | 新架构重载和独立冻结终评 |
| `straight_seam_v42_reuse` | 已批准的v4.2几何复用、合成来源白名单、监督修订及独立审计 |

Python3.10/3.11；实际远端PyTorch2.5.1+CUDA12.4，本地CPU2.2.2。依赖见仓库根
`requirements.txt`。从仓库根运行：

```bash
cd releases/2026-09-30/matcher_v2/source
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m \
  experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_v2_v1.verify_runtime_cpu \
  --source-root . --output-new ../my_cpu_verification.json
```

完整训练还需要私有数据、原曝光账本、来源划分、几何协议与新目录执行计划。
不能从仓库示例路径直接启动或覆盖旧任务。`joint_priority.py`是用户授权的服务器专用
暂停/接续器，不是通用启动入口；其他服务器不要直接运行。

## 数据已生成，但尚未正式混入训练

| 划分 | M | J | R | 总量 |
|---|---:|---:|---:|---:|
| TRAIN | 1,200 | 2,400 | 2,400 | 6,000 |
| SELECT | 200 | 400 | 300 | 900 |
| TEST | 200 | 400 | 300 | 900 |

每一类各自正负1:1；J子类型每个标签严格80%/12%/8%。三个独立种子、独立来源池。
7800对生成、逐像素/GT/监督审计完成；跨划分来源标识和模型输入重复数均为0。
直接生成素材限制为仿真Voronoi碎片，R为程序条带，未把真实评估碎片作为增强素材。
但跨库同写卷别名映射仍未完全证明，因此不能声称所有层级的来源泄漏已被排除。
官方loader去重、总体几何验收、E32仅SELECT校准和最终来源准入尚待完成；当前
`training_admitted=false`。TEST不参与调参、选择模型或定义样本难度。

三个参考生成器模块和参考测试数组由用户另行提供，未在公开仓库再分发：
`gen_straight_seam.py`、`gen_straight_seam_v3.py`、`gen_straight_seam_v4.py`。
复用wrapper用`--reference-dir`接收它们；生成还需已审计的仿真源。
带真实素材的参考NPZ、E32权重、原始图像、人工标注及案例清单均不在包中。

## 接下来的实验与边界

B0保留原结果；B1原Matcher+直缝数据；B2新Matcher+原数据；B3新Matcher+直缝数据。
按用户要求，B1/B3保留原24,000更新的全部曝光并额外增加7,667直缝更新，共31,667；
B0/B2仍24,000。新增数据不挤掉原曝光，但“加数据”对比也增加了训练量，不能写成纯数据效应。
每个新分支的Patch/Context和Q/几何Scorer使用该分支自己的选定Matcher，各训新头。

新分支真实选模只用敦煌来源隔离开发折，真实数据不反传；Turufan及保留TEST仅终评。
Turufan无Layout GT，Layout/Joint保持null。B4–B6及其他损失变化不自动启动。
是否改善分类和Layout，必须等待这些正式同条件对照完成。
