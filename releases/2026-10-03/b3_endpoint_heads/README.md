# B3 终点配套 Scorer：双卡训练代码与新 SELECT 扫描

这是论文阅读与复核用的独立代码快照，不覆盖历史 B3、原 runtime13 或标签修复候选。无 SSH 的对话可直接阅读所有链接。**新 Scorer 尚无完成后的效果数字，不能把旧头结果写成重训结果。**

## 三条实验必须分开

| 实验 | Matcher | Scorer | 已知状态 |
|---|---|---|---|
| 历史 B3 | 旧仿真 SELECT 选出的 U9667 | 原 Patch / Stats | 原训练与终评已完成 |
| 终点替换推理 | 用户指定原 B3 U31667 | 沿用在 U9667 上训练的原 Patch | 新 CAL＋单/四方向真实 TEST 已完成；见旧论文交接 |
| 本次 fresh-head | **固定原 B3 U31667，不再更新 Matcher** | 分别重新初始化 Patch / Stats | 15:48 启动核验：各双卡门控通过，正式训练进程已启动；尚无最终结果 |

新 SELECT 扫描是另一项已完成的**原检查点重新评价**：16个原 Matcher 检查点各推理1,596对，冻结规则选中 **U29667**。它不是本次用户指定的 U31667，不能据此自动更换正在训练的 Matcher 或再开一组头。

[完整 SELECT 曲线及各阶段统计](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_endpoint_heads/docs/SELECT_SCAN.md) · [可审核汇总 JSON](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/evidence/select_scan_aggregates.json) · [既有 B3 架构及真实结果](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/ec7091ae9d51e1f4da8ede4a388c427404441d94/releases/2026-10-03/b3_paper/README.md)

## 本次固定训练设计

| 项目 | Patch | Stats |
|---|---|---|
| 物理 GPU 分配 | 0、1 | 2、3 |
| DDP world size | 2 | 2 |
| 每 rank microbatch / accumulate | 16 / 1 | 16 / 1 |
| 有效全局 batch | 32 | 32 |
| 原定正式更新预算 | 31,667 | 31,667 |
| 头初始化 | 新随机初始化，未导入旧头或 optimizer | 同左 |
| Matcher | 原 B3 终点 U31667，冻结参数与 eval 模式 | 同左 |
| 头选轮次 / 标阈值 | 新 SELECT1596 / 新 CAL1587 | 同左 |

原 B3 TRAIN、原标签、课程/直缝顺序、seed、loss、LR计划和更新预算保持不变；只替换为终点 Matcher 并重新训练头。**Task3 FULL/TIGHT 标签覆盖层仍待人审，未用于这一轮。** 两头是独立替代分支，不是分数融合后的双头。

Matcher 文件 SHA256：`01c7070242c943861c1d44a6c5df4a5b463bb369a26439f9fbb053ea1faf6ac1`。

原门控为12次更新、update1保存、恢复至12次更新，并核对双rank梯度和完整模型/AdamW/各rank RNG的一致性。15:48证据中两头门控均真实return0，随后由同一有限pipeline启动正式训练。**门控12步不计入正式更新。** 当时首次formal/status尚未写出，所以公开的启动记录将正式更新数记为`null`，不是0或已完成。

## 代码入口（固定版本，不需要 SSH）

新增代码固定在提交 `be480140168cfbf102a0683292b9ba3a61af3c90`。`endpoint_head_source03` 的17个生产模块与已部署冻结副本逐字节相同；单独附上7项双卡路由测试源码。`select_scan_source01` 是实际完成扫描的独立旧源，不用可变开发目录替代。

| 目的 | 固定代码 |
|---|---|
| 指定原终点、拒绝混入未准入标签或旧optimizer | [head_endpoint.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/code/endpoint_head_source03/model_selection_v2/head_endpoint.py) |
| 加载冻结Matcher、初始化新头、绑定新SELECT/CAL | [head_bridge.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/code/endpoint_head_source03/model_selection_v2/head_bridge.py) |
| 原训练循环接入、保持TRAIN与全局batch | [head_execution.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/code/endpoint_head_source03/model_selection_v2/head_execution.py) |
| 双卡命令、GPU门控与正式接续 | [head_launcher.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/code/endpoint_head_source03/model_selection_v2/head_launcher.py)、[head_pipeline.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/code/endpoint_head_source03/model_selection_v2/head_pipeline.py) |
| 完成验收、已有观察复用、必需终评与新CAL阈值 | [head_terminal.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/code/endpoint_head_source03/model_selection_v2/head_terminal.py)、[head_terminal_queue.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/code/endpoint_head_source03/model_selection_v2/head_terminal_queue.py)、[head_readout.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/code/endpoint_head_source03/model_selection_v2/head_readout.py) |
| 已完成的新SELECT候选扫描 | [checkpoint_scan.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/code/select_scan_source01/model_selection_v2/checkpoint_scan.py) |
| 扫描前冻结的选择规则 | [protocol.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/code/select_scan_source01/model_selection_v2/protocol.py)、[selection.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/be480140168cfbf102a0683292b9ba3a61af3c90/releases/2026-10-03/b3_endpoint_heads/code/select_scan_source01/model_selection_v2/selection.py) |

原网络、训练循环与候选提取仍依赖 [原样保留的runtime13](https://github.com/YuqingZhangMirror12/DH-Pairwise/tree/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13)。本目录不是包含私有数据、模型和环境的一键复现包；不可将多个历史source混入同一个运行目录。

## 后续效果应怎样比较

先在同终点Matcher、同单方向推理下比较旧Patch与fresh Patch；Stats独立列示。原“终点＋旧Patch”的四方向结果可作系统参考，但不能把旋转集成收益计入重新训练头的收益。新CAL先封存阈值、之后再用TEST；不以TEST选epoch/阈值。Turu没有Layout GT，Layout/Joint必须为`null`。

有限pipeline包含原预算训练与必需终评，没有额外资源等待器或失败自动重试。必需评价涵盖仿真TEST3000、敦煌留出TEST161（已有开发639记录复用）、吐鲁番602和严格直缝TEST900；SIM/REAL两种头选模口径分开，主论文结果须标明选模口径。此处说明的是已部署流程，不是声称这些新头结果已经产生。

## 离线轻量核对与公开边界

下载目录后运行 `python analysis/verify_public.py`：只核对发布文件SHA/语法、16个候选聚合数字与冻结规则，**不访问GPU/网络，也不重新证明未公开的逐对预测**。全量逐对报告和实际返回已由所有者另行完成一次只读验收，公开汇总保留其SHA。

公开包不包含原始图像/NPZ、权重、缓存、逐例预测、pair/fragment ID、人工审核图、源注册清单或SSH连接信息。没有清理/改写远端数据或训练源。此前论文、许可和代码快照均保留。
