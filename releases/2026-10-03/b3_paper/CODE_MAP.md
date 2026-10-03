# B3 paper code map — 2026-10-03

所有代码链接固定到 `9dff51b32dc73b6a915c6954f6ecd40e48708d37`。阅读不需要SSH；执行完整训练/推理仍需要未公开的图像、权重、源绑定清单与第三方依赖。本次是**可阅读、可定位、可核对的代码/证据发布**，不是附带全部数据的一键复现实验包。

| 范围 | 固定入口 | 应如何使用 |
|---|---|---|
| B3实际Matcher V2 | [network.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/network.py)、[adapter.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/adapter.py) | Sep30 runtime13原字节，勿用其他早期副本覆盖 |
| 候选/T16 | [pose_consensus.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/pose_consensus.py)、[threshold_builder.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1/threshold_builder.py) | 确认冻结alias指向T16 |
| Patch/Stats | [head.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/binary_scorer_v1/head.py)、[loss.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/binary_scorer_v1/loss.py) | 两种替代头，非融合双头 |
| 训练/直缝交错 | [runtime_inputs.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/runtime_inputs.py)、[runtime_schedule.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matcher_v2_v1/runtime_schedule.py) | 实际31,667更新；WINDOWS按原24,000时钟 |
| 直缝包装器 | [generate.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-09-30/matcher_v2_runtime13/runtime_work_13/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/straight_seam_v42_reuse/generate.py) | 外部v4.2私有生成器不再分发 |
| 最新四方向+CAL/TEST | [core.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/rotation_ensemble_v1/core.py)、[endpoint_calibration.py](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/rotation_ensemble_v1/endpoint_calibration.py) | exact endpoint_cal_source02，对应完成protocol各文件SHA |
| B3终评根模式修复 | [独立修复入口](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/releases/2026-10-03/b3_paper/code/terminal_repair_source_03/entry.py) | 仅eval保护，不是重训或放宽错误验收 |
| v17/v17.5/v18增强 | [方法与参数](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_paper/docs/SIMULATION_METHODS.md) | code/data_methods保留原相对目录；依赖及数据另行提供 |
| 新SELECT/CAL | [生成/发布/配比](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_paper/docs/SELECT_CAL.md) | source07像素生成和source03清单发布分开，不修改旧B3 |
| 标签修复v2 | [方法与未训练边界](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_paper/docs/RELABEL.md) | 独立target覆盖层，待人审 |
| 基线适配 | [PairingNet](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/staging/pairwise_v0_2/baselines/PAIRINGNET_RACHEL_ADAPTATION.md)、[ShreddingNet](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/staging/pairwise_v0_2/baselines/RACHEL_SHREDDINGNET_BENCHMARK.md) | mask-only任务适配，非官方完整recipe复现 |

## 离线阅读与轻量核验

下载仓库后，以release为当前目录运行：

```bash
python analysis/verify_public.py
```

仅Python标准库；核对191份原样代码的SHA、语法和51行新增汇总算术。它不访问网络、SSH、GPU、权重或原始数据，也不声称重新证明预测语义。

`analysis/extract_evidence_private_inputs.py`是原保存预测重计数逻辑，需所有者原私有目录，公开包中仅供阅读，不能直接运行得到缺失数据。

## 发布边界

未发布：图像/NPZ/proof/target、训练权重、缓存、逐对预测或pair ID、人工标注、服务器身份/凭据、未获公开再分发许可的上游base_tearing/v4.2生成器和vendor。已有本地111例资料包未删除。上游许可范围继续见 [THIRD_PARTY_NOTICES.md](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/9dff51b32dc73b6a915c6954f6ecd40e48708d37/THIRD_PARTY_NOTICES.md)。

历史源码可能保留原实验绝对路径或对外部依赖的引用，作为冻结实现保留，不表示写论文需要服务器权限，也不承诺它们在任意目录无需配置即可执行。读代码优先按本地图，不要将多个发布快照混入同一个运行目录。

## 2026-10-03 下午的独立追加

[终点双卡新头代码与实验边界](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_endpoint_heads/README.md) · [新SELECT扫描结果](https://github.com/YuqingZhangMirror12/DH-Pairwise/blob/matcher-v2-20260930/releases/2026-10-03/b3_endpoint_heads/docs/SELECT_SCAN.md)。追加代码固定提交 `be480140168cfbf102a0683292b9ba3a61af3c90`；上表原代码链接和字节保持不变。新头未完成，不用旧头指标代填。
