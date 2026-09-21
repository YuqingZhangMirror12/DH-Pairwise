# 代码导航

以下链接均相对仓库，可在 GitHub 或本地 clone 内使用。

| 组件 | 实现 |
|---|---|
| Mask、contour、Patch 抽样和主体 Matcher | [rachel_n512.py](../staging/pairwise_v0_2/models/rachel_n512.py) |
| 共享 Patch CNN | [local_matcher.py](../staging/pairwise_v0_2/models/local_matcher.py) |
| Partial Sinkhorn 与 dustbin | [optimal_transport.py](../staging/pairwise_v0_2/models/optimal_transport.py) |
| 无旋转布局解码 | [translation_layout.py](../staging/pairwise_v0_2/models/translation_layout.py) |
| Matcher loss | [rachel_n512_loss.py](../staging/pairwise_v0_2/training/rachel_n512_loss.py) |
| S4/S6 等独立 CA 分类头 | [rachel_decoupled_score.py](../staging/pairwise_v0_2/models/rachel_decoupled_score.py) |
| 早期 all_tokens / matched_tokens / matched_edges 对照 | [matched_only/model.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/model.py) |
| edge_seed / edge_multi 与训练 | [matched_only/train.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/train.py) |
| 选择最终内点端点、C1/C2 | [candidate_local/model.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/candidate_local/model.py) |
| 多个分离位移候选 | [candidate_groups.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/candidate_groups.py) |
| 缓存 | [cache.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/cache.py)、[stage_cache.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/stage_cache.py) |
| 六输入模型组合与评估 | [inference.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/inference.py)、[evaluate.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/evaluate.py) |
| S7 数据与强腐蚀 | [rachel_s7_dataset.py](../staging/pairwise_v0_2/pairwise_data/rachel_s7_dataset.py)、[rachel_strong_weathering.py](../staging/pairwise_v0_2/pairwise_data/rachel_strong_weathering.py) |
| S7 固化生成入口 | [materialize_s7_training.py](../experiments/rachel_n512_formal_30k/materialize_s7_training.py) |
| 分阶段训练 | [train_score_decoupled.py](../experiments/rachel_n512_formal_30k/train_score_decoupled.py) |
| G0/G1 项目实验实现 | [scorer_feature_adaptation_v1/model.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/scorer_feature_adaptation_v1/model.py) |
| dustbin 解码诊断 | [probe.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/dustbin_probe_v1/probe.py) |
| PairingNet 项目适配 | [rachel_pairingnet_benchmark.py](../staging/pairwise_v0_2/baselines/rachel_pairingnet_benchmark.py) |
| ShreddingNet 项目适配 | [rachel_shreddingnet_benchmark.py](../staging/pairwise_v0_2/baselines/rachel_shreddingnet_benchmark.py) |
| 新局部证据八组：cap、heads、GCN、D 与 stable | [local_evidence_v2/model.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/model.py) |
| D 条件重采样／候选正确性标签 | [local_evidence_v2/data.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/data.py) |
| 新组训练预算与 loss | [local_evidence_v2/train.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/train.py) |
| 新组冻结推理／评估 | [local_evidence_v2/evaluate.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/evaluate.py) |
| 来源分折与构造负例 | [real_domain_calibration_v1/prepare.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/real_domain_calibration_v1/prepare.py) |
| 五折阈值选择与独立折测试 | [real_domain_calibration_v1/calibrate.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/real_domain_calibration_v1/calibrate.py) |
| 0.20–0.80 有界阈值策略 | [bounded_real_calibration_v2/common.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/bounded_real_calibration_v2/common.py) |
| 历史模型有界校准推理适配 | [bounded_real_calibration_v2/infer.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/bounded_real_calibration_v2/infer.py) |

新实验的逐项网络变动与训练配方见 [LOCAL_EVIDENCE_V2.md](LOCAL_EVIDENCE_V2.md)。

其余 Python 文件主要是训练、数据解析、评估或历史包初始化所需依赖，不表示全部都是当前推荐主线。`staging/pairwise_v0_1` 的少量依赖仅用于历史数据解析。
