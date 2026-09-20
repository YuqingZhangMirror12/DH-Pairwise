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
| 最新 all_tokens / matched_tokens / matched_edges | [matched_only/model.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/matched_only/model.py) |
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

其余 Python 文件主要是训练、数据解析、评估或历史包初始化所需依赖，不表示全部都是当前推荐主线。`staging/pairwise_v0_1` 的少量依赖仅用于历史数据解析。
