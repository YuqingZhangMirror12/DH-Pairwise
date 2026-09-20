# 运行与外部文件

## 源码与目录

从仓库根运行 `python -m ...`。保留 `staging/` 和 `experiments/` 的历史包路径，是为了兼容现有代码与 checkpoint；它们不是两个不同仓库。

代码快照不携带数据、checkpoint 或完整运行凭据。部分历史实验入口仍保留 `/root/autodl-tmp/...` 的原实验默认路径和内容指纹；这是复现实验身份，不是可公开下载的路径。换机器时需使用入口支持的路径参数，或在新实验配置中映射路径。不要删掉身份校验后将新数据输出冒充旧实验。

核心模型不依赖 PyG；ShreddingNet 适配版额外需要与本机 PyTorch/CUDA 相容的 `torch-geometric`。基线训练另有上游 checkout 的固定版本检查，通用 requirements 不替代这些基线环境要求。

公开版移除了两个历史数据默认路径中的本机用户名。网络和训练数学逻辑未因此改动；公开源码的字节哈希可能不同于历史 receipt，不能用于伪造“源码字节完全一致”的旧运行。

## 外部依赖

| 文件／目录 | 用途 | 本仓库是否提供 |
|---|---|---|
| Rachel prepared masks、contours、source lineage 与 train/val/test manifests | 训练与仿真验证 | 否 |
| 固定 S7 TRAIN24K materialized manifest 与样本 | 当前难化训练配方 | 否；生成实现提供 |
| S7 M12 checkpoint | 固定 Matcher | 否 |
| 冻结 token/候选 TRAIN/VAL cache | 独立 Scorer 训练 | 否；缓存实现提供 |
| Scorer checkpoint + freeze/protocol | 真实推理与阈值 | 否 |
| 敦煌／Turufan prepared inputs 与人工保留清单 | 真实域诊断 | 否 |
| PairingNet／ShreddingNet 官方固定 revision checkout | 适配基线 provenance 检查 | 否；上游链接提供 |

外部 checkpoint 仅加载可信的自有文件。若含 Python pickle 对象，不要加载陌生来源的文件。

## 主要入口

先查看命令帮助，不会启动训练：

```bash
python -m experiments.rachel_n512_formal_30k.train_score_decoupled --help
python -m experiments.rachel_n512_formal_30k.materialize_s7_training --help
python -m experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.cache --help
python -m experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.train --help
```

缓存准备完成后的独立分类头示例（路径为占位符）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.train \
  --arm matched_tokens \
  --train-cache /path/to/complete/train-cache \
  --val-cache /path/to/complete/val-cache \
  --output /path/to/new-run \
  --device cuda:0 --stop-after-head-epoch 16
```

该入口的 fresh 头固定 microbatch=16、累积=1，读取 S7 M12 缓存，训练 16 个分类轮次（不是继续训练 Matcher）。要求独占其可见 GPU；不要直接绕过互斥检查。`edge_seed`／`edge_multi` 还需要 train/val stage side-cache。旧缓存没有完整 dustbin 信息，新的 dustbin-aware 训练需要另建特征缓存。

`matched_only.inference.FrozenMatchedInference` 组合可信的 base Matcher 与独立 Scorer；推理输入只有 mask、轮廓与有效掩码，不输入 GT。它只替换分类输出；最高分候选暂未替换正式生产 Layout。

## 训练配方与检查点身份

- S7 M12 Matcher SHA-256：`d8a93af1eb5f3b02baaf7d42b9d8675242a446a1b11e43cde0561ba89e670e07`。
- matched_tokens C16 头 SHA-256：`2658c96453996138fd953dda24168d099cc1caef866d578aaae8e942e706eedf`。
- 主阈值约 0.834463；SIMVAL 99% 召回阈值 `0.3293727934360504`。只适用于绑定的这组权重／预处理，不能迁移成所有头的通用阈值。
- 所有外部文件的具体存放位置见所有者本地交接文档，不公开服务器登录信息。

## CPU 测试

```bash
python -m unittest \
  experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.test_model \
  experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.test_candidate_groups \
  experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local.test_candidate_local
```

源码快照不是“重新完成一次 GPU 训练”。更改源码、数据或采样后，应创建新 run identity、重建相关缓存、重新冻结仿真验证阈值，不覆盖旧结果。

本次发布前在独立代码副本上通过 Python 语法编译，以及 matched-only、候选选择、推理适配与 stage-cache 共 63 项合成 CPU 单测；没有运行全数据训练或真实域重评估。
