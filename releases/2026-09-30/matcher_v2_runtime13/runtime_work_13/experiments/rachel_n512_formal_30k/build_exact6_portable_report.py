#!/usr/bin/env python3
"""Build the canonical portable-report artifact from final exact-six evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SUMMARY_SCHEMA = "rachel-exact6-evidence-summary/1.0"
SUMMARY_STATUS = "complete_verified_exact_six_evidence"
METHODS = (
    "coarse_only",
    "full_n512",
    "matched_mm_converged",
    "matched_mm_same_exposure_epoch5",
    "pairingnet_adapted",
    "shreddingnet_adapted",
)
TRANSLATION_METHODS = (
    "full_n512",
    "pairingnet_adapted",
    "shreddingnet_adapted",
)
CONDITIONS = (
    "clean",
    "erosion_r2",
    "erosion_r4",
    "erosion_r8",
    "local_bites_k1_r8",
    "local_bites_k2_r8",
    "local_bites_k4_r8",
)
SOURCE_FILENAMES = {
    "canonical_summary": "final_summary.json",
    "synthetic_receipt": "synthetic_test_receipt.json",
    "corrosion_receipt": "corrosion_robustness_receipt.json",
    "real_pair_only": "real_pair_only.json",
    "real_translation_gt": "real_translation_gt.json",
    "synthetic_bootstrap": "synthetic_paired_endpoint_bootstrap_20000.json",
    "real_bootstrap": "real_paired_endpoint_bootstrap_20000.json",
    "resources": "resource_summary.json",
}
SOURCE_LABELS = {
    "canonical_summary": "经 fail-closed 校验的 exact-six canonical summary",
    "synthetic_receipt": "冻结 synthetic exact-six receipt",
    "corrosion_receipt": "固定 common-valid 腐蚀鲁棒性 receipt",
    "real_pair_only": "真实数据 pair-only exact-six receipt",
    "real_translation_gt": "预测冻结后的真实正样本 translation-GT receipt",
    "synthetic_bootstrap": "Synthetic 20,000 次 paired cluster bootstrap",
    "real_bootstrap": "Real balanced-1016 20,000 次 paired cluster bootstrap",
    "resources": "GPU、训练与推理资源证据汇总",
}
SHORT_METHODS = {
    "coarse_only": "Coarse-only",
    "full_n512": "Full",
    "matched_mm_converged": "Matched-MM converged",
    "matched_mm_same_exposure_epoch5": "Matched-MM epoch 5",
    "pairingnet_adapted": "PairingNet-adapted",
    "shreddingnet_adapted": "ShreddingNet-adapted",
}


class PortableReportError(RuntimeError):
    """A canonical-summary or portable-report precondition failed."""


def _duplicate_rejector(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise PortableReportError("duplicate JSON key: " + key)
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise PortableReportError("non-finite JSON constant: " + value)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _check_finite(value: object, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PortableReportError(location + " contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _check_finite(child, location + "." + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite(child, "{}[{}]".format(location, index))


def _read_summary(path_value: Path) -> Mapping[str, Any]:
    path = Path(path_value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise PortableReportError("summary must be a non-symlink regular file")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PortableReportError("summary changed while it was read")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_rejector,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortableReportError("summary is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise PortableReportError("summary root must be an object")
    _check_finite(value)
    if value.get("schema_version") != SUMMARY_SCHEMA or value.get("status") != SUMMARY_STATUS:
        raise PortableReportError("summary schema/status differs")
    if value.get("method_order") != list(METHODS):
        raise PortableReportError("summary exact-six method order differs")
    declared = value.get("content_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise PortableReportError("summary content SHA-256 is invalid")
    body = dict(value)
    del body["content_sha256"]
    if hashlib.sha256(_canonical_bytes(body)).hexdigest() != declared:
        raise PortableReportError("summary content SHA-256 differs")
    limits = value.get("claim_limits")
    if not isinstance(limits, Mapping) or limits.get(
        "same_data_benchmarks_are_adaptations_not_exact_reproductions"
    ) is not True:
        raise PortableReportError("adaptation claim limit is missing")
    return value


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PortableReportError(description + " must be an object")
    return value


def _number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PortableReportError(description + " must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise PortableReportError(description + " must be finite")
    return output


def _classification_row(
    method: str,
    display: Mapping[str, Any],
    metrics: Mapping[str, Any],
    **extra: object,
) -> Mapping[str, object]:
    row = _mapping(metrics.get("row"), method + ".row")
    cluster = _mapping(metrics.get("cluster_balanced"), method + ".cluster")
    result: Dict[str, object] = {
        "method_id": method,
        "method": display[method],
        "method_short": SHORT_METHODS[method],
    }
    result.update(extra)
    for prefix, source in (("row", row), ("cluster", cluster)):
        for metric in ("auroc", "auprc", "f1", "recall"):
            result[prefix + "_" + metric] = _number(
                source.get(metric), method + "." + prefix + "." + metric
            )
    return result


def _interval_text(interval: Mapping[str, Any], percent: bool = False) -> str:
    point = _number(interval.get("point_estimate"), "interval point")
    bounds = interval.get("percentile_95_ci")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise PortableReportError("interval bounds differ")
    low = _number(bounds[0], "interval low")
    high = _number(bounds[1], "interval high")
    if percent:
        return "{:.1%} [{:.1%}, {:.1%}]".format(point, low, high)
    return "{:.4f} [{:.4f}, {:.4f}]".format(point, low, high)


def _source_objects() -> Tuple[list, list]:
    manifest_sources = []
    sources = []
    for source_id, filename in SOURCE_FILENAMES.items():
        logical = PurePosixPath(filename)
        if logical.is_absolute() or len(logical.parts) != 1 or logical.name != filename:
            raise PortableReportError("source path must be one relative filename")
        row = {"id": source_id, "label": SOURCE_LABELS[source_id], "path": filename}
        manifest_sources.append(dict(row))
        sources.append(
            {
                **row,
                "query": {
                    "engine": "duckdb",
                    "sql": "SELECT * FROM read_json_auto('{}');".format(filename),
                    "description": (
                        "Loads the SHA-bound JSON source; the portable-report generator "
                        "selects and reshapes only reviewed aggregate fields."
                    ),
                    "tables_used": [filename],
                },
            }
        )
    return manifest_sources, sources


def build_artifact(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    display = _mapping(summary.get("method_display_names"), "display names")
    if set(display) != set(METHODS):
        raise PortableReportError("display-name inventory differs")
    for method in ("pairingnet_adapted", "shreddingnet_adapted"):
        if "adapted" not in str(display[method]).casefold():
            raise PortableReportError(method + " display name omits adaptation")

    classification = _mapping(summary.get("classification"), "classification")
    synthetic_source = _mapping(
        classification.get("synthetic_native_all_valid"), "synthetic classification"
    )
    synthetic_rows = [
        _classification_row(method, display, _mapping(synthetic_source[method], method))
        for method in METHODS
    ]

    corrosion_source = _mapping(
        classification.get("corrosion_fixed_all_method_all_condition_common_valid"),
        "corrosion classification",
    )
    corrosion_rows = []
    for condition in CONDITIONS:
        condition_source = _mapping(corrosion_source.get(condition), condition)
        if set(condition_source) != set(METHODS):
            raise PortableReportError(condition + " method inventory differs")
        for method in METHODS:
            corrosion_rows.append(
                _classification_row(
                    method,
                    display,
                    _mapping(condition_source[method], condition + "." + method),
                    condition=condition,
                )
            )

    real_source = _mapping(
        classification.get("real_balanced1016_inferential"),
        "real balanced classification",
    )
    real_methods = _mapping(real_source.get("methods"), "real methods")
    if set(real_methods) != set(METHODS):
        raise PortableReportError("real method inventory differs")
    real_rows = []
    real_ranking_rows = []
    for method in METHODS:
        fair = _mapping(
            _mapping(real_methods[method], method).get("fair_all_method_common_valid"),
            method + ".fair",
        )
        output = _classification_row(method, display, fair)
        real_rows.append(output)
        for metric in ("AUROC", "AUPRC"):
            real_ranking_rows.append(
                {
                    "method": output["method_short"],
                    "metric": metric,
                    "value": output["row_" + metric.casefold()],
                }
            )

    bootstrap_source = _mapping(summary.get("bootstrap"), "bootstrap")
    bootstrap_rows = []
    for population, source_key in (
        ("Synthetic", "synthetic_pair_classification"),
        ("Real balanced-1016", "real_pair_classification"),
    ):
        source = _mapping(bootstrap_source.get(source_key), source_key)
        deltas = _mapping(source.get("full_n512_minus_comparator"), source_key + ".deltas")
        for comparator in METHODS:
            if comparator == "full_n512":
                continue
            comparator_source = _mapping(deltas.get(comparator), comparator)
            for metric in ("auroc", "auprc"):
                interval = _mapping(comparator_source.get(metric), comparator + "." + metric)
                bounds = interval.get("percentile_95_ci")
                if not isinstance(bounds, list) or len(bounds) != 2:
                    raise PortableReportError("bootstrap interval differs")
                low, high = (_number(bounds[0], "CI low"), _number(bounds[1], "CI high"))
                bootstrap_rows.append(
                    {
                        "population": population,
                        "comparison": "Full − " + SHORT_METHODS[comparator],
                        "metric": metric.upper(),
                        "delta": _number(interval.get("point_estimate"), "delta"),
                        "ci95_low": low,
                        "ci95_high": high,
                        "probability_delta_gt_zero": _number(
                            interval.get("probability_delta_gt_zero"), "P(delta>0)"
                        ),
                        "ci_excludes_zero": low > 0.0 or high < 0.0,
                    }
                )

    pose = _mapping(summary.get("pose"), "pose")
    translation_root = _mapping(
        pose.get("real_strict547_positive_translation_gt"), "real translation"
    )
    translation_methods = _mapping(translation_root.get("methods"), "translation methods")
    translation_rows = []
    translation_error_rows = []
    translation_recall_rows = []
    for method in TRANSLATION_METHODS:
        method_source = _mapping(translation_methods.get(method), method + " translation")
        metrics = _mapping(method_source.get("case_bootstrap"), method + " bootstrap")

        def interval(name: str) -> Mapping[str, Any]:
            return _mapping(metrics.get(name), method + "." + name)

        row: Dict[str, object] = {
            "method_id": method,
            "method": display[method],
            "method_short": SHORT_METHODS[method],
        }
        for name in (
            "valid_translation_prediction_fraction",
            "median_l2_px",
            "p90_l2_px",
            "recall_at_2px",
            "recall_at_5px",
            "recall_at_8px",
            "recall_at_10px",
            "joint_frozen_threshold_and_translation_recall_at_2px",
            "joint_frozen_threshold_and_translation_recall_at_5px",
            "joint_frozen_threshold_and_translation_recall_at_8px",
            "joint_frozen_threshold_and_translation_recall_at_10px",
        ):
            source_interval = interval(name)
            row[name] = _number(source_interval.get("point_estimate"), method + "." + name)
            bounds = source_interval.get("percentile_95_ci")
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise PortableReportError(method + "." + name + " CI differs")
            row[name + "_ci"] = _interval_text(
                source_interval,
                percent=("fraction" in name or "recall" in name),
            )
        translation_rows.append(row)
        for statistic, field in (("Median", "median_l2_px"), ("P90", "p90_l2_px")):
            translation_error_rows.append(
                {
                    "method": row["method_short"],
                    "statistic": statistic,
                    "pixels": row[field],
                }
            )
        for tolerance in (2, 5, 8, 10):
            translation_recall_rows.append(
                {
                    "method": row["method_short"],
                    "tolerance": "≤{} px".format(tolerance),
                    "recall": row["recall_at_{}px".format(tolerance)],
                }
            )

    resources_root = _mapping(summary.get("resources"), "resources")
    resource_methods = _mapping(resources_root.get("methods"), "resource methods")
    hardware = _mapping(resources_root.get("hardware"), "resource hardware")
    concurrency = _mapping(
        resources_root.get("concurrency_assessment"), "resource concurrency"
    )
    pair_shred_concurrency = _mapping(
        concurrency.get("pairingnet_plus_shreddingnet"),
        "PairingNet plus ShreddingNet concurrency",
    )
    all_three_concurrency = _mapping(
        concurrency.get("all_three"), "all-three concurrency"
    )
    resource_rows = []
    throughput_rows = []
    memory_rows = []
    for method in TRANSLATION_METHODS:
        source = _mapping(resource_methods.get(method), method + " resources")
        parameter = _mapping(source.get("parameter_count"), method + ".parameter_count")
        if method == "shreddingnet_adapted":
            parameter_count = _number(
                parameter.get("end_to_end_inference_value"), method + ".parameters"
            )
        else:
            parameter_count = _number(parameter.get("value"), method + ".parameters")
        training = _mapping(source.get("training"), method + ".training")
        if method == "shreddingnet_adapted":
            training_hours = _number(
                training.get("pipeline_wallclock_hours"), method + ".training_hours"
            )
            training_scope = "pipeline filesystem wall clock"
        else:
            training_hours = _number(
                training.get("measured_epoch_hours"), method + ".training_hours"
            )
            training_scope = "summed embedded train+validation timers"
        inference = _mapping(
            source.get("sealed_synthetic_inference"), method + ".inference"
        )
        gpu = _mapping(source.get("gpu_memory_training_step"), method + ".gpu")
        if method == "shreddingnet_adapted":
            peak_allocated_gib = _number(
                gpu.get("maximum_peak_allocated_bytes"), method + ".allocated"
            ) / (1024.0**3)
            peak_reserved_gib = _number(
                gpu.get("maximum_peak_reserved_bytes"), method + ".reserved"
            ) / (1024.0**3)
            memory_scope = "maximum one-step smoke across the three stages"
        else:
            peak_allocated_gib = _number(
                gpu.get("peak_allocated_gibibytes"), method + ".allocated"
            )
            peak_reserved_gib = _number(
                gpu.get("peak_reserved_gibibytes"), method + ".reserved"
            )
            memory_scope = "one formal-batch optimizer-step smoke"
        pairs_per_second = _number(
            inference.get("pairs_per_second"), method + ".pairs_per_second"
        )
        elapsed_seconds = _number(
            inference.get("elapsed_seconds"), method + ".elapsed_seconds"
        )
        resource_rows.append(
            {
                "method_id": method,
                "method": display[method],
                "method_short": SHORT_METHODS[method],
                "parameters": parameter_count,
                "training_hours": training_hours,
                "training_timer_scope": training_scope,
                "synthetic_pairs_per_second": pairs_per_second,
                "synthetic_elapsed_seconds": elapsed_seconds,
                "peak_allocated_gib": peak_allocated_gib,
                "peak_reserved_gib": peak_reserved_gib,
                "gpu_measurement_scope": memory_scope,
            }
        )
        throughput_rows.append(
            {"method": SHORT_METHODS[method], "pairs_per_second": pairs_per_second}
        )
        memory_rows.extend(
            [
                {
                    "method": SHORT_METHODS[method],
                    "measurement": "Peak allocated",
                    "gib": peak_allocated_gib,
                },
                {
                    "method": SHORT_METHODS[method],
                    "measurement": "Peak reserved",
                    "gib": peak_reserved_gib,
                },
            ]
        )

    coverage = _mapping(summary.get("coverage"), "coverage")
    real_coverage = _mapping(coverage.get("real_balanced1016"), "real coverage")
    synthetic_coverage = _mapping(coverage.get("synthetic"), "synthetic coverage")
    corrosion_coverage = _mapping(
        coverage.get("corrosion_all_conditions"), "corrosion coverage"
    )
    full_real = next(row for row in real_rows if row["method_id"] == "full_n512")
    shred_translation = next(
        row for row in translation_rows if row["method_id"] == "shreddingnet_adapted"
    )
    full_resource = next(row for row in resource_rows if row["method_id"] == "full_n512")
    headline = [
        {
            "real_full_auroc": full_real["row_auroc"],
            "real_full_auprc": full_real["row_auprc"],
            "shred_translation_recall10": shred_translation["recall_at_10px"],
            "full_throughput": full_resource["synthetic_pairs_per_second"],
        }
    ]

    source_hashes = _mapping(summary.get("sources_file_sha256"), "source hashes")
    expected_hash_keys = set(SOURCE_FILENAMES) - {"canonical_summary"}
    if set(source_hashes) != expected_hash_keys:
        raise PortableReportError("summary source-hash inventory differs")
    integrity_rows = [
        {
            "source": SOURCE_FILENAMES["canonical_summary"],
            "sha256": str(summary["content_sha256"]),
            "hash_scope": "canonical JSON content (content_sha256)",
        }
    ]
    for source_id in SOURCE_FILENAMES:
        if source_id == "canonical_summary":
            continue
        value = source_hashes[source_id]
        if not isinstance(value, str) or len(value) != 64:
            raise PortableReportError(source_id + " SHA-256 differs")
        integrity_rows.append(
            {
                "source": SOURCE_FILENAMES[source_id],
                "sha256": value,
                "hash_scope": "source file bytes",
            }
        )

    real_bootstrap = _mapping(
        bootstrap_source.get("real_pair_classification"), "real bootstrap"
    )
    real_deltas = _mapping(
        real_bootstrap.get("full_n512_minus_comparator"), "real deltas"
    )
    pair_auroc = _mapping(
        _mapping(real_deltas.get("pairingnet_adapted"), "pair delta").get("auroc"),
        "pair AUROC delta",
    )
    shred_auroc = _mapping(
        _mapping(real_deltas.get("shreddingnet_adapted"), "shred delta").get("auroc"),
        "shred AUROC delta",
    )
    shred_auprc = _mapping(
        _mapping(real_deltas.get("shreddingnet_adapted"), "shred delta").get("auprc"),
        "shred AUPRC delta",
    )
    pair_translation = next(
        row for row in translation_rows if row["method_id"] == "pairingnet_adapted"
    )
    full_translation = next(
        row for row in translation_rows if row["method_id"] == "full_n512"
    )

    technical_summary = "\n".join(
        [
            "## 技术摘要",
            "",
            "- **真实 balanced-1016 的排序主指标上，Full 的 row AUROC 为 {:.4f}，高于 PairingNet-adapted 与 ShreddingNet-adapted；两项 Full-minus-adapted 的 paired 95% CI 均排除 0。**".format(
                full_real["row_auroc"]
            ),
            "- **AUPRC 不支持 Full 全面占优。** ShreddingNet-adapted 的 row AUPRC 为 {:.4f}，Full 为 {:.4f}；Full−ShreddingNet-adapted 的 95% CI 为 [{:.4f}, {:.4f}]，跨 0。".format(
                next(row for row in real_rows if row["method_id"] == "shreddingnet_adapted")[
                    "row_auprc"
                ],
                full_real["row_auprc"],
                shred_auprc["percentile_95_ci"][0],
                shred_auprc["percentile_95_ci"][1],
            ),
            "- **冻结阈值 F1/Recall 与 threshold-free 排序必须分开解释。** balanced-1016 exact-six 中 Coarse-only 的 row F1/Recall 最高；在 Full、PairingNet-adapted、ShreddingNet-adapted 三者中则是 ShreddingNet-adapted 最高。PairingNet-adapted 的低 F1/Recall 主要反映 validation threshold 向真实域迁移后的校准问题，不能当作其 AUROC/AUPRC。",
            "- **真实 508 个正样本的平移定位上，ShreddingNet-adapted 明显更强，但仍有长尾。** 其中位误差 {:.2f}px、≤10px 无条件召回 {:.1%}，而 P90 为 {:.2f}px；PairingNet-adapted 与 Full 的中位误差分别为 {:.2f}px 与 {:.2f}px。".format(
                shred_translation["median_l2_px"],
                shred_translation["recall_at_10px"],
                shred_translation["p90_l2_px"],
                pair_translation["median_l2_px"],
                full_translation["median_l2_px"],
            ),
            "- **资源侧 Full 的 synthetic 端到端评估吞吐最高。** Full 为 {:.2f} pairs/s，PairingNet-adapted 与 ShreddingNet-adapted 分别为 {:.2f} 与 {:.2f} pairs/s；显存数字是单个正式 batch 的训练步 smoke，而不是全训练连续峰值。".format(
                full_resource["synthetic_pairs_per_second"],
                next(row for row in resource_rows if row["method_id"] == "pairingnet_adapted")[
                    "synthetic_pairs_per_second"
                ],
                next(row for row in resource_rows if row["method_id"] == "shreddingnet_adapted")[
                    "synthetic_pairs_per_second"
                ],
            ),
        ]
    )
    definitions = "\n".join(
        [
            "## 数据与指标口径",
            "",
            "- **任务范围：** mask-only、已知 upright 条件下的 pairwise 匹配概率与相对二维平移；不等同于全局多碎片拼接。",
            "- **AUROC/AUPRC：** threshold-free 排序指标；报告 equal-row 与按 case/cluster 平衡后的点估计。",
            "- **F1/Recall：** 仅使用训练完成后冻结的 validation threshold，是 secondary 指标；没有在 test/real 上重新拟合阈值。",
            "- **推断人口：** synthetic 为 3,000/3,000 六方法 common-valid；腐蚀为七种条件共同有效的 1,822/3,000；real balanced-1016 为 1,016/1,016。",
            "- **不确定性：** synthetic 与 balanced-1016 的 equal-row AUROC/AUPRC 使用 20,000 次 paired endpoint-unit pigeonhole bootstrap；表中 95% percentile CI 未做多重比较校正。",
            "- **命名：** PairingNet-adapted 与 ShreddingNet-adapted 是针对本数据格式和 pairwise/upright 任务的同数据适配，不是原论文代码的 byte-exact 复现。",
            "- **Pose 口径：** median/P90 L2 只在 valid-pose 子集上统计；R@2/5/8/10px 以全部 508 个真实 positives 为分母，无效 pose 计为失败。PairingNet-style RR<4 指 `e_rmse < 4`，不是 translation error <4px。",
        ]
    )
    method_design = "\n".join(
        [
            "## 方法设计与适配",
            "",
            "- **Full (Sliding-window + Sinkhorn)：** coarse Siamese 提供全局 pair 线索，轮廓 sliding patches 提供局部对应，再通过可微 Sinkhorn 进行带 dustbin 的软分配；当前实验的 `full_n512` 上限为 512 个轮廓 token。",
            "- **PairingNet-adapted：** 在同一 mask-only、N=512、known-upright pairwise 数据协议下训练的适配版本，用于输出 pair score 与二维平移；报告不声称与论文环境或发布代码运行 byte-exact 等价。",
            "- **ShreddingNet-adapted：** 将 coarse、matching、classification 三阶段适配到相同 pairwise/upright 协议；全局 assembly、原生 CM/FM/SE/GA 不在本轮证据范围内。",
            "- 两个 adapted benchmark 与 Full 使用同一冻结 exact-six 测试人口和同一公平 common-valid 口径；适配命名在表、图和结论中始终保留。",
        ]
    )
    limitations = "\n".join(
        [
            "## 局限",
            "",
            "- strict-547 分类仅有 39 个 GT negatives，因此只作描述；balanced-1016 的 469 个构造 distractors **不是 GT negatives**。",
            "- translation-GT 仅覆盖 strict-547 中的 508 个正样本；无效 pose 计为无条件召回失败，median/P90 则条件于有效预测。",
            "- rotation 被条件为 upright，未训练、估计或评价旋转误差；pairwise 指标也不能证明全局组装质量。",
            "- paired bootstrap 只覆盖 equal-row AUROC/AUPRC。cluster-balanced 指标以及 frozen-threshold F1/Recall 目前只有点估计。",
            "- 腐蚀结果固定在跨六方法、跨七条件共同有效的 1,822 对交集上；这保证公平比较，但不能外推到被排除的 1,178 对。",
            "- GPU 显存是单步 smoke；训练计时在 Full/Pairing 与 Shredding 流水线之间计时范围不同。推理吞吐是 synthetic 全流程评估，不是纯网络 forward latency。",
            "- PairingNet-adapted + ShreddingNet-adapted 的并发适用性目前仅由两个单步 smoke 的 reserved 峰值保守相加推断，尚未做真实双进程压力测试；CPU、I/O、数据加载与吞吐竞争仍未知。",
        ]
    )
    conclusion = "\n".join(
        [
            "## 结论与下一步",
            "",
            "1. 若首要目标是 **real pair ranking 的 AUROC**，当前证据支持保留 Full：相对两个 adapted benchmark 的 paired CI 均为正（PairingNet-adapted {}；ShreddingNet-adapted {}）。".format(
                _interval_text(pair_auroc), _interval_text(shred_auroc)
            ),
            "2. 若首要目标是 **真实正样本平移定位**，当前实现应优先研究 ShreddingNet-adapted；但其 P90 很高，必须先定位长尾 case，而不能只看 median。",
            "3. **建议的工程主线是 Full 筛候选 + ShreddingNet-adapted 二阶段位姿。** 这是基于本轮 ranking 与 translation 的组合推断，尚未作为端到端 hybrid 实测；PairingNet-adapted 可作为速度—位姿质量折中（明显快于 ShreddingNet-adapted，但本轮真实 pose 较弱）。",
            "4. Full 的真实 translation 与 synthetic pose 表现出现明显域差异。下一轮应仅在 train/validation 上诊断坐标/尺度、局部对应与域适配，再进行新的预注册测试；不能用本轮 test/real 数值调参后仍宣称同一轮 confirmatory 结果。",
            "5. 若论文主张包含多碎片全局复原，应另设带完整候选图与全局 placement GT 的 assembly benchmark；本报告不把 selected-list pairwise 结果包装成原生 GA/CM/FM/SE。",
        ]
    )

    coarse_real = next(row for row in real_rows if row["method_id"] == "coarse_only")
    pairing_real = next(
        row for row in real_rows if row["method_id"] == "pairingnet_adapted"
    )
    shredding_real = next(
        row for row in real_rows if row["method_id"] == "shreddingnet_adapted"
    )
    real_chart_takeaway = "\n".join(
        [
            "### 如何读这张图",
            "",
            "AUROC/AUPRC 是不依赖阈值的排序能力；图中 Full 的 AUROC 最高，而 ShreddingNet-adapted 的 AUPRC 点估计略高。对应 paired CI 见下方表，不能仅凭柱高宣称差异。冻结阈值下，exact-six 的 row F1/Recall 最高其实是 Coarse-only（{:.4f}/{:.4f}）；仅在 Full、PairingNet-adapted、ShreddingNet-adapted 三者中，ShreddingNet-adapted 最高（{:.4f}/{:.4f}）。PairingNet-adapted 的 F1/Recall 为 {:.4f}/{:.4f}，这里反映真实域校准迁移，不能与其 AUROC {:.4f}、AUPRC {:.4f} 混为一谈。".format(
                coarse_real["row_f1"],
                coarse_real["row_recall"],
                shredding_real["row_f1"],
                shredding_real["row_recall"],
                pairing_real["row_f1"],
                pairing_real["row_recall"],
                pairing_real["row_auroc"],
                pairing_real["row_auprc"],
            ),
        ]
    )
    translation_error_takeaway = "\n".join(
        [
            "### 位姿误差图解读",
            "",
            "Median/P90 只在各方法自己的 valid-pose 子集上统计，不把无效 pose 填成大误差。ShreddingNet-adapted 的 median 最低（{:.2f}px），但 P90 仍达 {:.2f}px，说明真实域存在明显长尾；PairingNet-adapted 与 Full 的 median 分别为 {:.2f}px 和 {:.2f}px。".format(
                shred_translation["median_l2_px"],
                shred_translation["p90_l2_px"],
                pair_translation["median_l2_px"],
                full_translation["median_l2_px"],
            ),
        ]
    )
    translation_recall_takeaway = "\n".join(
        [
            "### 无条件召回图解读",
            "",
            "R@2/5/8/10px 始终以全部 508 个 positives 为分母，无效 pose 直接计为失败，因此可跨方法公平比较。R@10px：Full {:.1%}、PairingNet-adapted {:.1%}、ShreddingNet-adapted {:.1%}。这与上图的 valid-pose 条件 median/P90 是两个不同口径。".format(
                full_translation["recall_at_10px"],
                pair_translation["recall_at_10px"],
                shred_translation["recall_at_10px"],
            ),
        ]
    )
    pairing_resource = next(
        row for row in resource_rows if row["method_id"] == "pairingnet_adapted"
    )
    shredding_resource = next(
        row for row in resource_rows if row["method_id"] == "shreddingnet_adapted"
    )
    throughput_takeaway = "\n".join(
        [
            "### 吞吐图解读",
            "",
            "这是 3,000 synthetic pairs 的端到端正式方法评估，不是纯神经网络 forward latency。Full、PairingNet-adapted、ShreddingNet-adapted 分别为 {:.2f}、{:.2f}、{:.2f} pairs/s；PairingNet-adapted 是 ShreddingNet-adapted 的约 {:.1f} 倍。".format(
                full_resource["synthetic_pairs_per_second"],
                pairing_resource["synthetic_pairs_per_second"],
                shredding_resource["synthetic_pairs_per_second"],
                pairing_resource["synthetic_pairs_per_second"]
                / shredding_resource["synthetic_pairs_per_second"],
            ),
        ]
    )
    capacity_mib = _number(hardware.get("memory_total_mib"), "GPU capacity MiB")
    pair_shred_reserved_gb = _number(
        pair_shred_concurrency.get("combined_peak_reserved_gigabytes"),
        "Pair+Shred reserved GB",
    )
    pair_shred_reserved_gib = _number(
        pair_shred_concurrency.get("combined_peak_reserved_gibibytes"),
        "Pair+Shred reserved GiB",
    )
    pair_shred_fraction = _number(
        pair_shred_concurrency.get("fraction_of_device_capacity"),
        "Pair+Shred capacity fraction",
    )
    pair_shred_headroom = _number(
        pair_shred_concurrency.get("headroom_gibibytes"),
        "Pair+Shred headroom",
    )
    all_three_deficit = _number(
        all_three_concurrency.get("capacity_deficit_mebibytes"),
        "all-three capacity deficit",
    )
    memory_takeaway = "\n".join(
        [
            "### 显存与并发结论",
            "",
            "在 {}（{:.0f} MiB）上，把 PairingNet-adapted 与 ShreddingNet-adapted 各自正式 batch 单步 `peak_reserved` 保守相加为 {:.2f} GB（{:.2f} GiB），占设备 {:.1%}，仍有 {:.2f} GiB 余量；**按现有显存证据适合并发训练**。但尚未做真实双进程压力测试，吞吐、CUDA context、碎片化以及 CPU/I/O 竞争仍需实测。三者 reserved 合计已比设备容量高 {:.0f} MiB，尚未计额外开销，因此 **Full + PairingNet-adapted + ShreddingNet-adapted 不适合三者同时训练**。".format(
                hardware.get("device_name"),
                capacity_mib,
                pair_shred_reserved_gb,
                pair_shred_reserved_gib,
                pair_shred_fraction,
                pair_shred_headroom,
                all_three_deficit,
            ),
        ]
    )

    manifest_sources, top_sources = _source_objects()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )

    cards = [
        {
            "id": "real_full_auroc",
            "description": "Balanced-1016 六方法 common-valid equal-row AUROC。",
            "dataset": "headline",
            "sourceId": "real_pair_only",
            "metrics": [{"label": "Full real AUROC", "field": "real_full_auroc", "format": "percent"}],
        },
        {
            "id": "real_full_auprc",
            "description": "Balanced-1016 六方法 common-valid equal-row AUPRC。",
            "dataset": "headline",
            "sourceId": "real_pair_only",
            "metrics": [{"label": "Full real AUPRC", "field": "real_full_auprc", "format": "percent"}],
        },
        {
            "id": "shred_recall10",
            "description": "508 个真实正样本；无效 pose 计为失败。",
            "dataset": "headline",
            "sourceId": "real_translation_gt",
            "metrics": [{"label": "Shred-adapted translation recall ≤10px", "field": "shred_translation_recall10", "format": "percent"}],
        },
        {
            "id": "full_throughput",
            "description": "3,000 synthetic pairs 的端到端正式方法评估。",
            "dataset": "headline",
            "sourceId": "resources",
            "metrics": [{"label": "Full synthetic throughput", "field": "full_throughput", "format": "number", "unit": "pairs/s"}],
        },
    ]

    charts = [
        {
            "id": "real_ranking_chart",
            "title": "Balanced-1016 row-level ranking metrics",
            "subtitle": "Full 的 AUROC 最高；Shred-adapted 的 AUPRC 点估计略高于 Full。",
            "type": "bar",
            "dataset": "real_ranking",
            "sourceId": "real_pair_only",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {"field": "value", "type": "quantitative", "label": "Metric", "format": "percent"},
                "color": {"field": "metric", "type": "nominal", "label": "Metric"},
            },
        },
        {
            "id": "translation_error_chart",
            "title": "Real positive translation error",
            "subtitle": "Lower is better; P90 exposes a large failure tail for all three translation-producing methods.",
            "type": "bar",
            "dataset": "translation_error",
            "sourceId": "real_translation_gt",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {"field": "pixels", "type": "quantitative", "label": "Translation error (px)"},
                "color": {"field": "statistic", "type": "nominal", "label": "Statistic"},
            },
        },
        {
            "id": "translation_recall_chart",
            "title": "Unconditional real positive translation recall",
            "subtitle": "Invalid pose predictions count as failures; higher is better.",
            "type": "bar",
            "dataset": "translation_recall",
            "sourceId": "real_translation_gt",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "tolerance", "type": "ordinal", "label": "Tolerance"},
                "y": {"field": "recall", "type": "quantitative", "label": "Recall", "format": "percent"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
            },
        },
        {
            "id": "throughput_chart",
            "title": "Sealed synthetic end-to-end inference throughput",
            "subtitle": "3,000-pair formal method evaluation; higher is faster.",
            "type": "bar",
            "dataset": "throughput",
            "sourceId": "resources",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {"field": "pairs_per_second", "type": "quantitative", "label": "Pairs/s"},
            },
        },
        {
            "id": "gpu_memory_chart",
            "title": "Training-step smoke GPU memory",
            "subtitle": "Allocated and reserved memory for one formal-batch optimizer step; not a continuously sampled full-training maximum.",
            "type": "bar",
            "dataset": "gpu_memory",
            "sourceId": "resources",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {"field": "gib", "type": "quantitative", "label": "GPU memory (GiB)"},
                "color": {"field": "measurement", "type": "nominal", "label": "Measurement"},
            },
        },
    ]

    metric_columns = [
        {"field": "method", "label": "Method", "type": "text"},
        {"field": "row_auroc", "label": "Row AUROC", "format": "percent"},
        {"field": "row_auprc", "label": "Row AUPRC", "format": "percent"},
        {"field": "row_f1", "label": "Row F1", "format": "percent"},
        {"field": "row_recall", "label": "Row Recall", "format": "percent"},
        {"field": "cluster_auroc", "label": "Cluster AUROC", "format": "percent"},
        {"field": "cluster_auprc", "label": "Cluster AUPRC", "format": "percent"},
        {"field": "cluster_f1", "label": "Cluster F1", "format": "percent"},
        {"field": "cluster_recall", "label": "Cluster Recall", "format": "percent"},
    ]
    tables = [
        {
            "id": "synthetic_table",
            "title": "Synthetic exact-six classification",
            "subtitle": "3,000/3,000 all-method common-valid pairs; F1/Recall use frozen validation thresholds.",
            "dataset": "synthetic_classification",
            "sourceId": "synthetic_receipt",
            "defaultSort": {"field": "row_auroc", "direction": "desc"},
            "density": "dense",
            "columns": metric_columns,
        },
        {
            "id": "synthetic_bootstrap_table",
            "title": "Synthetic Full-minus-comparator paired bootstrap",
            "subtitle": "20,000 endpoint-unit pigeonhole draws; nominal percentile 95% CI.",
            "dataset": "synthetic_bootstrap",
            "sourceId": "synthetic_bootstrap",
            "defaultSort": {"field": "delta", "direction": "desc"},
            "density": "dense",
            "columns": [
                {"field": "comparison", "label": "Comparison", "type": "text"},
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "delta", "label": "Delta", "format": "number", "signed": True},
                {"field": "ci95_low", "label": "CI low", "format": "number", "signed": True},
                {"field": "ci95_high", "label": "CI high", "format": "number", "signed": True},
                {"field": "probability_delta_gt_zero", "label": "P(Δ>0)", "format": "percent"},
                {"field": "ci_excludes_zero", "label": "CI excludes 0", "type": "boolean"},
            ],
        },
        {
            "id": "corrosion_table",
            "title": "Corrosion exact-six classification",
            "subtitle": "Fixed 1,822-pair intersection valid for all six methods in all seven conditions.",
            "dataset": "corrosion_classification",
            "sourceId": "corrosion_receipt",
            "defaultSort": {"field": "condition", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "condition", "label": "Condition", "type": "text"},
                *metric_columns,
            ],
        },
        {
            "id": "real_table",
            "title": "Balanced-1016 exact-six classification",
            "subtitle": "Inference population: 508 positives and 508 negatives/distractors; all six methods valid.",
            "dataset": "real_classification",
            "sourceId": "real_pair_only",
            "defaultSort": {"field": "row_auroc", "direction": "desc"},
            "density": "dense",
            "columns": metric_columns,
        },
        {
            "id": "real_bootstrap_table",
            "title": "Real Full-minus-comparator paired bootstrap",
            "subtitle": "Balanced-1016, 20,000 endpoint-case pigeonhole draws; nominal percentile 95% CI.",
            "dataset": "real_bootstrap",
            "sourceId": "real_bootstrap",
            "defaultSort": {"field": "delta", "direction": "desc"},
            "density": "dense",
            "columns": [
                {"field": "comparison", "label": "Comparison", "type": "text"},
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "delta", "label": "Delta", "format": "number", "signed": True},
                {"field": "ci95_low", "label": "CI low", "format": "number", "signed": True},
                {"field": "ci95_high", "label": "CI high", "format": "number", "signed": True},
                {"field": "probability_delta_gt_zero", "label": "P(Δ>0)", "format": "percent"},
                {"field": "ci_excludes_zero", "label": "CI excludes 0", "type": "boolean"},
            ],
        },
        {
            "id": "translation_table",
            "title": "Strict-547 positive-subset translation metrics",
            "subtitle": "508 positives; entries include point estimate and case-bootstrap 95% CI.",
            "dataset": "translation",
            "sourceId": "real_translation_gt",
            "defaultSort": {"field": "method", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "valid_translation_prediction_fraction_ci", "label": "Valid pose fraction [95% CI]", "type": "text"},
                {"field": "median_l2_px_ci", "label": "Median L2 px [95% CI]", "type": "text"},
                {"field": "p90_l2_px_ci", "label": "P90 L2 px [95% CI]", "type": "text"},
                {"field": "recall_at_2px_ci", "label": "Recall ≤2px [95% CI]", "type": "text"},
                {"field": "recall_at_5px_ci", "label": "Recall ≤5px [95% CI]", "type": "text"},
                {"field": "recall_at_8px_ci", "label": "Recall ≤8px [95% CI]", "type": "text"},
                {"field": "recall_at_10px_ci", "label": "Recall ≤10px [95% CI]", "type": "text"},
            ],
        },
        {
            "id": "resource_table",
            "title": "Training, inference and GPU resource evidence",
            "subtitle": "Training timer scopes differ; GPU memory is one-step smoke; throughput is end-to-end synthetic evaluation.",
            "dataset": "resources",
            "sourceId": "resources",
            "defaultSort": {"field": "synthetic_pairs_per_second", "direction": "desc"},
            "density": "dense",
            "columns": [
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "parameters", "label": "Parameters", "format": "number"},
                {"field": "training_hours", "label": "Recorded training h", "format": "number"},
                {"field": "training_timer_scope", "label": "Training timer scope", "type": "text"},
                {"field": "synthetic_pairs_per_second", "label": "Synthetic pairs/s", "format": "number"},
                {"field": "peak_allocated_gib", "label": "Peak allocated GiB", "format": "number"},
                {"field": "peak_reserved_gib", "label": "Peak reserved GiB", "format": "number"},
                {"field": "gpu_measurement_scope", "label": "GPU scope", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# Dunhuang pairwise exact-six 技术报告"},
        {"id": "technical_summary", "type": "markdown", "body": technical_summary},
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {"id": "definitions", "type": "markdown", "body": definitions},
        {"id": "method_design", "type": "markdown", "body": method_design},
        {
            "id": "synthetic_heading",
            "type": "markdown",
            "sourceId": "synthetic_receipt",
            "body": "## Synthetic 分类\n\n六方法在同一 3,000 对 population 上均有效；表同时保留 equal-row 与 cluster-balanced AUROC/AUPRC/F1/Recall。",
        },
        {"id": "synthetic_table_block", "type": "table", "tableId": "synthetic_table", "layout": "full"},
        {"id": "synthetic_bootstrap_block", "type": "table", "tableId": "synthetic_bootstrap_table", "layout": "full"},
        {
            "id": "corrosion_heading",
            "type": "markdown",
            "sourceId": "corrosion_receipt",
            "body": "## 腐蚀鲁棒性\n\n公平比较固定在跨六方法、跨七条件共同有效的 {}/{} 对（{:.1%}）；不随条件或方法切换人口。这里的 clean 也是该固定 1,822 对子集，不能与 synthetic 3,000 对的 clean 指标直接相减并解释为腐蚀变化。".format(
                int(corrosion_coverage["valid_count"]),
                int(corrosion_coverage["population_count"]),
                corrosion_coverage["valid_fraction"],
            ),
        },
        {"id": "corrosion_table_block", "type": "table", "tableId": "corrosion_table", "layout": "full"},
        {
            "id": "real_heading",
            "type": "markdown",
            "sourceId": "real_pair_only",
            "body": "## 真实 balanced-1016 分类\n\n该 inferential population 为 {}/{} 六方法共同有效；包含 508 positives、39 strict GT negatives 与 469 个 constructed distractors。后者不宣称 GT negative。".format(
                int(real_coverage["valid_count"]), int(real_coverage["population_count"])
            ),
        },
        {"id": "real_chart_block", "type": "chart", "chartId": "real_ranking_chart"},
        {"id": "real_chart_takeaway", "type": "markdown", "sourceId": "real_pair_only", "body": real_chart_takeaway},
        {"id": "real_table_block", "type": "table", "tableId": "real_table", "layout": "full"},
        {"id": "real_bootstrap_block", "type": "table", "tableId": "real_bootstrap_table", "layout": "full"},
        {
            "id": "translation_heading",
            "type": "markdown",
            "sourceId": "real_translation_gt",
            "body": "## Strict-547 正样本 translation\n\n只评价其中 508 个具有 translation GT 的 positives；Coarse 与两个 matched-MM 方法不输出受监督二维平移，因此保持 N/A。",
        },
        {"id": "translation_error_block", "type": "chart", "chartId": "translation_error_chart"},
        {"id": "translation_error_takeaway", "type": "markdown", "sourceId": "real_translation_gt", "body": translation_error_takeaway},
        {"id": "translation_recall_block", "type": "chart", "chartId": "translation_recall_chart"},
        {"id": "translation_recall_takeaway", "type": "markdown", "sourceId": "real_translation_gt", "body": translation_recall_takeaway},
        {"id": "translation_table_block", "type": "table", "tableId": "translation_table", "layout": "full"},
        {
            "id": "resources_heading",
            "type": "markdown",
            "sourceId": "resources",
            "body": "## GPU、速度与训练资源\n\nFull、PairingNet-adapted 与 ShreddingNet-adapted 的参数量、训练计时、synthetic 端到端吞吐和正式 batch 单步显存 smoke 分开呈现。",
        },
        {"id": "throughput_block", "type": "chart", "chartId": "throughput_chart"},
        {"id": "throughput_takeaway", "type": "markdown", "sourceId": "resources", "body": throughput_takeaway},
        {"id": "gpu_memory_block", "type": "chart", "chartId": "gpu_memory_chart"},
        {"id": "memory_takeaway", "type": "markdown", "sourceId": "resources", "body": memory_takeaway},
        {"id": "resource_table_block", "type": "table", "tableId": "resource_table", "layout": "full"},
        {"id": "limitations", "type": "markdown", "body": limitations},
        {"id": "conclusion", "type": "markdown", "body": conclusion},
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Dunhuang pairwise exact-six 技术报告",
            "description": "六种冻结方法在 synthetic、腐蚀和真实 pairwise/upright 任务上的可复核技术比较。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "synthetic_classification": synthetic_rows,
                "corrosion_classification": corrosion_rows,
                "real_classification": real_rows,
                "real_ranking": real_ranking_rows,
                "synthetic_bootstrap": [
                    row for row in bootstrap_rows if row["population"] == "Synthetic"
                ],
                "real_bootstrap": [
                    row
                    for row in bootstrap_rows
                    if row["population"] == "Real balanced-1016"
                ],
                "translation": translation_rows,
                "translation_error": translation_error_rows,
                "translation_recall": translation_recall_rows,
                "resources": resource_rows,
                "throughput": throughput_rows,
                "gpu_memory": memory_rows,
                "integrity": integrity_rows,
            },
        },
        "sources": top_sources,
    }
    _check_finite(artifact)
    serialized = _canonical_bytes(artifact).decode("utf-8")
    if any(token in serialized for token in ("/Users/", "/root/", "ssh -p", "UWAH")):
        raise PortableReportError("portable artifact contains a forbidden local path or secret")
    if synthetic_coverage.get("valid_count") != 3000:
        raise PortableReportError("synthetic headline coverage differs")
    return artifact


def _write_atomic(path_value: Path, value: Mapping[str, Any]) -> None:
    path = Path(path_value).expanduser()
    if path.is_symlink():
        raise PortableReportError("output may not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = _read_summary(arguments.summary)
    artifact = build_artifact(summary)
    _write_atomic(arguments.output, artifact)
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
