#!/usr/bin/env python3
"""Render a verified, aggregate-only report from a completed Rachel final run.

This module is intentionally local and read-only with respect to ``final_root``.
It never opens pair-score JSONL files or model inputs.  Report publication is
reserved for a separate output namespace and uses first-writer-wins hard links.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "rachel-n512-final-report/2.0"
CONTROLLER_SCHEMA = "rachel-n512-final-pairwise-controller/2.0"
INVENTORY_RELATIVE = "control/content_sha256_inventory.json"
TERMINAL_RELATIVE = "control/terminal_receipt.json"
SYNTHETIC_STATS_RELATIVE = "stats_synthetic/paired_endpoint_bootstrap_20000.json"
REAL_STATS_RELATIVE = "stats_real/paired_endpoint_bootstrap_20000.json"
REAL_TRANSLATION_RELATIVE = (
    "real_translation_gt/strict547_positive_translation_gt_20000.json"
)
REAL_SOURCE_RELATIVE = "real/balanced1016_with_strict547.json"
BASE_METHODS = (
    "coarse_only",
    "full_n512",
    "matched_mm_converged",
    "matched_mm_same_exposure_epoch5",
)
BENCHMARK_METHODS = ("pairingnet_adapted", "shreddingnet_adapted")
METHODS = BASE_METHODS + BENCHMARK_METHODS
CORROSION_METHODS = METHODS
TRANSLATION_METHODS = ("full_n512",) + BENCHMARK_METHODS
NON_TRANSLATION_METHODS = tuple(
    method for method in METHODS if method not in TRANSLATION_METHODS
)
COMPARATORS = tuple(method for method in METHODS if method != "full_n512")
CORROSION_COMPARATORS = tuple(
    method for method in CORROSION_METHODS if method != "full_n512"
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
TOLERANCES = (2, 5, 8, 10)
ASSEMBLY_EDGE_TOLERANCES = (2, 5, 8, 10, 100)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_901
CORROSION_BOOTSTRAP_SEED = 260_901
REAL_TRANSLATION_BOOTSTRAP_SEED = (
    "rachel-real-translation-case-bootstrap-v1-fixed-20260901"
)

GATE_STATUS = {
    "control/same_data_benchmark_queue_gate.json": (
        "same_data_benchmark_queue_complete_and_verified_before_test_or_real_open"
    ),
    "control/train_validation_freeze_gate.json": (
        "complete_train_validation_freeze_before_test_or_real_open"
    ),
    "control/sealed_gate.json": "sealed_synthetic_complete_and_verified",
    "control/corrosion_gate.json": (
        "formal_exact_six_seven_condition_corrosion_complete_and_verified"
    ),
    "control/real_gate.json": (
        "real_balanced1016_single_forward_with_strict547_prefix_complete_and_verified"
    ),
    "control/synthetic_stats_gate.json": (
        "synthetic_statistics_20000_complete_and_verified"
    ),
    "control/real_stats_gate.json": "real_statistics_20000_complete_and_verified",
    "control/real_translation_gate.json": (
        "real_translation_gt_postprocessor_complete_and_verified"
    ),
    "control/source_code_freeze.json": "source_code_frozen_before_any_evaluation_open",
}
STAGE_ORDER = (
    "same_data_benchmark_queue_completion_and_freeze",
    "train_validation_exact_six_freeze",
    "combined_sealed_synthetic_exact_six_once",
    "formal_exact_six_seven_condition_corrosion_once",
    "combined_real_balanced1016_exact_six_single_forward_with_strict547_prefix",
    "synthetic_endpoint_statistics_exact_six_20000",
    "real_endpoint_statistics_exact_six_20000",
    "real_translation_gt_postprocessor_exact_six_20000",
)
INVENTORY_EXCLUDES = (INVENTORY_RELATIVE, TERMINAL_RELATIVE)

CANONICAL_RACHEL_N512_MODEL_CONFIG = {
    "canvas_size": 800,
    "coarse_size": 128,
    "contour_cap": 512,
    "window_sizes_px": [32.0, 64.0],
    "patch_size": 16,
    "feature_dim": 96,
    "num_heads": 4,
    "landmark_count": 32,
    "context_layers": 2,
    "evidence_dim": 24,
    "matcher_temperature": 0.25,
    "sinkhorn_iterations": 100,
    "sinkhorn_tolerance": 0.001,
    "translation_consensus_scale_px": 8.0,
    "translation_consensus_iterations": 2,
    "activation_checkpointing": True,
    "validate_runtime_inputs": False,
}
CANONICAL_RACHEL_N512_LOSS_CONFIG = {
    "fused_pair_weight": 1.0,
    "coarse_pair_weight": 0.25,
    "local_pair_weight": 0.5,
    "assignment_weight": 0.5,
    "translation_weight": 0.5,
    "sinkhorn_residual_weight": 0.05,
    "sinkhorn_residual_target": 0.001,
    "translation_scale_px": 32.0,
    "epsilon": 1e-8,
    "validate_runtime_targets": False,
    "collect_cpu_diagnostics": False,
}
CANONICAL_RACHEL_SELECTION_CONTRACT = {
    "seed": "rachel-pairwise-n512-v1",
    "total_pairs": 30_000,
    "pair_counts": {"train": 24_000, "val": 3_000, "test": 3_000},
    "positive_counts": {"train": 12_000, "val": 1_500, "test": 1_500},
    "negative_origin_counts_per_split": {
        "train": {
            "same_folder_hard": 6_000,
            "cross_folder_scale_matched": 6_000,
        },
        "val": {
            "same_folder_hard": 750,
            "cross_folder_scale_matched": 750,
        },
        "test": {
            "same_folder_hard": 750,
            "cross_folder_scale_matched": 750,
        },
    },
}


class FinalReportError(RuntimeError):
    """A provenance, schema, safety, or publication precondition failed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FinalReportError("value is not canonical finite JSON") from error


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash an authority member without parsing its contents or following a leaf link."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise FinalReportError("authority member cannot be opened safely") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FinalReportError("authority member is not a regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise FinalReportError("authority member changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FinalReportError("JSON contains a duplicate object key: " + key)
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise FinalReportError("JSON contains a non-finite constant: " + value)


def _check_finite(value: object, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FinalReportError(location + " contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_finite(item, location + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, "{}[{}]".format(location, index))


def _strict_json_bytes(payload: bytes, description: str) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except FinalReportError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FinalReportError(description + " is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise FinalReportError(description + " must be a JSON object")
    _check_finite(value)
    return value


def _resolve_directory(value: Path, description: str) -> Path:
    raw = Path(value).expanduser()
    try:
        resolved = raw.resolve(strict=True)
        info = os.lstat(resolved)
    except OSError as error:
        raise FinalReportError(description + " is missing") from error
    if not stat.S_ISDIR(info.st_mode) or raw.is_symlink():
        raise FinalReportError(description + " must be a non-symlink directory")
    return resolved


def _safe_member(root: Path, logical: str, description: str) -> Path:
    if not isinstance(logical, str) or "\\" in logical or "\x00" in logical:
        raise FinalReportError(description + " path is unsafe")
    relative = PurePosixPath(logical)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise FinalReportError(description + " path is unsafe")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError as error:
            raise FinalReportError(description + " is missing") from error
        if stat.S_ISLNK(info.st_mode):
            raise FinalReportError(description + " traverses a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise FinalReportError(description + " must be a regular file")
    return current


def _read_json_member(root: Path, logical: str, description: str) -> tuple[Mapping[str, Any], bytes]:
    if PurePosixPath(logical).suffix == ".jsonl":
        raise FinalReportError("pair-score JSONL files must never be opened")
    path = _safe_member(root, logical, description)
    payload = path.read_bytes()
    return _strict_json_bytes(payload, description), payload


def _verify_content_sha(document: Mapping[str, Any], description: str) -> None:
    declared = document.get("content_sha256")
    if not _is_sha256(declared):
        raise FinalReportError(description + " lacks a valid content SHA-256")
    body = dict(document)
    del body["content_sha256"]
    if _sha256_bytes(_canonical_bytes(body)) != declared:
        raise FinalReportError(description + " content SHA-256 differs")


def _load_terminal_inventory_bundle(
    final_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], str, str]:
    root = _resolve_directory(final_root, "final root")
    terminal, terminal_payload = _read_json_member(
        root, TERMINAL_RELATIVE, "terminal receipt"
    )
    _verify_content_sha(terminal, "terminal receipt")
    if terminal.get("schema_version") != CONTROLLER_SCHEMA:
        raise FinalReportError("terminal receipt schema differs")
    if terminal.get("status") != "complete_final_pairwise_protocol":
        raise FinalReportError("terminal receipt is not complete")
    if terminal.get("root") != str(root):
        raise FinalReportError("terminal receipt root differs")
    if (
        terminal.get("stage_order") != list(STAGE_ORDER)
        or terminal.get("fresh_sibling_output_directories")
        != [
            "sealed",
            "corrosion",
            "real",
            "stats_synthetic",
            "stats_real",
            "real_translation_gt",
        ]
        or terminal.get("logs_directory") != "control/logs"
        or terminal.get("terminal_receipt_first_hard_link_wins") is not True
        or terminal.get("formal_exact_six") is not True
        or terminal.get(
            "six_methods_verified_in_sealed_corrosion_real_statistics_and_real_translation_gt"
        )
        is not True
        or terminal.get("same_data_benchmark_adaptations_not_exact_reproductions")
        is not True
        or terminal.get("native_pairingnet_or_shreddingnet_CM_FM_SE_GA_claimed")
        is not False
        or terminal.get("automatic_performance_pass_fail_applied") is not False
    ):
        raise FinalReportError("terminal stage/output contract differs")
    corrosion_scope = terminal.get("corrosion_scope")
    if (
        not isinstance(corrosion_scope, Mapping)
        or corrosion_scope.get("formal_exact_six") is not True
        or corrosion_scope.get("method_inventory") != list(CORROSION_METHODS)
        or corrosion_scope.get("same_data_benchmark_methods_included") is not True
        or corrosion_scope.get("benchmark_correspondence")
        != "not_applicable_corrupted_contour_indices_have_no_current_condition_GT"
        or corrosion_scope.get("same_data_benchmark_adaptations_not_exact_reproductions")
        is not True
    ):
        raise FinalReportError("terminal corrosion scope differs")

    inventory, inventory_payload = _read_json_member(
        root, INVENTORY_RELATIVE, "content inventory"
    )
    _verify_content_sha(inventory, "content inventory")
    binding = terminal.get("content_inventory")
    if not isinstance(binding, Mapping):
        raise FinalReportError("terminal inventory binding is missing")
    if binding.get("path") != INVENTORY_RELATIVE:
        raise FinalReportError("terminal inventory path differs")
    if binding.get("file_sha256") != _sha256_bytes(inventory_payload):
        raise FinalReportError("terminal inventory file SHA-256 differs")
    if inventory.get("schema_version") != CONTROLLER_SCHEMA:
        raise FinalReportError("content inventory schema differs")
    if inventory.get("status") != "complete_content_sha256_inventory_before_terminal_receipt":
        raise FinalReportError("content inventory is not complete")
    if inventory.get("root") != str(root):
        raise FinalReportError("content inventory root differs")
    if terminal_payload == inventory_payload:
        raise FinalReportError("terminal and inventory unexpectedly alias")
    return (
        terminal,
        inventory,
        _sha256_bytes(terminal_payload),
        _sha256_bytes(inventory_payload),
    )


def _load_terminal_and_inventory(
    final_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    terminal, inventory, _, _ = _load_terminal_inventory_bundle(final_root)
    return terminal, inventory


@dataclass(frozen=True)
class InventoryEntry:
    logical_path: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class AggregateInputs:
    root: Path
    terminal: Mapping[str, Any]
    inventory: Mapping[str, InventoryEntry]
    gates: Mapping[str, Mapping[str, Any]]
    sealed_receipt: Mapping[str, Any]
    synthetic_stats: Mapping[str, Any]
    real_stats: Mapping[str, Any]
    corrosion_receipt: Mapping[str, Any]
    real_translation: Mapping[str, Any]
    artifact_sha256: Mapping[str, str]
    frozen_authority: Mapping[str, Any]
    terminal_file_sha256: str
    inventory_file_sha256: str
    member_lstat_snapshot: Mapping[str, Mapping[str, int]]


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalReportError(description + " must be an object")
    return value


def _list(value: object, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise FinalReportError(description + " must be an array")
    return value


def _integer(
    value: object, description: str, *, minimum: Optional[int] = None
) -> int:
    if type(value) is not int:
        raise FinalReportError(description + " must be an integer")
    if minimum is not None and value < minimum:
        raise FinalReportError(description + " is below its allowed minimum")
    return value


def _number(
    value: object,
    description: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalReportError(description + " must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise FinalReportError(description + " must be finite")
    if minimum is not None and output < minimum:
        raise FinalReportError(description + " is below its allowed minimum")
    if maximum is not None and output > maximum:
        raise FinalReportError(description + " exceeds its allowed maximum")
    return output


def _boolean(value: object, expected: bool, description: str) -> None:
    if value is not expected:
        raise FinalReportError(description + " differs")


def _same_number(first: float, second: float, description: str) -> None:
    if not math.isclose(first, second, rel_tol=0.0, abs_tol=1e-12):
        raise FinalReportError(description + " point estimates differ")


def _interval(
    value: object,
    description: str,
    *,
    delta: bool = False,
    nonnegative: bool = False,
    unbounded: bool = False,
) -> Mapping[str, Any]:
    row = _mapping(value, description)
    expected = {
        "point_estimate",
        "percentile_95_ci",
        "bootstrap_mean",
        "bootstrap_standard_error",
        "valid_replicates",
    }
    if delta:
        expected.add("probability_delta_gt_zero")
    if set(row) != expected:
        raise FinalReportError(description + " interval fields differ")
    low_bound = None if unbounded else (0.0 if nonnegative else (-1.0 if delta else 0.0))
    high_bound = None if (nonnegative or unbounded) else 1.0
    point = _number(
        row["point_estimate"], description + ".point", minimum=low_bound, maximum=high_bound
    )
    ci = _list(row["percentile_95_ci"], description + ".ci")
    if len(ci) != 2:
        raise FinalReportError(description + " CI must have two endpoints")
    low = _number(ci[0], description + ".ci.low", minimum=low_bound, maximum=high_bound)
    high = _number(ci[1], description + ".ci.high", minimum=low_bound, maximum=high_bound)
    if low > high:
        raise FinalReportError(description + " CI endpoints are reversed")
    mean = _number(
        row["bootstrap_mean"],
        description + ".bootstrap_mean",
        minimum=low_bound,
        maximum=high_bound,
    )
    standard_error = _number(
        row["bootstrap_standard_error"],
        description + ".bootstrap_standard_error",
        minimum=0.0,
    )
    valid = _integer(row["valid_replicates"], description + ".valid_replicates", minimum=1)
    if valid > BOOTSTRAP_REPLICATES:
        raise FinalReportError(description + " has too many bootstrap replicates")
    output: dict[str, Any] = {
        "point_estimate": point,
        "percentile_95_ci": [low, high],
        "bootstrap_mean": mean,
        "bootstrap_standard_error": standard_error,
        "valid_replicates": valid,
    }
    if delta:
        output["probability_delta_gt_zero"] = _number(
            row["probability_delta_gt_zero"],
            description + ".probability_delta_gt_zero",
            minimum=0.0,
            maximum=1.0,
        )
    return output


def _coverage(value: object, total: int, description: str) -> Mapping[str, Any]:
    coverage = _mapping(value, description)
    native = _mapping(coverage.get("native_by_method"), description + ".native")
    if set(native) != set(METHODS):
        raise FinalReportError(description + " native method set differs")

    def normalize(row_value: object, name: str) -> Mapping[str, Any]:
        row = _mapping(row_value, name)
        population = _integer(row.get("population_count"), name + ".population", minimum=1)
        if population != total:
            raise FinalReportError(name + " population count differs")
        valid = _integer(row.get("valid_count"), name + ".valid", minimum=0)
        positive = _integer(row.get("positive_count"), name + ".positive", minimum=0)
        negative = _integer(row.get("negative_count"), name + ".negative", minimum=0)
        positive_valid = _integer(
            row.get("positive_valid_count"), name + ".positive_valid", minimum=0
        )
        negative_valid = _integer(
            row.get("negative_valid_count"), name + ".negative_valid", minimum=0
        )
        fraction = _number(
            row.get("valid_fraction"), name + ".fraction", minimum=0.0, maximum=1.0
        )
        if (
            positive + negative != total
            or positive_valid + negative_valid != valid
            or valid > total
            or positive_valid > positive
            or negative_valid > negative
        ):
            raise FinalReportError(name + " coverage counts are inconsistent")
        _same_number(fraction, valid / total, name + ".valid_fraction")
        return {
            "valid_count": valid,
            "population_count": total,
            "valid_fraction": fraction,
            "positive_valid_count": positive_valid,
            "positive_count": positive,
            "negative_valid_count": negative_valid,
            "negative_count": negative,
        }

    normalized_native = {
        method: normalize(native[method], description + "." + method)
        for method in METHODS
    }
    common = normalize(
        coverage.get("all_method_common_valid"), description + ".common"
    )
    return {"native_by_method": normalized_native, "all_method_common_valid": common}


def _point_metrics(value: object, description: str) -> Mapping[str, Any]:
    point = _mapping(value, description)
    row_count = _integer(point.get("row_count"), description + ".row_count", minimum=1)
    positive = _integer(point.get("positive_count"), description + ".positive", minimum=1)
    negative = _integer(point.get("negative_count"), description + ".negative", minimum=1)
    if positive + negative != row_count:
        raise FinalReportError(description + " class counts differ")
    methods = _mapping(point.get("methods"), description + ".methods")
    if set(methods) != set(METHODS):
        raise FinalReportError(description + " method set differs")
    normalized_methods = {}
    for method in METHODS:
        row = _mapping(methods[method], description + "." + method)
        if set(row) != {"auroc", "auprc"}:
            raise FinalReportError(description + " metric fields differ")
        normalized_methods[method] = {
            metric: _number(
                row[metric],
                description + "." + method + "." + metric,
                minimum=0.0,
                maximum=1.0,
            )
            for metric in ("auroc", "auprc")
        }
    comparisons = _mapping(point.get("comparisons"), description + ".comparisons")
    expected = {"full_n512_minus_" + method for method in COMPARATORS}
    if set(comparisons) != expected:
        raise FinalReportError(description + " comparison set differs")
    normalized_comparisons = {}
    for name in sorted(expected):
        row = _mapping(comparisons[name], description + "." + name)
        if set(row) != {"auroc", "auprc"}:
            raise FinalReportError(description + " comparison metrics differ")
        normalized_comparisons[name] = {
            metric: _number(
                row[metric], description + "." + name + "." + metric, minimum=-1.0, maximum=1.0
            )
            for metric in ("auroc", "auprc")
        }
    return {
        "row_count": row_count,
        "positive_count": positive,
        "negative_count": negative,
        "methods": normalized_methods,
        "comparisons": normalized_comparisons,
    }


def _ranking_population(
    document: Mapping[str, Any], *, total: int, description: str
) -> Mapping[str, Any]:
    if (
        document.get("status") != "complete_formal_exact_six"
        or document.get("formal_evaluation") is not True
        or document.get("compatibility_mode") is not False
        or document.get("methods") != list(METHODS)
    ):
        raise FinalReportError(description + " declared method order differs")
    protocol = _mapping(document.get("protocol"), description + ".protocol")
    if (
        protocol.get("replicates") != BOOTSTRAP_REPLICATES
        or protocol.get("seed") != BOOTSTRAP_SEED
        or protocol.get("delta_direction") != "full_n512 minus comparator"
    ):
        raise FinalReportError(description + " bootstrap protocol differs")
    coverage = _coverage(document.get("coverage"), total, description + ".coverage")
    common = _mapping(document.get("common_population"), description + ".common")
    point = _point_metrics(document.get("point_metrics"), description + ".point")
    for field in ("row_count", "positive_count", "negative_count"):
        if common.get(field) != point[field]:
            raise FinalReportError(description + " common population counts differ")
    if coverage["all_method_common_valid"]["valid_count"] != point["row_count"]:
        raise FinalReportError(description + " common-valid count differs")
    paired = _mapping(
        document.get("paired_endpoint_pigeonhole_bootstrap"),
        description + ".bootstrap",
    )
    requested = _integer(
        paired.get("replicates_requested"), description + ".replicates", minimum=1
    )
    valid = _integer(paired.get("valid_replicates"), description + ".valid", minimum=1)
    skipped = _integer(
        paired.get("skipped_single_class_replicates"), description + ".skipped", minimum=0
    )
    if (
        paired.get("bootstrap") != "endpoint-unit_pigeonhole_product_multiplicity"
        or requested != BOOTSTRAP_REPLICATES
        or valid + skipped != requested
    ):
        raise FinalReportError(description + " paired bootstrap receipt differs")
    metrics = _mapping(paired.get("metrics"), description + ".bootstrap.metrics")
    if set(metrics) != {"auroc", "auprc"}:
        raise FinalReportError(description + " bootstrap metric set differs")
    ranking: dict[str, dict[str, Any]] = {method: {} for method in METHODS}
    deltas: dict[str, dict[str, Any]] = {
        comparator: {} for comparator in COMPARATORS
    }
    for metric in ("auroc", "auprc"):
        metric_row = _mapping(metrics[metric], description + "." + metric)
        method_rows = _mapping(metric_row.get("methods"), description + ".methods")
        delta_rows = _mapping(
            metric_row.get("paired_deltas"), description + ".deltas"
        )
        if set(method_rows) != set(METHODS) or set(delta_rows) != {
            "full_n512_minus_" + comparator for comparator in COMPARATORS
        }:
            raise FinalReportError(description + " bootstrap method/delta set differs")
        for method in METHODS:
            interval = _interval(method_rows[method], description + "." + method + "." + metric)
            _same_number(
                interval["point_estimate"], point["methods"][method][metric], description
            )
            ranking[method][metric] = interval
        for comparator in COMPARATORS:
            name = "full_n512_minus_" + comparator
            interval = _interval(
                delta_rows[name], description + "." + name + "." + metric, delta=True
            )
            _same_number(
                interval["point_estimate"], point["comparisons"][name][metric], description
            )
            deltas[comparator][metric] = interval
    return {
        "source_pair_count": total,
        "common_valid": {
            "row_count": point["row_count"],
            "positive_count": point["positive_count"],
            "negative_count": point["negative_count"],
            "semantic_sha256": common.get("semantic_sha256"),
        },
        "coverage": coverage,
        "ranking": ranking,
        "full_n512_minus_comparator": deltas,
        "bootstrap": {
            "method": paired["bootstrap"],
            "replicates_requested": requested,
            "valid_replicates": valid,
            "skipped_single_class_replicates": skipped,
        },
    }


def _descriptive_population(value: object) -> Mapping[str, Any]:
    document = _mapping(value, "real strict547 descriptive")
    if document.get("role") != "descriptive_only_no_inferential_CI":
        raise FinalReportError("strict547 must remain descriptive-only")
    coverage = _coverage(document.get("coverage"), 547, "strict547.coverage")
    point = _point_metrics(document.get("point_metrics"), "strict547.point")
    if coverage["all_method_common_valid"]["valid_count"] != point["row_count"]:
        raise FinalReportError("strict547 common-valid count differs")
    population_sha = document.get("common_population_sha256")
    if not _is_sha256(population_sha):
        raise FinalReportError("strict547 common population SHA-256 is invalid")
    return {
        "source_pair_count": 547,
        "role": "descriptive_only_no_inferential_CI",
        "common_valid": {
            "row_count": point["row_count"],
            "positive_count": point["positive_count"],
            "negative_count": point["negative_count"],
            "semantic_sha256": population_sha,
        },
        "coverage": coverage,
        "ranking_point_estimates": point["methods"],
        "full_n512_minus_comparator_point_estimates": {
            comparator: point["comparisons"]["full_n512_minus_" + comparator]
            for comparator in COMPARATORS
        },
        "inferential_confidence_intervals": None,
    }


def _synthetic_geometry(value: object) -> Mapping[str, Any]:
    geometry = _mapping(value, "synthetic geometry")
    if set(geometry) != {
        "scope",
        "bootstrap",
        "sampling_dependency_unit_count",
        "shared_draws_across_all_direct_geometry_metrics",
        "validation_threshold",
        "threshold_source",
        "full_n512",
        "full_n512_minus_coarse_only",
    }:
        raise FinalReportError("synthetic geometry schema differs")
    if (
        geometry.get("scope")
        != "full_n512_geometry_on_entire_sealed_synthetic_population"
        or geometry.get("bootstrap")
        != "endpoint-unit_pigeonhole_product_multiplicity"
        or geometry.get("sampling_dependency_unit_count") != 44
        or geometry.get("shared_draws_across_all_direct_geometry_metrics")
        is not True
        or geometry.get("threshold_source")
        != "frozen_validation_checkpoint_bound_artifact"
    ):
        raise FinalReportError("synthetic geometry protocol differs")
    threshold = _number(
        geometry.get("validation_threshold"),
        "synthetic geometry threshold",
        minimum=0.0,
        maximum=1.0,
    )
    full = _mapping(geometry.get("full_n512"), "synthetic full geometry")
    if set(full) != {
        "translation_l2_px",
        "correspondence",
        "pairingnet_style_registration",
        "assembly_edge_at_validation_threshold",
        "joint_success_at_validation_threshold",
    }:
        raise FinalReportError("synthetic full geometry schema differs")
    translation = _mapping(
        full.get("translation_l2_px"), "synthetic translation"
    )
    if set(translation) != {
        "definition",
        "eligible_positive_count",
        "valid_translation_count",
        "valid_translation_fraction",
        "median",
        "p90",
        "recall_definition",
        "recall_by_tolerance",
        "success_by_tolerance",
        "bootstrap_replicates_without_valid_translation",
    }:
        raise FinalReportError("synthetic translation schema differs")
    if (
        translation.get("definition")
        != "positive and full decision-valid; conditional on a finite translation"
        or translation.get("recall_definition")
        != "denominator is every positive pair; decision-invalid or missing translation is a failure"
    ):
        raise FinalReportError("synthetic translation definitions differ")
    eligible = _integer(
        translation.get("eligible_positive_count"),
        "synthetic eligible positives",
        minimum=1,
    )
    valid = _integer(
        translation.get("valid_translation_count"),
        "synthetic valid translations",
        minimum=1,
    )
    if eligible != 1500 or valid > eligible:
        raise FinalReportError("synthetic translation population differs")
    fraction = _number(
        translation.get("valid_translation_fraction"),
        "synthetic translation fraction",
        minimum=0.0,
        maximum=1.0,
    )
    _same_number(fraction, valid / eligible, "synthetic translation fraction")
    median = _interval(
        translation.get("median"), "synthetic median translation error", nonnegative=True
    )
    p90 = _interval(
        translation.get("p90"), "synthetic p90 translation error", nonnegative=True
    )
    success = _mapping(
        translation.get("success_by_tolerance"), "synthetic translation success"
    )
    expected_tolerances = {"success_at_{}px".format(item) for item in TOLERANCES}
    if set(success) != expected_tolerances:
        raise FinalReportError("synthetic translation tolerances differ")
    normalized_success = {
        name: _interval(success[name], "synthetic " + name)
        for name in sorted(expected_tolerances)
    }
    recall = _mapping(
        translation.get("recall_by_tolerance"), "synthetic translation recall"
    )
    expected_recall = {"recall_at_{}px".format(item) for item in TOLERANCES}
    if set(recall) != expected_recall:
        raise FinalReportError("synthetic translation recall tolerances differ")
    normalized_recall = {
        name: _interval(recall[name], "synthetic " + name)
        for name in sorted(expected_recall)
    }
    translation_skipped = _integer(
        translation.get("bootstrap_replicates_without_valid_translation"),
        "synthetic translation skipped replicates",
        minimum=0,
    )
    if translation_skipped > BOOTSTRAP_REPLICATES or any(
        row["valid_replicates"] + translation_skipped != BOOTSTRAP_REPLICATES
        for row in (median, p90, *normalized_success.values())
    ):
        raise FinalReportError("synthetic translation bootstrap counts differ")
    success_points = [
        normalized_success["success_at_{}px".format(tolerance)]["point_estimate"]
        for tolerance in TOLERANCES
    ]
    recall_points = [
        normalized_recall["recall_at_{}px".format(tolerance)]["point_estimate"]
        for tolerance in TOLERANCES
    ]
    if success_points != sorted(success_points) or recall_points != sorted(
        recall_points
    ):
        raise FinalReportError("synthetic translation tolerances are non-monotone")
    for success_point, recall_point in zip(success_points, recall_points):
        _same_number(
            recall_point,
            success_point * fraction,
            "synthetic conditional/unconditional translation",
        )

    def normalize_prf(value: object, description: str) -> Mapping[str, Any]:
        row = _mapping(value, description)
        if set(row) != {"point_counts", "precision", "recall", "harmonic_f1"}:
            raise FinalReportError(description + " PRF schema differs")
        counts = _mapping(row["point_counts"], description + ".counts")
        if set(counts) != {"true_positive", "predicted_count", "target_count"}:
            raise FinalReportError(description + " PRF count schema differs")
        true_positive = _integer(
            counts["true_positive"], description + ".tp", minimum=0
        )
        predicted = _integer(
            counts["predicted_count"], description + ".predicted", minimum=0
        )
        target = _integer(counts["target_count"], description + ".target", minimum=0)
        if true_positive > predicted or true_positive > target:
            raise FinalReportError(description + " PRF counts are inconsistent")
        normalized = {
            metric: _interval(row[metric], description + "." + metric)
            for metric in ("precision", "recall", "harmonic_f1")
        }
        precision = true_positive / predicted if predicted else 0.0
        recall_point = true_positive / target if target else 0.0
        harmonic = (
            2.0 * precision * recall_point / (precision + recall_point)
            if precision + recall_point
            else 0.0
        )
        for metric, point in (
            ("precision", precision),
            ("recall", recall_point),
            ("harmonic_f1", harmonic),
        ):
            _same_number(
                normalized[metric]["point_estimate"],
                point,
                description + "." + metric,
            )
        return {
            "point_counts": {
                "true_positive": true_positive,
                "predicted_count": predicted,
                "target_count": target,
            },
            **normalized,
        }

    correspondence = _mapping(
        full.get("correspondence"), "synthetic correspondence"
    )
    if set(correspondence) != {
        "scope",
        "strict_dustbin_aware",
        "mutual_top1",
        "dustbin_accuracy",
    }:
        raise FinalReportError("synthetic correspondence schema differs")
    if (
        correspondence.get("scope")
        != "entire sealed population; invalid decisions preserve targets and contribute no predicted matches/correct dustbins"
    ):
        raise FinalReportError("synthetic correspondence scope differs")
    strict = normalize_prf(
        correspondence.get("strict_dustbin_aware"),
        "synthetic strict correspondence",
    )
    mutual = normalize_prf(
        correspondence.get("mutual_top1"), "synthetic mutual correspondence"
    )
    dustbin = _mapping(
        correspondence.get("dustbin_accuracy"), "synthetic dustbin accuracy"
    )
    if set(dustbin) != {"point_counts", "accuracy"}:
        raise FinalReportError("synthetic dustbin schema differs")
    dustbin_counts = _mapping(
        dustbin["point_counts"], "synthetic dustbin counts"
    )
    if set(dustbin_counts) != {"correct", "token_count"}:
        raise FinalReportError("synthetic dustbin count schema differs")
    dustbin_correct = _integer(
        dustbin_counts["correct"], "synthetic dustbin correct", minimum=0
    )
    dustbin_total = _integer(
        dustbin_counts["token_count"], "synthetic dustbin total", minimum=1
    )
    if dustbin_correct > dustbin_total:
        raise FinalReportError("synthetic dustbin counts are inconsistent")
    dustbin_interval = _interval(
        dustbin["accuracy"], "synthetic dustbin accuracy"
    )
    _same_number(
        dustbin_interval["point_estimate"],
        dustbin_correct / dustbin_total,
        "synthetic dustbin accuracy",
    )

    registration = _mapping(
        full.get("pairingnet_style_registration"),
        "synthetic PairingNet-style registration",
    )
    if set(registration) != {
        "compatibility_source",
        "eligible_positive_count",
        "evaluated_positive_count",
        "valid_pose_count",
        "identity_fallback_count",
        "valid_pose_fraction",
        "invalid_pose_compatibility_fallback",
        "e_rmse_definition",
        "registration_recall_definition",
        "hausdorff_definition",
        "normalized_translation_error_definition",
        "mean_e_rmse",
        "registration_recall_e_rmse_lt4",
        "mean_symmetric_hausdorff_px",
        "mean_normalized_translation_error",
        "bootstrap_replicates_without_evaluable_positive",
        "rotation_error",
    }:
        raise FinalReportError("synthetic registration schema differs")
    if (
        registration.get("compatibility_source")
        != "PairingNet released matching_test.py, specialized to frozen upright translation-only Dunhuang inputs"
        or registration.get("e_rmse_definition")
        != "sqrt(mean(per_correspondence_euclidean_distance)); official compatibility definition, not conventional RMSE"
        or registration.get("registration_recall_definition")
        != "e_rmse_strictly_less_than_4"
        or registration.get("hausdorff_definition")
        != "max(directed_HD(transformed_source_seam,target_seam),directed_HD(target_seam,transformed_source_seam))"
        or registration.get("normalized_translation_error_definition")
        != "identity-fallback translation_l2_px divided by the sum of ordered N512 contour polygon areas after PairingNet int32 quantization"
        or registration.get("invalid_pose_compatibility_fallback")
        != "identity_translation_for_unconditional_official_style_aggregation"
    ):
        raise FinalReportError("synthetic registration definitions differ")
    registration_eligible = _integer(
        registration.get("eligible_positive_count"),
        "synthetic registration eligible",
        minimum=1,
    )
    registration_valid = _integer(
        registration.get("valid_pose_count"),
        "synthetic registration valid",
        minimum=1,
    )
    registration_evaluated = _integer(
        registration.get("evaluated_positive_count"),
        "synthetic registration evaluated",
        minimum=1,
    )
    identity_fallback = _integer(
        registration.get("identity_fallback_count"),
        "synthetic registration identity fallback",
        minimum=0,
    )
    registration_fraction = _number(
        registration.get("valid_pose_fraction"),
        "synthetic registration valid fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if (
        registration_eligible != 1500
        or registration_evaluated != registration_eligible
        or registration_valid + identity_fallback != registration_eligible
    ):
        raise FinalReportError("synthetic registration population differs")
    _same_number(
        registration_fraction,
        registration_valid / registration_eligible,
        "synthetic registration valid fraction",
    )
    rotation = _mapping(
        registration.get("rotation_error"), "synthetic rotation marker"
    )
    if (
        rotation.get("status")
        != "not_applicable_conditioned_upright_orientation"
        or rotation.get("estimated_or_supervised") is not False
    ):
        raise FinalReportError("synthetic upright rotation marker differs")
    normalized_registration = {
        "eligible_positive_count": registration_eligible,
        "evaluated_positive_count": registration_evaluated,
        "valid_pose_count": registration_valid,
        "identity_fallback_count": identity_fallback,
        "valid_pose_fraction": registration_fraction,
        "invalid_pose_compatibility_fallback": str(
            registration["invalid_pose_compatibility_fallback"]
        ),
        "mean_e_rmse": _interval(
            registration.get("mean_e_rmse"),
            "synthetic mean eRMSE",
            nonnegative=True,
        ),
        "registration_recall_e_rmse_lt4": _interval(
            registration.get("registration_recall_e_rmse_lt4"),
            "synthetic PairingNet RR",
        ),
        "mean_symmetric_hausdorff_px": _interval(
            registration.get("mean_symmetric_hausdorff_px"),
            "synthetic symmetric HD",
            nonnegative=True,
        ),
        "mean_normalized_translation_error": _interval(
            registration.get("mean_normalized_translation_error"),
            "synthetic NTE",
            nonnegative=True,
        ),
        "rotation_error": dict(rotation),
        "e_rmse_definition": str(registration["e_rmse_definition"]),
        "registration_recall_definition": str(
            registration["registration_recall_definition"]
        ),
        "hausdorff_definition": str(registration["hausdorff_definition"]),
        "normalized_translation_error_definition": str(
            registration["normalized_translation_error_definition"]
        ),
    }
    registration_skipped = _integer(
        registration.get("bootstrap_replicates_without_evaluable_positive"),
        "synthetic registration skipped replicates",
        minimum=0,
    )
    if registration_skipped > BOOTSTRAP_REPLICATES or any(
        normalized_registration[name]["valid_replicates"] + registration_skipped
        != BOOTSTRAP_REPLICATES
        for name in (
            "mean_e_rmse",
            "mean_symmetric_hausdorff_px",
            "mean_normalized_translation_error",
        )
    ):
        raise FinalReportError("synthetic registration bootstrap counts differ")

    assembly = _mapping(
        full.get("assembly_edge_at_validation_threshold"),
        "synthetic assembly edge",
    )
    if set(assembly) != {
        "definition",
        "predicted_edge_count",
        "target_edge_count",
        "by_tolerance",
    }:
        raise FinalReportError("synthetic assembly schema differs")
    if (
        assembly.get("definition")
        != "predicted edge requires decision-valid and frozen-threshold pair acceptance; true positive additionally requires adjacent GT and translation L2 error at or below the pixel tolerance"
    ):
        raise FinalReportError("synthetic assembly definition differs")
    predicted_edges = _integer(
        assembly.get("predicted_edge_count"),
        "synthetic predicted assembly edges",
        minimum=0,
    )
    target_edges = _integer(
        assembly.get("target_edge_count"),
        "synthetic target assembly edges",
        minimum=1,
    )
    if target_edges != 1500:
        raise FinalReportError("synthetic assembly target count differs")
    assembly_rows = _mapping(
        assembly.get("by_tolerance"), "synthetic assembly tolerances"
    )
    expected_assembly = {
        "at_{}".format(value) for value in ASSEMBLY_EDGE_TOLERANCES
    }
    if set(assembly_rows) != expected_assembly:
        raise FinalReportError("synthetic assembly tolerances differ")
    normalized_assembly = {
        name: normalize_prf(assembly_rows[name], "synthetic assembly " + name)
        for name in sorted(expected_assembly)
    }
    if any(
        row["point_counts"]["predicted_count"] != predicted_edges
        or row["point_counts"]["target_count"] != target_edges
        for row in normalized_assembly.values()
    ):
        raise FinalReportError("synthetic assembly point counts differ")
    assembly_true_positives = [
        normalized_assembly["at_{}".format(tolerance)]["point_counts"][
            "true_positive"
        ]
        for tolerance in ASSEMBLY_EDGE_TOLERANCES
    ]
    if assembly_true_positives != sorted(assembly_true_positives):
        raise FinalReportError("synthetic assembly tolerance outcomes are non-monotone")
    joint = _mapping(
        full.get("joint_success_at_validation_threshold"), "synthetic joint geometry"
    )
    if set(joint) != {
        "definition",
        "denominator_positive_count",
        "success_by_tolerance",
        "bootstrap_replicates_without_positive",
    }:
        raise FinalReportError("synthetic joint schema differs")
    if (
        joint.get("definition")
        != "denominator is every positive pair; success requires full decision-valid, score >= frozen validation threshold, and translation error <= tolerance"
        or joint.get("denominator_positive_count") != 1500
    ):
        raise FinalReportError("synthetic joint denominator differs")
    joint_success = _mapping(
        joint.get("success_by_tolerance"), "synthetic joint success"
    )
    if set(joint_success) != expected_tolerances:
        raise FinalReportError("synthetic joint tolerances differ")
    positive_skipped = _integer(
        joint.get("bootstrap_replicates_without_positive"),
        "synthetic positive skipped replicates",
        minimum=0,
    )
    if positive_skipped > BOOTSTRAP_REPLICATES:
        raise FinalReportError("synthetic positive bootstrap count differs")
    normalized_joint = {
        name: _interval(joint_success[name], "synthetic joint " + name)
        for name in sorted(expected_tolerances)
    }
    if any(
        interval["valid_replicates"] + positive_skipped != BOOTSTRAP_REPLICATES
        for interval in (
            *normalized_recall.values(),
            strict["precision"],
            strict["recall"],
            strict["harmonic_f1"],
            mutual["precision"],
            mutual["recall"],
            mutual["harmonic_f1"],
            dustbin_interval,
            normalized_registration["registration_recall_e_rmse_lt4"],
            *(
                metric
                for row in normalized_assembly.values()
                for metric in (
                    row["precision"],
                    row["recall"],
                    row["harmonic_f1"],
                )
            ),
            *normalized_joint.values(),
        )
    ):
        raise FinalReportError("synthetic direct-metric bootstrap counts differ")
    comparator = _mapping(
        geometry.get("full_n512_minus_coarse_only"), "synthetic geometry comparator"
    )
    if (
        set(comparator) != {"status", "reason"}
        or comparator.get("status") != "not_applicable"
        or comparator.get("reason")
        != "coarse_only emits pair scores but no correspondence or 2D translation, so translation and joint geometric deltas are undefined"
    ):
        raise FinalReportError("coarse geometry must be not-applicable")
    return {
        "scope": geometry["scope"],
        "frozen_validation_threshold": threshold,
        "correspondence": {
            "strict_dustbin_aware": strict,
            "mutual_top1": mutual,
            "dustbin_accuracy": {
                "point_counts": {
                    "correct": dustbin_correct,
                    "token_count": dustbin_total,
                },
                "accuracy": dustbin_interval,
            },
        },
        "pairingnet_style_registration": normalized_registration,
        "assembly_edge_at_frozen_validation_threshold": {
            "predicted_edge_count": predicted_edges,
            "target_edge_count": target_edges,
            "by_tolerance": normalized_assembly,
        },
        "translation_l2_px": {
            "eligible_positive_count": eligible,
            "valid_translation_count": valid,
            "valid_translation_fraction": fraction,
            "median": median,
            "p90": p90,
            "recall_by_tolerance": normalized_recall,
            "success_by_tolerance": normalized_success,
        },
        "joint_frozen_pair_and_translation_success": {
            "denominator_positive_count": 1500,
            "success_by_tolerance": normalized_joint,
        },
        "coarse_and_matched_translation_comparison": "not_applicable",
    }


def _direct_prf_point(value: object, description: str) -> Mapping[str, Any]:
    row = _mapping(value, description)
    expected = {
        "true_positive_count",
        "predicted_count",
        "target_count",
        "false_positive_count",
        "false_negative_count",
        "precision",
        "recall",
        "f1",
    }
    if set(row) != expected:
        raise FinalReportError(description + " direct PRF fields differ")
    true_positive = _integer(
        row["true_positive_count"], description + ".tp", minimum=0
    )
    predicted = _integer(row["predicted_count"], description + ".predicted", minimum=0)
    target = _integer(row["target_count"], description + ".target", minimum=0)
    false_positive = _integer(
        row["false_positive_count"], description + ".fp", minimum=0
    )
    false_negative = _integer(
        row["false_negative_count"], description + ".fn", minimum=0
    )
    precision = _number(row["precision"], description + ".precision", minimum=0, maximum=1)
    recall = _number(row["recall"], description + ".recall", minimum=0, maximum=1)
    f1 = _number(row["f1"], description + ".f1", minimum=0, maximum=1)
    if (
        true_positive + false_positive != predicted
        or true_positive + false_negative != target
    ):
        raise FinalReportError(description + " direct PRF counts differ")
    expected_precision = true_positive / predicted if predicted else 0.0
    expected_recall = true_positive / target if target else 0.0
    expected_f1 = (
        2.0 * expected_precision * expected_recall
        / (expected_precision + expected_recall)
        if expected_precision + expected_recall
        else 0.0
    )
    _same_number(precision, expected_precision, description + ".precision")
    _same_number(recall, expected_recall, description + ".recall")
    _same_number(f1, expected_f1, description + ".f1")
    return {
        "point_counts": {
            "true_positive": true_positive,
            "predicted_count": predicted,
            "target_count": target,
        },
        "precision": precision,
        "recall": recall,
        "harmonic_f1": f1,
        "confidence_interval": None,
    }


def _synthetic_direct_exact_six(
    sealed_receipt: Mapping[str, Any],
    full_geometry: Mapping[str, Any],
) -> Mapping[str, Any]:
    rows = sealed_receipt.get("arm_results")
    if not isinstance(rows, list) or [
        row.get("arm") for row in rows if isinstance(row, Mapping)
    ] != list(METHODS):
        raise FinalReportError("sealed direct-metric exact-six inventory differs")
    by_method = {str(row["arm"]): row for row in rows}
    output: dict[str, Mapping[str, Any]] = {}
    for method in NON_TRANSLATION_METHODS:
        output[method] = {
            "status": "not_applicable",
            "reason": "method_does_not_emit_a_supervised_2d_translation",
            "translation": None,
            "assembly_edge": None,
            "pairingnet_style_registration": None,
            "correspondence": None,
        }
    full_assembly = _mapping(
        full_geometry.get("assembly_edge_at_frozen_validation_threshold"),
        "full synthetic assembly edge",
    )
    full_tolerances = _mapping(
        full_assembly.get("by_tolerance"), "full synthetic assembly tolerances"
    )
    output["full_n512"] = {
        "status": "evaluated_direct_pairwise_geometry",
        "translation": dict(
            _mapping(full_geometry.get("translation_l2_px"), "full synthetic translation")
        ),
        "assembly_edge": {
            "predicted_edge_count": full_assembly.get("predicted_edge_count"),
            "target_edge_count": full_assembly.get("target_edge_count"),
            "by_tolerance": {
                "at_{}px".format(tolerance): full_tolerances[
                    "at_{}".format(tolerance)
                ]
                for tolerance in TOLERANCES
            },
        },
        "pairingnet_style_registration": dict(
            _mapping(
                full_geometry.get("pairingnet_style_registration"),
                "full synthetic registration",
            )
        ),
        "correspondence": dict(
            _mapping(full_geometry.get("correspondence"), "full synthetic correspondence")
        ),
        "native_global_assembly_GA": {
            "status": "not_applicable",
            "reason": "pairwise_only_no_global_placement",
        },
        "shreddingnet_native_CM_FM_SE": {
            "status": "not_reported",
            "reason": "not_a_native_exhaustive_same-parent_candidate_graph",
        },
    }
    for method in BENCHMARK_METHODS:
        arm = _mapping(by_method[method], method + " sealed direct row")
        metrics = _mapping(arm.get("metrics"), method + " direct metrics")
        scope = _mapping(metrics.get("scope"), method + " direct scope")
        unavailable = _mapping(
            metrics.get("unavailable_or_not_applicable"),
            method + " unavailable native metrics",
        )
        threshold = _mapping(metrics.get("threshold"), method + " direct threshold")
        translation = _mapping(metrics.get("translation"), method + " direct translation")
        assembly = _mapping(metrics.get("assembly_edge"), method + " direct assembly")
        assembly_rows = _mapping(
            assembly.get("by_tolerance"), method + " direct assembly tolerances"
        )
        registration = _mapping(
            metrics.get("pairingnet_style_registration"),
            method + " direct registration",
        )
        correspondence = _mapping(
            metrics.get("correspondence"), method + " direct correspondence"
        )
        native_ga = _mapping(
            unavailable.get("native_global_assembly_GA"), method + " native GA"
        )
        native_shred = _mapping(
            unavailable.get("shreddingnet_native_CM_FM_SE"),
            method + " native CM/FM/SE",
        )
        if (
            metrics.get("schema_version")
            != "rachel-common-direct-pairwise-report/1.0"
            or metrics.get("status") != "complete_frozen_direct_pairwise_evaluation"
            or metrics.get("method_key") != method
            or metrics.get("pair_count") != 3000
            or metrics.get("positive_count") != 1500
            or metrics.get("negative_count") != 1500
            or threshold.get("source") != "frozen_validation_only_artifact"
            or threshold.get("fit_performed_here") is not False
            or threshold.get("artifact_sha256")
            != arm.get("validation_threshold_sha256")
            or scope.get("same_data_method_adaptation_not_exact_reproduction")
            is not True
            or scope.get("global_assembly_performed") is not False
            or scope.get("native_cm_fm_se_or_ga_claimed") is not False
            or set(assembly_rows)
            != {"at_{}px".format(tolerance) for tolerance in TOLERANCES}
            or registration.get("rotation_error", {}).get("status")
            != "not_applicable"
            or native_ga.get("status") != "not_applicable"
            or native_shred.get("status") != "not_reported"
            or arm.get("adaptation_claim")
            != "same_data_method_adaptation_not_exact_reproduction"
            or arm.get("native_cm_fm_se_or_ga_claimed") is not False
        ):
            raise FinalReportError(method + " synthetic direct/native claim differs")
        conditioned = _mapping(
            translation.get("te_px_conditioned_on_valid_pose"),
            method + " conditioned TE",
        )
        recall = _mapping(
            translation.get("unconditional_positive_recall"),
            method + " direct translation recall",
        )
        if set(recall) != {"at_{}px".format(tolerance) for tolerance in TOLERANCES}:
            raise FinalReportError(method + " direct translation tolerance set differs")
        eligible = _integer(
            translation.get("eligible_positive_count"),
            method + " eligible positives",
            minimum=1,
        )
        valid_translation = _integer(
            translation.get("valid_prediction_count"),
            method + " valid translations",
            minimum=0,
        )
        valid_fraction = _number(
            translation.get("valid_prediction_fraction"),
            method + " valid translation fraction",
            minimum=0,
            maximum=1,
        )
        if (
            eligible != 1500
            or valid_translation > eligible
            or conditioned.get("count") != valid_translation
        ):
            raise FinalReportError(method + " direct translation counts differ")
        _same_number(
            valid_fraction,
            valid_translation / eligible,
            method + " valid translation fraction",
        )
        if (valid_translation == 0) != (
            conditioned.get("median") is None and conditioned.get("p90") is None
        ):
            raise FinalReportError(method + " conditioned TE availability differs")
        predicted_edges = _integer(
            assembly.get("predicted_edge_count"),
            method + " predicted edges",
            minimum=0,
        )
        target_edges = _integer(
            assembly.get("target_edge_count"),
            method + " target edges",
            minimum=0,
        )
        if target_edges != eligible:
            raise FinalReportError(method + " direct assembly target count differs")
        normalized_assembly = {
            name: _direct_prf_point(
                assembly_rows[name], method + " synthetic assembly " + name
            )
            for name in sorted(assembly_rows)
        }
        if any(
            row["point_counts"]["predicted_count"] != predicted_edges
            or row["point_counts"]["target_count"] != target_edges
            for row in normalized_assembly.values()
        ):
            raise FinalReportError(method + " direct assembly outer counts differ")
        registration_eligible = _integer(
            registration.get("eligible_positive_count"),
            method + " registration eligible",
            minimum=1,
        )
        registration_valid = _integer(
            registration.get("valid_pose_count"),
            method + " registration valid",
            minimum=0,
        )
        registration_fallback = _integer(
            registration.get("identity_fallback_count"),
            method + " registration fallback",
            minimum=0,
        )
        if (
            registration_eligible != eligible
            or registration_valid + registration_fallback != eligible
        ):
            raise FinalReportError(method + " registration coverage differs")
        _number(
            registration.get("rr_lt4"),
            method + " registration RR",
            minimum=0,
            maximum=1,
        )
        for field in (
            "mean_e_rmse",
            "mean_symmetric_hausdorff_px",
            "mean_normalized_translation_error",
        ):
            _number(registration.get(field), method + "." + field, minimum=0)
        correspondence_status = correspondence.get("status")
        if correspondence_status == "reported":
            _direct_prf_point(
                correspondence.get("thresholded_exact"),
                method + " thresholded correspondence",
            )
            _direct_prf_point(
                correspondence.get("reciprocal_top1_exact"),
                method + " reciprocal correspondence",
            )
            if correspondence.get("dustbin_aware", {}).get("status") != "not_applicable":
                raise FinalReportError(method + " dustbin comparability differs")
        elif (
            correspondence_status != "not_applicable"
            or correspondence.get("metrics") is not None
        ):
            raise FinalReportError(method + " correspondence availability differs")
        normalized_translation = {
            "eligible_positive_count": eligible,
            "valid_translation_count": valid_translation,
            "valid_translation_fraction": valid_fraction,
            "median_l2_px_conditioned_on_valid": (
                None
                if conditioned.get("median") is None
                else _number(conditioned.get("median"), method + " median TE", minimum=0)
            ),
            "p90_l2_px_conditioned_on_valid": (
                None
                if conditioned.get("p90") is None
                else _number(conditioned.get("p90"), method + " p90 TE", minimum=0)
            ),
            "unconditional_positive_recall": {
                name: _number(
                    recall[name], method + "." + name, minimum=0, maximum=1
                )
                for name in sorted(recall)
            },
            "confidence_intervals": None,
        }
        output[method] = {
            "status": "evaluated_direct_pairwise_geometry",
            "translation": normalized_translation,
            "assembly_edge": {
                "predicted_edge_count": predicted_edges,
                "target_edge_count": target_edges,
                "by_tolerance": normalized_assembly,
            },
            "pairingnet_style_registration": dict(registration),
            "correspondence": dict(correspondence),
            "native_global_assembly_GA": dict(native_ga),
            "shreddingnet_native_CM_FM_SE": dict(native_shred),
            "same_data_method_adaptation_not_exact_reproduction": True,
        }
    return {
        "method_inventory": list(METHODS),
        "translation_emitting_methods": list(TRANSLATION_METHODS),
        "methods": {method: output[method] for method in METHODS},
        "same_data_benchmarks_are_adaptations_not_exact_reproductions": True,
        "native_cm_fm_se_or_ga_claimed": False,
        "automatic_performance_pass_fail_applied": False,
    }


def _case_interval(
    value: object, description: str, *, nonnegative: bool = False
) -> Mapping[str, Any]:
    row = _mapping(value, description)
    if set(row) != {
        "estimate",
        "ci95_low",
        "ci95_high",
        "valid_bootstrap_replicates",
    }:
        raise FinalReportError(description + " case-bootstrap fields differ")
    minimum = 0.0
    maximum = None if nonnegative else 1.0
    estimate = _number(
        row["estimate"], description + ".estimate", minimum=minimum, maximum=maximum
    )
    low = _number(row["ci95_low"], description + ".low", minimum=minimum, maximum=maximum)
    high = _number(
        row["ci95_high"], description + ".high", minimum=minimum, maximum=maximum
    )
    if low > high:
        raise FinalReportError(description + " CI endpoints are reversed")
    valid = _integer(
        row["valid_bootstrap_replicates"], description + ".valid", minimum=1
    )
    if valid > BOOTSTRAP_REPLICATES:
        raise FinalReportError(description + " has too many replicates")
    return {
        "point_estimate": estimate,
        "percentile_95_ci": [low, high],
        "valid_replicates": valid,
    }


def _real_translation(value: Mapping[str, Any]) -> Mapping[str, Any]:
    population = _mapping(value.get("population"), "real translation population")
    if (
        population.get("strict_pair_count") != 547
        or population.get("positive_pair_count") != 508
        or population.get("negative_pairs_have_translation_gt") is not False
        or population.get("balanced_constructed_pairs_used_for_translation_or_GT")
        is not False
        or population.get(
            "balanced_constructed_pairs_claimed_GT_negative"
        )
        is not False
    ):
        raise FinalReportError("real translation population differs")
    methods = _mapping(value.get("method_metrics"), "real translation methods")
    if list(methods) != list(METHODS):
        raise FinalReportError("real translation method set differs")
    expected = {
        "valid_translation_prediction_fraction",
        "median_l2_px",
        "p90_l2_px",
        *("recall_at_{}px".format(item) for item in TOLERANCES),
        *(
            "joint_frozen_threshold_and_translation_recall_at_{}px".format(item)
            for item in TOLERANCES
        ),
    }
    normalized_methods: dict[str, Mapping[str, Any]] = {}
    for method in METHODS:
        row = _mapping(methods[method], "real translation " + method)
        if method in NON_TRANSLATION_METHODS:
            if (
                row.get("status") != "not_applicable"
                or row.get("translation_metrics") is not None
            ):
                raise FinalReportError(method + " translation must be not-applicable")
            normalized_methods[method] = {
                "status": "not_applicable",
                "reason": row.get("reason"),
                "translation_metrics": None,
                "assembly_edge_strict_547": None,
                "pairingnet_style_registration": None,
                "correspondence": None,
            }
            continue
        if (
            row.get("status") != "evaluated_positive_translation_gt"
            or row.get("eligible_positive_count") != 508
            or row.get("scope")
            != "strict_547_positive_pairs_plus_strict_and_balanced_edge_views"
            or row.get("threshold_fit_performed_here") is not False
            or row.get("invalid_translation_predictions_count_as_recall_failures")
            is not True
        ):
            raise FinalReportError(method + " real direct translation contract differs")
        threshold = _number(
            row.get("frozen_validation_threshold"),
            method + " real frozen threshold",
            minimum=0.0,
            maximum=1.0,
        )
        bootstrap = _mapping(
            row.get("case_bootstrap"), method + " real translation bootstrap"
        )
        if (
            bootstrap.get("schema") != "case_cluster_percentile_bootstrap/1.0"
            or bootstrap.get("seed") != REAL_TRANSLATION_BOOTSTRAP_SEED
            or bootstrap.get("repetitions") != BOOTSTRAP_REPLICATES
            or bootstrap.get("sampling_unit") != "authoritative_real_case_uid"
            or bootstrap.get("same_case_pairs_keep_endpoints_together") is not True
        ):
            raise FinalReportError(method + " real translation bootstrap differs")
        metrics = _mapping(
            bootstrap.get("metrics"), method + " real translation metrics"
        )
        point = _mapping(
            row.get("point_estimates"), method + " real translation points"
        )
        if set(metrics) != expected or set(point) != expected:
            raise FinalReportError(method + " real translation metric set differs")
        normalized = {
            name: _case_interval(
                metrics[name],
                method + " real translation " + name,
                nonnegative=name in {"median_l2_px", "p90_l2_px"},
            )
            for name in sorted(expected)
        }
        for name in expected:
            _same_number(
                normalized[name]["point_estimate"],
                _number(
                    point[name],
                    method + " real translation point " + name,
                    minimum=0.0,
                    maximum=(
                        None if name in {"median_l2_px", "p90_l2_px"} else 1.0
                    ),
                ),
                method + " real translation " + name,
            )
        assembly = _mapping(
            row.get("assembly_edge_strict_547"), method + " real assembly edge"
        )
        assembly_rows = _mapping(
            assembly.get("by_tolerance"), method + " real assembly tolerances"
        )
        registration = _mapping(
            row.get("pairingnet_style_registration"),
            method + " PairingNet-style registration",
        )
        correspondence = _mapping(
            row.get("correspondence"), method + " real correspondence"
        )
        native_ga = _mapping(
            row.get("native_global_assembly_GA"), method + " native GA disclosure"
        )
        native_shred = _mapping(
            row.get("shreddingnet_native_CM_FM_SE"),
            method + " native CM/FM/SE disclosure",
        )
        if set(assembly_rows) != {
            "at_{}px".format(tolerance) for tolerance in TOLERANCES
        }:
            raise FinalReportError(method + " real assembly tolerance set differs")
        predicted_edges = _integer(
            assembly.get("predicted_edge_count"),
            method + " real predicted edges",
            minimum=0,
        )
        target_edges = _integer(
            assembly.get("target_edge_count"),
            method + " real target edges",
            minimum=0,
        )
        normalized_assembly = {
            name: _direct_prf_point(
                assembly_rows[name], method + " real assembly " + name
            )
            for name in sorted(assembly_rows)
        }
        if target_edges != 508 or any(
            value["point_counts"]["predicted_count"] != predicted_edges
            or value["point_counts"]["target_count"] != target_edges
            for value in normalized_assembly.values()
        ):
            raise FinalReportError(method + " real assembly counts differ")
        registration_valid = _integer(
            registration.get("valid_pose_count"),
            method + " real registration valid",
            minimum=0,
        )
        registration_fallback = _integer(
            registration.get("identity_fallback_count"),
            method + " real registration fallback",
            minimum=0,
        )
        if registration_valid + registration_fallback != 508:
            raise FinalReportError(method + " real registration coverage differs")
        _number(
            registration.get("rr_lt4"),
            method + " real registration RR",
            minimum=0,
            maximum=1,
        )
        for field in (
            "mean_e_rmse",
            "mean_symmetric_hausdorff_px",
            "mean_normalized_translation_error",
        ):
            _number(registration.get(field), method + "." + field, minimum=0)
        if (
            correspondence.get("status") != "not_applicable"
            or native_ga.get("status") != "not_applicable"
            or native_shred.get("status") != "not_reported"
            or registration.get("rotation_error", {}).get("status")
            != "not_applicable"
        ):
            raise FinalReportError(method + " direct metric/native-claim contract differs")
        normalized_methods[method] = {
            "status": row["status"],
            "frozen_validation_threshold": threshold,
            "translation_metrics": normalized,
            "assembly_edge_strict_547": {
                "predicted_edge_count": predicted_edges,
                "target_edge_count": target_edges,
                "by_tolerance": normalized_assembly,
            },
            "pairingnet_style_registration": dict(registration),
            "correspondence": dict(correspondence),
            "native_global_assembly_GA": dict(native_ga),
            "shreddingnet_native_CM_FM_SE": dict(native_shred),
        }
    direct = _mapping(
        value.get("direct_pairwise_metric_contract"),
        "real direct pairwise metric contract",
    )
    if (
        direct.get("translation_tolerances_px") != list(TOLERANCES)
        or direct.get("rotation_error") != "not_applicable_known_upright"
        or direct.get("same_positive_pairs_and_case_bootstrap_draws_across_methods")
        is not True
        or direct.get(
            "balanced_selected_list_diagnostics_claimed_as_native_shreddingnet_metrics"
        )
        is not False
    ):
        raise FinalReportError("real direct pairwise metric contract differs")
    positives = _list(value.get("positive_pairs"), "real positive translation rows")
    if len(positives) != 508:
        raise FinalReportError("real positive translation row count differs")
    qa = _mapping(value.get("seam_and_correspondence_qa"), "real translation QA")
    if (
        qa.get("status") != "passed_all_translation_transform_and_residual_checks"
        or qa.get("positive_pair_count") != 508
    ):
        raise FinalReportError("real translation QA differs")
    return {
        "population": "strict547_positive_subset",
        "eligible_positive_count": 508,
        "method_inventory": list(METHODS),
        "methods": normalized_methods,
        "frozen_validation_threshold": normalized_methods["full_n512"][
            "frozen_validation_threshold"
        ],
        "metrics": normalized_methods["full_n512"]["translation_metrics"],
        "coarse_and_matched_translation_comparison": "not_applicable",
        "same_data_benchmarks_are_adaptations_not_exact_reproductions": True,
        "native_cm_fm_se_or_ga_claimed": False,
        "qa": {
            "status": qa["status"],
            "maximum_pair_p95_model_frame_residual_px": _number(
                qa.get("maximum_pair_p95_model_frame_residual_px"),
                "real maximum QA residual",
                minimum=0.0,
            ),
        },
    }


def _e5_disclosure(value: object, description: str) -> Mapping[str, Any]:
    disclosure = _mapping(value, description)
    expected = {
        "prior_epoch5_synthetic_test_completed": True,
        "prior_epoch5_synthetic_test_human_visible_before_continuation": True,
        "prior_epoch5_synthetic_test_used_by_automated_checkpoint_selection": False,
        "prior_epoch5_synthetic_test_used_by_automated_threshold_fitting": False,
        "prior_epoch5_synthetic_test_used_by_automated_early_stopping": False,
        "prior_epoch5_synthetic_test_used_by_automated_scheduler": False,
        "claim_no_human_cognitive_influence": False,
        "prior_epoch5_real_evaluator_started_then_stopped": True,
        "prior_epoch5_real_result_formed_or_read": False,
        "prior_convergence_time_synthetic_test_mask_morphology_review": True,
        "prior_convergence_time_synthetic_test_mask_sample_count": 500,
        "prior_convergence_time_synthetic_test_labels_read": False,
        "prior_convergence_time_synthetic_test_model_scores_read": False,
        "prior_convergence_time_synthetic_test_activity_used_for_training_checkpoint_or_threshold_selection": False,
        "prior_convergence_time_synthetic_test_activity_influenced_fixed_corrosion_conditions": False,
        "prior_convergence_time_synthetic_test_activity_used_for_corrosion_runtime_and_representation_QA": True,
        "real_data_accessed_in_that_activity": False,
    }
    for field, expected_value in expected.items():
        observed = disclosure.get(field)
        if (
            (type(expected_value) is bool and observed is not expected_value)
            or (type(expected_value) is not bool and observed != expected_value)
        ):
            raise FinalReportError(description + "." + field + " differs")
    return expected


def _corrosion(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if (
        value.get("schema_version") != "rachel-n512-corrosion-robustness/2.0"
        or value.get("status")
        != "complete_formal_exact_six_frozen_synthetic_corrosion_only"
        or value.get("formal_evaluation") is not True
        or value.get("compatibility_mode") is not False
        or value.get("test_accessed") is not True
        or value.get("real_external_test_accessed") is not False
    ):
        raise FinalReportError("corrosion exact-six completion status differs")
    population = _mapping(value.get("test_population"), "corrosion population")
    if (
        population.get("pair_count") != 3000
        or population.get("positive_count") != 1500
        or population.get("negative_count") != 1500
        or population.get(
            "same_pair_ids_labels_clusters_source_units_and_order_at_every_condition"
        )
        is not True
    ):
        raise FinalReportError("corrosion population differs")
    protocol = _mapping(value.get("protocol"), "corrosion protocol")
    if (
        protocol.get("condition_order") != list(CONDITIONS)
        or protocol.get("mask_only") is not True
        or protocol.get("rgb_text_or_ocr_used") is not False
        or protocol.get("known_upright_orientation") is not True
        or protocol.get("rotation_search") is not False
        or protocol.get("formal_evaluation") is not True
        or protocol.get("compatibility_mode") is not False
        or protocol.get("formal_exact_six") is not True
        or protocol.get("formal_method_inventory") != list(METHODS)
        or protocol.get("method_order") != list(METHODS)
        or protocol.get("same_data_benchmark_methods_included") is not True
        or protocol.get("same_corrupted_masks_supplied_to_all_six_methods") is not True
        or protocol.get("each_method_forwarded_each_pair_exactly_once_per_condition")
        is not True
        or protocol.get(
            "all_winners_and_validation_thresholds_frozen_before_first_test_path_access"
        )
        is not True
        or protocol.get(
            "exact_train_validation_manifest_alignment_verified_before_first_test_path_access"
        )
        is not True
        or protocol.get("frozen_validation_thresholds_are_secondary_only") is not True
        or protocol.get("threshold_fit_performed") is not False
        or protocol.get("checkpoint_selection_performed") is not False
        or protocol.get("training_performed") is not False
        or protocol.get("real_external_test_accessed") is not False
    ):
        raise FinalReportError("corrosion protocol differs")
    disclosure = _e5_disclosure(
        protocol.get("evaluation_history_disclosure"), "corrosion disclosure"
    )
    summary = _mapping(value.get("summary"), "corrosion summary")
    if (
        summary.get("condition_order") != list(CONDITIONS)
        or summary.get("method_order") != list(METHODS)
        or summary.get("formal_exact_six") is not True
    ):
        raise FinalReportError("corrosion summary condition order differs")
    primary = _mapping(summary.get("primary_population"), "corrosion primary population")
    primary_count = _integer(primary.get("pair_count"), "corrosion common count", minimum=1)
    primary_positive = _integer(
        primary.get("positive_count"), "corrosion common positive", minimum=1
    )
    primary_negative = _integer(
        primary.get("negative_count"), "corrosion common negative", minimum=1
    )
    if (
        primary.get("definition") != "all_six_methods_valid_at_all_seven_conditions"
        or primary_positive + primary_negative != primary_count
        or primary_count > 3000
    ):
        raise FinalReportError("corrosion primary population differs")
    paired = _mapping(
        value.get("paired_endpoint_pigeonhole_bootstrap"), "corrosion bootstrap"
    )
    if summary.get("paired_endpoint_pigeonhole_bootstrap") != paired:
        raise FinalReportError("corrosion bootstrap copies differ")
    requested = _integer(
        paired.get("replicates_requested"), "corrosion replicates", minimum=1
    )
    valid = _integer(paired.get("valid_replicates"), "corrosion valid replicates", minimum=1)
    skipped = _integer(
        paired.get("skipped_single_class_replicates"), "corrosion skipped", minimum=0
    )
    if (
        paired.get("bootstrap") != "endpoint-unit_pigeonhole_product_multiplicity"
        or paired.get("shared_draws_across_all_methods_conditions_and_geometry_metrics")
        is not True
        or paired.get("population") != "fixed_all_method_all_condition_common_valid"
        or paired.get("method_order") != list(METHODS)
        or paired.get("formal_exact_six") is not True
        or paired.get("seed") != CORROSION_BOOTSTRAP_SEED
        or requested != BOOTSTRAP_REPLICATES
        or valid + skipped != requested
    ):
        raise FinalReportError("corrosion bootstrap receipt differs")
    ranking = _mapping(paired.get("ranking"), "corrosion ranking")
    by_condition = _mapping(
        ranking.get("by_condition_method"), "corrosion condition ranking"
    )
    clean_delta = _mapping(
        ranking.get("condition_minus_clean_same_method"),
        "corrosion clean deltas",
    )
    comparator_delta = _mapping(
        ranking.get("full_n512_minus_comparator_within_condition"),
        "corrosion comparator deltas",
    )
    if not (
        set(by_condition) == set(clean_delta) == set(comparator_delta) == set(CONDITIONS)
    ):
        raise FinalReportError("corrosion bootstrap condition set differs")
    translation = _mapping(
        paired.get("positive_only_translation_gt_and_joint"),
        "corrosion translation",
    )
    if (
        translation.get("negative_pairs_excluded") is not True
        or translation.get("threshold_source")
        != "frozen_validation_checkpoint_bound_artifact"
    ):
        raise FinalReportError("corrosion translation protocol differs")
    threshold = _number(
        translation.get("threshold"),
        "corrosion frozen threshold",
        minimum=0.0,
        maximum=1.0,
    )
    translation_by_condition = _mapping(
        translation.get("by_condition"), "corrosion translation conditions"
    )
    translation_clean_delta = _mapping(
        translation.get("condition_minus_clean"), "corrosion translation clean deltas"
    )
    if set(translation_by_condition) != set(CONDITIONS) or set(
        translation_clean_delta
    ) != set(CONDITIONS):
        raise FinalReportError("corrosion translation condition set differs")
    translation_metrics = {
        "median_l2_px",
        "p90_l2_px",
        "valid_pose_fraction",
        *("recall_at_{}px".format(item) for item in TOLERANCES),
        *(
            "joint_frozen_threshold_and_translation_recall_at_{}px".format(item)
            for item in TOLERANCES
        ),
        *("assembly_precision_at_{}px".format(item) for item in TOLERANCES),
        *("assembly_recall_at_{}px".format(item) for item in TOLERANCES),
        *("assembly_f1_at_{}px".format(item) for item in TOLERANCES),
        "pairing_rr_lt4",
        "pairing_mean_e_rmse",
        "pairing_mean_symmetric_hausdorff_px",
        "pairing_mean_normalized_translation_error",
    }
    direct_geometry = _mapping(
        paired.get("direct_geometry_by_method"), "corrosion direct geometry"
    )
    if tuple(direct_geometry) != TRANSLATION_METHODS:
        raise FinalReportError("corrosion direct-geometry method order differs")
    direct_by_method: dict[str, Any] = {}
    for method in TRANSLATION_METHODS:
        method_row = _mapping(
            direct_geometry[method], "corrosion direct geometry " + method
        )
        _number(
            method_row.get("threshold"),
            method + " corrosion threshold",
            minimum=0.0,
            maximum=1.0,
        )
        if (
            method_row.get("threshold_source")
            != "frozen_validation_checkpoint_bound_artifact"
        ):
            raise FinalReportError(method + " corrosion threshold source differs")
        method_by_condition = _mapping(
            method_row.get("by_condition"), method + " corrosion conditions"
        )
        method_minus_clean = _mapping(
            method_row.get("condition_minus_clean"),
            method + " corrosion clean deltas",
        )
        correspondence = _mapping(
            method_row.get("correspondence"), method + " corrosion correspondence"
        )
        if (
            set(method_by_condition) != set(CONDITIONS)
            or set(method_minus_clean) != set(CONDITIONS)
            or correspondence.get("status") != "not_applicable"
        ):
            raise FinalReportError(method + " corrosion direct protocol differs")
        direct_by_method[method] = {
            "threshold": method_row["threshold"],
            "by_condition": method_by_condition,
            "condition_minus_clean": method_minus_clean,
            "correspondence": correspondence,
        }
    full_direct = direct_by_method["full_n512"]
    if (
        threshold != full_direct["threshold"]
        or translation_by_condition != full_direct["by_condition"]
        or translation_clean_delta != full_direct["condition_minus_clean"]
    ):
        raise FinalReportError(
            "corrosion full-N512 compatibility projection differs from direct geometry"
        )
    normalized_conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        methods = _mapping(by_condition[condition], condition + " methods")
        method_clean = _mapping(clean_delta[condition], condition + " clean deltas")
        method_comparators = _mapping(
            comparator_delta[condition], condition + " comparator deltas"
        )
        if (
            set(methods) != set(CORROSION_METHODS)
            or set(method_clean) != set(CORROSION_METHODS)
            or set(method_comparators) != set(CORROSION_COMPARATORS)
        ):
            raise FinalReportError(condition + " corrosion method set differs")
        normalized_methods = {}
        normalized_clean = {}
        normalized_comparators = {}
        for method in CORROSION_METHODS:
            rows = _mapping(methods[method], condition + "." + method)
            deltas = _mapping(method_clean[method], condition + "." + method + ".clean")
            if set(rows) != {"auroc", "auprc"} or set(deltas) != {"auroc", "auprc"}:
                raise FinalReportError(condition + " ranking metrics differ")
            normalized_methods[method] = {
                metric: _interval(rows[metric], condition + "." + method + "." + metric)
                for metric in ("auroc", "auprc")
            }
            normalized_clean[method] = {
                metric: _interval(
                    deltas[metric],
                    condition + "." + method + ".minus_clean." + metric,
                    delta=True,
                )
                for metric in ("auroc", "auprc")
            }
        for comparator in CORROSION_COMPARATORS:
            rows = _mapping(
                method_comparators[comparator], condition + "." + comparator
            )
            if set(rows) != {"auroc", "auprc"}:
                raise FinalReportError(condition + " comparator metrics differ")
            normalized_comparators[comparator] = {
                metric: _interval(
                    rows[metric],
                    condition + ".full_minus_" + comparator + "." + metric,
                    delta=True,
                )
                for metric in ("auroc", "auprc")
            }
        translation_rows = _mapping(
            translation_by_condition[condition], condition + " translation"
        )
        translation_deltas = _mapping(
            translation_clean_delta[condition], condition + " translation clean delta"
        )
        if set(translation_rows) != translation_metrics or set(
            translation_deltas
        ) != translation_metrics:
            raise FinalReportError(condition + " translation metric set differs")
        normalized_direct = {}
        nonnegative_metrics = {
            "median_l2_px",
            "p90_l2_px",
            "pairing_mean_e_rmse",
            "pairing_mean_symmetric_hausdorff_px",
            "pairing_mean_normalized_translation_error",
        }
        for method in TRANSLATION_METHODS:
            method_rows = _mapping(
                direct_by_method[method]["by_condition"][condition],
                condition + "." + method + ".direct",
            )
            method_deltas = _mapping(
                direct_by_method[method]["condition_minus_clean"][condition],
                condition + "." + method + ".direct_minus_clean",
            )
            if set(method_rows) != translation_metrics or set(
                method_deltas
            ) != translation_metrics:
                raise FinalReportError(
                    condition + "." + method + " direct metric set differs"
                )
            normalized_direct[method] = {
                "threshold": direct_by_method[method]["threshold"],
                "metrics": {
                    name: _interval(
                        method_rows[name],
                        condition + "." + method + "." + name,
                        nonnegative=name in nonnegative_metrics,
                    )
                    for name in sorted(translation_metrics)
                },
                "minus_clean": {
                    name: _interval(
                        method_deltas[name],
                        condition + "." + method + ".minus_clean." + name,
                        delta=True,
                        unbounded=name in nonnegative_metrics,
                    )
                    for name in sorted(translation_metrics)
                },
                "correspondence": dict(direct_by_method[method]["correspondence"]),
            }
        normalized_conditions[condition] = {
            "ranking": normalized_methods,
            "same_method_minus_clean": normalized_clean,
            "full_n512_minus_comparator": normalized_comparators,
            "full_n512_translation_gt": {
                name: _interval(
                    translation_rows[name],
                    condition + ".translation." + name,
                    nonnegative=name in nonnegative_metrics,
                )
                for name in sorted(translation_metrics)
            },
            "full_n512_translation_minus_clean": {
                name: _interval(
                    translation_deltas[name],
                    condition + ".translation_minus_clean." + name,
                    delta=True,
                    unbounded=name in nonnegative_metrics,
                )
                for name in sorted(translation_metrics)
            },
            "direct_geometry_by_method": normalized_direct,
        }
    results = _list(value.get("condition_results"), "corrosion condition results")
    if [row.get("condition") for row in results if isinstance(row, Mapping)] != list(
        CONDITIONS
    ) or len(results) != len(CONDITIONS):
        raise FinalReportError("corrosion condition result order differs")
    return (
        {
            "formal_exact_six": True,
            "method_inventory": list(CORROSION_METHODS),
            "same_data_benchmark_methods_included": True,
            "scope_disclosure": (
                "formal exact-six corrosion comparison; PairingNet and ShreddingNet "
                "are same-data adaptations, not exact paper reproductions"
            ),
            "source_pair_count": 3000,
            "common_valid": {
                "row_count": primary_count,
                "positive_count": primary_positive,
                "negative_count": primary_negative,
            },
            "frozen_full_n512_threshold": threshold,
            "conditions": normalized_conditions,
            "translation_methods": list(TRANSLATION_METHODS),
            "benchmark_correspondence": "not_applicable_under_corrosion",
            "bootstrap": {
                "replicates_requested": requested,
                "valid_replicates": valid,
                "skipped_single_class_replicates": skipped,
                "shared_draws": True,
            },
        },
        disclosure,
    )


def _no_tuning_disclosure(
    document: Mapping[str, Any], description: str, *, real: bool
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = _mapping(document.get("protocol"), description + ".protocol")
    evidence = _mapping(
        protocol.get("automated_no_test_or_real_tuning"), description + ".no_tuning"
    )
    if (
        evidence.get("automated_nonuse_verified") is not True
        or evidence.get("claim_no_human_cognitive_influence") is not False
    ):
        raise FinalReportError(description + " no-tuning disclosure differs")
    if real:
        if (
            evidence.get("winner_and_validation_threshold_frozen_before_real_open")
            is not True
            or evidence.get("threshold_fit_performed_on_real") is not False
            or evidence.get("real_geometry_ground_truth_read") is not False
            or evidence.get("constructed_negatives_used_for_tuning") is not False
        ):
            raise FinalReportError(description + " real no-tuning evidence differs")
    else:
        if (
            evidence.get("winner_frozen_before_test_open") is not True
            or evidence.get("training_performed") is not False
            or evidence.get("checkpoint_selection_performed") is not False
            or evidence.get("threshold_fit_performed") is not False
        ):
            raise FinalReportError(description + " synthetic no-tuning evidence differs")
    disclosure = _e5_disclosure(
        evidence.get("evaluation_history_disclosure"), description + ".disclosure"
    )
    return evidence, disclosure


def _authority_snapshot_payload(inputs: AggregateInputs) -> Mapping[str, Any]:
    return {
        "terminal_file_sha256": inputs.terminal_file_sha256,
        "inventory_file_sha256": inputs.inventory_file_sha256,
        "inventory_entries": {
            logical: {
                "logical_path": entry.logical_path,
                "size": entry.size,
                "sha256": entry.sha256,
            }
            for logical, entry in sorted(inputs.inventory.items())
        },
        "member_lstat": {
            logical: dict(value)
            for logical, value in sorted(inputs.member_lstat_snapshot.items())
        },
    }


def _normalize_aggregates(inputs: AggregateInputs) -> Mapping[str, Any]:
    synthetic = inputs.synthetic_stats
    real = inputs.real_stats
    synthetic_population = _ranking_population(
        synthetic, total=3000, description="synthetic balanced"
    )
    synthetic_sha = synthetic_population["common_valid"].get("semantic_sha256")
    if not _is_sha256(synthetic_sha):
        raise FinalReportError("synthetic common population SHA-256 is invalid")
    synthetic_geometry = _synthetic_geometry(
        synthetic.get("synthetic_geometry")
    )
    synthetic_direct = _synthetic_direct_exact_six(
        inputs.sealed_receipt, synthetic_geometry
    )
    synthetic_no_tuning, synthetic_disclosure = _no_tuning_disclosure(
        synthetic, "synthetic", real=False
    )

    real_population = _ranking_population(
        real, total=1016, description="real balanced1016"
    )
    real_sha = real_population["common_valid"].get("semantic_sha256")
    if not _is_sha256(real_sha):
        raise FinalReportError("real common population SHA-256 is invalid")
    if real.get("balanced1016_inference_role") != (
        "primary external paired endpoint-case pigeonhole bootstrap; "
        "constructed negatives remain non-GT distractors"
    ):
        raise FinalReportError("real balanced1016 inferential role differs")
    geometry_marker = _mapping(real.get("geometry"), "real ranking geometry marker")
    if (
        geometry_marker.get("status")
        != "not_read_or_reported_by_this_target_blind_pair_ranking_artifact"
        or geometry_marker.get("ground_truth_availability_claim_made") is not False
    ):
        raise FinalReportError("real ranking geometry isolation differs")
    strict = _descriptive_population(real.get("strict547_descriptive"))
    real_no_tuning, real_disclosure = _no_tuning_disclosure(
        real, "real", real=True
    )

    corrosion, corrosion_disclosure = _corrosion(inputs.corrosion_receipt)
    translation = _real_translation(inputs.real_translation)
    translation_protocol = _mapping(
        inputs.real_translation.get("protocol"), "real translation protocol"
    )
    if (
        translation_protocol.get(
            "pair_only_prediction_file_frozen_before_any_current_gt_open"
        )
        is not True
        or translation_protocol.get("model_forward_or_gpu_work_performed_here")
        is not False
        or translation_protocol.get(
            "gt_used_for_training_checkpoint_threshold_or_model_selection"
        )
        is not False
        or translation_protocol.get("threshold_fit_performed_here") is not False
        or translation_protocol.get("coarse_and_matched_translation_are_not_applicable")
        is not True
        or translation_protocol.get("formal_evaluation") is not True
        or translation_protocol.get("compatibility_mode") is not False
        or translation_protocol.get("formal_exact_six_required") is not True
        or translation_protocol.get(
            "pairingnet_and_shreddingnet_adapted_translation_evaluated"
        )
        is not True
        or translation_protocol.get(
            "shreddingnet_selected_list_diagnostic_claimed_native_CM_FM_SE_GA"
        )
        is not False
    ):
        raise FinalReportError("real translation leakage/selection protocol differs")
    translation_disclosure = _e5_disclosure(
        translation_protocol.get("evaluation_history_disclosure"),
        "real translation disclosure",
    )
    if not (
        synthetic_disclosure
        == real_disclosure
        == corrosion_disclosure
        == translation_disclosure
    ):
        raise FinalReportError("E5 disclosure differs across aggregate artifacts")
    terminal = inputs.terminal
    if (
        terminal.get("training_performed") is not False
        or terminal.get("checkpoint_or_threshold_selection_performed") is not False
        or terminal.get(
            "metric_values_used_for_adaptive_model_checkpoint_threshold_or_condition_selection"
        )
        is not False
        or terminal.get("all_commands_completed_without_overwrite") is not True
        or terminal.get("automatic_performance_pass_fail_applied") is not False
    ):
        raise FinalReportError("terminal no-selection/no-overwrite contract differs")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_aggregate_summary",
        "methods": list(METHODS),
        "artifact_sha256": dict(inputs.artifact_sha256),
        "frozen_authority": dict(inputs.frozen_authority),
        "populations": {
            "synthetic_balanced": {
                **synthetic_population,
                "role": "inferential_frozen_synthetic_test",
                "class_balance": "1500_positive_1500_negative_before_common_valid",
            },
            "real_strict547_descriptive": strict,
            "real_balanced1016_inferential": {
                **real_population,
                "role": "inferential_with_constructed_non_GT_distractor_negatives",
                "strict_prefix_count": 547,
                "positive_count_before_common_valid": 508,
                "constructed_negative_count": 469,
                "constructed_negatives_used_for_tuning": False,
            },
        },
        "geometry_translation": {
            "synthetic_full_n512": synthetic_geometry,
            "synthetic_exact_six_direct_pairwise": synthetic_direct,
            "real_strict547_positive_full_n512": translation,
            "real_strict547_positive_exact_six_direct_pairwise": translation,
        },
        "corrosion_seven_condition": corrosion,
        "frozen_threshold_policy": {
            "ranking_metrics_are_threshold_free": True,
            "thresholds_fitted_on_validation_before_test_or_real_open": True,
            "thresholds_used_for_direct_assembly_edge_and_secondary_joint_metrics": True,
            "threshold_fit_or_selection_performed_by_renderer": False,
        },
        "metric_role": {
            "primary": (
                "assembly_edge_precision_recall_harmonic_f1; positive_translation_"
                "median_p90_recall; PairingNet_style_RR_HD_NTE; correspondence"
            ),
            "diagnostic": "pair_classification_AUROC_AUPRC",
        },
        "disclosures": {
            "e5_human_visible": synthetic_disclosure,
            "pre_freeze_500_mask_morphology_only_review": {
                key: synthetic_disclosure[key]
                for key in synthetic_disclosure
                if key.startswith("prior_convergence_time_")
                or key == "real_data_accessed_in_that_activity"
            },
            "synthetic_aggregate_no_tuning": dict(synthetic_no_tuning),
            "real_aggregate_no_tuning": dict(real_no_tuning),
            "corrosion_evaluation_history": dict(
                _mapping(
                    _mapping(inputs.corrosion_receipt.get("protocol"), "corrosion protocol").get(
                        "evaluation_history_disclosure"
                    ),
                    "corrosion disclosure",
                )
            ),
            "real_translation_evaluation_history": dict(
                _mapping(
                    translation_protocol.get("evaluation_history_disclosure"),
                    "translation disclosure",
                )
            ),
        },
        "integrity": {
            "terminal_completion_and_inventory_bound": True,
            "consumed_aggregate_hashes_recomputed": True,
            "all_inventoried_regular_file_sha256_recomputed": True,
            "raw_pair_score_jsonl_sha256_verified_without_parsing": True,
            "raw_pair_score_jsonl_parsed": False,
            "model_inputs_opened": False,
            "metric_values_used_for_tuning_or_selection": False,
            "performance_pass_fail_threshold_applied": False,
            "complete_authority_snapshot_sha256": _sha256_bytes(
                _canonical_bytes(_authority_snapshot_payload(inputs))
            ),
        },
        "claim_limits": {
            "scope": "pairwise_match_probability_and_relative_2d_translation",
            "global_multi_fragment_assembly_demonstrated": False,
            "pairingnet_reproduction_claimed": False,
            "shreddingnet_reproduction_claimed": False,
            "same_data_benchmarks_are_adaptations_not_exact_reproductions": True,
            "pairingnet_native_global_assembly_GA_claimed": False,
            "shreddingnet_native_CM_FM_SE_or_GA_claimed": False,
            "rotation_estimated_or_supervised": False,
            "synthetic_correspondence_metric_aggregated_here": True,
        },
    }


def _format_number(value: float) -> str:
    return "{:.4f}".format(float(value))


def _format_interval(value: Mapping[str, Any]) -> str:
    point = _number(value.get("point_estimate"), "rendered point")
    interval = _list(value.get("percentile_95_ci"), "rendered CI")
    return "{} [{}, {}]".format(
        _format_number(point),
        _format_number(_number(interval[0], "rendered CI low")),
        _format_number(_number(interval[1], "rendered CI high")),
    )


def _format_direct_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, Mapping):
        if "point_estimate" in value and "percentile_95_ci" in value:
            return _format_interval(value)
        if "estimate" in value and "ci95_low" in value and "ci95_high" in value:
            return "{} [{}, {}]".format(
                _format_number(_number(value["estimate"], "direct estimate")),
                _format_number(_number(value["ci95_low"], "direct CI low")),
                _format_number(_number(value["ci95_high"], "direct CI high")),
            )
        raise FinalReportError("direct metric object has no supported display contract")
    return _format_number(_number(value, "direct point estimate"))


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def _ranking_tables(
    population: Mapping[str, Any], heading: str
) -> list[str]:
    common = _mapping(population.get("common_valid"), heading + " common")
    ranking = _mapping(population.get("ranking"), heading + " ranking")
    deltas = _mapping(
        population.get("full_n512_minus_comparator"), heading + " deltas"
    )
    lines = [
        "### " + heading,
        "",
        "Common-valid: {}/{} pairs ({} positive, {} negative).".format(
            common["row_count"],
            population["source_pair_count"],
            common["positive_count"],
            common["negative_count"],
        ),
        "",
    ]
    rows = []
    for method in METHODS:
        rows.append(
            [
                method,
                _format_interval(ranking[method]["auroc"]),
                _format_interval(ranking[method]["auprc"]),
            ]
        )
    lines.extend(_table(("Method", "AUROC [95% CI]", "AUPRC [95% CI]"), rows))
    lines.extend(("", "Full N=512 minus comparator (paired effect size):", ""))
    lines.extend(
        _table(
            ("Comparator", "Δ AUROC [95% CI]", "Δ AUPRC [95% CI]"),
            [
                [
                    comparator,
                    _format_interval(deltas[comparator]["auroc"]),
                    _format_interval(deltas[comparator]["auprc"]),
                ]
                for comparator in COMPARATORS
            ],
        )
    )
    return lines


def _render_markdown(summary: Mapping[str, Any]) -> str:
    populations = _mapping(summary.get("populations"), "summary populations")
    authority = _mapping(summary.get("frozen_authority"), "summary authority")
    geometry = _mapping(
        summary.get("geometry_translation"), "summary geometry"
    )
    corrosion = _mapping(
        summary.get("corrosion_seven_condition"), "summary corrosion"
    )
    synthetic_geometry = _mapping(
        geometry["synthetic_full_n512"], "synthetic geometry"
    )
    translation = _mapping(
        synthetic_geometry["translation_l2_px"], "synthetic translation"
    )
    assembly = _mapping(
        synthetic_geometry["assembly_edge_at_frozen_validation_threshold"],
        "synthetic assembly edge",
    )
    registration = _mapping(
        synthetic_geometry["pairingnet_style_registration"],
        "synthetic PairingNet registration",
    )
    correspondence = _mapping(
        synthetic_geometry["correspondence"], "synthetic correspondence"
    )
    synthetic_direct = _mapping(
        geometry["synthetic_exact_six_direct_pairwise"],
        "synthetic exact-six direct metrics",
    )
    synthetic_direct_methods = _mapping(
        synthetic_direct["methods"], "synthetic exact-six direct methods"
    )
    benchmark_adaptations = _mapping(
        authority.get("same_data_benchmark_adaptation_disclosures"),
        "same-data benchmark adaptation disclosures",
    )
    pairingnet_adaptation = _mapping(
        benchmark_adaptations.get("pairingnet_adapted"),
        "PairingNet adaptation disclosure",
    )
    shreddingnet_adaptation = _mapping(
        benchmark_adaptations.get("shreddingnet_adapted"),
        "ShreddingNet adaptation disclosure",
    )
    lines = [
        "# Rachel N=512 frozen pairwise final report",
        "",
        "This report renders only hash-verified aggregate artifacts. It did not open raw pair-score JSONL files or model inputs, did not tune or select a checkpoint/threshold, and applied no performance pass/fail threshold.",
        "",
        "Values shown as `point [low, high]` use the recorded 95% bootstrap interval. A displayed interval is not automatically interpreted as ‘better’, ‘worse’, or ‘passed’.",
        "",
        "## Methods and populations",
        "",
    ]
    lines.extend(
        _table(
            ("Method", "Role"),
            (
                ("coarse_only", "coarse Siamese pair classifier"),
                ("full_n512", "N=512 pair classifier + relative 2D translation"),
                ("matched_mm_converged", "matched-MM converged checkpoint"),
                (
                    "matched_mm_same_exposure_epoch5",
                    "matched-MM same-exposure epoch-5 checkpoint",
                ),
                (
                    "pairingnet_adapted",
                    "same-data PairingNet mask-only upright-translation adaptation",
                ),
                (
                    "shreddingnet_adapted",
                    "same-data ShreddingNet mask-only upright-translation adaptation",
                ),
            ),
        )
    )
    lines.extend(
        (
            "",
            "Synthetic: 3,000 balanced pairs (1,500/1,500). Real strict547: 508 positives and 39 negatives, descriptive only. Real balanced1016: the exact strict547 prefix plus 469 constructed non-GT distractor negatives; those constructed negatives were not used for tuning.",
            "",
            "PairingNet and ShreddingNet entries are same-data adaptations, not exact paper reproductions. Native PairingNet global assembly (GA) and native ShreddingNet CM/FM/SE/GA are not claimed.",
            "",
            "PairingNet adaptation disclosure: mask-only input; primary pose `{}`; the added pair head is not an official component; dual-softmax is not described as Sinkhorn.".format(
                pairingnet_adaptation["primary_pose"]
            ),
            "",
            "ShreddingNet adaptation disclosure: Rachel N=512 mask-only, known-upright translation-only primary pose; no global assembly is performed, and balanced selected-list diagnostics are not native CM/FM/SE." if shreddingnet_adaptation.get("balanced_pair_list_diagnostics_are_native_cm_fm_se") is False else "",
            "",
            "### Exact-six synthetic direct placement readout",
            "",
        )
    )
    synthetic_direct_rows = []
    for method in METHODS:
        row = _mapping(synthetic_direct_methods[method], method + " synthetic direct")
        if row.get("status") == "not_applicable":
            synthetic_direct_rows.append(
                [method, "N/A", "N/A", "N/A", "N/A", "N/A"]
            )
            continue
        direct_translation = _mapping(
            row.get("translation"), method + " synthetic direct translation"
        )
        direct_assembly = _mapping(
            row.get("assembly_edge"), method + " synthetic direct assembly"
        )
        direct_assembly_rows = _mapping(
            direct_assembly.get("by_tolerance"),
            method + " synthetic direct assembly tolerances",
        )
        direct_registration = _mapping(
            row.get("pairingnet_style_registration"),
            method + " synthetic direct registration",
        )
        direct_correspondence = _mapping(
            row.get("correspondence"), method + " synthetic direct correspondence"
        )
        if method == "full_n512":
            median_value = direct_translation.get("median")
            p90_value = direct_translation.get("p90")
            rr_value = direct_registration.get("registration_recall_e_rmse_lt4")
        else:
            median_value = direct_translation.get(
                "median_l2_px_conditioned_on_valid"
            )
            p90_value = direct_translation.get("p90_l2_px_conditioned_on_valid")
            rr_value = direct_registration.get("rr_lt4")
        synthetic_direct_rows.append(
            [
                method,
                _format_direct_value(median_value),
                _format_direct_value(p90_value),
                _format_direct_value(
                    direct_assembly_rows["at_8px"]["harmonic_f1"]
                ),
                _format_direct_value(rr_value),
                str(direct_correspondence.get("status", "reported")),
            ]
        )
    lines.extend(
        _table(
            (
                "Method",
                "Median TE px",
                "P90 TE px",
                "Assembly-edge F1@8px",
                "PairingNet-style RR",
                "Correspondence",
            ),
            synthetic_direct_rows,
        )
    )
    lines.extend(
        (
            "",
            "N/A means the frozen method does not emit a supervised 2D translation; it is not scored as zero. Benchmark direct synthetic rows are point estimates because their sealed receipt does not claim a geometry-bootstrap CI; Full N=512 retains its recorded endpoint-bootstrap interval.",
            "",
            "## Full N=512 detailed direct assembly metrics",
            "",
        )
    )
    assembly_rows = _mapping(assembly["by_tolerance"], "assembly rows")
    lines.extend(
        _table(
            (
                "Translation L2 tolerance (px)",
                "Assembly-edge precision [95% CI]",
                "Assembly-edge recall [95% CI]",
                "Assembly-edge harmonic-F1 [95% CI]",
            ),
            [
                [
                    "<= {}".format(tolerance),
                    _format_interval(assembly_rows["at_{}".format(tolerance)]["precision"]),
                    _format_interval(assembly_rows["at_{}".format(tolerance)]["recall"]),
                    _format_interval(
                        assembly_rows["at_{}".format(tolerance)]["harmonic_f1"]
                    ),
                ]
                for tolerance in ASSEMBLY_EDGE_TOLERANCES
            ],
        )
    )
    lines.extend(
        (
            "",
            "An assembly-edge prediction is correct only when the frozen validation threshold accepts the adjacent pair and its translation L2 error is at or below the stated pixel tolerance. Thus a high pair-classification score with a wrong placement is not counted as a correct join. PairingNet-compatible eRMSE is reported separately as RR and is not used for this assembly-edge criterion.",
            "",
            "Positive-pair translation L2: median {}; P90 {} ({} valid poses of {} positives). Translation recall uses every positive pair as denominator, so invalid poses are failures.".format(
                _format_interval(translation["median"]),
                _format_interval(translation["p90"]),
                translation["valid_translation_count"],
                translation["eligible_positive_count"],
            ),
            "",
        )
    )
    lines.extend(
        _table(
            ("Translation tolerance", "Positive-pair translation recall [95% CI]"),
            [
                [
                    "≤ {} px".format(tolerance),
                    _format_interval(
                        translation["recall_by_tolerance"][
                            "recall_at_{}px".format(tolerance)
                        ]
                    ),
                ]
                for tolerance in TOLERANCES
            ],
        )
    )
    lines.extend(
        (
            "",
            "PairingNet-style RR/HD/NTE use the released compatibility definitions over every positive pair; a decision-invalid pose uses identity translation, matching the released fallback, while direct translation and assembly metrics still count that pose as a failure. Contour areas are quantized to int32 before the NTE area calculation. Rotation error is N/A because orientation is fixed upright and no rotation is estimated or supervised.",
            "",
        )
    )
    lines.extend(
        _table(
            ("Registration metric", "Point [95% CI]"),
            (
                (
                    "RR (eRMSE < 4)",
                    _format_interval(registration["registration_recall_e_rmse_lt4"]),
                ),
                ("Mean eRMSE", _format_interval(registration["mean_e_rmse"])),
                (
                    "Mean symmetric HD (px)",
                    _format_interval(registration["mean_symmetric_hausdorff_px"]),
                ),
                (
                    "Mean NTE",
                    _format_interval(registration["mean_normalized_translation_error"]),
                ),
                ("Rotation error", "N/A (upright-conditioned)"),
            ),
        )
    )
    strict_correspondence = correspondence["strict_dustbin_aware"]
    mutual_correspondence = correspondence["mutual_top1"]
    lines.extend(("", "Correspondence and dustbin quality:", ""))
    lines.extend(
        _table(
            ("Readout", "Precision [95% CI]", "Recall [95% CI]", "Harmonic-F1 [95% CI]"),
            (
                (
                    "Strict dustbin-aware",
                    _format_interval(strict_correspondence["precision"]),
                    _format_interval(strict_correspondence["recall"]),
                    _format_interval(strict_correspondence["harmonic_f1"]),
                ),
                (
                    "Mutual top-1",
                    _format_interval(mutual_correspondence["precision"]),
                    _format_interval(mutual_correspondence["recall"]),
                    _format_interval(mutual_correspondence["harmonic_f1"]),
                ),
            ),
        )
    )
    lines.extend(
        (
            "",
            "Dustbin token accuracy: {}.".format(
                _format_interval(correspondence["dustbin_accuracy"]["accuracy"])
            ),
            "",
            "## Pair-classification diagnostics",
            "",
            "AUROC/AUPRC diagnose pair-score ranking only; they are not treated as direct evidence that fragments were placed correctly.",
            "",
        )
    )
    lines.extend(
        _ranking_tables(populations["synthetic_balanced"], "Synthetic balanced")
    )
    lines.extend(("",))
    lines.extend(
        _ranking_tables(
            populations["real_balanced1016_inferential"],
            "Real balanced1016 inferential",
        )
    )
    strict = _mapping(
        populations["real_strict547_descriptive"], "strict population"
    )
    strict_common = _mapping(strict.get("common_valid"), "strict common")
    strict_ranking = _mapping(
        strict.get("ranking_point_estimates"), "strict ranking"
    )
    lines.extend(
        (
            "",
            "### Real strict547 descriptive",
            "",
            "Common-valid: {}/547 pairs ({} positive, {} negative). No inferential CI is reported by design because the strict population has only 39 negatives before common-valid filtering.".format(
                strict_common["row_count"],
                strict_common["positive_count"],
                strict_common["negative_count"],
            ),
            "",
        )
    )
    lines.extend(
        _table(
            ("Method", "AUROC point", "AUPRC point"),
            [
                [
                    method,
                    _format_number(strict_ranking[method]["auroc"]),
                    _format_number(strict_ranking[method]["auprc"]),
                ]
                for method in METHODS
            ],
        )
    )

    joint = _mapping(
        synthetic_geometry["joint_frozen_pair_and_translation_success"],
        "synthetic joint",
    )
    lines.extend(
        (
            "",
            "## Supplementary conditional translation diagnostics",
            "",
            "These rows retain the earlier conditional-on-valid translation success and joint pair-plus-translation readout. The primary assembly-edge, unconditional translation-recall, registration, and correspondence metrics are reported above.",
            "",
            "Median translation L2: {} ({} valid of {} positive pairs).".format(
                _format_interval(translation["median"]),
                translation["valid_translation_count"],
                translation["eligible_positive_count"],
            ),
            "",
        )
    )
    lines.extend(
        _table(
            ("Tolerance", "Translation success [95% CI]", "Joint frozen-threshold success [95% CI]"),
            [
                [
                    "{} px".format(tolerance),
                    _format_interval(
                        translation["success_by_tolerance"][
                            "success_at_{}px".format(tolerance)
                        ]
                    ),
                    _format_interval(
                        joint["success_by_tolerance"][
                            "success_at_{}px".format(tolerance)
                        ]
                    ),
                ]
                for tolerance in TOLERANCES
            ],
        )
    )

    real_translation = _mapping(
        geometry["real_strict547_positive_full_n512"], "real translation"
    )
    real_translation_metrics = _mapping(
        real_translation["metrics"], "real translation metrics"
    )
    real_direct_methods = _mapping(
        real_translation["methods"], "real exact-six direct methods"
    )
    lines.extend(
        (
            "",
            "## Real strict547 exact-six direct placement/translation GT",
            "",
            "Direct translation is evaluated for Full N=512 plus the PairingNet and ShreddingNet adaptations on the same 508 authoritative positive pairs. Invalid translations count as recall failures. Coarse and both matched-MM checkpoints are explicitly N/A.",
            "",
        )
    )
    real_direct_rows = []
    for method in METHODS:
        row = _mapping(real_direct_methods[method], method + " real direct")
        if row.get("status") == "not_applicable":
            real_direct_rows.append(
                [method, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
            )
            continue
        metrics = _mapping(
            row.get("translation_metrics"), method + " real direct translation"
        )
        assembly_direct = _mapping(
            row.get("assembly_edge_strict_547"), method + " real direct assembly"
        )
        assembly_tolerances = _mapping(
            assembly_direct.get("by_tolerance"),
            method + " real direct assembly tolerances",
        )
        registration_direct = _mapping(
            row.get("pairingnet_style_registration"),
            method + " real direct registration",
        )
        correspondence_direct = _mapping(
            row.get("correspondence"), method + " real direct correspondence"
        )
        real_direct_rows.append(
            [
                method,
                _format_direct_value(metrics["median_l2_px"]),
                _format_direct_value(metrics["p90_l2_px"]),
                _format_direct_value(metrics["recall_at_8px"]),
                _format_direct_value(
                    assembly_tolerances["at_8px"]["harmonic_f1"]
                ),
                _format_direct_value(registration_direct["rr_lt4"]),
                str(correspondence_direct.get("status")),
            ]
        )
    lines.extend(
        _table(
            (
                "Method",
                "Median TE px [95% CI]",
                "P90 TE px [95% CI]",
                "Translation recall@8 [95% CI]",
                "Assembly-edge F1@8",
                "PairingNet-style RR",
                "Correspondence",
            ),
            real_direct_rows,
        )
    )
    lines.extend(
        (
            "",
            "Assembly-edge and registration cells are direct point estimates; the translation cells use the shared case-cluster bootstrap. No native CM/FM/SE/GA metric is inferred from the balanced selected-list diagnostics.",
            "",
            "Full N=512 real translation bootstrap detail:",
            "",
        )
    )
    lines.extend(
        _table(
            ("Metric", "Point [95% CI]"),
            [
                [name, _format_interval(real_translation_metrics[name])]
                for name in sorted(real_translation_metrics)
            ],
        )
    )

    condition_rows = []
    condition_values = _mapping(corrosion.get("conditions"), "corrosion conditions")
    for condition in CONDITIONS:
        row = _mapping(condition_values[condition], condition)
        ranking = _mapping(row["ranking"], condition + " ranking")
        effects = _mapping(
            row["full_n512_minus_comparator"], condition + " effects"
        )
        for metric in ("auroc", "auprc"):
            condition_rows.append(
                [
                    condition,
                    metric.upper(),
                    *[
                        _format_interval(ranking[method][metric])
                        for method in CORROSION_METHODS
                    ],
                    *[
                        _format_interval(effects[method][metric])
                        for method in CORROSION_COMPARATORS
                    ],
                ]
            )
    lines.extend(
        (
            "",
            "## Seven-condition mask-corrosion robustness",
            "",
            "This is the formal exact-six corruption comparison. Every method receives the same corrupted mask pair once per condition, and ranking uses one six-method/all-condition common-valid population. PairingNet and ShreddingNet remain same-data adaptations rather than exact paper reproductions; correspondence is N/A because corrupted contour-token indices have no current-condition GT.",
            "",
        )
    )
    lines.extend(
        _table(
            ("Condition", "Metric")
            + tuple(CORROSION_METHODS)
            + tuple("Full N512 Δ vs " + method for method in CORROSION_COMPARATORS),
            condition_rows,
        )
    )
    corrosion_translation_rows = []
    corrosion_assembly_rows = []
    corrosion_pairing_rows = []
    for condition in CONDITIONS:
        row = condition_values[condition]
        direct = _mapping(
            row["direct_geometry_by_method"], condition + " direct corrosion"
        )
        for method in TRANSLATION_METHODS:
            metrics = direct[method]["metrics"]
            corrosion_translation_rows.append(
                [
                    condition,
                    method,
                    _format_interval(metrics["median_l2_px"]),
                    _format_interval(metrics["p90_l2_px"]),
                    *[
                        _format_interval(
                            metrics["recall_at_{}px".format(tolerance)]
                        )
                        for tolerance in TOLERANCES
                    ],
                ]
            )
            corrosion_assembly_rows.append(
                [
                    condition,
                    method,
                    *[
                        _format_interval(
                            metrics["assembly_{}_at_{}px".format(name, tolerance)]
                        )
                        for tolerance in TOLERANCES
                        for name in ("precision", "recall", "f1")
                    ],
                ]
            )
            corrosion_pairing_rows.append(
                [
                    condition,
                    method,
                    _format_interval(metrics["pairing_rr_lt4"]),
                    _format_interval(metrics["pairing_mean_e_rmse"]),
                    _format_interval(
                        metrics["pairing_mean_symmetric_hausdorff_px"]
                    ),
                    _format_interval(
                        metrics["pairing_mean_normalized_translation_error"]
                    ),
                    str(direct[method]["correspondence"]["status"]),
                ]
            )
    lines.extend(("", "Exact-six direct geometry under corruption:", ""))
    lines.extend(
        _table(
            (
                "Condition",
                "Method",
                "Median TE",
                "P90 TE",
                "Recall@2",
                "Recall@5",
                "Recall@8",
                "Recall@10",
            ),
            corrosion_translation_rows,
        )
    )
    lines.extend(("", "Frozen-threshold assembly-edge precision/recall/F1:", ""))
    lines.extend(
        _table(
            ("Condition", "Method")
            + tuple(
                "{}@{}".format(name, tolerance)
                for tolerance in TOLERANCES
                for name in ("P", "R", "F1")
            ),
            corrosion_assembly_rows,
        )
    )
    lines.extend(("", "Pairing-compatible registration under corruption:", ""))
    lines.extend(
        _table(
            (
                "Condition",
                "Method",
                "Pairing RR",
                "Mean eRMSE",
                "Mean HD",
                "Mean NTE",
                "Correspondence",
            ),
            corrosion_pairing_rows,
        )
    )
    lines.extend(
        (
            "",
            "The machine-readable summary contains TE median/P90, 2/5/8/10 px recall, assembly-edge precision/recall/F1, Pairing-compatible RR/HD/NTE, and every same-method-vs-clean paired interval for Full N=512, PairingNet-adapted, and ShreddingNet-adapted.",
            "",
            "## Frozen architecture and Rachel data provenance",
            "",
            "The complete canonical model/loss dataclass mappings were checked field-for-field for both N=512 winners. Model config SHA-256: `{}`; loss config SHA-256: `{}`.".format(
                authority["canonical_model_config_sha256"],
                authority["canonical_loss_config_sha256"],
            ),
            "",
            "Rachel provenance sidecar SHA-256: `{}`. Preprocess receipt: `{}`; preprocess summary: `{}`; train manifest: `{}`; validation manifest: `{}`.".format(
                authority["rachel_data_provenance_sidecar_sha256"],
                authority["rachel_preprocess_receipt_sha256"],
                authority["rachel_preprocess_summary_sha256"],
                authority["train_manifest_content_sha256"],
                authority["validation_manifest_content_sha256"],
            ),
            "",
            "The frozen source authority is Rachel RGB JPEG plus colocated label.csv only, with no Shredding data read. The preprocess receipt is complete and embeds the separately hashed preprocess summary exactly. The train and validation manifest hashes are bound to both training fingerprints; the formal test manifest content was not read or hashed by the pre-test provenance receipt.",
            "",
            "## Frozen-threshold and disclosure notes",
            "",
            "AUROC/AUPRC are threshold-free diagnostics. The direct assembly-edge metric uses the validation-frozen pair threshold plus seam registration tolerance; no test/real threshold fitting or adaptive checkpoint/condition selection occurred in this render.",
            "",
            "The prior epoch-5 synthetic test was completed and human-visible before continuation. It was not used by automated checkpoint selection, threshold fitting, early stopping, or scheduling. Therefore this report does not claim absence of possible human cognitive influence.",
            "",
            "Before the final freeze, a morphology-only review inspected 500 synthetic-test masks. No labels or model scores were read; it was not used for training, checkpoint/threshold selection, or choosing the fixed corrosion conditions. It was used only for corrosion runtime/representation QA, and no real data was accessed in that activity.",
            "",
            "## Claim limits",
            "",
            "This is pairwise relative pose only: pair compatibility and upright, translation-only 2D relative placement for Full N=512 and the two same-data benchmark adaptations. It is not global multi-fragment assembly; PairingNet/ShreddingNet are not exact reproductions, and no native PairingNet GA or ShreddingNet CM/FM/SE/GA claim is made. Rotation is neither supervised nor estimated. No automatic performance pass/fail decision is applied.",
            "",
        )
    )
    return "\n".join(lines)


def _stage_file(directory: Path, name: str, payload: bytes) -> None:
    path = directory / name
    descriptor = os.open(
        str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _output_path(final_root: Path, value: Path) -> Path:
    raw = Path(value).expanduser()
    if raw.exists() or raw.is_symlink():
        raise FinalReportError("output directory must be fresh and non-symlink")
    parent = _resolve_directory(raw.parent, "output parent")
    output = parent / raw.name
    if not raw.name or raw.name in {".", ".."}:
        raise FinalReportError("output directory name is invalid")
    try:
        output.relative_to(final_root)
    except ValueError:
        pass
    else:
        raise FinalReportError("output directory may not be nested in final root")
    try:
        final_root.relative_to(output)
    except ValueError:
        pass
    else:
        raise FinalReportError("output directory may not contain final root")
    return output


def _publish_report_directory(
    output: Path, markdown_payload: bytes, summary_payload: bytes, receipt_payload: bytes
) -> None:
    parent = output.parent
    partial = Path(
        tempfile.mkdtemp(prefix="." + output.name + ".partial-", dir=str(parent))
    )
    published = False
    try:
        _stage_file(partial, "final_report.md", markdown_payload)
        _stage_file(partial, "final_report_summary.json", summary_payload)
        _stage_file(partial, "completion_receipt.json", receipt_payload)
        _fsync_directory(partial)
        try:
            os.mkdir(output, 0o700)
        except FileExistsError as error:
            raise FinalReportError("refusing to overwrite report output directory") from error
        published = True
        for name in ("final_report.md", "final_report_summary.json"):
            os.link(partial / name, output / name)
        _fsync_directory(output)
        os.link(partial / "completion_receipt.json", output / "completion_receipt.json")
        _fsync_directory(output)
        _fsync_directory(parent)
    except BaseException:
        if published:
            for name in (
                "completion_receipt.json",
                "final_report_summary.json",
                "final_report.md",
            ):
                try:
                    (output / name).unlink()
                except FileNotFoundError:
                    pass
            try:
                output.rmdir()
            except FileNotFoundError:
                pass
        raise
    finally:
        shutil.rmtree(partial, ignore_errors=True)


def _render_and_publish(inputs: AggregateInputs, output_value: Path) -> Path:
    output = _output_path(inputs.root, output_value)
    normalized = _normalize_aggregates(inputs)
    reloaded = _load_aggregate_inputs(inputs.root)
    if (
        reloaded.terminal_file_sha256 != inputs.terminal_file_sha256
        or reloaded.inventory_file_sha256 != inputs.inventory_file_sha256
        or reloaded.inventory != inputs.inventory
        or reloaded.member_lstat_snapshot != inputs.member_lstat_snapshot
    ):
        raise FinalReportError(
            "complete authority snapshot changed before report publication"
        )
    reloaded_normalized = _normalize_aggregates(reloaded)
    if _canonical_bytes(reloaded_normalized) != _canonical_bytes(normalized):
        raise FinalReportError(
            "final root changed between initial load and pre-publication revalidation"
        )
    markdown = _render_markdown(normalized).encode("utf-8")
    summary_body = {
        **dict(normalized),
        "rendered_markdown": {
            "path": "final_report.md",
            "sha256": _sha256_bytes(markdown),
        },
    }
    summary = {**summary_body, "content_sha256": _sha256_bytes(_canonical_bytes(summary_body))}
    summary_payload = _canonical_bytes(summary) + b"\n"
    receipt_body = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_verified_final_report_render",
        "source_final_root": str(inputs.root),
        "output_directory": str(output),
        "completion_receipt_published_last": True,
        "files": [
            {
                "path": "final_report.md",
                "size": len(markdown),
                "sha256": _sha256_bytes(markdown),
            },
            {
                "path": "final_report_summary.json",
                "size": len(summary_payload),
                "sha256": _sha256_bytes(summary_payload),
            },
        ],
        "raw_pair_score_jsonl_sha256_verified_without_parsing": True,
        "raw_pair_score_jsonl_parsed": False,
        "model_inputs_opened": False,
        "tuning_or_selection_performed": False,
        "performance_pass_fail_threshold_applied": False,
    }
    receipt = {
        **receipt_body,
        "content_sha256": _sha256_bytes(_canonical_bytes(receipt_body)),
    }
    receipt_payload = _canonical_bytes(receipt) + b"\n"
    _publish_report_directory(output, markdown, summary_payload, receipt_payload)
    return output


def _inventory_index(
    root: Path, terminal: Mapping[str, Any], inventory: Mapping[str, Any]
) -> Mapping[str, InventoryEntry]:
    rows = inventory.get("files")
    count = inventory.get("file_count")
    if inventory.get("excludes") != list(INVENTORY_EXCLUDES):
        raise FinalReportError("content inventory exclusion set differs")
    if not isinstance(rows, list) or type(count) is not int or count != len(rows):
        raise FinalReportError("content inventory file rows/count differ")
    binding = terminal.get("content_inventory")
    if not isinstance(binding, Mapping) or binding.get("inventory_file_count") != count:
        raise FinalReportError("terminal inventory file count differs")
    index: dict[str, InventoryEntry] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FinalReportError("inventory row {} is not an object".format(position))
        logical = row.get("path")
        size = row.get("size")
        digest = row.get("sha256")
        if not isinstance(logical, str):
            raise FinalReportError("inventory row path is invalid")
        if logical in index:
            raise FinalReportError("content inventory has a duplicate path: " + logical)
        if type(size) is not int or size < 0 or not _is_sha256(digest):
            raise FinalReportError("inventory row size/SHA-256 is invalid: " + logical)
        path = _safe_member(root, logical, "inventory member")
        if path.stat().st_size != size:
            raise FinalReportError("inventory member size differs: " + logical)
        if _sha256_file(path) != digest:
            raise FinalReportError("inventory member SHA-256 differs: " + logical)
        index[logical] = InventoryEntry(logical, path, size, str(digest))
    actual: set[str] = set()

    def visit(directory: Path) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise FinalReportError("final root tree cannot be enumerated") from error
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise FinalReportError("final root member cannot be stated") from error
            logical = path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode):
                raise FinalReportError("symlinks are forbidden in final root: " + logical)
            if stat.S_ISDIR(info.st_mode):
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                actual.add(logical)
            else:
                raise FinalReportError("special files are forbidden in final root: " + logical)

    visit(root)
    expected = set(index) | set(INVENTORY_EXCLUDES)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise FinalReportError(
            "final root file set differs from inventory; missing={} unexpected={}".format(
                missing, unexpected
            )
        )
    return index


def _member_lstat_snapshot(
    root: Path, index: Mapping[str, InventoryEntry]
) -> Mapping[str, Mapping[str, int]]:
    snapshot: dict[str, Mapping[str, int]] = {}

    def visit(directory: Path) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise FinalReportError("final root authority tree cannot be enumerated") from error
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise FinalReportError("final root authority member cannot be stated") from error
            logical = path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode):
                raise FinalReportError("symlinks are forbidden in final root: " + logical)
            if stat.S_ISDIR(info.st_mode):
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                snapshot[logical] = {
                    "device": int(info.st_dev),
                    "inode": int(info.st_ino),
                    "mode": int(info.st_mode),
                    "link_count": int(info.st_nlink),
                    "size": int(info.st_size),
                    "mtime_ns": int(info.st_mtime_ns),
                    "ctime_ns": int(info.st_ctime_ns),
                }
            else:
                raise FinalReportError("special files are forbidden in final root: " + logical)

    visit(root)
    expected = set(index) | set(INVENTORY_EXCLUDES)
    if set(snapshot) != expected:
        raise FinalReportError("final root authority snapshot file set differs")
    return {logical: snapshot[logical] for logical in sorted(snapshot)}


def _read_inventoried_json(
    root: Path,
    index: Mapping[str, InventoryEntry],
    logical: str,
    description: str,
) -> tuple[Mapping[str, Any], str]:
    entry = index.get(logical)
    if entry is None:
        raise FinalReportError(description + " is absent from the content inventory")
    document, payload = _read_json_member(root, logical, description)
    digest = _sha256_bytes(payload)
    if len(payload) != entry.size or digest != entry.sha256:
        raise FinalReportError(description + " differs from the content inventory")
    return document, digest


def _terminal_gate_rows(terminal: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    rows = terminal.get("gate_receipts")
    if not isinstance(rows, list):
        raise FinalReportError("terminal gate receipt index is missing")
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise FinalReportError("terminal gate receipt row is invalid")
        path = str(row["path"])
        if path in output:
            raise FinalReportError("terminal gate receipt path is duplicated")
        if not _is_sha256(row.get("file_sha256")) or not isinstance(
            row.get("status"), str
        ):
            raise FinalReportError("terminal gate receipt binding is invalid")
        output[path] = row
    if set(output) != set(GATE_STATUS):
        raise FinalReportError("terminal must bind the exact nine controller gates")
    return output


def _load_gate(
    root: Path,
    index: Mapping[str, InventoryEntry],
    terminal_rows: Mapping[str, Mapping[str, Any]],
    logical: str,
) -> Mapping[str, Any]:
    expected_status = GATE_STATUS[logical]
    terminal_row = terminal_rows.get(logical)
    if terminal_row is None or terminal_row.get("status") != expected_status:
        raise FinalReportError(logical + " terminal status binding differs")
    gate, digest = _read_inventoried_json(root, index, logical, logical)
    _verify_content_sha(gate, logical)
    if (
        gate.get("schema_version") != CONTROLLER_SCHEMA
        or gate.get("status") != expected_status
        or terminal_row.get("file_sha256") != digest
    ):
        raise FinalReportError(logical + " gate schema/status/SHA binding differs")
    if logical != "control/source_code_freeze.json" and (
        gate.get("metric_artifacts_read_for_integrity_validation") is not True
        or gate.get(
            "metric_values_used_for_adaptive_model_checkpoint_threshold_or_condition_selection"
        )
        is not False
        or gate.get("metric_values_emitted_to_controller_stdout") is not False
        or gate.get("predeclared_integrity_gates_may_abort") is not True
    ):
        raise FinalReportError(logical + " integrity/non-selection flags differ")
    return gate


def _require_declared_path(
    root: Path, value: object, logical: str, description: str
) -> None:
    expected = root.joinpath(*PurePosixPath(logical).parts)
    if not isinstance(value, str) or value != str(expected):
        raise FinalReportError(description + " declared path differs")


def _bind_result(
    root: Path,
    gate: Mapping[str, Any],
    logical: str,
    digest: str,
    description: str,
) -> None:
    _require_declared_path(root, gate.get("result"), logical, description)
    if gate.get("result_sha256") != digest:
        raise FinalReportError(description + " gate result SHA-256 differs")


def _sha_map(
    value: object, names: Sequence[str], description: str
) -> Mapping[str, str]:
    mapping = _mapping(value, description)
    if set(mapping) != set(names) or any(not _is_sha256(mapping.get(name)) for name in names):
        raise FinalReportError(description + " method/SHA set differs")
    return {name: str(mapping[name]) for name in names}


def _declared_directory(root: Path, value: object, description: str) -> tuple[Path, str]:
    if not isinstance(value, str):
        raise FinalReportError(description + " is missing")
    candidate = Path(value)
    if not candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts[1:]):
        raise FinalReportError(description + " path is unsafe")
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise FinalReportError(description + " escapes final root") from error
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError as error:
            raise FinalReportError(description + " is missing") from error
        if stat.S_ISLNK(info.st_mode):
            raise FinalReportError(description + " traverses a symlink")
    if not stat.S_ISDIR(info.st_mode):
        raise FinalReportError(description + " must be a directory")
    return current, relative.as_posix()


def _artifact_rows_bound_without_open(
    root: Path,
    index: Mapping[str, InventoryEntry],
    directory_value: object,
    rows_value: object,
    *,
    identity_name: str,
    identities: Sequence[str],
    description: str,
) -> Mapping[str, Mapping[str, Any]]:
    _, directory_logical = _declared_directory(root, directory_value, description)
    rows = _list(rows_value, description + " artifacts")
    by_identity: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        item = _mapping(row, description + " artifact")
        identity = item.get(identity_name)
        if not isinstance(identity, str) or identity in by_identity:
            raise FinalReportError(description + " artifact identity is invalid")
        logical = item.get("path")
        if not isinstance(logical, str) or PurePosixPath(logical).suffix != ".jsonl":
            raise FinalReportError(description + " artifact must name JSONL")
        relative = PurePosixPath(logical)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise FinalReportError(description + " artifact path is unsafe")
        combined = str(PurePosixPath(directory_logical) / relative)
        entry = index.get(combined)
        if entry is None or item.get("sha256") != entry.sha256:
            raise FinalReportError(description + " artifact inventory SHA differs")
        by_identity[identity] = item
    if set(by_identity) != set(identities):
        raise FinalReportError(description + " artifact identity set differs")
    return by_identity


def _frozen_authority(
    root: Path,
    terminal: Mapping[str, Any],
    index: Mapping[str, InventoryEntry],
    gates: Mapping[str, Mapping[str, Any]],
    corrosion: Mapping[str, Any],
    translation: Mapping[str, Any],
) -> Mapping[str, Any]:
    freeze = gates["control/train_validation_freeze_gate.json"]
    n512 = _mapping(freeze.get("n512"), "N512 freeze authority")
    matched = _mapping(freeze.get("matched_mm"), "matched freeze authority")
    benchmark = _mapping(
        freeze.get("same_data_benchmarks"), "same-data benchmark freeze authority"
    )
    benchmark_methods = _mapping(
        benchmark.get("methods"), "same-data benchmark frozen methods"
    )
    queue_gate = gates["control/same_data_benchmark_queue_gate.json"]
    queue_source = _mapping(
        queue_gate.get("queue_source_authority"),
        "same-data benchmark queue source authority",
    )
    queue_reviews = _mapping(
        queue_source.get("reviewed_GO_markers"),
        "same-data benchmark reviewed GO markers",
    )
    expected_queue_reviews = {
        "queue_controller",
        "train_val_asset_freeze",
        "benchmark_environment",
        "pairingnet_runner",
        "shreddingnet_runner",
    }
    if set(queue_reviews) != expected_queue_reviews:
        raise FinalReportError("same-data benchmark reviewed GO inventory differs")
    for name in sorted(expected_queue_reviews):
        review = _mapping(queue_reviews[name], name + " reviewed GO marker")
        if (
            not isinstance(review.get("path"), str)
            or not str(review["path"]).startswith("/")
            or not isinstance(review.get("relative_path"), str)
            or not _is_sha256(review.get("file_sha256"))
            or not isinstance(review.get("schema_version"), str)
            or review.get("status") != "GO"
        ):
            raise FinalReportError(name + " reviewed GO marker binding differs")
    if (
        freeze.get("formal_method_inventory") != list(METHODS)
        or freeze.get(
            "formal_exact_six_winners_and_validation_thresholds_frozen_before_test_or_real_open"
        )
        is not True
        or set(benchmark_methods) != set(BENCHMARK_METHODS)
        or benchmark.get("all_winners_and_validation_thresholds_strict_loaded_cpu")
        is not True
        or benchmark.get("both_same_data_adaptations_not_exact_reproductions")
        is not True
        or benchmark.get("native_cm_fm_se_or_ga_claimed") is not False
        or queue_gate.get("exact_two_benchmark_methods_verified") is not True
        or queue_gate.get("sealed_test_or_real_opened") is not False
        or benchmark.get("queue_gate_file_sha256")
        != index["control/same_data_benchmark_queue_gate.json"].sha256
    ):
        raise FinalReportError("same-data benchmark exact-six freeze authority differs")
    n512_receipt = n512.get("receipt_sha256")
    matched_receipt = matched.get("receipt_sha256")
    convergence_receipt = n512.get("convergence_receipt_sha256")
    if (
        not _is_sha256(n512_receipt)
        or not _is_sha256(matched_receipt)
        or not _is_sha256(convergence_receipt)
    ):
        raise FinalReportError("frozen training receipt SHA-256 differs")
    n512_checkpoints = _sha_map(
        n512.get("winner_checkpoint_sha256_by_arm"),
        ("coarse_only", "full_n512"),
        "N512 checkpoint authority",
    )
    n512_thresholds = _sha_map(
        n512.get("validation_threshold_sha256_by_arm"),
        ("coarse_only", "full_n512"),
        "N512 threshold authority",
    )
    config_authority = _mapping(
        n512.get("canonical_model_and_loss_config_authority"),
        "canonical N512 model/loss config authority",
    )
    expected_config_keys = {
        "comparison",
        "model_config",
        "model_config_sha256",
        "loss_config",
        "loss_config_sha256",
        "model_config_sha256_by_arm",
        "loss_config_sha256_by_arm",
        "all_winner_configs_exactly_equal",
    }
    if set(config_authority) != expected_config_keys:
        raise FinalReportError("canonical N512 config authority schema differs")
    canonical_model_sha = _sha256_bytes(
        _canonical_bytes(CANONICAL_RACHEL_N512_MODEL_CONFIG)
    )
    canonical_loss_sha = _sha256_bytes(
        _canonical_bytes(CANONICAL_RACHEL_N512_LOSS_CONFIG)
    )
    model_sha_by_arm = _sha_map(
        config_authority.get("model_config_sha256_by_arm"),
        ("coarse_only", "full_n512"),
        "N512 model-config SHA authority",
    )
    loss_sha_by_arm = _sha_map(
        config_authority.get("loss_config_sha256_by_arm"),
        ("coarse_only", "full_n512"),
        "N512 loss-config SHA authority",
    )
    if (
        config_authority.get("comparison")
        != "complete_dataclass_field_mapping_exact_equality"
        or _canonical_bytes(config_authority.get("model_config"))
        != _canonical_bytes(CANONICAL_RACHEL_N512_MODEL_CONFIG)
        or _canonical_bytes(config_authority.get("loss_config"))
        != _canonical_bytes(CANONICAL_RACHEL_N512_LOSS_CONFIG)
        or config_authority.get("model_config_sha256") != canonical_model_sha
        or config_authority.get("loss_config_sha256") != canonical_loss_sha
        or any(value != canonical_model_sha for value in model_sha_by_arm.values())
        or any(value != canonical_loss_sha for value in loss_sha_by_arm.values())
        or config_authority.get("all_winner_configs_exactly_equal") is not True
    ):
        raise FinalReportError(
            "winner model/loss config differs from the complete canonical authority"
        )
    matched_checkpoints = _sha_map(
        matched.get("winner_checkpoint_sha256_by_method"),
        ("matched_mm_converged", "matched_mm_same_exposure_epoch5"),
        "matched checkpoint authority",
    )
    matched_thresholds = _sha_map(
        matched.get("validation_threshold_sha256_by_method"),
        ("matched_mm_converged", "matched_mm_same_exposure_epoch5"),
        "matched threshold authority",
    )
    benchmark_checkpoint_stages: dict[str, Mapping[str, str]] = {}
    benchmark_checkpoints: dict[str, str] = {}
    benchmark_thresholds: dict[str, str] = {}
    benchmark_adaptations: dict[str, Mapping[str, Any]] = {}
    queue_by_method = {
        "pairingnet_adapted": _mapping(
            queue_gate.get("pairingnet"), "queue PairingNet authority"
        ),
        "shreddingnet_adapted": _mapping(
            queue_gate.get("shreddingnet"), "queue ShreddingNet authority"
        ),
    }
    for method in BENCHMARK_METHODS:
        row = _mapping(benchmark_methods[method], method + " frozen authority")
        stages = _mapping(
            row.get("checkpoint_sha256_by_stage"), method + " checkpoint stages"
        )
        if not stages or any(not _is_sha256(value) for value in stages.values()):
            raise FinalReportError(method + " checkpoint-stage authority differs")
        threshold = _mapping(
            row.get("validation_threshold"), method + " validation threshold"
        )
        threshold_checkpoint = row.get("validation_threshold_checkpoint_sha256")
        adaptation = _mapping(row.get("adaptation"), method + " adaptation")
        if (
            not _is_sha256(row.get("freeze_authority_sha256"))
            or not _is_sha256(row.get("validation_threshold_sha256"))
            or not _is_sha256(threshold_checkpoint)
            or threshold.get("checkpoint_sha256") != threshold_checkpoint
            or threshold_checkpoint not in set(stages.values())
            or row.get("same_data_method_adaptation_not_exact_reproduction")
            is not True
            or row.get("native_cm_fm_se_or_ga_claimed") is not False
            or adaptation.get("claim")
            != "same_data_method_adaptation_not_exact_reproduction"
            or adaptation.get("mask_only") is not True
        ):
            raise FinalReportError(method + " frozen threshold/adaptation differs")
        if method == "pairingnet_adapted":
            if (
                adaptation.get("primary_pose")
                != "upright_translation_only_consensus"
                or adaptation.get("pair_head_is_official_component") is not False
                or adaptation.get("dual_softmax_is_sinkhorn") is not False
            ):
                raise FinalReportError("PairingNet adaptation disclosure differs")
        elif (
            adaptation.get("rachel_n512") is not True
            or adaptation.get("upright_known_translation_only_primary") is not True
            or adaptation.get("global_assembly_performed") is not False
            or adaptation.get("native_cm_fm_se_or_ga_claimed") is not False
            or adaptation.get("balanced_pair_list_diagnostics_are_native_cm_fm_se")
            is not False
        ):
            raise FinalReportError("ShreddingNet adaptation disclosure differs")
        queue_row = queue_by_method[method]
        if method == "pairingnet_adapted":
            queue_stages = {"winner": queue_row.get("winner_checkpoint_sha256")}
            queue_freeze_sha = queue_row.get("completion_receipt_sha256")
        else:
            queue_stages = queue_row.get("winner_checkpoint_sha256_by_stage")
            queue_freeze_sha = queue_row.get("train_val_freeze_file_sha256")
        if (
            dict(stages) != queue_stages
            or row.get("freeze_authority_sha256") != queue_freeze_sha
            or row.get("validation_threshold_sha256")
            != queue_row.get("validation_threshold_artifact_sha256")
        ):
            raise FinalReportError(method + " queue/freeze authority differs")
        benchmark_checkpoint_stages[method] = {
            str(name): str(value) for name, value in stages.items()
        }
        benchmark_checkpoints[method] = str(threshold_checkpoint)
        benchmark_thresholds[method] = str(row["validation_threshold_sha256"])
        benchmark_adaptations[method] = dict(adaptation)
    if (
        n512.get("both_arms_validation_plateau_verified") is not True
        or matched.get("converged_and_epoch5_validation_winners_frozen") is not True
        or matched.get("validation_plateau_verified") is not True
        or freeze.get("same_dataset_seed_population_and_exposure_alignment_verified")
        is not True
        or freeze.get("test_accessed_by_controller_preflight") is not False
        or freeze.get("real_external_test_accessed_by_controller_preflight") is not False
    ):
        raise FinalReportError("frozen winner/dataset authority facts differ")
    provenance = _mapping(
        freeze.get("rachel_data_provenance"), "Rachel data provenance authority"
    )
    if set(provenance) != {
        "sidecar_path",
        "sidecar_sha256",
        "schema_version",
        "status",
        "dataset_root",
        "source_authority",
        "selection_contract",
        "files",
        "preprocess_receipt_schema_version",
        "preprocess_receipt_status",
        "preprocess_summary_exactly_embedded_in_receipt",
        "selection_summary_schema_version",
        "test_manifest_content_read_or_hashed_by_pretest_sidecar",
        "verified_before_current_test_or_real_open",
    }:
        raise FinalReportError("Rachel data provenance schema differs")
    source_authority = _mapping(
        provenance.get("source_authority"), "Rachel source authority"
    )
    selection_contract = _mapping(
        provenance.get("selection_contract"), "Rachel selection contract"
    )
    provenance_files = _mapping(provenance.get("files"), "Rachel provenance files")
    if (
        provenance.get("schema_version")
        != "rachel-n512-pretest-data-provenance/1.0"
        or provenance.get("status")
        != "frozen_before_current_final_controller_test_open"
        or not isinstance(provenance.get("sidecar_path"), str)
        or not str(provenance["sidecar_path"]).startswith("/")
        or not _is_sha256(provenance.get("sidecar_sha256"))
        or not isinstance(provenance.get("dataset_root"), str)
        or not str(provenance["dataset_root"]).startswith("/")
        or _canonical_bytes(source_authority)
        != _canonical_bytes(
            {
                "description": "Rachel RGB JPEG plus colocated label.csv only",
                "shredding_data_read": False,
            }
        )
        or _canonical_bytes(selection_contract)
        != _canonical_bytes(CANONICAL_RACHEL_SELECTION_CONTRACT)
        or provenance.get("preprocess_receipt_schema_version")
        != "rachel-pairwise-n512-preprocessing/1.0"
        or provenance.get("preprocess_receipt_status")
        != "complete_rachel_pairwise_n512_30k"
        or provenance.get("preprocess_summary_exactly_embedded_in_receipt")
        is not True
        or provenance.get("selection_summary_schema_version")
        != "rachel-pairwise-30k-selection/1.0"
        or provenance.get(
            "test_manifest_content_read_or_hashed_by_pretest_sidecar"
        )
        is not False
        or provenance.get("verified_before_current_test_or_real_open") is not True
    ):
        raise FinalReportError("Rachel data provenance authority differs")
    expected_provenance_files = {
        "preprocess_receipt": ("preprocess_receipt.json", None),
        "preprocess_summary": ("qa/preprocess_summary.json", None),
        "train_manifest": ("pairs/train.jsonl", 24_000),
        "validation_manifest": ("pairs/val.jsonl", 3_000),
    }
    if set(provenance_files) != set(expected_provenance_files):
        raise FinalReportError("Rachel provenance file inventory differs")
    normalized_provenance_files: dict[str, Mapping[str, Any]] = {}
    for name, (expected_path, expected_lines) in expected_provenance_files.items():
        row = _mapping(provenance_files[name], "Rachel " + name)
        expected_keys = {"relative_path", "sha256"}
        if expected_lines is not None:
            expected_keys.add("line_count")
        if (
            set(row) != expected_keys
            or row.get("relative_path") != expected_path
            or not _is_sha256(row.get("sha256"))
            or (
                expected_lines is not None
                and row.get("line_count") != expected_lines
            )
        ):
            raise FinalReportError("Rachel " + name + " provenance differs")
        normalized_provenance_files[name] = dict(row)
    terminal_authority = _mapping(
        terminal.get("frozen_training_authority"), "terminal frozen authority"
    )
    if (
        set(terminal_authority)
        != {
            "n512_receipt_sha256",
            "matched_mm_receipt_sha256",
            "same_data_benchmarks",
            "formal_method_inventory",
            "formal_exact_six_winners_and_thresholds_frozen_before_test_or_real_open",
            "canonical_model_and_loss_config_authority",
            "rachel_data_provenance",
            "receipts_winner_checkpoints_configs_and_data_provenance_reverified_after_all_evaluations",
        }
        or terminal_authority.get("n512_receipt_sha256") != n512_receipt
        or terminal_authority.get("matched_mm_receipt_sha256") != matched_receipt
        or _canonical_bytes(terminal_authority.get("same_data_benchmarks"))
        != _canonical_bytes(benchmark)
        or terminal_authority.get("formal_method_inventory") != list(METHODS)
        or terminal_authority.get(
            "formal_exact_six_winners_and_thresholds_frozen_before_test_or_real_open"
        )
        is not True
        or _canonical_bytes(
            terminal_authority.get("canonical_model_and_loss_config_authority")
        )
        != _canonical_bytes(config_authority)
        or _canonical_bytes(terminal_authority.get("rachel_data_provenance"))
        != _canonical_bytes(provenance)
        or terminal_authority.get(
            "receipts_winner_checkpoints_configs_and_data_provenance_reverified_after_all_evaluations"
        )
        is not True
    ):
        raise FinalReportError("terminal frozen training authority differs")
    checkpoints = {
        **n512_checkpoints,
        **matched_checkpoints,
        **benchmark_checkpoints,
    }
    thresholds = {
        **n512_thresholds,
        **matched_thresholds,
        **benchmark_thresholds,
    }
    alignment = _mapping(
        freeze.get("alignment_hash_evidence"), "training alignment hash evidence"
    )
    train_alignment = _mapping(
        alignment.get("train_manifest"), "train manifest alignment"
    )
    validation_alignment = _mapping(
        alignment.get("validation_manifest"), "validation manifest alignment"
    )
    fingerprints = _sha_map(
        alignment.get("training_fingerprints_recomputed"),
        ("rachel_n512", "matched_mm"),
        "training fingerprints",
    )
    validation_order = _mapping(
        alignment.get("validation_pair_order"), "validation pair-order alignment"
    )
    validation_sources = _mapping(
        validation_order.get("sources"), "validation score sources"
    )
    limitations = _mapping(alignment.get("limitations"), "alignment limitations")
    pair_order_sha = validation_alignment.get("pair_order_fingerprint_sha256")
    if (
        alignment.get("claim_level")
        != "same_frozen_train_val_manifest_content_and_exact_validation_pair_order;per_epoch_training_presentation_order_not_provable"
        or not isinstance(alignment.get("canonical_dataset_root"), str)
        or not str(alignment["canonical_dataset_root"]).startswith("/")
        or alignment.get("canonical_dataset_root")
        != provenance.get("dataset_root")
        or type(alignment.get("canonical_seed")) is not int
        or train_alignment.get("pair_count") != 24000
        or validation_alignment.get("pair_count") != 3000
        or train_alignment.get("bound_to_both_training_fingerprints") is not True
        or validation_alignment.get("bound_to_both_training_fingerprints") is not True
        or validation_alignment.get(
            "exact_order_equal_to_all_four_validation_score_artifacts"
        )
        is not True
        or not _is_sha256(train_alignment.get("content_sha256"))
        or not _is_sha256(validation_alignment.get("content_sha256"))
        or not _is_sha256(train_alignment.get("pair_order_fingerprint_sha256"))
        or not _is_sha256(pair_order_sha)
        or validation_order.get("pair_count") != 3000
        or validation_order.get("exact_order_equal_across_four_frozen_thresholds")
        is not True
        or validation_order.get("pair_order_fingerprint_sha256") != pair_order_sha
        or set(validation_sources) != set(BASE_METHODS)
        or limitations.get("exact_per_epoch_training_pair_presentation_order_saved")
        is not False
        or limitations.get("exact_per_epoch_training_pair_presentation_order_claimed_equal")
        is not False
    ):
        raise FinalReportError("training dataset/exposure alignment evidence differs")
    if (
        train_alignment.get("content_sha256")
        != normalized_provenance_files["train_manifest"]["sha256"]
        or validation_alignment.get("content_sha256")
        != normalized_provenance_files["validation_manifest"]["sha256"]
    ):
        raise FinalReportError(
            "Rachel provenance and frozen training manifest SHA authority differ"
        )
    for method in BASE_METHODS:
        row = _mapping(validation_sources[method], method + " validation source")
        if (
            row.get("pair_count") != 3000
            or row.get("pair_order_fingerprint_sha256") != pair_order_sha
            or row.get("threshold_bound_to_pair_order") is not True
            or row.get("threshold_checkpoint_sha256") != checkpoints[method]
            or not _is_sha256(row.get("artifact_sha256"))
        ):
            raise FinalReportError(method + " validation source alignment differs")

    benchmark_alignment = _mapping(
        benchmark.get("training_manifest_alignment"),
        "same-data benchmark training alignment",
    )
    benchmark_manifest_sha = _mapping(
        benchmark_alignment.get("manifest_content_sha256"),
        "same-data benchmark manifest hashes",
    )
    benchmark_alignment_methods = _mapping(
        benchmark_alignment.get("methods"),
        "same-data benchmark alignment methods",
    )
    if (
        benchmark_alignment.get(
            "exact_manifest_bytes_equal_across_n512_pairingnet_and_shreddingnet"
        )
        is not True
        or benchmark_manifest_sha
        != {
            "train": train_alignment.get("content_sha256"),
            "val": validation_alignment.get("content_sha256"),
        }
        or set(benchmark_alignment_methods) != set(BENCHMARK_METHODS)
    ):
        raise FinalReportError("same-data benchmark manifest alignment differs")
    for method in BENCHMARK_METHODS:
        aligned = _mapping(
            benchmark_alignment_methods[method], method + " aligned provenance"
        )
        frozen = _mapping(benchmark_methods[method], method + " frozen provenance")
        if (
            aligned.get("training_manifest_sha256") != benchmark_manifest_sha
            or aligned.get("checkpoint_sha256_by_stage")
            != frozen.get("checkpoint_sha256_by_stage")
            or aligned.get("freeze_authority_sha256")
            != frozen.get("freeze_authority_sha256")
            or aligned.get("validation_threshold_sha256")
            != frozen.get("validation_threshold_sha256")
            or aligned.get("adaptation") != frozen.get("adaptation")
        ):
            raise FinalReportError(method + " training alignment provenance differs")

    sealed = gates["control/sealed_gate.json"]
    sealed_rows = _artifact_rows_bound_without_open(
        root,
        index,
        sealed.get("result_directory"),
        sealed.get("pair_score_artifacts"),
        identity_name="method",
        identities=METHODS,
        description="sealed pair scores",
    )
    if (
        sealed.get("formal_exact_six") is not True
        or sealed.get("six_methods_verified") is not True
        or sealed.get("method_inventory") != list(METHODS)
        or sealed.get("same_data_benchmark_adaptations_not_exact_reproductions")
        is not True
        or sealed.get("native_cm_fm_se_or_ga_claimed") is not False
        or sealed.get("pair_count") != 3000
        or any(
            sealed_rows[method].get("checkpoint_sha256") != checkpoints[method]
            or sealed_rows[method].get("validation_threshold_sha256")
            != thresholds[method]
            for method in METHODS
        )
        or any(
            sealed_rows[method].get("model_config_sha256")
            != model_sha_by_arm[method]
            or sealed_rows[method].get("loss_config_sha256")
            != loss_sha_by_arm[method]
            for method in ("coarse_only", "full_n512")
        )
        or any(
            sealed_rows[method].get("checkpoint_sha256_by_stage")
            != benchmark_checkpoint_stages[method]
            or sealed_rows[method].get("freeze_authority_sha256")
            != benchmark_methods[method].get("freeze_authority_sha256")
            or sealed_rows[method].get("adaptation")
            != benchmark_adaptations[method]
            or sealed_rows[method].get("native_cm_fm_se_or_ga_claimed") is not False
            for method in BENCHMARK_METHODS
        )
    ):
        raise FinalReportError(
            "sealed checkpoint/threshold/model/loss authority differs"
        )
    corrosion_gate = gates["control/corrosion_gate.json"]
    if (
        corrosion_gate.get("formal_exact_six") is not True
        or corrosion_gate.get("six_methods_verified") is not True
        or corrosion_gate.get("method_inventory") != list(CORROSION_METHODS)
        or corrosion_gate.get("same_data_benchmark_methods_included") is not True
        or corrosion_gate.get(
            "same_data_benchmark_adaptations_not_exact_reproductions"
        )
        is not True
        or corrosion_gate.get("benchmark_correspondence_not_applicable_under_corrosion")
        is not True
    ):
        raise FinalReportError("corrosion gate exact-six disclosure differs")
    _artifact_rows_bound_without_open(
        root,
        index,
        corrosion_gate.get("result_directory"),
        corrosion_gate.get("condition_pair_score_artifacts"),
        identity_name="condition",
        identities=CONDITIONS,
        description="corrosion pair scores",
    )
    sources = _mapping(corrosion.get("source_training_runs"), "corrosion sources")
    corrosion_n512 = _mapping(sources.get("rachel_n512"), "corrosion N512 source")
    corrosion_matched = _mapping(sources.get("matched_mm"), "corrosion matched source")
    corrosion_benchmarks = _mapping(
        sources.get("same_data_benchmarks"), "corrosion benchmark sources"
    )
    if (
        corrosion_n512.get("receipt_sha256") != n512_receipt
        or corrosion_matched.get("receipt_sha256") != matched_receipt
        or corrosion_n512.get("formal_plateau_verified_before_test_open") is not True
        or corrosion_matched.get("formal_plateau_verified_before_test_open") is not True
        or corrosion_benchmarks.get(
            "all_winners_and_validation_thresholds_frozen_before_test_open"
        )
        is not True
    ):
        raise FinalReportError("corrosion frozen training authority differs")
    translation_source = _mapping(
        translation.get("source_pair_only_evaluation"), "translation source authority"
    )
    if (
        translation_source.get("source_training_receipt_sha256") != n512_receipt
        or translation_source.get("source_matched_receipt_sha256") != matched_receipt
        or translation_source.get("formal_combined_authority_gate_passed") is not True
        or translation_source.get("exact_method_order") != list(METHODS)
        or translation_source.get("checkpoint_sha256_by_method") != checkpoints
    ):
        raise FinalReportError("translation frozen training authority differs")

    source_gate = gates["control/source_code_freeze.json"]
    source_files = _list(source_gate.get("files"), "source freeze files")
    if source_gate.get("file_count") != len(source_files):
        raise FinalReportError("source freeze file count differs")
    source_by_path: dict[str, Mapping[str, Any]] = {}
    for row in source_files:
        item = _mapping(row, "source freeze row")
        path = item.get("path")
        if (
            not isinstance(path, str)
            or path in source_by_path
            or type(item.get("size")) is not int
            or item["size"] < 0
            or not _is_sha256(item.get("sha256"))
        ):
            raise FinalReportError("source freeze row differs")
        source_by_path[path] = item
    required_source = {
        "experiments/rachel_n512_formal_30k/paired_cluster_bootstrap.py",
        "experiments/rachel_n512_formal_30k/run_final_pairwise_protocol.sh",
        "staging/pairwise_v0_2/training/rachel_n512_sealed_test.py",
        "staging/pairwise_v0_2/baselines/rachel_same_data_benchmark_eval_adapter.py",
        "staging/pairwise_v0_2/baselines/rachel_pairingnet_benchmark.py",
        "staging/pairwise_v0_2/baselines/rachel_shreddingnet_benchmark.py",
        "staging/pairwise_v0_2/baselines/rachel_n512_corrosion_robustness.py",
        "staging/pairwise_v0_2/baselines/rachel_n512_real_external.py",
        "staging/pairwise_v0_2/baselines/rachel_n512_real_translation_gt.py",
    }
    if (
        not required_source.issubset(source_by_path)
        or source_gate.get("test_or_real_manifest_read") is not False
        or source_gate.get("scope")
        != "all regular .py/.sh under staging/pairwise_v0_2 and experiments/rachel_n512_formal_30k"
        or not isinstance(source_gate.get("source_root"), str)
        or not str(source_gate["source_root"]).startswith("/")
    ):
        raise FinalReportError("source-code freeze authority differs")
    return {
        "n512_training_receipt_sha256": str(n512_receipt),
        "matched_mm_training_receipt_sha256": str(matched_receipt),
        "n512_convergence_receipt_sha256": str(convergence_receipt),
        "winner_checkpoint_sha256_by_method": checkpoints,
        "winner_checkpoint_sha256_by_stage_for_same_data_benchmarks": (
            benchmark_checkpoint_stages
        ),
        "validation_threshold_sha256_by_method": thresholds,
        "same_data_benchmark_adaptation_disclosures": benchmark_adaptations,
        "same_data_benchmark_queue_gate_file_sha256": index[
            "control/same_data_benchmark_queue_gate.json"
        ].sha256,
        "same_data_benchmark_queue_terminal": dict(
            _mapping(queue_gate.get("queue_terminal"), "queue terminal binding")
        ),
        "same_data_benchmark_reviewed_GO_markers": {
            name: dict(queue_reviews[name]) for name in sorted(queue_reviews)
        },
        "formal_method_inventory": list(METHODS),
        "formal_exact_six_frozen_before_test_or_real_open": True,
        "same_dataset_seed_population_and_exposure_alignment_verified": True,
        "training_alignment_evidence_sha256": _sha256_bytes(
            _canonical_bytes(alignment)
        ),
        "training_fingerprints_recomputed": fingerprints,
        "canonical_model_config": dict(CANONICAL_RACHEL_N512_MODEL_CONFIG),
        "canonical_model_config_sha256": canonical_model_sha,
        "canonical_loss_config": dict(CANONICAL_RACHEL_N512_LOSS_CONFIG),
        "canonical_loss_config_sha256": canonical_loss_sha,
        "canonical_model_and_loss_config_authority_sha256": _sha256_bytes(
            _canonical_bytes(config_authority)
        ),
        "rachel_data_provenance_sidecar_path": str(provenance["sidecar_path"]),
        "rachel_data_provenance_sidecar_sha256": str(
            provenance["sidecar_sha256"]
        ),
        "rachel_dataset_root": str(provenance["dataset_root"]),
        "rachel_source_authority": dict(source_authority),
        "rachel_selection_contract": dict(selection_contract),
        "rachel_preprocess_receipt_schema_version": str(
            provenance["preprocess_receipt_schema_version"]
        ),
        "rachel_preprocess_receipt_status": str(
            provenance["preprocess_receipt_status"]
        ),
        "rachel_preprocess_summary_exactly_embedded_in_receipt": True,
        "rachel_selection_summary_schema_version": str(
            provenance["selection_summary_schema_version"]
        ),
        "rachel_provenance_files": normalized_provenance_files,
        "rachel_preprocess_receipt_sha256": str(
            normalized_provenance_files["preprocess_receipt"]["sha256"]
        ),
        "rachel_preprocess_summary_sha256": str(
            normalized_provenance_files["preprocess_summary"]["sha256"]
        ),
        "train_manifest_content_sha256": str(train_alignment["content_sha256"]),
        "validation_manifest_content_sha256": str(
            validation_alignment["content_sha256"]
        ),
        "validation_pair_order_fingerprint_sha256": str(pair_order_sha),
        "all_winners_and_receipts_reverified_after_evaluations": True,
        "source_code_freeze_gate_file_sha256": index[
            "control/source_code_freeze.json"
        ].sha256,
        "source_code_file_count": len(source_files),
        "required_runtime_source_files_present": sorted(required_source),
    }


def _load_aggregate_inputs(final_root: Path) -> AggregateInputs:
    root = _resolve_directory(final_root, "final root")
    (
        terminal,
        inventory_document,
        terminal_file_sha256,
        inventory_file_sha256,
    ) = _load_terminal_inventory_bundle(root)
    index = _inventory_index(root, terminal, inventory_document)
    terminal_rows = _terminal_gate_rows(terminal)
    gates = {
        logical: _load_gate(root, index, terminal_rows, logical)
        for logical in GATE_STATUS
    }

    sealed_candidates = [
        logical
        for logical in index
        if len(PurePosixPath(logical).parts) == 3
        and PurePosixPath(logical).parts[0] == "sealed"
        and PurePosixPath(logical).name == "test_receipt.json"
    ]
    if len(sealed_candidates) != 1:
        raise FinalReportError("content inventory must name one sealed receipt")
    sealed_logical = sealed_candidates[0]
    sealed, sealed_sha = _read_inventoried_json(
        root, index, sealed_logical, "sealed exact-six aggregate receipt"
    )
    sealed_gate = gates["control/sealed_gate.json"]
    _require_declared_path(
        root,
        sealed_gate.get("result_directory"),
        str(PurePosixPath(sealed_logical).parent),
        "sealed result directory",
    )
    sealed_rows = sealed.get("arm_results")
    if (
        sealed_gate.get("receipt_sha256") != sealed_sha
        or sealed_gate.get("formal_exact_six") is not True
        or sealed_gate.get("six_methods_verified") is not True
        or sealed_gate.get("method_inventory") != list(METHODS)
        or sealed.get("schema_version")
        != "rachel-n512-sealed-synthetic-test/1.0"
        or sealed.get("status") != "complete_frozen_synthetic_test_only"
        or sealed.get("formal_evaluation") is not True
        or sealed.get("compatibility_mode") is not False
        or not isinstance(sealed_rows, list)
        or [row.get("arm") for row in sealed_rows if isinstance(row, Mapping)]
        != list(METHODS)
        or len(sealed_rows) != len(METHODS)
    ):
        raise FinalReportError("sealed exact-six aggregate receipt differs")

    synthetic, synthetic_sha = _read_inventoried_json(
        root, index, SYNTHETIC_STATS_RELATIVE, "synthetic aggregate statistics"
    )
    _bind_result(
        root,
        gates["control/synthetic_stats_gate.json"],
        SYNTHETIC_STATS_RELATIVE,
        synthetic_sha,
        "synthetic aggregate statistics",
    )
    if (
        synthetic.get("schema_version")
        != "rachel-n512-paired-endpoint-bootstrap/1.1"
        or synthetic.get("status") != "complete_formal_exact_six"
        or synthetic.get("formal_evaluation") is not True
        or synthetic.get("compatibility_mode") is not False
        or synthetic.get("methods") != list(METHODS)
        or not isinstance(synthetic.get("input"), Mapping)
        or synthetic["input"].get("source_kind") != "sealed_synthetic"
    ):
        raise FinalReportError("synthetic aggregate statistics schema/status differs")
    synthetic_source = synthetic["input"].get("source_sha256")
    if (
        not _is_sha256(synthetic_source)
        or gates["control/synthetic_stats_gate.json"].get("source_receipt_sha256")
        != synthetic_source
        or gates["control/sealed_gate.json"].get("receipt_sha256")
        != synthetic_source
    ):
        raise FinalReportError("synthetic source receipt SHA binding differs")

    real_stats, real_stats_sha = _read_inventoried_json(
        root, index, REAL_STATS_RELATIVE, "real aggregate statistics"
    )
    _bind_result(
        root,
        gates["control/real_stats_gate.json"],
        REAL_STATS_RELATIVE,
        real_stats_sha,
        "real aggregate statistics",
    )
    if (
        real_stats.get("schema_version")
        != "rachel-n512-paired-endpoint-bootstrap/1.1"
        or real_stats.get("status") != "complete_formal_exact_six"
        or real_stats.get("formal_evaluation") is not True
        or real_stats.get("compatibility_mode") is not False
        or real_stats.get("methods") != list(METHODS)
        or not isinstance(real_stats.get("input"), Mapping)
        or real_stats["input"].get("source_kind") != "real_balanced1016"
    ):
        raise FinalReportError("real aggregate statistics schema/status differs")
    real_source = real_stats["input"].get("source_sha256")
    real_gate = gates["control/real_gate.json"]
    _require_declared_path(root, real_gate.get("result"), REAL_SOURCE_RELATIVE, "real source")
    source_entry = index.get(REAL_SOURCE_RELATIVE)
    if (
        source_entry is None
        or not _is_sha256(real_source)
        or real_gate.get("result_sha256") != source_entry.sha256
        or gates["control/real_stats_gate.json"].get("source_result_sha256")
        != real_source
        or real_source != source_entry.sha256
    ):
        raise FinalReportError("real aggregate source SHA binding differs")

    translation, translation_sha = _read_inventoried_json(
        root, index, REAL_TRANSLATION_RELATIVE, "real translation-GT aggregate"
    )
    _verify_content_sha(translation, "real translation-GT aggregate")
    _bind_result(
        root,
        gates["control/real_translation_gate.json"],
        REAL_TRANSLATION_RELATIVE,
        translation_sha,
        "real translation-GT aggregate",
    )
    translation_source = translation.get("source_pair_only_evaluation")
    if (
        translation.get("schema_version")
        != "rachel-n512-real-translation-gt-posteval/1.0"
        or translation.get("status")
        != "complete_postprediction_real_positive_translation_gt"
        or translation.get("formal_evaluation") is not True
        or translation.get("compatibility_mode") is not False
        or not isinstance(translation_source, Mapping)
        or translation_source.get("file_sha256_frozen_before_gt_open")
        != real_source
        or gates["control/real_translation_gate.json"].get(
            "source_pair_only_result_sha256"
        )
        != real_source
    ):
        raise FinalReportError("real translation-GT source SHA binding differs")

    candidates = [
        logical
        for logical in index
        if len(PurePosixPath(logical).parts) == 3
        and PurePosixPath(logical).parts[0] == "corrosion"
        and PurePosixPath(logical).name == "robustness_receipt.json"
    ]
    if len(candidates) != 1:
        raise FinalReportError("content inventory must name one corrosion receipt")
    corrosion_logical = candidates[0]
    corrosion, corrosion_sha = _read_inventoried_json(
        root, index, corrosion_logical, "corrosion aggregate receipt"
    )
    corrosion_gate = gates["control/corrosion_gate.json"]
    _require_declared_path(
        root,
        corrosion_gate.get("result_directory"),
        str(PurePosixPath(corrosion_logical).parent),
        "corrosion result directory",
    )
    if (
        corrosion_gate.get("receipt_sha256") != corrosion_sha
        or corrosion.get("schema_version")
        != "rachel-n512-corrosion-robustness/2.0"
        or corrosion.get("status")
        != "complete_formal_exact_six_frozen_synthetic_corrosion_only"
        or corrosion.get("formal_evaluation") is not True
        or corrosion.get("compatibility_mode") is not False
        or not isinstance(corrosion.get("clean_reference"), Mapping)
        or corrosion_gate.get("formal_exact_six") is not True
        or corrosion_gate.get("same_data_benchmark_methods_included") is not True
        or corrosion_gate.get("method_inventory") != list(CORROSION_METHODS)
        or not isinstance(corrosion_gate.get("clean_reference"), Mapping)
        or corrosion["clean_reference"].get("receipt_sha256")
        != corrosion_gate["clean_reference"].get("receipt_sha256")
        or corrosion_gate["clean_reference"].get("receipt_sha256")
        != synthetic_source
    ):
        raise FinalReportError("corrosion aggregate receipt/source SHA binding differs")

    authority = _frozen_authority(
        root, terminal, index, gates, corrosion, translation
    )
    member_snapshot = _member_lstat_snapshot(root, index)

    return AggregateInputs(
        root=root,
        terminal=terminal,
        inventory=index,
        gates=gates,
        sealed_receipt=sealed,
        synthetic_stats=synthetic,
        real_stats=real_stats,
        corrosion_receipt=corrosion,
        real_translation=translation,
        artifact_sha256={
            "synthetic_stats": synthetic_sha,
            "sealed_exact_six_receipt": sealed_sha,
            "real_stats": real_stats_sha,
            "corrosion_receipt": corrosion_sha,
            "real_translation_gt": translation_sha,
            "real_pair_only_unopened": str(real_source),
        },
        frozen_authority=authority,
        terminal_file_sha256=terminal_file_sha256,
        inventory_file_sha256=inventory_file_sha256,
        member_lstat_snapshot=member_snapshot,
    )


def _atomic_write_no_replace(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FinalReportError("refusing to overwrite report output")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + target.name + ".tmp-", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FinalReportError("refusing to overwrite report output") from error
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    inputs = _load_aggregate_inputs(arguments.final_root)
    output = _render_and_publish(inputs, arguments.output_directory)
    print(str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
