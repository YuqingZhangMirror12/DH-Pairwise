#!/usr/bin/env python3
"""Extract a fail-closed, aggregate-only summary from Rachel exact-six evidence.

The extractor is deliberately independent from the evaluation launcher.  It
opens only the six aggregate JSON inputs named on the command line (plus an
optional resource receipt), validates their identities and cross-file hash
bindings, and writes canonical JSON and an optional concise Markdown view.  It
does not open model checkpoints, masks, images, or pair-score JSONL members.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "rachel-exact6-evidence-summary/1.0"
RESOURCE_SCHEMA_VERSION = "rachel-pairwise-resource-summary/1.0"
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
TOLERANCES = (2, 5, 8, 10)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_901


class EvidenceExtractionError(RuntimeError):
    """An input identity, metric, population, or hash binding failed."""


def _reject_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Mapping[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceExtractionError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvidenceExtractionError("non-finite JSON constant: " + value)


def _check_finite(value: object, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceExtractionError(location + " contains a non-finite value")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_finite(item, location + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, "{}[{}]".format(location, index))


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
        raise EvidenceExtractionError("value is not canonical finite JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json(path_value: Path, description: str) -> Tuple[Mapping[str, Any], str]:
    path = Path(path_value).expanduser()
    try:
        info = os.lstat(path)
    except OSError as error:
        raise EvidenceExtractionError(description + " is missing") from error
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise EvidenceExtractionError(description + " must be a non-symlink regular file")
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except EvidenceExtractionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceExtractionError(description + " is not strict UTF-8 JSON") from error
    after = os.lstat(path)
    if (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise EvidenceExtractionError(description + " changed while it was read")
    if not isinstance(value, Mapping):
        raise EvidenceExtractionError(description + " root must be an object")
    _check_finite(value)
    return value, _sha256_bytes(payload)


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceExtractionError(description + " must be an object")
    return value


def _list(value: object, description: str) -> list:
    if not isinstance(value, list):
        raise EvidenceExtractionError(description + " must be an array")
    return value


def _integer(value: object, description: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:  # noqa: E721
        raise EvidenceExtractionError(description + " must be an integer >= {}".format(minimum))
    return int(value)


def _number(
    value: object,
    description: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceExtractionError(description + " must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise EvidenceExtractionError(description + " must be finite")
    if minimum is not None and output < minimum:
        raise EvidenceExtractionError(description + " is below its allowed minimum")
    if maximum is not None and output > maximum:
        raise EvidenceExtractionError(description + " exceeds its allowed maximum")
    return output


def _probability(value: object, description: str) -> float:
    return _number(value, description, minimum=0.0, maximum=1.0)


def _require_identity(
    value: Mapping[str, Any],
    *,
    schema: str,
    status: str,
    description: str,
) -> None:
    if value.get("schema_version") != schema or value.get("status") != status:
        raise EvidenceExtractionError(description + " schema/status differs")


def _require_formal(value: Mapping[str, Any], description: str) -> None:
    if value.get("formal_evaluation") is not True or value.get("compatibility_mode") is not False:
        raise EvidenceExtractionError(description + " is not a formal, non-compatibility artifact")


def _require_methods(value: object, description: str) -> None:
    if value != list(METHODS):
        raise EvidenceExtractionError(description + " method order differs")


def _verify_content_sha(value: Mapping[str, Any], description: str) -> str:
    declared = value.get("content_sha256")
    if not _is_sha256(declared):
        raise EvidenceExtractionError(description + " lacks a valid content_sha256")
    body = dict(value)
    del body["content_sha256"]
    if _sha256_bytes(_canonical_bytes(body)) != declared:
        raise EvidenceExtractionError(description + " content_sha256 differs")
    return str(declared)


def _classification_metrics(value: object, description: str) -> Mapping[str, Any]:
    row = _mapping(value, description)
    output = {}
    for metric in ("auroc", "auprc", "f1", "recall"):
        output[metric] = _probability(row.get(metric), description + "." + metric)
    return output


def _normalize_coverage(value: object, total: int, description: str) -> Mapping[str, Any]:
    row = _mapping(value, description)
    population = row.get("population_count", row.get("record_count"))
    valid_positive = row.get("positive_valid_count", row.get("valid_positive_count"))
    valid_negative = row.get("negative_valid_count", row.get("valid_negative_count"))
    population_count = _integer(population, description + ".population_count", 1)
    if population_count != total:
        raise EvidenceExtractionError(description + " population count differs")
    valid_count = _integer(row.get("valid_count"), description + ".valid_count")
    positive_count = _integer(row.get("positive_count"), description + ".positive_count")
    negative_count = _integer(row.get("negative_count"), description + ".negative_count")
    positive_valid_count = _integer(valid_positive, description + ".positive_valid_count")
    negative_valid_count = _integer(valid_negative, description + ".negative_valid_count")
    valid_fraction = _probability(row.get("valid_fraction"), description + ".valid_fraction")
    if (
        positive_count + negative_count != total
        or positive_valid_count + negative_valid_count != valid_count
        or valid_count > total
        or positive_valid_count > positive_count
        or negative_valid_count > negative_count
        or not math.isclose(valid_fraction, valid_count / total, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise EvidenceExtractionError(description + " coverage counts/fraction differ")
    return {
        "population_count": total,
        "valid_count": valid_count,
        "valid_fraction": valid_fraction,
        "positive_count": positive_count,
        "positive_valid_count": positive_valid_count,
        "negative_count": negative_count,
        "negative_valid_count": negative_valid_count,
    }


def _interval(value: object, description: str, *, delta: bool = False) -> Mapping[str, Any]:
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
        raise EvidenceExtractionError(description + " interval fields differ")
    point = _number(row["point_estimate"], description + ".point")
    ci = _list(row["percentile_95_ci"], description + ".ci")
    if len(ci) != 2:
        raise EvidenceExtractionError(description + " CI must have two endpoints")
    low = _number(ci[0], description + ".ci.low")
    high = _number(ci[1], description + ".ci.high")
    if low > high:
        raise EvidenceExtractionError(description + " CI endpoints are reversed")
    valid = _integer(row["valid_replicates"], description + ".valid_replicates", 1)
    if valid > BOOTSTRAP_REPLICATES:
        raise EvidenceExtractionError(description + " has too many bootstrap replicates")
    output = {
        "point_estimate": point,
        "percentile_95_ci": [low, high],
        "bootstrap_mean": _number(row["bootstrap_mean"], description + ".mean"),
        "bootstrap_standard_error": _number(
            row["bootstrap_standard_error"], description + ".se", minimum=0.0
        ),
        "valid_replicates": valid,
    }
    if delta:
        output["probability_delta_gt_zero"] = _probability(
            row["probability_delta_gt_zero"], description + ".probability_delta_gt_zero"
        )
    return output


def _synthetic_classification(arm: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    metrics = _mapping(arm.get("metrics"), "synthetic " + method + " metrics")
    value = (
        metrics.get("pair_classification_diagnostic")
        if method in {"pairingnet_adapted", "shreddingnet_adapted"}
        else metrics.get("main_pairwise")
    )
    view = _mapping(value, "synthetic " + method + " classification")
    if _integer(view.get("sample_count"), method + ".sample_count", 1) != 3000:
        raise EvidenceExtractionError(method + " synthetic classification coverage differs")
    if _integer(view.get("positive_count"), method + ".positive_count") != 1500 or _integer(
        view.get("negative_count"), method + ".negative_count"
    ) != 1500:
        raise EvidenceExtractionError(method + " synthetic class counts differ")
    return {
        "row": _classification_metrics(view.get("row"), method + ".row"),
        "cluster_balanced": _classification_metrics(
            view.get("cluster_balanced"), method + ".cluster_balanced"
        ),
    }


def _synthetic_pose(arm: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    metrics = _mapping(arm.get("metrics"), "synthetic " + method + " metrics")
    translation = _mapping(metrics.get("translation"), method + " translation")
    if method == "full_n512":
        median = _number(translation.get("median_l2_px"), method + ".median", minimum=0.0)
        p90 = _number(translation.get("p90_l2_px"), method + ".p90", minimum=0.0)
        valid_fraction = _probability(translation.get("valid_fraction"), method + ".valid")
        recall = {
            "at_{}px".format(tolerance): _probability(
                translation.get("success_at_{}px".format(tolerance)),
                method + ".recall_at_{}px".format(tolerance),
            )
            for tolerance in TOLERANCES
        }
        assembly_key_format = "at_{}"
    else:
        te = _mapping(translation.get("te_px_conditioned_on_valid_pose"), method + ".te")
        median = _number(te.get("median"), method + ".median", minimum=0.0)
        p90 = _number(te.get("p90"), method + ".p90", minimum=0.0)
        valid_fraction = _probability(
            translation.get("valid_prediction_fraction"), method + ".valid"
        )
        recalls = _mapping(
            translation.get("unconditional_positive_recall"), method + ".recall"
        )
        recall = {
            "at_{}px".format(tolerance): _probability(
                recalls.get("at_{}px".format(tolerance)),
                method + ".recall_at_{}px".format(tolerance),
            )
            for tolerance in TOLERANCES
        }
        assembly_key_format = "at_{}px"
    assembly_source = _mapping(metrics.get("assembly_edge"), method + ".assembly")
    by_tolerance = _mapping(assembly_source.get("by_tolerance"), method + ".assembly.by_tolerance")
    assembly = {}
    for tolerance in TOLERANCES:
        row = _mapping(
            by_tolerance.get(assembly_key_format.format(tolerance)),
            method + ".assembly",
        )
        assembly["at_{}px".format(tolerance)] = {
            name: _probability(row.get(name), method + ".assembly." + name)
            for name in ("precision", "recall", "f1")
        }
    registration = _mapping(
        metrics.get("pairingnet_style_registration"), method + ".registration"
    )
    rr_key = "registration_recall_lt4" if method == "full_n512" else "rr_lt4"
    output = {
        "eligible_positive_count": _integer(
            translation.get("eligible_positive_count"), method + ".eligible", 1
        ),
        "valid_pose_fraction": valid_fraction,
        "median_l2_px_conditioned_on_valid_pose": median,
        "p90_l2_px_conditioned_on_valid_pose": p90,
        "unconditional_positive_translation_recall": recall,
        "assembly_edge": assembly,
        "pairingnet_style_registration": {
            "rr_lt4": _probability(registration.get(rr_key), method + ".rr_lt4"),
            "mean_e_rmse": _number(
                registration.get("mean_e_rmse"), method + ".e_rmse", minimum=0.0
            ),
            "mean_symmetric_hausdorff_px": _number(
                registration.get("mean_symmetric_hausdorff_px"),
                method + ".hausdorff",
                minimum=0.0,
            ),
            "mean_normalized_translation_error": _number(
                registration.get("mean_normalized_translation_error"),
                method + ".nte",
                minimum=0.0,
            ),
            "rotation_error": "not_applicable_known_upright",
        },
        "correspondence": metrics.get("correspondence"),
    }
    if output["eligible_positive_count"] != 1500:
        raise EvidenceExtractionError(method + " synthetic pose population differs")
    return output


def extract_synthetic(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_identity(
        value,
        schema="rachel-n512-sealed-synthetic-test/1.0",
        status="complete_frozen_synthetic_test_only",
        description="synthetic receipt",
    )
    _require_formal(value, "synthetic receipt")
    protocol = _mapping(value.get("protocol"), "synthetic protocol")
    _require_methods(protocol.get("formal_method_inventory"), "synthetic")
    if protocol.get("formal_exact_six_frozen_before_test_open") is not True:
        raise EvidenceExtractionError("synthetic exact-six freeze gate differs")
    population = _mapping(value.get("test_population"), "synthetic population")
    expected = {
        "observed_count": 3000,
        "positive_count": 1500,
        "negative_count": 1500,
        "cluster_count": 130,
    }
    for key, expected_value in expected.items():
        if population.get(key) != expected_value:
            raise EvidenceExtractionError("synthetic population {} differs".format(key))
    pair_sha = population.get("pair_ids_sha256")
    if not _is_sha256(pair_sha):
        raise EvidenceExtractionError("synthetic pair-order SHA-256 is invalid")
    raw_arms = _list(value.get("arm_results"), "synthetic arm results")
    arms = {
        str(_mapping(row, "synthetic arm").get("arm")): _mapping(row, "synthetic arm")
        for row in raw_arms
    }
    if len(arms) != len(raw_arms) or set(arms) != set(METHODS):
        raise EvidenceExtractionError("synthetic arm inventory differs")
    classification = {}
    elapsed = {}
    declared_members = {}
    for method in METHODS:
        arm = arms[method]
        if method in {"pairingnet_adapted", "shreddingnet_adapted"}:
            adaptation = _mapping(arm.get("adaptation"), method + ".adaptation")
            if (
                arm.get("adaptation_claim")
                != "same_data_method_adaptation_not_exact_reproduction"
                or adaptation.get("claim")
                != "same_data_method_adaptation_not_exact_reproduction"
            ):
                raise EvidenceExtractionError(method + " adaptation disclosure differs")
        if arm.get("pair_scores_count") != 3000 or arm.get("metrics", {}).get(
            "pair_ids_sha256"
        ) != pair_sha:
            raise EvidenceExtractionError(method + " synthetic pair binding differs")
        member_sha = arm.get("pair_scores_sha256")
        if not _is_sha256(member_sha):
            raise EvidenceExtractionError(method + " pair-score SHA-256 is invalid")
        declared_members[method] = {
            "logical_path": arm.get("pair_scores"),
            "sha256": member_sha,
            "record_count": 3000,
            "bytes_recomputed_by_this_extractor": False,
        }
        classification[method] = _synthetic_classification(arm, method)
        metric_source = _mapping(arm.get("metrics"), method + ".metrics")
        elapsed_value = metric_source.get("seconds")
        if method in {"pairingnet_adapted", "shreddingnet_adapted"}:
            elapsed_value = _mapping(
                metric_source.get("single_forward_population_contract"),
                method + ".single_forward_population_contract",
            ).get("elapsed_seconds")
        elapsed[method] = _number(
            elapsed_value, method + ".seconds", minimum=0.0
        )
    return {
        "population": {
            "pair_count": 3000,
            "positive_count": 1500,
            "negative_count": 1500,
            "cluster_count": 130,
            "pair_order_sha256": pair_sha,
            "common_valid": {
                "population_count": 3000,
                "valid_count": 3000,
                "valid_fraction": 1.0,
                "positive_count": 1500,
                "positive_valid_count": 1500,
                "negative_count": 1500,
                "negative_valid_count": 1500,
            },
        },
        "classification_native_all_valid": classification,
        "pose": {
            method: _synthetic_pose(arms[method], method)
            for method in TRANSLATION_METHODS
        },
        "elapsed_seconds_by_method": elapsed,
        "declared_pair_score_members": declared_members,
    }


def _corrosion_metric_view(value: object, description: str) -> Mapping[str, Any]:
    view = _mapping(value, description)
    ranking = _mapping(view.get("primary_threshold_free_ranking"), description + ".ranking")
    thresholded = _mapping(
        view.get("secondary_frozen_validation_threshold"), description + ".thresholded"
    )
    output = {}
    for grain, source_grain in (("row", "row"), ("cluster_balanced", "cluster_balanced")):
        ranking_row = _mapping(ranking.get(source_grain), description + "." + grain)
        threshold_row = _mapping(thresholded.get(source_grain), description + "." + grain)
        output[grain] = {
            "auroc": _probability(ranking_row.get("auroc"), description + ".auroc"),
            "auprc": _probability(ranking_row.get("auprc"), description + ".auprc"),
            "f1": _probability(threshold_row.get("f1"), description + ".f1"),
            "recall": _probability(threshold_row.get("recall"), description + ".recall"),
        }
    return output


def extract_corrosion(value: Mapping[str, Any], synthetic_sha: str) -> Mapping[str, Any]:
    _require_identity(
        value,
        schema="rachel-n512-corrosion-robustness/2.0",
        status="complete_formal_exact_six_frozen_synthetic_corrosion_only",
        description="corrosion receipt",
    )
    _require_formal(value, "corrosion receipt")
    protocol = _mapping(value.get("protocol"), "corrosion protocol")
    _require_methods(protocol.get("formal_method_inventory"), "corrosion")
    if (
        protocol.get("formal_exact_six") is not True
        or protocol.get("same_corrupted_masks_supplied_to_all_six_methods") is not True
        or protocol.get("same_data_benchmark_adaptations_not_exact_reproductions") is not True
    ):
        raise EvidenceExtractionError("corrosion exact-six/adaptation contract differs")
    clean_reference = _mapping(value.get("clean_reference"), "corrosion clean reference")
    if clean_reference.get("receipt_sha256") != synthetic_sha:
        raise EvidenceExtractionError("corrosion clean reference is not bound to synthetic input")
    summary = _mapping(value.get("summary"), "corrosion summary")
    _require_methods(summary.get("method_order"), "corrosion summary")
    if summary.get("condition_order") != list(CONDITIONS) or summary.get("formal_exact_six") is not True:
        raise EvidenceExtractionError("corrosion condition order differs")
    primary = _mapping(summary.get("primary_population"), "corrosion primary population")
    if (
        primary.get("definition") != "all_six_methods_valid_at_all_seven_conditions"
        or primary.get("pair_count") != 1822
        or primary.get("positive_count") != 923
        or primary.get("negative_count") != 899
    ):
        raise EvidenceExtractionError("corrosion primary common-valid population differs")
    coverage = _probability(primary.get("coverage"), "corrosion coverage")
    if not math.isclose(coverage, 1822 / 3000, rel_tol=0.0, abs_tol=1e-12):
        raise EvidenceExtractionError("corrosion coverage fraction differs")
    conditions_source = _mapping(summary.get("conditions"), "corrosion conditions")
    if set(conditions_source) != set(CONDITIONS):
        raise EvidenceExtractionError("corrosion condition inventory differs")
    conditions = {}
    for condition in CONDITIONS:
        source = _mapping(conditions_source[condition], condition)
        methods = _mapping(source.get("methods"), condition + ".methods")
        if set(methods) != set(METHODS):
            raise EvidenceExtractionError(condition + " method inventory differs")
        rows = {}
        for method in METHODS:
            method_source = _mapping(methods[method], condition + "." + method)
            fair = _mapping(
                method_source.get("fixed_all_method_all_condition_common_valid"),
                condition + "." + method + ".fair",
            )
            method_coverage = _mapping(fair.get("coverage"), condition + ".coverage")
            if (
                method_coverage.get("record_count") != 3000
                or method_coverage.get("valid_count") != 1822
                or method_coverage.get("positive_valid_count") != 923
                or method_coverage.get("negative_valid_count") != 899
                or not math.isclose(
                    _probability(method_coverage.get("valid_fraction"), condition + ".coverage"),
                    coverage,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise EvidenceExtractionError(condition + "/" + method + " coverage differs")
            rows[method] = _corrosion_metric_view(
                fair, condition + "." + method + ".classification"
            )
        direct_source = _mapping(source.get("direct_geometry_by_method"), condition + ".direct")
        if set(direct_source) != set(TRANSLATION_METHODS):
            raise EvidenceExtractionError(condition + " direct method inventory differs")
        conditions[condition] = {
            "classification": rows,
            "pose_point_estimates": {
                method: _mapping(direct_source[method], condition + "." + method).get(
                    "fixed_all_method_all_condition_common_valid_positive_population"
                )
                for method in TRANSLATION_METHODS
            },
        }
    bootstrap = _mapping(
        value.get("paired_endpoint_pigeonhole_bootstrap"), "corrosion bootstrap"
    )
    _require_methods(bootstrap.get("method_order"), "corrosion bootstrap")
    if (
        bootstrap.get("population") != "fixed_all_method_all_condition_common_valid"
        or bootstrap.get("replicates_requested") != BOOTSTRAP_REPLICATES
        or bootstrap.get("shared_draws_across_all_methods_conditions_and_geometry_metrics") is not True
    ):
        raise EvidenceExtractionError("corrosion bootstrap protocol differs")
    valid = _integer(bootstrap.get("valid_replicates"), "corrosion valid replicates", 1)
    skipped = _integer(
        bootstrap.get("skipped_single_class_replicates"), "corrosion skipped replicates"
    )
    if valid + skipped != BOOTSTRAP_REPLICATES:
        raise EvidenceExtractionError("corrosion bootstrap replicate accounting differs")
    ranking = _mapping(bootstrap.get("ranking"), "corrosion bootstrap ranking")
    by_condition = _mapping(ranking.get("by_condition_method"), "corrosion bootstrap by condition")
    delta_source = _mapping(
        ranking.get("full_n512_minus_comparator_within_condition"),
        "corrosion bootstrap deltas",
    )
    classification_bootstrap = {}
    for condition in CONDITIONS:
        method_rows = _mapping(by_condition.get(condition), condition + ".bootstrap")
        delta_rows = _mapping(delta_source.get(condition), condition + ".deltas")
        classification_bootstrap[condition] = {
            "methods": {
                method: {
                    metric: _interval(
                        _mapping(method_rows.get(method), method).get(metric),
                        "{}.{}.{}".format(condition, method, metric),
                    )
                    for metric in ("auroc", "auprc")
                }
                for method in METHODS
            },
            "full_n512_minus_comparator": {
                method: {
                    metric: _interval(
                        _mapping(delta_rows.get(method), method).get(metric),
                        "{}.delta.{}.{}".format(condition, method, metric),
                        delta=True,
                    )
                    for metric in ("auroc", "auprc")
                }
                for method in METHODS
                if method != "full_n512"
            },
        }
        for method in METHODS:
            for metric in ("auroc", "auprc"):
                point_value = conditions[condition]["classification"][method]["row"][metric]
                interval_value = classification_bootstrap[condition]["methods"][method][metric][
                    "point_estimate"
                ]
                if not math.isclose(point_value, interval_value, rel_tol=0.0, abs_tol=1e-12):
                    raise EvidenceExtractionError(
                        "corrosion bootstrap/classification point estimate differs"
                    )
    direct_bootstrap_source = _mapping(
        bootstrap.get("direct_geometry_by_method"), "corrosion direct bootstrap"
    )
    if set(direct_bootstrap_source) != set(TRANSLATION_METHODS):
        raise EvidenceExtractionError("corrosion direct-bootstrap methods differ")
    direct_bootstrap = {}
    for method in TRANSLATION_METHODS:
        by_method = _mapping(direct_bootstrap_source[method], method + ".direct bootstrap")
        by_method_condition = _mapping(by_method.get("by_condition"), method + ".by_condition")
        if set(by_method_condition) != set(CONDITIONS):
            raise EvidenceExtractionError(method + " direct-bootstrap conditions differ")
        direct_bootstrap[method] = {
            condition: {
                metric: _interval(interval, "{}.{}.{}".format(method, condition, metric))
                for metric, interval in _mapping(
                    by_method_condition[condition], method + "." + condition
                ).items()
            }
            for condition in CONDITIONS
        }
    return {
        "population": {
            "definition": primary["definition"],
            "population_count": 3000,
            "valid_count": 1822,
            "valid_fraction": coverage,
            "positive_valid_count": 923,
            "negative_valid_count": 899,
            "pair_ids_sha256": primary.get("pair_ids_sha256"),
        },
        "conditions": conditions,
        "classification_bootstrap": {
            "metrics_with_intervals": ["row_auroc", "row_auprc"],
            "cluster_balanced_f1_recall_intervals_available": False,
            "replicates_requested": BOOTSTRAP_REPLICATES,
            "valid_replicates": valid,
            "skipped_single_class_replicates": skipped,
            "by_condition": classification_bootstrap,
        },
        "pose_bootstrap": direct_bootstrap,
    }


def _real_method_view(value: object, description: str) -> Mapping[str, Any]:
    view = _mapping(value, description)
    ranking = _mapping(view.get("ranking_primary_threshold_free"), description + ".ranking")
    thresholded = _mapping(
        view.get("frozen_validation_threshold_secondary"), description + ".thresholded"
    )
    output = {}
    for output_grain, source_grain in (("row", "row"), ("cluster_balanced", "case_cluster_balanced")):
        rank = _mapping(ranking.get(source_grain), description + "." + source_grain)
        cut = _mapping(thresholded.get(source_grain), description + "." + source_grain)
        output[output_grain] = {
            "auroc": _probability(rank.get("auroc"), description + ".auroc"),
            "auprc": _probability(rank.get("auprc"), description + ".auprc"),
            "f1": _probability(cut.get("f1"), description + ".f1"),
            "recall": _probability(cut.get("recall"), description + ".recall"),
        }
    output["coverage"] = view.get("coverage")
    return output


def _real_population(value: object, name: str, total: int) -> Mapping[str, Any]:
    source = _mapping(value, name)
    methods = _mapping(source.get("methods"), name + ".methods")
    if set(methods) != set(METHODS):
        raise EvidenceExtractionError(name + " method inventory differs")
    common = _normalize_coverage(source.get("all_method_common_valid"), total, name + ".common")
    output_methods = {}
    for method in METHODS:
        method_source = _mapping(methods[method], name + "." + method)
        fair = _real_method_view(
            method_source.get("all_method_common_valid"), name + "." + method + ".fair"
        )
        fair_coverage = _normalize_coverage(fair.get("coverage"), total, name + "." + method)
        if fair_coverage != common:
            raise EvidenceExtractionError(name + "/" + method + " common-valid coverage differs")
        native_key = (
            "pair_classification_diagnostic"
            if method in {"pairingnet_adapted", "shreddingnet_adapted"}
            else "native"
        )
        native = _real_method_view(method_source.get(native_key), name + "." + method + ".native")
        output_methods[method] = {
            "fair_all_method_common_valid": {
                "row": fair["row"],
                "cluster_balanced": fair["cluster_balanced"],
            },
            "native": {
                "row": native["row"],
                "cluster_balanced": native["cluster_balanced"],
                "coverage": _normalize_coverage(
                    native.get("coverage"), total, name + "." + method + ".native.coverage"
                ),
            },
        }
    return {"common_valid": common, "methods": output_methods}


def extract_real(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_identity(
        value,
        schema="rachel-n512-real-external/1.0",
        status="complete_strict_and_balanced_single_forward_population",
        description="real pair-only result",
    )
    forward = _mapping(value.get("forward_contract"), "real forward contract")
    _require_methods(forward.get("methods"), "real pair-only")
    root_protocol = _mapping(value.get("protocol"), "real pair-only protocol")
    _require_methods(root_protocol.get("formal_method_inventory"), "real pair-only protocol")
    if (
        root_protocol.get("formal_evaluation") is not True
        or root_protocol.get("compatibility_mode") is not False
        or root_protocol.get("formal_exact_six_frozen_before_real_open") is not True
        or root_protocol.get("same_data_benchmark_exact_reproduction_claimed") is not False
    ):
        raise EvidenceExtractionError("real pair-only formal/adaptation protocol differs")
    if (
        forward.get("forward_pair_count_per_arm") != 1016
        or forward.get("strict_547_derived_from_exact_prediction_prefix") is not True
        or forward.get("strict_pairs_forwarded_twice") is not False
    ):
        raise EvidenceExtractionError("real single-forward contract differs")
    strict_source = _mapping(value.get("strict_547"), "strict547")
    balanced_source = _mapping(value.get("balanced_1016"), "balanced1016")
    _require_identity(
        strict_source,
        schema="rachel-n512-real-external/1.0",
        status="complete_strict_547_target_blind_external_test",
        description="strict547",
    )
    _require_identity(
        balanced_source,
        schema="rachel-n512-real-external/1.0",
        status="complete_balanced_1016_target_blind_external_test",
        description="balanced1016",
    )
    strict_dataset = _mapping(strict_source.get("dataset"), "strict547 dataset")
    balanced_dataset = _mapping(balanced_source.get("dataset"), "balanced1016 dataset")
    for key, expected in (("pair_count", 547), ("positive_count", 508), ("negative_count", 39)):
        if strict_dataset.get(key) != expected:
            raise EvidenceExtractionError("strict547 {} differs".format(key))
    for key, expected in (("pair_count", 1016), ("positive_count", 508), ("negative_count", 508)):
        if balanced_dataset.get(key) != expected:
            raise EvidenceExtractionError("balanced1016 {} differs".format(key))
    if (
        balanced_dataset.get("strict_prefix_count") != 547
        or balanced_dataset.get("constructed_count") != 469
        or balanced_dataset.get("strict_prefix_preserved_exactly") is not True
    ):
        raise EvidenceExtractionError("balanced1016 constructed/prefix contract differs")
    protocol = _mapping(balanced_source.get("protocol"), "balanced1016 protocol")
    _require_methods(protocol.get("formal_method_inventory"), "balanced1016 protocol")
    if (
        protocol.get("same_data_benchmark_exact_reproduction_claimed") is not False
        or protocol.get("formal_evaluation") is not True
        or protocol.get("compatibility_mode") is not False
    ):
        raise EvidenceExtractionError("real benchmark adaptation disclosure differs")
    return {
        "strict547_descriptive": {
            "role": "descriptive_only_few_GT_negatives",
            **_real_population(strict_source, "strict547", 547),
        },
        "balanced1016_inferential": {
            "role": "strict_GT_547_plus_469_constructed_not_GT_negative_distractors",
            **_real_population(balanced_source, "balanced1016", 1016),
        },
    }


def _extract_ranking_bootstrap(
    value: Mapping[str, Any], source_sha: str, source_kind: str, total: int
) -> Mapping[str, Any]:
    _require_identity(
        value,
        schema="rachel-n512-paired-endpoint-bootstrap/1.1",
        status="complete_formal_exact_six",
        description=source_kind + " bootstrap",
    )
    _require_formal(value, source_kind + " bootstrap")
    _require_methods(value.get("methods"), source_kind + " bootstrap")
    source = _mapping(value.get("input"), source_kind + " bootstrap input")
    if source.get("source_kind") != source_kind or source.get("source_sha256") != source_sha:
        raise EvidenceExtractionError(source_kind + " bootstrap source hash/kind differs")
    protocol = _mapping(value.get("protocol"), source_kind + " bootstrap protocol")
    if protocol.get("replicates") != BOOTSTRAP_REPLICATES or protocol.get("seed") != BOOTSTRAP_SEED:
        raise EvidenceExtractionError(source_kind + " bootstrap repetitions/seed differs")
    coverage = _mapping(value.get("coverage"), source_kind + " bootstrap coverage")
    common = _normalize_coverage(
        coverage.get("all_method_common_valid"), total, source_kind + " bootstrap common"
    )
    point = _mapping(value.get("point_metrics"), source_kind + " point metrics")
    if point.get("row_count") != common["valid_count"]:
        raise EvidenceExtractionError(source_kind + " bootstrap common count differs")
    methods = _mapping(point.get("methods"), source_kind + " point methods")
    if set(methods) != set(METHODS):
        raise EvidenceExtractionError(source_kind + " bootstrap point methods differ")
    paired = _mapping(
        value.get("paired_endpoint_pigeonhole_bootstrap"), source_kind + " paired bootstrap"
    )
    if paired.get("replicates_requested") != BOOTSTRAP_REPLICATES:
        raise EvidenceExtractionError(source_kind + " paired bootstrap count differs")
    valid = _integer(paired.get("valid_replicates"), source_kind + " valid replicates", 1)
    skipped = _integer(
        paired.get("skipped_single_class_replicates"), source_kind + " skipped replicates"
    )
    if valid + skipped != BOOTSTRAP_REPLICATES:
        raise EvidenceExtractionError(source_kind + " bootstrap accounting differs")
    metrics = _mapping(paired.get("metrics"), source_kind + " bootstrap metrics")
    if set(metrics) != {"auroc", "auprc"}:
        raise EvidenceExtractionError(source_kind + " bootstrap metric inventory differs")
    output_methods = {method: {} for method in METHODS}
    output_deltas = {method: {} for method in METHODS if method != "full_n512"}
    for metric in ("auroc", "auprc"):
        metric_source = _mapping(metrics.get(metric), source_kind + "." + metric)
        method_source = _mapping(metric_source.get("methods"), source_kind + ".methods")
        delta_source = _mapping(metric_source.get("paired_deltas"), source_kind + ".deltas")
        if set(method_source) != set(METHODS):
            raise EvidenceExtractionError(source_kind + " bootstrap method inventory differs")
        if set(delta_source) != {
            "full_n512_minus_" + method
            for method in METHODS
            if method != "full_n512"
        }:
            raise EvidenceExtractionError(source_kind + " bootstrap delta inventory differs")
        for method in METHODS:
            interval = _interval(method_source[method], source_kind + "." + method + "." + metric)
            point_value = _probability(
                _mapping(methods[method], source_kind + ".point").get(metric),
                source_kind + ".point." + metric,
            )
            if not math.isclose(interval["point_estimate"], point_value, rel_tol=0.0, abs_tol=1e-12):
                raise EvidenceExtractionError(source_kind + " bootstrap point estimate differs")
            output_methods[method][metric] = interval
        for method in output_deltas:
            name = "full_n512_minus_" + method
            output_deltas[method][metric] = _interval(
                delta_source.get(name), source_kind + "." + name + "." + metric, delta=True
            )
    return {
        "scope": "all_method_common_valid_equal_row_ranking",
        "coverage": common,
        "methods": output_methods,
        "full_n512_minus_comparator": output_deltas,
        "replicates_requested": BOOTSTRAP_REPLICATES,
        "valid_replicates": valid,
        "skipped_single_class_replicates": skipped,
        "cluster_balanced_or_f1_recall_intervals_available": False,
    }


def extract_translation(value: Mapping[str, Any], real_sha: str) -> Mapping[str, Any]:
    _require_identity(
        value,
        schema="rachel-n512-real-translation-gt-posteval/1.0",
        status="complete_postprediction_real_positive_translation_gt",
        description="real translation-GT",
    )
    _require_formal(value, "real translation-GT")
    _verify_content_sha(value, "real translation-GT")
    source = _mapping(value.get("source_pair_only_evaluation"), "translation source")
    if source.get("file_sha256_frozen_before_gt_open") != real_sha:
        raise EvidenceExtractionError("translation-GT is not bound to real pair-only input")
    _require_methods(source.get("exact_method_order"), "translation-GT")
    population = _mapping(value.get("population"), "translation population")
    if population.get("strict_pair_count") != 547 or population.get("positive_pair_count") != 508:
        raise EvidenceExtractionError("translation-GT population differs")
    methods_source = _mapping(value.get("method_metrics"), "translation method metrics")
    if set(methods_source) != set(METHODS):
        raise EvidenceExtractionError("translation-GT method inventory differs")
    methods = {}
    for method in METHODS:
        row = _mapping(methods_source[method], method + " translation metrics")
        if method not in TRANSLATION_METHODS:
            if row.get("status") != "not_applicable" or row.get("translation_metrics") is not None:
                raise EvidenceExtractionError(method + " translation should be not applicable")
            methods[method] = {"status": "not_applicable"}
            continue
        if row.get("status") != "evaluated_positive_translation_gt" or row.get(
            "eligible_positive_count"
        ) != 508:
            raise EvidenceExtractionError(method + " translation evaluation differs")
        case_bootstrap = _mapping(row.get("case_bootstrap"), method + ".case_bootstrap")
        if (
            case_bootstrap.get("schema") != "case_cluster_percentile_bootstrap/1.0"
            or case_bootstrap.get("repetitions") != BOOTSTRAP_REPLICATES
            or case_bootstrap.get("sampling_unit") != "authoritative_real_case_uid"
        ):
            raise EvidenceExtractionError(method + " translation bootstrap differs")
        metrics = {}
        raw_bootstrap_metrics = _mapping(
            case_bootstrap.get("metrics"), method + ".bootstrap.metrics"
        )
        expected_metric_names = {
            "valid_translation_prediction_fraction",
            "median_l2_px",
            "p90_l2_px",
            *("recall_at_{}px".format(tolerance) for tolerance in TOLERANCES),
            *(
                "joint_frozen_threshold_and_translation_recall_at_{}px".format(
                    tolerance
                )
                for tolerance in TOLERANCES
            ),
        }
        if set(raw_bootstrap_metrics) != expected_metric_names:
            raise EvidenceExtractionError(method + " translation metric inventory differs")
        for name, interval_value in raw_bootstrap_metrics.items():
            interval_source = _mapping(interval_value, method + "." + name)
            if set(interval_source) != {
                "estimate",
                "ci95_low",
                "ci95_high",
                "valid_bootstrap_replicates",
            }:
                raise EvidenceExtractionError(method + "/" + name + " interval fields differ")
            estimate = _number(interval_source.get("estimate"), method + "." + name)
            low = _number(interval_source.get("ci95_low"), method + "." + name + ".low")
            high = _number(interval_source.get("ci95_high"), method + "." + name + ".high")
            if low > high:
                raise EvidenceExtractionError(method + "/" + name + " CI is reversed")
            metrics[name] = {
                "point_estimate": estimate,
                "percentile_95_ci": [low, high],
                "valid_replicates": _integer(
                    interval_source.get("valid_bootstrap_replicates"),
                    method + "." + name + ".valid",
                    1,
                ),
            }
        point = _mapping(row.get("point_estimates"), method + ".point_estimates")
        if set(point) != set(metrics) or any(
            not math.isclose(
                _number(point[name], method + ".point." + name),
                metrics[name]["point_estimate"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name in metrics
        ):
            raise EvidenceExtractionError(method + " translation point/bootstrap binding differs")
        methods[method] = {
            "status": row["status"],
            "case_bootstrap": metrics,
            "assembly_edge_strict547": row.get("assembly_edge_strict_547"),
            "assembly_edge_balanced1016_selected_list_diagnostic": row.get(
                "assembly_edge_balanced_1016_selected_list_diagnostic"
            ),
            "pairingnet_style_registration": row.get("pairingnet_style_registration"),
            "correspondence": row.get("correspondence"),
        }
    return {
        "population": {
            "strict_pair_count": 547,
            "positive_pair_count": 508,
            "negative_pairs_have_translation_gt": False,
        },
        "methods": methods,
    }


def extract_resources(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_identity(
        value,
        schema=RESOURCE_SCHEMA_VERSION,
        status="complete_from_existing_evidence",
        description="resource evidence",
    )
    methods = _mapping(value.get("methods"), "resource methods")
    if set(methods) != set(TRANSLATION_METHODS):
        raise EvidenceExtractionError("resource method inventory differs")
    source_hash_count = 0

    def validate_source_hashes(item: object, location: str) -> None:
        nonlocal source_hash_count
        if isinstance(item, Mapping):
            if "sha256" in item:
                if not _is_sha256(item.get("sha256")):
                    raise EvidenceExtractionError(location + " has an invalid source SHA-256")
                source_hash_count += 1
            for key, child in item.items():
                validate_source_hashes(child, location + "." + str(key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                validate_source_hashes(child, "{}[{}]".format(location, index))

    validate_source_hashes(value, "resources")
    if source_hash_count == 0:
        raise EvidenceExtractionError("resource evidence contains no source SHA-256")
    for method in ("pairingnet_adapted", "shreddingnet_adapted"):
        label = _mapping(methods[method], method + " resources").get("method_label")
        if not isinstance(label, str) or "adapt" not in label.casefold():
            raise EvidenceExtractionError(method + " resource label omits adaptation")

    hardware = _mapping(value.get("hardware"), "resource hardware")
    capacity = _integer(
        hardware.get("memory_total_bytes"), "resource GPU capacity bytes", 1
    )
    capacity_mib = _integer(
        hardware.get("memory_total_mib"), "resource GPU capacity MiB", 1
    )
    if capacity != capacity_mib * 1024 * 1024:
        raise EvidenceExtractionError("resource GPU capacity units disagree")
    device_name = hardware.get("device_name")
    if not isinstance(device_name, str) or not device_name.strip():
        raise EvidenceExtractionError("resource GPU device name is invalid")

    reserved = {
        "full_n512": _integer(
            _mapping(methods["full_n512"].get("gpu_memory_training_step"), "full GPU").get(
                "peak_reserved_bytes"
            ),
            "full peak reserved bytes",
            1,
        ),
        "pairingnet_adapted": _integer(
            _mapping(
                methods["pairingnet_adapted"].get("gpu_memory_training_step"),
                "PairingNet GPU",
            ).get("peak_reserved_bytes"),
            "PairingNet peak reserved bytes",
            1,
        ),
        "shreddingnet_adapted": _integer(
            _mapping(
                methods["shreddingnet_adapted"].get("gpu_memory_training_step"),
                "ShreddingNet GPU",
            ).get("maximum_peak_reserved_bytes"),
            "ShreddingNet maximum peak reserved bytes",
            1,
        ),
    }
    concurrency = _mapping(value.get("concurrency_assessment"), "resource concurrency")
    if (
        concurrency.get("measurement_type")
        != "derived_from_smoke_peak_reserved_bytes"
        or concurrency.get("capacity_bytes") != capacity
    ):
        raise EvidenceExtractionError("resource concurrency capacity/protocol differs")
    expected_scenarios = {
        "pairingnet_plus_shreddingnet": (
            reserved["pairingnet_adapted"] + reserved["shreddingnet_adapted"],
            "suitable_by_gpu_memory",
        ),
        "full_n512_plus_pairingnet": (
            reserved["full_n512"] + reserved["pairingnet_adapted"],
            "likely_suitable_by_smoke_memory_only",
        ),
        "full_n512_plus_shreddingnet": (
            reserved["full_n512"] + reserved["shreddingnet_adapted"],
            "possible_but_lower_headroom_by_smoke_memory_only",
        ),
        "all_three": (sum(reserved.values()), "not_suitable"),
    }
    concurrency_output = {
        "measurement_type": concurrency["measurement_type"],
        "capacity_bytes": capacity,
    }
    for scenario, (expected_bytes, expected_verdict) in expected_scenarios.items():
        row = _mapping(concurrency.get(scenario), "resource concurrency " + scenario)
        if (
            row.get("combined_peak_reserved_bytes") != expected_bytes
            or row.get("verdict") != expected_verdict
        ):
            raise EvidenceExtractionError(scenario + " concurrency arithmetic/verdict differs")
        fraction = _number(
            row.get("fraction_of_device_capacity"), scenario + " capacity fraction"
        )
        if not math.isclose(fraction, expected_bytes / capacity, rel_tol=0.0, abs_tol=1e-12):
            raise EvidenceExtractionError(scenario + " capacity fraction differs")
        if expected_bytes <= capacity:
            if row.get("headroom_bytes") != capacity - expected_bytes:
                raise EvidenceExtractionError(scenario + " headroom differs")
        elif row.get("capacity_deficit_bytes") != expected_bytes - capacity:
            raise EvidenceExtractionError(scenario + " capacity deficit differs")
        concurrency_output[scenario] = dict(row)

    limitations = _list(value.get("limitations"), "resource limitations")
    if not limitations or any(
        not isinstance(item, str) or not item.strip() for item in limitations
    ):
        raise EvidenceExtractionError("resource limitations are invalid")
    # Preserve method profiler payloads plus the arithmetically verified
    # hardware/concurrency evidence. The outer resource file hash remains the
    # byte-level authority for fields that retain profiler-specific structure.
    return {
        "hardware": {
            "device_name": device_name,
            "memory_total_bytes": capacity,
            "memory_total_mib": capacity_mib,
        },
        "methods": {method: methods[method] for method in TRANSLATION_METHODS},
        "concurrency_assessment": concurrency_output,
        "limitations": list(limitations),
        "declared_source_sha256_count": source_hash_count,
    }


def build_summary(
    *,
    synthetic: Mapping[str, Any],
    synthetic_sha: str,
    corrosion: Mapping[str, Any],
    corrosion_sha: str,
    real: Mapping[str, Any],
    real_sha: str,
    translation: Mapping[str, Any],
    translation_sha: str,
    synthetic_bootstrap: Mapping[str, Any],
    synthetic_bootstrap_sha: str,
    real_bootstrap: Mapping[str, Any],
    real_bootstrap_sha: str,
    resources: Optional[Mapping[str, Any]] = None,
    resources_sha: Optional[str] = None,
) -> Mapping[str, Any]:
    synthetic_out = extract_synthetic(synthetic)
    corrosion_out = extract_corrosion(corrosion, synthetic_sha)
    real_out = extract_real(real)
    translation_out = extract_translation(translation, real_sha)
    synthetic_stats = _extract_ranking_bootstrap(
        synthetic_bootstrap, synthetic_sha, "sealed_synthetic", 3000
    )
    real_stats = _extract_ranking_bootstrap(
        real_bootstrap, real_sha, "real_balanced1016", 1016
    )
    real_common = real_out["balanced1016_inferential"]["common_valid"]
    if real_stats["coverage"] != real_common:
        raise EvidenceExtractionError("real bootstrap/raw common-valid population differs")
    synthetic_common = synthetic_out["population"]["common_valid"]
    if synthetic_stats["coverage"] != synthetic_common:
        raise EvidenceExtractionError("synthetic bootstrap/raw common-valid population differs")
    sources = {
        "synthetic_receipt": synthetic_sha,
        "corrosion_receipt": corrosion_sha,
        "real_pair_only": real_sha,
        "real_translation_gt": translation_sha,
        "synthetic_bootstrap": synthetic_bootstrap_sha,
        "real_bootstrap": real_bootstrap_sha,
    }
    resource_output = None
    if resources is not None:
        if resources_sha is None:
            raise EvidenceExtractionError("resource SHA-256 is missing")
        resource_output = extract_resources(resources)
        sources["resources"] = resources_sha
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_verified_exact_six_evidence",
        "method_order": list(METHODS),
        "method_display_names": {
            "coarse_only": "Coarse-only Siamese",
            "full_n512": "Sliding-window + Sinkhorn (Full N=512)",
            "matched_mm_converged": "Matched-MM Siamese (converged)",
            "matched_mm_same_exposure_epoch5": "Matched-MM Siamese (epoch 5)",
            "pairingnet_adapted": "PairingNet-adapted",
            "shreddingnet_adapted": "ShreddingNet-adapted",
        },
        "sources_file_sha256": sources,
        "classification": {
            "synthetic_native_all_valid": synthetic_out["classification_native_all_valid"],
            "corrosion_fixed_all_method_all_condition_common_valid": {
                condition: corrosion_out["conditions"][condition]["classification"]
                for condition in CONDITIONS
            },
            "real_strict547_descriptive": real_out["strict547_descriptive"],
            "real_balanced1016_inferential": real_out["balanced1016_inferential"],
        },
        "pose": {
            "synthetic": synthetic_out["pose"],
            "corrosion_fixed_common_valid": {
                "point_estimates": {
                    condition: corrosion_out["conditions"][condition]["pose_point_estimates"]
                    for condition in CONDITIONS
                },
                "bootstrap": corrosion_out["pose_bootstrap"],
            },
            "real_strict547_positive_translation_gt": translation_out,
        },
        "coverage": {
            "synthetic": synthetic_out["population"]["common_valid"],
            "corrosion_all_conditions": corrosion_out["population"],
            "real_strict547": real_out["strict547_descriptive"]["common_valid"],
            "real_balanced1016": real_out["balanced1016_inferential"]["common_valid"],
        },
        "bootstrap": {
            "synthetic_pair_classification": synthetic_stats,
            "corrosion": corrosion_out["classification_bootstrap"],
            "real_pair_classification": real_stats,
            "availability_note": (
                "paired classification bootstrap covers equal-row AUROC/AUPRC; "
                "cluster-balanced AUROC/AUPRC and frozen-threshold F1/Recall remain point estimates"
            ),
        },
        "resources": resource_output,
        "runtime_seconds": {
            "synthetic_total": _number(synthetic.get("seconds"), "synthetic seconds", minimum=0.0),
            "synthetic_by_method": synthetic_out["elapsed_seconds_by_method"],
            "corrosion_total": _number(corrosion.get("seconds"), "corrosion seconds", minimum=0.0),
        },
        "integrity": {
            "all_six_input_file_sha256_recomputed": True,
            "translation_content_sha256_verified": True,
            "cross_artifact_source_hashes_verified": True,
            "common_valid_population_counts_cross_checked": True,
            "raw_pair_score_jsonl_opened": False,
            "raw_pair_score_member_hashes_recomputed": False,
            "declared_synthetic_pair_score_members": synthetic_out[
                "declared_pair_score_members"
            ],
        },
        "claim_limits": {
            "scope": "pairwise_match_probability_and_upright_relative_2d_translation",
            "global_multi_fragment_assembly_demonstrated": False,
            "rotation_estimated_or_supervised": False,
            "pairingnet_reproduction_claimed": False,
            "shreddingnet_reproduction_claimed": False,
            "same_data_benchmarks_are_adaptations_not_exact_reproductions": True,
            "real_balanced_constructed_negatives_are_GT_negatives": False,
        },
    }
    summary["content_sha256"] = _sha256_bytes(_canonical_bytes(summary))
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    display = _mapping(summary.get("method_display_names"), "display names")
    lines = [
        "# Rachel exact-six Pairwise benchmark evidence",
        "",
        "PairingNet-adapted and ShreddingNet-adapted are same-data task adaptations, not byte-exact paper reproductions.",
        "",
        "## Synthetic classification (3,000 pairs; all methods valid)",
        "",
        "| Method | Row AUROC | Row AUPRC | Row F1 | Row Recall | Cluster AUROC | Cluster AUPRC | Cluster F1 | Cluster Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    synthetic = summary["classification"]["synthetic_native_all_valid"]
    for method in METHODS:
        row = synthetic[method]["row"]
        cluster = synthetic[method]["cluster_balanced"]
        lines.append(
            "| {} | {} |".format(
                display[method],
                " | ".join(
                    "{:.4f}".format(value)
                    for value in (
                        row["auroc"],
                        row["auprc"],
                        row["f1"],
                        row["recall"],
                        cluster["auroc"],
                        cluster["auprc"],
                        cluster["f1"],
                        cluster["recall"],
                    )
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Corrosion classification on the fixed all-method/all-condition common-valid population",
            "",
            "Common-valid coverage: {}/{} ({:.2%}); {} positive and {} negative pairs.".format(
                summary["coverage"]["corrosion_all_conditions"]["valid_count"],
                summary["coverage"]["corrosion_all_conditions"]["population_count"],
                summary["coverage"]["corrosion_all_conditions"]["valid_fraction"],
                summary["coverage"]["corrosion_all_conditions"]["positive_valid_count"],
                summary["coverage"]["corrosion_all_conditions"]["negative_valid_count"],
            ),
            "",
            "| Condition | Method | Row AUROC | Row AUPRC | Row F1 | Row Recall | Cluster AUROC | Cluster AUPRC | Cluster F1 | Cluster Recall |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    corrosion = summary["classification"][
        "corrosion_fixed_all_method_all_condition_common_valid"
    ]
    for condition in CONDITIONS:
        for method in METHODS:
            row = corrosion[condition][method]["row"]
            cluster = corrosion[condition][method]["cluster_balanced"]
            lines.append(
                "| {} | {} | {} |".format(
                    condition,
                    display[method],
                    " | ".join(
                        "{:.4f}".format(value)
                        for value in (
                            row["auroc"],
                            row["auprc"],
                            row["f1"],
                            row["recall"],
                            cluster["auroc"],
                            cluster["auprc"],
                            cluster["f1"],
                            cluster["recall"],
                        )
                    ),
                )
            )
    lines.extend(
        [
            "",
            "## Real pair classification",
            "",
            "Balanced-1016 is the inferential selected-list population; strict-547 is descriptive because it contains only 39 GT negatives. Constructed distractors are not GT negatives.",
            "",
            "| Population | Method | Row AUROC | Row AUPRC | Row F1 | Row Recall | Cluster AUROC | Cluster AUPRC | Cluster F1 | Cluster Recall |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for population_key, label in (
        ("real_strict547_descriptive", "strict547"),
        ("real_balanced1016_inferential", "balanced1016"),
    ):
        population = summary["classification"][population_key]
        for method in METHODS:
            fair = population["methods"][method]["fair_all_method_common_valid"]
            row = fair["row"]
            cluster = fair["cluster_balanced"]
            lines.append(
                "| {} | {} | {} |".format(
                    label,
                    display[method],
                    " | ".join(
                        "{:.4f}".format(value)
                        for value in (
                            row["auroc"],
                            row["auprc"],
                            row["f1"],
                            row["recall"],
                            cluster["auroc"],
                            cluster["auprc"],
                            cluster["f1"],
                            cluster["recall"],
                        )
                    ),
                )
            )
    lines.extend(
        [
            "",
            "## Evidence limits",
            "",
            "- Bootstrap classification intervals cover equal-row AUROC/AUPRC only; no cluster-balanced or F1/Recall interval is inferred.",
            "- Rotation is conditioned as known upright and is not estimated.",
            "- This evidence covers pairwise compatibility and relative translation, not global multi-fragment assembly.",
            "",
            "Summary content SHA-256: `{}`".format(summary["content_sha256"]),
            "",
        ]
    )
    return "\n".join(lines)


def _write_new(path_value: Path, payload: bytes) -> None:
    path = Path(path_value).expanduser()
    if path.exists() or path.is_symlink():
        raise EvidenceExtractionError("refusing to overwrite output: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".tmp-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise EvidenceExtractionError("refusing to overwrite output") from error
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--synthetic-receipt", type=Path, required=True)
    parser.add_argument("--corrosion-receipt", type=Path, required=True)
    parser.add_argument("--real-pair-json", type=Path, required=True)
    parser.add_argument("--translation-json", type=Path, required=True)
    parser.add_argument("--synthetic-bootstrap", type=Path, required=True)
    parser.add_argument("--real-bootstrap", type=Path, required=True)
    parser.add_argument("--resources-json", type=Path)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    synthetic, synthetic_sha = _read_json(arguments.synthetic_receipt, "synthetic receipt")
    corrosion, corrosion_sha = _read_json(arguments.corrosion_receipt, "corrosion receipt")
    real, real_sha = _read_json(arguments.real_pair_json, "real pair-only result")
    translation, translation_sha = _read_json(arguments.translation_json, "real translation-GT")
    synthetic_bootstrap, synthetic_bootstrap_sha = _read_json(
        arguments.synthetic_bootstrap, "synthetic bootstrap"
    )
    real_bootstrap, real_bootstrap_sha = _read_json(
        arguments.real_bootstrap, "real bootstrap"
    )
    resources = None
    resources_sha = None
    if arguments.resources_json is not None:
        resources, resources_sha = _read_json(arguments.resources_json, "resource evidence")
    summary = build_summary(
        synthetic=synthetic,
        synthetic_sha=synthetic_sha,
        corrosion=corrosion,
        corrosion_sha=corrosion_sha,
        real=real,
        real_sha=real_sha,
        translation=translation,
        translation_sha=translation_sha,
        synthetic_bootstrap=synthetic_bootstrap,
        synthetic_bootstrap_sha=synthetic_bootstrap_sha,
        real_bootstrap=real_bootstrap,
        real_bootstrap_sha=real_bootstrap_sha,
        resources=resources,
        resources_sha=resources_sha,
    )
    _write_new(arguments.summary_output, _canonical_bytes(summary) + b"\n")
    if arguments.markdown_output is not None:
        _write_new(arguments.markdown_output, render_markdown(summary).encode("utf-8"))
    print(str(arguments.summary_output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
