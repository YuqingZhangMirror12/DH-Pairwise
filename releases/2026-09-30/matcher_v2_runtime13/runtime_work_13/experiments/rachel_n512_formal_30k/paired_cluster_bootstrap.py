#!/usr/bin/env python3
"""Paired endpoint-unit bootstrap inference for frozen Rachel evaluations.

The script consumes either the sealed synthetic ``test_receipt.json`` (and its
SHA-bound per-arm JSONL files) or one target-blind real-evaluation JSON.  It
freezes a single all-method common-valid population, recomputes equal-row
AUROC/AUPRC, and applies one shared pigeonhole-bootstrap draw over the source
units/cases incident to every pair.  This preserves dependencies created when
one manuscript lineage or real case occurs in several pair rows.  No
threshold, checkpoint, or model is selected here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "rachel-n512-paired-endpoint-bootstrap/1.1"
SEALED_SCHEMA_VERSION = "rachel-n512-sealed-synthetic-test/1.0"
SEALED_PAIR_SCHEMA_VERSION = "rachel-n512-sealed-test-pair/1.0"
REAL_SCHEMA_VERSION = "rachel-n512-real-external/1.0"
FORMAL_SEALED_STATUS = "complete_frozen_synthetic_test_only"
COMPATIBILITY_SEALED_STATUS = (
    "complete_non_formal_compatibility_synthetic_evaluation"
)
FORMAL_REAL_COMBINED_STATUS = (
    "complete_strict_and_balanced_single_forward_population"
)
COMPATIBILITY_REAL_COMBINED_STATUS = (
    "complete_non_formal_compatibility_strict_and_balanced_single_forward_population"
)
COMPATIBILITY_REAL_STRICT_STATUS = (
    "complete_non_formal_compatibility_strict_target_blind_external_test"
)
COMPATIBILITY_REAL_BALANCED_STATUS = (
    "complete_non_formal_compatibility_balanced_target_blind_external_test"
)
REQUIRED_METHODS = ("coarse_only", "full_n512")
SAME_DATA_BENCHMARK_METHODS = ("pairingnet_adapted", "shreddingnet_adapted")
DEFAULT_REPLICATES = 20_000
DEFAULT_SEED = 20_260_901
EXPECTED_SYNTHETIC_COUNT = 3_000
EXPECTED_SYNTHETIC_POSITIVE = 1_500
EXPECTED_SYNTHETIC_UNIT_COUNT = 44
EXPECTED_SYNTHETIC_CLUSTER_COUNT = 130
EXPECTED_SYNTHETIC_MANIFEST_SHA256 = (
    "fc9fd23f3c1d9c85303f8e80c4ad191bb0fa8b076ddb592dfe43ef6931b4ea78"
)
EXPECTED_SYNTHETIC_PAIR_ORDER_SHA256 = (
    "5aef5602f3af0c93e9187d3a702ba97de3aba6f0618f70aabd80000c16f8203c"
)
EXPECTED_REAL_MANIFEST_SHA256 = (
    "210ab081b2e70f39f35888a7b198458f6562847a2717d71758002c0225d32610"
)
EXPECTED_STRICT_COUNT = 547
EXPECTED_STRICT_POSITIVE = 508
EXPECTED_STRICT_CASE_COUNT = 445
EXPECTED_STRICT_PAIR_ORDER_SHA256 = (
    "f49232e468672466fee6236c9460101de8a5c0ef551866ef6815f62adf202ae5"
)
EXPECTED_BALANCED_COUNT = 1_016
EXPECTED_BALANCED_POSITIVE = 508
EXPECTED_CONSTRUCTED_COUNT = 469
EXPECTED_CONSTRUCTED_SELECTION_SHA256 = (
    "ba469ad0bbc1f9f61c8e74bf6159b8846c1e37853724379e5cfaa1b42d60db6b"
)
PAIRINGNET_RR_THRESHOLD = 4.0
ASSEMBLY_EDGE_TOLERANCES = (2, 5, 8, 10, 100)


class PairedBootstrapError(ValueError):
    """An input provenance, alignment, or statistical precondition failed."""


@dataclass(frozen=True)
class PairRow:
    pair_id: str
    label: bool
    cluster: str
    dependency_units: Tuple[str, ...]
    stratum: str
    probability: Mapping[str, float]
    valid: Mapping[str, bool]
    geometry: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class LoadedEvaluation:
    source_kind: str
    source_path: Path
    source_sha256: str
    methods: Tuple[str, ...]
    rows: Tuple[PairRow, ...]
    strict_prefix_count: Optional[int]
    validation_thresholds: Mapping[str, float]
    no_test_tuning_evidence: Mapping[str, object]
    input_files: Tuple[Mapping[str, object], ...]
    formal_evaluation: bool = True


def _duplicate_rejector(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PairedBootstrapError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PairedBootstrapError("non-finite JSON number: " + value)


def _loads(payload: str, description: str) -> object:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_duplicate_rejector,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise PairedBootstrapError(description + " is not strict JSON") from error


def _read_object(path: Path, description: str) -> Mapping[str, object]:
    try:
        value = _loads(Path(path).read_text(encoding="utf-8"), description)
    except OSError as error:
        raise PairedBootstrapError(description + " is unreadable") from error
    if not isinstance(value, Mapping):
        raise PairedBootstrapError(description + " root must be an object")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _number(value: object, name: str, *, probability: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairedBootstrapError(name + " must be numeric")
    output = float(value)
    if not math.isfinite(output) or (probability and not 0.0 <= output <= 1.0):
        raise PairedBootstrapError(name + " is outside its finite domain")
    return output


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721
        raise PairedBootstrapError(name + " must be an explicit bool")
    return bool(value)


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PairedBootstrapError(name + " must be a non-empty string")
    return value


def _sha256_string(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PairedBootstrapError(name + " must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise PairedBootstrapError(name + " must be hexadecimal") from error
    return value.casefold()


def _dependency_units(value: object, name: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 2:
        raise PairedBootstrapError(name + " must contain one or two source units")
    units = tuple(_nonempty_string(item, name) for item in value)
    if units != tuple(sorted(set(units))):
        raise PairedBootstrapError(name + " must be sorted and unique")
    return units


def _validation_threshold(value: object, arm: str) -> float:
    if not isinstance(value, Mapping):
        raise PairedBootstrapError(arm + " validation threshold is missing")
    if value.get("schema_version") != "dunhuang-pairwise-threshold/0.2":
        raise PairedBootstrapError(arm + " validation threshold schema differs")
    source = _nonempty_string(value.get("source_split"), arm + ".source_split")
    normalized = source.strip().casefold()
    validation_only = normalized in {"val", "validation", "dev"} or (
        ("validation" in normalized or normalized.startswith("val_"))
        and "train" not in normalized
        and "test" not in normalized
    )
    if not validation_only:
        raise PairedBootstrapError(arm + " threshold was not fitted on validation")
    if value.get("fit_method") != "maximize_cluster_balanced_f1":
        raise PairedBootstrapError(arm + " threshold fit method differs")
    for name in (
        "validation_fingerprint_sha256",
        "checkpoint_sha256",
        "model_config_sha256",
        "aggregation_config_sha256",
    ):
        if not _is_sha256(value.get(name)):
            raise PairedBootstrapError(arm + " threshold has invalid " + name)
    return _number(value.get("threshold"), arm + ".threshold", probability=True)


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:  # noqa: E721
        raise PairedBootstrapError(name + " must be a non-negative integer")
    return int(value)


def _same_translation_metric(float64_value: float, float32_value: float) -> bool:
    """Accept the producer's documented float32 serialization boundary only.

    Direct pose errors are emitted by a PyTorch float32 norm, while the
    PairingNet-style compatibility view recomputes the same norm in float64.
    Preserve the strict comparison first.  At the serialization boundary,
    require the direct value to be exactly representable as float32 and allow
    its bit pattern to differ by at most one ULP from the rounded float64 norm.
    One ULP covers the observed final-rounding difference between a float32
    vector norm and the float64 recomputation without creating a broad
    numerical-tolerance escape hatch.
    """

    if math.isclose(
        float64_value,
        float32_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return True
    rounded_compatibility = np.float32(float64_value)
    serialized_direct = np.float32(float32_value)
    if float(serialized_direct) != float32_value:
        return False
    compatibility_bits = int(rounded_compatibility.view(np.uint32))
    direct_bits = int(serialized_direct.view(np.uint32))
    return abs(compatibility_bits - direct_bits) <= 1


def _validate_direct_geometry(
    geometry: Mapping[str, object],
    *,
    pair_id: str,
    label: bool,
    full_valid: bool,
    probability: float,
    threshold: float,
) -> None:
    if set(geometry) != {
        "decision_valid",
        "translation_target_valid",
        "translation_hat_rc",
        "translation_l2_px",
        "correspondence",
        "pairingnet_style_registration",
        "assembly_edge",
    }:
        raise PairedBootstrapError(pair_id + " direct geometry schema differs")
    if (
        _bool(geometry.get("decision_valid"), pair_id + ".geometry.decision_valid")
        != full_valid
        or _bool(
            geometry.get("translation_target_valid"),
            pair_id + ".geometry.translation_target_valid",
        )
        != label
    ):
        raise PairedBootstrapError(pair_id + " direct geometry validity differs")
    translation_hat = geometry.get("translation_hat_rc")
    if (
        not isinstance(translation_hat, list)
        or len(translation_hat) != 2
        or any(
            not math.isfinite(_number(value, pair_id + ".translation_hat_rc"))
            for value in translation_hat
        )
    ):
        raise PairedBootstrapError(pair_id + " translation prediction is invalid")
    correspondence = geometry.get("correspondence")
    if not isinstance(correspondence, Mapping) or set(correspondence) != {
        "strict_true_positive",
        "strict_predicted_count",
        "mutual_top1_true_positive",
        "mutual_top1_predicted_count",
        "target_count",
        "dustbin_correct",
        "dustbin_total",
    }:
        raise PairedBootstrapError(pair_id + " correspondence count schema differs")
    counts = {
        name: _nonnegative_integer(value, pair_id + ".correspondence." + name)
        for name, value in correspondence.items()
    }
    if (
        counts["strict_true_positive"] > counts["strict_predicted_count"]
        or counts["strict_true_positive"] > counts["target_count"]
        or counts["mutual_top1_true_positive"]
        > counts["mutual_top1_predicted_count"]
        or counts["mutual_top1_true_positive"] > counts["target_count"]
        or counts["dustbin_correct"] > counts["dustbin_total"]
        ):
        raise PairedBootstrapError(pair_id + " correspondence counts are inconsistent")
    if (
        counts["strict_true_positive"] > counts["mutual_top1_true_positive"]
        or counts["strict_predicted_count"]
        > counts["mutual_top1_predicted_count"]
        or counts["dustbin_total"] <= 0
        or (label and counts["target_count"] <= 0)
        or (not label and counts["target_count"] != 0)
    ):
        raise PairedBootstrapError(pair_id + " correspondence semantics differ")
    if not full_valid and any(
        counts[name] != 0
        for name in (
            "strict_true_positive",
            "strict_predicted_count",
            "mutual_top1_true_positive",
            "mutual_top1_predicted_count",
            "dustbin_correct",
        )
    ):
        raise PairedBootstrapError(
            pair_id + " invalid decision has predicted correspondence evidence"
        )

    registration = geometry.get("pairingnet_style_registration")
    if not isinstance(registration, Mapping) or set(registration) != {
        "target_valid",
        "prediction_valid",
        "identity_fallback_used",
        "e_rmse",
        "registration_recall_lt4_success",
        "symmetric_hausdorff_px",
        "translation_l2_px",
        "normalized_translation_error",
        "source_contour_area_px2",
        "target_contour_area_px2",
        "rotation_error",
    }:
        raise PairedBootstrapError(pair_id + " PairingNet registration schema differs")
    if (
        _bool(registration.get("target_valid"), pair_id + ".registration.target")
        != label
        or _bool(
            registration.get("prediction_valid"), pair_id + ".registration.valid"
        )
        != full_valid
    ):
        raise PairedBootstrapError(pair_id + " registration validity differs")
    rotation = registration.get("rotation_error")
    if (
        not isinstance(rotation, Mapping)
        or set(rotation)
        != {"status", "estimated_or_supervised", "ground_truth_rotation_degrees"}
        or rotation.get("status")
        != "not_applicable_conditioned_upright_orientation"
        or rotation.get("estimated_or_supervised") is not False
        or _number(
            rotation.get("ground_truth_rotation_degrees"),
            pair_id + ".rotation.ground_truth",
        )
        != 0.0
    ):
        raise PairedBootstrapError(pair_id + " upright rotation N/A contract differs")
    metric_names = (
        "e_rmse",
        "symmetric_hausdorff_px",
        "translation_l2_px",
        "normalized_translation_error",
        "source_contour_area_px2",
        "target_contour_area_px2",
    )
    e_rmse: Optional[float] = None
    direct_translation_l2: Optional[float] = None
    if label:
        if (
            _bool(
                registration.get("identity_fallback_used"),
                pair_id + ".registration.identity_fallback",
            )
            != (not full_valid)
        ):
            raise PairedBootstrapError(pair_id + " identity fallback differs")
        values = {
            name: _number(registration.get(name), pair_id + ".registration." + name)
            for name in metric_names
        }
        if any(value < 0.0 for value in values.values()) or (
            values["source_contour_area_px2"]
            + values["target_contour_area_px2"]
            <= 0.0
        ):
            raise PairedBootstrapError(pair_id + " registration metric is negative")
        expected_nte = values["translation_l2_px"] / (
            values["source_contour_area_px2"]
            + values["target_contour_area_px2"]
        )
        if not math.isclose(
            values["normalized_translation_error"],
            expected_nte,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise PairedBootstrapError(pair_id + " normalized translation differs")
        e_rmse = values["e_rmse"]
        compatibility_translation_l2 = values["translation_l2_px"]
        if (
            _bool(
                registration.get("registration_recall_lt4_success"),
                pair_id + ".registration.rr",
            )
            != (e_rmse < PAIRINGNET_RR_THRESHOLD)
        ):
            raise PairedBootstrapError(pair_id + " RR<4 outcome differs")
        if full_valid:
            direct_translation_l2 = _number(
                geometry.get("translation_l2_px"), pair_id + ".translation_l2_px"
            )
            if not _same_translation_metric(
                compatibility_translation_l2, direct_translation_l2
            ):
                raise PairedBootstrapError(pair_id + " translation metrics disagree")
        elif geometry.get("translation_l2_px") is not None:
            raise PairedBootstrapError(
                pair_id + " invalid pose has a direct translation metric"
            )
    else:
        if any(registration.get(name) is not None for name in metric_names):
            raise PairedBootstrapError(pair_id + " ineligible registration is non-null")
        if registration.get("identity_fallback_used") is not None:
            raise PairedBootstrapError(pair_id + " negative fallback marker is non-null")
        if registration.get("registration_recall_lt4_success") is not None:
            raise PairedBootstrapError(pair_id + " ineligible RR outcome differs")

    assembly = geometry.get("assembly_edge")
    if not isinstance(assembly, Mapping) or set(assembly) != {
        "target_edge",
        "predicted_edge",
        "true_positive_by_tolerance",
    }:
        raise PairedBootstrapError(pair_id + " assembly-edge schema differs")
    predicted_edge = full_valid and probability >= threshold
    if (
        _bool(assembly.get("target_edge"), pair_id + ".assembly.target") != label
        or _bool(assembly.get("predicted_edge"), pair_id + ".assembly.predicted")
        != predicted_edge
    ):
        raise PairedBootstrapError(pair_id + " assembly-edge decision differs")
    outcomes = assembly.get("true_positive_by_tolerance")
    expected_names = {"at_{}".format(value) for value in ASSEMBLY_EDGE_TOLERANCES}
    if not isinstance(outcomes, Mapping) or set(outcomes) != expected_names:
        raise PairedBootstrapError(pair_id + " assembly-edge tolerances differ")
    for tolerance in ASSEMBLY_EDGE_TOLERANCES:
        expected = bool(
            label
            and predicted_edge
            and direct_translation_l2 is not None
            and direct_translation_l2 <= tolerance
        )
        if (
            _bool(
                outcomes.get("at_{}".format(tolerance)),
                pair_id + ".assembly.at_{}".format(tolerance),
            )
            != expected
        ):
            raise PairedBootstrapError(pair_id + " assembly-edge outcome differs")


def _safe_relative_file(root: Path, value: object, description: str) -> Path:
    logical_value = _nonempty_string(value, description)
    if "\\" in logical_value or "\x00" in logical_value:
        raise PairedBootstrapError(description + " is unsafe")
    logical = PurePosixPath(logical_value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise PairedBootstrapError(description + " is unsafe")
    current = root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise PairedBootstrapError(description + " may not traverse a symlink")
    try:
        path = current.resolve(strict=True)
        path.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise PairedBootstrapError(description + " escapes or is missing") from error
    if not path.is_file():
        raise PairedBootstrapError(description + " must be a regular file")
    return path


def _read_jsonl(path: Path, description: str) -> Tuple[Mapping[str, object], ...]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = _loads(line, "{} line {}".format(description, line_number))
                if not isinstance(value, Mapping):
                    raise PairedBootstrapError(
                        "{} line {} is not an object".format(description, line_number)
                    )
                rows.append(value)
    except OSError as error:
        raise PairedBootstrapError(description + " is unreadable") from error
    if not rows:
        raise PairedBootstrapError(description + " is empty")
    return tuple(rows)


def _method_order(methods: Iterable[str]) -> Tuple[str, ...]:
    available = set(methods)
    if not set(REQUIRED_METHODS).issubset(available):
        raise PairedBootstrapError("coarse_only and full_n512 are both required")
    matched = sorted(
        value
        for value in available
        - set(REQUIRED_METHODS)
        - set(SAME_DATA_BENCHMARK_METHODS)
        if value in {"matched_mm", "matched_route_a_siamese"}
        or ("matched" in value and ("mm" in value or "siamese" in value))
    )
    present_benchmarks = tuple(
        value for value in SAME_DATA_BENCHMARK_METHODS if value in available
    )
    if present_benchmarks not in ((), SAME_DATA_BENCHMARK_METHODS):
        raise PairedBootstrapError(
            "PairingNet and ShreddingNet adapted methods must be present together"
        )
    recognized = set(REQUIRED_METHODS) | set(matched) | set(present_benchmarks)
    if recognized != available:
        raise PairedBootstrapError("unrecognized method inventory")
    return tuple(REQUIRED_METHODS) + tuple(matched) + present_benchmarks


def _validate_sealed_receipt(
    receipt: Mapping[str, object], *, compatibility_mode: bool
) -> None:
    expected_status = (
        COMPATIBILITY_SEALED_STATUS
        if compatibility_mode
        else FORMAL_SEALED_STATUS
    )
    if (
        receipt.get("schema_version") != SEALED_SCHEMA_VERSION
        or receipt.get("status") != expected_status
        or receipt.get("test_accessed") is not True
        or receipt.get("real_external_test_accessed") is not False
    ):
        raise PairedBootstrapError("sealed receipt is not a completed synthetic test")
    source = receipt.get("source_training_run")
    protocol = receipt.get("protocol")
    if not isinstance(source, Mapping) or not isinstance(protocol, Mapping):
        raise PairedBootstrapError("sealed receipt provenance/protocol is missing")
    if compatibility_mode:
        if (
            receipt.get("formal_evaluation") is not False
            or receipt.get("compatibility_mode") is not True
            or protocol.get("formal_evaluation") is not False
            or protocol.get("compatibility_mode") is not True
        ):
            raise PairedBootstrapError(
                "sealed compatibility receipt is not explicitly non-formal"
            )
    elif (
        receipt.get("formal_evaluation") is not True
        or receipt.get("compatibility_mode") is not False
        or protocol.get("formal_evaluation") is not True
        or protocol.get("compatibility_mode") is not False
        or protocol.get("formal_exact_six_frozen_before_test_open") is not True
        or protocol.get("formal_method_inventory")
        != list(REQUIRED_METHODS)
        + ["matched_mm_converged", "matched_mm_same_exposure_epoch5"]
        + list(SAME_DATA_BENCHMARK_METHODS)
    ):
        raise PairedBootstrapError("sealed formal exact-six receipt differs")
    if (
        source.get("status_at_open") != "complete_train_validation_only"
        or source.get("winners_frozen_before_test_open") is not True
        or source.get("both_arms_validation_plateau_verified") is not True
        or source.get("formal_config_verified") is not True
        or protocol.get("training_performed") is not False
        or protocol.get("checkpoint_selection_performed") is not False
        or protocol.get("threshold_fit_performed") is not False
        or protocol.get("cross_arm_winner_selected_on_test") is not False
        or protocol.get("main_threshold")
        != "validation_fit_checkpoint_bound_artifact_only"
        or protocol.get(
            "all_requested_winners_and_thresholds_frozen_before_current_test_open"
        )
        is not True
        or protocol.get("formal_config_verified") is not True
        or protocol.get("both_arms_validation_plateau_verified") is not True
    ):
        raise PairedBootstrapError("sealed test shows tuning or an unfrozen winner")
    _sha256_string(source.get("receipt_sha256"), "sealed N=512 source receipt_sha256")
    disclosure = protocol.get("evaluation_history_disclosure")
    if (
        not isinstance(disclosure, Mapping)
        or disclosure.get("prior_epoch5_synthetic_test_completed") is not True
        or disclosure.get(
            "prior_epoch5_synthetic_test_human_visible_before_continuation"
        ) is not True
        or disclosure.get(
            "prior_epoch5_synthetic_test_used_by_automated_checkpoint_selection"
        ) is not False
        or disclosure.get(
            "prior_epoch5_synthetic_test_used_by_automated_threshold_fitting"
        ) is not False
        or disclosure.get(
            "prior_epoch5_synthetic_test_used_by_automated_early_stopping"
        ) is not False
        or disclosure.get(
            "prior_epoch5_synthetic_test_used_by_automated_scheduler"
        ) is not False
        or disclosure.get("claim_no_human_cognitive_influence") is not False
        or disclosure.get("prior_epoch5_real_evaluator_started_then_stopped")
        is not True
        or disclosure.get("prior_epoch5_real_result_formed_or_read") is not False
        or disclosure.get(
            "prior_convergence_time_synthetic_test_mask_morphology_review"
        )
        is not True
        or disclosure.get("prior_convergence_time_synthetic_test_mask_sample_count")
        != 500
        or disclosure.get("prior_convergence_time_synthetic_test_labels_read")
        is not False
        or disclosure.get("prior_convergence_time_synthetic_test_model_scores_read")
        is not False
        or disclosure.get(
            "prior_convergence_time_synthetic_test_activity_used_for_training_checkpoint_or_threshold_selection"
        )
        is not False
        or disclosure.get(
            "prior_convergence_time_synthetic_test_activity_influenced_fixed_corrosion_conditions"
        )
        is not False
        or disclosure.get(
            "prior_convergence_time_synthetic_test_activity_used_for_corrosion_runtime_and_representation_QA"
        )
        is not True
        or disclosure.get("real_data_accessed_in_that_activity") is not False
        or disclosure.get("claim_of_pristine_first_project_test") is not False
    ):
        raise PairedBootstrapError("sealed evaluation-history disclosure is incomplete")


def load_sealed(
    receipt_path: Path, *, compatibility_mode: bool = False
) -> LoadedEvaluation:
    """Load a sealed synthetic receipt and verify all score bytes/alignment."""

    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = _read_object(receipt_path, "sealed test receipt")
    if type(compatibility_mode) is not bool:  # noqa: E721
        raise TypeError("compatibility_mode must be bool")
    _validate_sealed_receipt(receipt, compatibility_mode=compatibility_mode)
    protocol = receipt.get("protocol")
    assert isinstance(protocol, Mapping)
    disclosure = protocol.get("evaluation_history_disclosure")
    assert isinstance(disclosure, Mapping)
    root = receipt_path.parent.resolve(strict=True)
    arm_results = receipt.get("arm_results")
    if not isinstance(arm_results, list) or not arm_results:
        raise PairedBootstrapError("sealed receipt has no arm results")
    result_by_arm: Dict[str, Mapping[str, object]] = {}
    for result in arm_results:
        if not isinstance(result, Mapping):
            raise PairedBootstrapError("sealed arm result is malformed")
        arm = _nonempty_string(result.get("arm"), "sealed arm")
        if arm in result_by_arm:
            raise PairedBootstrapError("duplicate sealed arm: " + arm)
        result_by_arm[arm] = result
    methods = _method_order(result_by_arm)
    if not compatibility_mode and methods != (
        tuple(REQUIRED_METHODS)
        + ("matched_mm_converged", "matched_mm_same_exposure_epoch5")
        + SAME_DATA_BENCHMARK_METHODS
    ):
        raise PairedBootstrapError(
            "formal sealed evaluation must contain exact-six methods"
        )
    if set(result_by_arm) != set(methods):
        raise PairedBootstrapError("sealed receipt contains an unsupported extra arm")
    matched_methods = tuple(
        method
        for method in methods
        if method not in REQUIRED_METHODS
        and method not in SAME_DATA_BENCHMARK_METHODS
    )
    benchmark_methods = tuple(
        method for method in methods if method in SAME_DATA_BENCHMARK_METHODS
    )
    matched_source = receipt.get("source_matched_mm_training_run")
    if matched_methods:
        if (
            not isinstance(matched_source, Mapping)
            or matched_source.get("status_at_open") != "complete_train_validation_only"
            or matched_source.get(
                "converged_and_epoch5_winners_frozen_before_test_open"
            )
            is not True
            or matched_source.get("formal_config_verified") is not True
        ):
            raise PairedBootstrapError(
                "sealed matched-MM source was not frozen before test open"
            )
        _sha256_string(
            matched_source.get("receipt_sha256"),
            "sealed matched-MM source receipt_sha256",
        )
    elif matched_source is not None:
        raise PairedBootstrapError(
            "sealed receipt names a matched-MM source without matched methods"
        )
    benchmark_source = receipt.get("source_same_data_benchmark_training_runs")
    if benchmark_methods:
        if (
            not isinstance(benchmark_source, Mapping)
            or benchmark_source.get("all_methods_frozen_before_test_open") is not True
            or protocol.get("same_data_benchmark_methods_requested") is not True
            or protocol.get(
                "same_data_benchmark_winners_and_thresholds_frozen_before_test_open"
            )
            is not True
        ):
            raise PairedBootstrapError(
                "sealed same-data benchmark source was not frozen before test open"
            )
        benchmark_provenance = benchmark_source.get("methods")
        if not isinstance(benchmark_provenance, Mapping) or set(
            benchmark_provenance
        ) != set(SAME_DATA_BENCHMARK_METHODS):
            raise PairedBootstrapError(
                "sealed same-data benchmark provenance inventory differs"
            )
    elif benchmark_source is not None:
        raise PairedBootstrapError(
            "sealed receipt names benchmark source without benchmark methods"
        )

    thresholds: Dict[str, float] = {}
    input_files: list[Mapping[str, object]] = [
        {
            "role": "test_receipt",
            "path": str(receipt_path),
            "sha256": _file_sha256(receipt_path),
        }
    ]
    raw_by_arm: Dict[str, Tuple[Mapping[str, object], ...]] = {}
    for arm in methods:
        result = result_by_arm[arm]
        if (
            result.get("training_performed") is not False
            or result.get("checkpoint_selection_performed") is not False
            or result.get("threshold_fit_performed") is not False
        ):
            raise PairedBootstrapError(arm + " result performed test-time tuning")
        if (
            arm in matched_methods
            and result.get("score_semantics") != "historical_mm_probability"
        ):
            raise PairedBootstrapError(arm + " sealed score semantics differs")
        if (
            arm in benchmark_methods
            and (
                result.get("score_semantics") != "adapted_pair_probability"
                or result.get("native_cm_fm_se_or_ga_claimed") is not False
            )
        ):
            raise PairedBootstrapError(arm + " adapted score semantics differs")
        _sha256_string(
            result.get("winner_checkpoint_sha256"),
            arm + " winner_checkpoint_sha256",
        )
        threshold_value = result.get("validation_threshold")
        threshold = _validation_threshold(threshold_value, arm)
        thresholds[arm] = threshold
        expected_threshold_sha = result.get("validation_threshold_sha256")
        if expected_threshold_sha != _canonical_sha256(threshold_value):
            raise PairedBootstrapError(arm + " validation-threshold SHA differs")
        if isinstance(threshold_value, Mapping) and threshold_value.get(
            "checkpoint_sha256"
        ) != result.get("winner_checkpoint_sha256"):
            raise PairedBootstrapError(arm + " threshold/checkpoint binding differs")
        score_path = _safe_relative_file(
            root, result.get("pair_scores"), arm + " pair_scores"
        )
        observed_sha = _file_sha256(score_path)
        if observed_sha != result.get("pair_scores_sha256"):
            raise PairedBootstrapError(arm + " pair_scores SHA-256 differs")
        raw = _read_jsonl(score_path, arm + " pair_scores")
        if result.get("pair_scores_count") != len(raw):
            raise PairedBootstrapError(arm + " pair_scores count differs")
        raw_by_arm[arm] = raw
        input_files.append(
            {
                "role": arm + "_pair_scores",
                "path": str(score_path),
                "sha256": observed_sha,
            }
        )

    first_arm = methods[0]
    ordered_ids: list[str] = []
    reference: Dict[str, Tuple[bool, str, Tuple[str, ...]]] = {}
    parsed_by_arm: Dict[
        str, Dict[str, Tuple[float, bool, Optional[Mapping[str, object]]]]
    ] = {}
    for arm in methods:
        parsed: Dict[str, Tuple[float, bool, Optional[Mapping[str, object]]]] = {}
        for raw in raw_by_arm[arm]:
            if (
                raw.get("schema_version") != SEALED_PAIR_SCHEMA_VERSION
                or raw.get("arm") != arm
            ):
                raise PairedBootstrapError(arm + " pair row schema/identity differs")
            pair_id = _nonempty_string(raw.get("pair_id"), arm + ".pair_id")
            if pair_id in parsed:
                raise PairedBootstrapError(arm + " duplicate pair_id=" + pair_id)
            label = _bool(raw.get("label"), pair_id + ".label")
            cluster = _nonempty_string(raw.get("cluster_id"), pair_id + ".cluster_id")
            dependency_units = _dependency_units(
                raw.get("source_unit_ids"), pair_id + ".source_unit_ids"
            )
            if arm == first_arm:
                ordered_ids.append(pair_id)
                reference[pair_id] = (label, cluster, dependency_units)
            elif pair_id not in reference:
                raise PairedBootstrapError(arm + " has unexpected pair_id=" + pair_id)
            elif reference[pair_id] != (label, cluster, dependency_units):
                if reference[pair_id][0] != label:
                    field = "label"
                elif reference[pair_id][1] != cluster:
                    field = "cluster_id"
                else:
                    field = "source_unit_ids"
                raise PairedBootstrapError(field + " differs for pair_id=" + pair_id)
            main_score = _nonempty_string(
                raw.get("main_score"), pair_id + ".main_score"
            )
            expected_score = (
                "coarse"
                if arm == "coarse_only"
                else "fused"
                if arm == "full_n512"
                else "pair_probability"
                if arm in benchmark_methods
                else "historical_mm_probability"
            )
            if main_score != expected_score:
                raise PairedBootstrapError(arm + " main score differs")
            scores = raw.get("scores")
            decision = raw.get("decision")
            if not isinstance(scores, Mapping) or not isinstance(decision, Mapping):
                raise PairedBootstrapError(pair_id + " scores/decision missing")
            main = scores.get(main_score)
            if not isinstance(main, Mapping):
                raise PairedBootstrapError(pair_id + " main score missing")
            probability = _number(
                main.get("probability"), pair_id + ".probability", probability=True
            )
            valid = _bool(main.get("valid"), pair_id + ".valid")
            if _bool(decision.get("valid"), pair_id + ".decision.valid") != valid:
                raise PairedBootstrapError(pair_id + " score/decision validity differs")
            row_threshold = _number(
                decision.get("validation_threshold"),
                pair_id + ".decision.threshold",
                probability=True,
            )
            if row_threshold != thresholds[arm]:
                raise PairedBootstrapError(pair_id + " uses a non-frozen threshold")
            expected_decision: Optional[bool] = (
                probability >= row_threshold if valid else None
            )
            if decision.get("predicted_label") is not expected_decision:
                raise PairedBootstrapError(
                    pair_id + " frozen-threshold decision differs"
                )
            geometry = raw.get("geometry")
            if geometry is not None and not isinstance(geometry, Mapping):
                raise PairedBootstrapError(pair_id + " geometry is malformed")
            parsed[pair_id] = (probability, valid, geometry)
        parsed_by_arm[arm] = parsed
    if any(set(parsed_by_arm[arm]) != set(reference) for arm in methods):
        raise PairedBootstrapError("sealed arms do not contain the same pair_id set")

    population = receipt.get("test_population")
    if not isinstance(population, Mapping):
        raise PairedBootstrapError("sealed test population receipt is missing")
    if (
        population.get("observed_count") != len(reference)
        or population.get("expected_count") != len(reference)
        or population.get("exact_one_to_one") is not True
        or population.get("positive_count")
        != sum(value[0] for value in reference.values())
        or population.get("negative_count")
        != sum(not value[0] for value in reference.values())
        or population.get("cluster_count")
        != len({value[1] for value in reference.values()})
        or population.get("pair_ids_sha256") != _canonical_sha256(ordered_ids)
    ):
        raise PairedBootstrapError("sealed population receipt differs from score rows")
    if sum(value[0] for value in reference.values()) * 2 != len(reference):
        raise PairedBootstrapError("sealed population is not exact 1:1")
    source_units = {unit for value in reference.values() for unit in value[2]}
    if (
        len(reference) != EXPECTED_SYNTHETIC_COUNT
        or sum(value[0] for value in reference.values()) != EXPECTED_SYNTHETIC_POSITIVE
        or len(source_units) != EXPECTED_SYNTHETIC_UNIT_COUNT
        or len({value[1] for value in reference.values()})
        != EXPECTED_SYNTHETIC_CLUSTER_COUNT
        or population.get("manifest_sha256") != EXPECTED_SYNTHETIC_MANIFEST_SHA256
        or population.get("pair_ids_sha256") != EXPECTED_SYNTHETIC_PAIR_ORDER_SHA256
    ):
        raise PairedBootstrapError("sealed population is not the frozen formal 3000")

    rows = []
    for pair_id in ordered_ids:
        label, cluster, dependency_units = reference[pair_id]
        geometry = parsed_by_arm.get("full_n512", {}).get(pair_id, (0.0, False, None))[
            2
        ]
        full_valid = parsed_by_arm["full_n512"][pair_id][1]
        if not isinstance(geometry, Mapping):
            raise PairedBootstrapError(pair_id + " lacks full_n512 geometry")
        if (
            _bool(geometry.get("decision_valid"), pair_id + ".geometry.decision_valid")
            != full_valid
            or _bool(
                geometry.get("translation_target_valid"),
                pair_id + ".geometry.translation_target_valid",
            )
            != label
        ):
            raise PairedBootstrapError(pair_id + " geometry validity/label differs")
        vector = geometry.get("translation_hat_rc")
        if (
            not isinstance(vector, list)
            or len(vector) != 2
            or any(
                not math.isfinite(float(value))
                for value in vector
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            )
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in vector
            )
        ):
            raise PairedBootstrapError(pair_id + " translation vector is invalid")
        error = geometry.get("translation_l2_px")
        if label and full_valid:
            if _number(error, pair_id + ".translation_l2_px") < 0.0:
                raise PairedBootstrapError(pair_id + " translation error is negative")
        elif error is not None:
            raise PairedBootstrapError(
                pair_id + " ineligible translation error is non-null"
            )
        _validate_direct_geometry(
            geometry,
            pair_id=pair_id,
            label=label,
            full_valid=full_valid,
            probability=parsed_by_arm["full_n512"][pair_id][0],
            threshold=thresholds["full_n512"],
        )
        rows.append(
            PairRow(
                pair_id=pair_id,
                label=label,
                cluster=cluster,
                dependency_units=dependency_units,
                stratum="synthetic_positive" if label else "synthetic_negative",
                probability={arm: parsed_by_arm[arm][pair_id][0] for arm in methods},
                valid={arm: parsed_by_arm[arm][pair_id][1] for arm in methods},
                geometry=geometry,
            )
        )
    return LoadedEvaluation(
        source_kind="sealed_synthetic",
        source_path=receipt_path,
        source_sha256=_file_sha256(receipt_path),
        methods=methods,
        rows=tuple(rows),
        strict_prefix_count=None,
        validation_thresholds=thresholds,
        no_test_tuning_evidence={
            "automated_nonuse_verified": True,
            "claim_no_human_cognitive_influence": False,
            "evaluation_history_disclosure": dict(disclosure),
            "winner_frozen_before_test_open": True,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_fit_performed": False,
            "threshold_source": "validation_checkpoint_bound_artifact",
        },
        input_files=tuple(input_files),
        formal_evaluation=not compatibility_mode,
    )


def _real_stratum(
    raw: Mapping[str, object], index: int, label: bool, strict_prefix_count: int
) -> str:
    expected = (
        ("strict_manifest_positive" if label else "strict_manifest_negative")
        if index < strict_prefix_count
        else "constructed_not_GT_negative"
    )
    supplied = raw.get("stratum", raw.get("label_origin"))
    if supplied is not None:
        supplied = _nonempty_string(supplied, "real pair stratum")
        aliases = {
            "strict_manifest_positive": "strict_manifest_positive",
            "strict_manifest_negative": "strict_manifest_negative",
            "constructed_not_GT_negative": "constructed_not_GT_negative",
            "constructed_distractor_not_gt_negative": "constructed_not_GT_negative",
        }
        normalized = aliases.get(supplied, supplied)
        if normalized != expected:
            raise PairedBootstrapError("real pair stratum/order semantics differ")
    return expected


def _constructed_receipt_units(
    value: Mapping[str, object], description: str
) -> Tuple[str, ...]:
    explicit = value.get("source_case_uids")
    if explicit is not None:
        return _dependency_units(explicit, description + ".source_case_uids")
    first = value.get("fragment_a")
    second = value.get("fragment_b")
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise PairedBootstrapError(description + " lacks constructed endpoints")
    units = sorted(
        {
            _nonempty_string(
                first.get("source_case_uid"), description + ".fragment_a.case"
            ),
            _nonempty_string(
                second.get("source_case_uid"), description + ".fragment_b.case"
            ),
        }
    )
    return _dependency_units(units, description + ".source_case_uids")


def _unwrap_real_document(
    document: Mapping[str, object], *, compatibility_mode: bool
) -> Mapping[str, object]:
    """Validate the single-forward wrapper and return its balanced population."""

    expected_wrapper_status = (
        COMPATIBILITY_REAL_COMBINED_STATUS
        if compatibility_mode
        else FORMAL_REAL_COMBINED_STATUS
    )
    if document.get("status") != expected_wrapper_status:
        if compatibility_mode and document.get("status") in {
            COMPATIBILITY_REAL_STRICT_STATUS,
            COMPATIBILITY_REAL_BALANCED_STATUS,
        }:
            return document
        raise PairedBootstrapError(
            "formal real input must be combined balanced1016 exact-prefix wrapper"
            if not compatibility_mode
            else "real input status differs from explicit compatibility mode"
        )
    if document.get("schema_version") != REAL_SCHEMA_VERSION:
        raise PairedBootstrapError("combined real wrapper schema differs")
    contract = document.get("forward_contract")
    strict = document.get("strict_547")
    balanced = document.get("balanced_1016")
    if (
        not isinstance(contract, Mapping)
        or not isinstance(strict, Mapping)
        or not isinstance(balanced, Mapping)
        or contract.get("forward_pair_count_per_arm") != EXPECTED_BALANCED_COUNT
        or contract.get("strict_547_derived_from_exact_prediction_prefix") is not True
        or contract.get("strict_pairs_forwarded_twice") is not False
        or contract.get("all_methods_share_exact_strict_prediction_prefix") is not True
        or strict.get("status")
        != (
            COMPATIBILITY_REAL_STRICT_STATUS
            if compatibility_mode
            else "complete_strict_547_target_blind_external_test"
        )
        or balanced.get("status")
        != (
            COMPATIBILITY_REAL_BALANCED_STATUS
            if compatibility_mode
            else "complete_balanced_1016_target_blind_external_test"
        )
    ):
        raise PairedBootstrapError("combined real single-forward contract differs")
    strict_pairs = strict.get("pairs")
    balanced_pairs = balanced.get("pairs")
    if (
        not isinstance(strict_pairs, list)
        or not isinstance(balanced_pairs, list)
        or len(strict_pairs) != EXPECTED_STRICT_COUNT
        or len(balanced_pairs) != EXPECTED_BALANCED_COUNT
        or strict_pairs != balanced_pairs[:EXPECTED_STRICT_COUNT]
    ):
        raise PairedBootstrapError("combined real strict prefix is not exact")
    declared_methods = contract.get("methods")
    balanced_methods = balanced.get("methods")
    if (
        not isinstance(declared_methods, list)
        or not isinstance(balanced_methods, Mapping)
        or set(declared_methods) != set(balanced_methods)
    ):
        raise PairedBootstrapError("combined real method inventory differs")
    return balanced


def load_real(path: Path, *, compatibility_mode: bool = False) -> LoadedEvaluation:
    """Load strict547 or balanced1016 target-blind real evaluation JSON."""

    path = Path(path).resolve(strict=True)
    if type(compatibility_mode) is not bool:  # noqa: E721
        raise TypeError("compatibility_mode must be bool")
    outer_document = _read_object(path, "real evaluation")
    document = _unwrap_real_document(
        outer_document, compatibility_mode=compatibility_mode
    )
    status = document.get("status")
    expected_statuses = (
        {COMPATIBILITY_REAL_STRICT_STATUS, COMPATIBILITY_REAL_BALANCED_STATUS}
        if compatibility_mode
        else {"complete_balanced_1016_target_blind_external_test"}
    )
    if (
        document.get("schema_version") != REAL_SCHEMA_VERSION
        or status not in expected_statuses
    ):
        raise PairedBootstrapError("unsupported or incomplete Rachel real evaluation")
    protocol = document.get("protocol")
    dataset = document.get("dataset")
    method_metadata = document.get("methods")
    raw_rows = document.get("pairs")
    if (
        not isinstance(protocol, Mapping)
        or not isinstance(dataset, Mapping)
        or not isinstance(method_metadata, Mapping)
        or not isinstance(raw_rows, list)
        or not raw_rows
    ):
        raise PairedBootstrapError(
            "real evaluation lacks protocol/dataset/methods/pairs"
        )
    if (
        protocol.get("winner_and_validation_threshold_frozen_before_real_open")
        is not True
        or protocol.get("real_correspondence_or_translation_gt_read") is not False
        or protocol.get("primary_metrics") != "threshold_free_AUROC_AUPRC"
        or protocol.get("thresholded_metrics")
        != "secondary_frozen_validation_threshold_only"
        or protocol.get(
            "all_requested_winners_and_thresholds_frozen_before_current_real_open"
        )
        is not True
        or protocol.get("formal_config_verified") is not True
        or protocol.get("both_arms_validation_plateau_verified") is not True
        or protocol.get("positive_direction_derived_or_bbox_read") is not False
    ):
        raise PairedBootstrapError("real evaluation permits tuning or target leakage")
    if compatibility_mode:
        if (
            protocol.get("formal_evaluation") is not False
            or protocol.get("compatibility_mode") is not True
        ):
            raise PairedBootstrapError(
                "real compatibility input is not explicitly non-formal"
            )
    elif (
        protocol.get("formal_evaluation") is not True
        or protocol.get("compatibility_mode") is not False
        or protocol.get("formal_exact_six_frozen_before_real_open") is not True
        or protocol.get("balanced1016_single_forward_required") is not True
        or protocol.get("formal_method_inventory")
        != list(REQUIRED_METHODS)
        + ["matched_mm_converged", "matched_mm_same_exposure_epoch5"]
        + list(SAME_DATA_BENCHMARK_METHODS)
    ):
        raise PairedBootstrapError("formal real exact-six protocol differs")
    disclosure = protocol.get("evaluation_history_disclosure")
    if (
        not isinstance(disclosure, Mapping)
        or disclosure.get("prior_epoch5_synthetic_test_completed") is not True
        or disclosure.get(
            "prior_epoch5_synthetic_test_human_visible_before_continuation"
        ) is not True
        or disclosure.get(
            "prior_epoch5_synthetic_test_used_by_automated_checkpoint_selection"
        ) is not False
        or disclosure.get(
            "prior_epoch5_synthetic_test_used_by_automated_threshold_fitting"
        ) is not False
        or disclosure.get(
            "prior_epoch5_synthetic_test_used_by_automated_early_stopping"
        ) is not False
        or disclosure.get(
            "prior_epoch5_synthetic_test_used_by_automated_scheduler"
        ) is not False
        or disclosure.get("claim_no_human_cognitive_influence") is not False
        or disclosure.get("prior_epoch5_real_evaluator_started_then_stopped")
        is not True
        or disclosure.get("prior_epoch5_real_result_formed_or_read") is not False
        or disclosure.get(
            "prior_convergence_time_synthetic_test_mask_morphology_review"
        )
        is not True
        or disclosure.get("prior_convergence_time_synthetic_test_mask_sample_count")
        != 500
        or disclosure.get("prior_convergence_time_synthetic_test_labels_read")
        is not False
        or disclosure.get("prior_convergence_time_synthetic_test_model_scores_read")
        is not False
        or disclosure.get(
            "prior_convergence_time_synthetic_test_activity_used_for_training_checkpoint_or_threshold_selection"
        )
        is not False
        or disclosure.get(
            "prior_convergence_time_synthetic_test_activity_influenced_fixed_corrosion_conditions"
        )
        is not False
        or disclosure.get(
            "prior_convergence_time_synthetic_test_activity_used_for_corrosion_runtime_and_representation_QA"
        )
        is not True
        or disclosure.get("real_data_accessed_in_that_activity") is not False
        or disclosure.get("claim_of_project_first_real_access") is not False
    ):
        raise PairedBootstrapError("real evaluation-history disclosure is incomplete")

    source_training = document.get("source_training_run")
    if (
        not isinstance(source_training, Mapping)
        or source_training.get("status_at_open") != "complete_train_validation_only"
        or source_training.get("both_arms_validation_plateau_verified") is not True
        or source_training.get("formal_config_verified") is not True
    ):
        raise PairedBootstrapError("real N=512 source provenance is incomplete")
    _sha256_string(
        source_training.get("receipt_sha256"),
        "real N=512 source receipt_sha256",
    )

    methods = _method_order(method_metadata)
    if not compatibility_mode and methods != (
        tuple(REQUIRED_METHODS)
        + ("matched_mm_converged", "matched_mm_same_exposure_epoch5")
        + SAME_DATA_BENCHMARK_METHODS
    ):
        raise PairedBootstrapError(
            "formal real evaluation must contain exact-six methods"
        )
    matched_methods = tuple(
        method
        for method in methods
        if method not in REQUIRED_METHODS
        and method not in SAME_DATA_BENCHMARK_METHODS
    )
    benchmark_methods = tuple(
        method for method in methods if method in SAME_DATA_BENCHMARK_METHODS
    )
    matched_source = document.get("source_matched_mm_training_run")
    if matched_methods:
        if (
            not isinstance(matched_source, Mapping)
            or matched_source.get("status_at_open") != "complete_train_validation_only"
            or matched_source.get(
                "converged_and_epoch5_winners_frozen_before_current_real_open"
            )
            is not True
            or matched_source.get("formal_config_verified") is not True
        ):
            raise PairedBootstrapError(
                "real matched-MM source provenance is incomplete"
            )
        _sha256_string(
            matched_source.get("receipt_sha256"),
            "real matched-MM source receipt_sha256",
        )
    elif matched_source is not None:
        raise PairedBootstrapError(
            "real result names a matched-MM source without matched methods"
        )
    benchmark_source = document.get("source_same_data_benchmark_training_runs")
    if benchmark_methods:
        if (
            not isinstance(benchmark_source, Mapping)
            or benchmark_source.get(
                "all_winners_and_validation_thresholds_frozen_before_current_real_open"
            )
            is not True
            or protocol.get(
                "same_data_benchmark_winners_and_validation_thresholds_frozen_before_current_real_open"
            )
            is not True
            or protocol.get("same_data_benchmark_mask_only") is not True
            or protocol.get(
                "same_data_benchmark_upright_translation_only_adaptation"
            )
            is not True
            or protocol.get("same_data_benchmark_exact_reproduction_claimed")
            is not False
            or protocol.get(
                "shreddingnet_balanced_selected_list_claimed_as_native_CM_FM_SE_or_GA"
            )
            is not False
        ):
            raise PairedBootstrapError(
                "real same-data benchmark provenance is incomplete"
            )
        benchmark_provenance = benchmark_source.get("methods")
        if not isinstance(benchmark_provenance, Mapping) or set(
            benchmark_provenance
        ) != set(SAME_DATA_BENCHMARK_METHODS):
            raise PairedBootstrapError(
                "real same-data benchmark provenance inventory differs"
            )
    elif benchmark_source is not None:
        raise PairedBootstrapError(
            "real result names same-data benchmark source without methods"
        )
    thresholds: Dict[str, float] = {}
    for method in methods:
        metadata = method_metadata.get(method)
        if not isinstance(metadata, Mapping):
            raise PairedBootstrapError(method + " real method metadata is missing")
        if (
            method in matched_methods
            and metadata.get("score_semantics") != "historical_mm_probability"
        ):
            raise PairedBootstrapError(method + " real score semantics differs")
        if (
            method in benchmark_methods
            and metadata.get("score_semantics") != "adapted_pair_probability"
        ):
            raise PairedBootstrapError(method + " adapted score semantics differs")
        winner = metadata.get("winner")
        if (
            isinstance(winner, Mapping)
            and winner.get("validation_threshold") is not None
        ):
            threshold_value = winner.get("validation_threshold")
            threshold = _validation_threshold(threshold_value, method)
            if method in benchmark_methods:
                stages = winner.get("checkpoint_sha256_by_stage")
                if (
                    not isinstance(stages, Mapping)
                    or not stages
                    or any(
                        _sha256_string(value, method + " checkpoint stage") != value
                        for value in stages.values()
                    )
                    or not isinstance(threshold_value, Mapping)
                    or threshold_value.get("checkpoint_sha256")
                    not in set(stages.values())
                    or metadata.get("native_cm_fm_se_or_ga_claimed") is not False
                ):
                    raise PairedBootstrapError(
                        method + " adapted threshold/checkpoint binding differs"
                    )
            else:
                _sha256_string(
                    winner.get("checkpoint_sha256"),
                    method + " checkpoint_sha256",
                )
                if isinstance(threshold_value, Mapping) and threshold_value.get(
                    "checkpoint_sha256"
                ) != winner.get("checkpoint_sha256"):
                    raise PairedBootstrapError(
                        method + " real threshold binding differs"
                    )
            thresholds[method] = threshold
        else:
            raise PairedBootstrapError(method + " lacks a frozen validation threshold")

    is_balanced = status in {
        "complete_balanced_1016_target_blind_external_test",
        COMPATIBILITY_REAL_BALANCED_STATUS,
    }
    constructed_receipt_rows: Tuple[Mapping[str, object], ...] = ()
    if is_balanced:
        strict_prefix = dataset.get("strict_prefix_count")
        constructed = dataset.get("constructed_count")
        semantics = document.get("negative_semantics")
        construction = document.get("construction_receipt")
        if (
            type(strict_prefix) is not int  # noqa: E721
            or strict_prefix <= 0
            or type(constructed) is not int  # noqa: E721
            or constructed <= 0
            or strict_prefix + constructed != len(raw_rows)
            or dataset.get("strict_prefix_preserved_exactly") is not True
            or not isinstance(semantics, Mapping)
            or semantics.get("constructed_are_GT_negatives") is not False
            or semantics.get("never_used_for_training_threshold_or_tuning") is not True
            or not isinstance(construction, Mapping)
        ):
            raise PairedBootstrapError("balanced1016 construction semantics differ")
        construction_payload = dict(construction)
        content_sha = construction_payload.pop("content_sha256", None)
        raw_constructed_rows = construction.get("constructed_pairs")
        if (
            construction.get("status")
            != "complete_label_blind_constructed_distractor_plan"
            or construction.get("constructed_selection_sha256")
            != EXPECTED_CONSTRUCTED_SELECTION_SHA256
            or not isinstance(raw_constructed_rows, list)
            or any(not isinstance(row, Mapping) for row in raw_constructed_rows)
            or construction.get("constructed_selection_sha256")
            != _canonical_sha256(raw_constructed_rows)
            or content_sha != _canonical_sha256(construction_payload)
            or len(raw_constructed_rows) != EXPECTED_CONSTRUCTED_COUNT
            or construction.get("selection_uses_model_scores") is not False
            or construction.get("selection_uses_pair_labels") is not False
            or construction.get("selection_uses_gt_bbox_or_direction") is not False
            or construction.get("same_case_forbidden") is not True
            or construction.get("same_alpha_sha256_forbidden") is not True
            or construction.get("uses_per_fragment_occurrence") != 1
            or construction.get("max_pairs_per_unordered_case_pair") != 1
            or construction.get("negative_semantics") != "constructed_not_GT_negative"
        ):
            raise PairedBootstrapError("balanced1016 construction receipt differs")
        constructed_receipt_rows = tuple(raw_constructed_rows)
    else:
        strict_prefix = len(raw_rows)

    rows = []
    seen = set()
    ordered_ids = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise PairedBootstrapError("real pair row is malformed")
        pair_id = _nonempty_string(raw.get("pair_id"), "real pair_id")
        if pair_id in seen:
            raise PairedBootstrapError("duplicate real pair_id=" + pair_id)
        seen.add(pair_id)
        ordered_ids.append(pair_id)
        label = _bool(raw.get("label"), pair_id + ".label")
        cluster = _nonempty_string(
            raw.get("case_cluster", raw.get("cluster_id")), pair_id + ".case_cluster"
        )
        dependency_units = _dependency_units(
            raw.get("source_case_uids"), pair_id + ".source_case_uids"
        )
        stratum = _real_stratum(raw, index, label, strict_prefix)
        method_rows = raw.get("methods")
        if not isinstance(method_rows, Mapping):
            raise PairedBootstrapError(pair_id + " lacks method predictions")
        if not set(methods).issubset(method_rows):
            raise PairedBootstrapError(pair_id + " lacks a compared method")
        probability: Dict[str, float] = {}
        validity: Dict[str, bool] = {}
        for method in methods:
            prediction = method_rows.get(method)
            if not isinstance(prediction, Mapping):
                raise PairedBootstrapError(pair_id + " has malformed " + method)
            if (
                method in matched_methods
                and prediction.get("score_semantics") != "historical_mm_probability"
            ):
                raise PairedBootstrapError(
                    pair_id + "." + method + " score semantics differs"
                )
            if (
                method in benchmark_methods
                and prediction.get("score_semantics") != "adapted_pair_probability"
            ):
                raise PairedBootstrapError(
                    pair_id + "." + method + " adapted score semantics differs"
                )
            probability[method] = _number(
                prediction.get("probability"),
                pair_id + "." + method + ".probability",
                probability=True,
            )
            validity[method] = _bool(
                prediction.get("valid"), pair_id + "." + method + ".valid"
            )
            method_stratum = prediction.get("stratum")
            if method_stratum is not None and method_stratum != stratum:
                raise PairedBootstrapError(
                    "stratum differs across methods for pair_id=" + pair_id
                )
            if method in thresholds:
                expected = (
                    probability[method] >= thresholds[method]
                    if validity[method]
                    else None
                )
                decision = prediction.get("decision_at_frozen_validation_threshold")
                if decision is not expected:
                    raise PairedBootstrapError(
                        pair_id + " has a non-frozen real threshold decision"
                    )
        coarse_full_valid = validity["coarse_only"] and validity["full_n512"]
        if raw.get("coarse_full_common_valid") is not coarse_full_valid:
            raise PairedBootstrapError(pair_id + " common-valid flag differs")
        rows.append(
            PairRow(
                pair_id=pair_id,
                label=label,
                cluster=cluster,
                dependency_units=dependency_units,
                stratum=stratum,
                probability=probability,
                valid=validity,
            )
        )

    positive_count = sum(row.label for row in rows)
    expected_count = EXPECTED_BALANCED_COUNT if is_balanced else EXPECTED_STRICT_COUNT
    expected_positive = (
        EXPECTED_BALANCED_POSITIVE if is_balanced else EXPECTED_STRICT_POSITIVE
    )
    if (
        dataset.get("pair_count") != len(rows)
        or dataset.get("positive_count") != positive_count
        or dataset.get("negative_count") != len(rows) - positive_count
        or dataset.get("case_cluster_count") != len({row.cluster for row in rows})
        or dataset.get("pair_order_sha256") != _canonical_sha256(ordered_ids)
        or dataset.get("manifest_sha256") != EXPECTED_REAL_MANIFEST_SHA256
        or _canonical_sha256(ordered_ids[:strict_prefix])
        != EXPECTED_STRICT_PAIR_ORDER_SHA256
        or len(rows) != expected_count
        or positive_count != expected_positive
        or (not is_balanced and any(len(row.dependency_units) != 1 for row in rows))
        or (
            not is_balanced
            and len({row.dependency_units[0] for row in rows})
            != EXPECTED_STRICT_CASE_COUNT
        )
        or (
            not is_balanced
            and dataset.get("pair_order_sha256") != EXPECTED_STRICT_PAIR_ORDER_SHA256
        )
    ):
        raise PairedBootstrapError("real dataset receipt differs from pair rows")
    if is_balanced:
        for index, (receipt_row, result_row) in enumerate(
            zip(constructed_receipt_rows, rows[strict_prefix:])
        ):
            description = "constructed receipt row {}".format(index)
            if (
                receipt_row.get("pair_id") != result_row.pair_id
                or _constructed_receipt_units(receipt_row, description)
                != result_row.dependency_units
                or receipt_row.get("constructed_is_ground_truth_negative", False)
                is not False
            ):
                raise PairedBootstrapError(
                    "balanced result rows differ from constructed receipt"
                )
            declared_cluster = receipt_row.get("case_pair_cluster_id")
            if declared_cluster is not None and declared_cluster != result_row.cluster:
                raise PairedBootstrapError(
                    "balanced result cluster differs from constructed receipt"
                )
        if (
            strict_prefix != EXPECTED_STRICT_COUNT
            or len(rows) - strict_prefix != EXPECTED_CONSTRUCTED_COUNT
            or sum(row.label for row in rows[:strict_prefix])
            != EXPECTED_STRICT_POSITIVE
            or any(row.label for row in rows[strict_prefix:])
            or any(len(row.dependency_units) != 1 for row in rows[:strict_prefix])
            or any(len(row.dependency_units) != 2 for row in rows[strict_prefix:])
        ):
            raise PairedBootstrapError(
                "balanced1016 formal prefix/endpoint units differ"
            )
    return LoadedEvaluation(
        source_kind="real_balanced1016" if is_balanced else "real_strict547",
        source_path=path,
        source_sha256=_file_sha256(path),
        methods=methods,
        rows=tuple(rows),
        strict_prefix_count=int(strict_prefix),
        validation_thresholds=thresholds,
        no_test_tuning_evidence={
            "automated_nonuse_verified": True,
            "claim_no_human_cognitive_influence": False,
            "evaluation_history_disclosure": dict(disclosure),
            "winner_and_validation_threshold_frozen_before_real_open": True,
            "threshold_fit_performed_on_real": False,
            "real_geometry_ground_truth_read": False,
            "constructed_negatives_used_for_tuning": False if is_balanced else None,
        },
        input_files=(
            {
                "role": "real_evaluation",
                "path": str(path),
                "sha256": _file_sha256(path),
            },
        ),
        formal_evaluation=not compatibility_mode,
    )


@dataclass(frozen=True)
class _RankingPlan:
    labels: np.ndarray
    order: np.ndarray
    starts: np.ndarray

    @classmethod
    def build(cls, labels: np.ndarray, scores: np.ndarray) -> "_RankingPlan":
        order = np.argsort(-scores, kind="mergesort")
        ordered_scores = scores[order]
        starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
        return cls(labels=labels[order], order=order, starts=starts)

    def evaluate(self, weights: np.ndarray) -> Tuple[float, float]:
        ordered_weights = weights[self.order]
        positive_total = float(ordered_weights[self.labels].sum())
        negative_total = float(ordered_weights[~self.labels].sum())
        if positive_total <= 0.0 or negative_total <= 0.0:
            raise PairedBootstrapError("ranking metrics require both sampled classes")
        positive = np.add.reduceat(ordered_weights * self.labels, self.starts)
        negative = np.add.reduceat(ordered_weights * ~self.labels, self.starts)
        true_positive = np.cumsum(positive)
        false_positive = np.cumsum(negative)
        tpr = true_positive / positive_total
        fpr = false_positive / negative_total
        previous_tpr = np.r_[0.0, tpr[:-1]]
        previous_fpr = np.r_[0.0, fpr[:-1]]
        auroc = np.sum((fpr - previous_fpr) * (tpr + previous_tpr) * 0.5)
        denominator = true_positive + false_positive
        precision = np.divide(
            true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator > 0.0,
        )
        auprc = np.sum((tpr - previous_tpr) * precision)
        return float(auroc), float(auprc)


def _interval(
    values: Sequence[float], point: float, *, delta: bool = False
) -> Mapping[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise PairedBootstrapError("no finite bootstrap distribution is available")
    lower, upper = np.quantile(array, (0.025, 0.975))
    result: Dict[str, object] = {
        "point_estimate": float(point),
        "percentile_95_ci": [float(lower), float(upper)],
        "bootstrap_mean": float(array.mean()),
        "bootstrap_standard_error": (
            float(array.std(ddof=1)) if array.size > 1 else 0.0
        ),
        "valid_replicates": int(array.size),
    }
    if delta:
        result["probability_delta_gt_zero"] = float(np.mean(array > 0.0))
    return result


def _coverage(rows: Sequence[PairRow], methods: Sequence[str]) -> Mapping[str, object]:
    labels = np.asarray([row.label for row in rows], dtype=np.bool_)
    native = {
        method: np.asarray([row.valid[method] for row in rows], dtype=np.bool_)
        for method in methods
    }
    common = np.logical_and.reduce(tuple(native.values()))

    def view(mask: np.ndarray) -> Mapping[str, object]:
        return {
            "valid_count": int(mask.sum()),
            "population_count": int(mask.size),
            "valid_fraction": float(mask.mean()),
            "positive_valid_count": int((mask & labels).sum()),
            "positive_count": int(labels.sum()),
            "negative_valid_count": int((mask & ~labels).sum()),
            "negative_count": int((~labels).sum()),
        }

    return {
        "native_by_method": {method: view(mask) for method, mask in native.items()},
        "all_method_common_valid": view(common),
        "common_mask": common,
    }


def _point_metrics(
    rows: Sequence[PairRow], methods: Sequence[str], mask: np.ndarray
) -> Mapping[str, object]:
    selected = [row for row, keep in zip(rows, mask) if keep]
    labels = np.asarray([row.label for row in selected], dtype=np.bool_)
    if not labels.size or not labels.any() or labels.all():
        raise PairedBootstrapError("common-valid population must contain both classes")
    weights = np.ones(labels.size, dtype=np.float64)
    by_method = {}
    raw = {}
    for method in methods:
        scores = np.asarray(
            [row.probability[method] for row in selected], dtype=np.float64
        )
        raw[method] = _RankingPlan.build(labels, scores).evaluate(weights)
        by_method[method] = {"auroc": raw[method][0], "auprc": raw[method][1]}
    comparisons = {}
    for comparator in methods:
        if comparator == "full_n512":
            continue
        comparisons["full_n512_minus_" + comparator] = {
            "auroc": raw["full_n512"][0] - raw[comparator][0],
            "auprc": raw["full_n512"][1] - raw[comparator][1],
        }
    return {
        "row_count": len(selected),
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
        "methods": by_method,
        "comparisons": comparisons,
    }


def _dependency_index(
    rows: Sequence[PairRow],
) -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    units = tuple(sorted({unit for row in rows for unit in row.dependency_units}))
    if len(units) < 2:
        raise PairedBootstrapError(
            "endpoint bootstrap requires at least two source units"
        )
    lookup = {unit: index for index, unit in enumerate(units)}
    first = np.asarray(
        [lookup[row.dependency_units[0]] for row in rows], dtype=np.int64
    )
    second = np.asarray(
        [
            lookup[row.dependency_units[1]] if len(row.dependency_units) == 2 else -1
            for row in rows
        ],
        dtype=np.int64,
    )
    return units, first, second


def _pigeonhole_weights(
    unit_count: int,
    first: np.ndarray,
    second: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = rng.integers(0, unit_count, size=unit_count)
    multiplicity = np.bincount(selected, minlength=unit_count).astype(np.int64)
    weights = multiplicity[first].copy()
    paired = second >= 0
    weights[paired] *= multiplicity[second[paired]]
    return weights.astype(np.float64)


def _paired_bootstrap(
    rows: Sequence[PairRow],
    methods: Sequence[str],
    mask: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> Mapping[str, object]:
    selected = [row for row, keep in zip(rows, mask) if keep]
    labels = np.asarray([row.label for row in selected], dtype=np.bool_)
    dependency_units, first_unit, second_unit = _dependency_index(selected)
    plans = {
        method: _RankingPlan.build(
            labels,
            np.asarray([row.probability[method] for row in selected], dtype=np.float64),
        )
        for method in methods
    }
    unit_weights = np.ones(labels.size, dtype=np.float64)
    points = {method: plans[method].evaluate(unit_weights) for method in methods}
    draws: Dict[str, Dict[str, list[float]]] = {
        metric: {method: [] for method in methods} for metric in ("auroc", "auprc")
    }
    delta_draws: Dict[str, Dict[str, list[float]]] = {
        "full_n512_minus_" + comparator: {"auroc": [], "auprc": []}
        for comparator in methods
        if comparator != "full_n512"
    }
    skipped = 0
    for _ in range(replicates):
        weights = _pigeonhole_weights(
            len(dependency_units), first_unit, second_unit, rng
        )
        if not np.any(weights[labels] > 0.0) or not np.any(weights[~labels] > 0.0):
            skipped += 1
            continue
        values = {method: plans[method].evaluate(weights) for method in methods}
        for metric_index, metric in enumerate(("auroc", "auprc")):
            for method in methods:
                draws[metric][method].append(values[method][metric_index])
            for comparator in methods:
                if comparator == "full_n512":
                    continue
                name = "full_n512_minus_" + comparator
                delta_draws[name][metric].append(
                    values["full_n512"][metric_index] - values[comparator][metric_index]
                )
    if replicates - skipped <= 0:
        raise PairedBootstrapError("all bootstrap replicates were single-class")
    metrics: Dict[str, object] = {}
    for metric_index, metric in enumerate(("auroc", "auprc")):
        method_results = {
            method: _interval(draws[metric][method], points[method][metric_index])
            for method in methods
        }
        comparison_results = {}
        for comparator in methods:
            if comparator == "full_n512":
                continue
            name = "full_n512_minus_" + comparator
            point = points["full_n512"][metric_index] - points[comparator][metric_index]
            comparison_results[name] = _interval(
                delta_draws[name][metric], point, delta=True
            )
        metrics[metric] = {
            "methods": method_results,
            "paired_deltas": comparison_results,
        }
    return {
        "sampling_dependency_unit_count": len(dependency_units),
        "sampling_pair_cluster_count_descriptive": len(
            {row.cluster for row in selected}
        ),
        "bootstrap": "endpoint-unit_pigeonhole_product_multiplicity",
        "replicates_requested": replicates,
        "valid_replicates": replicates - skipped,
        "skipped_single_class_replicates": skipped,
        "metrics": metrics,
    }


def _population_sha(rows: Sequence[PairRow], mask: np.ndarray) -> str:
    semantics = sorted(
        (
            {
                "pair_id": row.pair_id,
                "label": row.label,
                "cluster": row.cluster,
                "dependency_units": list(row.dependency_units),
                "stratum": row.stratum,
            }
            for row, keep in zip(rows, mask)
            if keep
        ),
        key=lambda row: row["pair_id"],
    )
    return _canonical_sha256(semantics)


def _descriptive_population(
    rows: Sequence[PairRow], methods: Sequence[str]
) -> Mapping[str, object]:
    coverage = dict(_coverage(rows, methods))
    mask = coverage.pop("common_mask")
    point = _point_metrics(rows, methods, mask)
    return {
        "role": "descriptive_only_no_inferential_CI",
        "coverage": coverage,
        "common_population_sha256": _population_sha(rows, mask),
        "point_metrics": point,
    }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    keep = weights > 0.0
    values = values[keep]
    weights = weights[keep]
    if not values.size or not np.isfinite(values).all():
        raise PairedBootstrapError("weighted median has no finite observations")
    integer_weights = np.rint(weights).astype(np.int64)
    if not np.allclose(weights, integer_weights) or integer_weights.sum() <= 0:
        raise PairedBootstrapError("bootstrap weights must be non-negative integers")
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    cumulative = np.cumsum(integer_weights[order])
    total = int(cumulative[-1])
    left_position = (total - 1) // 2
    right_position = total // 2
    left = values[np.searchsorted(cumulative, left_position, side="right")]
    right = values[np.searchsorted(cumulative, right_position, side="right")]
    return float((left + right) * 0.5)


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """NumPy-linear quantile of the integer-multiplicity bootstrap sample."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    keep = weights > 0.0
    selected_values = values[keep]
    selected_weights = weights[keep]
    if not selected_values.size or not np.isfinite(selected_values).all():
        raise PairedBootstrapError("weighted quantile has no finite observations")
    integer_weights = np.rint(selected_weights).astype(np.int64)
    if (
        not np.allclose(selected_weights, integer_weights)
        or integer_weights.sum() <= 0
    ):
        raise PairedBootstrapError("bootstrap weights must be non-negative integers")
    order = np.argsort(selected_values, kind="mergesort")
    selected_values = selected_values[order]
    cumulative = np.cumsum(integer_weights[order])
    position = quantile * (int(cumulative[-1]) - 1)
    left_position = int(math.floor(position))
    right_position = int(math.ceil(position))
    left = selected_values[
        np.searchsorted(cumulative, left_position, side="right")
    ]
    right = selected_values[
        np.searchsorted(cumulative, right_position, side="right")
    ]
    return float(left + (right - left) * (position - left_position))


def _synthetic_geometry(
    loaded: LoadedEvaluation,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> Mapping[str, object]:
    rows = loaded.rows
    labels = np.asarray([row.label for row in rows], dtype=np.bool_)
    valid = np.asarray([row.valid["full_n512"] for row in rows], dtype=np.bool_)
    scores = np.asarray(
        [row.probability["full_n512"] for row in rows], dtype=np.float64
    )
    errors = np.full(len(rows), np.nan, dtype=np.float64)
    e_rmse = np.full(len(rows), np.nan, dtype=np.float64)
    hausdorff = np.full(len(rows), np.nan, dtype=np.float64)
    nte = np.full(len(rows), np.nan, dtype=np.float64)
    correspondence_names = (
        "strict_true_positive",
        "strict_predicted_count",
        "mutual_top1_true_positive",
        "mutual_top1_predicted_count",
        "target_count",
        "dustbin_correct",
        "dustbin_total",
    )
    correspondence = {
        name: np.zeros(len(rows), dtype=np.float64)
        for name in correspondence_names
    }
    assembly_success = {
        tolerance: np.zeros(len(rows), dtype=np.bool_)
        for tolerance in ASSEMBLY_EDGE_TOLERANCES
    }
    for index, row in enumerate(rows):
        assert row.geometry is not None
        value = row.geometry.get("translation_l2_px")
        if value is not None:
            errors[index] = float(value)
        registration = row.geometry["pairingnet_style_registration"]
        assert isinstance(registration, Mapping)
        if registration.get("e_rmse") is not None:
            e_rmse[index] = float(registration["e_rmse"])
            hausdorff[index] = float(registration["symmetric_hausdorff_px"])
            nte[index] = float(registration["normalized_translation_error"])
        counts = row.geometry["correspondence"]
        assert isinstance(counts, Mapping)
        for name in correspondence_names:
            correspondence[name][index] = float(counts[name])
        assembly = row.geometry["assembly_edge"]
        assert isinstance(assembly, Mapping)
        outcomes = assembly["true_positive_by_tolerance"]
        assert isinstance(outcomes, Mapping)
        for tolerance in ASSEMBLY_EDGE_TOLERANCES:
            assembly_success[tolerance][index] = bool(
                outcomes["at_{}".format(tolerance)]
            )
    translation_usable = labels & valid & np.isfinite(errors)
    registration_usable = (
        labels & np.isfinite(e_rmse) & np.isfinite(hausdorff) & np.isfinite(nte)
    )
    if not translation_usable.any():
        raise PairedBootstrapError("synthetic full_n512 has no usable translations")
    if int(registration_usable.sum()) != int(labels.sum()):
        raise PairedBootstrapError(
            "synthetic full_n512 identity-fallback registration is incomplete"
        )
    threshold = loaded.validation_thresholds["full_n512"]
    predicted_positive = valid & (scores >= threshold)
    tolerances = (2, 5, 8, 10)
    dependency_units, first_unit, second_unit = _dependency_index(rows)
    unit = np.ones(len(rows), dtype=np.float64)

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0.0 else 0.0

    def prf(tp: float, predicted: float, target: float) -> Mapping[str, float]:
        precision = ratio(tp, predicted)
        recall = ratio(tp, target)
        harmonic = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "harmonic_f1": harmonic,
        }

    def statistics(weights: np.ndarray) -> Optional[Mapping[str, object]]:
        positive_weight = float(weights[labels].sum())
        translation_weight = float(weights[translation_usable].sum())
        registration_weight = float(weights[registration_usable].sum())
        if positive_weight <= 0.0:
            return None
        predicted_edge_weight = float(weights[predicted_positive].sum())
        strict = prf(
            float(np.dot(weights, correspondence["strict_true_positive"])),
            float(np.dot(weights, correspondence["strict_predicted_count"])),
            float(np.dot(weights, correspondence["target_count"])),
        )
        mutual = prf(
            float(np.dot(weights, correspondence["mutual_top1_true_positive"])),
            float(np.dot(weights, correspondence["mutual_top1_predicted_count"])),
            float(np.dot(weights, correspondence["target_count"])),
        )
        dustbin_total = float(np.dot(weights, correspondence["dustbin_total"]))
        output: Dict[str, object] = {
            "joint": {
                tolerance: float(
                    weights[
                        labels
                        & valid
                        & predicted_positive
                        & np.isfinite(errors)
                        & (errors <= tolerance)
                    ].sum()
                    / positive_weight
                )
                for tolerance in tolerances
            },
            "translation_recall": {
                tolerance: float(
                    weights[
                        labels
                        & valid
                        & np.isfinite(errors)
                        & (errors <= tolerance)
                    ].sum()
                    / positive_weight
                )
                for tolerance in tolerances
            },
            "correspondence": {
                "strict": strict,
                "mutual": mutual,
                "dustbin_accuracy": ratio(
                    float(np.dot(weights, correspondence["dustbin_correct"])),
                    dustbin_total,
                ),
            },
            "registration_recall_lt4": float(
                weights[
                    labels
                    & registration_usable
                    & (e_rmse < PAIRINGNET_RR_THRESHOLD)
                ].sum()
                / positive_weight
            ),
            "assembly": {
                tolerance: prf(
                    float(weights[assembly_success[tolerance]].sum()),
                    predicted_edge_weight,
                    positive_weight,
                )
                for tolerance in ASSEMBLY_EDGE_TOLERANCES
            },
        }
        if translation_weight > 0.0:
            output["median"] = _weighted_median(
                errors[translation_usable], weights[translation_usable]
            )
            output["p90"] = _weighted_quantile(
                errors[translation_usable], weights[translation_usable], 0.9
            )
            output["translation_success"] = {
                tolerance: float(
                    weights[translation_usable & (errors <= tolerance)].sum()
                    / translation_weight
                )
                for tolerance in tolerances
            }
        else:
            output["median"] = None
            output["p90"] = None
            output["translation_success"] = None
        if registration_weight > 0.0:
            output["registration"] = {
                "mean_e_rmse": float(
                    np.dot(
                        weights[registration_usable], e_rmse[registration_usable]
                    )
                    / registration_weight
                ),
                "mean_symmetric_hausdorff_px": float(
                    np.dot(
                        weights[registration_usable],
                        hausdorff[registration_usable],
                    )
                    / registration_weight
                ),
                "mean_normalized_translation_error": float(
                    np.dot(weights[registration_usable], nte[registration_usable])
                    / registration_weight
                ),
            }
        else:
            output["registration"] = None
        return output

    point = statistics(unit)
    assert (
        point is not None
        and point["median"] is not None
        and point["p90"] is not None
        and point["registration"] is not None
    )
    median_draws: list[float] = []
    p90_draws: list[float] = []
    translation_draws = {tolerance: [] for tolerance in tolerances}
    translation_recall_draws = {tolerance: [] for tolerance in tolerances}
    joint_draws = {tolerance: [] for tolerance in tolerances}
    correspondence_draws = {
        method: {metric: [] for metric in ("precision", "recall", "harmonic_f1")}
        for method in ("strict", "mutual")
    }
    dustbin_draws: list[float] = []
    registration_draws = {
        name: []
        for name in (
            "mean_e_rmse",
            "mean_symmetric_hausdorff_px",
            "mean_normalized_translation_error",
        )
    }
    registration_recall_draws: list[float] = []
    assembly_draws = {
        tolerance: {
            metric: [] for metric in ("precision", "recall", "harmonic_f1")
        }
        for tolerance in ASSEMBLY_EDGE_TOLERANCES
    }
    positive_skipped = 0
    translation_skipped = 0
    registration_skipped = 0
    for _ in range(replicates):
        weights = _pigeonhole_weights(
            len(dependency_units), first_unit, second_unit, rng
        )
        value = statistics(weights)
        if value is None:
            positive_skipped += 1
            translation_skipped += 1
            registration_skipped += 1
            continue
        for tolerance in tolerances:
            joint_draws[tolerance].append(value["joint"][tolerance])
            translation_recall_draws[tolerance].append(
                value["translation_recall"][tolerance]
            )
        for method in ("strict", "mutual"):
            for metric in ("precision", "recall", "harmonic_f1"):
                correspondence_draws[method][metric].append(
                    value["correspondence"][method][metric]
                )
        dustbin_draws.append(value["correspondence"]["dustbin_accuracy"])
        registration_recall_draws.append(value["registration_recall_lt4"])
        for tolerance in ASSEMBLY_EDGE_TOLERANCES:
            for metric in ("precision", "recall", "harmonic_f1"):
                assembly_draws[tolerance][metric].append(
                    value["assembly"][tolerance][metric]
                )
        if value["median"] is None:
            translation_skipped += 1
        else:
            median_draws.append(float(value["median"]))
            p90_draws.append(float(value["p90"]))
            for tolerance in tolerances:
                translation_draws[tolerance].append(
                    value["translation_success"][tolerance]
                )
        if value["registration"] is None:
            registration_skipped += 1
        else:
            for name in registration_draws:
                registration_draws[name].append(value["registration"][name])

    point_counts = {
        name: int(correspondence[name].sum()) for name in correspondence_names
    }
    predicted_edge_count = int(predicted_positive.sum())
    target_edge_count = int(labels.sum())
    return {
        "scope": "full_n512_geometry_on_entire_sealed_synthetic_population",
        "bootstrap": "endpoint-unit_pigeonhole_product_multiplicity",
        "sampling_dependency_unit_count": len(dependency_units),
        "shared_draws_across_all_direct_geometry_metrics": True,
        "validation_threshold": threshold,
        "threshold_source": "frozen_validation_checkpoint_bound_artifact",
        "full_n512": {
            "translation_l2_px": {
                "definition": "positive and full decision-valid; conditional on a finite translation",
                "eligible_positive_count": int(labels.sum()),
                "valid_translation_count": int(translation_usable.sum()),
                "valid_translation_fraction": float(
                    translation_usable.sum() / labels.sum()
                ),
                "median": _interval(median_draws, float(point["median"])),
                "p90": _interval(p90_draws, float(point["p90"])),
                "recall_definition": (
                    "denominator is every positive pair; decision-invalid or missing "
                    "translation is a failure"
                ),
                "recall_by_tolerance": {
                    "recall_at_{}px".format(tolerance): _interval(
                        translation_recall_draws[tolerance],
                        point["translation_recall"][tolerance],
                    )
                    for tolerance in tolerances
                },
                "success_by_tolerance": {
                    "success_at_{}px".format(tolerance): _interval(
                        translation_draws[tolerance],
                        point["translation_success"][tolerance],
                    )
                    for tolerance in tolerances
                },
                "bootstrap_replicates_without_valid_translation": translation_skipped,
            },
            "correspondence": {
                "scope": (
                    "entire sealed population; invalid decisions preserve targets and "
                    "contribute no predicted matches/correct dustbins"
                ),
                "strict_dustbin_aware": {
                    "point_counts": {
                        "true_positive": point_counts["strict_true_positive"],
                        "predicted_count": point_counts[
                            "strict_predicted_count"
                        ],
                        "target_count": point_counts["target_count"],
                    },
                    **{
                        metric: _interval(
                            correspondence_draws["strict"][metric],
                            point["correspondence"]["strict"][metric],
                        )
                        for metric in ("precision", "recall", "harmonic_f1")
                    },
                },
                "mutual_top1": {
                    "point_counts": {
                        "true_positive": point_counts[
                            "mutual_top1_true_positive"
                        ],
                        "predicted_count": point_counts[
                            "mutual_top1_predicted_count"
                        ],
                        "target_count": point_counts["target_count"],
                    },
                    **{
                        metric: _interval(
                            correspondence_draws["mutual"][metric],
                            point["correspondence"]["mutual"][metric],
                        )
                        for metric in ("precision", "recall", "harmonic_f1")
                    },
                },
                "dustbin_accuracy": {
                    "point_counts": {
                        "correct": point_counts["dustbin_correct"],
                        "token_count": point_counts["dustbin_total"],
                    },
                    "accuracy": _interval(
                        dustbin_draws,
                        point["correspondence"]["dustbin_accuracy"],
                    ),
                },
            },
            "pairingnet_style_registration": {
                "compatibility_source": (
                    "PairingNet released matching_test.py, specialized to frozen "
                    "upright translation-only Dunhuang inputs"
                ),
                "eligible_positive_count": int(labels.sum()),
                "evaluated_positive_count": int(registration_usable.sum()),
                "valid_pose_count": int((labels & valid).sum()),
                "identity_fallback_count": int((labels & ~valid).sum()),
                "valid_pose_fraction": float(
                    (labels & valid).sum() / labels.sum()
                ),
                "invalid_pose_compatibility_fallback": (
                    "identity_translation_for_unconditional_official_style_aggregation"
                ),
                "e_rmse_definition": (
                    "sqrt(mean(per_correspondence_euclidean_distance)); official "
                    "compatibility definition, not conventional RMSE"
                ),
                "registration_recall_definition": "e_rmse_strictly_less_than_4",
                "hausdorff_definition": (
                    "max(directed_HD(transformed_source_seam,target_seam),"
                    "directed_HD(target_seam,transformed_source_seam))"
                ),
                "normalized_translation_error_definition": (
                    "identity-fallback translation_l2_px divided by the sum of ordered "
                    "N512 contour polygon areas after PairingNet int32 quantization"
                ),
                "mean_e_rmse": _interval(
                    registration_draws["mean_e_rmse"],
                    point["registration"]["mean_e_rmse"],
                ),
                "registration_recall_e_rmse_lt4": _interval(
                    registration_recall_draws,
                    point["registration_recall_lt4"],
                ),
                "mean_symmetric_hausdorff_px": _interval(
                    registration_draws["mean_symmetric_hausdorff_px"],
                    point["registration"]["mean_symmetric_hausdorff_px"],
                ),
                "mean_normalized_translation_error": _interval(
                    registration_draws["mean_normalized_translation_error"],
                    point["registration"]["mean_normalized_translation_error"],
                ),
                "bootstrap_replicates_without_evaluable_positive": (
                    registration_skipped
                ),
                "rotation_error": {
                    "status": "not_applicable_conditioned_upright_orientation",
                    "estimated_or_supervised": False,
                },
            },
            "assembly_edge_at_validation_threshold": {
                "definition": (
                    "predicted edge requires decision-valid and frozen-threshold pair "
                    "acceptance; true positive additionally requires adjacent GT and "
                    "translation L2 error at or below the pixel tolerance"
                ),
                "predicted_edge_count": predicted_edge_count,
                "target_edge_count": target_edge_count,
                "by_tolerance": {
                    "at_{}".format(tolerance): {
                        "point_counts": {
                            "true_positive": int(
                                assembly_success[tolerance].sum()
                            ),
                            "predicted_count": predicted_edge_count,
                            "target_count": target_edge_count,
                        },
                        **{
                            metric: _interval(
                                assembly_draws[tolerance][metric],
                                point["assembly"][tolerance][metric],
                            )
                            for metric in (
                                "precision",
                                "recall",
                                "harmonic_f1",
                            )
                        },
                    }
                    for tolerance in ASSEMBLY_EDGE_TOLERANCES
                },
            },
            "joint_success_at_validation_threshold": {
                "definition": (
                    "denominator is every positive pair; success requires full decision-valid, "
                    "score >= frozen validation threshold, and translation error <= tolerance"
                ),
                "denominator_positive_count": int(labels.sum()),
                "success_by_tolerance": {
                    "success_at_{}px".format(tolerance): _interval(
                        joint_draws[tolerance], point["joint"][tolerance]
                    )
                    for tolerance in tolerances
                },
                "bootstrap_replicates_without_positive": positive_skipped,
            },
        },
        "full_n512_minus_coarse_only": {
            "status": "not_applicable",
            "reason": (
                "coarse_only emits pair scores but no correspondence or 2D translation, "
                "so translation and joint geometric deltas are undefined"
            ),
        },
    }


def analyze_loaded(
    loaded: LoadedEvaluation,
    *,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> Mapping[str, object]:
    """Analyze one already-validated evaluation using a frozen population."""

    formal_methods = (
        tuple(REQUIRED_METHODS)
        + ("matched_mm_converged", "matched_mm_same_exposure_epoch5")
        + SAME_DATA_BENCHMARK_METHODS
    )
    if loaded.formal_evaluation and loaded.methods != formal_methods:
        raise PairedBootstrapError(
            "formal analysis cannot emit status without exact-six methods"
        )
    if loaded.formal_evaluation and loaded.source_kind not in {
        "sealed_synthetic",
        "real_balanced1016",
    }:
        raise PairedBootstrapError(
            "formal analysis requires sealed3000 or balanced1016 source"
        )
    if type(replicates) is not int or replicates <= 0:  # noqa: E721
        raise PairedBootstrapError("replicates must be a positive integer")
    if type(seed) is not int or seed < 0:  # noqa: E721
        raise PairedBootstrapError("seed must be a non-negative integer")
    if not loaded.rows:
        raise PairedBootstrapError("evaluation population is empty")
    coverage = dict(_coverage(loaded.rows, loaded.methods))
    common_mask = coverage.pop("common_mask")
    common_points = _point_metrics(loaded.rows, loaded.methods, common_mask)
    common_sha = _population_sha(loaded.rows, common_mask)
    result: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete_formal_exact_six"
            if loaded.formal_evaluation
            else "complete_non_formal_compatibility_analysis"
        ),
        "formal_evaluation": loaded.formal_evaluation,
        "compatibility_mode": not loaded.formal_evaluation,
        "input": {
            "source_kind": loaded.source_kind,
            "source_path": str(loaded.source_path),
            "source_sha256": loaded.source_sha256,
            "files": list(loaded.input_files),
        },
        "methods": list(loaded.methods),
        "protocol": {
            "formal_evaluation": loaded.formal_evaluation,
            "compatibility_mode": not loaded.formal_evaluation,
            "formal_status_claimed": loaded.formal_evaluation,
            "replicates": replicates,
            "seed": seed,
            "point_metrics": "equal-row tie-aware AUROC and average precision",
            "population": "single all-compared-method common-valid pair intersection",
            "sampling_unit": "unique source split-unit or real source case",
            "sampling": (
                "pigeonhole bootstrap: sample source units with replacement; "
                "weight one-unit rows by m(u) and two-unit rows by m(u)*m(v); "
                "apply identical row weights to every method"
            ),
            "confidence_interval": "two-sided 95% percentile interval",
            "delta_direction": "full_n512 minus comparator",
            "automated_no_test_or_real_tuning": loaded.no_test_tuning_evidence,
        },
        "coverage": coverage,
        "common_population": {
            "row_count": common_points["row_count"],
            "positive_count": common_points["positive_count"],
            "negative_count": common_points["negative_count"],
            "cluster_count": len(
                {row.cluster for row, keep in zip(loaded.rows, common_mask) if keep}
            ),
            "dependency_unit_count": len(
                {
                    unit
                    for row, keep in zip(loaded.rows, common_mask)
                    if keep
                    for unit in row.dependency_units
                }
            ),
            "semantic_sha256": common_sha,
            "sha256_payload": (
                "sorted [{pair_id,label,cluster,dependency_units,stratum}] canonical JSON"
            ),
        },
        "point_metrics": common_points,
    }
    seed_sequence = np.random.SeedSequence(seed)
    ranking_seed, geometry_seed = seed_sequence.spawn(2)
    if loaded.source_kind == "real_strict547":
        result["inference"] = {
            "status": "not_run_by_design",
            "reason": "strict547 has few negatives and is descriptive only",
        }
    else:
        result["paired_endpoint_pigeonhole_bootstrap"] = _paired_bootstrap(
            loaded.rows,
            loaded.methods,
            common_mask,
            replicates=replicates,
            rng=np.random.default_rng(ranking_seed),
        )
    if loaded.source_kind == "sealed_synthetic":
        result["synthetic_geometry"] = _synthetic_geometry(
            loaded,
            replicates=replicates,
            rng=np.random.default_rng(geometry_seed),
        )
    if loaded.source_kind == "real_balanced1016":
        assert loaded.strict_prefix_count is not None
        strict_rows = loaded.rows[: loaded.strict_prefix_count]
        result["strict547_descriptive"] = _descriptive_population(
            strict_rows, loaded.methods
        )
        result["balanced1016_inference_role"] = (
            "primary external paired endpoint-case pigeonhole bootstrap; "
            "constructed negatives remain non-GT distractors"
        )
    if loaded.source_kind.startswith("real_"):
        result["geometry"] = {
            "status": "not_read_or_reported_by_this_target_blind_pair_ranking_artifact",
            "reason": (
                "this target-blind pair-ranking artifact does not read or report "
                "correspondence/translation targets; geometry is evaluated by the "
                "independent post-prediction evaluator"
            ),
            "ground_truth_availability_claim_made": False,
        }
    return result


def analyze_synthetic(
    receipt_path: Path,
    *,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    compatibility_mode: bool = False,
) -> Mapping[str, object]:
    return analyze_loaded(
        load_sealed(receipt_path, compatibility_mode=compatibility_mode),
        replicates=replicates,
        seed=seed,
    )


def analyze_real(
    path: Path,
    *,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    compatibility_mode: bool = False,
) -> Mapping[str, object]:
    return analyze_loaded(
        load_real(path, compatibility_mode=compatibility_mode),
        replicates=replicates,
        seed=seed,
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise PairedBootstrapError("analysis output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + target.name + ".tmp-", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise PairedBootstrapError("analysis output already exists") from error
        parent_descriptor = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic-receipt", type=Path)
    source.add_argument("--real-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--compatibility-non-formal",
        action="store_true",
        help="accept only explicitly non-formal compatibility evaluations",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    source_path = (
        arguments.synthetic_receipt
        if arguments.synthetic_receipt is not None
        else arguments.real_json
    ).resolve(strict=True)
    output_path = arguments.output.expanduser().resolve()
    if output_path == source_path:
        raise PairedBootstrapError("analysis output may not overwrite its input")
    if arguments.synthetic_receipt is not None:
        try:
            output_path.relative_to(source_path.parent)
        except ValueError:
            pass
        else:
            raise PairedBootstrapError(
                "analysis output may not be written inside the sealed result"
            )
    if arguments.synthetic_receipt is not None:
        result = analyze_synthetic(
            arguments.synthetic_receipt,
            replicates=arguments.replicates,
            seed=arguments.seed,
            compatibility_mode=arguments.compatibility_non_formal,
        )
    else:
        result = analyze_real(
            arguments.real_json,
            replicates=arguments.replicates,
            seed=arguments.seed,
            compatibility_mode=arguments.compatibility_non_formal,
        )
    _write_json(output_path, result)
    print(str(output_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
