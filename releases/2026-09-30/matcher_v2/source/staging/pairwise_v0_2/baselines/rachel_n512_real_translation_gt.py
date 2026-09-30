"""Post-prediction real-Dunhuang translation-GT evaluation.

The target-blind real evaluator must run first.  This module freezes that
completed JSON result byte-for-byte, verifies its strict-547 prediction prefix
and frozen validation threshold, and only then opens label-side bounding boxes
and alpha masks.  It never restores a model and never performs a forward pass.

For a fragment whose local alpha image has parent-canvas origin ``o``, the
existing real evaluator applies one case-wide scale ``s``, tight-crops the
resized alpha, and centre-pads it.  If ``p`` is the pad start and ``m`` is the
resized alpha minimum, its parent-scaled-to-model offset is

``offset = p - m - s * o``.

Consequently the supervised A-to-B point-coordinate translation is
``offset_B - offset_A``.  To place the centred B mask into A's frame, the
assembly shift is its negative.  Absolute placement still has the usual global
translation gauge and requires one anchored fragment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from staging.pairwise_v0_2.baselines.rachel_matched_mm_evaluation import (
    MATCHED_METHODS,
)
from staging.pairwise_v0_2.baselines import (
    rachel_same_data_benchmark_eval_adapter as benchmark_adapter,
)
from staging.pairwise_v0_2.baselines import rachel_n512_real_external as real_eval
from staging.pairwise_v0_2.pairwise_data.real_dunhuang_representations import (
    RealDunhuangRepresentationError,
    RealExternalTestSpec,
    RealFragmentSpec,
    load_real_external_test_spec,
)
from experiments.rachel_n512_formal_30k import paired_cluster_bootstrap as formal_stats


SCHEMA_VERSION = "rachel-n512-real-translation-gt-posteval/1.0"
EXPECTED_MANIFEST_SHA256 = (
    "210ab081b2e70f39f35888a7b198458f6562847a2717d71758002c0225d32610"
)
EXPECTED_STRICT_PAIR_ORDER_SHA256 = (
    "f49232e468672466fee6236c9460101de8a5c0ef551866ef6815f62adf202ae5"
)
BOOTSTRAP_SEED = "rachel-real-translation-case-bootstrap-v1-fixed-20260901"
TOLERANCES_PX = (2, 5, 8, 10)
SOURCE_CONTACT_TOLERANCE_PX = 2.0
MIN_CONTACT_PIXELS = 8
MIN_CONTACT_BOUNDARY_FRACTION = 0.002
BOUNDARY_WORKING_SET_PIXELS = 4_000_000
MAX_BOUNDARY_POINTS_PER_FRAGMENT = 2_000_000
EXPECTED_METHODS = (
    "coarse_only",
    "full_n512",
    "matched_mm_converged",
    "matched_mm_same_exposure_epoch5",
)
COMPLETE_BENCHMARK_METHODS = EXPECTED_METHODS + benchmark_adapter.BENCHMARK_METHODS
TRANSLATION_METHODS = (
    "full_n512",
    benchmark_adapter.PAIRINGNET_METHOD_KEY,
    benchmark_adapter.SHREDDINGNET_METHOD_KEY,
)
EXPECTED_SCORE_SEMANTICS = {
    "coarse_only": "coarse_probability",
    "full_n512": "fused_probability",
    "matched_mm_converged": "historical_mm_probability",
    "matched_mm_same_exposure_epoch5": "historical_mm_probability",
    benchmark_adapter.PAIRINGNET_METHOD_KEY: "adapted_pair_probability",
    benchmark_adapter.SHREDDINGNET_METHOD_KEY: "adapted_pair_probability",
}


class RachelRealTranslationGTError(RuntimeError):
    """The prediction-freeze or post-prediction GT contract was violated."""


@dataclass(frozen=True)
class RealTranslationGTAuthority:
    """Exact population authority; custom values exist only for unit fixtures."""

    manifest_sha256: str = EXPECTED_MANIFEST_SHA256
    strict_pair_order_sha256: str = EXPECTED_STRICT_PAIR_ORDER_SHA256
    case_count: int = 445
    fragment_count: int = 938
    strict_pair_count: int = 547
    strict_positive_count: int = 508
    strict_negative_count: int = 39
    balanced_pair_count: int = 1016
    canvas_size: int = 800
    contour_cap: int = 512
    require_matched_controls: bool = True
    enforce_formal_combined_gate: bool = True

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_sha256, "authority manifest SHA-256")
        _require_sha256(
            self.strict_pair_order_sha256, "authority strict pair-order SHA-256"
        )
        for name in (
            "case_count",
            "fragment_count",
            "strict_pair_count",
            "strict_positive_count",
            "strict_negative_count",
            "balanced_pair_count",
            "canvas_size",
            "contour_cap",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        if self.strict_positive_count + self.strict_negative_count != (
            self.strict_pair_count
        ):
            raise ValueError("strict class counts do not sum to strict_pair_count")
        if self.balanced_pair_count < self.strict_pair_count:
            raise ValueError("balanced population cannot be shorter than strict")
        for name in ("require_matched_controls", "enforce_formal_combined_gate"):
            if type(getattr(self, name)) is not bool:  # noqa: E721
                raise TypeError(name + " must be bool")


@dataclass(frozen=True)
class FrozenRealPredictions:
    """Validated prediction-side state, held entirely in memory before GT open."""

    source_path: Path
    source_sha256: str
    source_payload: bytes
    strict_result: Mapping[str, object]
    strict_pairs: Tuple[Mapping[str, object], ...]
    balanced_pairs: Tuple[Mapping[str, object], ...]
    pair_ids: Tuple[str, ...]
    labels: Tuple[bool, ...]
    full_threshold: float
    full_checkpoint_sha256: str
    method_names: Tuple[str, ...]
    validation_thresholds: Mapping[str, float]
    checkpoint_sha256_by_method: Mapping[str, str]
    source_training_receipt_sha256: str
    source_matched_receipt_sha256: Optional[str]
    formal_evaluation: bool


@dataclass(frozen=True)
class _FrozenLabelAuthority:
    manifest_path: Path
    manifest_payload: bytes
    manifest_file_sha256: str
    manifest: Mapping[str, object]
    local_receipt_path: Path
    local_receipt_payload: bytes
    local_receipt_file_sha256: str
    local_receipt: Mapping[str, object]


@dataclass(frozen=True)
class _GTFragment:
    token: str
    case_uid: str
    fragment_id: int
    mask: np.ndarray
    model_mask: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    canvas_wh: Tuple[int, int]
    scale: float
    resized_tight_min_rc: Tuple[int, int]
    resized_tight_hw: Tuple[int, int]
    pad_start_rc: Tuple[int, int]
    parent_scaled_to_model_offset_rc: Tuple[float, float]
    model_points_rc: np.ndarray


@dataclass(frozen=True)
class _PositiveGT:
    pair_id: str
    case_uid: str
    fragment_a_token: str
    fragment_b_token: str
    translation_a_to_b_rc: Tuple[float, float]
    translation_a_to_b_xy_cartesian: Tuple[float, float]
    placement_shift_b_into_a_rc: Tuple[float, float]
    seam_qa: Mapping[str, object]
    source_seam_model_rc: np.ndarray
    target_seam_model_rc: np.ndarray
    source_contour_model_rc: np.ndarray
    target_contour_model_rc: np.ndarray


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RachelRealTranslationGTError(description + " must be lowercase SHA-256")
    return value


def _require_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RachelRealTranslationGTError(description + " must be an object")
    return value


def _require_probability(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RachelRealTranslationGTError(description + " must be numeric")
    output = float(value)
    if not math.isfinite(output) or not 0.0 <= output <= 1.0:
        raise RachelRealTranslationGTError(description + " must be in [0,1]")
    return output


def _require_finite_pair(value: object, description: str) -> Tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise RachelRealTranslationGTError(description + " must contain two numbers")
    try:
        output = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as error:
        raise RachelRealTranslationGTError(
            description + " must contain two numbers"
        ) from error
    if not all(math.isfinite(item) for item in output):
        raise RachelRealTranslationGTError(description + " must be finite")
    return output


def _read_frozen_json(
    path: Path,
) -> Tuple[Path, bytes, str, Mapping[str, object]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RachelRealTranslationGTError(
            "pair-only real evaluation must be a regular non-symlink file"
        )
    resolved = source.resolve(strict=True)
    try:
        payload = resolved.read_bytes()
        value = formal_stats._loads(
            payload.decode("utf-8"), "pair-only real evaluation"
        )
    except (OSError, UnicodeError, formal_stats.PairedBootstrapError) as error:
        raise RachelRealTranslationGTError(
            "pair-only real evaluation is not readable JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise RachelRealTranslationGTError(
            "pair-only real evaluation root must be an object"
        )
    return resolved, payload, _sha256_bytes(payload), value


def _method_names(
    strict: Mapping[str, object], *, compatibility_mode: bool
) -> Tuple[str, ...]:
    methods = _require_mapping(strict.get("methods"), "strict methods")
    if not compatibility_mode and set(methods) == set(COMPLETE_BENCHMARK_METHODS):
        return COMPLETE_BENCHMARK_METHODS
    if compatibility_mode and set(methods) == set(EXPECTED_METHODS):
        return EXPECTED_METHODS
    if compatibility_mode and set(methods) == set(COMPLETE_BENCHMARK_METHODS):
        return COMPLETE_BENCHMARK_METHODS
    raise RachelRealTranslationGTError(
        (
            "formal strict result must contain exact-six complete methods"
            if not compatibility_mode
            else "compatibility strict result must contain exact four or exact six methods"
        )
    )


def _validate_history_disclosure(protocol: Mapping[str, object]) -> None:
    disclosure = _require_mapping(
        protocol.get("evaluation_history_disclosure"),
        "real evaluation-history disclosure",
    )
    required_true = {
        "prior_epoch5_synthetic_test_completed": True,
        "prior_epoch5_synthetic_test_human_visible_before_continuation": True,
        "prior_epoch5_real_evaluator_started_then_stopped": True,
        "prior_convergence_time_synthetic_test_mask_morphology_review": True,
        "prior_convergence_time_synthetic_test_activity_used_for_corrosion_runtime_and_representation_QA": True,
    }
    required_false = {
        "prior_epoch5_synthetic_test_used_by_automated_checkpoint_selection": False,
        "prior_epoch5_synthetic_test_used_by_automated_threshold_fitting": False,
        "prior_epoch5_synthetic_test_used_by_automated_early_stopping": False,
        "prior_epoch5_synthetic_test_used_by_automated_scheduler": False,
        "claim_no_human_cognitive_influence": False,
        "prior_epoch5_real_result_formed_or_read": False,
        "prior_convergence_time_synthetic_test_labels_read": False,
        "prior_convergence_time_synthetic_test_model_scores_read": False,
        "prior_convergence_time_synthetic_test_activity_used_for_training_checkpoint_or_threshold_selection": False,
        "prior_convergence_time_synthetic_test_activity_influenced_fixed_corrosion_conditions": False,
        "real_data_accessed_in_that_activity": False,
        "claim_of_project_first_real_access": False,
    }
    if (
        any(disclosure.get(name) is not value for name, value in required_true.items())
        or any(
            disclosure.get(name) is not value
            for name, value in required_false.items()
        )
        or disclosure.get(
            "prior_convergence_time_synthetic_test_mask_sample_count"
        )
        != 500
    ):
        raise RachelRealTranslationGTError(
            "real evaluation-history disclosure is incomplete"
        )


def _validate_source_protocol(
    strict: Mapping[str, object], authority: RealTranslationGTAuthority, *,
    compatibility_mode: bool,
) -> None:
    protocol = _require_mapping(strict.get("protocol"), "strict protocol")
    required_true = (
        "winner_and_validation_threshold_frozen_before_real_open",
        "case_common_parent_canvas_scale",
        "tight_crop_then_centerpad",
        "formal_config_verified",
        "both_arms_validation_plateau_verified",
        "all_requested_winners_and_thresholds_frozen_before_current_real_open",
    )
    if any(protocol.get(name) is not True for name in required_true):
        raise RachelRealTranslationGTError(
            "pair-only source did not complete the frozen formal protocol"
        )
    required_false = (
        "per_fragment_independent_resize",
        "rgb_used",
        "bbox_or_gt_canvas_origin_exposed_to_model",
        "real_correspondence_or_translation_gt_read",
        "positive_direction_derived_or_bbox_read",
    )
    if any(protocol.get(name) is not False for name in required_false):
        raise RachelRealTranslationGTError(
            "pair-only source already exposed real geometry to inference"
        )
    if (
        protocol.get("canvas_size") != authority.canvas_size
        or protocol.get("contour_cap") != authority.contour_cap
        or protocol.get("pixel_source") != "PNG_alpha_ge_128_only"
    ):
        raise RachelRealTranslationGTError(
            "pair-only source preprocessing differs from translation authority"
        )
    if _method_names(strict, compatibility_mode=compatibility_mode) == COMPLETE_BENCHMARK_METHODS:
        benchmark_required = (
            "same_data_benchmark_winners_and_validation_thresholds_frozen_before_current_real_open",
            "same_data_benchmark_mask_only",
            "same_data_benchmark_upright_translation_only_adaptation",
        )
        if any(protocol.get(name) is not True for name in benchmark_required) or (
            protocol.get("same_data_benchmark_exact_reproduction_claimed")
            is not False
            or protocol.get(
                "shreddingnet_balanced_selected_list_claimed_as_native_CM_FM_SE_or_GA"
            )
            is not False
        ):
            raise RachelRealTranslationGTError(
                "same-data benchmark real adaptation disclosure is incomplete"
            )
    protocol_formal = protocol.get("formal_evaluation")
    protocol_compatibility = protocol.get("compatibility_mode")
    if compatibility_mode:
        if protocol_formal is not False or protocol_compatibility is not True:
            raise RachelRealTranslationGTError(
                "compatibility source must be explicitly non-formal"
            )
    elif (
        protocol_formal is not True
        or protocol_compatibility is not False
        or protocol.get("formal_exact_six_frozen_before_real_open") is not True
        or protocol.get("balanced1016_single_forward_required") is not True
        or protocol.get("formal_method_inventory")
        != list(COMPLETE_BENCHMARK_METHODS)
    ):
        raise RachelRealTranslationGTError(
            "formal source exact-six protocol disclosure is incomplete"
        )
    _validate_history_disclosure(protocol)


def _run_formal_combined_authority_gate(
    payload: bytes, source_sha256: str
) -> formal_stats.LoadedEvaluation:
    """Apply the publication statistics loader to immutable frozen bytes."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix="rachel-real-combined-frozen-", suffix=".json"
    )
    snapshot = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            loaded = formal_stats.load_real(snapshot)
        except formal_stats.PairedBootstrapError as error:
            raise RachelRealTranslationGTError(
                "formal combined-real authority gate failed: " + str(error)
            ) from error
        if (
            loaded.source_kind != "real_balanced1016"
            or loaded.source_sha256 != source_sha256
            or loaded.methods != COMPLETE_BENCHMARK_METHODS
            or loaded.strict_prefix_count != formal_stats.EXPECTED_STRICT_COUNT
            or len(loaded.rows) != formal_stats.EXPECTED_BALANCED_COUNT
            or set(loaded.validation_thresholds) != set(loaded.methods)
            or loaded.no_test_tuning_evidence.get(
                "winner_and_validation_threshold_frozen_before_real_open"
            )
            is not True
            or loaded.no_test_tuning_evidence.get("real_geometry_ground_truth_read")
            is not False
        ):
            raise RachelRealTranslationGTError(
                "formal combined-real loaded authority differs"
            )
        return loaded
    finally:
        snapshot.unlink(missing_ok=True)


def freeze_real_prediction_result(
    path: Path,
    *,
    authority: RealTranslationGTAuthority = RealTranslationGTAuthority(),
    compatibility_mode: bool = False,
) -> FrozenRealPredictions:
    """Freeze and fully validate predictions without accepting a GT path."""

    if type(compatibility_mode) is not bool:  # noqa: E721
        raise TypeError("compatibility_mode must be bool")
    source, source_payload, source_sha256, root = _read_frozen_json(path)
    expected_root_status = (
        real_eval.COMPATIBILITY_COMBINED_STATUS
        if compatibility_mode
        else real_eval.FORMAL_COMBINED_STATUS
    )
    if (
        root.get("schema_version") != real_eval.SCHEMA_VERSION
        or root.get("status") != expected_root_status
    ):
        raise RachelRealTranslationGTError(
            "source combined formal/compatibility status differs from requested mode"
        )
    strict = _require_mapping(root.get("strict_547"), "strict_547")
    balanced = _require_mapping(root.get("balanced_1016"), "balanced_1016")
    method_names = _method_names(strict, compatibility_mode=compatibility_mode)
    forward_contract = _require_mapping(
        root.get("forward_contract"), "combined forward contract"
    )
    if (
        forward_contract.get("forward_pair_count_per_arm")
        != authority.balanced_pair_count
        or forward_contract.get("strict_547_derived_from_exact_prediction_prefix")
        is not True
        or forward_contract.get("strict_pairs_forwarded_twice") is not False
        or forward_contract.get("all_methods_share_exact_strict_prediction_prefix")
        is not True
        or forward_contract.get("methods") != list(method_names)
    ):
        raise RachelRealTranslationGTError(
            "combined real single-forward authority differs"
        )
    expected_strict_status = (
        real_eval.COMPATIBILITY_STRICT_STATUS
        if compatibility_mode
        else "complete_strict_547_target_blind_external_test"
    )
    expected_balanced_status = (
        real_eval.COMPATIBILITY_BALANCED_STATUS
        if compatibility_mode
        else "complete_balanced_1016_target_blind_external_test"
    )
    if strict.get("status") != expected_strict_status:
        raise RachelRealTranslationGTError("strict real source is incomplete")
    if balanced.get("status") != expected_balanced_status:
        raise RachelRealTranslationGTError("balanced real source is incomplete")
    _validate_source_protocol(
        strict, authority, compatibility_mode=compatibility_mode
    )
    _validate_source_protocol(
        balanced, authority, compatibility_mode=compatibility_mode
    )
    balanced_method_names = _method_names(
        balanced, compatibility_mode=compatibility_mode
    )
    if method_names != balanced_method_names:
        raise RachelRealTranslationGTError("strict/balanced method inventories differ")
    if authority.require_matched_controls and tuple(MATCHED_METHODS) != (
        "matched_mm_converged",
        "matched_mm_same_exposure_epoch5",
    ):
        raise RachelRealTranslationGTError(
            "combined formal source lacks both matched controls"
        )

    strict_dataset = _require_mapping(strict.get("dataset"), "strict dataset")
    balanced_dataset = _require_mapping(balanced.get("dataset"), "balanced dataset")
    expected_dataset_values = {
        "manifest_sha256": authority.manifest_sha256,
        "pair_count": authority.strict_pair_count,
        "positive_count": authority.strict_positive_count,
        "negative_count": authority.strict_negative_count,
        "pair_order_sha256": authority.strict_pair_order_sha256,
    }
    if any(strict_dataset.get(name) != value for name, value in expected_dataset_values.items()):
        raise RachelRealTranslationGTError("strict dataset authority differs")
    if (
        balanced_dataset.get("pair_count") != authority.balanced_pair_count
        or balanced_dataset.get("strict_prefix_count") != authority.strict_pair_count
        or balanced_dataset.get("strict_prefix_preserved_exactly") is not True
    ):
        raise RachelRealTranslationGTError("balanced strict-prefix authority differs")

    strict_rows_value = strict.get("pairs")
    balanced_rows_value = balanced.get("pairs")
    if (
        not isinstance(strict_rows_value, list)
        or len(strict_rows_value) != authority.strict_pair_count
        or not isinstance(balanced_rows_value, list)
        or len(balanced_rows_value) != authority.balanced_pair_count
    ):
        raise RachelRealTranslationGTError("source pair-row counts differ")
    strict_rows = tuple(
        _require_mapping(row, "strict prediction row") for row in strict_rows_value
    )
    balanced_rows = tuple(
        _require_mapping(row, "balanced prediction row")
        for row in balanced_rows_value
    )
    balanced_prefix = balanced_rows[: authority.strict_pair_count]
    if strict_rows != balanced_prefix:
        raise RachelRealTranslationGTError(
            "strict rows differ from the balanced single-forward prefix"
        )

    pair_ids = []
    labels = []
    seen = set()
    for row in strict_rows:
        pair_id = row.get("pair_id")
        label = row.get("label")
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or pair_id in seen
            or type(label) is not bool  # noqa: E721
        ):
            raise RachelRealTranslationGTError("strict pair identity/label is invalid")
        seen.add(pair_id)
        pair_ids.append(pair_id)
        labels.append(bool(label))
    if (
        _canonical_sha256(tuple(pair_ids)) != authority.strict_pair_order_sha256
        or sum(labels) != authority.strict_positive_count
    ):
        raise RachelRealTranslationGTError("strict pair order or class count differs")

    methods = _require_mapping(strict.get("methods"), "strict methods")
    balanced_methods = _require_mapping(balanced.get("methods"), "balanced methods")
    thresholds: Dict[str, float] = {}
    checkpoint_sha_by_method: Dict[str, str] = {}
    for method in method_names:
        metadata = _require_mapping(methods.get(method), method + " metadata")
        balanced_metadata = _require_mapping(
            balanced_methods.get(method), method + " balanced metadata"
        )
        expected_semantics = EXPECTED_SCORE_SEMANTICS[method]
        for description, value in (
            ("strict", metadata),
            ("balanced", balanced_metadata),
        ):
            declared_semantics = value.get("score_semantics")
            if (
                method in MATCHED_METHODS
                and declared_semantics != expected_semantics
            ) or (
                method not in MATCHED_METHODS
                and declared_semantics not in {None, expected_semantics}
            ):
                raise RachelRealTranslationGTError(
                    method + " " + description + " score semantics differs"
                )
        winner = _require_mapping(metadata.get("winner"), method + " winner")
        balanced_winner = _require_mapping(
            balanced_metadata.get("winner"), method + " balanced winner"
        )
        if winner != balanced_winner:
            raise RachelRealTranslationGTError(
                method + " strict/balanced winner receipts differ"
            )
        threshold_receipt = _require_mapping(
            winner.get("validation_threshold"), method + " threshold"
        )
        threshold = _require_probability(
            threshold_receipt.get("threshold"), method + " frozen threshold"
        )
        source_split = threshold_receipt.get("source_split")
        normalized_source = (
            source_split.strip().casefold()
            if isinstance(source_split, str)
            else ""
        )
        validation_only = normalized_source in {"val", "validation", "dev"} or (
            ("validation" in normalized_source or normalized_source.startswith("val_"))
            and "train" not in normalized_source
            and "test" not in normalized_source
        )
        if method in benchmark_adapter.BENCHMARK_METHODS:
            stages = _require_mapping(
                winner.get("checkpoint_sha256_by_stage"),
                method + " checkpoint stages",
            )
            if not stages:
                raise RachelRealTranslationGTError(
                    method + " checkpoint stage inventory is empty"
                )
            stage_hashes = tuple(
                _require_sha256(value, method + " checkpoint stage SHA-256")
                for value in stages.values()
            )
            _require_sha256(
                winner.get("freeze_authority_sha256"),
                method + " freeze authority SHA-256",
            )
            checkpoint_sha = _require_sha256(
                threshold_receipt.get("checkpoint_sha256"),
                method + " threshold checkpoint SHA-256",
            )
            if (
                checkpoint_sha not in stage_hashes
                or metadata.get("native_cm_fm_se_or_ga_claimed") is not False
            ):
                raise RachelRealTranslationGTError(
                    method + " adaptation/checkpoint binding differs"
                )
        else:
            if (
                type(winner.get("epoch")) is not int  # noqa: E721
                or int(winner["epoch"]) <= 0
            ):
                raise RachelRealTranslationGTError(
                    method + " winner epoch is invalid"
                )
            checkpoint_sha = _require_sha256(
                winner.get("checkpoint_sha256"), method + " checkpoint SHA-256"
            )
        if (
            not validation_only
            or threshold_receipt.get("fit_method")
            != "maximize_cluster_balanced_f1"
            or threshold_receipt.get("checkpoint_sha256") != checkpoint_sha
        ):
            raise RachelRealTranslationGTError(
                method + " threshold is not validation-only/checkpoint-bound"
            )
        thresholds[method] = threshold
        checkpoint_sha_by_method[method] = checkpoint_sha
    threshold = thresholds["full_n512"]
    checkpoint_sha256 = checkpoint_sha_by_method["full_n512"]
    for row in strict_rows:
        row_methods = _require_mapping(row.get("methods"), "strict row methods")
        source_case_uids = row.get("source_case_uids")
        if (
            not isinstance(source_case_uids, list)
            or len(source_case_uids) != 1
            or not isinstance(source_case_uids[0], str)
            or not source_case_uids[0]
        ):
            raise RachelRealTranslationGTError(
                "strict row must name exactly one source case"
            )
        if set(row_methods) != set(method_names):
            raise RachelRealTranslationGTError(
                "strict row method inventory is not exact"
            )
        for method in method_names:
            method_row = _require_mapping(
                row_methods.get(method), "strict row method " + method
            )
            probability = _require_probability(
                method_row.get("probability"), method + " probability"
            )
            valid = method_row.get("valid")
            if type(valid) is not bool:  # noqa: E721
                raise RachelRealTranslationGTError(method + " valid must be bool")
            expected_semantics = EXPECTED_SCORE_SEMANTICS[method]
            declared_semantics = method_row.get("score_semantics")
            if (
                method in MATCHED_METHODS
                and declared_semantics != expected_semantics
            ) or (
                method not in MATCHED_METHODS
                and declared_semantics not in {None, expected_semantics}
            ):
                raise RachelRealTranslationGTError(
                    method + " row score semantics differs"
                )
            decision = method_row.get("decision_at_frozen_validation_threshold")
            expected_decision = (
                probability >= thresholds[method] if valid else None
            )
            if decision is not expected_decision:
                raise RachelRealTranslationGTError(
                    method + " stored decision differs from frozen threshold"
                )
            if method == "full_n512":
                translation = method_row.get("translation_hat_rc_unsupervised")
                if valid:
                    _require_finite_pair(translation, "full_n512 translation")
                elif translation is not None:
                    _require_finite_pair(
                        translation, "invalid full_n512 translation diagnostic"
                    )
            elif method in benchmark_adapter.BENCHMARK_METHODS:
                translation = method_row.get("translation_hat_rc_unsupervised")
                if translation is not None:
                    _require_finite_pair(translation, method + " translation")
            elif method_row.get("translation_hat_rc_unsupervised") is not None:
                raise RachelRealTranslationGTError(
                    method + " must not emit a translation"
                )

    source_training = _require_mapping(
        strict.get("source_training_run"), "source training run"
    )
    balanced_source_training = _require_mapping(
        balanced.get("source_training_run"), "balanced source training run"
    )
    root_source_training = _require_mapping(
        root.get("source_training_run"), "wrapper source training run"
    )
    if (
        source_training != balanced_source_training
        or source_training != root_source_training
        or source_training.get("status_at_open")
        != "complete_train_validation_only"
        or source_training.get("both_arms_validation_plateau_verified") is not True
        or source_training.get("formal_config_verified") is not True
    ):
        raise RachelRealTranslationGTError(
            "strict/balanced/wrapper training authority differs"
        )
    training_receipt_sha = _require_sha256(
        source_training.get("receipt_sha256"), "source training receipt SHA-256"
    )
    source_matched = strict.get("source_matched_mm_training_run")
    balanced_source_matched = balanced.get("source_matched_mm_training_run")
    root_source_matched = root.get("source_matched_mm_training_run")
    matched_receipt_sha = None
    if source_matched is not None:
        matched_mapping = _require_mapping(source_matched, "source matched run")
        alignment_key = "training_alignment_hash_evidence"
        nested_alignment = matched_mapping.get(alignment_key)
        root_alignment = root.get("matched_training_alignment_hash_evidence")
        if (
            source_matched == balanced_source_matched
            and isinstance(root_source_matched, Mapping)
            and alignment_key in matched_mapping
            and alignment_key not in root_source_matched
            and set(matched_mapping) == set(root_source_matched) | {alignment_key}
            and isinstance(nested_alignment, Mapping)
            and root_alignment == nested_alignment
        ):
            normalized_root_source_matched = dict(root_source_matched)
            normalized_root_source_matched[alignment_key] = nested_alignment
            if normalized_root_source_matched == source_matched:
                root_source_matched = normalized_root_source_matched
        if (
            source_matched != balanced_source_matched
            or source_matched != root_source_matched
            or matched_mapping.get("status_at_open")
            != "complete_train_validation_only"
            or matched_mapping.get(
                "converged_and_epoch5_winners_frozen_before_current_real_open"
            )
            is not True
            or matched_mapping.get("formal_config_verified") is not True
        ):
            raise RachelRealTranslationGTError(
                "strict/balanced/wrapper matched authority differs"
            )
        matched_receipt_sha = _require_sha256(
            matched_mapping.get("receipt_sha256"),
            "source matched receipt SHA-256",
        )
    elif authority.require_matched_controls:
        raise RachelRealTranslationGTError("source matched run receipt is missing")

    source_benchmarks = strict.get("source_same_data_benchmark_training_runs")
    if method_names == COMPLETE_BENCHMARK_METHODS:
        benchmark_mapping = _require_mapping(
            source_benchmarks, "source same-data benchmark runs"
        )
        if (
            source_benchmarks
            != balanced.get("source_same_data_benchmark_training_runs")
            or source_benchmarks
            != root.get("source_same_data_benchmark_training_runs")
            or benchmark_mapping.get(
                "all_winners_and_validation_thresholds_frozen_before_current_real_open"
            )
            is not True
        ):
            raise RachelRealTranslationGTError(
                "strict/balanced/wrapper same-data benchmark authority differs"
            )
        benchmark_methods = _require_mapping(
            benchmark_mapping.get("methods"), "same-data benchmark methods"
        )
        if set(benchmark_methods) != set(benchmark_adapter.BENCHMARK_METHODS):
            raise RachelRealTranslationGTError(
                "same-data benchmark provenance inventory differs"
            )
        for method in benchmark_adapter.BENCHMARK_METHODS:
            provenance = _require_mapping(
                benchmark_methods.get(method), method + " benchmark provenance"
            )
            if (
                provenance.get("all_winners_and_validation_threshold_frozen")
                is not True
                or provenance.get("sealed_synthetic_accessed_during_freeze")
                is not False
                or provenance.get("real_data_accessed_during_freeze") is not False
            ):
                raise RachelRealTranslationGTError(
                    method + " frozen provenance disclosure differs"
                )
    elif source_benchmarks is not None:
        raise RachelRealTranslationGTError(
            "same-data benchmark authority exists without both benchmark methods"
        )

    strict_protocol = _require_mapping(strict.get("protocol"), "strict protocol")
    balanced_protocol = _require_mapping(
        balanced.get("protocol"), "balanced protocol"
    )
    if strict_protocol.get("evaluation_history_disclosure") != (
        balanced_protocol.get("evaluation_history_disclosure")
    ):
        raise RachelRealTranslationGTError(
            "strict/balanced evaluation-history disclosures differ"
        )
    if authority.enforce_formal_combined_gate and not compatibility_mode:
        _run_formal_combined_authority_gate(source_payload, source_sha256)

    return FrozenRealPredictions(
        source_path=source,
        source_sha256=source_sha256,
        source_payload=source_payload,
        strict_result=strict,
        strict_pairs=strict_rows,
        balanced_pairs=balanced_rows,
        pair_ids=tuple(pair_ids),
        labels=tuple(labels),
        full_threshold=threshold,
        full_checkpoint_sha256=checkpoint_sha256,
        method_names=method_names,
        validation_thresholds=thresholds,
        checkpoint_sha256_by_method=checkpoint_sha_by_method,
        source_training_receipt_sha256=training_receipt_sha,
        source_matched_receipt_sha256=matched_receipt_sha,
        formal_evaluation=not compatibility_mode,
    )


def _validate_output_path_before_gt(
    output_path: Path,
    *,
    prediction_path: Path,
    forbidden_roots: Sequence[Path],
) -> Path:
    target = Path(output_path)
    if target.exists() or target.is_symlink():
        raise RachelRealTranslationGTError("refusing to overwrite post-evaluation output")
    resolved = target.expanduser().resolve()
    if resolved == prediction_path.resolve(strict=True):
        raise RachelRealTranslationGTError("output aliases pair-only prediction input")
    for value in forbidden_roots:
        root_value = Path(value).expanduser()
        try:
            root = root_value.resolve(strict=True)
            resolved.relative_to(root)
        except ValueError:
            continue
        except (OSError, RuntimeError) as error:
            raise RachelRealTranslationGTError(
                "cannot validate a forbidden output root"
            ) from error
        raise RachelRealTranslationGTError(
            "post-evaluation output may not be inside real label/source roots"
        )
    return resolved


def _positive_int_pair(value: object, description: str) -> Tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)  # noqa: E721
    ):
        raise RachelRealTranslationGTError(
            description + " must contain two positive integers"
        )
    return int(value[0]), int(value[1])


def _bbox(value: object, description: str) -> Tuple[int, int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(type(item) is not int for item in value)  # noqa: E721
    ):
        raise RachelRealTranslationGTError(description + " must be four integers")
    x1, y1, x2, y2 = (int(item) for item in value)
    if x2 <= x1 or y2 <= y1:
        raise RachelRealTranslationGTError(description + " must be positive")
    return x1, y1, x2, y2


def _read_label_authority_file_once(
    path: Path, description: str
) -> Tuple[Path, bytes, str, Mapping[str, object]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RachelRealTranslationGTError(
            description + " must be a regular non-symlink file"
        )
    resolved = source.resolve(strict=True)
    try:
        payload = resolved.read_bytes()
        value = formal_stats._loads(payload.decode("utf-8"), description)
    except (OSError, UnicodeError, formal_stats.PairedBootstrapError) as error:
        raise RachelRealTranslationGTError(description + " is not strict JSON") from error
    if not isinstance(value, Mapping):
        raise RachelRealTranslationGTError(description + " root must be an object")
    return resolved, payload, _sha256_bytes(payload), value


def _freeze_label_authority_after_predictions(
    manifest_path: Path,
    local_path_receipt_path: Path,
    *,
    authority: RealTranslationGTAuthority,
) -> _FrozenLabelAuthority:
    manifest_path, manifest_payload, manifest_file_sha, manifest = (
        _read_label_authority_file_once(manifest_path, "real manifest")
    )
    receipt_path, receipt_payload, receipt_file_sha, receipt = (
        _read_label_authority_file_once(
            local_path_receipt_path, "real local-path receipt"
        )
    )
    if manifest.get("schema_version") != (
        "pairwise-v0.2-real-external-test/0.1"
    ):
        raise RachelRealTranslationGTError("real manifest schema differs")
    declared_manifest_sha = _require_sha256(
        manifest.get("manifest_sha256"), "real manifest canonical SHA-256"
    )
    manifest_payload_without_sha = dict(manifest)
    manifest_payload_without_sha.pop("manifest_sha256", None)
    if (
        declared_manifest_sha != _canonical_sha256(manifest_payload_without_sha)
        or declared_manifest_sha != authority.manifest_sha256
    ):
        raise RachelRealTranslationGTError("real manifest canonical authority differs")
    if (
        receipt.get("schema_version")
        != "pairwise-v0.2-real-external-test-local-receipt/0.1"
        or receipt.get("portable_manifest_sha256") != declared_manifest_sha
    ):
        raise RachelRealTranslationGTError(
            "real local-path receipt is not bound to the manifest"
        )
    return _FrozenLabelAuthority(
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        manifest_file_sha256=manifest_file_sha,
        manifest=manifest,
        local_receipt_path=receipt_path,
        local_receipt_payload=receipt_payload,
        local_receipt_file_sha256=receipt_file_sha,
        local_receipt=receipt,
    )


@contextmanager
def _frozen_label_authority_snapshot(
    frozen: _FrozenLabelAuthority,
):
    """Expose immutable byte snapshots to the existing spec loader."""

    with tempfile.TemporaryDirectory(prefix="rachel-real-gt-authority-") as directory:
        root = Path(directory)
        manifest = root / "manifest.json"
        receipt = root / "local_path_receipt.json"
        for path, payload in (
            (manifest, frozen.manifest_payload),
            (receipt, frozen.local_receipt_payload),
        ):
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        yield manifest, receipt


def _verify_label_authority_unchanged(frozen: _FrozenLabelAuthority) -> None:
    for path, expected, description in (
        (
            frozen.manifest_path,
            frozen.manifest_file_sha256,
            "real manifest",
        ),
        (
            frozen.local_receipt_path,
            frozen.local_receipt_file_sha256,
            "real local-path receipt",
        ),
    ):
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise RachelRealTranslationGTError(
                description + " changed after its bytes were frozen"
            )


def _label_geometry_from_manifest(
    manifest: Mapping[str, object],
    spec: RealExternalTestSpec,
    *,
    authority: RealTranslationGTAuthority,
) -> Tuple[
    Mapping[Tuple[str, int], Tuple[int, int, int, int]],
    Mapping[str, Tuple[int, int]],
    Mapping[str, Mapping[str, object]],
]:
    if manifest.get("manifest_sha256") != authority.manifest_sha256:
        raise RachelRealTranslationGTError("real manifest authority SHA-256 differs")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise RachelRealTranslationGTError("real manifest cases are missing")
    selected_cases = {fragment.case_uid for fragment in spec.fragments}
    bboxes: Dict[Tuple[str, int], Tuple[int, int, int, int]] = {}
    canvases: Dict[str, Tuple[int, int]] = {}
    pair_authority: Dict[str, Mapping[str, object]] = {}
    spec_pair_by_key = {
        (pair.case_uid, pair.fragment_a_id, pair.fragment_b_id): pair
        for pair in spec.pairs
    }
    for raw_case in cases:
        case = _require_mapping(raw_case, "real case")
        case_uid = case.get("case_uid")
        if case_uid not in selected_cases:
            continue
        metadata = _require_mapping(case.get("numeric_metadata"), "case metadata")
        canvas = _positive_int_pair(metadata.get("canvas_wh"), "case canvas_wh")
        if case_uid in canvases:
            raise RachelRealTranslationGTError("selected case UID is duplicated")
        canvases[str(case_uid)] = canvas
        fragments = case.get("fragments")
        if not isinstance(fragments, list):
            raise RachelRealTranslationGTError("case fragments are missing")
        for raw_fragment in fragments:
            fragment = _require_mapping(raw_fragment, "real fragment")
            fragment_id = fragment.get("fragment_id")
            if type(fragment_id) is not int:  # noqa: E721
                raise RachelRealTranslationGTError("fragment ID must be integer")
            value = _bbox(fragment.get("bbox_xyxy"), "fragment bbox")
            size = _positive_int_pair(fragment.get("size_wh"), "fragment size_wh")
            if size != (value[2] - value[0], value[3] - value[1]):
                raise RachelRealTranslationGTError(
                    "fragment PNG size and bbox extent differ"
                )
            key = (str(case_uid), int(fragment_id))
            if key in bboxes:
                raise RachelRealTranslationGTError("fragment bbox is duplicated")
            bboxes[key] = value
        raw_pairs = case.get("pair_labels")
        if not isinstance(raw_pairs, list):
            raise RachelRealTranslationGTError("case pair labels are missing")
        for raw_pair in raw_pairs:
            pair = _require_mapping(raw_pair, "real pair label")
            first = pair.get("fragment_a")
            second = pair.get("fragment_b")
            if type(first) is not int or type(second) is not int:  # noqa: E721
                raise RachelRealTranslationGTError("real pair endpoints are invalid")
            spec_pair = spec_pair_by_key.get((str(case_uid), int(first), int(second)))
            if spec_pair is None:
                raise RachelRealTranslationGTError(
                    "manifest pair is absent from selected real spec"
                )
            label_source = pair.get("label_source")
            if label_source not in {
                "gt_numeric_bbox_plus_fragment_alpha",
                "curated_two_fragment_conjunction_category",
            }:
                raise RachelRealTranslationGTError(
                    "real pair label source is unsupported"
                )
            if spec_pair.pair_id in pair_authority:
                raise RachelRealTranslationGTError("real pair authority is duplicated")
            pair_authority[spec_pair.pair_id] = {
                "label_source": label_source,
                "geometry_diagnostic_label": pair.get(
                    "geometry_diagnostic_label", pair.get("label")
                ),
                "reason": pair.get("reason"),
            }
    expected_keys = {(row.case_uid, row.fragment_id) for row in spec.fragments}
    if (
        set(bboxes) != expected_keys
        or set(canvases) != selected_cases
        or set(pair_authority) != {pair.pair_id for pair in spec.pairs}
    ):
        raise RachelRealTranslationGTError("label geometry coverage differs from spec")
    return bboxes, canvases, pair_authority


def _boundary_points_parent_rc(
    mask: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]
) -> np.ndarray:
    """Mirror builder 4-neighbour boundaries with bounded working memory."""

    value = np.asarray(mask, dtype=np.bool_)
    if value.ndim != 2 or not bool(value.any()):
        raise RachelRealTranslationGTError("boundary mask must be nonempty 2-D")
    height, width = value.shape
    # The collection contains plates above 270M pixels.  A full-sized chain of
    # boolean temporaries can exceed a GB; scan row blocks while preserving the
    # exact builder definition.
    rows_out = []
    columns_out = []
    boundary_count = 0
    rows_per_block = max(1, BOUNDARY_WORKING_SET_PIXELS // max(1, width))
    for start in range(0, height, rows_per_block):
        end = min(height, start + rows_per_block)
        center = value[start:end]
        boundary = center.copy()
        if start == 0:
            boundary[0] = False
        else:
            boundary[0] &= value[start - 1]
        if len(boundary) > 1:
            boundary[1:] &= center[:-1]
        if end == height:
            boundary[-1] = False
        else:
            boundary[-1] &= value[end]
        if len(boundary) > 1:
            boundary[:-1] &= center[1:]
        if width < 3:
            boundary[:] = False
        else:
            boundary[:, 0] = False
            boundary[:, -1] = False
            boundary[:, 1:] &= center[:, :-1]
            boundary[:, :-1] &= center[:, 1:]
        np.logical_not(boundary, out=boundary)
        np.logical_and(boundary, center, out=boundary)
        rows, columns = np.nonzero(boundary)
        if len(rows):
            boundary_count += len(rows)
            if boundary_count > MAX_BOUNDARY_POINTS_PER_FRAGMENT:
                raise RachelRealTranslationGTError(
                    "alpha boundary exceeds the fixed QA point limit"
                )
            rows_out.append(rows + start)
            columns_out.append(columns)
    if not rows_out:
        raise RachelRealTranslationGTError("alpha boundary is empty")
    rows = np.concatenate(rows_out)
    columns = np.concatenate(columns_out)
    x1, y1, _, _ = bbox_xyxy
    return np.column_stack((rows + y1, columns + x1)).astype(np.float64)


def _one_way_nearest(
    first_parent_rc: np.ndarray,
    second_parent_rc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    distance, neighbor = cKDTree(second_parent_rc).query(first_parent_rc, k=1)
    return np.asarray(distance, dtype=np.float64), np.asarray(
        neighbor, dtype=np.int64
    )


def _summary(values: np.ndarray) -> Mapping[str, Optional[float]]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _build_fragment_gt(
    fragment: RealFragmentSpec,
    *,
    mask: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    canvas_wh: Tuple[int, int],
    canvas_size: int,
    contour_cap: int,
) -> _GTFragment:
    x1, y1, x2, y2 = bbox_xyxy
    if mask.shape != (y2 - y1, x2 - x1):
        raise RachelRealTranslationGTError("decoded alpha shape differs from bbox")
    scale = canvas_size / float(max(canvas_wh))
    scaled = real_eval._resize_bool_mask(mask, scale)
    rows, columns = np.nonzero(scaled)
    minimum = np.asarray((rows.min(), columns.min()), dtype=np.int64)
    maximum = np.asarray((rows.max() + 1, columns.max() + 1), dtype=np.int64)
    tight_hw = maximum - minimum
    if np.any(tight_hw > canvas_size):
        raise RachelRealTranslationGTError("real fragment cannot fit model canvas")
    pad = (canvas_size - tight_hw) // 2
    origin_rc = np.asarray((y1, x1), dtype=np.float64)
    offset = pad.astype(np.float64) - minimum.astype(np.float64) - scale * origin_rc
    model_mask = np.zeros((canvas_size, canvas_size), dtype=np.bool_)
    tight = scaled[minimum[0] : maximum[0], minimum[1] : maximum[1]]
    model_mask[
        pad[0] : pad[0] + tight_hw[0], pad[1] : pad[1] + tight_hw[1]
    ] = tight
    points, valid = real_eval.extract_ordered_outer_contour(
        model_mask, cap=contour_cap
    )
    if not bool(np.all(valid)):
        raise RachelRealTranslationGTError("prepared real contour validity changed")
    token = "{}/fragment/{}".format(fragment.case_uid, fragment.fragment_id)
    return _GTFragment(
        token=token,
        case_uid=fragment.case_uid,
        fragment_id=fragment.fragment_id,
        mask=np.ascontiguousarray(mask, dtype=np.bool_),
        model_mask=np.ascontiguousarray(model_mask, dtype=np.bool_),
        bbox_xyxy=bbox_xyxy,
        canvas_wh=canvas_wh,
        scale=scale,
        resized_tight_min_rc=(int(minimum[0]), int(minimum[1])),
        resized_tight_hw=(int(tight_hw[0]), int(tight_hw[1])),
        pad_start_rc=(int(pad[0]), int(pad[1])),
        parent_scaled_to_model_offset_rc=(float(offset[0]), float(offset[1])),
        model_points_rc=np.asarray(points, dtype=np.float64),
    )


def _source_to_model_points(
    parent_points_rc: np.ndarray, fragment: _GTFragment
) -> np.ndarray:
    return (
        np.asarray(parent_points_rc, dtype=np.float64) * fragment.scale
        + np.asarray(fragment.parent_scaled_to_model_offset_rc, dtype=np.float64)
    )


def _positive_gt(
    first: _GTFragment,
    second: _GTFragment,
    pair_id: str,
    pair_authority: Mapping[str, object],
) -> _PositiveGT:
    if first.case_uid != second.case_uid or first.canvas_wh != second.canvas_wh:
        raise RachelRealTranslationGTError("positive real pair must share one case")
    if not math.isclose(first.scale, second.scale, rel_tol=0.0, abs_tol=0.0):
        raise RachelRealTranslationGTError("positive endpoints have different scales")
    first_offset = np.asarray(
        first.parent_scaled_to_model_offset_rc, dtype=np.float64
    )
    second_offset = np.asarray(
        second.parent_scaled_to_model_offset_rc, dtype=np.float64
    )
    translation = second_offset - first_offset
    reverse = first_offset - second_offset
    if not np.allclose(reverse, -translation, rtol=0.0, atol=1e-12):
        raise RachelRealTranslationGTError("A/B swap does not negate translation")

    boundary_a = _boundary_points_parent_rc(first.mask, first.bbox_xyxy)
    boundary_b = _boundary_points_parent_rc(second.mask, second.bbox_xyxy)
    all_distances_ab, all_neighbors_ab = _one_way_nearest(boundary_a, boundary_b)
    all_distances_ba, all_neighbors_ba = _one_way_nearest(boundary_b, boundary_a)
    contact_ab = all_distances_ab <= SOURCE_CONTACT_TOLERANCE_PX
    contact_ba = all_distances_ba <= SOURCE_CONTACT_TOLERANCE_PX
    required = max(
        MIN_CONTACT_PIXELS,
        int(math.ceil(MIN_CONTACT_BOUNDARY_FRACTION * min(len(boundary_a), len(boundary_b)))),
    )
    label_source = pair_authority.get("label_source")
    strong_geometry_contract = label_source == "gt_numeric_bbox_plus_fragment_alpha"
    if strong_geometry_contract and (
        int(np.count_nonzero(contact_ab)) < required
        or int(np.count_nonzero(contact_ba)) < required
    ):
        raise RachelRealTranslationGTError(
            "positive pair fails authoritative alpha-boundary contact reenactment"
        )
    if label_source == "curated_two_fragment_conjunction_category":
        # Two-fragment folders are curator-positive even if the builder's
        # alpha-gap diagnostic is not a strong contact.  Use all <=2 px support
        # when present; otherwise validate the transform on the closest point
        # in each direction without silently changing the positive label.
        selected_mask_ab = contact_ab.copy()
        selected_mask_ba = contact_ba.copy()
        if not bool(selected_mask_ab.any()):
            selected_mask_ab[int(np.argmin(all_distances_ab))] = True
        if not bool(selected_mask_ba.any()):
            selected_mask_ba[int(np.argmin(all_distances_ba))] = True
    elif strong_geometry_contract:
        selected_mask_ab = contact_ab
        selected_mask_ba = contact_ba
    else:
        raise RachelRealTranslationGTError("positive pair label source is unsupported")

    distances_ab = all_distances_ab[selected_mask_ab]
    distances_ba = all_distances_ba[selected_mask_ba]
    selected_a = boundary_a[selected_mask_ab]
    paired_b = boundary_b[all_neighbors_ab[selected_mask_ab]]
    selected_b = boundary_b[selected_mask_ba]
    paired_a = boundary_a[all_neighbors_ba[selected_mask_ba]]
    source_seam_model_rc = _source_to_model_points(selected_a, first)
    target_seam_model_rc = _source_to_model_points(paired_b, second)
    residual_ab = (
        target_seam_model_rc
        - source_seam_model_rc
        - translation[None, :]
    )
    reverse_translation = -translation
    residual_ba = (
        _source_to_model_points(paired_a, first)
        - _source_to_model_points(selected_b, second)
        - reverse_translation[None, :]
    )
    residual_norm = np.concatenate(
        (
            np.linalg.norm(residual_ab, axis=1),
            np.linalg.norm(residual_ba, axis=1),
        )
    )
    expected_norm = np.concatenate((distances_ab, distances_ba)) * first.scale
    if not np.allclose(residual_norm, expected_norm, rtol=0.0, atol=1e-8):
        raise RachelRealTranslationGTError(
            "model-frame translation/contact residual identity failed"
        )
    residual_limit = float(
        max(np.max(distances_ab), np.max(distances_ba)) * first.scale + 1e-8
    )
    if not np.all(np.isfinite(residual_norm)) or np.max(residual_norm) > residual_limit:
        raise RachelRealTranslationGTError("positive translation residual gate failed")

    # The exact N=512 contour support is diagnostic, not a second source of GT.
    world_a = first.model_points_rc - first_offset[None, :]
    world_b = second.model_points_rc - second_offset[None, :]
    cap_distance_ab = cKDTree(world_b).query(world_a, k=1)[0]
    cap_distance_ba = cKDTree(world_a).query(world_b, k=1)[0]
    cap_tolerance = SOURCE_CONTACT_TOLERANCE_PX * first.scale + 3.0
    cap_support_ab = int(np.count_nonzero(cap_distance_ab <= cap_tolerance))
    cap_support_ba = int(np.count_nonzero(cap_distance_ba <= cap_tolerance))

    translation_rc = (float(translation[0]), float(translation[1]))
    return _PositiveGT(
        pair_id=pair_id,
        case_uid=first.case_uid,
        fragment_a_token=first.token,
        fragment_b_token=second.token,
        translation_a_to_b_rc=translation_rc,
        translation_a_to_b_xy_cartesian=(translation_rc[1], -translation_rc[0]),
        placement_shift_b_into_a_rc=(-translation_rc[0], -translation_rc[1]),
        source_seam_model_rc=np.ascontiguousarray(
            source_seam_model_rc, dtype=np.float64
        ),
        target_seam_model_rc=np.ascontiguousarray(
            target_seam_model_rc, dtype=np.float64
        ),
        source_contour_model_rc=np.ascontiguousarray(
            first.model_points_rc, dtype=np.float64
        ),
        target_contour_model_rc=np.ascontiguousarray(
            second.model_points_rc, dtype=np.float64
        ),
        seam_qa={
            "status": "passed",
            "authoritative_pair_label_source": label_source,
            "geometry_diagnostic_label": pair_authority.get(
                "geometry_diagnostic_label"
            ),
            "strong_geometry_contact_contract_required": strong_geometry_contract,
            "strong_geometry_contact_contract_passed": (
                int(np.count_nonzero(contact_ab)) >= required
                and int(np.count_nonzero(contact_ba)) >= required
            ),
            "authoritative_source_contact_tolerance_px": SOURCE_CONTACT_TOLERANCE_PX,
            "required_bidirectional_contact_count": required,
            "a_to_b_contact_count": int(np.count_nonzero(contact_ab)),
            "b_to_a_contact_count": int(np.count_nonzero(contact_ba)),
            "source_contact_distance_px": _summary(
                np.concatenate((distances_ab, distances_ba))
            ),
            "model_frame_correspondence_residual_px": _summary(residual_norm),
            "model_frame_residual_limit_px": residual_limit,
            "n512_cap_diagnostic": {
                "not_used_to_define_gt": True,
                "tolerance_px": cap_tolerance,
                "a_to_b_supported_token_count": cap_support_ab,
                "b_to_a_supported_token_count": cap_support_ba,
                "a_token_count": len(world_a),
                "b_token_count": len(world_b),
            },
        },
    )


def _load_gt_after_prediction_freeze(
    frozen: FrozenRealPredictions,
    frozen_labels: _FrozenLabelAuthority,
    *,
    main_root: Optional[Path],
    supp_root: Optional[Path],
    authority: RealTranslationGTAuthority,
) -> Tuple[Mapping[str, _PositiveGT], Mapping[str, object]]:
    """First function in the run path permitted to open real label authority."""

    try:
        with _frozen_label_authority_snapshot(frozen_labels) as snapshot:
            spec = load_real_external_test_spec(
                snapshot[0],
                snapshot[1],
                main_root=main_root,
                supp_root=supp_root,
                expected_case_count=authority.case_count,
                expected_fragment_count=authority.fragment_count,
                expected_pair_count=authority.strict_pair_count,
                expected_positive_count=authority.strict_positive_count,
                expected_negative_count=authority.strict_negative_count,
                derive_positive_direction=False,
            )
    except (OSError, RealDunhuangRepresentationError) as error:
        raise RachelRealTranslationGTError(
            "cannot load authoritative real label specification: " + str(error)
        ) from error
    if spec.manifest_sha256 != authority.manifest_sha256:
        raise RachelRealTranslationGTError("loaded real spec manifest differs")
    if tuple(pair.pair_id for pair in spec.pairs) != frozen.pair_ids:
        raise RachelRealTranslationGTError(
            "prediction pair order differs from authoritative real spec"
        )
    if tuple(pair.label for pair in spec.pairs) != frozen.labels:
        raise RachelRealTranslationGTError(
            "prediction labels differ from authoritative real spec"
        )
    bboxes, canvases, pair_authority = _label_geometry_from_manifest(
        frozen_labels.manifest, spec, authority=authority
    )
    ordered_endpoints = []
    for pair, frozen_row in zip(spec.pairs, frozen.strict_pairs):
        expected_case = [pair.case_uid]
        if frozen_row.get("source_case_uids") != expected_case:
            raise RachelRealTranslationGTError(
                "pair result source case differs from endpoint authority"
            )
        ordered_endpoints.append(
            {
                "pair_id": pair.pair_id,
                "source_pair_uid": pair.source_pair_uid,
                "case_uid": pair.case_uid,
                "fragment_a_token": "{}/fragment/{}".format(
                    pair.case_uid, pair.fragment_a_id
                ),
                "fragment_b_token": "{}/fragment/{}".format(
                    pair.case_uid, pair.fragment_b_id
                ),
                "label": pair.label,
            }
        )

    alpha_sha_values = []
    specs_by_case: Dict[str, list] = {}
    pairs_by_case: Dict[str, list] = {}
    frozen_row_by_id = {
        str(row["pair_id"]): row for row in frozen.strict_pairs
    }
    for fragment in spec.fragments:
        specs_by_case.setdefault(fragment.case_uid, []).append(fragment)
    for pair in spec.pairs:
        pairs_by_case.setdefault(pair.case_uid, []).append(pair)

    positives: Dict[str, _PositiveGT] = {}
    # Process one case at a time.  Some source PNGs are very large; retaining
    # all 938 decoded alpha masks would create unnecessary multi-GB peak memory.
    for case_uid, case_specs in specs_by_case.items():
        fragments: Dict[str, _GTFragment] = {}
        for fragment in case_specs:
            try:
                mask = real_eval._load_strict_alpha(fragment)
            except real_eval.RachelRealExternalError as error:
                raise RachelRealTranslationGTError(str(error)) from error
            observed_alpha_sha = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
            alpha_sha_values.append(observed_alpha_sha)
            value = _build_fragment_gt(
                fragment,
                mask=mask,
                bbox_xyxy=bboxes[(fragment.case_uid, fragment.fragment_id)],
                canvas_wh=canvases[fragment.case_uid],
                canvas_size=authority.canvas_size,
                contour_cap=authority.contour_cap,
            )
            if value.token in fragments:
                raise RachelRealTranslationGTError("GT fragment token is duplicated")
            fragments[value.token] = value

        # Independent equality check against the exact mask-preparation entry
        # point used by target-blind inference.
        tokens = tuple(fragments)
        exact = real_eval.prepare_case_alpha_masks(
            {token: fragments[token].mask for token in tokens},
            case_uid=case_uid,
            canvas_wh=canvases[case_uid],
            canvas_size=authority.canvas_size,
            contour_cap=authority.contour_cap,
        )
        for token in tokens:
            expected = fragments[token]
            observed = exact[token]
            point_count = len(expected.model_points_rc)
            if (
                not np.array_equal(observed.mask, expected.model_mask)
                or observed.tight_crop_hw != expected.resized_tight_hw
                or not math.isclose(
                    observed.shared_case_scale, expected.scale, rel_tol=0.0, abs_tol=0.0
                )
                or int(np.count_nonzero(observed.contour_valid)) != point_count
                or not np.allclose(
                    observed.points_rc[:point_count],
                    expected.model_points_rc,
                    rtol=0.0,
                    atol=1e-6,
                )
            ):
                raise RachelRealTranslationGTError(
                    "GT transform reconstruction differs from real evaluator preprocessing"
                )
        for pair in pairs_by_case.get(case_uid, ()):
            if not pair.label:
                continue
            first_token = "{}/fragment/{}".format(
                pair.case_uid, pair.fragment_a_id
            )
            second_token = "{}/fragment/{}".format(
                pair.case_uid, pair.fragment_b_id
            )
            gt = _positive_gt(
                fragments[first_token],
                fragments[second_token],
                pair.pair_id,
                pair_authority[pair.pair_id],
            )
            frozen_row = frozen_row_by_id.get(pair.pair_id)
            if (
                gt.pair_id in positives
                or frozen_row is None
                or frozen_row.get("label") is not True
            ):
                raise RachelRealTranslationGTError(
                    "positive GT identity is inconsistent"
                )
            positives[gt.pair_id] = gt
    if len(positives) != authority.strict_positive_count:
        raise RachelRealTranslationGTError("positive GT count differs")
    receipt = {
        "manifest_canonical_sha256": spec.manifest_sha256,
        "manifest_file_sha256": frozen_labels.manifest_file_sha256,
        "local_path_receipt_file_sha256": (
            frozen_labels.local_receipt_file_sha256
        ),
        "authority_files_frozen_once_before_spec_or_geometry_parse": True,
        "authority_file_hashes_reverified_before_publication": True,
        "verified_alpha_mask_count": len(alpha_sha_values),
        "verified_alpha_mask_digest_list_sha256": _canonical_sha256(
            tuple(alpha_sha_values)
        ),
        "case_count": authority.case_count,
        "fragment_count": authority.fragment_count,
        "positive_gt_count": len(positives),
        "ordered_endpoint_authority": {
            "source": "authoritative_real_spec_manifest_order",
            "pair_count": len(ordered_endpoints),
            "ordered_pair_endpoint_sha256": _canonical_sha256(ordered_endpoints),
            "translation_orientation": "fragment_a_to_fragment_b",
            "rows": ordered_endpoints,
        },
        "maximum_boundary_points_per_fragment": MAX_BOUNDARY_POINTS_PER_FRAGMENT,
        "bbox_values_emitted": False,
        "source_paths_emitted": False,
    }
    return positives, receipt


def _metric_point(
    errors: np.ndarray,
    valid: np.ndarray,
    threshold_positive: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Mapping[str, Optional[float]]:
    count = len(errors)
    if weights is None:
        weight = np.ones(count, dtype=np.int64)
    else:
        weight = np.asarray(weights, dtype=np.int64)
    denominator = int(np.sum(weight))
    valid_weight = weight * valid.astype(np.int64)
    valid_count = int(np.sum(valid_weight))
    repeated_errors = np.repeat(errors[valid], valid_weight[valid])
    output: Dict[str, Optional[float]] = {
        "valid_translation_prediction_fraction": (
            valid_count / denominator if denominator else None
        ),
        "median_l2_px": (
            float(np.median(repeated_errors)) if len(repeated_errors) else None
        ),
        "p90_l2_px": (
            float(np.quantile(repeated_errors, 0.9)) if len(repeated_errors) else None
        ),
    }
    for tolerance in TOLERANCES_PX:
        success = valid & (errors <= tolerance)
        joint = success & threshold_positive
        output["recall_at_{}px".format(tolerance)] = (
            int(np.sum(weight * success.astype(np.int64))) / denominator
            if denominator
            else None
        )
        output[
            "joint_frozen_threshold_and_translation_recall_at_{}px".format(
                tolerance
            )
        ] = (
            int(np.sum(weight * joint.astype(np.int64))) / denominator
            if denominator
            else None
        )
    return output


def _case_bootstrap(
    *,
    errors: np.ndarray,
    valid: np.ndarray,
    threshold_positive: np.ndarray,
    cases: Sequence[str],
    repetitions: int,
    seed: str,
) -> Mapping[str, object]:
    if type(repetitions) is not int or repetitions <= 0:  # noqa: E721
        raise ValueError("bootstrap repetitions must be a positive integer")
    units = tuple(sorted(set(cases)))
    unit_index = {unit: index for index, unit in enumerate(units)}
    row_unit = np.asarray([unit_index[value] for value in cases], dtype=np.int64)
    rng_seed = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    point = _metric_point(errors, valid, threshold_positive)
    draws: Dict[str, list] = {name: [] for name in point}
    for _ in range(repetitions):
        sampled = rng.integers(0, len(units), size=len(units))
        multiplicity = np.bincount(sampled, minlength=len(units))
        weights = multiplicity[row_unit]
        value = _metric_point(errors, valid, threshold_positive, weights)
        for name, result in value.items():
            if result is not None and math.isfinite(float(result)):
                draws[name].append(float(result))
    metrics: Dict[str, object] = {}
    for name, estimate in point.items():
        values = np.asarray(draws[name], dtype=np.float64)
        metrics[name] = {
            "estimate": estimate,
            "ci95_low": float(np.quantile(values, 0.025)) if len(values) else None,
            "ci95_high": float(np.quantile(values, 0.975)) if len(values) else None,
            "valid_bootstrap_replicates": len(values),
        }
    return {
        "schema": "case_cluster_percentile_bootstrap/1.0",
        "seed": seed,
        "repetitions": repetitions,
        "sampling_unit": "authoritative_real_case_uid",
        "case_count": len(units),
        "same_case_pairs_keep_endpoints_together": True,
        "metrics": metrics,
    }


def _polygon_area_rc(points_rc: np.ndarray) -> float:
    points = np.asarray(points_rc, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
        raise RachelRealTranslationGTError("registration contour is malformed")
    quantized = points.astype(np.int32).astype(np.float64)
    row = quantized[:, 0]
    column = quantized[:, 1]
    return float(
        abs(np.dot(column, np.roll(row, 1)) - np.dot(row, np.roll(column, 1)))
        / 2.0
    )


def _registration_record(
    gt: _PositiveGT,
    prediction: Optional[Tuple[float, float]],
) -> Mapping[str, object]:
    pose_valid = prediction is not None
    effective = np.asarray(
        prediction if prediction is not None else (0.0, 0.0), dtype=np.float64
    )
    transformed_seam = gt.source_seam_model_rc + effective[None, :]
    seam_residual = np.linalg.norm(
        transformed_seam - gt.target_seam_model_rc, axis=1
    )
    if not len(seam_residual) or not np.all(np.isfinite(seam_residual)):
        raise RachelRealTranslationGTError("registration seam residual is invalid")
    e_rmse = float(np.sqrt(np.mean(seam_residual)))
    hausdorff = float(
        max(
            np.max(
                cKDTree(gt.target_seam_model_rc).query(
                    transformed_seam, k=1
                )[0]
            ),
            np.max(
                cKDTree(transformed_seam).query(
                    gt.target_seam_model_rc, k=1
                )[0]
            ),
        )
    )
    translation_error = float(
        np.linalg.norm(effective - np.asarray(gt.translation_a_to_b_rc))
    )
    area_sum = _polygon_area_rc(gt.source_contour_model_rc) + _polygon_area_rc(
        gt.target_contour_model_rc
    )
    if area_sum <= 0.0:
        raise RachelRealTranslationGTError("registration contour area is non-positive")
    return {
        "prediction_valid": pose_valid,
        "identity_fallback_used": not pose_valid,
        "e_rmse": e_rmse,
        "registration_recall_lt4_success": e_rmse < 4.0,
        "symmetric_hausdorff_px": hausdorff,
        "normalized_translation_error": translation_error / area_sum,
    }


def _direct_prf(true_positive: int, predicted: int, target: int) -> Mapping[str, object]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / target if target else 0.0
    return {
        "true_positive_count": true_positive,
        "predicted_count": predicted,
        "target_count": target,
        "false_positive_count": predicted - true_positive,
        "false_negative_count": target - true_positive,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }


def _assembly_view(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    threshold: float,
    positive_error: Mapping[str, Tuple[bool, float]],
    negative_semantics: str,
) -> Mapping[str, object]:
    predicted_ids = set()
    target_ids = set()
    for row in rows:
        pair_id = str(row.get("pair_id"))
        label = row.get("label")
        if type(label) is not bool:  # noqa: E721
            raise RachelRealTranslationGTError("assembly row label is invalid")
        method_rows = _require_mapping(row.get("methods"), "assembly row methods")
        value = _require_mapping(method_rows.get(method), method + " assembly row")
        probability = _require_probability(
            value.get("probability"), method + " assembly probability"
        )
        valid = value.get("valid")
        if type(valid) is not bool:  # noqa: E721
            raise RachelRealTranslationGTError("assembly method validity is invalid")
        if valid and probability >= threshold:
            predicted_ids.add(pair_id)
        if label:
            target_ids.add(pair_id)
    output = {}
    for tolerance in TOLERANCES_PX:
        correct = {
            pair_id
            for pair_id in target_ids & predicted_ids
            if pair_id in positive_error
            and positive_error[pair_id][0]
            and positive_error[pair_id][1] <= tolerance
        }
        output["at_{}px".format(tolerance)] = _direct_prf(
            len(correct), len(predicted_ids), len(target_ids)
        )
    return {
        "definition": (
            "threshold-accepted edge is correct only for an authoritative positive "
            "with a valid translation within tolerance; wrong/invalid pose is FP+FN"
        ),
        "negative_semantics": negative_semantics,
        "predicted_edge_count": len(predicted_ids),
        "target_edge_count": len(target_ids),
        "by_tolerance": output,
    }


def summarize_translation_gt(
    frozen: FrozenRealPredictions,
    positive_gt: Mapping[str, _PositiveGT],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: str = BOOTSTRAP_SEED,
) -> Mapping[str, object]:
    """Score every frozen translation-emitting method on common direct metrics."""

    evaluated_methods = tuple(
        method for method in TRANSLATION_METHODS if method in frozen.method_names
    )
    if not evaluated_methods or evaluated_methods[0] != "full_n512":
        raise RachelRealTranslationGTError("full-N512 translation method is missing")
    per_method = {
        method: {
            "errors": [],
            "valid": [],
            "threshold_positive": [],
            "positive_error": {},
            "registration": [],
        }
        for method in evaluated_methods
    }
    rows = []
    cases = []
    positive_ids = []
    qa_residual_p95 = []
    qa_n512_bilateral_support = []
    qa_strong_required = []
    qa_strong_passed = []
    for source_row in frozen.strict_pairs:
        if source_row.get("label") is not True:
            continue
        pair_id = str(source_row["pair_id"])
        gt = positive_gt.get(pair_id)
        if gt is None:
            raise RachelRealTranslationGTError("strict positive lacks translation GT")
        source_methods = _require_mapping(
            source_row.get("methods"), "strict row methods"
        )
        row: Dict[str, object] = {
            "pair_id": pair_id,
            "case_uid": gt.case_uid,
            "fragment_a_token": gt.fragment_a_token,
            "fragment_b_token": gt.fragment_b_token,
            "translation_gt_a_to_b_rc": list(gt.translation_a_to_b_rc),
            "translation_gt_a_to_b_xy_cartesian": list(
                gt.translation_a_to_b_xy_cartesian
            ),
            "placement_shift_b_into_a_rc": list(gt.placement_shift_b_into_a_rc),
            "seam_and_correspondence_qa": gt.seam_qa,
        }
        for method in evaluated_methods:
            source_method = _require_mapping(
                source_methods.get(method), method + " positive row"
            )
            probability = _require_probability(
                source_method.get("probability"), method + " probability"
            )
            classification_valid = bool(source_method.get("valid"))
            prediction_value = source_method.get(
                "translation_hat_rc_unsupervised"
            )
            prediction = (
                _require_finite_pair(prediction_value, method + " translation")
                if classification_valid and prediction_value is not None
                else None
            )
            pose_valid = prediction is not None
            error_value = math.inf
            if pose_valid:
                assert prediction is not None
                error_value = math.hypot(
                    prediction[0] - gt.translation_a_to_b_rc[0],
                    prediction[1] - gt.translation_a_to_b_rc[1],
                )
            decision = classification_valid and (
                probability >= frozen.validation_thresholds[method]
            )
            detail = {
                "probability_frozen": probability,
                "validation_threshold_frozen": frozen.validation_thresholds[method],
                "decision_at_frozen_threshold": (
                    decision if classification_valid else None
                ),
                "translation_prediction_valid": pose_valid,
                "translation_hat_a_to_b_rc_frozen": (
                    list(prediction) if prediction is not None else None
                ),
                "translation_hat_a_to_b_xy_cartesian_frozen": (
                    [prediction[1], -prediction[0]]
                    if prediction is not None
                    else None
                ),
                "placement_shift_hat_b_into_a_rc_frozen": (
                    [-prediction[0], -prediction[1]]
                    if prediction is not None
                    else None
                ),
                "translation_l2_error_px": error_value if pose_valid else None,
                **{
                    "translation_within_{}px".format(tolerance): bool(
                        pose_valid and error_value <= tolerance
                    )
                    for tolerance in TOLERANCES_PX
                },
            }
            row[method] = detail
            per_method[method]["errors"].append(error_value)
            per_method[method]["valid"].append(pose_valid)
            per_method[method]["threshold_positive"].append(decision)
            per_method[method]["positive_error"][pair_id] = (
                pose_valid,
                error_value,
            )
            per_method[method]["registration"].append(
                _registration_record(gt, prediction)
            )
        rows.append(row)
        cases.append(gt.case_uid)
        positive_ids.append(pair_id)
        residual_summary = _require_mapping(
            gt.seam_qa.get("model_frame_correspondence_residual_px"),
            "GT correspondence residual summary",
        )
        residual_p95 = residual_summary.get("p95")
        if not isinstance(residual_p95, (int, float)) or not math.isfinite(
            float(residual_p95)
        ):
            raise RachelRealTranslationGTError("GT residual summary is invalid")
        qa_residual_p95.append(float(residual_p95))
        cap = _require_mapping(
            gt.seam_qa.get("n512_cap_diagnostic"), "N512 cap diagnostic"
        )
        qa_n512_bilateral_support.append(
            int(cap.get("a_to_b_supported_token_count", 0)) > 0
            and int(cap.get("b_to_a_supported_token_count", 0)) > 0
        )
        strong_required = gt.seam_qa.get(
            "strong_geometry_contact_contract_required"
        )
        strong_passed = gt.seam_qa.get(
            "strong_geometry_contact_contract_passed"
        )
        if type(strong_required) is not bool or type(strong_passed) is not bool:  # noqa: E721
            raise RachelRealTranslationGTError("GT contact-contract QA is invalid")
        qa_strong_required.append(bool(strong_required))
        qa_strong_passed.append(bool(strong_passed))
    if len(rows) != len(positive_gt):
        raise RachelRealTranslationGTError("positive prediction/GT coverage differs")

    method_metrics: Dict[str, object] = {}
    for method in frozen.method_names:
        if method not in evaluated_methods:
            method_metrics[method] = {
                "status": "not_applicable",
                "reason": "method_does_not_emit_a_supervised_2d_translation",
                "translation_metrics": None,
            }
            continue
        values = per_method[method]
        error_array = np.asarray(values["errors"], dtype=np.float64)
        valid_array = np.asarray(values["valid"], dtype=np.bool_)
        threshold_array = np.asarray(
            values["threshold_positive"], dtype=np.bool_
        )
        bootstrap = _case_bootstrap(
            errors=error_array,
            valid=valid_array,
            threshold_positive=threshold_array,
            cases=cases,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        bootstrap_metrics = _require_mapping(
            bootstrap.get("metrics"), "case-bootstrap metrics"
        )
        registrations = values["registration"]
        method_metrics[method] = {
            "status": "evaluated_positive_translation_gt",
            "scope": "strict_547_positive_pairs_plus_strict_and_balanced_edge_views",
            "eligible_positive_count": len(rows),
            "frozen_validation_threshold": frozen.validation_thresholds[method],
            "pair_score_semantics": EXPECTED_SCORE_SEMANTICS[method],
            "threshold_fit_performed_here": False,
            "invalid_translation_predictions_count_as_recall_failures": True,
            "point_estimates": {
                name: _require_mapping(value, "bootstrap metric").get("estimate")
                for name, value in bootstrap_metrics.items()
            },
            "case_bootstrap": bootstrap,
            "assembly_edge_strict_547": _assembly_view(
                frozen.strict_pairs,
                method=method,
                threshold=frozen.validation_thresholds[method],
                positive_error=values["positive_error"],
                negative_semantics="authoritative_strict_manifest_labels",
            ),
            "assembly_edge_balanced_1016_selected_list_diagnostic": _assembly_view(
                frozen.balanced_pairs,
                method=method,
                threshold=frozen.validation_thresholds[method],
                positive_error=values["positive_error"],
                negative_semantics=(
                    "strict_GT_negatives_plus_constructed_not_GT_negative_distractors"
                ),
            ),
            "pairingnet_style_registration": {
                "compatibility_source": (
                    "PairingNet matching_test.py upright translation-only specialization"
                ),
                "e_rmse_definition": (
                    "sqrt(mean(per_correspondence_euclidean_distance))"
                ),
                "invalid_pose_fallback": "identity_translation",
                "rr_lt4": float(
                    np.mean(
                        [
                            bool(value["registration_recall_lt4_success"])
                            for value in registrations
                        ]
                    )
                ),
                "mean_e_rmse": float(
                    np.mean([float(value["e_rmse"]) for value in registrations])
                ),
                "mean_symmetric_hausdorff_px": float(
                    np.mean(
                        [
                            float(value["symmetric_hausdorff_px"])
                            for value in registrations
                        ]
                    )
                ),
                "mean_normalized_translation_error": float(
                    np.mean(
                        [
                            float(value["normalized_translation_error"])
                            for value in registrations
                        ]
                    )
                ),
                "valid_pose_count": int(sum(values["valid"])),
                "identity_fallback_count": len(rows) - int(sum(values["valid"])),
                "rotation_error": {
                    "status": "not_applicable",
                    "reason": "known_upright_orientation_is_conditioned",
                },
            },
            "correspondence": {
                "status": "not_applicable",
                "reason": "real_target_blind_artifact_did_not_store_correspondence_matrix",
            },
            "native_global_assembly_GA": {
                "status": "not_applicable",
                "reason": "pairwise_only_no_global_placement",
            },
            "shreddingnet_native_CM_FM_SE": {
                "status": "not_reported",
                "reason": (
                    "balanced selected list is not exhaustive same-parent candidate graph"
                ),
            },
        }
    return {
        "method_metrics": method_metrics,
        "direct_pairwise_metric_contract": {
            "translation_tolerances_px": list(TOLERANCES_PX),
            "rotation_error": "not_applicable_known_upright",
            "same_positive_pairs_and_case_bootstrap_draws_across_methods": True,
            "balanced_selected_list_diagnostics_claimed_as_native_shreddingnet_metrics": False,
        },
        "seam_and_correspondence_qa": {
            "status": "passed_all_translation_transform_and_residual_checks",
            "positive_pair_count": len(rows),
            "strong_geometry_contact_contract_pair_count": int(
                sum(qa_strong_required)
            ),
            "strong_geometry_contact_contract_pass_count": int(
                sum(
                    required and passed
                    for required, passed in zip(
                        qa_strong_required, qa_strong_passed
                    )
                )
            ),
            "curated_two_fragment_positive_pair_count": int(
                len(rows) - sum(qa_strong_required)
            ),
            "maximum_pair_p95_model_frame_residual_px": max(qa_residual_p95),
            "n512_bilateral_token_support_pair_count": int(
                sum(qa_n512_bilateral_support)
            ),
            "n512_bilateral_token_support_fraction": float(
                np.mean(np.asarray(qa_n512_bilateral_support, dtype=np.float64))
            ),
            "n512_support_is_diagnostic_not_gt_eligibility": True,
        },
        "positive_pairs": rows,
        "positive_pair_order_sha256": _canonical_sha256(tuple(positive_ids)),
    }


def _write_atomic_no_replace(path: Path, value: Mapping[str, object]) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise RachelRealTranslationGTError("refusing to overwrite post-evaluation output")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
            linked = True
        except FileExistsError as error:
            raise RachelRealTranslationGTError(
                "refusing to overwrite post-evaluation output"
            ) from error
        temporary.unlink(missing_ok=True)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(str(target.parent), flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
        if linked and not target.exists():
            raise RachelRealTranslationGTError(
                "published post-evaluation output disappeared before directory fsync"
            )


def evaluate_real_translation_gt_postprediction(
    pair_only_result_path: Path,
    manifest_path: Path,
    local_path_receipt_path: Path,
    *,
    output_path: Path,
    main_root: Optional[Path] = None,
    supp_root: Optional[Path] = None,
    bootstrap_repetitions: int = 20_000,
    bootstrap_seed: str = BOOTSTRAP_SEED,
    authority: RealTranslationGTAuthority = RealTranslationGTAuthority(),
    compatibility_mode: bool = False,
) -> Mapping[str, object]:
    """Freeze predictions, then open GT, score, disclose, and publish once."""

    # Protocol boundary: this function accepts no real-data path and is called
    # before output preflight or any manifest/alpha/bbox read.
    frozen = freeze_real_prediction_result(
        pair_only_result_path,
        authority=authority,
        compatibility_mode=compatibility_mode,
    )
    _validate_output_path_before_gt(
        output_path,
        prediction_path=frozen.source_path,
        forbidden_roots=tuple(
            Path(value).parent
            for value in (manifest_path, local_path_receipt_path)
        )
        + tuple(Path(value) for value in (main_root, supp_root) if value is not None),
    )

    frozen_labels = _freeze_label_authority_after_predictions(
        manifest_path,
        local_path_receipt_path,
        authority=authority,
    )
    positive_gt, label_receipt = _load_gt_after_prediction_freeze(
        frozen,
        frozen_labels,
        main_root=main_root,
        supp_root=supp_root,
        authority=authority,
    )
    summary = summarize_translation_gt(
        frozen,
        positive_gt,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )
    if _sha256_file(frozen.source_path) != frozen.source_sha256:
        raise RachelRealTranslationGTError(
            "pair-only prediction file changed after it was frozen"
        )
    _verify_label_authority_unchanged(frozen_labels)
    result: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete_postprediction_real_positive_translation_gt"
            if frozen.formal_evaluation
            else "complete_non_formal_compatibility_real_translation_gt"
        ),
        "formal_evaluation": frozen.formal_evaluation,
        "compatibility_mode": not frozen.formal_evaluation,
        "source_pair_only_evaluation": {
            "path": str(frozen.source_path),
            "file_sha256_frozen_before_gt_open": frozen.source_sha256,
            "source_training_receipt_sha256": (
                frozen.source_training_receipt_sha256
            ),
            "source_matched_receipt_sha256": frozen.source_matched_receipt_sha256,
            "full_n512_checkpoint_sha256": frozen.full_checkpoint_sha256,
            "full_n512_validation_threshold": frozen.full_threshold,
            "checkpoint_sha256_by_method": dict(
                frozen.checkpoint_sha256_by_method
            ),
            "validation_thresholds": dict(frozen.validation_thresholds),
            "strict_pair_order_sha256": authority.strict_pair_order_sha256,
            "formal_combined_authority_gate_passed": (
                authority.enforce_formal_combined_gate
            ),
            "exact_method_order": list(frozen.method_names),
            "score_semantics": {
                method: EXPECTED_SCORE_SEMANTICS[method]
                for method in frozen.method_names
            },
        },
        "label_authority_opened_after_prediction_freeze": label_receipt,
        "population": {
            "source": "strict_547_positive_subset_only",
            "strict_pair_count": authority.strict_pair_count,
            "positive_pair_count": authority.strict_positive_count,
            "negative_pairs_have_translation_gt": False,
            "balanced_constructed_pairs_used_for_translation_or_GT": False,
            "balanced_constructed_pairs_used_for_selected_list_edge_diagnostic": True,
            "balanced_constructed_pairs_claimed_GT_negative": False,
        },
        "coordinate_contract": {
            "model_input_canvas_hw": [authority.canvas_size, authority.canvas_size],
            "one_shared_scale_per_case": True,
            "scale": "canvas_size/max(parent_canvas_width,parent_canvas_height)",
            "resize": "nearest_with_each_fragment_shape_rounded_exactly_as_pair_only_evaluator",
            "preprocess": "shared_case_scale_then_alpha_tight_crop_then_centerpad",
            "a_to_b_rc_semantics": "point_B_model = point_A_model + translation_A_to_B_RC",
            "cartesian_conversion": "(dx,dy)=(delta_col,-delta_row)",
            "assembly_semantics": "shift_B_into_A_frame = -translation_A_to_B_RC",
            "absolute_position_global_translation_gauge": True,
            "rotation_supervised_or_estimated": False,
        },
        "protocol": {
            "formal_evaluation": frozen.formal_evaluation,
            "compatibility_mode": not frozen.formal_evaluation,
            "formal_status_claimed": frozen.formal_evaluation,
            "formal_exact_six_required": frozen.formal_evaluation,
            "pair_only_prediction_file_frozen_before_any_current_gt_open": True,
            "manifest_and_local_receipt_bytes_frozen_once_before_gt_parse": True,
            "manifest_and_local_receipt_hashes_reverified_before_publication": True,
            "all_model_predictions_scores_thresholds_preexisting": True,
            "model_checkpoint_opened_here": False,
            "model_forward_or_gpu_work_performed_here": False,
            "bbox_and_translation_gt_read_after_predictions": True,
            "positive_direction_label_derived_or_used_here": False,
            "bbox_and_alpha_used_label_side_only": True,
            "gt_used_for_training_checkpoint_threshold_or_model_selection": False,
            "threshold_fit_performed_here": False,
            "coarse_and_matched_translation_are_not_applicable": True,
            "pairingnet_and_shreddingnet_adapted_translation_evaluated": (
                frozen.method_names == COMPLETE_BENCHMARK_METHODS
            ),
            "shreddingnet_selected_list_diagnostic_claimed_native_CM_FM_SE_GA": False,
            "evaluation_history_disclosure": {
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
                "claim_of_project_first_real_access": False,
                "current_translation_gt_postevaluation_used_for_any_selection": False,
            },
        },
        **summary,
    }
    result["content_sha256"] = _canonical_sha256(result)
    _write_atomic_no_replace(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-only-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-path-receipt", type=Path, required=True)
    parser.add_argument("--main-root", type=Path)
    parser.add_argument("--supp-root", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--compatibility-non-formal",
        action="store_true",
        help="accept only explicitly non-formal compatibility prediction receipts",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    evaluate_real_translation_gt_postprediction(
        arguments.pair_only_result,
        arguments.manifest,
        arguments.local_path_receipt,
        output_path=arguments.output,
        main_root=arguments.main_root,
        supp_root=arguments.supp_root,
        bootstrap_repetitions=arguments.bootstrap_repetitions,
        bootstrap_seed=arguments.bootstrap_seed,
        compatibility_mode=arguments.compatibility_non_formal,
    )
    print(str(arguments.output), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_SEED",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_STRICT_PAIR_ORDER_SHA256",
    "FrozenRealPredictions",
    "RachelRealTranslationGTError",
    "RealTranslationGTAuthority",
    "SCHEMA_VERSION",
    "evaluate_real_translation_gt_postprediction",
    "freeze_real_prediction_result",
    "summarize_translation_gt",
]
