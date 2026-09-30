"""Dependency-light authority for captured LOCAL-Q1 Route-A payloads.

This module validates and freezes already-produced eligibility decisions for a
metadata-only exact replay.  It deliberately imports no candidate builder,
geometry batch bridge, fragment/cache implementation, SciPy, torch, model, or
GPU module.  The producer-side geometry qualifier re-exports these exact public
classes while retaining its separate heavy execution helpers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

from staging.pairwise_v0_2.pairwise_data.training_stream import MaskMemberRef
from staging.pairwise_v0_2.training.local_q1_pair_qualification_contract import (
    LOCAL_Q1_PAIR_QUALIFICATION_VERSION,
    LocalQ1PairQualification,
)

# These are receipt vocabulary constants, not geometry execution code.  The
# producer-side typed loader independently recomputes the same commitments and
# the equivalence tests bind both views field by field.
FRAGMENT_GEOMETRY_ARTIFACT_VERSION = "upright-role-neutral-fragment/v0.3"
GEOMETRY_VERSION = "upright-facing-multirun-patches/v0.3"


class _ReceiptDirection(str, Enum):
    B_LEFT_OF_A = "b_left_of_a"
    B_RIGHT_OF_A = "b_right_of_a"
    B_ABOVE_A = "b_above_a"
    B_BELOW_A = "b_below_a"


DEFAULT_DIRECTION_ORDER = tuple(_ReceiptDirection)


class GeometryStatus(str, Enum):
    OK = "ok"
    INVALID_INPUT = "invalid_input"
    INVALID_CONTOUR = "invalid_contour"
    NO_USABLE_CANDIDATES = "no_usable_candidates"
    UNSUPPORTED_CONFIGURATION = "unsupported_configuration"

LOCAL_Q1_ELIGIBILITY_INDEX_SCHEMA_VERSION = (
    "dunhuang-local-q1-geometry-eligibility-index/0.1"
)
LOCAL_Q1_ELIGIBILITY_RECEIPT_SCHEMA_VERSION = (
    "dunhuang-local-q1-geometry-eligibility-receipt/0.1"
)
ROUTE_A_POLICY_SCHEMA_VERSION = "dunhuang-local-q1-refreeze-route-decision/0.2"
ROUTE_A_POLICY_FILE_SHA256 = (
    "3cf8781b65ad0d49fd12f47bb8c8f02f2052ee9f11b178b9c1e3001d3e52b5e3"
)
ROUTE_A_POLICY_CONTENT_SHA256 = (
    "be35c86a2504c4904106b0421f1456f8a7d64844d2135f084075107c4423543a"
)
CANONICAL_ROUTE_A_POLICY_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments/local_q1_refreeze_decision/route_a_policy_v0_2.json"
)
ROUTE_A_CONFIG_AUTHORITY_SCHEMA_VERSION = (
    "dunhuang-local-q1-route-a-config-authority/0.1"
)
ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256 = (
    "a7b238d7913be08d988277ba98dd21c170669e97002f28afe1b9112bae802379"
)
ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256 = (
    "200f8bfeadf957d2eec5e75d1448609433fab5a33fc33aa69723d4ae97b37041"
)
ROUTE_A_GEOMETRY_BATCH_CONFIG_SHA256 = (
    "dc994bf15596eea1b3d9245ac41505b1e72739eb49a91fef7a02012c744dcfd3"
)
ROUTE_A_PLANNING_GUARD_CONFIG_SHA256 = (
    "60d3f585f6f5ac27fdb3d2dbad29b8f52c9adb1d1955e88bfaa956ba51798d56"
)
ROUTE_A_INVENTORY_CONFIG_SHA256 = (
    "0270d5b7d7f042943cd52be28d6665ada11a037880fe81abf38f6d7fd008c123"
)
ROUTE_A_BATCH_PLAN_CONFIG_SHA256 = (
    "c27d3bc0629f65f33cecfb339b4b29e241a7f1c84c35e06d91eb140a9b4476b3"
)
ROUTE_A_CANDIDATE_BUILDER_CONFIG_SHA256 = (
    "1c5b6e4352bdbf686106e705989a734aa274d8fbff4b75854ffe092572fb7db9"
)
CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments/local_q1_refreeze_decision/route_a_config_authority_v0_1.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INPUT_ROLE_NAMES = frozenset(
    {
        "freeze_receipt",
        "mm_archive",
        "eccv_archive",
        "mm_fingerprint_cache",
        "eccv_fingerprint_cache",
        "historical_split",
        "synthetic_manifest",
        "synthetic_archive",
    }
)
_FORBIDDEN_PORTABLE_KEYS = frozenset(
    {
        "absolute_path",
        "archive_member",
        "canonical_group_id",
        "component_id",
        "fragment_id",
        "group_id",
        "label",
        "local_path",
        "member_path",
        "pair_id",
        "path",
        "secret",
    }
)



class LocalQ1GeometryQualificationError(RuntimeError):
    """An eligibility artifact, decision, or lazy geometry gate is unsafe."""


def _canonical_json(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(child) for child in item]
        return item

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _commit_order(tokens: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(str(token).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _commit_set(tokens: Sequence[str]) -> str:
    return _commit_order(sorted(tokens))


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise LocalQ1GeometryQualificationError(
            "{} must be lowercase SHA-256".format(name)
        )
    return value


def _portable(value: Any, location: str = "eligibility") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and (
            value.startswith(("/", "~/", "file://"))
            or re.match(r"^[A-Za-z]:[\\/]", value)
        ):
            raise LocalQ1GeometryQualificationError(
                "{} contains a machine-local path".format(location)
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LocalQ1GeometryQualificationError(
                "{} contains NaN/Inf".format(location)
            )
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LocalQ1GeometryQualificationError(
                    "{} contains a non-string key".format(location)
                )
            if key.casefold() in _FORBIDDEN_PORTABLE_KEYS:
                raise LocalQ1GeometryQualificationError(
                    "{} exposes a path, identity, or supervision field".format(location)
                )
            result[key] = _portable(item, "{}.{}".format(location, key))
        return result
    if isinstance(value, (list, tuple)):
        return [
            _portable(item, "{}[{}]".format(location, index))
            for index, item in enumerate(value)
        ]
    raise LocalQ1GeometryQualificationError("{} is not JSON-portable".format(location))


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _parse_strict_json(payload: bytes, name: str) -> Mapping[str, Any]:
    if type(payload) is not bytes:  # noqa: E721
        raise TypeError("{} bytes are required".format(name))

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant: " + token)
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise LocalQ1GeometryQualificationError(
            "{} is not strict JSON".format(name)
        ) from exc
    if not isinstance(value, Mapping):
        raise LocalQ1GeometryQualificationError(
            "{} root must be an object".format(name)
        )
    return value


def _strict_json(payload: bytes, name: str) -> Mapping[str, Any]:
    value = _parse_strict_json(payload, name)
    if payload != _canonical_json(value):
        raise LocalQ1GeometryQualificationError("{} is not canonical JSON".format(name))
    return value


def _content_sha256(value: Mapping[str, Any], name: str) -> str:
    unsigned = dict(value)
    claimed = _require_sha256(unsigned.pop("content_sha256", None), name + " content")
    observed = _sha256(_canonical_json(unsigned))
    if not hmac.compare_digest(claimed, observed):
        raise LocalQ1GeometryQualificationError("{} content hash mismatch".format(name))
    return claimed


@dataclass(frozen=True)
class LocalQ1ExternalFileLock:
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, int)
            or self.bytes <= 0
        ):
            raise ValueError("external file-lock bytes must be a positive integer")
        _require_sha256(self.sha256, "external file-lock sha256")

    def portable_dict(self) -> Dict[str, Any]:
        return {"bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class LocalQ1EligibilityTrust:
    """Independent locks required before an index/receipt can be trusted."""

    predecessor_freeze_file_sha256: str
    predecessor_freeze_content_sha256: str
    input_role_locks: Mapping[str, LocalQ1ExternalFileLock]
    source_bundle_manifest_sha256: str
    geometry_batch_config_sha256: str
    planning_guard_config_sha256: str
    eligibility_policy_file_sha256: str
    eligibility_policy_content_sha256: str
    config_authority_file_sha256: str
    config_authority_content_sha256: str
    eligibility_index_file_sha256: str
    eligibility_index_content_sha256: str
    eligibility_receipt_file_sha256: str
    eligibility_receipt_content_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "predecessor_freeze_file_sha256",
            "predecessor_freeze_content_sha256",
            "source_bundle_manifest_sha256",
            "geometry_batch_config_sha256",
            "planning_guard_config_sha256",
            "eligibility_policy_file_sha256",
            "eligibility_policy_content_sha256",
            "config_authority_file_sha256",
            "config_authority_content_sha256",
            "eligibility_index_file_sha256",
            "eligibility_index_content_sha256",
            "eligibility_receipt_file_sha256",
            "eligibility_receipt_content_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            self.eligibility_policy_file_sha256 != ROUTE_A_POLICY_FILE_SHA256
            or self.eligibility_policy_content_sha256 != ROUTE_A_POLICY_CONTENT_SHA256
        ):
            raise LocalQ1GeometryQualificationError(
                "only the corrected Route-A policy v0.2 may authorize eligibility"
            )
        if (
            self.config_authority_file_sha256 != ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256
            or self.config_authority_content_sha256
            != ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256
            or self.geometry_batch_config_sha256 != ROUTE_A_GEOMETRY_BATCH_CONFIG_SHA256
            or self.planning_guard_config_sha256 != ROUTE_A_PLANNING_GUARD_CONFIG_SHA256
        ):
            raise LocalQ1GeometryQualificationError(
                "only the preregistered Route-A production configs may authorize eligibility"
            )
        if (
            not isinstance(self.input_role_locks, Mapping)
            or set(self.input_role_locks) != _INPUT_ROLE_NAMES
        ):
            raise LocalQ1GeometryQualificationError(
                "eligibility trust requires all eight input roles exactly"
            )
        normalized = {}
        for role in sorted(_INPUT_ROLE_NAMES):
            lock = self.input_role_locks[role]
            if not isinstance(lock, LocalQ1ExternalFileLock):
                raise TypeError("eligibility input locks must be file locks")
            normalized[role] = lock
        if normalized["freeze_receipt"].sha256 != self.predecessor_freeze_file_sha256:
            raise LocalQ1GeometryQualificationError(
                "predecessor freeze and freeze input-role lock disagree"
            )
        object.__setattr__(self, "input_role_locks", MappingProxyType(normalized))

    def portable_dict(self) -> Dict[str, Any]:
        return {
            "predecessor_freeze_file_sha256": self.predecessor_freeze_file_sha256,
            "predecessor_freeze_content_sha256": self.predecessor_freeze_content_sha256,
            "input_role_locks": {
                role: self.input_role_locks[role].portable_dict()
                for role in sorted(self.input_role_locks)
            },
            "source_bundle_manifest_sha256": self.source_bundle_manifest_sha256,
            "geometry_batch_config_sha256": self.geometry_batch_config_sha256,
            "planning_guard_config_sha256": self.planning_guard_config_sha256,
            "eligibility_policy_file_sha256": self.eligibility_policy_file_sha256,
            "eligibility_policy_content_sha256": self.eligibility_policy_content_sha256,
            "config_authority_file_sha256": self.config_authority_file_sha256,
            "config_authority_content_sha256": self.config_authority_content_sha256,
            "eligibility_index_file_sha256": self.eligibility_index_file_sha256,
            "eligibility_index_content_sha256": self.eligibility_index_content_sha256,
            "eligibility_receipt_file_sha256": self.eligibility_receipt_file_sha256,
            "eligibility_receipt_content_sha256": (
                self.eligibility_receipt_content_sha256
            ),
        }


@dataclass(frozen=True)
class LocalQ1EligibilityDecision:
    pair_identity_sha256: str
    assessment: LocalQ1PairQualification

    def __post_init__(self) -> None:
        _require_sha256(self.pair_identity_sha256, "pair identity commitment")
        if not isinstance(self.assessment, LocalQ1PairQualification):
            raise TypeError("assessment must be LocalQ1PairQualification")

    def portable_dict(self) -> Dict[str, Any]:
        return {
            "pair_identity_sha256": self.pair_identity_sha256,
            "assessment": self.assessment.portable_dict(),
        }


def _parse_assessment(value: Any) -> LocalQ1PairQualification:
    if not isinstance(value, Mapping):
        raise LocalQ1GeometryQualificationError(
            "eligibility assessment must be an object"
        )
    expected = {
        "qualification_version",
        "eligible",
        "failure_stage",
        "failure_reason",
        "fragment_statuses",
        "pair_status",
        "emitted_directions",
        "candidate_count",
        "max_sequence_a",
        "max_sequence_b",
        "local_tensor_elements",
        "attention_elements_per_head",
        "affinity_elements",
        "sinkhorn_elements",
        "candidate_semantic_sha256",
    }
    if set(value) != expected or value.get("qualification_version") != (
        LOCAL_Q1_PAIR_QUALIFICATION_VERSION
    ):
        raise LocalQ1GeometryQualificationError(
            "eligibility assessment schema/version changed"
        )
    payload = dict(value)
    payload.pop("qualification_version")
    payload["fragment_statuses"] = tuple(payload["fragment_statuses"])
    payload["emitted_directions"] = tuple(payload["emitted_directions"])
    try:
        assessment = LocalQ1PairQualification(**payload)
    except (TypeError, ValueError) as exc:
        raise LocalQ1GeometryQualificationError(
            "eligibility assessment is invalid"
        ) from exc
    ok = GeometryStatus.OK.value
    frozen_directions = tuple(direction.value for direction in DEFAULT_DIRECTION_ORDER)
    fragment_ok = assessment.fragment_statuses == (ok, ok)
    pair_ok = assessment.pair_status == ok
    directions_ok = (
        len(assessment.emitted_directions) == len(frozen_directions)
        and len(set(assessment.emitted_directions)) == len(frozen_directions)
        and set(assessment.emitted_directions) == set(frozen_directions)
    )
    if assessment.eligible:
        if not (
            fragment_ok
            and pair_ok
            and directions_ok
            and assessment.candidate_count > 0
            and assessment.max_sequence_a > 0
            and assessment.max_sequence_b > 0
        ):
            raise LocalQ1GeometryQualificationError(
                "eligible decision violates fragment/pair/direction/resource semantics"
            )
    elif assessment.failure_stage == "fragment":
        if fragment_ok or assessment.pair_status != "not_evaluated":
            raise LocalQ1GeometryQualificationError(
                "fragment failure assessment is semantically inconsistent"
            )
    elif assessment.failure_stage == "pair":
        if not fragment_ok or directions_ok:
            raise LocalQ1GeometryQualificationError(
                "pair failure assessment is semantically inconsistent"
            )
    elif assessment.failure_stage == "direction":
        if not fragment_ok or not pair_ok or directions_ok:
            raise LocalQ1GeometryQualificationError(
                "direction failure assessment is semantically inconsistent"
            )
    elif assessment.failure_stage == "resource":
        if not (
            fragment_ok and pair_ok and directions_ok and assessment.candidate_count > 0
        ):
            raise LocalQ1GeometryQualificationError(
                "resource failure assessment is semantically inconsistent"
            )
    else:  # pragma: no cover - dataclass owns the finite stage vocabulary
        raise LocalQ1GeometryQualificationError(
            "ineligible decision lacks a recognized failure stage"
        )
    return assessment


def _validate_qualification_contract(
    value: Any, *, expected_geometry_config_sha256: str
) -> None:
    expected = {
        "qualification_version": LOCAL_Q1_PAIR_QUALIFICATION_VERSION,
        "candidate_generation_direction_argument": None,
        "candidate_generation_directions": [
            direction.value for direction in DEFAULT_DIRECTION_ORDER
        ],
        "supervision_fields_read": [],
        "fragment_gate": "both_fragment_geometry_results_are_ok",
        "pair_gate": "combine_direction_none_ok_and_nonempty_candidates",
        "direction_gate": "all_four_frozen_upright_directions_exactly_once",
        "individual_resource_gate": (
            "all_candidate_sequence_attention_affinity_sinkhorn_id_and_single_"
            "record_batch_bounds"
        ),
        "geometry_batch_config_sha256": expected_geometry_config_sha256,
    }
    if not isinstance(value, Mapping) or _canonical_json(value) != _canonical_json(
        expected
    ):
        raise LocalQ1GeometryQualificationError(
            "eligibility index qualification/config contract changed"
        )


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LocalQ1GeometryQualificationError(
            "{} must be a non-negative integer".format(name)
        )
    return value


def _validate_diff(value: Any, name: str) -> None:
    expected = {
        "old_count",
        "new_count",
        "unchanged_count",
        "removed_count",
        "replacement_count",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LocalQ1GeometryQualificationError("{} diff schema changed".format(name))
    counts = {key: _nonnegative_int(value[key], name + "." + key) for key in expected}
    if (
        counts["old_count"] != counts["new_count"]
        or counts["unchanged_count"] + counts["removed_count"] != counts["old_count"]
        or counts["unchanged_count"] + counts["replacement_count"]
        != counts["new_count"]
        or counts["removed_count"] != counts["replacement_count"]
    ):
        raise LocalQ1GeometryQualificationError(
            "{} diff arithmetic is inconsistent".format(name)
        )


def _validate_selection_proof(
    value: Any, *, decision_count: int, require_production: bool = True
) -> None:
    expected_root = {
        "frontier_version",
        "frontier_storage",
        "arbitrary_fixed_reserve_cutoff_used",
        "manual_exclusions_used",
        "unexplored_higher_priority_record_count",
        "training",
        "validation",
        "old_to_new_population_difference",
        "predecessor_population",
        "decision_count",
        "decision_order_sha256",
        "decision_trace_sha256",
        "decision_failure_stage_reason_counts",
        "selected_training_count",
        "selected_validation_count",
        "selected_record_order_sha256",
        "selected_record_set_sha256",
        "selected_pair_commitment_order_sha256",
        "selected_pair_commitment_set_sha256",
        "selected_failure_counts",
    }
    if not isinstance(value, Mapping) or set(value) != expected_root:
        raise LocalQ1GeometryQualificationError("selection proof root schema changed")
    if (
        value["frontier_version"]
        != "dunhuang-local-q1-complete-lazy-selection-frontier/0.1"
        or value["frontier_storage"]
        != "complete_metadata_sqlite_external_sort_geometry_consumed_lazily"
        or value["arbitrary_fixed_reserve_cutoff_used"] is not False
        or value["manual_exclusions_used"] is not False
        or value["unexplored_higher_priority_record_count"] != 0
        or value["decision_count"] != decision_count
    ):
        raise LocalQ1GeometryQualificationError("selection frontier proof changed")
    for name in (
        "decision_order_sha256",
        "decision_trace_sha256",
        "selected_record_order_sha256",
        "selected_record_set_sha256",
        "selected_pair_commitment_order_sha256",
        "selected_pair_commitment_set_sha256",
    ):
        _require_sha256(value[name], "selection proof " + name)
    selected_train = _nonnegative_int(
        value["selected_training_count"], "selected training count"
    )
    selected_validation = _nonnegative_int(
        value["selected_validation_count"], "selected validation count"
    )
    predecessor = value["predecessor_population"]
    predecessor_fields = {
        "training_count",
        "validation_count",
        "training_record_order_sha256",
        "training_record_set_sha256",
        "validation_record_order_sha256",
        "validation_record_set_sha256",
    }
    if not isinstance(predecessor, Mapping) or set(predecessor) != (predecessor_fields):
        raise LocalQ1GeometryQualificationError(
            "selection proof predecessor population schema changed"
        )
    predecessor_training = _nonnegative_int(
        predecessor["training_count"], "predecessor training count"
    )
    predecessor_validation = _nonnegative_int(
        predecessor["validation_count"], "predecessor validation count"
    )
    if (
        predecessor_training != selected_train
        or predecessor_validation != selected_validation
    ):
        raise LocalQ1GeometryQualificationError(
            "selection proof predecessor population counts changed"
        )
    for name in predecessor_fields - {"training_count", "validation_count"}:
        _require_sha256(predecessor[name], "predecessor population " + name)
    expected_train_strata = {
        "mm_augmented|negative",
        "mm_augmented|positive",
        "eccv_1113data|negative",
        "eccv_1113data|positive",
        "canonical_new|negative",
        "canonical_new|positive",
    }
    training = value["training"]
    if not isinstance(training, Mapping) or set(training) != expected_train_strata:
        raise LocalQ1GeometryQualificationError(
            "selection proof training strata changed"
        )
    train_selected_sum = 0
    train_fields = {
        "target",
        "selected_count",
        "component_count",
        "rounds_started",
        "final_round_component_cutoff_ordinal",
        "decided_frontier_row_count",
        "ineligible_predecessor_count",
        "exhausted_component_count",
        "unexplored_higher_priority_record_count",
        "arbitrary_fixed_reserve_cutoff_used",
        "frontier_trace_sha256",
        "selected_record_order_sha256",
        "selected_record_set_sha256",
    }
    for stratum, row in training.items():
        if not isinstance(row, Mapping) or set(row) != train_fields:
            raise LocalQ1GeometryQualificationError(
                "selection proof training row schema changed"
            )
        counts = {
            name: _nonnegative_int(row[name], stratum + "." + name)
            for name in train_fields
            if name
            not in {
                "arbitrary_fixed_reserve_cutoff_used",
                "frontier_trace_sha256",
                "selected_record_order_sha256",
                "selected_record_set_sha256",
            }
        }
        if (
            row["arbitrary_fixed_reserve_cutoff_used"] is not False
            or counts["unexplored_higher_priority_record_count"] != 0
            or counts["target"] <= 0
            or counts["selected_count"] != counts["target"]
            or counts["component_count"] <= 0
            or counts["rounds_started"] <= 0
            or counts["final_round_component_cutoff_ordinal"]
            >= counts["component_count"]
            or counts["decided_frontier_row_count"]
            != counts["selected_count"] + counts["ineligible_predecessor_count"]
            or counts["exhausted_component_count"] > counts["component_count"]
        ):
            raise LocalQ1GeometryQualificationError(
                "selection proof training prefix arithmetic changed"
            )
        for name in (
            "frontier_trace_sha256",
            "selected_record_order_sha256",
            "selected_record_set_sha256",
        ):
            _require_sha256(row[name], "training proof " + name)
        train_selected_sum += counts["selected_count"]
    if train_selected_sum != selected_train:
        raise LocalQ1GeometryQualificationError(
            "selection proof training total is inconsistent"
        )

    validation = value["validation"]
    if not isinstance(validation, Mapping) or set(validation) != {
        "mm_augmented",
        "eccv_1113data",
    }:
        raise LocalQ1GeometryQualificationError(
            "selection proof validation domains changed"
        )
    validation_fields = {
        "cap_per_component",
        "component_count",
        "selected_count",
        "decided_frontier_row_count",
        "ineligible_predecessor_count",
        "minimum_decided_prefix_per_component",
        "maximum_decided_prefix_per_component",
        "component_cap_shortfall_count",
        "unexplored_higher_priority_record_count",
        "arbitrary_fixed_reserve_cutoff_used",
        "frontier_trace_sha256",
        "selected_record_set_sha256",
    }
    validation_selected_sum = 0
    for dataset, row in validation.items():
        if not isinstance(row, Mapping) or set(row) != validation_fields:
            raise LocalQ1GeometryQualificationError(
                "selection proof validation row schema changed"
            )
        counts = {
            name: _nonnegative_int(row[name], dataset + "." + name)
            for name in validation_fields
            if name
            not in {
                "arbitrary_fixed_reserve_cutoff_used",
                "frontier_trace_sha256",
                "selected_record_set_sha256",
            }
        }
        if (
            row["arbitrary_fixed_reserve_cutoff_used"] is not False
            or counts["cap_per_component"] <= 0
            or counts["component_count"] <= 0
            or counts["selected_count"]
            != counts["cap_per_component"] * counts["component_count"]
            or counts["decided_frontier_row_count"]
            != counts["selected_count"] + counts["ineligible_predecessor_count"]
            or counts["minimum_decided_prefix_per_component"]
            < counts["cap_per_component"]
            or counts["maximum_decided_prefix_per_component"]
            < counts["minimum_decided_prefix_per_component"]
            or counts["component_cap_shortfall_count"] != 0
            or counts["unexplored_higher_priority_record_count"] != 0
        ):
            raise LocalQ1GeometryQualificationError(
                "selection proof validation prefix arithmetic changed"
            )
        _require_sha256(row["frontier_trace_sha256"], "validation trace hash")
        _require_sha256(row["selected_record_set_sha256"], "validation set hash")
        validation_selected_sum += counts["selected_count"]
    if validation_selected_sum != selected_validation:
        raise LocalQ1GeometryQualificationError(
            "selection proof validation total is inconsistent"
        )
    if require_production:
        production_train = {stratum: 512 for stratum in expected_train_strata}
        production_validation = {
            "mm_augmented": {"cap": 32, "components": 55, "selected": 1760},
            "eccv_1113data": {"cap": 4, "components": 533, "selected": 2132},
        }
        if (
            selected_train != 3072
            or selected_validation != 3892
            or decision_count < 6964
            or any(
                training[stratum]["target"] != target
                or training[stratum]["selected_count"] != target
                for stratum, target in production_train.items()
            )
            or any(
                validation[dataset]["cap_per_component"] != expected["cap"]
                or validation[dataset]["component_count"] != expected["components"]
                or validation[dataset]["selected_count"] != expected["selected"]
                for dataset, expected in production_validation.items()
            )
        ):
            raise LocalQ1GeometryQualificationError(
                "selection proof is not the preregistered production population"
            )

    differences = value["old_to_new_population_difference"]
    if (
        not isinstance(differences, Mapping)
        or set(differences)
        != {
            "training_by_dataset_label",
            "validation_by_dataset",
            "path_or_row_identifiers_present",
        }
        or differences["path_or_row_identifiers_present"] is not False
    ):
        raise LocalQ1GeometryQualificationError(
            "selection proof population-difference schema changed"
        )
    train_diff = differences["training_by_dataset_label"]
    val_diff = differences["validation_by_dataset"]
    if not isinstance(train_diff, Mapping) or set(train_diff) != expected_train_strata:
        raise LocalQ1GeometryQualificationError("training diff strata changed")
    if not isinstance(val_diff, Mapping) or set(val_diff) != {
        "mm_augmented",
        "eccv_1113data",
    }:
        raise LocalQ1GeometryQualificationError("validation diff domains changed")
    for name, row in list(train_diff.items()) + list(val_diff.items()):
        _validate_diff(row, str(name))

    failure_counts = value["decision_failure_stage_reason_counts"]
    if (
        not isinstance(failure_counts, Mapping)
        or any(
            not isinstance(key, str)
            or not key
            or _nonnegative_int(count, "decision failure count") < 0
            for key, count in failure_counts.items()
        )
        or sum(failure_counts.values()) != decision_count
    ):
        raise LocalQ1GeometryQualificationError(
            "selection proof decision census is inconsistent"
        )
    if value["selected_failure_counts"] != {
        "fragment": 0,
        "pair": 0,
        "direction_coverage": 0,
        "resource": 0,
    }:
        raise LocalQ1GeometryQualificationError(
            "selection proof selected failures are nonzero"
        )


def _validate_decision_census(
    value: Any, decisions: Mapping[str, LocalQ1EligibilityDecision]
) -> None:
    expected_fields = {
        "decision_count",
        "eligible_count",
        "ineligible_count",
        "by_failure_stage",
        "by_failure_stage_reason",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise LocalQ1GeometryQualificationError("decision census schema changed")
    eligible = sum(decision.assessment.eligible for decision in decisions.values())
    ineligible = len(decisions) - eligible
    stage = Counter(
        decision.assessment.failure_stage or "eligible"
        for decision in decisions.values()
    )
    stage_reason = Counter(
        "{}|{}".format(
            decision.assessment.failure_stage or "eligible",
            decision.assessment.failure_reason or "eligible",
        )
        for decision in decisions.values()
    )
    expected = {
        "decision_count": len(decisions),
        "eligible_count": eligible,
        "ineligible_count": ineligible,
        "by_failure_stage": dict(sorted(stage.items())),
        "by_failure_stage_reason": dict(sorted(stage_reason.items())),
    }
    if _canonical_json(value) != _canonical_json(expected):
        raise LocalQ1GeometryQualificationError(
            "decision census differs from locked eligibility index"
        )


def _validate_scratch_cache_evidence(
    value: Any,
    *,
    decision_count: int,
    selected_record_count: int,
    selected_pair_commitment_order_sha256: str,
    selected_pair_commitment_set_sha256: str,
) -> None:
    expected_fields = {
        "cache_role",
        "fresh_path_atomically_claimed",
        "initial_artifact_count",
        "external_preexisting_artifact_count",
        "predecessor_partial_cache_role_count",
        "predecessor_partial_cache_tree_accessed",
        "mask_decode_count",
        "scratch_cache_hit_count",
        "scratch_cache_miss_count",
        "final_artifact_count",
        "final_artifact_file_bytes",
        "scratch_cache_artifact_set_sha256",
        "selected_population_guard",
        "decision_count",
        "path_or_fragment_identifiers_present",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise LocalQ1GeometryQualificationError(
            "qualification scratch-cache evidence schema changed"
        )
    if (
        value["cache_role"] != "new_local_scratch_output_never_membership_input"
        or value["fresh_path_atomically_claimed"] is not True
        or value["predecessor_partial_cache_tree_accessed"] is not False
        or value["path_or_fragment_identifiers_present"] is not False
    ):
        raise LocalQ1GeometryQualificationError(
            "qualification scratch cache was not independent/private"
        )
    counts = {
        name: _nonnegative_int(value[name], "scratch cache " + name)
        for name in (
            "initial_artifact_count",
            "external_preexisting_artifact_count",
            "predecessor_partial_cache_role_count",
            "mask_decode_count",
            "scratch_cache_hit_count",
            "scratch_cache_miss_count",
            "final_artifact_count",
            "final_artifact_file_bytes",
            "decision_count",
        )
    }
    _require_sha256(
        value["scratch_cache_artifact_set_sha256"],
        "scratch cache artifact-set commitment",
    )
    selected_guard = value["selected_population_guard"]
    expected_selected_guard_fields = {
        "inventory_config_sha256",
        "selected_record_count",
        "selected_pair_commitment_order_sha256",
        "selected_pair_commitment_set_sha256",
        "selected_unique_reference_count",
        "selected_max_mask_pixels_per_reference",
        "max_records_bound",
        "max_unique_references_bound",
        "max_mask_pixels_per_reference_bound",
        "train_val_canonical_fragment_overlap_count",
        "all_selected_decisions_eligible",
        "additional_mask_decode_count",
        "path_or_fragment_identifiers_present",
    }
    if not isinstance(selected_guard, Mapping) or set(selected_guard) != (
        expected_selected_guard_fields
    ):
        raise LocalQ1GeometryQualificationError(
            "selected-population guard evidence schema changed"
        )
    _require_sha256(
        selected_guard["inventory_config_sha256"],
        "selected population inventory config",
    )
    for name in (
        "selected_pair_commitment_order_sha256",
        "selected_pair_commitment_set_sha256",
    ):
        _require_sha256(selected_guard[name], "selected population " + name)
    guard_counts = {
        name: _nonnegative_int(
            selected_guard[name], "selected population guard " + name
        )
        for name in expected_selected_guard_fields
        if name
        not in {
            "inventory_config_sha256",
            "selected_pair_commitment_order_sha256",
            "selected_pair_commitment_set_sha256",
            "all_selected_decisions_eligible",
            "path_or_fragment_identifiers_present",
        }
    }
    if (
        selected_guard["inventory_config_sha256"] != ROUTE_A_INVENTORY_CONFIG_SHA256
        or selected_guard["all_selected_decisions_eligible"] is not True
        or selected_guard["path_or_fragment_identifiers_present"] is not False
        or guard_counts["selected_record_count"] != selected_record_count
        or selected_guard["selected_pair_commitment_order_sha256"]
        != selected_pair_commitment_order_sha256
        or selected_guard["selected_pair_commitment_set_sha256"]
        != selected_pair_commitment_set_sha256
        or guard_counts["selected_record_count"] > guard_counts["max_records_bound"]
        or guard_counts["selected_unique_reference_count"] <= 0
        or guard_counts["selected_unique_reference_count"]
        > guard_counts["max_unique_references_bound"]
        or guard_counts["selected_max_mask_pixels_per_reference"] <= 0
        or guard_counts["selected_max_mask_pixels_per_reference"]
        > guard_counts["max_mask_pixels_per_reference_bound"]
        or guard_counts["max_records_bound"] != 20_000
        or guard_counts["max_unique_references_bound"] != 40_000
        or guard_counts["max_mask_pixels_per_reference_bound"] != 100_000_000
        or guard_counts["train_val_canonical_fragment_overlap_count"] != 0
        or guard_counts["additional_mask_decode_count"] != 0
    ):
        raise LocalQ1GeometryQualificationError(
            "selected-population planning guards are inconsistent"
        )
    if (
        counts["initial_artifact_count"] != 0
        or counts["external_preexisting_artifact_count"] != 0
        or counts["predecessor_partial_cache_role_count"] != 0
        or counts["decision_count"] != decision_count
        or counts["scratch_cache_hit_count"] + counts["scratch_cache_miss_count"]
        != counts["mask_decode_count"]
        or counts["final_artifact_count"] != counts["scratch_cache_miss_count"]
        or (
            counts["final_artifact_count"] == 0
            and counts["final_artifact_file_bytes"] != 0
        )
        or (
            counts["final_artifact_count"] > 0
            and counts["final_artifact_file_bytes"] <= 0
        )
    ):
        raise LocalQ1GeometryQualificationError(
            "qualification scratch-cache counts are inconsistent"
        )


@dataclass(frozen=True)
class LocalQ1GeometryEligibilityAuthority:
    """Externally verified decisions plus the aggregate frontier proof."""

    decisions: Mapping[str, LocalQ1EligibilityDecision]
    decision_order: Tuple[str, ...]
    index: Mapping[str, Any]
    receipt: Mapping[str, Any]
    trust: LocalQ1EligibilityTrust
    production_authorized: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.trust, LocalQ1EligibilityTrust):
            raise TypeError("trust must be LocalQ1EligibilityTrust")
        if type(self.production_authorized) is not bool:  # noqa: E721
            raise TypeError("production_authorized must be bool")
        normalized = {}
        for token, decision in self.decisions.items():
            if token != decision.pair_identity_sha256:
                raise LocalQ1GeometryQualificationError(
                    "eligibility decision lookup key changed"
                )
            normalized[token] = decision
        order = tuple(self.decision_order)
        if len(order) != len(set(order)) or set(order) != set(normalized):
            raise LocalQ1GeometryQualificationError(
                "eligibility decision order is not unique/exhaustive"
            )
        object.__setattr__(self, "decisions", MappingProxyType(normalized))
        object.__setattr__(self, "decision_order", order)
        frozen_index = _deep_freeze(_portable(dict(self.index), "eligibility_index"))
        frozen_receipt = _deep_freeze(
            _portable(dict(self.receipt), "eligibility_receipt")
        )
        if (
            _content_sha256(frozen_index, "eligibility index")
            != self.trust.eligibility_index_content_sha256
            or _content_sha256(frozen_receipt, "eligibility receipt")
            != self.trust.eligibility_receipt_content_sha256
        ):
            raise LocalQ1GeometryQualificationError(
                "eligibility authority content changed before deep freeze"
            )
        object.__setattr__(self, "index", frozen_index)
        object.__setattr__(self, "receipt", frozen_receipt)

    def require_decision(self, pair_identity_sha256: str) -> LocalQ1EligibilityDecision:
        _require_sha256(pair_identity_sha256, "frontier pair identity commitment")
        try:
            return self.decisions[pair_identity_sha256]
        except KeyError as exc:
            raise LocalQ1GeometryQualificationError(
                "complete lazy frontier reached a pair without an external decision"
            ) from exc

    def __call__(self, item: "LocalQ1PairGeometryInput") -> LocalQ1EligibilityDecision:
        if not isinstance(item, LocalQ1PairGeometryInput):
            raise TypeError(
                "eligibility authority accepts LocalQ1PairGeometryInput only"
            )
        return self.require_decision(item.pair_identity_sha256)

    def assert_selection_proof(self, observed: Mapping[str, Any]) -> None:
        if _content_sha256(self.receipt, "eligibility receipt") != (
            self.trust.eligibility_receipt_content_sha256
        ):
            raise LocalQ1GeometryQualificationError(
                "eligibility receipt changed after authority construction"
            )
        expected = self.receipt.get("selection_proof")
        portable = _portable(dict(observed), "observed_selection_proof")
        if not isinstance(expected, Mapping) or _canonical_json(
            expected
        ) != _canonical_json(portable):
            raise LocalQ1GeometryQualificationError(
                "refreeze selection replay differs from eligibility authority"
            )




@dataclass(frozen=True)
class LocalQ1RouteAConfigContractAuthority:
    """Dependency-light view of the exact externally locked Route-A config."""

    document: Mapping[str, Any]
    geometry_batch_config_sha256: str = ROUTE_A_GEOMETRY_BATCH_CONFIG_SHA256
    planning_guard_config_sha256: str = ROUTE_A_PLANNING_GUARD_CONFIG_SHA256
    file_sha256: str = ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256
    content_sha256: str = ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256

    def __post_init__(self) -> None:
        if (
            self.geometry_batch_config_sha256
            != ROUTE_A_GEOMETRY_BATCH_CONFIG_SHA256
            or self.planning_guard_config_sha256
            != ROUTE_A_PLANNING_GUARD_CONFIG_SHA256
            or self.file_sha256 != ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256
            or self.content_sha256 != ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256
        ):
            raise LocalQ1GeometryQualificationError(
                "Route-A config contract external commitments changed"
            )
        frozen = _deep_freeze(
            _portable(dict(self.document), "route_a_config_authority")
        )
        if _content_sha256(frozen, "Route-A config authority") != self.content_sha256:
            raise LocalQ1GeometryQualificationError(
                "Route-A config authority changed before deep freeze"
            )
        object.__setattr__(self, "document", frozen)


def load_local_q1_route_a_config_authority_payload_contract(
    *,
    payload: bytes,
    expected_file_sha256: str = ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256,
    expected_content_sha256: str = ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256,
) -> LocalQ1RouteAConfigContractAuthority:
    """Validate every replay-relevant field/hash without heavy geometry imports."""

    if (
        expected_file_sha256 != ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256
        or expected_content_sha256 != ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256
    ):
        raise LocalQ1GeometryQualificationError(
            "only the frozen Route-A config authority may be loaded"
        )
    if type(payload) is not bytes:  # noqa: E721
        raise LocalQ1GeometryQualificationError(
            "Route-A config authority payload must be immutable bytes"
        )
    if not hmac.compare_digest(_sha256(payload), expected_file_sha256):
        raise LocalQ1GeometryQualificationError(
            "Route-A config authority external file hash mismatch"
        )
    document = _parse_strict_json(payload, "Route-A config authority")
    expected_root = {
        "schema_version",
        "status",
        "scope",
        "authority_contract",
        "hash_contract",
        "route_a_policy_lock",
        "geometry_batch_config",
        "geometry_batch_config_sha256",
        "planning_guard_config",
        "planning_guard_config_sha256",
        "component_config_hashes",
        "production_population_contract",
        "content_sha256",
    }
    if (
        set(document) != expected_root
        or document.get("schema_version") != ROUTE_A_CONFIG_AUTHORITY_SCHEMA_VERSION
        or document.get("status")
        != "preregistered_exact_configs_and_population_before_real_feasibility_result"
        or _content_sha256(document, "Route-A config authority")
        != expected_content_sha256
    ):
        raise LocalQ1GeometryQualificationError(
            "Route-A config authority schema/status/content changed"
        )
    if document.get("scope") != {
        "decision_made_before_real_route_a_feasibility_result": True,
        "historical_test_pair_stream_read": False,
        "mask_pixels_decoded": False,
        "model_or_gpu_used": False,
        "sealed_real_read": False,
    } or document.get("authority_contract") != {
        "config_values_caller_substitutable": False,
        "fixture_scale_positive_values_authorize_real_scan": False,
        "production_driver_must_bind_this_file_and_content_sha256_externally": True,
        "production_driver_must_recompute_all_component_and_combined_hashes": True,
    }:
        raise LocalQ1GeometryQualificationError(
            "Route-A config authority scope/contract changed"
        )
    if document.get("hash_contract") != {
        "algorithm": "sha256",
        "canonicalization": (
            "json_utf8_ensure_ascii_false_sort_keys_true_separators_comma_colon_"
            "allow_nan_false_no_trailing_newline"
        ),
        "excluded_top_level_fields": ["content_sha256"],
    } or document.get("route_a_policy_lock") != {
        "schema_version": ROUTE_A_POLICY_SCHEMA_VERSION,
        "file_sha256": ROUTE_A_POLICY_FILE_SHA256,
        "content_sha256": ROUTE_A_POLICY_CONTENT_SHA256,
    }:
        raise LocalQ1GeometryQualificationError(
            "Route-A config authority hash/policy contract changed"
        )

    geometry = document.get("geometry_batch_config")
    planning = document.get("planning_guard_config")
    if (
        not isinstance(geometry, Mapping)
        or not isinstance(planning, Mapping)
        or set(planning) != {"inventory_config", "batch_plan_config"}
        or not isinstance(geometry.get("geometry"), Mapping)
        or not isinstance(planning.get("inventory_config"), Mapping)
        or not isinstance(planning.get("batch_plan_config"), Mapping)
    ):
        raise LocalQ1GeometryQualificationError(
            "Route-A embedded config contract changed"
        )
    observed_geometry = _sha256(_canonical_json(geometry))
    observed_planning = _sha256(_canonical_json(planning))
    observed_inventory = _sha256(
        _canonical_json(planning["inventory_config"])
    )
    observed_batch_plan = _sha256(
        _canonical_json(planning["batch_plan_config"])
    )
    observed_candidate = _sha256(
        _canonical_json(
            {
                "artifact_version": FRAGMENT_GEOMETRY_ARTIFACT_VERSION,
                "geometry_version": GEOMETRY_VERSION,
                "geometry_config": geometry["geometry"],
            }
        )
    )
    if (
        observed_geometry != ROUTE_A_GEOMETRY_BATCH_CONFIG_SHA256
        or document.get("geometry_batch_config_sha256") != observed_geometry
        or observed_planning != ROUTE_A_PLANNING_GUARD_CONFIG_SHA256
        or document.get("planning_guard_config_sha256") != observed_planning
        or observed_inventory != ROUTE_A_INVENTORY_CONFIG_SHA256
        or observed_batch_plan != ROUTE_A_BATCH_PLAN_CONFIG_SHA256
        or observed_candidate != ROUTE_A_CANDIDATE_BUILDER_CONFIG_SHA256
        or document.get("component_config_hashes")
        != {
            "candidate_builder_config_sha256": observed_candidate,
            "local_cache_inventory_config_sha256": observed_inventory,
            "local_q1_batch_plan_config_sha256": observed_batch_plan,
        }
        or planning["inventory_config"].get("geometry") != geometry["geometry"]
    ):
        raise LocalQ1GeometryQualificationError(
            "Route-A embedded config commitment recomputation failed"
        )
    if document.get("production_population_contract") != {
        "selection_seed": "local-q1-260828",
        "training_dataset_order": [
            "mm_augmented",
            "eccv_1113data",
            "canonical_new",
        ],
        "training_label_order": ["negative", "positive"],
        "training_target_per_dataset_label": 512,
        "training_total": 3072,
        "validation_caps": {"mm_augmented": 32, "eccv_1113data": 4},
        "validation_component_counts": {
            "mm_augmented": 55,
            "eccv_1113data": 533,
        },
        "validation_total": 3892,
        "selected_pair_total": 6964,
        "required_all_four_direction_pair_count": 6964,
        "required_fragment_pair_direction_resource_failure_count": 0,
        "required_train_validation_component_member_content_overlap_count": 0,
    }:
        raise LocalQ1GeometryQualificationError(
            "Route-A production population contract changed"
        )
    return LocalQ1RouteAConfigContractAuthority(
        document=document,
        geometry_batch_config_sha256=observed_geometry,
        planning_guard_config_sha256=observed_planning,
    )


def _validate_immutable_payload_lock(
    *,
    payload: bytes,
    expected_bytes: int,
    expected_file_sha256: str,
    name: str,
) -> None:
    if type(payload) is not bytes:  # noqa: E721
        raise LocalQ1GeometryQualificationError(
            "{} payload must be immutable bytes".format(name)
        )
    if type(expected_bytes) is not int or expected_bytes <= 0:  # noqa: E721
        raise LocalQ1GeometryQualificationError(
            "{} expected bytes must be a positive integer".format(name)
        )
    _require_sha256(expected_file_sha256, name + " expected file hash")
    if len(payload) != expected_bytes:
        raise LocalQ1GeometryQualificationError(
            "{} external byte length mismatch".format(name)
        )
    if not hmac.compare_digest(_sha256(payload), expected_file_sha256):
        raise LocalQ1GeometryQualificationError(
            "{} external file hash mismatch".format(name)
        )


def load_local_q1_route_a_policy_payload(
    *,
    payload: bytes,
    expected_bytes: int,
    expected_file_sha256: str = ROUTE_A_POLICY_FILE_SHA256,
    expected_content_sha256: str = ROUTE_A_POLICY_CONTENT_SHA256,
) -> Mapping[str, Any]:
    """Validate captured corrected Route-A policy bytes without reopening a path."""

    if (
        expected_file_sha256 != ROUTE_A_POLICY_FILE_SHA256
        or expected_content_sha256 != ROUTE_A_POLICY_CONTENT_SHA256
    ):
        raise LocalQ1GeometryQualificationError(
            "only the corrected Route-A policy v0.2 may authorize eligibility"
        )
    _validate_immutable_payload_lock(
        payload=payload,
        expected_bytes=expected_bytes,
        expected_file_sha256=expected_file_sha256,
        name="corrected Route-A policy",
    )
    policy = _parse_strict_json(payload, "corrected Route-A policy")
    if (
        policy.get("schema_version") != ROUTE_A_POLICY_SCHEMA_VERSION
        or policy.get("status")
        != "corrected_route_a_preregistered_before_real_feasibility_result"
        or _content_sha256(policy, "corrected Route-A policy")
        != expected_content_sha256
    ):
        raise LocalQ1GeometryQualificationError(
            "corrected Route-A policy content/status changed"
        )
    return _deep_freeze(_portable(dict(policy), "corrected_route_a_policy"))


def load_local_q1_geometry_eligibility_authority_payloads_v2(
    *,
    index_payload: bytes,
    receipt_payload: bytes,
    policy_payload: bytes,
    config_authority_payload: bytes,
    trust: LocalQ1EligibilityTrust,
    expected_index_bytes: int,
    expected_index_file_sha256: str,
    expected_index_content_sha256: str,
    expected_receipt_bytes: int,
    expected_receipt_file_sha256: str,
    expected_receipt_content_sha256: str,
    expected_policy_bytes: int,
    expected_policy_file_sha256: str,
    expected_policy_content_sha256: str,
    expected_config_authority_bytes: int,
    expected_config_authority_file_sha256: str,
    expected_config_authority_content_sha256: str,
    _fixture_scale_test_only: bool = False,
) -> LocalQ1GeometryEligibilityAuthority:
    """Validate four captured authority payloads without reopening their paths."""

    if not isinstance(trust, LocalQ1EligibilityTrust):
        raise TypeError("trust must be LocalQ1EligibilityTrust")
    if type(_fixture_scale_test_only) is not bool:  # noqa: E721
        raise TypeError("_fixture_scale_test_only must be bool")
    for value, name in (
        (expected_index_file_sha256, "eligibility index expected file hash"),
        (expected_index_content_sha256, "eligibility index expected content hash"),
        (expected_receipt_file_sha256, "eligibility receipt expected file hash"),
        (expected_receipt_content_sha256, "eligibility receipt expected content hash"),
        (expected_policy_file_sha256, "Route-A policy expected file hash"),
        (expected_policy_content_sha256, "Route-A policy expected content hash"),
        (
            expected_config_authority_file_sha256,
            "Route-A config authority expected file hash",
        ),
        (
            expected_config_authority_content_sha256,
            "Route-A config authority expected content hash",
        ),
    ):
        _require_sha256(value, name)
    if (
        expected_index_file_sha256 != trust.eligibility_index_file_sha256
        or expected_index_content_sha256 != trust.eligibility_index_content_sha256
        or expected_receipt_file_sha256 != trust.eligibility_receipt_file_sha256
        or expected_receipt_content_sha256 != trust.eligibility_receipt_content_sha256
        or expected_policy_file_sha256 != trust.eligibility_policy_file_sha256
        or expected_policy_content_sha256 != trust.eligibility_policy_content_sha256
        or expected_config_authority_file_sha256 != trust.config_authority_file_sha256
        or expected_config_authority_content_sha256
        != trust.config_authority_content_sha256
    ):
        raise LocalQ1GeometryQualificationError(
            "eligibility payload locks differ from trust"
        )
    _validate_immutable_payload_lock(
        payload=index_payload,
        expected_bytes=expected_index_bytes,
        expected_file_sha256=expected_index_file_sha256,
        name="eligibility index",
    )
    _validate_immutable_payload_lock(
        payload=receipt_payload,
        expected_bytes=expected_receipt_bytes,
        expected_file_sha256=expected_receipt_file_sha256,
        name="eligibility receipt",
    )
    _validate_immutable_payload_lock(
        payload=policy_payload,
        expected_bytes=expected_policy_bytes,
        expected_file_sha256=expected_policy_file_sha256,
        name="corrected Route-A policy",
    )
    _validate_immutable_payload_lock(
        payload=config_authority_payload,
        expected_bytes=expected_config_authority_bytes,
        expected_file_sha256=expected_config_authority_file_sha256,
        name="Route-A config authority",
    )

    config_authority = load_local_q1_route_a_config_authority_payload_contract(
        payload=config_authority_payload,
        expected_file_sha256=trust.config_authority_file_sha256,
        expected_content_sha256=trust.config_authority_content_sha256,
    )
    if (
        config_authority.geometry_batch_config_sha256
        != trust.geometry_batch_config_sha256
        or config_authority.document["planning_guard_config_sha256"]
        != trust.planning_guard_config_sha256
    ):
        raise LocalQ1GeometryQualificationError(
            "eligibility trust differs from Route-A config authority"
        )

    load_local_q1_route_a_policy_payload(
        payload=policy_payload,
        expected_bytes=expected_policy_bytes,
        expected_file_sha256=trust.eligibility_policy_file_sha256,
        expected_content_sha256=trust.eligibility_policy_content_sha256,
    )

    index_bytes = index_payload
    receipt_bytes = receipt_payload
    index = _strict_json(index_bytes, "eligibility index")
    receipt = _strict_json(receipt_bytes, "eligibility receipt")
    if not hmac.compare_digest(
        _content_sha256(index, "eligibility index"),
        trust.eligibility_index_content_sha256,
    ):
        raise LocalQ1GeometryQualificationError(
            "eligibility index external content hash mismatch"
        )
    if not hmac.compare_digest(
        _content_sha256(receipt, "eligibility receipt"),
        trust.eligibility_receipt_content_sha256,
    ):
        raise LocalQ1GeometryQualificationError(
            "eligibility receipt external content hash mismatch"
        )

    expected_index_fields = {
        "schema_version",
        "status",
        "qualification_contract",
        "decision_count",
        "decision_order_sha256",
        "decision_set_sha256",
        "decisions",
        "content_sha256",
    }
    if set(index) != expected_index_fields or (
        index.get("schema_version") != LOCAL_Q1_ELIGIBILITY_INDEX_SCHEMA_VERSION
        or index.get("status")
        != "complete_lazy_frontier_decisions_label_direction_blind"
    ):
        raise LocalQ1GeometryQualificationError("eligibility index contract changed")
    _validate_qualification_contract(
        index.get("qualification_contract"),
        expected_geometry_config_sha256=trust.geometry_batch_config_sha256,
    )
    decisions_value = index.get("decisions")
    if not isinstance(decisions_value, list) or index.get("decision_count") != len(
        decisions_value
    ):
        raise LocalQ1GeometryQualificationError("eligibility decision count changed")
    decisions = {}
    order = []
    for row in decisions_value:
        if not isinstance(row, Mapping) or set(row) != {
            "pair_identity_sha256",
            "assessment",
        }:
            raise LocalQ1GeometryQualificationError(
                "eligibility decision row schema changed"
            )
        token = _require_sha256(row["pair_identity_sha256"], "decision pair token")
        if token in decisions:
            raise LocalQ1GeometryQualificationError("duplicate eligibility decision")
        decision = LocalQ1EligibilityDecision(
            pair_identity_sha256=token,
            assessment=_parse_assessment(row["assessment"]),
        )
        decisions[token] = decision
        order.append(token)
    if index.get("decision_order_sha256") != _sha256(
        _canonical_json(order)
    ) or index.get("decision_set_sha256") != _sha256(_canonical_json(sorted(order))):
        raise LocalQ1GeometryQualificationError(
            "eligibility decision order/set commitment mismatch"
        )

    expected_receipt_fields = {
        "schema_version",
        "status",
        "external_locks",
        "index_lock",
        "frontier_contract",
        "selection_proof",
        "decision_census",
        "qualification_scratch_cache",
        "partial_cache_exclusion",
        "scope",
        "portable_privacy",
        "content_sha256",
    }
    expected_status = (
        "fixture_scale_test_only_not_authorized_for_real_scan"
        if _fixture_scale_test_only
        else "qualified_lazy_complete_frontier_no_model_no_test"
    )
    if set(receipt) != expected_receipt_fields or (
        receipt.get("schema_version") != LOCAL_Q1_ELIGIBILITY_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != expected_status
    ):
        raise LocalQ1GeometryQualificationError("eligibility receipt contract changed")
    expected_locks = trust.portable_dict()
    # The two eligibility document locks are represented by index_lock and the
    # enclosing external authority, so external_locks contains upstream/config
    # locks only and cannot circularly embed its own file hash.
    for name in (
        "eligibility_index_file_sha256",
        "eligibility_index_content_sha256",
        "eligibility_receipt_file_sha256",
        "eligibility_receipt_content_sha256",
    ):
        expected_locks.pop(name)
    if receipt.get("external_locks") != expected_locks:
        raise LocalQ1GeometryQualificationError(
            "eligibility receipt upstream/config locks differ from authority"
        )
    if receipt.get("index_lock") != {
        "bytes": len(index_bytes),
        "file_sha256": trust.eligibility_index_file_sha256,
        "content_sha256": trust.eligibility_index_content_sha256,
        "decision_count": len(order),
        "decision_order_sha256": index["decision_order_sha256"],
        "decision_set_sha256": index["decision_set_sha256"],
    }:
        raise LocalQ1GeometryQualificationError(
            "eligibility receipt/index cross-lock mismatch"
        )
    if receipt.get("partial_cache_exclusion") != {
        "partial_cache_as_membership_input_permitted": False,
        "partial_cache_role_count": 0,
        "partial_cache_tree_accessed": False,
    }:
        raise LocalQ1GeometryQualificationError(
            "eligibility receipt did not exclude the partial cache"
        )
    scope = receipt.get("scope")
    if scope != {
        "input_modality": "canonical_bool_mask_only",
        "pair_stream_splits_read": ["train", "val"],
        "historical_test_pair_stream_read": False,
        "sealed_real_read": False,
        "model_imported": False,
        "model_executed": False,
        "supervision_fields_read_by_geometry_qualification": [],
        "fixture_scale_test_only": _fixture_scale_test_only,
    }:
        raise LocalQ1GeometryQualificationError("eligibility scope contract changed")
    frontier = receipt.get("frontier_contract")
    if frontier != {
        "policy": "complete_deterministic_frontier_consumed_lazily",
        "arbitrary_fixed_reserve_cutoff_used": False,
        "manual_exclusions_used": False,
        "unexplored_higher_priority_record_count": 0,
        "decision_requirement": (
            "every_candidate_preceding_or_equal_to_each_selected_frontier_slot"
        ),
    }:
        raise LocalQ1GeometryQualificationError("eligibility frontier contract changed")
    _validate_selection_proof(
        receipt.get("selection_proof"),
        decision_count=len(decisions),
        require_production=not _fixture_scale_test_only,
    )
    _validate_decision_census(receipt.get("decision_census"), decisions)
    _validate_scratch_cache_evidence(
        receipt.get("qualification_scratch_cache"),
        decision_count=len(decisions),
        selected_record_count=(
            receipt["selection_proof"]["selected_training_count"]
            + receipt["selection_proof"]["selected_validation_count"]
        ),
        selected_pair_commitment_order_sha256=receipt["selection_proof"][
            "selected_pair_commitment_order_sha256"
        ],
        selected_pair_commitment_set_sha256=receipt["selection_proof"][
            "selected_pair_commitment_set_sha256"
        ],
    )
    if receipt.get("portable_privacy") != {
        "absolute_paths_present": False,
        "member_group_component_fragment_or_pair_identifiers_present": False,
        "supervision_labels_or_historical_directions_present_in_index": False,
        "opaque_pair_commitments_present": True,
        "aggregate_population_difference_only": True,
        "secrets_present": False,
    }:
        raise LocalQ1GeometryQualificationError(
            "eligibility portable-privacy contract changed"
        )
    _portable(receipt, "eligibility_receipt")
    return LocalQ1GeometryEligibilityAuthority(
        decisions=decisions,
        decision_order=tuple(order),
        index=index,
        receipt=receipt,
        trust=trust,
        production_authorized=not _fixture_scale_test_only,
    )


@dataclass(frozen=True)
class LocalQ1PairGeometryInput:
    """Supervision-free input exposed to a lazy geometry decision function."""

    fragment_a: MaskMemberRef
    fragment_b: MaskMemberRef
    pair_id: str
    pair_identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.fragment_a, MaskMemberRef) or not isinstance(
            self.fragment_b, MaskMemberRef
        ):
            raise TypeError("pair geometry input requires two mask references")
        if not isinstance(self.pair_id, str) or not re.fullmatch(
            r"pair/sha256/[0-9a-f]{64}", self.pair_id
        ):
            raise ValueError("pair geometry input requires a label-blind pair_id")
        _require_sha256(self.pair_identity_sha256, "pair geometry identity")
        # A pair commitment is unordered.  Canonicalize the two endpoint
        # references before any geometry is decoded so a reversed metadata row
        # cannot make the first observed assessment win by traversal accident.
        references = sorted(
            (self.fragment_a, self.fragment_b), key=_reference_geometry_order_key
        )
        object.__setattr__(self, "fragment_a", references[0])
        object.__setattr__(self, "fragment_b", references[1])


def _reference_geometry_order_key(reference: MaskMemberRef) -> str:
    if not isinstance(reference, MaskMemberRef):
        raise TypeError("geometry endpoint must be MaskMemberRef")
    return _sha256(
        _canonical_json(
            {
                "binding_logical_id": reference.binding.logical_id,
                "binding_sha256": reference.binding.sha256,
                "archive_member": reference.archive_member,
                "threshold_rule": reference.threshold_rule,
                "content_sha256": reference.content_sha256,
            }
        )
    )



__all__ = [
    "CANONICAL_ROUTE_A_CONFIG_AUTHORITY_PATH",
    "CANONICAL_ROUTE_A_POLICY_PATH",
    "LOCAL_Q1_ELIGIBILITY_INDEX_SCHEMA_VERSION",
    "LOCAL_Q1_ELIGIBILITY_RECEIPT_SCHEMA_VERSION",
    "LocalQ1EligibilityDecision",
    "LocalQ1EligibilityTrust",
    "LocalQ1ExternalFileLock",
    "LocalQ1GeometryEligibilityAuthority",
    "LocalQ1GeometryQualificationError",
    "LocalQ1PairGeometryInput",
    "LocalQ1RouteAConfigContractAuthority",
    "ROUTE_A_CONFIG_AUTHORITY_CONTENT_SHA256",
    "ROUTE_A_CONFIG_AUTHORITY_FILE_SHA256",
    "ROUTE_A_GEOMETRY_BATCH_CONFIG_SHA256",
    "ROUTE_A_PLANNING_GUARD_CONFIG_SHA256",
    "ROUTE_A_POLICY_CONTENT_SHA256",
    "ROUTE_A_POLICY_FILE_SHA256",
    "load_local_q1_geometry_eligibility_authority_payloads_v2",
    "load_local_q1_route_a_config_authority_payload_contract",
    "load_local_q1_route_a_policy_payload",
]
