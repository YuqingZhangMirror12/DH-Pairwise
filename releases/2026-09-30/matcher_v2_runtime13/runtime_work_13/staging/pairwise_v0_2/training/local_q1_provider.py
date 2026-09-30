"""Frozen candidate-complexity plan and read-only LOCAL-Q1 batch provider.

The cache builder proves that the complete LOCAL-Q1 fragment cache exists and
is immutable.  This module is the next boundary: it combines those same
role-neutral artifacts in all four upright directions, plans padding-aware
batches before training, and later materializes only exact frozen batches from
that same cache.  It imports no model, optimizer, checkpoint, or runner.

The plan never uses ``label`` or ``direction_b_wrt_a``.  Supervision remains in
the :class:`TrainingPairRecord` values passed to ``prepare`` and can affect
targets, but it cannot affect record ordering, bucketing, candidate selection,
or allocation.  Portable receipts contain anonymous per-record candidate
counts, sequence lengths, buckets, exact costs, ordinals, and commitments so
their resource claims can be independently recomputed.  They contain no raw
archive paths/members or pair/component/fragment/dataset IDs.  Anonymous
complexity can still fingerprint a record; this is data minimization, not a
cryptographic privacy guarantee.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from torch import Tensor

from staging.pairwise_v0_2.geometry import (
    CHANNEL_ORDER,
    DEFAULT_DIRECTION_ORDER,
    CandidateBuilderConfig,
    ContourKeypointConfig,
    CorrosionConfig,
    GeometryStatus,
    combine_fragment_results,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.fragment_geometry_cache import (
    FragmentGeometryCacheLookup,
    load_or_build_fragment_geometry,
)
from staging.pairwise_v0_2.training.geometry_batch import (
    KEYPOINT_REPRESENTATION,
    MULTIRUN_REPRESENTATION,
    GeometryBatchConfig,
    RaggedGeometryBatch,
    build_geometry_batch,
    geometry_cache_key,
)
from staging.pairwise_v0_2.training.geometry_cache import (
    GeometryArtifactCache,
    fragment_cache_identity,
)
from staging.pairwise_v0_2.training.local_cache_inventory import (
    LocalCacheInventoryConfig,
)
from staging.pairwise_v0_2.training.local_q1_cache_builder import (
    OpenedLocalQ1Cache,
)
from staging.pairwise_v0_2.training.local_q1_pair_qualification import (
    qualification_contract,
    qualify_local_q1_pair_geometry,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    ApprovedCoarsePreprocessing,
    BatchProviderContract,
    PreparedAblationBatch,
    record_sequence_fingerprint,
)


LOCAL_Q1_BATCH_PLAN_SCHEMA_VERSION = "dunhuang-local-q1-batch-plan/0.4"
LOCAL_Q1_BATCH_PROVIDER_VERSION = "dunhuang-local-q1-read-only-provider/0.7"
LOCAL_Q1_PROVIDER_RUNTIME_SCHEMA_VERSION = "dunhuang-local-q1-provider-runtime/0.7"
LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION = (
    "dunhuang-local-q1-planned-safety-metadata/0.2"
)
LOCAL_Q1_PROVENANCE_SAFETY_SCAN_VERSION = "dunhuang-local-q1-provenance-safety-scan/0.1"
LOCAL_Q1_PROVENANCE_SAFETY_MAX_DEPTH = 32
LOCAL_Q1_PROVENANCE_SAFETY_MAX_NODES = 4096
LOCAL_Q1_VALIDATION_ASSIGNMENT_SCHEMA_VERSION = (
    "dunhuang-local-q1-validation-assignment/0.1"
)
LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE = (
    "dunhuang-local-q1/validation-component-assignment/v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VALIDATION_PHASES = (
    "validation_select",
    "validation_calibration",
    "validation_report",
)
_FORMAL_PHASES = ("train",) + _VALIDATION_PHASES
_PHASE_TO_SPLIT = MappingProxyType(
    {
        "train": "train",
        "validation_select": "val",
        "validation_calibration": "val",
        "validation_report": "val",
    }
)
_FORBIDDEN_PORTABLE_KEYS = frozenset(
    {
        "archive_member",
        "component_id",
        "dataset_id",
        "fragment_id",
        "group_id",
        "member_path",
        "pair_id",
        "path",
        "sample_id",
        "secret",
    }
)


class LocalQ1ProviderError(RuntimeError):
    """A LOCAL-Q1 plan/provider invariant failed before model execution."""


def _canonical_json(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: normalize(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(value) for value in item]
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


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("{} must be lowercase SHA-256".format(name))
    return value


def _portable(value: Any, location: str = "receipt") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and (
            value.startswith(("/", "~/", "file://"))
            or re.match(r"^[A-Za-z]:[\\/]", value)
        ):
            raise LocalQ1ProviderError(
                "{} contains a machine-local path".format(location)
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LocalQ1ProviderError("{} contains NaN/Inf".format(location))
        return value
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LocalQ1ProviderError(
                    "{} contains a non-string key".format(location)
                )
            if key.casefold() in _FORBIDDEN_PORTABLE_KEYS:
                raise LocalQ1ProviderError(
                    "{} exposes a local or row identity".format(location)
                )
            output[key] = _portable(item, "{}.{}".format(location, key))
        return output
    if isinstance(value, (tuple, list)):
        return [
            _portable(item, "{}[{}]".format(location, index))
            for index, item in enumerate(value)
        ]
    raise LocalQ1ProviderError(
        "{} is not JSON-portable: {}".format(location, type(value).__name__)
    )


@dataclass(frozen=True)
class LocalQ1ProvenanceSafetyMarkers:
    """Identity-free outcome of the bounded shared provenance scan."""

    top_level_sealed_marker_explicit_false: bool
    sealed_scope_marker: bool
    historical_test_marker: bool

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool  # noqa: E721
            for value in (
                self.top_level_sealed_marker_explicit_false,
                self.sealed_scope_marker,
                self.historical_test_marker,
            )
        ):
            raise TypeError("provenance safety markers must be exact bools")


def local_q1_provenance_safety_policy() -> Mapping[str, Any]:
    """Return the immutable, portable policy shared by provider and runner."""

    return MappingProxyType(
        {
            "schema_version": LOCAL_Q1_PROVENANCE_SAFETY_SCAN_VERSION,
            "algorithm": "bounded_recursive_mapping_list_tuple_scope_markers_v1",
            "mapping_key_requirement": "built_in_str",
            "mapping_key_normalization": "strip_then_casefold",
            "false_like_values": (
                "literal_false",
                "null",
                "numeric_zero",
                "empty_string_after_strip",
                "casefold_false_string",
                "casefold_no_string",
            ),
            "top_level_required_literal_false_key": ("real_dunhuang_sealed_test"),
            "containers": ("Mapping", "list", "tuple"),
            "supported_scalars": ("null", "bool", "int", "finite_float", "str"),
            "max_depth": LOCAL_Q1_PROVENANCE_SAFETY_MAX_DEPTH,
            "max_nodes": LOCAL_Q1_PROVENANCE_SAFETY_MAX_NODES,
            "cycle_policy": "reject",
            "depth_budget_policy": "reject",
            "node_budget_policy": "reject",
            "unsupported_value_policy": "reject",
            "raw_keys_values_paths_or_scan_counts_emitted": False,
        }
    )


def _false_scope_marker(value: Any) -> bool:
    if value is False or value is None:
        return True
    if type(value) in {int, float}:  # noqa: E721
        return value == 0
    if isinstance(value, str):
        return value.strip().casefold() in {"", "false", "no"}
    return False


def scan_local_q1_provenance_safety(
    provenance: Mapping[str, Any],
) -> LocalQ1ProvenanceSafetyMarkers:
    """Boundedly scan recursive provenance without emitting its contents.

    Mapping/list/tuple containers are traversed depth-first.  Cycles, excessive
    depth or node count, non-string keys, non-finite floats, and unsupported
    values are rejected rather than treated as safe.
    """

    if not isinstance(provenance, Mapping):
        raise LocalQ1ProviderError("provenance safety scan requires a Mapping")
    try:
        top_level_explicit_false = (
            "real_dunhuang_sealed_test" in provenance
            and provenance["real_dunhuang_sealed_test"] is False
        )
    except Exception as exc:
        raise LocalQ1ProviderError(
            "provenance safety scan top-level marker access failed"
        ) from exc
    active_containers = set()
    visited_nodes = 0
    sealed_scope_marker = False
    historical_test_marker = False
    split_keys = {
        "split",
        "source_split",
        "upstream_split",
        "experiment_split",
    }
    historical_boolean_keys = {"is_test", "test_record", "withheld_test"}

    def visit(value: Any, depth: int) -> None:
        nonlocal visited_nodes, sealed_scope_marker, historical_test_marker
        if depth > LOCAL_Q1_PROVENANCE_SAFETY_MAX_DEPTH:
            raise LocalQ1ProviderError("provenance safety scan depth exceeded")
        visited_nodes += 1
        if visited_nodes > LOCAL_Q1_PROVENANCE_SAFETY_MAX_NODES:
            raise LocalQ1ProviderError("provenance safety scan node budget exceeded")
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active_containers:
                raise LocalQ1ProviderError("provenance safety scan found a cycle")
            active_containers.add(identity)
            try:
                for raw_key, item in value.items():
                    if type(raw_key) is not str:  # noqa: E721
                        raise LocalQ1ProviderError(
                            "provenance safety scan requires built-in string keys"
                        )
                    key = raw_key.strip().casefold()
                    false_like = _false_scope_marker(item)
                    if key == "real_dunhuang_sealed_test":
                        if item is not False:
                            sealed_scope_marker = True
                    elif "sealed" in key and not false_like:
                        sealed_scope_marker = True
                    if (
                        ("historical" in key and "test" in key)
                        or key in historical_boolean_keys
                    ) and not false_like:
                        historical_test_marker = True
                    if (
                        key in split_keys
                        and isinstance(item, str)
                        and "test" in item.casefold()
                    ):
                        historical_test_marker = True
                    visit(item, depth + 1)
            except LocalQ1ProviderError:
                raise
            except Exception as exc:
                raise LocalQ1ProviderError(
                    "provenance safety scan mapping traversal failed"
                ) from exc
            finally:
                active_containers.remove(identity)
            return
        if isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in active_containers:
                raise LocalQ1ProviderError("provenance safety scan found a cycle")
            active_containers.add(identity)
            try:
                for item in value:
                    visit(item, depth + 1)
            except LocalQ1ProviderError:
                raise
            except Exception as exc:
                raise LocalQ1ProviderError(
                    "provenance safety scan sequence traversal failed"
                ) from exc
            finally:
                active_containers.remove(identity)
            return
        if value is None or type(value) in {bool, int, str}:  # noqa: E721
            return
        if type(value) is float:  # noqa: E721
            if not math.isfinite(value):
                raise LocalQ1ProviderError(
                    "provenance safety scan rejects non-finite floats"
                )
            return
        raise LocalQ1ProviderError(
            "provenance safety scan rejects unsupported {}".format(type(value).__name__)
        )

    visit(provenance, 0)
    return LocalQ1ProvenanceSafetyMarkers(
        top_level_sealed_marker_explicit_false=top_level_explicit_false,
        sealed_scope_marker=sealed_scope_marker,
        historical_test_marker=historical_test_marker,
    )


def _content_sha256(receipt: Mapping[str, Any], name: str) -> str:
    value = dict(receipt)
    claimed = value.pop("content_sha256", None)
    _require_sha256(claimed, name + " content sha256")
    observed = _sha256(_canonical_json(value))
    if not hmac.compare_digest(claimed, observed):
        raise LocalQ1ProviderError("{} content hash mismatch".format(name))
    return claimed


def _strict_json_loads(payload: bytes, name: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise LocalQ1ProviderError("{} is not strict JSON".format(name)) from exc
    if not isinstance(value, Mapping):
        raise LocalQ1ProviderError("{} root must be an object".format(name))
    return value


def _close_loader(loader: Any) -> None:
    close = getattr(loader, "close", None)
    if callable(close):
        close()


def _reference_runtime_key(reference: MaskMemberRef) -> Tuple[str, ...]:
    """Physical decode key used only in memory and never serialized."""

    return (
        reference.binding.logical_id,
        reference.binding.sha256,
        reference.archive_member,
        reference.threshold_rule,
    )


def _validated_mask(
    loader: Callable[[MaskMemberRef], np.ndarray],
    reference: MaskMemberRef,
    max_pixels: int,
) -> np.ndarray:
    value = np.asarray(loader(reference))
    if value.ndim != 2 or value.size == 0 or value.dtype != np.bool_:
        raise LocalQ1ProviderError(
            "mask loader must return a non-empty two-dimensional bool array"
        )
    if int(value.size) > max_pixels:
        raise LocalQ1ProviderError("decoded mask exceeds configured pixel bound")
    return np.ascontiguousarray(value, dtype=np.bool_)


def _prefetch_planning_masks(
    records: Sequence[TrainingPairRecord],
    *,
    loader: Callable[[MaskMemberRef], np.ndarray],
    max_pixels: int,
    mask_memo: Dict[Tuple[str, ...], np.ndarray],
    counters: Dict[str, int],
) -> None:
    """Decode each physical planning mask once in archive-local member order."""

    references: Dict[Tuple[str, ...], MaskMemberRef] = {}
    for record in records:
        if not isinstance(record, TrainingPairRecord):
            raise TypeError("every planning record must be TrainingPairRecord")
        for reference in (record.fragment_a, record.fragment_b):
            references.setdefault(_reference_runtime_key(reference), reference)
    for key in sorted(references):
        if key in mask_memo:
            continue
        mask_memo[key] = _validated_mask(loader, references[key], max_pixels)
        counters["mask_decode_count"] += 1


@dataclass(frozen=True)
class LocalQ1ExternalLocks:
    """Experiment-owned trust roots required before planning or preparing.

    The source bundle commitment is supplied by the enclosing production run
    plan.  The remaining commitments are checked against the already reopened
    cache/population and their canonical portable receipts.  Batch-plan hashes
    are absent only during the initial plan build; the provider requires both.
    """

    run_plan_file_sha256: str
    run_plan_content_sha256: str
    source_bundle_manifest_sha256: str
    planning_config_receipt_file_sha256: str
    planning_config_receipt_content_sha256: str
    freeze_file_sha256: str
    freeze_content_sha256: str
    cache_receipt_file_sha256: str
    cache_receipt_content_sha256: str
    inventory_receipt_file_sha256: str
    inventory_receipt_content_sha256: str
    inventory_semantic_sha256: str
    build_receipt_file_sha256: str
    build_receipt_content_sha256: str
    batch_plan_file_sha256: Optional[str] = None
    batch_plan_content_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name.startswith("batch_plan_") and value is None:
                continue
            _require_sha256(value, name)
        if (self.batch_plan_file_sha256 is None) != (
            self.batch_plan_content_sha256 is None
        ):
            raise ValueError("batch-plan file/content locks must be supplied together")

    def without_batch_plan(self) -> Dict[str, str]:
        value = asdict(self)
        value.pop("batch_plan_file_sha256")
        value.pop("batch_plan_content_sha256")
        return value


@dataclass(frozen=True)
class LocalQ1BatchPlanConfig:
    """Deterministic packing and conservative resource-estimation settings."""

    packing_max_records: int = 256
    attention_head_count: int = 4
    score_element_bytes: int = 4
    planning_elements_per_second: int = 50_000_000
    time_safety_factor: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "packing_max_records",
            "attention_head_count",
            "score_element_bytes",
            "planning_elements_per_second",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if (
            type(self.time_safety_factor) is not float
            or not math.isfinite(self.time_safety_factor)
            or self.time_safety_factor < 1.0
        ):
            raise ValueError("time_safety_factor must be finite and >= 1")

    def portable_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenLocalQ1ValidationAssignment:
    """In-memory component partition plus its identity-free portable proof.

    ``phase_by_component`` is deliberately runtime-only.  The portable receipt
    commits to salted component hashes, counts, phases, and population ordinals
    without serializing a component identifier or a per-component hash list.
    """

    phase_by_component: Mapping[str, str]
    record_ordinals_by_phase: Mapping[str, Tuple[int, ...]]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        phases = dict(self.phase_by_component)
        if not phases or any(
            not isinstance(component, str)
            or not component.strip()
            or phase not in _VALIDATION_PHASES
            for component, phase in phases.items()
        ):
            raise LocalQ1ProviderError("validation component assignment is invalid")
        ordinals = {
            phase: tuple(self.record_ordinals_by_phase.get(phase, ()))
            for phase in _VALIDATION_PHASES
        }
        if set(self.record_ordinals_by_phase) != set(_VALIDATION_PHASES) or any(
            not values
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            )
            or tuple(sorted(set(values))) != values
            for values in ordinals.values()
        ):
            raise LocalQ1ProviderError("validation record assignment is invalid")
        receipt = _portable(dict(self.receipt), "validation_assignment")
        _validate_validation_assignment_receipt(receipt)
        if receipt["validation_component_count"] != len(phases):
            raise LocalQ1ProviderError(
                "validation component assignment count differs from receipt"
            )
        for phase in _VALIDATION_PHASES:
            if receipt["phases"][phase]["record_count"] != len(ordinals[phase]):
                raise LocalQ1ProviderError(
                    "validation record assignment count differs from receipt"
                )
        object.__setattr__(self, "phase_by_component", MappingProxyType(phases))
        object.__setattr__(self, "record_ordinals_by_phase", MappingProxyType(ordinals))
        object.__setattr__(self, "receipt", MappingProxyType(dict(receipt)))


def _validation_component_token(component_id: str) -> str:
    if not isinstance(component_id, str) or not component_id.strip():
        raise LocalQ1ProviderError("validation component_id is required")
    return _sha256(
        LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE.encode("utf-8")
        + b"\0"
        + component_id.encode("utf-8")
    )


def _validate_validation_assignment_receipt(receipt: Mapping[str, Any]) -> None:
    expected_root = {
        "schema_version",
        "status",
        "namespace",
        "algorithm",
        "phase_order",
        "fields_read",
        "supervision_fields_read",
        "validation_record_count",
        "validation_component_count",
        "population_component_set_sha256",
        "assignment_sha256",
        "validation_record_phase_order_sha256",
        "phases",
        "partition_proof",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_root:
        raise LocalQ1ProviderError(
            "validation assignment receipt has missing or extra fields"
        )
    if (
        receipt["schema_version"] != LOCAL_Q1_VALIDATION_ASSIGNMENT_SCHEMA_VERSION
        or receipt["status"]
        != "frozen_label_blind_component_partition_nonempty_disjoint_exhaustive"
        or receipt["namespace"] != LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE
        or receipt["algorithm"]
        != "sha256_namespace_component_rank_round_robin_three_phase_v1"
        or receipt["phase_order"] != list(_VALIDATION_PHASES)
        or receipt["fields_read"] != ["split", "component_id"]
        or receipt["supervision_fields_read"] != []
    ):
        raise LocalQ1ProviderError("validation assignment contract changed")
    for name in (
        "population_component_set_sha256",
        "assignment_sha256",
        "validation_record_phase_order_sha256",
    ):
        _require_sha256(receipt.get(name), "validation assignment " + name)
    record_count = receipt.get("validation_record_count")
    component_count = receipt.get("validation_component_count")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 3
        or isinstance(component_count, bool)
        or not isinstance(component_count, int)
        or component_count < 3
    ):
        raise LocalQ1ProviderError(
            "formal validation assignment requires at least three records/components"
        )
    phases = receipt.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != set(_VALIDATION_PHASES):
        raise LocalQ1ProviderError("validation assignment phases are incomplete")
    observed_records = 0
    observed_components = 0
    for phase in _VALIDATION_PHASES:
        value = phases[phase]
        if not isinstance(value, Mapping) or set(value) != {
            "phase",
            "record_count",
            "component_count",
            "component_set_sha256",
            "record_ordinal_set_sha256",
        }:
            raise LocalQ1ProviderError("validation phase assignment is malformed")
        if value["phase"] != phase:
            raise LocalQ1ProviderError("validation phase assignment identity changed")
        for name in ("record_count", "component_count"):
            count = value[name]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise LocalQ1ProviderError(
                    "validation phase assignment must be nonempty"
                )
        _require_sha256(
            value.get("component_set_sha256"),
            "validation phase component set sha256",
        )
        _require_sha256(
            value.get("record_ordinal_set_sha256"),
            "validation phase record ordinal set sha256",
        )
        observed_records += value["record_count"]
        observed_components += value["component_count"]
    if observed_records != record_count or observed_components != component_count:
        raise LocalQ1ProviderError("validation assignment totals are inconsistent")
    if receipt.get("partition_proof") != {
        "unit": "component_id",
        "all_phases_nonempty": True,
        "component_overlap_count": 0,
        "record_overlap_count": 0,
        "component_assignment_exhaustive": True,
        "record_assignment_exhaustive": True,
        "labels_or_directions_read": False,
    }:
        raise LocalQ1ProviderError("validation assignment proof changed")


def freeze_local_q1_validation_assignment(
    records: Sequence[TrainingPairRecord],
) -> FrozenLocalQ1ValidationAssignment:
    """Partition validation components without reading labels or directions."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("validation records must be a finite sequence")
    population = tuple(records)
    if not population:
        raise LocalQ1ProviderError("validation assignment population is empty")
    component_tokens: Dict[str, str] = {}
    for record in population:
        if not isinstance(record, TrainingPairRecord) or record.split != "val":
            raise LocalQ1ProviderError("validation assignment accepts val records only")
        token = _validation_component_token(record.component_id)
        previous = component_tokens.get(token)
        if previous is not None and previous != record.component_id:
            raise LocalQ1ProviderError("validation component hash collision")
        component_tokens[token] = record.component_id
    if len(component_tokens) < len(_VALIDATION_PHASES):
        raise LocalQ1ProviderError(
            "formal validation assignment requires at least three components"
        )
    phase_by_component: Dict[str, str] = {}
    component_tokens_by_phase = {phase: [] for phase in _VALIDATION_PHASES}
    assignment_rows = []
    for rank, token in enumerate(sorted(component_tokens)):
        phase = _VALIDATION_PHASES[rank % len(_VALIDATION_PHASES)]
        component = component_tokens[token]
        phase_by_component[component] = phase
        component_tokens_by_phase[phase].append(token)
        assignment_rows.append([token, phase])
    record_ordinals_by_phase = {phase: [] for phase in _VALIDATION_PHASES}
    record_phase_rows = []
    for ordinal, record in enumerate(population):
        token = _validation_component_token(record.component_id)
        phase = phase_by_component[record.component_id]
        record_ordinals_by_phase[phase].append(ordinal)
        record_phase_rows.append([ordinal, token, phase])
    if any(not values for values in record_ordinals_by_phase.values()):
        raise LocalQ1ProviderError("validation phase record population is empty")
    phase_receipts = {}
    for phase in _VALIDATION_PHASES:
        tokens = sorted(component_tokens_by_phase[phase])
        ordinals = tuple(record_ordinals_by_phase[phase])
        phase_receipts[phase] = {
            "phase": phase,
            "record_count": len(ordinals),
            "component_count": len(tokens),
            "component_set_sha256": _sha256(_canonical_json(tokens)),
            "record_ordinal_set_sha256": _sha256(_canonical_json(sorted(ordinals))),
        }
    receipt = {
        "schema_version": LOCAL_Q1_VALIDATION_ASSIGNMENT_SCHEMA_VERSION,
        "status": (
            "frozen_label_blind_component_partition_nonempty_disjoint_exhaustive"
        ),
        "namespace": LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE,
        "algorithm": ("sha256_namespace_component_rank_round_robin_three_phase_v1"),
        "phase_order": list(_VALIDATION_PHASES),
        "fields_read": ["split", "component_id"],
        "supervision_fields_read": [],
        "validation_record_count": len(population),
        "validation_component_count": len(component_tokens),
        "population_component_set_sha256": _sha256(
            _canonical_json(sorted(component_tokens))
        ),
        "assignment_sha256": _sha256(_canonical_json(assignment_rows)),
        "validation_record_phase_order_sha256": _sha256(
            _canonical_json(record_phase_rows)
        ),
        "phases": phase_receipts,
        "partition_proof": {
            "unit": "component_id",
            "all_phases_nonempty": True,
            "component_overlap_count": 0,
            "record_overlap_count": 0,
            "component_assignment_exhaustive": True,
            "record_assignment_exhaustive": True,
            "labels_or_directions_read": False,
        },
    }
    return FrozenLocalQ1ValidationAssignment(
        phase_by_component=phase_by_component,
        record_ordinals_by_phase={
            phase: tuple(values) for phase, values in record_ordinals_by_phase.items()
        },
        receipt=receipt,
    )


@dataclass(frozen=True)
class _RecordComplexity:
    split: str
    original_index: int
    record_geometry_sha256: str
    candidate_semantic_sha256: str
    candidate_count: int
    max_sequence_a: int
    max_sequence_b: int
    sequence_tokens_a: int
    sequence_tokens_b: int
    exact_attention_elements: int
    exact_affinity_elements: int
    exact_sinkhorn_elements: int
    keypoint_candidate_count: int
    keypoint_max_sequence_a: int
    keypoint_max_sequence_b: int
    keypoint_sequence_tokens_a: int
    keypoint_sequence_tokens_b: int
    keypoint_exact_attention_elements: int
    keypoint_exact_affinity_elements: int
    keypoint_exact_sinkhorn_elements: int
    canonical_fragment_keys: Tuple[str, str]
    candidate_rows: Tuple[Tuple[str, str, int, int], ...]
    bucket: Tuple[int, int, int]


def _limit_bucket(value: int, edges: Sequence[int]) -> int:
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def _candidate_cost(length_a: int, length_b: int) -> Tuple[int, int, int]:
    return (
        (length_a + length_b) ** 2,
        length_a * length_b,
        (length_a + 1) * (length_b + 1),
    )


def _record_geometry_payload(
    record: TrainingPairRecord,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    geometry_config: GeometryBatchConfig,
) -> Dict[str, Any]:
    # Intentionally excludes label, direction, label origin, hard-negative
    # score, provenance, row/member IDs, and pair ID.
    return {
        "split": record.split,
        "geometry_cache_key": geometry_cache_key(
            record, geometry_config, mask_a=mask_a, mask_b=mask_b
        ),
    }


def _batch_metrics(
    items: Sequence[_RecordComplexity], geometry_config: GeometryBatchConfig
) -> Dict[str, int]:
    candidate_count = sum(item.candidate_count for item in items)
    max_a = max((item.max_sequence_a for item in items), default=0)
    max_b = max((item.max_sequence_b for item in items), default=0)
    patch_height, patch_width = geometry_config.geometry.output_size
    local_elements = (
        candidate_count
        * len(CHANNEL_ORDER)
        * patch_height
        * patch_width
        * (max_a + max_b)
    )
    attention = candidate_count * (max_a + max_b) ** 2
    affinity = candidate_count * max_a * max_b
    sinkhorn = candidate_count * (max_a + 1) * (max_b + 1)
    return {
        "record_count": len(items),
        "candidate_count": candidate_count,
        "padded_sequence_a": max_a,
        "padded_sequence_b": max_b,
        "local_tensor_elements": local_elements,
        "attention_score_elements_per_head": attention,
        "affinity_elements": affinity,
        "sinkhorn_elements": sinkhorn,
        "exact_sequence_tokens_a": sum(item.sequence_tokens_a for item in items),
        "exact_sequence_tokens_b": sum(item.sequence_tokens_b for item in items),
        "exact_attention_elements": sum(
            item.exact_attention_elements for item in items
        ),
        "exact_affinity_elements": sum(item.exact_affinity_elements for item in items),
        "exact_sinkhorn_elements": sum(item.exact_sinkhorn_elements for item in items),
    }


def _keypoint_batch_metrics(
    items: Sequence[_RecordComplexity], geometry_config: GeometryBatchConfig
) -> Dict[str, int]:
    candidate_count = sum(item.keypoint_candidate_count for item in items)
    max_a = max((item.keypoint_max_sequence_a for item in items), default=0)
    max_b = max((item.keypoint_max_sequence_b for item in items), default=0)
    patch_height, patch_width = geometry_config.geometry.output_size
    return {
        "record_count": len(items),
        "candidate_count": candidate_count,
        "padded_sequence_a": max_a,
        "padded_sequence_b": max_b,
        "local_tensor_elements": (
            candidate_count
            * len(CHANNEL_ORDER)
            * patch_height
            * patch_width
            * (max_a + max_b)
        ),
        "attention_score_elements_per_head": (candidate_count * (max_a + max_b) ** 2),
        "affinity_elements": candidate_count * max_a * max_b,
        "sinkhorn_elements": candidate_count * (max_a + 1) * (max_b + 1),
        "exact_sequence_tokens_a": sum(
            item.keypoint_sequence_tokens_a for item in items
        ),
        "exact_sequence_tokens_b": sum(
            item.keypoint_sequence_tokens_b for item in items
        ),
        "exact_attention_elements": sum(
            item.keypoint_exact_attention_elements for item in items
        ),
        "exact_affinity_elements": sum(
            item.keypoint_exact_affinity_elements for item in items
        ),
        "exact_sinkhorn_elements": sum(
            item.keypoint_exact_sinkhorn_elements for item in items
        ),
    }


def _metrics_fit(
    metrics: Mapping[str, int],
    geometry_config: GeometryBatchConfig,
    plan_config: LocalQ1BatchPlanConfig,
) -> bool:
    return (
        metrics["record_count"]
        <= min(geometry_config.max_batch_size, plan_config.packing_max_records)
        and metrics["candidate_count"] <= geometry_config.max_candidates_per_batch
        and metrics["local_tensor_elements"]
        <= geometry_config.max_local_tensor_elements
        and metrics["attention_score_elements_per_head"]
        <= geometry_config.max_attention_score_elements_per_batch
        and metrics["affinity_elements"]
        <= geometry_config.max_affinity_elements_per_batch
        and metrics["sinkhorn_elements"]
        <= geometry_config.max_sinkhorn_elements_per_batch
    )


def _batch_fits(
    items: Sequence[_RecordComplexity],
    geometry_config: GeometryBatchConfig,
    plan_config: LocalQ1BatchPlanConfig,
) -> bool:
    return _metrics_fit(
        _batch_metrics(items, geometry_config), geometry_config, plan_config
    ) and _metrics_fit(
        _keypoint_batch_metrics(items, geometry_config),
        geometry_config,
        plan_config,
    )


def _validate_external_locks(
    opened: OpenedLocalQ1Cache,
    locks: LocalQ1ExternalLocks,
    *,
    require_batch_plan: bool,
) -> None:
    if not isinstance(opened, OpenedLocalQ1Cache):
        raise TypeError("opened must be OpenedLocalQ1Cache")
    if not isinstance(locks, LocalQ1ExternalLocks):
        raise TypeError("locks must be LocalQ1ExternalLocks")
    cache = opened.cache
    cache_trust = cache.frozen_receipt_trust
    if not cache.read_only or cache_trust is None:
        raise LocalQ1ProviderError(
            "LOCAL-Q1 provider requires an externally bound read-only cache"
        )
    population = opened.population
    replay = opened.inventory_replay
    build = opened.build_receipt
    if (
        population.freeze_file_sha256 != locks.freeze_file_sha256
        or population.freeze_content_sha256 != locks.freeze_content_sha256
        or population.precache_authority.run_plan_file_sha256
        != locks.run_plan_file_sha256
        or population.precache_authority.run_plan_content_sha256
        != locks.run_plan_content_sha256
        or population.precache_authority.source_bundle_manifest_sha256
        != locks.source_bundle_manifest_sha256
        or cache_trust.expected_file_sha256 != locks.cache_receipt_file_sha256
        or cache_trust.expected_content_sha256 != locks.cache_receipt_content_sha256
        or replay.semantic_commitment_sha256 != locks.inventory_semantic_sha256
        or _content_sha256(build, "LOCAL-Q1 cache build")
        != locks.build_receipt_content_sha256
        or _sha256(_canonical_json(build)) != locks.build_receipt_file_sha256
    ):
        raise LocalQ1ProviderError("LOCAL-Q1 external lock mismatch")
    if (
        build.get("precache_authority", {}).get("run_plan_file_sha256")
        != locks.run_plan_file_sha256
        or build.get("precache_authority", {}).get("run_plan_content_sha256")
        != locks.run_plan_content_sha256
        or build.get("precache_authority", {}).get("source_bundle_manifest_sha256")
        != locks.source_bundle_manifest_sha256
        or build.get("cache", {}).get("receipt_file_sha256")
        != locks.cache_receipt_file_sha256
        or build.get("cache", {}).get("receipt_content_sha256")
        != locks.cache_receipt_content_sha256
        or build.get("inventory", {}).get("receipt_file_sha256")
        != locks.inventory_receipt_file_sha256
        or build.get("inventory", {}).get("receipt_content_sha256")
        != locks.inventory_receipt_content_sha256
        or build.get("inventory", {}).get("semantic_commitment_sha256")
        != locks.inventory_semantic_sha256
    ):
        raise LocalQ1ProviderError("LOCAL-Q1 build cross-lock mismatch")
    if require_batch_plan and locks.batch_plan_file_sha256 is None:
        raise LocalQ1ProviderError("provider requires external batch-plan locks")


def _build_record_complexities(
    records: Sequence[TrainingPairRecord],
    *,
    split: str,
    loader: Callable[[MaskMemberRef], np.ndarray],
    cache: GeometryArtifactCache,
    geometry_config: GeometryBatchConfig,
    inventory_config: LocalCacheInventoryConfig,
    mask_memo: Dict[Tuple[str, ...], np.ndarray],
    fragment_memo: Dict[str, FragmentGeometryCacheLookup],
    counters: Dict[str, int],
) -> Tuple[_RecordComplexity, ...]:
    output = []

    def fragment(reference: MaskMemberRef) -> FragmentGeometryCacheLookup:
        reference_key = _reference_runtime_key(reference)
        mask = mask_memo.get(reference_key)
        if mask is None:
            mask = _validated_mask(
                loader, reference, geometry_config.max_input_pixels_per_mask
            )
            mask_memo[reference_key] = mask
            counters["mask_decode_count"] += 1
        identity = fragment_cache_identity(
            mask, reference.threshold_rule, geometry_config.geometry
        )
        existing = fragment_memo.get(identity.key)
        if existing is not None:
            return existing
        lookup = load_or_build_fragment_geometry(
            mask, reference.threshold_rule, geometry_config.geometry, cache
        )
        if not lookup.cache_hit:
            raise LocalQ1ProviderError("frozen cache replay produced a cache miss")
        fragment_memo[lookup.identity.key] = lookup
        counters["cache_read_count"] += 1
        return lookup

    for original_index, record in enumerate(records):
        if not isinstance(record, TrainingPairRecord) or record.split != split:
            raise LocalQ1ProviderError("plan population contains the wrong split")
        first = fragment(record.fragment_a)
        second = fragment(record.fragment_b)
        qualification = qualify_local_q1_pair_geometry(
            first.result,
            second.result,
            pair_id=record.pair_id,
            geometry_config=geometry_config,
            require_candidates=inventory_config.require_all_pairs_ok,
        )
        if not qualification.eligible:
            raise LocalQ1ProviderError(
                "LOCAL-Q1 pair cannot fit shared {} qualification: {}".format(
                    qualification.failure_stage,
                    qualification.failure_reason,
                )
            )
        result = combine_fragment_results(
            first.result,
            second.result,
            direction_b_wrt_a=None,
            config=geometry_config.geometry,
        )
        if result.status is not GeometryStatus.OK:
            raise LocalQ1ProviderError(
                "LOCAL-Q1 pair geometry failed during frozen planning"
            )
        if inventory_config.require_all_pairs_ok and not result.candidates:
            raise LocalQ1ProviderError("qualified pair has no candidates")
        emitted = tuple(group.direction for group in result.direction_groups)
        if len(set(emitted)) != len(emitted) or any(
            direction not in DEFAULT_DIRECTION_ORDER for direction in emitted
        ):
            raise LocalQ1ProviderError("geometry emitted invalid direction groups")
        if len(emitted) != len(DEFAULT_DIRECTION_ORDER):
            raise LocalQ1ProviderError(
                "LOCAL-Q1 planning requires candidates in all four directions"
            )
        rows = tuple(
            (
                candidate.candidate_id,
                candidate.direction.value,
                candidate.sequence_a.length,
                candidate.sequence_b.length,
            )
            for candidate in result.candidates
        )
        candidate_payload = [
            {
                "candidate": candidate_id,
                "direction": direction,
                "sequence_a": length_a,
                "sequence_b": length_b,
            }
            for candidate_id, direction, length_a, length_b in rows
        ]
        mask_a = mask_memo[_reference_runtime_key(record.fragment_a)]
        mask_b = mask_memo[_reference_runtime_key(record.fragment_b)]
        geometry_payload = _record_geometry_payload(
            record, mask_a, mask_b, geometry_config
        )
        lengths_a = [row[2] for row in rows]
        lengths_b = [row[3] for row in rows]
        costs = [_candidate_cost(row[2], row[3]) for row in rows]
        selector = ContourKeypointConfig()
        scale_count = len(geometry_config.geometry.window_scale_fractions)
        keypoint_token_cap = selector.max_keypoints_per_side * scale_count
        # Static upper bound: one bounded sequence pair in each of the four
        # known-upright directions.  Planning therefore stays cache-only and
        # does not materialize keypoint tensors merely to pack records.
        keypoint_lengths = [
            (keypoint_token_cap, keypoint_token_cap)
            for _direction in DEFAULT_DIRECTION_ORDER
        ]
        keypoint_costs = [
            _candidate_cost(length_a, length_b)
            for length_a, length_b in keypoint_lengths
        ]
        item = _RecordComplexity(
            split=split,
            original_index=original_index,
            record_geometry_sha256=_sha256(_canonical_json(geometry_payload)),
            candidate_semantic_sha256=_sha256(
                _canonical_json(
                    {
                        "record_geometry": geometry_payload,
                        "ordered_candidates": candidate_payload,
                    }
                )
            ),
            candidate_count=len(rows),
            max_sequence_a=max(lengths_a, default=0),
            max_sequence_b=max(lengths_b, default=0),
            sequence_tokens_a=sum(lengths_a),
            sequence_tokens_b=sum(lengths_b),
            exact_attention_elements=sum(value[0] for value in costs),
            exact_affinity_elements=sum(value[1] for value in costs),
            exact_sinkhorn_elements=sum(value[2] for value in costs),
            keypoint_candidate_count=len(keypoint_lengths),
            keypoint_max_sequence_a=max(
                (value[0] for value in keypoint_lengths), default=0
            ),
            keypoint_max_sequence_b=max(
                (value[1] for value in keypoint_lengths), default=0
            ),
            keypoint_sequence_tokens_a=sum(value[0] for value in keypoint_lengths),
            keypoint_sequence_tokens_b=sum(value[1] for value in keypoint_lengths),
            keypoint_exact_attention_elements=sum(value[0] for value in keypoint_costs),
            keypoint_exact_affinity_elements=sum(value[1] for value in keypoint_costs),
            keypoint_exact_sinkhorn_elements=sum(value[2] for value in keypoint_costs),
            canonical_fragment_keys=(first.identity.key, second.identity.key),
            candidate_rows=rows,
            bucket=(
                _limit_bucket(len(rows), inventory_config.candidate_count_bucket_edges),
                _limit_bucket(
                    max(lengths_a, default=0),
                    inventory_config.sequence_length_bucket_edges,
                ),
                _limit_bucket(
                    max(lengths_b, default=0),
                    inventory_config.sequence_length_bucket_edges,
                ),
            ),
        )
        one_record_metrics = _batch_metrics((item,), geometry_config)
        qualification_candidate_sha256 = _sha256(_canonical_json(candidate_payload))
        if (
            item.candidate_count != qualification.candidate_count
            or item.max_sequence_a != qualification.max_sequence_a
            or item.max_sequence_b != qualification.max_sequence_b
            or qualification_candidate_sha256 != qualification.candidate_semantic_sha256
            or one_record_metrics["local_tensor_elements"]
            != qualification.local_tensor_elements
            or one_record_metrics["attention_score_elements_per_head"]
            != qualification.attention_elements_per_head
            or one_record_metrics["affinity_elements"]
            != qualification.affinity_elements
            or one_record_metrics["sinkhorn_elements"]
            != qualification.sinkhorn_elements
        ):
            raise LocalQ1ProviderError(
                "shared qualification and provider complexity disagree"
            )
        output.append(item)
    return tuple(output)


def _pack_records(
    items: Sequence[_RecordComplexity],
    geometry_config: GeometryBatchConfig,
    plan_config: LocalQ1BatchPlanConfig,
) -> Tuple[Tuple[_RecordComplexity, ...], ...]:
    # Descending buckets/complexities reduce padding blow-up.  The final hash
    # and original ordinal are deterministic tie-breakers independent of
    # supervision.
    ordered = sorted(
        items,
        key=lambda item: (
            -item.bucket[1],
            -item.bucket[2],
            -item.bucket[0],
            -item.max_sequence_a,
            -item.max_sequence_b,
            -item.candidate_count,
            -item.keypoint_max_sequence_a,
            -item.keypoint_max_sequence_b,
            item.record_geometry_sha256,
            item.original_index,
        ),
    )
    bins = []  # type: ignore[var-annotated]
    for item in ordered:
        placed = False
        for batch in bins:
            if _batch_fits(tuple(batch) + (item,), geometry_config, plan_config):
                batch.append(item)
                placed = True
                break
        if not placed:
            if not _batch_fits((item,), geometry_config, plan_config):
                raise LocalQ1ProviderError(
                    "one record cannot fit the frozen packing guards"
                )
            bins.append([item])
    return tuple(tuple(batch) for batch in bins)


def _batch_candidate_digest(items: Sequence[_RecordComplexity]) -> str:
    return _sha256(
        _canonical_json(
            [
                {
                    "record_geometry_sha256": item.record_geometry_sha256,
                    "candidate_semantic_sha256": item.candidate_semantic_sha256,
                    "ordered_candidates": [list(row) for row in item.candidate_rows],
                }
                for item in items
            ]
        )
    )


def _batch_geometry_digest(items: Sequence[_RecordComplexity]) -> str:
    return _sha256(_canonical_json([item.record_geometry_sha256 for item in items]))


def _record_exact_cost(item: _RecordComplexity) -> Dict[str, int]:
    return {
        "sequence_tokens_a": item.sequence_tokens_a,
        "sequence_tokens_b": item.sequence_tokens_b,
        "attention_elements": item.exact_attention_elements,
        "affinity_elements": item.exact_affinity_elements,
        "sinkhorn_elements": item.exact_sinkhorn_elements,
    }


def _resource_estimate(
    metrics: Mapping[str, int],
    batch_size: int,
    geometry_config: GeometryBatchConfig,
    plan_config: LocalQ1BatchPlanConfig,
) -> Dict[str, Any]:
    coarse_elements = (
        batch_size
        * 2
        * geometry_config.coarse_output_size[0]
        * geometry_config.coarse_output_size[1]
    )
    tensor_bytes = (
        metrics["local_tensor_elements"] * 4
        + coarse_elements * 4
        + metrics["candidate_count"]
        * (metrics["padded_sequence_a"] + metrics["padded_sequence_b"])
        + metrics["candidate_count"] * 3 * 8
        + batch_size * 16
    )
    score_elements = (
        metrics["attention_score_elements_per_head"] * plan_config.attention_head_count
        + metrics["affinity_elements"]
        + metrics["sinkhorn_elements"]
    )
    score_bytes = score_elements * plan_config.score_element_bytes
    work_elements = metrics["local_tensor_elements"] + score_elements
    seconds = work_elements / float(plan_config.planning_elements_per_second)
    return {
        "tensor_bytes": tensor_bytes,
        "score_workspace_bytes": score_bytes,
        "conservative_peak_bytes": tensor_bytes + score_bytes,
        "work_elements": work_elements,
        "estimated_seconds_nominal": seconds,
        "estimated_seconds_upper": seconds * plan_config.time_safety_factor,
    }


def _serialize_phase_plan(
    phase: str,
    batches: Sequence[Sequence[_RecordComplexity]],
    records: Sequence[_RecordComplexity],
    geometry_config: GeometryBatchConfig,
    plan_config: LocalQ1BatchPlanConfig,
) -> Dict[str, Any]:
    candidate_cursor = 0
    sequence_a_cursor = 0
    sequence_b_cursor = 0
    serialized_batches = []
    for ordinal, items in enumerate(batches):
        metrics = _batch_metrics(items, geometry_config)
        candidate_stop = candidate_cursor + metrics["candidate_count"]
        sequence_a_stop = sequence_a_cursor + metrics["exact_sequence_tokens_a"]
        sequence_b_stop = sequence_b_cursor + metrics["exact_sequence_tokens_b"]
        serialized_batches.append(
            {
                "ordinal": ordinal,
                "record_ordinals": [item.original_index for item in items],
                "record_geometry_sha256": [
                    item.record_geometry_sha256 for item in items
                ],
                "record_candidate_sha256": [
                    item.candidate_semantic_sha256 for item in items
                ],
                "record_candidate_counts": [item.candidate_count for item in items],
                "record_candidate_sequence_lengths": [
                    [[row[2], row[3]] for row in item.candidate_rows] for item in items
                ],
                "record_max_sequence_a": [item.max_sequence_a for item in items],
                "record_max_sequence_b": [item.max_sequence_b for item in items],
                "record_buckets": [list(item.bucket) for item in items],
                "record_exact_costs": [_record_exact_cost(item) for item in items],
                "batch_geometry_sha256": _batch_geometry_digest(items),
                "candidate_semantic_sha256": _batch_candidate_digest(items),
                "candidate_slice": [candidate_cursor, candidate_stop],
                "sequence_a_token_slice": [sequence_a_cursor, sequence_a_stop],
                "sequence_b_token_slice": [sequence_b_cursor, sequence_b_stop],
                "complexity": metrics,
                "resource_estimate": _resource_estimate(
                    metrics, len(items), geometry_config, plan_config
                ),
            }
        )
        candidate_cursor = candidate_stop
        sequence_a_cursor = sequence_a_stop
        sequence_b_cursor = sequence_b_stop
    order = [
        record_hash
        for batch in serialized_batches
        for record_hash in batch["record_geometry_sha256"]
    ]
    return {
        "phase": phase,
        "record_count": len(records),
        "batch_count": len(serialized_batches),
        "population_ordinal_set_sha256": _sha256(
            _canonical_json(sorted(item.original_index for item in records))
        ),
        "planned_record_order_sha256": _sha256(_canonical_json(order)),
        "population_geometry_set_sha256": _sha256(
            _canonical_json(sorted(item.record_geometry_sha256 for item in records))
        ),
        "candidate_slice": [0, candidate_cursor],
        "sequence_a_token_slice": [0, sequence_a_cursor],
        "sequence_b_token_slice": [0, sequence_b_cursor],
        "batches": serialized_batches,
    }


@dataclass(frozen=True)
class FrozenLocalQ1BatchPlan:
    """Validated path-free plan receipt plus its canonical commitments."""

    receipt: Mapping[str, Any]
    content_sha256: str
    canonical_file_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.content_sha256, "plan content sha256")
        _require_sha256(self.canonical_file_sha256, "plan file sha256")
        portable = _portable(dict(self.receipt), "batch_plan")
        if portable.get("schema_version") != LOCAL_Q1_BATCH_PLAN_SCHEMA_VERSION:
            raise LocalQ1ProviderError("batch-plan schema changed")
        if portable.get("status") != (
            "frozen_four_phase_zero_truncation_read_only_cache_only"
        ):
            raise LocalQ1ProviderError("batch plan is not frozen and qualified")
        if _content_sha256(portable, "batch plan") != self.content_sha256:
            raise LocalQ1ProviderError("batch-plan content lock mismatch")
        if _sha256(_canonical_json(portable)) != self.canonical_file_sha256:
            raise LocalQ1ProviderError("batch-plan file lock mismatch")
        _validate_plan_structure(portable)
        object.__setattr__(self, "receipt", MappingProxyType(dict(portable)))

    def phase_batches(self, phase: str) -> Tuple[Mapping[str, Any], ...]:
        if phase not in _FORMAL_PHASES:
            raise ValueError("phase must be one of the four frozen formal phases")
        return tuple(self.receipt["phases"][phase]["batches"])

    @property
    def validation_assignment(self) -> Mapping[str, Any]:
        return self.receipt["validation_assignment"]


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


@dataclass(frozen=True)
class LocalQ1PlannedSafetyRecord:
    """Identity-minimized preflight metadata for one authoritative record."""

    population_ordinal: int
    expected_split: str
    component_token_sha256: str
    sealed_real_test_marker_explicit_false: bool
    sealed_scope_marker: bool
    historical_test_marker: bool

    def __post_init__(self) -> None:
        if type(self.population_ordinal) is not int or self.population_ordinal < 0:
            raise LocalQ1ProviderError("safety metadata population ordinal is invalid")
        if self.expected_split not in {"train", "val"}:
            raise LocalQ1ProviderError("safety metadata expected split is invalid")
        _require_sha256(
            self.component_token_sha256,
            "safety metadata component token",
        )
        if type(self.sealed_real_test_marker_explicit_false) is not bool:
            raise LocalQ1ProviderError("safety metadata sealed marker is not bool")
        if type(self.sealed_scope_marker) is not bool:
            raise LocalQ1ProviderError(
                "safety metadata sealed scope marker is not bool"
            )
        if type(self.historical_test_marker) is not bool:
            raise LocalQ1ProviderError("safety metadata historical marker is not bool")

    def portable_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SAFETY_METADATA_PRIVACY = {
    "fields_read": [
        "split",
        "component_id",
        "provenance.recursive_container_shape_keys_and_marker_relevant_values",
    ],
    "supervision_fields_read": [],
    "raw_pair_fragment_member_mask_or_path_fields_read": [],
    "raw_identifiers_or_paths_present": False,
    "raw_component_ids_present": False,
    "raw_provenance_keys_or_values_present": False,
    "provenance_shape_or_scan_counts_present": False,
    "component_tokens_are_opaque_sha256": True,
    "component_tokens_may_form_a_fingerprint": True,
    "cryptographic_privacy_claimed": False,
    "provenance_safety_scan": _portable(
        dict(local_q1_provenance_safety_policy()),
        "provenance_safety_policy",
    ),
}


@dataclass(frozen=True)
class FrozenLocalQ1PlannedSafetyMetadata:
    """Validated immutable pre-session view that never materializes a batch."""

    receipt: Mapping[str, Any]
    content_sha256: str
    records: Tuple[LocalQ1PlannedSafetyRecord, ...] = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.content_sha256, "planned safety metadata content sha256")
        portable = _portable(dict(self.receipt), "planned_safety_metadata")
        expected_root = {
            "schema_version",
            "status",
            "provider_version",
            "batch_plan_file_sha256",
            "batch_plan_content_sha256",
            "validation_assignment_sha256",
            "phase",
            "batch_ordinal",
            "expected_split",
            "plan_batch_geometry_sha256",
            "plan_phase_population_ordinal_set_sha256",
            "component_commitment_kind",
            "phase_component_set_sha256",
            "assignment_phase_component_set_sha256",
            "record_count",
            "population_ordinal_sequence_sha256",
            "batch_component_set_sha256",
            "records",
            "privacy",
            "content_sha256",
        }
        if not isinstance(portable, Mapping) or set(portable) != expected_root:
            raise LocalQ1ProviderError(
                "planned safety metadata has missing or extra fields"
            )
        if (
            portable["schema_version"]
            != LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION
            or portable["status"]
            != "frozen_authoritative_population_safety_metadata_no_materialization"
            or portable["provider_version"] != LOCAL_Q1_BATCH_PROVIDER_VERSION
        ):
            raise LocalQ1ProviderError("planned safety metadata contract changed")
        for name in (
            "batch_plan_file_sha256",
            "batch_plan_content_sha256",
            "validation_assignment_sha256",
            "plan_batch_geometry_sha256",
            "plan_phase_population_ordinal_set_sha256",
            "phase_component_set_sha256",
            "population_ordinal_sequence_sha256",
            "batch_component_set_sha256",
        ):
            _require_sha256(portable[name], "planned safety metadata " + name)
        phase = portable["phase"]
        if phase not in _PHASE_TO_SPLIT:
            raise LocalQ1ProviderError("planned safety metadata phase is invalid")
        expected_split = _PHASE_TO_SPLIT[phase]
        expected_kind = (
            "authoritative_train_population_component_tokens"
            if phase == "train"
            else "frozen_validation_assignment_phase_component_tokens"
        )
        if (
            portable["expected_split"] != expected_split
            or portable["component_commitment_kind"] != expected_kind
            or type(portable["batch_ordinal"]) is not int
            or portable["batch_ordinal"] < 0
            or type(portable["record_count"]) is not int
            or portable["record_count"] <= 0
        ):
            raise LocalQ1ProviderError(
                "planned safety metadata phase/batch/count is invalid"
            )
        assignment_component = portable["assignment_phase_component_set_sha256"]
        if phase == "train":
            if assignment_component is not None:
                raise LocalQ1ProviderError(
                    "training safety metadata cannot claim assignment components"
                )
        elif (
            not isinstance(assignment_component, str)
            or assignment_component != portable["phase_component_set_sha256"]
        ):
            raise LocalQ1ProviderError(
                "validation safety component commitment differs from assignment"
            )
        if portable["privacy"] != _SAFETY_METADATA_PRIVACY:
            raise LocalQ1ProviderError("planned safety metadata privacy changed")
        rows = portable["records"]
        expected_row_fields = {
            "population_ordinal",
            "expected_split",
            "component_token_sha256",
            "sealed_real_test_marker_explicit_false",
            "sealed_scope_marker",
            "historical_test_marker",
        }
        if not isinstance(rows, list) or not rows:
            raise LocalQ1ProviderError("planned safety metadata records are invalid")
        typed_rows = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != expected_row_fields:
                raise LocalQ1ProviderError(
                    "planned safety record has missing or extra fields"
                )
            typed = LocalQ1PlannedSafetyRecord(**dict(row))
            if typed.expected_split != expected_split:
                raise LocalQ1ProviderError(
                    "planned safety record split differs from phase"
                )
            typed_rows.append(typed)
        if (
            portable["record_count"] != len(typed_rows)
            or len({row.population_ordinal for row in typed_rows}) != len(typed_rows)
            or portable["population_ordinal_sequence_sha256"]
            != _sha256(_canonical_json([row.population_ordinal for row in typed_rows]))
            or portable["batch_component_set_sha256"]
            != _sha256(
                _canonical_json(
                    sorted({row.component_token_sha256 for row in typed_rows})
                )
            )
        ):
            raise LocalQ1ProviderError(
                "planned safety record order/component commitments changed"
            )
        if _content_sha256(portable, "planned safety metadata") != self.content_sha256:
            raise LocalQ1ProviderError("planned safety metadata content lock mismatch")
        object.__setattr__(self, "records", tuple(typed_rows))
        object.__setattr__(self, "receipt", _deep_freeze(portable))


def _validate_slice(value: Any, start: int, name: str) -> int:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] != start
        or value[1] < value[0]
    ):
        raise LocalQ1ProviderError("{} is not a contiguous ordinal slice".format(name))
    return value[1]


def _exact_dataclass_mapping(value: Any, cls: Any, name: str) -> Dict[str, Any]:
    expected = {field.name for field in fields(cls)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LocalQ1ProviderError(
            "{} must contain every {} field exactly".format(name, cls.__name__)
        )
    return dict(value)


def _candidate_config_from_receipt(value: Any) -> CandidateBuilderConfig:
    values = _exact_dataclass_mapping(value, CandidateBuilderConfig, "candidate config")
    corrosion = _exact_dataclass_mapping(
        values["corrosion"], CorrosionConfig, "corrosion config"
    )
    values["corrosion"] = CorrosionConfig(**corrosion)
    values["window_scale_fractions"] = tuple(values["window_scale_fractions"])
    values["output_size"] = tuple(values["output_size"])
    return CandidateBuilderConfig(**values)


def _trusted_plan_configs(
    config: Any,
) -> Tuple[LocalQ1BatchPlanConfig, GeometryBatchConfig, LocalCacheInventoryConfig]:
    if not isinstance(config, Mapping) or set(config) != {
        "plan",
        "geometry",
        "inventory",
        "validation_assignment_policy",
    }:
        raise LocalQ1ProviderError("batch-plan config fields changed")
    if config["validation_assignment_policy"] != {
        "schema_version": LOCAL_Q1_VALIDATION_ASSIGNMENT_SCHEMA_VERSION,
        "namespace": LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE,
        "algorithm": ("sha256_namespace_component_rank_round_robin_three_phase_v1"),
        "phase_order": list(_VALIDATION_PHASES),
    }:
        raise LocalQ1ProviderError("validation assignment policy changed")
    try:
        plan_values = _exact_dataclass_mapping(
            config["plan"], LocalQ1BatchPlanConfig, "batch plan config"
        )
        plan_config = LocalQ1BatchPlanConfig(**plan_values)
        inventory_values = _exact_dataclass_mapping(
            config["inventory"], LocalCacheInventoryConfig, "inventory config"
        )
        inventory_values["geometry"] = _candidate_config_from_receipt(
            inventory_values["geometry"]
        )
        inventory_values["sequence_length_bucket_edges"] = tuple(
            inventory_values["sequence_length_bucket_edges"]
        )
        inventory_values["candidate_count_bucket_edges"] = tuple(
            inventory_values["candidate_count_bucket_edges"]
        )
        inventory_config = LocalCacheInventoryConfig(**inventory_values)
        geometry_receipt = config["geometry"]
        if not isinstance(geometry_receipt, Mapping):
            raise LocalQ1ProviderError("geometry config receipt must be an object")
        bounds = geometry_receipt.get("bounds")
        expected_bounds = {
            "max_batch_size",
            "max_input_pixels_per_mask",
            "max_candidates_per_sample",
            "max_candidates_per_batch",
            "max_sequence_length",
            "max_local_tensor_elements",
            "max_attention_score_elements_per_candidate",
            "max_attention_score_elements_per_batch",
            "max_affinity_elements_per_candidate",
            "max_affinity_elements_per_batch",
            "max_sinkhorn_elements_per_candidate",
            "max_sinkhorn_elements_per_batch",
            "max_candidate_id_bytes",
        }
        if not isinstance(bounds, Mapping) or set(bounds) != expected_bounds:
            raise LocalQ1ProviderError("geometry bounds fields changed")
        geometry_config = GeometryBatchConfig(
            geometry=_candidate_config_from_receipt(geometry_receipt["geometry"]),
            coarse_output_size=tuple(geometry_receipt["coarse_output_size"]),
            coarse_resize_mode=geometry_receipt["coarse_resize_mode"],
            coarse_preprocess_mode=geometry_receipt["coarse_preprocess_mode"],
            coarse_content_fraction=geometry_receipt["coarse_content_fraction"],
            coarse_component_connectivity=geometry_receipt[
                "coarse_component_connectivity"
            ],
            **dict(bounds),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalQ1ProviderError("batch-plan typed config is invalid") from exc
    if (
        _canonical_json(plan_config.portable_dict()) != _canonical_json(config["plan"])
        or _canonical_json(inventory_config.portable_dict())
        != _canonical_json(config["inventory"])
        or _canonical_json(geometry_config.provenance_dict())
        != _canonical_json(config["geometry"])
        or geometry_config.geometry != inventory_config.geometry
    ):
        raise LocalQ1ProviderError("batch-plan typed config reconstruction changed")
    if plan_config.packing_max_records > geometry_config.max_batch_size:
        raise LocalQ1ProviderError("packing_max_records exceeds max_batch_size")
    return plan_config, geometry_config, inventory_config


_COMPLEXITY_KEYS = frozenset(
    {
        "record_count",
        "candidate_count",
        "padded_sequence_a",
        "padded_sequence_b",
        "local_tensor_elements",
        "attention_score_elements_per_head",
        "affinity_elements",
        "sinkhorn_elements",
        "exact_sequence_tokens_a",
        "exact_sequence_tokens_b",
        "exact_attention_elements",
        "exact_affinity_elements",
        "exact_sinkhorn_elements",
    }
)
_RECORD_EXACT_COST_KEYS = frozenset(
    {
        "sequence_tokens_a",
        "sequence_tokens_b",
        "attention_elements",
        "affinity_elements",
        "sinkhorn_elements",
    }
)
_BATCH_RESOURCE_KEYS = frozenset(
    {
        "tensor_bytes",
        "score_workspace_bytes",
        "conservative_peak_bytes",
        "work_elements",
        "estimated_seconds_nominal",
        "estimated_seconds_upper",
    }
)
_ROOT_RESOURCE_KEYS = frozenset(
    {
        "kind",
        "attention_head_count",
        "score_element_bytes",
        "planning_elements_per_second",
        "time_safety_factor",
        "estimated_peak_bytes",
        "total_work_elements",
        "estimated_seconds_nominal",
        "estimated_seconds_upper",
    }
)


def _validate_exact_int_mapping(
    value: Any, expected_keys: frozenset, name: str
) -> Mapping[str, int]:
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or any(type(item) is not int or item < 0 for item in value.values())
    ):
        raise LocalQ1ProviderError(name + " fields/types/ranges are invalid")
    return value


def _validate_batch_resource(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BATCH_RESOURCE_KEYS:
        raise LocalQ1ProviderError(
            "batch resource estimate has missing or extra fields"
        )
    for name in (
        "tensor_bytes",
        "score_workspace_bytes",
        "conservative_peak_bytes",
        "work_elements",
    ):
        if type(value[name]) is not int or value[name] < 0:
            raise LocalQ1ProviderError("batch resource integer is invalid")
    for name in ("estimated_seconds_nominal", "estimated_seconds_upper"):
        if (
            type(value[name]) is not float
            or not math.isfinite(value[name])
            or value[name] < 0
        ):
            raise LocalQ1ProviderError("batch resource time is invalid")
    return value


def _validate_root_resource(
    value: Any,
    *,
    expected: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping) or set(value) != _ROOT_RESOURCE_KEYS:
        raise LocalQ1ProviderError("root resource estimate has missing or extra fields")
    if value["kind"] != "planning_estimate_not_measured_runtime":
        raise LocalQ1ProviderError("root resource estimate kind changed")
    for name in (
        "attention_head_count",
        "score_element_bytes",
        "planning_elements_per_second",
    ):
        if type(value[name]) is not int or value[name] <= 0:
            raise LocalQ1ProviderError("root resource config integer is invalid")
    for name in ("estimated_peak_bytes", "total_work_elements"):
        if type(value[name]) is not int or value[name] < 0:
            raise LocalQ1ProviderError("root resource total integer is invalid")
    if (
        type(value["time_safety_factor"]) is not float
        or not math.isfinite(value["time_safety_factor"])
        or value["time_safety_factor"] < 1.0
    ):
        raise LocalQ1ProviderError("root resource safety factor is invalid")
    for name in ("estimated_seconds_nominal", "estimated_seconds_upper"):
        if (
            type(value[name]) is not float
            or not math.isfinite(value[name])
            or value[name] < 0
        ):
            raise LocalQ1ProviderError("root resource time is invalid")
    if value != expected:
        raise LocalQ1ProviderError(
            "root resource estimate differs from exact recomputation"
        )


def _recompute_batch_entry(
    batch: Any,
    *,
    ordinal: int,
    geometry_config: GeometryBatchConfig,
    inventory_config: LocalCacheInventoryConfig,
    plan_config: LocalQ1BatchPlanConfig,
) -> Mapping[str, Any]:
    expected_fields = {
        "ordinal",
        "record_ordinals",
        "record_geometry_sha256",
        "record_candidate_sha256",
        "record_candidate_counts",
        "record_candidate_sequence_lengths",
        "record_max_sequence_a",
        "record_max_sequence_b",
        "record_buckets",
        "record_exact_costs",
        "batch_geometry_sha256",
        "candidate_semantic_sha256",
        "candidate_slice",
        "sequence_a_token_slice",
        "sequence_b_token_slice",
        "complexity",
        "resource_estimate",
    }
    if not isinstance(batch, Mapping) or set(batch) != expected_fields:
        raise LocalQ1ProviderError("batch has missing or extra fields")
    if type(batch["ordinal"]) is not int or batch["ordinal"] != ordinal:
        raise LocalQ1ProviderError("batch ordinal/order changed")
    vectors = (
        batch["record_ordinals"],
        batch["record_geometry_sha256"],
        batch["record_candidate_sha256"],
        batch["record_candidate_counts"],
        batch["record_candidate_sequence_lengths"],
        batch["record_max_sequence_a"],
        batch["record_max_sequence_b"],
        batch["record_buckets"],
        batch["record_exact_costs"],
    )
    if any(not isinstance(value, list) for value in vectors):
        raise LocalQ1ProviderError("batch record vectors must be lists")
    record_count = len(batch["record_ordinals"])
    if record_count <= 0 or any(len(value) != record_count for value in vectors):
        raise LocalQ1ProviderError("batch record vectors have inconsistent lengths")
    if record_count > min(
        geometry_config.max_batch_size, plan_config.packing_max_records
    ):
        raise LocalQ1ProviderError("batch record count exceeds trusted config")
    if any(type(value) is not int or value < 0 for value in batch["record_ordinals"]):
        raise LocalQ1ProviderError("batch record ordinals are invalid")
    for digest in batch["record_geometry_sha256"] + batch["record_candidate_sha256"]:
        _require_sha256(digest, "batch record digest")
    _require_sha256(
        batch["candidate_semantic_sha256"], "batch candidate semantic digest"
    )
    if batch["batch_geometry_sha256"] != _sha256(
        _canonical_json(batch["record_geometry_sha256"])
    ):
        raise LocalQ1ProviderError("batch geometry digest mismatch")

    candidate_counts = []
    maxima_a = []
    maxima_b = []
    buckets = []
    exact_costs = []
    for shapes in batch["record_candidate_sequence_lengths"]:
        if (
            not isinstance(shapes, list)
            or not shapes
            or len(shapes) > geometry_config.max_candidates_per_sample
        ):
            raise LocalQ1ProviderError("record candidate shape list is invalid")
        lengths_a = []
        lengths_b = []
        attention = 0
        affinity = 0
        sinkhorn = 0
        for shape in shapes:
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(length) is not int or length <= 0 for length in shape)
            ):
                raise LocalQ1ProviderError("candidate sequence shape is invalid")
            length_a, length_b = shape
            if (
                length_a > geometry_config.max_sequence_length
                or length_b > geometry_config.max_sequence_length
            ):
                raise LocalQ1ProviderError("candidate sequence exceeds trusted bound")
            candidate_attention, candidate_affinity, candidate_sinkhorn = (
                _candidate_cost(length_a, length_b)
            )
            if (
                candidate_attention
                > geometry_config.max_attention_score_elements_per_candidate
                or candidate_affinity
                > geometry_config.max_affinity_elements_per_candidate
                or candidate_sinkhorn
                > geometry_config.max_sinkhorn_elements_per_candidate
            ):
                raise LocalQ1ProviderError("candidate exact cost exceeds trusted bound")
            lengths_a.append(length_a)
            lengths_b.append(length_b)
            attention += candidate_attention
            affinity += candidate_affinity
            sinkhorn += candidate_sinkhorn
        count = len(shapes)
        max_a = max(lengths_a)
        max_b = max(lengths_b)
        candidate_counts.append(count)
        maxima_a.append(max_a)
        maxima_b.append(max_b)
        buckets.append(
            [
                _limit_bucket(count, inventory_config.candidate_count_bucket_edges),
                _limit_bucket(max_a, inventory_config.sequence_length_bucket_edges),
                _limit_bucket(max_b, inventory_config.sequence_length_bucket_edges),
            ]
        )
        exact_costs.append(
            {
                "sequence_tokens_a": sum(lengths_a),
                "sequence_tokens_b": sum(lengths_b),
                "attention_elements": attention,
                "affinity_elements": affinity,
                "sinkhorn_elements": sinkhorn,
            }
        )
    for cost in batch["record_exact_costs"]:
        _validate_exact_int_mapping(cost, _RECORD_EXACT_COST_KEYS, "record exact cost")
    if any(
        type(value) is not int or value <= 0
        for vector in (
            batch["record_candidate_counts"],
            batch["record_max_sequence_a"],
            batch["record_max_sequence_b"],
        )
        for value in vector
    ) or any(
        not isinstance(bucket, list)
        or len(bucket) != 3
        or any(type(value) is not int or value < 0 for value in bucket)
        for bucket in batch["record_buckets"]
    ):
        raise LocalQ1ProviderError("anonymous record complexity types are invalid")
    if (
        batch["record_candidate_counts"] != candidate_counts
        or batch["record_max_sequence_a"] != maxima_a
        or batch["record_max_sequence_b"] != maxima_b
        or batch["record_buckets"] != buckets
        or batch["record_exact_costs"] != exact_costs
    ):
        raise LocalQ1ProviderError("anonymous record complexity was not recomputed")
    candidate_count = sum(candidate_counts)
    padded_a = max(maxima_a)
    padded_b = max(maxima_b)
    patch_height, patch_width = geometry_config.geometry.output_size
    complexity = {
        "record_count": record_count,
        "candidate_count": candidate_count,
        "padded_sequence_a": padded_a,
        "padded_sequence_b": padded_b,
        "local_tensor_elements": (
            candidate_count
            * len(CHANNEL_ORDER)
            * patch_height
            * patch_width
            * (padded_a + padded_b)
        ),
        "attention_score_elements_per_head": (
            candidate_count * (padded_a + padded_b) ** 2
        ),
        "affinity_elements": candidate_count * padded_a * padded_b,
        "sinkhorn_elements": candidate_count * (padded_a + 1) * (padded_b + 1),
        "exact_sequence_tokens_a": sum(
            cost["sequence_tokens_a"] for cost in exact_costs
        ),
        "exact_sequence_tokens_b": sum(
            cost["sequence_tokens_b"] for cost in exact_costs
        ),
        "exact_attention_elements": sum(
            cost["attention_elements"] for cost in exact_costs
        ),
        "exact_affinity_elements": sum(
            cost["affinity_elements"] for cost in exact_costs
        ),
        "exact_sinkhorn_elements": sum(
            cost["sinkhorn_elements"] for cost in exact_costs
        ),
    }
    _validate_exact_int_mapping(batch["complexity"], _COMPLEXITY_KEYS, "complexity")
    if batch["complexity"] != complexity:
        raise LocalQ1ProviderError("batch complexity differs from exact recomputation")
    if (
        complexity["candidate_count"] > geometry_config.max_candidates_per_batch
        or complexity["local_tensor_elements"]
        > geometry_config.max_local_tensor_elements
        or complexity["attention_score_elements_per_head"]
        > geometry_config.max_attention_score_elements_per_batch
        or complexity["affinity_elements"]
        > geometry_config.max_affinity_elements_per_batch
        or complexity["sinkhorn_elements"]
        > geometry_config.max_sinkhorn_elements_per_batch
    ):
        raise LocalQ1ProviderError("batch exact cost exceeds trusted batch bound")
    resource = _resource_estimate(
        complexity, record_count, geometry_config, plan_config
    )
    _validate_batch_resource(batch["resource_estimate"])
    if batch["resource_estimate"] != resource:
        raise LocalQ1ProviderError("batch resource estimate was not recomputed")
    return {
        "ordinals": batch["record_ordinals"],
        "record_geometry_sha256": batch["record_geometry_sha256"],
        "complexity": complexity,
        "resource_estimate": resource,
    }


def _validate_plan_structure(receipt: Mapping[str, Any]) -> None:
    expected_root = {
        "schema_version",
        "status",
        "provider_version",
        "external_locks",
        "config",
        "config_sha256",
        "geometry_config_sha256",
        "inventory_config_sha256",
        "pair_qualification_contract",
        "authority_boundary",
        "validation_assignment",
        "ordering_contract",
        "phases",
        "totals",
        "resource_estimate",
        "operational_counts",
        "portable_privacy",
        "content_sha256",
    }
    if set(receipt) != expected_root:
        raise LocalQ1ProviderError("batch-plan root has missing or extra fields")
    if receipt.get("provider_version") != LOCAL_Q1_BATCH_PROVIDER_VERSION:
        raise LocalQ1ProviderError("batch-plan provider version changed")
    for name in (
        "config_sha256",
        "geometry_config_sha256",
        "inventory_config_sha256",
    ):
        _require_sha256(receipt.get(name), "batch-plan " + name)
    if not isinstance(receipt.get("config"), Mapping) or receipt[
        "config_sha256"
    ] != _sha256(_canonical_json(receipt["config"])):
        raise LocalQ1ProviderError("batch-plan config digest mismatch")
    config = receipt["config"]
    plan_config, geometry_config, inventory_config = _trusted_plan_configs(config)
    if receipt["geometry_config_sha256"] != _sha256(
        _canonical_json(config["geometry"])
    ) or receipt["inventory_config_sha256"] != _sha256(
        _canonical_json(config["inventory"])
    ):
        raise LocalQ1ProviderError("batch-plan config sub-digest mismatch")
    if receipt.get("pair_qualification_contract") != qualification_contract(
        geometry_config
    ):
        raise LocalQ1ProviderError(
            "batch-plan shared pair-qualification contract changed"
        )
    if receipt.get("authority_boundary") != {
        "planning_config_hash_arguments_source": (
            "self_provided_by_enclosing_cli_caller"
        ),
        "planning_config_independent_authority_verified_here": False,
        "enclosing_planning_authority_status": "pending",
        "result_bearing_authorized": False,
        "promotion_requirement": (
            "later_independent_run_authority_must_lock_batch_plan_file_and_"
            "content_sha256_before_any_result_bearing_execution"
        ),
    }:
        raise LocalQ1ProviderError("batch-plan authority boundary changed")
    locks = receipt.get("external_locks")
    if not isinstance(locks, Mapping) or set(locks) != {
        "run_plan_file_sha256",
        "run_plan_content_sha256",
        "source_bundle_manifest_sha256",
        "planning_config_receipt_file_sha256",
        "planning_config_receipt_content_sha256",
        "freeze_file_sha256",
        "freeze_content_sha256",
        "cache_receipt_file_sha256",
        "cache_receipt_content_sha256",
        "inventory_receipt_file_sha256",
        "inventory_receipt_content_sha256",
        "inventory_semantic_sha256",
        "build_receipt_file_sha256",
        "build_receipt_content_sha256",
    }:
        raise LocalQ1ProviderError("batch-plan external locks are incomplete")
    for name, digest in locks.items():
        _require_sha256(digest, "batch-plan external lock " + name)
    assignment = receipt.get("validation_assignment")
    _validate_validation_assignment_receipt(assignment)
    if receipt.get("ordering_contract") != {
        "supervision_fields_read": [],
        "validation_assignment_fields_read": ["split", "component_id"],
        "validation_assignment_unit": "component_id",
        "validation_phases_independently_packed": True,
        "candidate_generation_direction_argument": None,
        "candidate_generation_directions": [
            direction.value for direction in DEFAULT_DIRECTION_ORDER
        ],
        "bucketing_keys": ["candidate_count", "max_sequence_a", "max_sequence_b"],
        "packing": "deterministic_descending_bucket_first_fit_no_truncation",
        "packing_fit_representations": [
            MULTIRUN_REPRESENTATION,
            KEYPOINT_REPRESENTATION,
        ],
        "packing_fit_rule": "every_batch_must_fit_both_representations",
        "contour_keypoint_config": _portable(
            asdict(ContourKeypointConfig()), "batch_plan.keypoint_config"
        ),
        "cross_arm_shared": [
            "local_dual_softmax",
            "local_dustbin_sinkhorn",
            "keypoint_dual_softmax",
            "keypoint_dustbin_sinkhorn",
            "fused",
        ],
    }:
        raise LocalQ1ProviderError("batch-plan ordering contract changed")
    phases = receipt.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != set(_FORMAL_PHASES):
        raise LocalQ1ProviderError("batch plan must contain four formal phases")
    total_records = 0
    total_batches = 0
    total_candidates = 0
    total_sequence_a = 0
    total_sequence_b = 0
    validation_ordinals = []
    all_batch_resources = []
    for phase in _FORMAL_PHASES:
        value = phases[phase]
        if not isinstance(value, Mapping) or set(value) != {
            "phase",
            "record_count",
            "batch_count",
            "population_ordinal_set_sha256",
            "planned_record_order_sha256",
            "population_geometry_set_sha256",
            "candidate_slice",
            "sequence_a_token_slice",
            "sequence_b_token_slice",
            "batches",
        }:
            raise LocalQ1ProviderError("phase plan has missing or extra fields")
        if (
            value["phase"] != phase
            or type(value["record_count"]) is not int
            or value["record_count"] < 0
            or type(value["batch_count"]) is not int
            or value["batch_count"] < 0
            or not isinstance(value["batches"], list)
        ):
            raise LocalQ1ProviderError("phase plan identity/type is invalid")
        batches = value["batches"]
        if value["batch_count"] != len(batches):
            raise LocalQ1ProviderError("split batch count mismatch")
        seen = []
        candidate_cursor = 0
        sequence_a_cursor = 0
        sequence_b_cursor = 0
        order = []
        for ordinal, batch in enumerate(batches):
            recomputed = _recompute_batch_entry(
                batch,
                ordinal=ordinal,
                geometry_config=geometry_config,
                inventory_config=inventory_config,
                plan_config=plan_config,
            )
            ordinals = recomputed["ordinals"]
            hashes = recomputed["record_geometry_sha256"]
            complexity = recomputed["complexity"]
            seen.extend(ordinals)
            order.extend(hashes)
            expected_candidate_stop = candidate_cursor + complexity["candidate_count"]
            expected_sequence_a_stop = (
                sequence_a_cursor + complexity["exact_sequence_tokens_a"]
            )
            expected_sequence_b_stop = (
                sequence_b_cursor + complexity["exact_sequence_tokens_b"]
            )
            if (
                _validate_slice(
                    batch["candidate_slice"], candidate_cursor, "candidate slice"
                )
                != expected_candidate_stop
            ):
                raise LocalQ1ProviderError("candidate slice was not recomputed")
            if (
                _validate_slice(
                    batch["sequence_a_token_slice"],
                    sequence_a_cursor,
                    "sequence A slice",
                )
                != expected_sequence_a_stop
            ):
                raise LocalQ1ProviderError("sequence A slice was not recomputed")
            if (
                _validate_slice(
                    batch["sequence_b_token_slice"],
                    sequence_b_cursor,
                    "sequence B slice",
                )
                != expected_sequence_b_stop
            ):
                raise LocalQ1ProviderError("sequence B slice was not recomputed")
            candidate_cursor = expected_candidate_stop
            sequence_a_cursor = expected_sequence_a_stop
            sequence_b_cursor = expected_sequence_b_stop
            all_batch_resources.append(recomputed["resource_estimate"])
        if len(seen) != value["record_count"] or len(set(seen)) != len(seen):
            raise LocalQ1ProviderError("phase plan has duplicate record ordinals")
        if value["population_ordinal_set_sha256"] != _sha256(
            _canonical_json(sorted(seen))
        ):
            raise LocalQ1ProviderError("phase population ordinal digest mismatch")
        if phase == "train":
            if sorted(seen) != list(range(value["record_count"])):
                raise LocalQ1ProviderError(
                    "training phase has missing/extra record ordinals"
                )
        else:
            expected_phase = assignment["phases"][phase]
            if (
                value["record_count"] != expected_phase["record_count"]
                or value["population_ordinal_set_sha256"]
                != expected_phase["record_ordinal_set_sha256"]
            ):
                raise LocalQ1ProviderError(
                    "phase plan differs from validation assignment"
                )
            validation_ordinals.extend(seen)
        if value["planned_record_order_sha256"] != _sha256(_canonical_json(order)):
            raise LocalQ1ProviderError("planned record order digest mismatch")
        if value["population_geometry_set_sha256"] != _sha256(
            _canonical_json(sorted(order))
        ):
            raise LocalQ1ProviderError("population geometry set digest mismatch")
        if (
            _validate_slice(value["candidate_slice"], 0, "phase candidate slice")
            != candidate_cursor
        ):
            raise LocalQ1ProviderError("split candidate slice mismatch")
        if (
            _validate_slice(
                value["sequence_a_token_slice"], 0, "phase sequence A slice"
            )
            != sequence_a_cursor
        ):
            raise LocalQ1ProviderError("split sequence A slice mismatch")
        if (
            _validate_slice(
                value["sequence_b_token_slice"], 0, "phase sequence B slice"
            )
            != sequence_b_cursor
        ):
            raise LocalQ1ProviderError("split sequence B slice mismatch")
        total_records += value["record_count"]
        total_batches += value["batch_count"]
        total_candidates += candidate_cursor
        total_sequence_a += sequence_a_cursor
        total_sequence_b += sequence_b_cursor
    if sorted(validation_ordinals) != list(
        range(assignment["validation_record_count"])
    ):
        raise LocalQ1ProviderError(
            "validation phase record ordinals are not disjoint and exhaustive"
        )
    totals = receipt.get("totals")
    expected_totals = {
        "batch_count": total_batches,
        "record_count": total_records,
        "candidate_count": total_candidates,
        "sequence_a_token_count": total_sequence_a,
        "sequence_b_token_count": total_sequence_b,
    }
    if (
        not isinstance(totals, Mapping)
        or set(totals) != set(expected_totals)
        or any(type(value) is not int or value < 0 for value in totals.values())
        or totals != expected_totals
    ):
        raise LocalQ1ProviderError("batch-plan totals are inconsistent")
    expected_root_resource = {
        "kind": "planning_estimate_not_measured_runtime",
        "attention_head_count": plan_config.attention_head_count,
        "score_element_bytes": plan_config.score_element_bytes,
        "planning_elements_per_second": (plan_config.planning_elements_per_second),
        "time_safety_factor": plan_config.time_safety_factor,
        "estimated_peak_bytes": max(
            (resource["conservative_peak_bytes"] for resource in all_batch_resources),
            default=0,
        ),
        "total_work_elements": sum(
            resource["work_elements"] for resource in all_batch_resources
        ),
        "estimated_seconds_nominal": sum(
            resource["estimated_seconds_nominal"] for resource in all_batch_resources
        ),
        "estimated_seconds_upper": sum(
            resource["estimated_seconds_upper"] for resource in all_batch_resources
        ),
    }
    _validate_root_resource(
        receipt.get("resource_estimate"), expected=expected_root_resource
    )
    counts = receipt.get("operational_counts")
    if not isinstance(counts, Mapping) or set(counts) != {
        "planning_mask_decode_count",
        "planning_cache_read_count",
        "planning_cache_miss_count",
        "planning_cache_write_count",
        "planning_geometry_build_count",
    }:
        raise LocalQ1ProviderError("batch-plan operational counts changed")
    if (
        any(type(value) is not int or value < 0 for value in counts.values())
        or any(
            counts[name] <= 0
            for name in ("planning_mask_decode_count", "planning_cache_read_count")
        )
        or counts["planning_cache_miss_count"] != 0
        or counts["planning_cache_write_count"] != 0
        or counts["planning_geometry_build_count"] != 0
    ):
        raise LocalQ1ProviderError(
            "batch-plan must prove positive reads and zero miss/write/build"
        )
    if receipt.get("portable_privacy") != {
        "raw_pair_component_member_or_path_identifiers_present": False,
        "anonymous_per_record_fields_present": [
            "ordinal",
            "geometry_commitment",
            "candidate_commitment",
            "candidate_count",
            "candidate_sequence_lengths",
            "max_sequence_a",
            "max_sequence_b",
            "bucket",
            "exact_costs",
        ],
        "anonymous_complexity_may_form_a_fingerprint": True,
        "cryptographic_privacy_claimed": False,
    }:
        raise LocalQ1ProviderError("batch-plan portable privacy contract changed")


def build_local_q1_batch_plan(
    *,
    opened: OpenedLocalQ1Cache,
    loader_factory: Callable[[], Callable[[MaskMemberRef], np.ndarray]],
    geometry_config: GeometryBatchConfig,
    inventory_config: LocalCacheInventoryConfig,
    external_locks: LocalQ1ExternalLocks,
    plan_config: LocalQ1BatchPlanConfig,
) -> FrozenLocalQ1BatchPlan:
    """Inventory and independently pack train/select/calibration/report.

    Every candidate is retained.  A record that cannot fit by itself raises an
    exception; candidates are never capped, shortened, dropped, or moved across
    the frozen label-blind validation component partition.
    """

    _validate_external_locks(opened, external_locks, require_batch_plan=False)
    if external_locks.batch_plan_file_sha256 is not None:
        raise LocalQ1ProviderError("initial plan build must not trust itself")
    if not callable(loader_factory):
        raise TypeError("loader_factory must be callable")
    if not isinstance(geometry_config, GeometryBatchConfig):
        raise TypeError("geometry_config must be GeometryBatchConfig")
    if not isinstance(inventory_config, LocalCacheInventoryConfig):
        raise TypeError("inventory_config must be LocalCacheInventoryConfig")
    settings = plan_config
    if not isinstance(settings, LocalQ1BatchPlanConfig):
        raise TypeError("plan_config must be LocalQ1BatchPlanConfig")
    if geometry_config.geometry != inventory_config.geometry:
        raise LocalQ1ProviderError("geometry and inventory configs differ")
    if settings.packing_max_records > geometry_config.max_batch_size:
        raise LocalQ1ProviderError("packing_max_records exceeds max_batch_size")
    expected_inventory_config = opened.inventory_replay.receipt.get("config")
    if _canonical_json(expected_inventory_config) != _canonical_json(
        inventory_config.portable_dict()
    ):
        raise LocalQ1ProviderError("inventory config differs from frozen replay")
    validation_assignment = freeze_local_q1_validation_assignment(
        opened.population.validation_records
    )

    loader = loader_factory()
    if not callable(loader):
        raise TypeError("loader factory must return a callable")
    counters = {"mask_decode_count": 0, "cache_read_count": 0}
    mask_memo: Dict[Tuple[str, ...], np.ndarray] = {}
    fragment_memo: Dict[str, FragmentGeometryCacheLookup] = {}
    try:
        _prefetch_planning_masks(
            opened.population.records,
            loader=loader,
            max_pixels=geometry_config.max_input_pixels_per_mask,
            mask_memo=mask_memo,
            counters=counters,
        )
        train_records = _build_record_complexities(
            opened.population.training_records,
            split="train",
            loader=loader,
            cache=opened.cache,
            geometry_config=geometry_config,
            inventory_config=inventory_config,
            mask_memo=mask_memo,
            fragment_memo=fragment_memo,
            counters=counters,
        )
        val_records = _build_record_complexities(
            opened.population.validation_records,
            split="val",
            loader=loader,
            cache=opened.cache,
            geometry_config=geometry_config,
            inventory_config=inventory_config,
            mask_memo=mask_memo,
            fragment_memo=fragment_memo,
            counters=counters,
        )
    finally:
        _close_loader(loader)
    expected_fragments = opened.inventory_replay.receipt["operational_counts"][
        "unique_canonical_fragment_count"
    ]
    if counters["cache_read_count"] != expected_fragments:
        raise LocalQ1ProviderError(
            "planning did not read every frozen canonical fragment exactly once"
        )
    validation_complexities_by_phase = {
        phase: tuple(
            item
            for item in val_records
            if validation_assignment.phase_by_component[
                opened.population.validation_records[item.original_index].component_id
            ]
            == phase
        )
        for phase in _VALIDATION_PHASES
    }
    phase_complexities = {
        "train": train_records,
        **validation_complexities_by_phase,
    }
    phase_receipts = {}
    for phase in _FORMAL_PHASES:
        complexities = phase_complexities[phase]
        batches = _pack_records(complexities, geometry_config, settings)
        phase_receipts[phase] = _serialize_phase_plan(
            phase, batches, complexities, geometry_config, settings
        )
    all_batches = [
        batch for phase in _FORMAL_PHASES for batch in phase_receipts[phase]["batches"]
    ]
    peak = max(
        (
            batch["resource_estimate"]["conservative_peak_bytes"]
            for batch in all_batches
        ),
        default=0,
    )
    total_work = sum(
        batch["resource_estimate"]["work_elements"] for batch in all_batches
    )
    nominal_seconds = sum(
        batch["resource_estimate"]["estimated_seconds_nominal"] for batch in all_batches
    )
    upper_seconds = sum(
        batch["resource_estimate"]["estimated_seconds_upper"] for batch in all_batches
    )
    totals = {
        "batch_count": sum(value["batch_count"] for value in phase_receipts.values()),
        "record_count": sum(value["record_count"] for value in phase_receipts.values()),
        "candidate_count": sum(
            value["candidate_slice"][1] for value in phase_receipts.values()
        ),
        "sequence_a_token_count": sum(
            value["sequence_a_token_slice"][1] for value in phase_receipts.values()
        ),
        "sequence_b_token_count": sum(
            value["sequence_b_token_slice"][1] for value in phase_receipts.values()
        ),
    }
    config_payload = {
        "plan": settings.portable_dict(),
        "geometry": geometry_config.provenance_dict(),
        "inventory": inventory_config.portable_dict(),
        "validation_assignment_policy": {
            "schema_version": LOCAL_Q1_VALIDATION_ASSIGNMENT_SCHEMA_VERSION,
            "namespace": LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE,
            "algorithm": ("sha256_namespace_component_rank_round_robin_three_phase_v1"),
            "phase_order": list(_VALIDATION_PHASES),
        },
    }
    receipt = {
        "schema_version": LOCAL_Q1_BATCH_PLAN_SCHEMA_VERSION,
        "status": "frozen_four_phase_zero_truncation_read_only_cache_only",
        "provider_version": LOCAL_Q1_BATCH_PROVIDER_VERSION,
        "external_locks": external_locks.without_batch_plan(),
        "config": config_payload,
        "config_sha256": _sha256(_canonical_json(config_payload)),
        "geometry_config_sha256": geometry_config.fingerprint,
        "inventory_config_sha256": _sha256(
            _canonical_json(inventory_config.portable_dict())
        ),
        "pair_qualification_contract": qualification_contract(geometry_config),
        "authority_boundary": {
            "planning_config_hash_arguments_source": (
                "self_provided_by_enclosing_cli_caller"
            ),
            "planning_config_independent_authority_verified_here": False,
            "enclosing_planning_authority_status": "pending",
            "result_bearing_authorized": False,
            "promotion_requirement": (
                "later_independent_run_authority_must_lock_batch_plan_file_and_"
                "content_sha256_before_any_result_bearing_execution"
            ),
        },
        "validation_assignment": dict(validation_assignment.receipt),
        "ordering_contract": {
            "supervision_fields_read": [],
            "validation_assignment_fields_read": ["split", "component_id"],
            "validation_assignment_unit": "component_id",
            "validation_phases_independently_packed": True,
            "candidate_generation_direction_argument": None,
            "candidate_generation_directions": [
                direction.value for direction in DEFAULT_DIRECTION_ORDER
            ],
            "bucketing_keys": ["candidate_count", "max_sequence_a", "max_sequence_b"],
            "packing": "deterministic_descending_bucket_first_fit_no_truncation",
            "packing_fit_representations": [
                MULTIRUN_REPRESENTATION,
                KEYPOINT_REPRESENTATION,
            ],
            "packing_fit_rule": "every_batch_must_fit_both_representations",
            "contour_keypoint_config": _portable(
                asdict(ContourKeypointConfig()), "batch_plan.keypoint_config"
            ),
            "cross_arm_shared": [
                "local_dual_softmax",
                "local_dustbin_sinkhorn",
                "keypoint_dual_softmax",
                "keypoint_dustbin_sinkhorn",
                "fused",
            ],
        },
        "phases": phase_receipts,
        "totals": totals,
        "resource_estimate": {
            "kind": "planning_estimate_not_measured_runtime",
            "attention_head_count": settings.attention_head_count,
            "score_element_bytes": settings.score_element_bytes,
            "planning_elements_per_second": settings.planning_elements_per_second,
            "time_safety_factor": settings.time_safety_factor,
            "estimated_peak_bytes": peak,
            "total_work_elements": total_work,
            "estimated_seconds_nominal": nominal_seconds,
            "estimated_seconds_upper": upper_seconds,
        },
        "operational_counts": {
            "planning_mask_decode_count": counters["mask_decode_count"],
            "planning_cache_read_count": counters["cache_read_count"],
            "planning_cache_miss_count": 0,
            "planning_cache_write_count": 0,
            "planning_geometry_build_count": 0,
        },
        "portable_privacy": {
            "raw_pair_component_member_or_path_identifiers_present": False,
            "anonymous_per_record_fields_present": [
                "ordinal",
                "geometry_commitment",
                "candidate_commitment",
                "candidate_count",
                "candidate_sequence_lengths",
                "max_sequence_a",
                "max_sequence_b",
                "bucket",
                "exact_costs",
            ],
            "anonymous_complexity_may_form_a_fingerprint": True,
            "cryptographic_privacy_claimed": False,
        },
    }
    receipt["content_sha256"] = _sha256(_canonical_json(receipt))
    portable = _portable(receipt, "batch_plan")
    return FrozenLocalQ1BatchPlan(
        receipt=portable,
        content_sha256=portable["content_sha256"],
        canonical_file_sha256=_sha256(_canonical_json(portable)),
    )


def write_local_q1_batch_plan(path: Path, plan: FrozenLocalQ1BatchPlan) -> str:
    """Write one canonical plan without overwriting an existing artifact."""

    if not isinstance(plan, FrozenLocalQ1BatchPlan):
        raise TypeError("plan must be FrozenLocalQ1BatchPlan")
    target = Path(path)
    if target.exists():
        raise LocalQ1ProviderError("batch-plan output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(plan.receipt)
    try:
        with target.open("xb") as stream:
            stream.write(payload)
            stream.flush()
    except OSError as exc:
        raise LocalQ1ProviderError("cannot write batch-plan receipt") from exc
    observed = _sha256(payload)
    if observed != plan.canonical_file_sha256:
        raise LocalQ1ProviderError("canonical plan serialization changed")
    return observed


def reopen_local_q1_batch_plan(
    path: Path, *, external_locks: LocalQ1ExternalLocks
) -> FrozenLocalQ1BatchPlan:
    """Open only an externally file/content-locked canonical plan."""

    if not isinstance(external_locks, LocalQ1ExternalLocks):
        raise TypeError("external_locks must be LocalQ1ExternalLocks")
    if external_locks.batch_plan_file_sha256 is None:
        raise LocalQ1ProviderError("batch-plan external locks are required")
    target = Path(path)
    try:
        if target.is_symlink() or not target.is_file():
            raise LocalQ1ProviderError("batch-plan receipt is missing")
        payload = target.read_bytes()
    except OSError as exc:
        raise LocalQ1ProviderError("cannot read batch-plan receipt") from exc
    if not hmac.compare_digest(_sha256(payload), external_locks.batch_plan_file_sha256):
        raise LocalQ1ProviderError("batch-plan external file hash mismatch")
    value = _strict_json_loads(payload, "batch-plan receipt")
    if payload != _canonical_json(value):
        raise LocalQ1ProviderError("batch-plan receipt is not canonical JSON")
    plan = FrozenLocalQ1BatchPlan(
        receipt=value,
        content_sha256=external_locks.batch_plan_content_sha256,
        canonical_file_sha256=external_locks.batch_plan_file_sha256,
    )
    if plan.receipt["external_locks"] != external_locks.without_batch_plan():
        raise LocalQ1ProviderError("batch plan was built under different locks")
    return plan


def _tensor_digest(digest: "hashlib._Hash", name: str, value: Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    array = tensor.numpy()
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))


def local_q1_prepared_digests(
    batch: RaggedGeometryBatch,
) -> Tuple[str, str]:
    """Return authoritative ``(prepared_input, local_candidate)`` SHA-256s."""

    # ``local_candidate_sha256`` is deliberately supervision-blind.  It binds
    # every local model input, the per-sample geometry-validity masks consumed
    # by local aggregation, and the immutable candidate/config identities, but
    # excludes labels and direction targets.  Matchers using the same
    # representation therefore share one candidate artifact even when
    # supervision is audited separately.
    local = hashlib.sha256()
    local.update(b"dunhuang-local-candidate-digest/0.2\0")
    for name in (
        "local_a",
        "local_b",
        "token_mask_a",
        "token_mask_b",
        "correspondence_mask",
        "sample_index",
        "direction_index",
        "candidate_valid",
        "direction_slot_valid",
        "geometry_valid",
    ):
        _tensor_digest(local, name, getattr(batch, name))
    local.update(
        _canonical_json(
            {
                "candidate_ids": list(batch.candidate_ids),
                "candidate_representation": batch.candidate_representation,
                "config_fingerprint": batch.config_fingerprint,
                "direction_names": list(batch.direction_names),
                "geometry_cache_keys": list(batch.geometry_cache_keys),
                "sequence_length_buckets": list(batch.sequence_length_buckets),
            }
        )
    )
    local_sha = local.hexdigest()

    # ``prepared_input_sha256`` commits the complete backend-visible batch:
    # all model inputs, all supervision tensors, all validity tensors, the
    # frozen config, and the supervision-blind local artifact digest above.
    prepared = hashlib.sha256()
    prepared.update(b"dunhuang-prepared-input-digest/0.2\0")
    for name in (
        "coarse_a",
        "coarse_b",
        "local_a",
        "local_b",
        "token_mask_a",
        "token_mask_b",
        "correspondence_mask",
        "sample_index",
        "direction_index",
        "candidate_valid",
        "labels",
        "direction_target",
        "direction_target_valid",
        "direction_slot_valid",
        "geometry_valid",
    ):
        _tensor_digest(prepared, name, getattr(batch, name))
    if batch.exact_assignment_target_a is not None:
        _tensor_digest(
            prepared,
            "exact_assignment_target_a",
            batch.exact_assignment_target_a,
        )
        _tensor_digest(
            prepared,
            "exact_assignment_target_b",
            batch.exact_assignment_target_b,
        )
        prepared.update(
            _canonical_json(
                {
                    "exact_supervision_config": _portable(
                        batch.exact_supervision_config,
                        "exact_supervision_config",
                    ),
                    "exact_targets_are_model_inputs": False,
                }
            )
        )
    prepared.update(
        _canonical_json(
            {
                "config_fingerprint": batch.config_fingerprint,
                "candidate_representation": batch.candidate_representation,
                "local_candidate_sha256": local_sha,
            }
        )
    )
    return prepared.hexdigest(), local_sha


def _semantic_candidate_rows(batch: RaggedGeometryBatch) -> Tuple[Tuple[Any, ...], ...]:
    rows = []
    for index, global_id in enumerate(batch.candidate_ids):
        parts = global_id.split(":", 2)
        if len(parts) != 3:
            raise LocalQ1ProviderError("geometry batch candidate id is malformed")
        length_a = int(batch.token_mask_a[index].shape[0])
        length_b = int(batch.token_mask_b[index].shape[0])
        # Token-mask shapes are batch padding, so use the true valid counts.
        true_a = int(batch.token_mask_a[index].sum().item())
        true_b = int(batch.token_mask_b[index].sum().item())
        if true_a <= 0 or true_b <= 0 or true_a > length_a or true_b > length_b:
            raise LocalQ1ProviderError("geometry batch token lengths are invalid")
        rows.append(
            (
                int(batch.sample_index[index].item()),
                parts[2],
                batch.direction_names[int(batch.direction_index[index].item())],
                true_a,
                true_b,
            )
        )
    return tuple(rows)


def _candidate_digest_from_batch(
    batch: RaggedGeometryBatch, record_hashes: Sequence[str]
) -> str:
    by_record = [[] for _ in record_hashes]
    for (
        sample_index,
        candidate_id,
        direction,
        length_a,
        length_b,
    ) in _semantic_candidate_rows(batch):
        by_record[sample_index].append([candidate_id, direction, length_a, length_b])
    return _sha256(
        _canonical_json(
            [
                {
                    "record_geometry_sha256": record_hashes[index],
                    "ordered_candidates": candidates,
                }
                for index, candidates in enumerate(by_record)
            ]
        )
    )


class LocalQ1ReadOnlyBatchProvider:
    """Materialize frozen-record LOCAL-Q1 batches for all local representations."""

    def __init__(
        self,
        *,
        opened: OpenedLocalQ1Cache,
        loader_factory: Callable[[], Callable[[MaskMemberRef], np.ndarray]],
        plan: FrozenLocalQ1BatchPlan,
        geometry_config: GeometryBatchConfig,
        inventory_config: LocalCacheInventoryConfig,
        external_locks: LocalQ1ExternalLocks,
    ) -> None:
        _validate_external_locks(opened, external_locks, require_batch_plan=True)
        if not callable(loader_factory):
            raise TypeError("loader_factory must be callable")
        if not isinstance(plan, FrozenLocalQ1BatchPlan):
            raise TypeError("plan must be FrozenLocalQ1BatchPlan")
        if not isinstance(geometry_config, GeometryBatchConfig):
            raise TypeError("geometry_config must be GeometryBatchConfig")
        if not isinstance(inventory_config, LocalCacheInventoryConfig):
            raise TypeError("inventory_config must be LocalCacheInventoryConfig")
        if (
            plan.canonical_file_sha256 != external_locks.batch_plan_file_sha256
            or plan.content_sha256 != external_locks.batch_plan_content_sha256
            or plan.receipt["external_locks"] != external_locks.without_batch_plan()
            or plan.receipt["geometry_config_sha256"] != geometry_config.fingerprint
            or plan.receipt["inventory_config_sha256"]
            != _sha256(_canonical_json(inventory_config.portable_dict()))
        ):
            raise LocalQ1ProviderError("provider config/plan external lock mismatch")
        validation_assignment = freeze_local_q1_validation_assignment(
            opened.population.validation_records
        )
        if _canonical_json(validation_assignment.receipt) != _canonical_json(
            plan.validation_assignment
        ):
            raise LocalQ1ProviderError(
                "provider population differs from frozen validation assignment"
            )
        approved = ApprovedCoarsePreprocessing.from_geometry_config(geometry_config)
        self.contract = BatchProviderContract(
            coarse_preprocess_mode=geometry_config.coarse_preprocess_mode,
            coarse_preprocessing_sha256=approved.preprocessing_sha256,
            geometry_config_sha256=geometry_config.fingerprint,
            cache_interface=(
                "same_externally_verified_read_only_LOCAL_Q1_fragment_cache_"
                "zero_miss_zero_write"
            ),
            provider_version=LOCAL_Q1_BATCH_PROVIDER_VERSION,
        )
        self._opened = opened
        self._loader_factory = loader_factory
        self._plan = plan
        self._geometry_config = geometry_config
        self._inventory_config = inventory_config
        self._approved = approved
        self._validation_assignment = validation_assignment
        self._entries: Dict[Tuple[str, str], Mapping[str, Any]] = {}
        for phase in _FORMAL_PHASES:
            for entry in plan.phase_batches(phase):
                key = (phase, entry["batch_geometry_sha256"])
                if key in self._entries:
                    raise LocalQ1ProviderError("plan has an ambiguous batch geometry")
                self._entries[key] = entry
        self._counts = {
            "planned_safety_metadata_call_count": 0,
            "planned_safety_metadata_record_count": 0,
            "planned_records_call_count": 0,
            "planned_records_record_count": 0,
            "prepare_call_count": 0,
            "prepared_record_count": 0,
            "prepared_candidate_count": 0,
            "mask_decode_count": 0,
            "cache_read_count": 0,
            "cache_miss_count": 0,
            "cache_write_count": 0,
            "geometry_build_count": 0,
        }
        self._arm_counts: Dict[str, int] = {}
        self._prepared_digest = hashlib.sha256()
        self._candidate_digest = hashlib.sha256()

    def _checked_plan_batch(
        self, phase: str, batch_ordinal: int
    ) -> Tuple[Mapping[str, Any], Sequence[TrainingPairRecord]]:
        if phase not in _PHASE_TO_SPLIT:
            raise LocalQ1ProviderError("phase must be one of the four formal phases")
        if type(batch_ordinal) is not int or batch_ordinal < 0:
            raise ValueError("batch_ordinal must be a non-negative integer")
        batches = self._plan.phase_batches(phase)
        if batch_ordinal >= len(batches):
            raise LocalQ1ProviderError("batch ordinal is outside the frozen phase")
        source = (
            self._opened.population.training_records
            if phase == "train"
            else self._opened.population.validation_records
        )
        return batches[batch_ordinal], source

    def planned_safety_metadata(
        self, phase: str, batch_ordinal: int
    ) -> FrozenLocalQ1PlannedSafetyMetadata:
        """Return path/ID-free preflight metadata without materializing records.

        Only population ordinals, split, component ID hashing, and the bounded
        shared recursive provenance safety scan are used.  Labels, directions,
        masks, members, pairs, fragments, and their identifiers are untouched.
        """

        entry, source = self._checked_plan_batch(phase, batch_ordinal)
        expected_split = _PHASE_TO_SPLIT[phase]
        phase_receipt = self._plan.receipt["phases"][phase]
        phase_ordinals = tuple(
            ordinal
            for batch in phase_receipt["batches"]
            for ordinal in batch["record_ordinals"]
        )
        phase_components: Dict[str, str] = {}
        for population_ordinal in phase_ordinals:
            try:
                record = source[population_ordinal]
            except IndexError as exc:  # pragma: no cover - frozen plan owns this
                raise LocalQ1ProviderError(
                    "frozen safety metadata ordinal is out of range"
                ) from exc
            if (
                not isinstance(record, TrainingPairRecord)
                or record.split != expected_split
            ):
                raise LocalQ1ProviderError(
                    "safety metadata population split differs from phase"
                )
            token = _validation_component_token(record.component_id)
            previous = phase_components.get(token)
            if previous is not None and previous != record.component_id:
                raise LocalQ1ProviderError("safety metadata component hash collision")
            phase_components[token] = record.component_id
            if (
                phase != "train"
                and self._validation_assignment.phase_by_component.get(
                    record.component_id
                )
                != phase
            ):
                raise LocalQ1ProviderError(
                    "safety metadata differs from validation assignment"
                )
        phase_component_set_sha256 = _sha256(_canonical_json(sorted(phase_components)))
        assignment = self._plan.validation_assignment
        assignment_phase_component_set_sha256 = (
            None
            if phase == "train"
            else assignment["phases"][phase]["component_set_sha256"]
        )
        if (
            assignment_phase_component_set_sha256 is not None
            and phase_component_set_sha256 != assignment_phase_component_set_sha256
        ):
            raise LocalQ1ProviderError(
                "safety metadata phase components differ from assignment"
            )

        rows = []
        batch_components: Dict[str, str] = {}
        for population_ordinal in entry["record_ordinals"]:
            try:
                record = source[population_ordinal]
            except IndexError as exc:  # pragma: no cover - frozen plan owns this
                raise LocalQ1ProviderError(
                    "frozen safety metadata ordinal is out of range"
                ) from exc
            if (
                not isinstance(record, TrainingPairRecord)
                or record.split != expected_split
            ):
                raise LocalQ1ProviderError(
                    "safety metadata population split differs from phase"
                )
            component_token = _validation_component_token(record.component_id)
            previous = batch_components.get(component_token)
            if previous is not None and previous != record.component_id:
                raise LocalQ1ProviderError("safety metadata component hash collision")
            batch_components[component_token] = record.component_id
            provenance_markers = scan_local_q1_provenance_safety(record.provenance)
            row = LocalQ1PlannedSafetyRecord(
                population_ordinal=population_ordinal,
                expected_split=expected_split,
                component_token_sha256=component_token,
                sealed_real_test_marker_explicit_false=(
                    provenance_markers.top_level_sealed_marker_explicit_false
                ),
                sealed_scope_marker=provenance_markers.sealed_scope_marker,
                historical_test_marker=provenance_markers.historical_test_marker,
            )
            rows.append(row.portable_dict())
        receipt = {
            "schema_version": LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION,
            "status": (
                "frozen_authoritative_population_safety_metadata_no_materialization"
            ),
            "provider_version": LOCAL_Q1_BATCH_PROVIDER_VERSION,
            "batch_plan_file_sha256": self._plan.canonical_file_sha256,
            "batch_plan_content_sha256": self._plan.content_sha256,
            "validation_assignment_sha256": assignment["assignment_sha256"],
            "phase": phase,
            "batch_ordinal": batch_ordinal,
            "expected_split": expected_split,
            "plan_batch_geometry_sha256": entry["batch_geometry_sha256"],
            "plan_phase_population_ordinal_set_sha256": phase_receipt[
                "population_ordinal_set_sha256"
            ],
            "component_commitment_kind": (
                "authoritative_train_population_component_tokens"
                if phase == "train"
                else "frozen_validation_assignment_phase_component_tokens"
            ),
            "phase_component_set_sha256": phase_component_set_sha256,
            "assignment_phase_component_set_sha256": (
                assignment_phase_component_set_sha256
            ),
            "record_count": len(rows),
            "population_ordinal_sequence_sha256": _sha256(
                _canonical_json(list(entry["record_ordinals"]))
            ),
            "batch_component_set_sha256": _sha256(
                _canonical_json(sorted(batch_components))
            ),
            "records": rows,
            "privacy": dict(_SAFETY_METADATA_PRIVACY),
        }
        receipt["content_sha256"] = _sha256(_canonical_json(receipt))
        metadata = FrozenLocalQ1PlannedSafetyMetadata(
            receipt=receipt,
            content_sha256=receipt["content_sha256"],
        )
        if (
            tuple(row.population_ordinal for row in metadata.records)
            != tuple(entry["record_ordinals"])
            or metadata.receipt["plan_batch_geometry_sha256"]
            != entry["batch_geometry_sha256"]
            or metadata.receipt["plan_phase_population_ordinal_set_sha256"]
            != phase_receipt["population_ordinal_set_sha256"]
        ):
            raise LocalQ1ProviderError(
                "planned safety metadata differs from frozen batch authority"
            )
        self._counts["planned_safety_metadata_call_count"] += 1
        self._counts["planned_safety_metadata_record_count"] += len(rows)
        return metadata

    def planned_records(
        self, phase: str, batch_ordinal: int
    ) -> Tuple[TrainingPairRecord, ...]:
        """Return one complete plan-authoritative record batch by ordinal."""

        entry, source = self._checked_plan_batch(phase, batch_ordinal)
        try:
            records = tuple(source[index] for index in entry["record_ordinals"])
        except IndexError as exc:  # pragma: no cover - plan constructor owns this
            raise LocalQ1ProviderError("frozen phase ordinal is out of range") from exc
        if phase != "train" and any(
            self._validation_assignment.phase_by_component[record.component_id] != phase
            for record in records
        ):
            raise LocalQ1ProviderError(
                "frozen phase records differ from component assignment"
            )
        self._counts["planned_records_call_count"] += 1
        self._counts["planned_records_record_count"] += len(records)
        return records

    def prepare(
        self,
        records: Sequence[TrainingPairRecord],
        *,
        arm: AblationArm,
        phase: str,
    ) -> PreparedAblationBatch:
        if not isinstance(arm, AblationArm):
            raise TypeError("arm must be AblationArm")
        if not arm.uses_local_geometry:
            raise LocalQ1ProviderError(
                "LOCAL-Q1 provider serves local and fused arms only"
            )
        if phase not in _PHASE_TO_SPLIT:
            raise LocalQ1ProviderError("phase must be one of the four formal phases")
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("records must be a finite sequence")
        population = tuple(records)
        if not population:
            raise LocalQ1ProviderError("provider batch cannot be empty")
        split = _PHASE_TO_SPLIT[phase]
        if any(
            not isinstance(record, TrainingPairRecord) or record.split != split
            for record in population
        ):
            raise LocalQ1ProviderError("provider batch split differs from phase")
        if phase != "train" and any(
            self._validation_assignment.phase_by_component.get(record.component_id)
            != phase
            for record in population
        ):
            raise LocalQ1ProviderError(
                "provider validation batch crosses frozen component phases"
            )

        raw_loader = self._loader_factory()
        if not callable(raw_loader):
            raise TypeError("loader factory must return a callable")
        masks: Dict[MaskMemberRef, np.ndarray] = {}
        try:
            for record in population:
                for reference in (record.fragment_a, record.fragment_b):
                    if reference not in masks:
                        masks[reference] = _validated_mask(
                            raw_loader,
                            reference,
                            self._geometry_config.max_input_pixels_per_mask,
                        )
            record_hashes = tuple(
                _sha256(
                    _canonical_json(
                        _record_geometry_payload(
                            record,
                            masks[record.fragment_a],
                            masks[record.fragment_b],
                            self._geometry_config,
                        )
                    )
                )
                for record in population
            )
            batch_geometry_sha = _sha256(_canonical_json(list(record_hashes)))
            entry = self._entries.get((phase, batch_geometry_sha))
            if entry is None or list(record_hashes) != entry["record_geometry_sha256"]:
                raise LocalQ1ProviderError(
                    "records are not one complete frozen batch in frozen order"
                )
            keypoint_config = None
            if arm.candidate_representation == KEYPOINT_REPRESENTATION:
                declared_keypoints = arm.model_config.get("contour_keypoint_config")
                if not isinstance(declared_keypoints, Mapping):
                    raise LocalQ1ProviderError(
                        "keypoint arm lacks a frozen selector configuration"
                    )
                try:
                    keypoint_config = ContourKeypointConfig(**dict(declared_keypoints))
                except (TypeError, ValueError) as exc:
                    raise LocalQ1ProviderError(
                        "keypoint selector configuration is invalid"
                    ) from exc
            batch = build_geometry_batch(
                population,
                masks.__getitem__,
                self._geometry_config,
                geometry_artifact_cache=self._opened.cache,
                candidate_representation=arm.candidate_representation,
                keypoint_config=keypoint_config,
            )
        finally:
            _close_loader(raw_loader)
        if (
            batch.candidate_representation == MULTIRUN_REPRESENTATION
            and batch.config_fingerprint != self._geometry_config.fingerprint
        ):
            raise LocalQ1ProviderError("prepared geometry config changed")
        observed_complexity = batch.complexity_receipt.to_dict()
        expected_complexity = entry["complexity"]
        if batch.candidate_representation == MULTIRUN_REPRESENTATION:
            for name in (
                "candidate_count",
                "padded_sequence_a",
                "padded_sequence_b",
                "local_tensor_elements",
                "attention_score_elements_per_head",
                "affinity_elements",
                "sinkhorn_elements",
            ):
                if observed_complexity[name] != expected_complexity[name]:
                    raise LocalQ1ProviderError(
                        "prepared batch complexity differs from frozen plan"
                    )
        observed_candidate_semantic = _candidate_digest_from_batch(batch, record_hashes)
        expected_candidate_semantic = _sha256(
            _canonical_json(
                [
                    {
                        "record_geometry_sha256": record_hashes[index],
                        "ordered_candidates": [
                            list(row[1:])
                            for row in _semantic_candidate_rows(batch)
                            if row[0] == index
                        ],
                    }
                    for index in range(len(record_hashes))
                ]
            )
        )
        if observed_candidate_semantic != expected_candidate_semantic:
            raise LocalQ1ProviderError("internal candidate digest is unstable")
        if batch.candidate_representation == MULTIRUN_REPRESENTATION:
            # The original plan freezes multi-run candidate semantics.  The
            # keypoint arms deliberately share its record ordinals and cache,
            # then derive a different label-blind representation at prepare.
            plan_candidate = _sha256(
                _canonical_json(
                    [
                        {
                            "record_geometry_sha256": record_hashes[index],
                            "candidate_semantic_sha256": entry[
                                "record_candidate_sha256"
                            ][index],
                            "ordered_candidates": [
                                list(row[1:])
                                for row in _semantic_candidate_rows(batch)
                                if row[0] == index
                            ],
                        }
                        for index in range(len(record_hashes))
                    ]
                )
            )
            if plan_candidate != entry["candidate_semantic_sha256"]:
                raise LocalQ1ProviderError(
                    "prepared candidates differ from frozen plan"
                )
        elif batch.candidate_representation != KEYPOINT_REPRESENTATION:
            raise LocalQ1ProviderError("unsupported candidate representation")
        prepared_sha, local_sha = local_q1_prepared_digests(batch)
        unique_references = len(masks)
        unique_cache_keys = len(
            {
                fragment_cache_identity(
                    mask,
                    reference.threshold_rule,
                    self._geometry_config.geometry,
                ).key
                for reference, mask in masks.items()
            }
        )
        counts = {
            "coarse_preprocess_count": 2 * len(population),
            "geometry_build_count": 0,
            "geometry_cache_read_count": unique_cache_keys,
            "geometry_cache_write_count": 0,
            "local_candidate_count": batch.candidate_count,
            "mask_load_count": unique_references,
        }
        self._counts["prepare_call_count"] += 1
        self._counts["prepared_record_count"] += len(population)
        self._counts["prepared_candidate_count"] += batch.candidate_count
        self._counts["mask_decode_count"] += unique_references
        self._counts["cache_read_count"] += unique_cache_keys
        self._arm_counts[arm.name.value] = self._arm_counts.get(arm.name.value, 0) + 1
        self._prepared_digest.update(prepared_sha.encode("ascii"))
        self._candidate_digest.update(local_sha.encode("ascii"))
        return PreparedAblationBatch(
            payload=batch,
            sample_count=len(population),
            record_sequence_sha256=record_sequence_fingerprint(population),
            prepared_input_sha256=prepared_sha,
            local_candidate_sha256=local_sha,
            coarse_preprocessing_sha256=self._approved.preprocessing_sha256,
            geometry_config_sha256=self._geometry_config.fingerprint,
            processing_counts=counts,
            candidate_representation=batch.candidate_representation,
        )

    def portable_receipt(self) -> Mapping[str, Any]:
        """Return current aggregate provider counters without row identities."""

        receipt = {
            "schema_version": LOCAL_Q1_PROVIDER_RUNTIME_SCHEMA_VERSION,
            "status": "read_only_zero_miss_zero_write_no_model_backend",
            "provider_version": LOCAL_Q1_BATCH_PROVIDER_VERSION,
            "batch_plan_file_sha256": self._plan.canonical_file_sha256,
            "batch_plan_content_sha256": self._plan.content_sha256,
            "counts": dict(self._counts),
            "arm_prepare_call_counts": dict(sorted(self._arm_counts.items())),
            "prepared_call_sequence_sha256": self._prepared_digest.hexdigest(),
            "candidate_call_sequence_sha256": self._candidate_digest.hexdigest(),
            "resource_estimate": self._plan.receipt["resource_estimate"],
            "safety_metadata_api": {
                "schema_version": (LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION),
                "method": "planned_safety_metadata",
                "return_type": "FrozenLocalQ1PlannedSafetyMetadata",
                "source": "plan_ordinals_and_authoritative_population_safety_fields",
                "calls_planned_records": False,
                "materializes_masks_or_batches": False,
                "provenance_safety_scan": _portable(
                    dict(local_q1_provenance_safety_policy()),
                    "provider_runtime_provenance_safety_policy",
                ),
            },
            "portable_privacy": {
                "paths_present": False,
                "row_or_fragment_ids_present": False,
                "aggregate_counters_and_hashes_only": True,
            },
        }
        receipt["content_sha256"] = _sha256(_canonical_json(receipt))
        return MappingProxyType(_portable(receipt, "provider_receipt"))


__all__ = [
    "LOCAL_Q1_BATCH_PLAN_SCHEMA_VERSION",
    "LOCAL_Q1_BATCH_PROVIDER_VERSION",
    "LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION",
    "LOCAL_Q1_PROVENANCE_SAFETY_MAX_DEPTH",
    "LOCAL_Q1_PROVENANCE_SAFETY_MAX_NODES",
    "LOCAL_Q1_PROVENANCE_SAFETY_SCAN_VERSION",
    "LOCAL_Q1_PROVIDER_RUNTIME_SCHEMA_VERSION",
    "LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE",
    "LOCAL_Q1_VALIDATION_ASSIGNMENT_SCHEMA_VERSION",
    "FrozenLocalQ1BatchPlan",
    "FrozenLocalQ1PlannedSafetyMetadata",
    "FrozenLocalQ1ValidationAssignment",
    "LocalQ1BatchPlanConfig",
    "LocalQ1ExternalLocks",
    "LocalQ1ProviderError",
    "LocalQ1PlannedSafetyRecord",
    "LocalQ1ProvenanceSafetyMarkers",
    "LocalQ1ReadOnlyBatchProvider",
    "build_local_q1_batch_plan",
    "freeze_local_q1_validation_assignment",
    "local_q1_prepared_digests",
    "local_q1_provenance_safety_policy",
    "reopen_local_q1_batch_plan",
    "scan_local_q1_provenance_safety",
    "write_local_q1_batch_plan",
]
