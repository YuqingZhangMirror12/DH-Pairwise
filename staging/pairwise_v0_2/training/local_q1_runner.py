"""Dedicated, fail-closed LOCAL-Q1 train/validation runner.

This runner deliberately does not reuse the generic short-ablation scheduler.
Every train and validation batch is addressed by the externally locked
``FrozenLocalQ1BatchPlan``.  The plan is also the sole authority for the
label-blind, component-disjoint validation partition:

* ``validation_select`` is replayed after every epoch and selects a winner;
* ``validation_calibration`` is read once after winner selection and fits the
  validation-only threshold; and
* ``validation_report`` is read once, with the frozen threshold, for the final
  synthetic validation report.

The runner trains dual-softmax and dustbin-Sinkhorn on both the established
multi-run sliding-window representation and the contour-keypoint
representation.  All four arms replay identical plan ordinals and seeds.
Fused remains a separate second-stage entry.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np
import torch
from torch import nn

from staging.pairwise_v0_2.models.local_matcher import MatcherMode
from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.training.checkpoint import (
    CheckpointReceipt,
    canonical_config_hash,
    load_trusted_checkpoint,
    save_checkpoint,
)
from staging.pairwise_v0_2.training.evaluation import (
    evaluate_pairwise,
    fit_pairwise_threshold,
)
from staging.pairwise_v0_2.training.local_q1_backend import (
    CheckpointModelScope,
    FusedCheckpointBindings,
    LOCAL_Q1_AUTHORITY_KIND,
    LOCAL_Q1_AUTHORITY_STATUS,
    LOCAL_Q1_BACKEND_VERSION,
    LocalQ1Backend,
    LocalQ1BackendError,
    LocalQ1BackendMode,
    LocalQ1OptimizerConfig,
    LocalQ1StepConfig,
    TrustedCheckpointBinding,
    checkpoint_authority_identity_sha256,
    checkpoint_semantic_projection,
)
from staging.pairwise_v0_2.training.local_q1_provider import (
    LOCAL_Q1_BATCH_PROVIDER_VERSION,
    LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION,
    LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE,
    FrozenLocalQ1BatchPlan,
    FrozenLocalQ1PlannedSafetyMetadata,
    LocalQ1PlannedSafetyRecord,
    LocalQ1ReadOnlyBatchProvider,
    local_q1_provenance_safety_policy,
    scan_local_q1_provenance_safety,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    BackendContract,
    BatchProviderContract,
    EvidenceMode,
    ExecutionKind,
    PredictionBatch,
    PreparedAblationBatch,
    TrainBatchResult,
    record_sequence_fingerprint,
)


LOCAL_Q1_RUNNER_SCHEMA_VERSION = "dunhuang-local-q1-runner/0.5"
LOCAL_Q1_RUNNER_CONTRACT_SCHEMA_VERSION = "dunhuang-local-q1-runner-contract/0.5"
LOCAL_Q1_CHECKPOINT_AUTHORITY_SCHEMA_VERSION = (
    "dunhuang-local-q1-checkpoint-authority/0.5"
)
LOCAL_Q1_CHECKPOINT_CONFIG_SCHEMA_VERSION = "dunhuang-local-q1-checkpoint-config/0.5"

_TRAIN_PHASE = "train"
_SELECT_PHASE = "validation_select"
_CALIBRATION_PHASE = "validation_calibration"
_REPORT_PHASE = "validation_report"
_PHASES = (_TRAIN_PHASE, _SELECT_PHASE, _CALIBRATION_PHASE, _REPORT_PHASE)
_VALIDATION_PHASES = (_SELECT_PHASE, _CALIBRATION_PHASE, _REPORT_PHASE)
_LOCAL_ARMS = (
    AblationArmName.LOCAL_DUAL_SOFTMAX,
    AblationArmName.LOCAL_DUSTBIN_SINKHORN,
    AblationArmName.KEYPOINT_DUAL_SOFTMAX,
    AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
)
_SELECTION_POLICY = (
    "max_equal_dataset_macro_cluster_auroc_then_auprc_then_earlier_epoch"
)
_FORMAL_BACKEND_MODEL_FAMILY = "DunhuangPairwiseV02-LOCAL-Q1-five-arm"
_FORMAL_BACKEND_VERSION = (
    LOCAL_Q1_BACKEND_VERSION + "/" + LocalQ1BackendMode.FORMAL.value
)
_SHA256_CHARS = frozenset("0123456789abcdef")


class LocalQ1RunnerError(RuntimeError):
    """A frozen runner, plan, result, or authority invariant failed."""


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not set(value).issubset(_SHA256_CHARS)
    ):
        raise LocalQ1RunnerError("{} must be lowercase SHA-256".format(name))
    return value


def _portable(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return _portable(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _portable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_portable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LocalQ1RunnerError("portable value contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _portable(value.item())
    raise TypeError("unsupported portable type: {}".format(type(value).__name__))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _portable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _self_content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return _content_sha256(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_string(value: str) -> bool:
    return not (
        value.startswith(("/", "~/", "file:", "\\\\"))
        or (len(value) >= 3 and value[1:3] in {":\\", ":/"})
    )


def _assert_portable(value: Any) -> None:
    forbidden = {
        "archive_member",
        "canonical_group_id",
        "component_id",
        "fragment_id",
        "pair_id",
        "path",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in forbidden:
                raise LocalQ1RunnerError(
                    "portable receipt exposes an identity/path field"
                )
            _assert_portable(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_portable(item)
    elif isinstance(value, str) and not _portable_string(value):
        raise LocalQ1RunnerError("portable receipt contains a machine-local path")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = _canonical_json(value)
    target = Path(path)
    if target.exists():
        raise LocalQ1RunnerError("artifact overwrite is forbidden")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + target.name + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LocalQ1RunnerContract:
    """Externally preregistered execution constants for four local arms."""

    epochs: int
    initialization_seed: int
    batch_plan_file_sha256: str
    batch_plan_content_sha256: str
    min_training_valid_fraction: float = 0.5
    min_validation_valid_fraction: float = 0.5
    production: bool = True
    schema_version: str = LOCAL_Q1_RUNNER_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOCAL_Q1_RUNNER_CONTRACT_SCHEMA_VERSION:
            raise LocalQ1RunnerError("LOCAL-Q1 runner contract schema changed")
        if type(self.epochs) is not int or self.epochs <= 0:  # noqa: E721
            raise LocalQ1RunnerError("epochs must be a positive built-in int")
        if (
            type(self.initialization_seed) is not int  # noqa: E721
            or self.initialization_seed < 0
        ):
            raise LocalQ1RunnerError(
                "initialization_seed must be a non-negative built-in int"
            )
        _require_sha256(self.batch_plan_file_sha256, "batch-plan file SHA-256")
        _require_sha256(self.batch_plan_content_sha256, "batch-plan content SHA-256")
        for name in (
            "min_training_valid_fraction",
            "min_validation_valid_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise LocalQ1RunnerError("{} must be in (0, 1]".format(name))
        if type(self.production) is not bool:  # noqa: E721
            raise TypeError("production must be bool")

    def portable_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "epochs": self.epochs,
                "initialization_seed": self.initialization_seed,
                "batch_plan_file_sha256": self.batch_plan_file_sha256,
                "batch_plan_content_sha256": self.batch_plan_content_sha256,
                "min_training_valid_fraction": self.min_training_valid_fraction,
                "min_validation_valid_fraction": self.min_validation_valid_fraction,
                "production": self.production,
                "arms": [name.value for name in _LOCAL_ARMS],
                "phase_policy": {
                    "pre_session_safety_metadata_reads": list(_PHASES),
                    "pre_session_safety_metadata_supervision_fields_read": [],
                    "provenance_safety_scan": _portable(
                        dict(local_q1_provenance_safety_policy())
                    ),
                    "epoch_selection_reads": [_SELECT_PHASE],
                    "threshold_fitting_reads": [_CALIBRATION_PHASE],
                    "final_reporting_reads": [_REPORT_PHASE],
                    "validation_partitions_component_disjoint": True,
                    "validation_assignment_label_blind": True,
                    "all_winner_replays_before_calibration_population_lookup": True,
                    "all_thresholds_frozen_before_report_population_lookup": True,
                    "production_external_validation_records_executed": False,
                },
                "winner_selection_policy": _SELECTION_POLICY,
                "fused_in_this_stage": False,
            }
        )

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.portable_dict())


class _RunnerProvider(Protocol):
    contract: BatchProviderContract

    def planned_safety_metadata(
        self, phase: str, batch_ordinal: int
    ) -> FrozenLocalQ1PlannedSafetyMetadata: ...

    def planned_records(
        self, phase: str, batch_ordinal: int
    ) -> Tuple[TrainingPairRecord, ...]: ...

    def prepare(
        self,
        records: Sequence[TrainingPairRecord],
        *,
        arm: AblationArm,
        phase: str,
    ) -> PreparedAblationBatch: ...

    def portable_receipt(self) -> Mapping[str, Any]: ...


class _RunnerSession(Protocol):
    model: nn.Module
    optimizer: torch.optim.Optimizer
    initial_model_state_sha256: str

    def train_batch(self, batch: PreparedAblationBatch) -> TrainBatchResult: ...

    def predict_batch(
        self, batch: PreparedAblationBatch, *, evidence: EvidenceMode
    ) -> PredictionBatch: ...


class _RunnerBackend(Protocol):
    contract: BackendContract
    optimizer_config: Mapping[str, Any]
    aggregation_config: Mapping[str, Any]
    model_template: Any

    def model_config_for(self, name: AblationArmName) -> Mapping[str, Any]: ...

    def create_session(self, arm: AblationArm, *, seed: int) -> _RunnerSession: ...


@dataclass(frozen=True)
class LocalQ1RunArtifacts:
    run_directory: Path
    receipt_path: Path
    checkpoint_paths: Mapping[str, Tuple[Path, ...]]
    authority_receipt_paths: Mapping[str, Path]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class _SavedCheckpoint:
    path: Path
    receipt: CheckpointReceipt
    config: Mapping[str, Any]
    selection_report: Mapping[str, Any]


@dataclass
class _ArmWinnerState:
    arm: AblationArm
    reload_arm: AblationArm
    reload_session: _RunnerSession
    saved: Tuple[_SavedCheckpoint, ...]
    winner: _SavedCheckpoint
    selection_replay: Mapping[str, Any]
    initial_model_state_sha256: str
    optimizer_steps: int
    training_presented_count: int
    training_valid_count: int
    calibration_report: Optional[Mapping[str, Any]] = None
    threshold: Optional[Mapping[str, Any]] = None
    validation_report: Optional[Mapping[str, Any]] = None


def _checked_backend(value: Any, *, production: bool) -> _RunnerBackend:
    contract = getattr(value, "contract", None)
    if not isinstance(contract, BackendContract):
        raise LocalQ1RunnerError("backend lacks a typed BackendContract")
    if contract.sealed_real_test_capability:
        raise LocalQ1RunnerError("backend exposes sealed-real-test capability")
    if production:
        if type(value) is not LocalQ1Backend:  # noqa: E721 - exact trust boundary
            raise LocalQ1RunnerError(
                "production runner requires the exact LocalQ1Backend implementation"
            )
        if (
            value.mode is not LocalQ1BackendMode.FORMAL
            or contract.execution_kind is not ExecutionKind.SYNTHETIC_TRAIN_VALIDATION
            or contract.backend_version != _FORMAL_BACKEND_VERSION
            or contract.model_family != _FORMAL_BACKEND_MODEL_FAMILY
            or contract.device_type != value.device.type
            or value.fused_checkpoint_bindings is not None
        ):
            raise LocalQ1RunnerError("production backend formal identity changed")
        reference = LocalQ1Backend(device="cpu", mode=LocalQ1BackendMode.FORMAL)
        if (
            value.model_template != reference.model_template
            or type(value._optimizer_spec) is not LocalQ1OptimizerConfig  # noqa: SLF001,E721
            or _portable(value._optimizer_spec.to_dict())  # noqa: SLF001
            != _portable(LocalQ1OptimizerConfig().to_dict())
            or _portable(value.optimizer_config)
            != _portable(LocalQ1OptimizerConfig().to_dict())
            or type(value._step_config) is not LocalQ1StepConfig  # noqa: SLF001,E721
            or _portable(value.step_config.to_dict())
            != _portable(LocalQ1StepConfig().to_dict())
            or _portable(value.aggregation_config)
            != _portable(reference.aggregation_config)
        ):
            raise LocalQ1RunnerError("production backend frozen configuration changed")
        for arm_name in _LOCAL_ARMS:
            observed = LocalQ1Backend.model_config_for(value, arm_name)
            expected = LocalQ1Backend.model_config_for(reference, arm_name)
            if not hmac.compare_digest(
                canonical_config_hash(observed), canonical_config_hash(expected)
            ):
                raise LocalQ1RunnerError("production backend arm configuration changed")
    for name in (
        "model_config_for",
        "create_session",
        "optimizer_config",
        "aggregation_config",
        "model_template",
    ):
        if not hasattr(value, name):
            raise LocalQ1RunnerError("backend is missing {}".format(name))
    return value


def _checked_provider(
    value: Any, *, production: bool, plan: FrozenLocalQ1BatchPlan
) -> _RunnerProvider:
    contract = getattr(value, "contract", None)
    if not isinstance(contract, BatchProviderContract):
        raise LocalQ1RunnerError("provider lacks a typed BatchProviderContract")
    for name in (
        "planned_safety_metadata",
        "planned_records",
        "prepare",
        "portable_receipt",
    ):
        if not callable(getattr(value, name, None)):
            raise LocalQ1RunnerError("provider lacks {}".format(name))
    if production:
        if type(value) is not LocalQ1ReadOnlyBatchProvider:  # noqa: E721
            raise LocalQ1RunnerError(
                "production runner requires the exact read-only LOCAL-Q1 provider"
            )
        if any(
            name in vars(value)
            for name in (
                "planned_safety_metadata",
                "planned_records",
                "prepare",
                "portable_receipt",
            )
        ):
            raise LocalQ1RunnerError(
                "production provider methods cannot be instance-overridden"
            )
        provider_plan = getattr(value, "_plan", None)
        if (
            contract.provider_version != LOCAL_Q1_BATCH_PROVIDER_VERSION
            or not isinstance(provider_plan, FrozenLocalQ1BatchPlan)
            or provider_plan.canonical_file_sha256 != plan.canonical_file_sha256
            or provider_plan.content_sha256 != plan.content_sha256
        ):
            raise LocalQ1RunnerError("production provider/plan identity changed")
    return value


def _arm_for(
    backend: _RunnerBackend, name: AblationArmName, *, production: bool
) -> AblationArm:
    matcher = {
        AblationArmName.LOCAL_DUAL_SOFTMAX: MatcherMode.DUAL_SOFTMAX.value,
        AblationArmName.LOCAL_DUSTBIN_SINKHORN: MatcherMode.DUSTBIN_SINKHORN.value,
        AblationArmName.KEYPOINT_DUAL_SOFTMAX: MatcherMode.DUAL_SOFTMAX.value,
        AblationArmName.KEYPOINT_DUSTBIN_SINKHORN: (MatcherMode.DUSTBIN_SINKHORN.value),
    }.get(name)
    if matcher is None:
        raise LocalQ1RunnerError("dedicated local runner accepts four local arms only")
    model_config = (
        LocalQ1Backend.model_config_for(backend, name)
        if production
        else backend.model_config_for(name)
    )
    return AblationArm(
        name=name,
        evidence=EvidenceMode.LOCAL,
        matcher_mode=matcher,
        model_config=model_config,
        optimizer_config=backend.optimizer_config,
        aggregation_config=backend.aggregation_config,
        arc_pooling=backend.model_template.arc_pooling,
    )


def _create_session(
    backend: _RunnerBackend,
    arm: AblationArm,
    *,
    seed: int,
    production: bool,
) -> _RunnerSession:
    if production:
        return LocalQ1Backend.create_session(backend, arm, seed=seed)
    return backend.create_session(arm, seed=seed)


def _checked_session(value: Any) -> _RunnerSession:
    if not isinstance(getattr(value, "model", None), nn.Module):
        raise LocalQ1RunnerError("session model must be torch.nn.Module")
    if not isinstance(getattr(value, "optimizer", None), torch.optim.Optimizer):
        raise LocalQ1RunnerError("session optimizer must be a torch optimizer")
    _require_sha256(
        getattr(value, "initial_model_state_sha256", None),
        "session initial model state SHA-256",
    )
    for name in ("train_batch", "predict_batch"):
        if not callable(getattr(value, name, None)):
            raise LocalQ1RunnerError("session lacks {}".format(name))
    return value


def _plan_phase_receipt(plan: FrozenLocalQ1BatchPlan, phase: str) -> Mapping[str, Any]:
    """Return one exact phase receipt from the four-phase plan schema."""

    if phase not in _PHASES:
        raise LocalQ1RunnerError("unsupported LOCAL-Q1 phase")
    phases = plan.receipt.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != set(_PHASES):
        raise LocalQ1RunnerError(
            "batch plan lacks exact train/select/calibration/report phases"
        )
    value = phases.get(phase)
    if not isinstance(value, Mapping):
        raise LocalQ1RunnerError("batch-plan phase receipt is invalid")
    if value.get("phase") != phase or not isinstance(value.get("batches"), list):
        raise LocalQ1RunnerError("batch-plan phase identity/batches changed")
    return value


def _checked_plan(
    plan: FrozenLocalQ1BatchPlan, contract: LocalQ1RunnerContract
) -> FrozenLocalQ1BatchPlan:
    if contract.production and not isinstance(plan, FrozenLocalQ1BatchPlan):
        raise TypeError("plan must be FrozenLocalQ1BatchPlan")
    if not all(
        hasattr(plan, name)
        for name in ("receipt", "content_sha256", "canonical_file_sha256")
    ):
        raise TypeError("plan lacks frozen receipt/hash commitments")
    if not hmac.compare_digest(
        plan.canonical_file_sha256, contract.batch_plan_file_sha256
    ) or not hmac.compare_digest(
        plan.content_sha256, contract.batch_plan_content_sha256
    ):
        raise LocalQ1RunnerError("batch plan differs from preregistered locks")
    if contract.production:
        try:
            plan = FrozenLocalQ1BatchPlan(
                receipt=plan.receipt,
                content_sha256=plan.content_sha256,
                canonical_file_sha256=plan.canonical_file_sha256,
            )
        except Exception as exc:
            raise LocalQ1RunnerError(
                "production batch plan failed independent reconstruction"
            ) from exc
    for phase in _PHASES:
        _plan_phase_receipt(plan, phase)
    assignment = plan.receipt.get("validation_assignment")
    if not isinstance(assignment, Mapping):
        raise LocalQ1RunnerError("batch plan lacks validation partition authority")
    if (
        assignment.get("status")
        != "frozen_label_blind_component_partition_nonempty_disjoint_exhaustive"
        or assignment.get("phase_order") != list(_VALIDATION_PHASES)
        or assignment.get("fields_read") != ["split", "component_id"]
        or assignment.get("supervision_fields_read") != []
    ):
        raise LocalQ1RunnerError(
            "validation partition is not label-blind/component-disjoint/complete"
        )
    phases = assignment.get("phases")
    proof = assignment.get("partition_proof")
    if not isinstance(phases, Mapping) or set(phases) != set(_VALIDATION_PHASES):
        raise LocalQ1RunnerError("validation partition phase set changed")
    if proof != {
        "unit": "component_id",
        "all_phases_nonempty": True,
        "component_overlap_count": 0,
        "record_overlap_count": 0,
        "component_assignment_exhaustive": True,
        "record_assignment_exhaustive": True,
        "labels_or_directions_read": False,
    }:
        raise LocalQ1RunnerError("validation partition proof changed")
    return plan


@dataclass(frozen=True)
class _ProviderSafetyPhaseAuthority:
    """One phase's supervision-blind provider safety preflight."""

    phase: str
    batches: Tuple[FrozenLocalQ1PlannedSafetyMetadata, ...]
    records: Tuple[LocalQ1PlannedSafetyRecord, ...]
    population_ordinals: Tuple[int, ...]
    component_tokens: frozenset
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class _ProviderPhaseAuthority:
    """One phase, acquired exactly once from provider-owned plan batches."""

    phase: str
    batches: Tuple[Tuple[TrainingPairRecord, ...], ...]
    records: Tuple[TrainingPairRecord, ...]
    population_ordinals: Tuple[int, ...]
    component_ids: frozenset
    pair_ids: frozenset
    evidence: Mapping[str, Any]


@dataclass
class _PhaseAccessLedger:
    planned_safety_metadata_read_count: Dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in _PHASES}
    )
    planned_safety_metadata_batch_call_count: Dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in _PHASES}
    )
    planned_records_read_count: Dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in _PHASES}
    )
    planned_records_batch_call_count: Dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in _PHASES}
    )
    prepare_call_count: Dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in _PHASES}
    )
    predict_call_count: Dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in _PHASES}
    )
    predict_pass_count: Dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in _PHASES}
    )
    events: List[str] = field(default_factory=list)

    def add_event(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise LocalQ1RunnerError("execution event must be a nonempty string")
        self.events.append(value)

    def portable_dict(self) -> Mapping[str, Any]:
        return {
            "planned_safety_metadata_read_count": dict(
                self.planned_safety_metadata_read_count
            ),
            "planned_safety_metadata_batch_call_count": dict(
                self.planned_safety_metadata_batch_call_count
            ),
            "planned_records_read_count": dict(self.planned_records_read_count),
            "planned_records_batch_call_count": dict(
                self.planned_records_batch_call_count
            ),
            "prepare_call_count": dict(self.prepare_call_count),
            "predict_call_count": dict(self.predict_call_count),
            "predict_pass_count": dict(self.predict_pass_count),
            "event_log": list(self.events),
            "event_log_sha256": _content_sha256(self.events),
        }


def _component_token(component_id: str) -> str:
    if not isinstance(component_id, str) or not component_id.strip():
        raise LocalQ1RunnerError("component identity is missing")
    return hashlib.sha256(
        LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE.encode("utf-8")
        + b"\0"
        + component_id.encode("utf-8")
    ).hexdigest()


def _load_provider_safety_phase(
    provider: _RunnerProvider,
    plan: FrozenLocalQ1BatchPlan,
    *,
    phase: str,
    production: bool,
    access: _PhaseAccessLedger,
) -> _ProviderSafetyPhaseAuthority:
    """Acquire one full phase's supervision-blind safety view pre-session."""

    if access.planned_safety_metadata_read_count[phase] != 0:
        raise LocalQ1RunnerError("provider safety phase was acquired more than once")
    access.planned_safety_metadata_read_count[phase] = 1
    access.add_event("planned_safety_metadata_read:" + phase)
    receipt = _plan_phase_receipt(plan, phase)
    expected_split = "train" if phase == _TRAIN_PHASE else "val"
    population_count = (
        int(receipt["record_count"])
        if phase == _TRAIN_PHASE
        else sum(
            int(_plan_phase_receipt(plan, item)["record_count"])
            for item in _VALIDATION_PHASES
        )
    )
    assignment = plan.receipt.get("validation_assignment")
    if not isinstance(assignment, Mapping):
        raise LocalQ1RunnerError("batch plan lacks validation assignment")
    assignment_sha256 = _require_sha256(
        assignment.get("assignment_sha256"),
        "validation assignment SHA-256",
    )
    batches: List[FrozenLocalQ1PlannedSafetyMetadata] = []
    rows: List[LocalQ1PlannedSafetyRecord] = []
    population_ordinals: List[int] = []
    phase_component_commitment: Optional[str] = None
    for entry in receipt["batches"]:
        ordinal = entry.get("ordinal")
        if type(ordinal) is not int or ordinal != len(batches):  # noqa: E721
            raise LocalQ1RunnerError("batch-plan safety ordinal changed")
        try:
            metadata = (
                LocalQ1ReadOnlyBatchProvider.planned_safety_metadata(
                    provider, phase, ordinal
                )
                if production
                else provider.planned_safety_metadata(phase, ordinal)
            )
        except Exception as exc:
            raise LocalQ1RunnerError(
                "provider planned safety metadata rejected {}: {}".format(phase, exc)
            ) from exc
        access.planned_safety_metadata_batch_call_count[phase] += 1
        if type(metadata) is not FrozenLocalQ1PlannedSafetyMetadata:  # noqa: E721
            raise LocalQ1RunnerError(
                "provider returned untyped planned safety metadata"
            )
        try:
            metadata = FrozenLocalQ1PlannedSafetyMetadata(
                receipt=metadata.receipt,
                content_sha256=metadata.content_sha256,
            )
        except Exception as exc:
            raise LocalQ1RunnerError(
                "provider planned safety metadata failed reconstruction"
            ) from exc
        value = metadata.receipt
        record_ordinals = entry.get("record_ordinals")
        batch_geometry_sha256 = entry.get("batch_geometry_sha256")
        phase_population_ordinal_set_sha256 = receipt.get(
            "population_ordinal_set_sha256"
        )
        if (
            value["phase"] != phase
            or value["batch_ordinal"] != ordinal
            or value["expected_split"] != expected_split
            or value["batch_plan_file_sha256"] != plan.canonical_file_sha256
            or value["batch_plan_content_sha256"] != plan.content_sha256
            or value["validation_assignment_sha256"] != assignment_sha256
            or value["plan_batch_geometry_sha256"] != batch_geometry_sha256
            or value["plan_phase_population_ordinal_set_sha256"]
            != phase_population_ordinal_set_sha256
            or not isinstance(record_ordinals, list)
            or tuple(row.population_ordinal for row in metadata.records)
            != tuple(record_ordinals)
        ):
            raise LocalQ1RunnerError(
                "provider safety metadata differs from frozen batch plan"
            )
        observed_phase_commitment = value["phase_component_set_sha256"]
        if phase_component_commitment is None:
            phase_component_commitment = observed_phase_commitment
        elif not hmac.compare_digest(
            phase_component_commitment, observed_phase_commitment
        ):
            raise LocalQ1RunnerError(
                "provider safety phase component commitment changed across batches"
            )
        for row in metadata.records:
            if (
                row.expected_split != expected_split
                or row.population_ordinal < 0
                or row.population_ordinal >= population_count
                or row.population_ordinal in population_ordinals
            ):
                raise LocalQ1RunnerError(
                    "provider safety metadata ordinals/splits are not exact"
                )
            if (
                not row.sealed_real_test_marker_explicit_false
                or row.sealed_scope_marker
            ):
                raise LocalQ1RunnerError(
                    "provider safety metadata admits sealed or missing sealed marker"
                )
            if row.historical_test_marker:
                raise LocalQ1RunnerError(
                    "provider safety metadata admits historical-test provenance"
                )
        batches.append(metadata)
        rows.extend(metadata.records)
        population_ordinals.extend(record_ordinals)
    if len(rows) != receipt["record_count"]:
        raise LocalQ1RunnerError("provider safety phase cardinality changed")
    ordinal_tuple = tuple(population_ordinals)
    if phase == _TRAIN_PHASE and set(ordinal_tuple) != set(range(population_count)):
        raise LocalQ1RunnerError("provider safety train ordinals are not exhaustive")
    component_tokens = frozenset(row.component_token_sha256 for row in rows)
    component_set_sha256 = _content_sha256(sorted(component_tokens))
    if phase_component_commitment is None or not hmac.compare_digest(
        phase_component_commitment, component_set_sha256
    ):
        raise LocalQ1RunnerError(
            "provider safety phase component population is incomplete"
        )
    evidence: Dict[str, Any] = {
        "phase": phase,
        "batch_count": len(batches),
        "record_count": len(rows),
        "component_count": len(component_tokens),
        "metadata_content_sequence_sha256": _content_sha256(
            [metadata.content_sha256 for metadata in batches]
        ),
        "population_ordinal_sequence_sha256": _content_sha256(list(ordinal_tuple)),
        "component_set_sha256": component_set_sha256,
        "sealed_or_missing_marker_count": 0,
        "historical_test_marker_count": 0,
        "split_violation_count": 0,
    }
    evidence["content_sha256"] = _self_content_sha256(evidence)
    return _ProviderSafetyPhaseAuthority(
        phase=phase,
        batches=tuple(batches),
        records=tuple(rows),
        population_ordinals=ordinal_tuple,
        component_tokens=component_tokens,
        evidence=MappingProxyType(evidence),
    )


def _provider_safety_authority(
    provider: _RunnerProvider,
    plan: FrozenLocalQ1BatchPlan,
    *,
    production: bool,
    access: _PhaseAccessLedger,
) -> Tuple[Mapping[str, _ProviderSafetyPhaseAuthority], Mapping[str, Any]]:
    phases: Dict[str, _ProviderSafetyPhaseAuthority] = {}
    for phase in _PHASES:
        phases[phase] = _load_provider_safety_phase(
            provider,
            plan,
            phase=phase,
            production=production,
            access=access,
        )
    validation_count = sum(
        len(phases[phase].population_ordinals) for phase in _VALIDATION_PHASES
    )
    validation_ordinals = {
        ordinal
        for phase in _VALIDATION_PHASES
        for ordinal in phases[phase].population_ordinals
    }
    if validation_ordinals != set(range(validation_count)):
        raise LocalQ1RunnerError(
            "provider safety validation ordinals are not disjoint and exhaustive"
        )
    train_components = phases[_TRAIN_PHASE].component_tokens
    validation_components = set().union(
        *(phases[phase].component_tokens for phase in _VALIDATION_PHASES)
    )
    if train_components.intersection(validation_components):
        raise LocalQ1RunnerError("provider safety train/validation components overlap")
    for index, phase in enumerate(_VALIDATION_PHASES):
        for other in _VALIDATION_PHASES[index + 1 :]:
            if phases[phase].component_tokens.intersection(
                phases[other].component_tokens
            ):
                raise LocalQ1RunnerError(
                    "provider safety validation phases share a component"
                )
    read_order = [
        event.split(":", 1)[1]
        for event in access.events
        if event.startswith("planned_safety_metadata_read:")
    ]
    if read_order != list(_PHASES):
        raise LocalQ1RunnerError("provider safety phases were not read in order")
    value: Dict[str, Any] = {
        "authority": (
            "provider.planned_safety_metadata_exact_frozen_phase_batches_"
            "supervision_blind_pre_session"
        ),
        "metadata_schema_version": (LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION),
        "provider_version": LOCAL_Q1_BATCH_PROVIDER_VERSION,
        "provenance_safety_scan": _portable(dict(local_q1_provenance_safety_policy())),
        "phase_read_order": read_order,
        "planned_safety_metadata_read_count": dict(
            access.planned_safety_metadata_read_count
        ),
        "planned_safety_metadata_batch_call_count": dict(
            access.planned_safety_metadata_batch_call_count
        ),
        "provider_safety_record_count": sum(
            len(phases[phase].records) for phase in _PHASES
        ),
        "phases": {phase: dict(phases[phase].evidence) for phase in _PHASES},
        "supervision_fields_read": [],
        "raw_component_ids_present": False,
        "sealed_real_test_accessed": False,
        "historical_test_accessed": False,
        "train_validation_component_overlap_count": 0,
        "validation_component_overlap_count": 0,
        "validation_record_ordinal_overlap_count": 0,
        "validation_record_ordinals_exhaustive": True,
        "completed_before_backend_session_or_output": True,
    }
    value["content_sha256"] = _self_content_sha256(value)
    return MappingProxyType(phases), MappingProxyType(value)


def _provider_scope_attestation(
    provider: _RunnerProvider, *, production: bool
) -> Mapping[str, Any]:
    if production:
        opened = getattr(provider, "_opened", None)
        build_receipt = getattr(opened, "build_receipt", None)
        attested_scope = (
            build_receipt.get("scope") if isinstance(build_receipt, Mapping) else None
        )
        expected_attested_scope = {
            "experiment": "LOCAL-Q1",
            "input_modality": "canonical_bool_mask_only",
            "splits": ["train", "val"],
            "sealed_real_capability": False,
            "historical_test_capability": False,
            "model_backend_imported": False,
            "model_executed": False,
        }
        if attested_scope != expected_attested_scope:
            raise LocalQ1RunnerError(
                "provider cache-build test/sealed scope attestation changed"
            )
        value: Dict[str, Any] = {
            "kind": "externally_verified_provider_cache_build_scope",
            "source_scope_sha256": _content_sha256(attested_scope),
            "sealed_real_test_accessed": attested_scope["sealed_real_capability"],
            "historical_test_accessed": attested_scope["historical_test_capability"],
            "result_eligible": True,
        }
    else:
        value = {
            "kind": "fixture_non_result_no_external_scope_authority",
            "source_scope_sha256": _content_sha256(
                {"fixture_non_result": True, "external_scope_authority": False}
            ),
            "sealed_real_test_accessed": False,
            "historical_test_accessed": False,
            "result_eligible": False,
        }
    value["content_sha256"] = _self_content_sha256(value)
    return MappingProxyType(value)


def _phase_scope_evidence(
    *,
    phase: str,
    records: Tuple[TrainingPairRecord, ...],
    population_ordinals: Tuple[int, ...],
) -> Mapping[str, Any]:
    expected_split = "train" if phase == _TRAIN_PHASE else "val"
    sealed_markers = 0
    historical_test_markers = 0
    split_violations = 0
    for record in records:
        try:
            provenance_markers = scan_local_q1_provenance_safety(record.provenance)
        except Exception as exc:
            raise LocalQ1RunnerError(
                "provider planned phase provenance safety scan failed"
            ) from exc
        sealed_markers += int(
            not provenance_markers.top_level_sealed_marker_explicit_false
            or provenance_markers.sealed_scope_marker
        )
        historical_test_markers += int(provenance_markers.historical_test_marker)
        if (
            record.split != expected_split
            or record.fragment_a.split != expected_split
            or record.fragment_b.split != expected_split
        ):
            split_violations += 1
    if sealed_markers:
        raise LocalQ1RunnerError(
            "provider planned phase has sealed or missing sealed-test provenance"
        )
    if historical_test_markers:
        raise LocalQ1RunnerError(
            "provider planned phase has historical-test provenance"
        )
    if split_violations:
        raise LocalQ1RunnerError("provider planned phase crosses train/val splits")
    pair_ids = tuple(record.pair_id for record in records)
    if len(set(pair_ids)) != len(pair_ids):
        raise LocalQ1RunnerError("provider planned phase repeats a pair identity")
    component_tokens = sorted(
        {_component_token(record.component_id) for record in records}
    )
    evidence: Dict[str, Any] = {
        "phase": phase,
        "record_count": len(records),
        "component_count": len(component_tokens),
        "record_sequence_sha256": record_sequence_fingerprint(records),
        "population_ordinal_sequence_sha256": _content_sha256(
            list(population_ordinals)
        ),
        "component_set_sha256": _content_sha256(component_tokens),
        "sealed_or_missing_marker_count": sealed_markers,
        "historical_test_marker_count": historical_test_markers,
        "split_violation_count": split_violations,
        "sealed_real_test_accessed": False,
        "historical_test_accessed": False,
    }
    evidence["content_sha256"] = _self_content_sha256(evidence)
    return MappingProxyType(evidence)


def _load_provider_phase(
    provider: _RunnerProvider,
    plan: FrozenLocalQ1BatchPlan,
    *,
    phase: str,
    safety: _ProviderSafetyPhaseAuthority,
    production: bool,
    access: _PhaseAccessLedger,
) -> _ProviderPhaseAuthority:
    if safety.phase != phase:
        raise LocalQ1RunnerError("provider safety/full phase identity changed")
    if access.planned_records_read_count[phase] != 0:
        raise LocalQ1RunnerError("provider phase was acquired more than once")
    access.planned_records_read_count[phase] = 1
    access.add_event("planned_records_read:" + phase)
    receipt = _plan_phase_receipt(plan, phase)
    if phase == _TRAIN_PHASE:
        population_count = int(receipt["record_count"])
    else:
        population_count = sum(
            int(_plan_phase_receipt(plan, validation_phase)["record_count"])
            for validation_phase in _VALIDATION_PHASES
        )
    expected_split = "train" if phase == _TRAIN_PHASE else "val"
    batches: List[Tuple[TrainingPairRecord, ...]] = []
    rows: List[TrainingPairRecord] = []
    population_ordinals: List[int] = []
    for entry in receipt["batches"]:
        ordinal = entry.get("ordinal")
        if type(ordinal) is not int or ordinal != len(batches):  # noqa: E721
            raise LocalQ1RunnerError("batch-plan ordinal changed")
        records = tuple(
            LocalQ1ReadOnlyBatchProvider.planned_records(provider, phase, ordinal)
            if production
            else provider.planned_records(phase, ordinal)
        )
        access.planned_records_batch_call_count[phase] += 1
        record_ordinals = entry.get("record_ordinals")
        if (
            not records
            or not isinstance(record_ordinals, list)
            or len(records) != len(record_ordinals)
            or any(
                not isinstance(record, TrainingPairRecord)
                or record.split != expected_split
                for record in records
            )
        ):
            raise LocalQ1RunnerError(
                "provider planned batch differs from frozen phase contract"
            )
        for population_ordinal in record_ordinals:
            if (
                type(population_ordinal) is not int  # noqa: E721
                or population_ordinal < 0
                or population_ordinal >= population_count
                or population_ordinal in population_ordinals
            ):
                raise LocalQ1RunnerError(
                    "provider planned phase ordinals are not exact"
                )
        batches.append(records)
        rows.extend(records)
        population_ordinals.extend(record_ordinals)
    if len(rows) != receipt["record_count"]:
        raise LocalQ1RunnerError("provider planned phase cardinality changed")
    if phase == _TRAIN_PHASE and set(population_ordinals) != set(
        range(population_count)
    ):
        raise LocalQ1RunnerError("provider planned train ordinals are not exhaustive")
    row_tuple = tuple(rows)
    ordinal_tuple = tuple(population_ordinals)
    if ordinal_tuple != safety.population_ordinals or len(row_tuple) != len(
        safety.records
    ):
        raise LocalQ1RunnerError(
            "provider full phase differs from safety metadata ordinals"
        )
    for record, safety_row in zip(row_tuple, safety.records):
        if record.split != safety_row.expected_split or not hmac.compare_digest(
            _component_token(record.component_id),
            safety_row.component_token_sha256,
        ):
            raise LocalQ1RunnerError(
                "provider full phase differs from pre-session safety metadata"
            )
    evidence = _phase_scope_evidence(
        phase=phase,
        records=row_tuple,
        population_ordinals=ordinal_tuple,
    )
    for name in (
        "record_count",
        "component_count",
        "population_ordinal_sequence_sha256",
        "component_set_sha256",
        "sealed_or_missing_marker_count",
        "historical_test_marker_count",
        "split_violation_count",
    ):
        if evidence[name] != safety.evidence[name]:
            raise LocalQ1RunnerError(
                "provider full phase scope differs from pre-session safety metadata"
            )
    return _ProviderPhaseAuthority(
        phase=phase,
        batches=tuple(batches),
        records=row_tuple,
        population_ordinals=ordinal_tuple,
        component_ids=frozenset(record.component_id for record in row_tuple),
        pair_ids=frozenset(record.pair_id for record in row_tuple),
        evidence=evidence,
    )


def _require_phase_compatible(
    new: _ProviderPhaseAuthority,
    loaded: Mapping[str, _ProviderPhaseAuthority],
) -> None:
    for previous in loaded.values():
        if new.phase != _TRAIN_PHASE and previous.phase != _TRAIN_PHASE:
            if set(new.population_ordinals).intersection(previous.population_ordinals):
                raise LocalQ1RunnerError(
                    "provider validation phases share population ordinals"
                )
        if new.component_ids.intersection(previous.component_ids):
            raise LocalQ1RunnerError(
                "provider planned train/validation components overlap"
                if _TRAIN_PHASE in {new.phase, previous.phase}
                else "provider validation phases share a component"
            )
        if new.pair_ids.intersection(previous.pair_ids):
            raise LocalQ1RunnerError("provider planned phases share a pair identity")


def _audit_external_phase(
    supplied: Sequence[TrainingPairRecord],
    authority: _ProviderPhaseAuthority,
    *,
    name: str,
    expected_population_count: int,
) -> None:
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise TypeError("external {} population must be a finite sequence".format(name))
    if len(supplied) != expected_population_count:
        raise LocalQ1RunnerError(
            "external {} population count differs from provider authority".format(name)
        )
    observed: List[TrainingPairRecord] = []
    for population_ordinal, expected in zip(
        authority.population_ordinals, authority.records
    ):
        candidate = supplied[population_ordinal]
        if not isinstance(candidate, TrainingPairRecord):
            raise TypeError("external population contains a non-record value")
        for dataclass_field in fields(TrainingPairRecord):
            if getattr(candidate, dataclass_field.name) != getattr(
                expected, dataclass_field.name
            ):
                raise LocalQ1RunnerError(
                    "external {} population differs from provider authority at "
                    "record {} field {}".format(
                        name, population_ordinal, dataclass_field.name
                    )
                )
        observed.append(candidate)
    if not hmac.compare_digest(
        record_sequence_fingerprint(observed),
        record_sequence_fingerprint(authority.records),
    ):
        raise LocalQ1RunnerError(
            "external {} phase order differs from provider authority".format(name)
        )


def _final_phase_authority(
    *,
    attestation: Mapping[str, Any],
    safety_authority: Mapping[str, Any],
    phases: Mapping[str, _ProviderPhaseAuthority],
    access: _PhaseAccessLedger,
) -> Mapping[str, Any]:
    if set(phases) != set(_PHASES):
        raise LocalQ1RunnerError("provider phase authority is incomplete")
    validation_count = sum(len(phases[phase].records) for phase in _VALIDATION_PHASES)
    validation_ordinals = {
        ordinal
        for phase in _VALIDATION_PHASES
        for ordinal in phases[phase].population_ordinals
    }
    if validation_ordinals != set(range(validation_count)):
        raise LocalQ1RunnerError(
            "provider validation phase ordinals are not disjoint and exhaustive"
        )
    if any(access.planned_records_read_count[phase] != 1 for phase in _PHASES):
        raise LocalQ1RunnerError("each provider phase must be acquired exactly once")
    phase_read_order = [
        event.split(":", 1)[1]
        for event in access.events
        if event.startswith("planned_records_read:")
    ]
    if phase_read_order != list(_PHASES):
        raise LocalQ1RunnerError("provider phases were not lazily opened in order")
    value: Dict[str, Any] = {
        "authority": "provider.planned_records_lazy_exact_frozen_phase_batches",
        "provider_scope_attestation_kind": attestation["kind"],
        "provider_scope_attestation_sha256": attestation["content_sha256"],
        "provider_safety_metadata_authority": dict(safety_authority),
        "provider_safety_metadata_authority_sha256": safety_authority["content_sha256"],
        "provider_planned_record_count": sum(
            len(phases[phase].records) for phase in _PHASES
        ),
        "phase_read_order": phase_read_order,
        "planned_records_read_count": dict(access.planned_records_read_count),
        "planned_records_batch_call_count": dict(
            access.planned_records_batch_call_count
        ),
        "phases": {phase: dict(phases[phase].evidence) for phase in _PHASES},
        "sealed_real_test_accessed": bool(
            attestation["sealed_real_test_accessed"]
            or any(
                phases[phase].evidence["sealed_real_test_accessed"] for phase in _PHASES
            )
        ),
        "historical_test_accessed": bool(
            attestation["historical_test_accessed"]
            or any(
                phases[phase].evidence["historical_test_accessed"] for phase in _PHASES
            )
        ),
        "train_validation_component_overlap_count": 0,
        "validation_component_overlap_count": 0,
        "validation_record_ordinal_overlap_count": 0,
        "validation_record_ordinals_exhaustive": True,
    }
    if value["sealed_real_test_accessed"] or value["historical_test_accessed"]:
        raise LocalQ1RunnerError("provider phase authority admits test records")
    value["content_sha256"] = _self_content_sha256(value)
    return MappingProxyType(value)


def _phase_access_receipt(access: _PhaseAccessLedger) -> Mapping[str, Any]:
    events = access.events

    def position(value: str) -> int:
        try:
            return events.index(value)
        except ValueError as exc:
            raise LocalQ1RunnerError(
                "execution event log is missing {}".format(value)
            ) from exc

    backend_position = position("backend_created")
    calibration_position = position("planned_records_read:" + _CALIBRATION_PHASE)
    report_position = position("planned_records_read:" + _REPORT_PHASE)
    all_winners_position = position("all_winners_selection_reproduced")
    all_thresholds_position = position("all_thresholds_frozen")
    session_positions = [
        index
        for index, value in enumerate(events)
        if value.startswith("training_session_created:")
        or value.startswith("winner_reload_session_created:")
    ]
    epoch_positions = [
        index
        for index, value in enumerate(events)
        if value.startswith("epoch_selection_complete:")
    ]
    winner_positions = [
        index
        for index, value in enumerate(events)
        if value.startswith("winner_selection_reproduced:")
    ]
    threshold_positions = [
        index
        for index, value in enumerate(events)
        if value.startswith("threshold_frozen:")
    ]
    safety_positions = [
        position("planned_safety_metadata_read:" + phase) for phase in _PHASES
    ]
    if (
        not session_positions
        or not epoch_positions
        or len(winner_positions) != len(_LOCAL_ARMS)
        or len(threshold_positions) != len(_LOCAL_ARMS)
        or any(
            access.planned_safety_metadata_read_count[phase] != 1 for phase in _PHASES
        )
        or safety_positions != sorted(safety_positions)
        or max(safety_positions) >= position("planned_records_read:" + _TRAIN_PHASE)
        or position("planned_records_read:" + _TRAIN_PHASE) >= backend_position
        or position("planned_records_read:" + _SELECT_PHASE) >= backend_position
        or backend_position >= min(session_positions)
        or max(epoch_positions) >= all_winners_position
        or max(winner_positions) >= calibration_position
        or all_winners_position >= calibration_position
        or max(threshold_positions) >= report_position
        or all_thresholds_position >= report_position
    ):
        raise LocalQ1RunnerError("lazy phase execution event order changed")
    value = dict(access.portable_dict())
    value.update(
        {
            "count_semantics": {
                "planned_safety_metadata_read_count": (
                    "phase_level_supervision_blind_preflight_lookup_count"
                ),
                "planned_safety_metadata_batch_call_count": (
                    "provider_planned_safety_metadata_calls_one_per_frozen_batch"
                ),
                "planned_records_read_count": ("phase_level_population_lookup_count"),
                "planned_records_batch_call_count": (
                    "provider_planned_records_method_calls_one_per_frozen_batch"
                ),
                "prepare_call_count": "provider_prepare_method_calls",
                "predict_call_count": "backend_predict_batch_method_calls",
                "predict_pass_count": "complete_phase_prediction_passes",
            },
            "ordering_proof": {
                "all_phase_safety_metadata_before_backend": True,
                "safety_metadata_supervision_fields_read": [],
                "train_and_select_lookup_before_backend": True,
                "no_calibration_lookup_before_all_winner_replays": True,
                "no_report_lookup_before_all_thresholds_frozen": True,
                "calibration_population_lookup_count": 1,
                "report_population_lookup_count": 1,
                "calibration_prediction_pass_count_per_arm": 1,
                "report_prediction_pass_count_per_arm": 1,
            },
        }
    )
    if (
        access.planned_records_read_count[_CALIBRATION_PHASE] != 1
        or access.planned_records_read_count[_REPORT_PHASE] != 1
        or access.predict_pass_count[_CALIBRATION_PHASE] != len(_LOCAL_ARMS)
        or access.predict_pass_count[_REPORT_PHASE] != len(_LOCAL_ARMS)
    ):
        raise LocalQ1RunnerError("lazy calibration/report access counts changed")
    value["content_sha256"] = _self_content_sha256(value)
    return MappingProxyType(value)


def _prepare_checked(
    provider: _RunnerProvider,
    records: Sequence[TrainingPairRecord],
    *,
    arm: AblationArm,
    phase: str,
    shared_prepared: Dict[Tuple[str, int, str], Tuple[str, str, str]],
    batch_ordinal: int,
    access: _PhaseAccessLedger,
) -> PreparedAblationBatch:
    prepared = provider.prepare(records, arm=arm, phase=phase)
    access.prepare_call_count[phase] += 1
    if not isinstance(prepared, PreparedAblationBatch):
        raise LocalQ1RunnerError("provider returned an invalid prepared batch")
    if prepared.sample_count != len(records):
        raise LocalQ1RunnerError("provider changed batch cardinality")
    expected_sequence = record_sequence_fingerprint(records)
    if not hmac.compare_digest(prepared.record_sequence_sha256, expected_sequence):
        raise LocalQ1RunnerError("provider changed the frozen record sequence")
    if prepared.local_candidate_sha256 is None:
        raise LocalQ1RunnerError("LOCAL-Q1 provider omitted candidate authority")
    if prepared.candidate_representation != arm.candidate_representation:
        raise LocalQ1RunnerError(
            "provider candidate representation differs from local arm"
        )
    if not hmac.compare_digest(
        prepared.coarse_preprocessing_sha256,
        provider.contract.coarse_preprocessing_sha256,
    ) or not hmac.compare_digest(
        str(prepared.geometry_config_sha256),
        provider.contract.geometry_config_sha256,
    ):
        raise LocalQ1RunnerError("prepared preprocessing/geometry contract changed")
    identity = (
        prepared.record_sequence_sha256,
        prepared.prepared_input_sha256,
        prepared.local_candidate_sha256,
    )
    key = (phase, batch_ordinal, prepared.candidate_representation)
    previous = shared_prepared.setdefault(key, identity)
    if previous != identity:
        raise LocalQ1RunnerError(
            "prepared inputs/candidates differ within representation"
        )
    return prepared


def _prediction_commitment(
    probabilities: Sequence[float], valid: Sequence[bool]
) -> str:
    payload = [
        {"probability": float(probability) if is_valid else None, "valid": is_valid}
        for probability, is_valid in zip(probabilities, valid)
    ]
    return _content_sha256(payload)


def _dataset_evaluation(
    records: Sequence[TrainingPairRecord],
    probabilities: Sequence[float],
    valid: Sequence[bool],
    *,
    threshold: float,
) -> Mapping[str, Any]:
    labels = [record.label for record in records]
    clusters = [record.component_id for record in records]
    overall = evaluate_pairwise(
        probabilities, labels, valid, clusters, threshold=threshold
    )
    by_dataset: Dict[str, Any] = {}
    for dataset in sorted({record.dataset_id for record in records}):
        indices = [
            index
            for index, record in enumerate(records)
            if record.dataset_id == dataset
        ]
        by_dataset[dataset] = evaluate_pairwise(
            [probabilities[index] for index in indices],
            [labels[index] for index in indices],
            [valid[index] for index in indices],
            [clusters[index] for index in indices],
            threshold=threshold,
        )
    macro = {
        metric: float(
            np.mean(
                [
                    float(value["cluster_balanced"][metric])
                    for value in by_dataset.values()
                ]
            )
        )
        for metric in ("auroc", "auprc", "brier", "ece")
    }
    coverage = {
        "record_count": len(records),
        "valid_count": sum(bool(value) for value in valid),
        "valid_fraction": sum(bool(value) for value in valid) / len(records),
        "by_dataset": {
            dataset: {
                "record_count": sum(record.dataset_id == dataset for record in records),
                "valid_count": sum(
                    bool(is_valid) and record.dataset_id == dataset
                    for record, is_valid in zip(records, valid)
                ),
            }
            for dataset in sorted(by_dataset)
        },
    }
    return {
        "threshold": threshold,
        "coverage": coverage,
        "overall": overall,
        "by_dataset": by_dataset,
        "equal_dataset_macro_cluster": macro,
        "prediction_commitment_sha256": _prediction_commitment(probabilities, valid),
        "record_sequence_sha256": record_sequence_fingerprint(records),
    }


def _predict_phase(
    *,
    provider: _RunnerProvider,
    session: _RunnerSession,
    arm: AblationArm,
    planned_batches: Tuple[Tuple[TrainingPairRecord, ...], ...],
    phase_receipt: Mapping[str, Any],
    phase: str,
    shared_prepared: Dict[Tuple[str, int, str], Tuple[str, str, str]],
    threshold: float,
    min_valid_fraction: float,
    access: _PhaseAccessLedger,
) -> Tuple[Mapping[str, Any], Tuple[float, ...], Tuple[bool, ...]]:
    access.predict_pass_count[phase] += 1
    records: List[TrainingPairRecord] = []
    probabilities: List[float] = []
    validity: List[bool] = []
    diagnostic_hashes: List[str] = []
    if len(planned_batches) != len(phase_receipt["batches"]):
        raise LocalQ1RunnerError("provider planned validation batch count changed")
    for entry, batch_records in zip(phase_receipt["batches"], planned_batches):
        batch_ordinal = int(entry["ordinal"])
        prepared = _prepare_checked(
            provider,
            batch_records,
            arm=arm,
            phase=phase,
            shared_prepared=shared_prepared,
            batch_ordinal=batch_ordinal,
            access=access,
        )
        prediction = session.predict_batch(prepared, evidence=EvidenceMode.LOCAL)
        access.predict_call_count[phase] += 1
        if not isinstance(prediction, PredictionBatch):
            raise LocalQ1RunnerError("backend returned an invalid prediction batch")
        if prediction.probability.shape[0] != len(batch_records):
            raise LocalQ1RunnerError("prediction cardinality changed")
        records.extend(batch_records)
        probabilities.extend(
            float(value) for value in prediction.probability.detach().cpu().tolist()
        )
        validity.extend(
            bool(value) for value in prediction.valid.detach().cpu().tolist()
        )
        diagnostic_hashes.append(_content_sha256(dict(prediction.diagnostics)))
    if len(records) != phase_receipt["record_count"]:
        raise LocalQ1RunnerError("validation phase cardinality changed")
    valid_fraction = sum(validity) / len(validity)
    if valid_fraction < min_valid_fraction:
        raise LocalQ1RunnerError(
            "{} valid coverage is below the preregistered minimum".format(phase)
        )
    report = dict(
        _dataset_evaluation(records, probabilities, validity, threshold=threshold)
    )
    report["phase"] = phase
    report["batch_count"] = len(phase_receipt["batches"])
    report["diagnostic_sequence_sha256"] = _content_sha256(diagnostic_hashes)
    return report, tuple(probabilities), tuple(validity)


def _prediction_replay_projection(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the deterministic part of one prediction-phase report.

    Backend diagnostics intentionally include execution telemetry such as elapsed
    time and device-memory deltas.  A fresh checkpoint session cannot reproduce
    that telemetry, even when every prediction bit is identical.  The remaining
    report binds the raw probability/validity sequence, record sequence, coverage,
    metrics, threshold, phase, and batch count and is therefore the appropriate
    equality boundary for winner replay.
    """

    if "diagnostic_sequence_sha256" not in report:
        raise LocalQ1RunnerError(
            "prediction report omitted its diagnostic sequence commitment"
        )
    return MappingProxyType(
        {
            str(key): value
            for key, value in report.items()
            if key != "diagnostic_sequence_sha256"
        }
    )


def _checkpoint_config(
    *,
    contract: LocalQ1RunnerContract,
    plan: FrozenLocalQ1BatchPlan,
    provider: _RunnerProvider,
    backend: _RunnerBackend,
    arm: AblationArm,
    run_fingerprint: str,
    scope_attestation: Mapping[str, Any],
    safety_authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = {
        "schema_version": LOCAL_Q1_CHECKPOINT_CONFIG_SCHEMA_VERSION,
        "model_config": _portable(arm.model_config),
        "optimizer_config": _portable(arm.optimizer_config),
        "aggregation_config": _portable(arm.aggregation_config),
        "arm": arm.to_dict(),
        "runner_contract": dict(contract.portable_dict()),
        "batch_plan": {
            "file_sha256": plan.canonical_file_sha256,
            "content_sha256": plan.content_sha256,
            "phase_commitment_sha256": _content_sha256(plan.receipt["phases"]),
            "validation_assignment_commitment_sha256": _content_sha256(
                plan.receipt["validation_assignment"]
            ),
        },
        "provider_contract": provider.contract.to_dict(),
        "backend_contract": backend.contract.to_dict(),
        "run_fingerprint_sha256": run_fingerprint,
        "scope": {
            "mask_only_known_orientation": True,
            "train_and_validation_only": True,
            "sealed_real_test_accessed": scope_attestation["sealed_real_test_accessed"],
            "historical_test_accessed": scope_attestation["historical_test_accessed"],
            "provider_scope_attestation_sha256": scope_attestation["content_sha256"],
            "provider_safety_metadata_authority_sha256": safety_authority[
                "content_sha256"
            ],
            "fused": False,
        },
    }
    _assert_portable(value)
    return MappingProxyType(value)


def _checkpoint_claim(receipt: CheckpointReceipt) -> Mapping[str, Any]:
    return {
        "epoch": receipt.epoch,
        "file_sha256": receipt.file_sha256,
        "canonical_content_sha256": receipt.canonical_content_sha256,
        "config_sha256": receipt.config_hash,
        "model_state_sha256": receipt.model_state_sha256,
        "optimizer_state_sha256": receipt.optimizer_state_sha256,
    }


def _authority_receipt(
    *,
    result_eligible: bool,
    arm: AblationArm,
    plan: FrozenLocalQ1BatchPlan,
    run_fingerprint: str,
    winner: _SavedCheckpoint,
    reload_selection: Mapping[str, Any],
    calibration: Mapping[str, Any],
    threshold: Mapping[str, Any],
    report: Mapping[str, Any],
    phase_authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    scope = CheckpointModelScope.PAIRWISE_MODEL
    projection = checkpoint_semantic_projection(winner.config, scope)
    value: Dict[str, Any] = {
        "schema_version": LOCAL_Q1_CHECKPOINT_AUTHORITY_SCHEMA_VERSION,
        "status": LOCAL_Q1_AUTHORITY_STATUS,
        "authority_kind": LOCAL_Q1_AUTHORITY_KIND,
        "result_eligible": result_eligible,
        "arm": arm.name.value,
        "model_scope": scope.value,
        "checkpoint": _checkpoint_claim(winner.receipt),
        "checkpoint_config": _portable(winner.config),
        "semantic_projection": _portable(projection),
        "semantic_projection_sha256": canonical_config_hash(projection),
        "provider_phase_authority": _portable(phase_authority),
        "run_binding": {
            "run_fingerprint_sha256": run_fingerprint,
            "batch_plan_file_sha256": plan.canonical_file_sha256,
            "batch_plan_content_sha256": plan.content_sha256,
        },
        "validation_authority": {
            "selection_policy": _SELECTION_POLICY,
            "winner_epoch": winner.receipt.epoch,
            "restricted_fresh_session_reload": True,
            "selection_replay_match": True,
            "selection_report_sha256": _content_sha256(reload_selection),
            "calibration_prediction_pass_count": 1,
            "calibration_report_sha256": _content_sha256(calibration),
            "threshold_content_sha256": _content_sha256(threshold),
            "report_prediction_pass_count": 1,
            "report_sha256": _content_sha256(report),
        },
        "scope": {
            "synthetic_train_validation_only": True,
            "sealed_real_test_accessed": phase_authority["sealed_real_test_accessed"],
            "historical_test_accessed": phase_authority["historical_test_accessed"],
            "provider_phase_authority_sha256": phase_authority["content_sha256"],
            "provider_safety_metadata_authority_sha256": phase_authority[
                "provider_safety_metadata_authority_sha256"
            ],
            "threshold_fit_phase": _CALIBRATION_PHASE,
            "final_metric_phase": _REPORT_PHASE,
        },
    }
    _assert_portable(value)
    value["content_sha256"] = _self_content_sha256(value)
    return MappingProxyType(value)


def _fit_calibration_threshold(
    *,
    records: Sequence[TrainingPairRecord],
    probabilities: Sequence[float],
    valid: Sequence[bool],
    arm: AblationArm,
    checkpoint_sha256: str,
) -> Mapping[str, Any]:
    artifact = fit_pairwise_threshold(
        probabilities,
        [record.label for record in records],
        valid,
        [record.component_id for record in records],
        source_split=_CALIBRATION_PHASE,
        validation_fingerprint_sha256=record_sequence_fingerprint(records),
        checkpoint_sha256=checkpoint_sha256,
        model_config_sha256=arm.model_config_sha256,
        aggregation_config_sha256=arm.aggregation_config_sha256,
    )
    return MappingProxyType(artifact.to_dict())


def _fresh_output_directory(root: Path, run_fingerprint: str) -> Tuple[Path, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / ("run-" + run_fingerprint[:16])
    if destination.exists():
        raise LocalQ1RunnerError("run destination already exists; overwrite forbidden")
    partial = Path(tempfile.mkdtemp(prefix=".partial-local-q1-", dir=str(root)))
    return partial, destination


def run_local_q1(
    *,
    contract: LocalQ1RunnerContract,
    plan: FrozenLocalQ1BatchPlan,
    train_records: Sequence[TrainingPairRecord],
    validation_records: Optional[Sequence[TrainingPairRecord]] = None,
    provider: _RunnerProvider,
    backend_factory: Callable[[], _RunnerBackend],
    output_root: Path,
) -> LocalQ1RunArtifacts:
    """Run four local arms with genuinely lazy validation-phase authority.

    All four phases first expose only typed, supervision-blind provider safety
    metadata.  That preflight must finish before backend construction, session
    creation, checkpoint writes, or output-directory creation.  Full
    calibration/report records remain unavailable until their legitimate
    winner/threshold gates.

    ``train_records`` remains a compatibility audit view.  In fixture mode,
    ``validation_records`` is audited one phase at a time only after that phase
    is legitimately opened.  Production ignores the external validation view:
    the exact provider and frozen plan are the only validation authorities.
    No caller-supplied record is executed.
    """

    if not isinstance(contract, LocalQ1RunnerContract):
        raise TypeError("contract must be LocalQ1RunnerContract")
    if not callable(backend_factory):
        raise TypeError("backend_factory must be callable")
    access = _PhaseAccessLedger()
    frozen_plan = _checked_plan(plan, contract)
    checked_provider = _checked_provider(
        provider, production=contract.production, plan=frozen_plan
    )
    scope_attestation = _provider_scope_attestation(
        checked_provider,
        production=contract.production,
    )
    safety_phases, safety_authority = _provider_safety_authority(
        checked_provider,
        frozen_plan,
        production=contract.production,
        access=access,
    )
    access.add_event("all_phase_safety_metadata_preflight_complete")
    phase_authorities: Dict[str, _ProviderPhaseAuthority] = {}
    train_authority = _load_provider_phase(
        checked_provider,
        frozen_plan,
        phase=_TRAIN_PHASE,
        safety=safety_phases[_TRAIN_PHASE],
        production=contract.production,
        access=access,
    )
    phase_authorities[_TRAIN_PHASE] = train_authority
    select_authority = _load_provider_phase(
        checked_provider,
        frozen_plan,
        phase=_SELECT_PHASE,
        safety=safety_phases[_SELECT_PHASE],
        production=contract.production,
        access=access,
    )
    _require_phase_compatible(select_authority, phase_authorities)
    phase_authorities[_SELECT_PHASE] = select_authority
    _audit_external_phase(
        train_records,
        train_authority,
        name="train",
        expected_population_count=int(
            _plan_phase_receipt(frozen_plan, _TRAIN_PHASE)["record_count"]
        ),
    )
    access.add_event("external_audit:" + _TRAIN_PHASE)
    validation_population_count = sum(
        int(_plan_phase_receipt(frozen_plan, phase)["record_count"])
        for phase in _VALIDATION_PHASES
    )
    if contract.production:
        access.add_event("external_validation_audit_omitted:provider_plan_is_authority")
    else:
        if validation_records is None:
            raise TypeError("fixture validation audit view is required")
        _audit_external_phase(
            validation_records,
            select_authority,
            name=_SELECT_PHASE,
            expected_population_count=validation_population_count,
        )
        access.add_event("external_audit:" + _SELECT_PHASE)

    backend = _checked_backend(backend_factory(), production=contract.production)
    access.add_event("backend_created")
    train_phase = _plan_phase_receipt(frozen_plan, _TRAIN_PHASE)
    arms = tuple(
        _arm_for(backend, name, production=contract.production) for name in _LOCAL_ARMS
    )
    run_binding = {
        "schema_version": LOCAL_Q1_RUNNER_SCHEMA_VERSION,
        "contract_sha256": contract.content_sha256,
        "batch_plan_file_sha256": frozen_plan.canonical_file_sha256,
        "batch_plan_content_sha256": frozen_plan.content_sha256,
        "provider_contract_sha256": _content_sha256(
            checked_provider.contract.to_dict()
        ),
        "backend_contract_sha256": _content_sha256(backend.contract.to_dict()),
        "provider_scope_attestation_sha256": scope_attestation["content_sha256"],
        "provider_safety_metadata_authority_sha256": safety_authority["content_sha256"],
        "arm_config_sha256": {
            arm.name.value: _content_sha256(arm.to_dict()) for arm in arms
        },
    }
    run_fingerprint = _content_sha256(run_binding)
    partial, destination = _fresh_output_directory(output_root, run_fingerprint)
    shared_prepared: Dict[Tuple[str, int, str], Tuple[str, str, str]] = {}
    checkpoint_paths: Dict[str, Tuple[Path, ...]] = {}
    authority_paths: Dict[str, Path] = {}
    arm_rows: List[Mapping[str, Any]] = []
    arm_states: List[_ArmWinnerState] = []
    common_initial_state: Optional[str] = None
    try:
        for arm in arms:
            session = _checked_session(
                _create_session(
                    backend,
                    arm,
                    seed=contract.initialization_seed,
                    production=contract.production,
                )
            )
            access.add_event("training_session_created:" + arm.name.value)
            if common_initial_state is None:
                common_initial_state = session.initial_model_state_sha256
            elif not hmac.compare_digest(
                common_initial_state, session.initial_model_state_sha256
            ):
                raise LocalQ1RunnerError("LOCAL-Q1 arm initial model states differ")
            checkpoint_config = _checkpoint_config(
                contract=contract,
                plan=frozen_plan,
                provider=checked_provider,
                backend=backend,
                arm=arm,
                run_fingerprint=run_fingerprint,
                scope_attestation=scope_attestation,
                safety_authority=safety_authority,
            )
            saved: List[_SavedCheckpoint] = []
            optimizer_steps = 0
            presented = 0
            training_valid = 0
            for epoch in range(1, contract.epochs + 1):
                losses: List[float] = []
                diagnostics: List[str] = []
                for entry, batch_records in zip(
                    train_phase["batches"],
                    train_authority.batches,
                ):
                    prepared = _prepare_checked(
                        checked_provider,
                        batch_records,
                        arm=arm,
                        phase=_TRAIN_PHASE,
                        shared_prepared=shared_prepared,
                        batch_ordinal=int(entry["ordinal"]),
                        access=access,
                    )
                    result = session.train_batch(prepared)
                    if not isinstance(result, TrainBatchResult):
                        raise LocalQ1RunnerError(
                            "backend returned an invalid training result"
                        )
                    if result.valid_count > len(batch_records):
                        raise LocalQ1RunnerError(
                            "training valid_count exceeds batch cardinality"
                        )
                    optimizer_steps += 1
                    presented += len(batch_records)
                    training_valid += result.valid_count
                    losses.append(float(result.loss))
                    diagnostics.append(_content_sha256(dict(result.diagnostics)))
                expected_steps = epoch * len(train_phase["batches"])
                if optimizer_steps != expected_steps:
                    raise LocalQ1RunnerError("optimizer step count differs from plan")
                selection, _, _ = _predict_phase(
                    provider=checked_provider,
                    session=session,
                    arm=arm,
                    planned_batches=select_authority.batches,
                    phase_receipt=_plan_phase_receipt(frozen_plan, _SELECT_PHASE),
                    phase=_SELECT_PHASE,
                    shared_prepared=shared_prepared,
                    threshold=0.5,
                    min_valid_fraction=contract.min_validation_valid_fraction,
                    access=access,
                )
                access.add_event(
                    "epoch_selection_complete:{}:{}".format(arm.name.value, epoch)
                )
                epoch_metrics = {
                    "epoch": epoch,
                    "train": {
                        "batch_count": len(train_phase["batches"]),
                        "optimizer_steps_cumulative": optimizer_steps,
                        "presented_count_cumulative": presented,
                        "valid_count_cumulative": training_valid,
                        "mean_loss": float(np.mean(losses)),
                        "last_loss": float(losses[-1]),
                        "diagnostic_sequence_sha256": _content_sha256(diagnostics),
                    },
                    "validation_select": selection,
                }
                checkpoint_path = partial / (
                    arm.name.value + "-epoch-{:02d}.pt".format(epoch)
                )
                checkpoint = save_checkpoint(
                    checkpoint_path,
                    session.model,
                    config=checkpoint_config,
                    epoch=epoch,
                    optimizer=session.optimizer,
                    metrics=epoch_metrics,
                    provenance={
                        "run_fingerprint_sha256": run_fingerprint,
                        "batch_plan_file_sha256": frozen_plan.canonical_file_sha256,
                        "batch_plan_content_sha256": frozen_plan.content_sha256,
                        "selection_phase_only_for_epoch_ranking": True,
                        "calibration_accessed_before_winner": False,
                        "report_accessed_before_winner": False,
                        "calibration_planned_records_read_before_winner": False,
                        "report_planned_records_read_before_threshold": False,
                        "sealed_real_test_accessed": scope_attestation[
                            "sealed_real_test_accessed"
                        ],
                        "historical_test_accessed": scope_attestation[
                            "historical_test_accessed"
                        ],
                        "provider_scope_attestation_sha256": scope_attestation[
                            "content_sha256"
                        ],
                    },
                )
                saved.append(
                    _SavedCheckpoint(
                        path=checkpoint_path,
                        receipt=checkpoint,
                        config=checkpoint_config,
                        selection_report=selection,
                    )
                )
            if training_valid / presented < contract.min_training_valid_fraction:
                raise LocalQ1RunnerError(
                    "training valid coverage is below the preregistered minimum"
                )
            winner = max(
                saved,
                key=lambda item: (
                    float(
                        item.selection_report["equal_dataset_macro_cluster"]["auroc"]
                    ),
                    float(
                        item.selection_report["equal_dataset_macro_cluster"]["auprc"]
                    ),
                    -item.receipt.epoch,
                ),
            )
            access.add_event(
                "winner_selected:{}:{}".format(arm.name.value, winner.receipt.epoch)
            )

            reload_backend = _checked_backend(
                backend_factory(), production=contract.production
            )
            if reload_backend.contract.to_dict() != backend.contract.to_dict():
                raise LocalQ1RunnerError("fresh reload backend contract changed")
            reload_arm = _arm_for(
                reload_backend, arm.name, production=contract.production
            )
            if reload_arm.to_dict() != arm.to_dict():
                raise LocalQ1RunnerError("fresh reload arm configuration changed")
            reload_session = _checked_session(
                _create_session(
                    reload_backend,
                    reload_arm,
                    seed=contract.initialization_seed,
                    production=contract.production,
                )
            )
            access.add_event("winner_reload_session_created:" + arm.name.value)
            if not hmac.compare_digest(
                reload_session.initial_model_state_sha256,
                session.initial_model_state_sha256,
            ):
                raise LocalQ1RunnerError("fresh reload initialization changed")
            loaded = load_trusted_checkpoint(
                winner.path,
                reload_session.model,
                expected_config=winner.config,
                expected_file_sha256=winner.receipt.file_sha256,
                expected_canonical_content_sha256=(
                    winner.receipt.canonical_content_sha256
                ),
                optimizer=reload_session.optimizer,
                map_location="cpu",
                trusted=True,
            )
            if loaded["epoch"] != winner.receipt.epoch:
                raise LocalQ1RunnerError("restricted reload returned wrong epoch")
            selection_replay, _, _ = _predict_phase(
                provider=checked_provider,
                session=reload_session,
                arm=reload_arm,
                planned_batches=select_authority.batches,
                phase_receipt=_plan_phase_receipt(frozen_plan, _SELECT_PHASE),
                phase=_SELECT_PHASE,
                shared_prepared=shared_prepared,
                threshold=0.5,
                min_valid_fraction=contract.min_validation_valid_fraction,
                access=access,
            )
            if _content_sha256(
                _prediction_replay_projection(selection_replay)
            ) != _content_sha256(
                _prediction_replay_projection(winner.selection_report)
            ):
                raise LocalQ1RunnerError(
                    "winner restricted reload selection replay differs"
                )
            access.add_event("winner_selection_reproduced:" + arm.name.value)
            arm_states.append(
                _ArmWinnerState(
                    arm=arm,
                    reload_arm=reload_arm,
                    reload_session=reload_session,
                    saved=tuple(saved),
                    winner=winner,
                    selection_replay=selection_replay,
                    initial_model_state_sha256=(session.initial_model_state_sha256),
                    optimizer_steps=optimizer_steps,
                    training_presented_count=presented,
                    training_valid_count=training_valid,
                )
            )
            checkpoint_paths[arm.name.value] = tuple(item.path for item in saved)

        access.add_event("all_winners_selection_reproduced")
        if (
            access.planned_records_read_count[_CALIBRATION_PHASE] != 0
            or access.planned_records_read_count[_REPORT_PHASE] != 0
        ):
            raise LocalQ1RunnerError(
                "calibration/report population was opened before all winners"
            )
        calibration_authority = _load_provider_phase(
            checked_provider,
            frozen_plan,
            phase=_CALIBRATION_PHASE,
            safety=safety_phases[_CALIBRATION_PHASE],
            production=contract.production,
            access=access,
        )
        _require_phase_compatible(calibration_authority, phase_authorities)
        phase_authorities[_CALIBRATION_PHASE] = calibration_authority
        if not contract.production:
            if validation_records is None:  # pragma: no cover - checked above
                raise TypeError("fixture validation audit view is required")
            _audit_external_phase(
                validation_records,
                calibration_authority,
                name=_CALIBRATION_PHASE,
                expected_population_count=validation_population_count,
            )
            access.add_event("external_audit:" + _CALIBRATION_PHASE)

        for state in arm_states:
            calibration_report, calibration_probability, calibration_valid = (
                _predict_phase(
                    provider=checked_provider,
                    session=state.reload_session,
                    arm=state.reload_arm,
                    planned_batches=calibration_authority.batches,
                    phase_receipt=_plan_phase_receipt(frozen_plan, _CALIBRATION_PHASE),
                    phase=_CALIBRATION_PHASE,
                    shared_prepared=shared_prepared,
                    threshold=0.5,
                    min_valid_fraction=contract.min_validation_valid_fraction,
                    access=access,
                )
            )
            state.calibration_report = calibration_report
            state.threshold = _fit_calibration_threshold(
                records=calibration_authority.records,
                probabilities=calibration_probability,
                valid=calibration_valid,
                arm=state.arm,
                checkpoint_sha256=state.winner.receipt.file_sha256,
            )
            access.add_event("threshold_frozen:" + state.arm.name.value)

        access.add_event("all_thresholds_frozen")
        if access.planned_records_read_count[_REPORT_PHASE] != 0:
            raise LocalQ1RunnerError(
                "report population was opened before all thresholds"
            )
        report_authority = _load_provider_phase(
            checked_provider,
            frozen_plan,
            phase=_REPORT_PHASE,
            safety=safety_phases[_REPORT_PHASE],
            production=contract.production,
            access=access,
        )
        _require_phase_compatible(report_authority, phase_authorities)
        phase_authorities[_REPORT_PHASE] = report_authority
        if not contract.production:
            if validation_records is None:  # pragma: no cover - checked above
                raise TypeError("fixture validation audit view is required")
            _audit_external_phase(
                validation_records,
                report_authority,
                name=_REPORT_PHASE,
                expected_population_count=validation_population_count,
            )
            access.add_event("external_audit:" + _REPORT_PHASE)

        final_phase_authority = _final_phase_authority(
            attestation=scope_attestation,
            safety_authority=safety_authority,
            phases=phase_authorities,
            access=access,
        )
        for state in arm_states:
            if state.threshold is None or state.calibration_report is None:
                raise LocalQ1RunnerError("arm calibration state is incomplete")
            report, _, _ = _predict_phase(
                provider=checked_provider,
                session=state.reload_session,
                arm=state.reload_arm,
                planned_batches=report_authority.batches,
                phase_receipt=_plan_phase_receipt(frozen_plan, _REPORT_PHASE),
                phase=_REPORT_PHASE,
                shared_prepared=shared_prepared,
                threshold=float(state.threshold["threshold"]),
                min_valid_fraction=contract.min_validation_valid_fraction,
                access=access,
            )
            state.validation_report = report
            authority = _authority_receipt(
                result_eligible=contract.production,
                arm=state.arm,
                plan=frozen_plan,
                run_fingerprint=run_fingerprint,
                winner=state.winner,
                reload_selection=state.selection_replay,
                calibration=state.calibration_report,
                threshold=state.threshold,
                report=report,
                phase_authority=final_phase_authority,
            )
            authority_path = partial / (state.arm.name.value + ".authority.json")
            authority_file_sha = _atomic_json(authority_path, authority)
            arm_rows.append(
                {
                    "arm": state.arm.to_dict(),
                    "initial_model_state_sha256": (state.initial_model_state_sha256),
                    "optimizer_steps": state.optimizer_steps,
                    "training_presented_count": state.training_presented_count,
                    "training_valid_count": state.training_valid_count,
                    "training_valid_fraction": (
                        state.training_valid_count / state.training_presented_count
                    ),
                    "epochs": [
                        {
                            "epoch": item.receipt.epoch,
                            "checkpoint": _checkpoint_claim(item.receipt),
                            "validation_select": item.selection_report,
                        }
                        for item in state.saved
                    ],
                    "winner": {
                        "epoch": state.winner.receipt.epoch,
                        "selection_policy": _SELECTION_POLICY,
                        "checkpoint": _checkpoint_claim(state.winner.receipt),
                        "restricted_fresh_session_reload": True,
                        "selection_replay_match": True,
                    },
                    "validation_calibration": state.calibration_report,
                    "threshold": dict(state.threshold),
                    "validation_report": report,
                    "checkpoint_authority": {
                        "content_sha256": authority["content_sha256"],
                        "file_sha256": authority_file_sha,
                        "result_eligible": contract.production,
                    },
                }
            )
            authority_paths[state.arm.name.value] = authority_path
            access.add_event("report_complete:" + state.arm.name.value)

        access.add_event("all_reports_complete")
        phase_access = _phase_access_receipt(access)

        provider_receipt = dict(
            LocalQ1ReadOnlyBatchProvider.portable_receipt(checked_provider)
            if contract.production
            else checked_provider.portable_receipt()
        )
        _assert_portable(provider_receipt)
        receipt: Dict[str, Any] = {
            "schema_version": LOCAL_Q1_RUNNER_SCHEMA_VERSION,
            "status": (
                "complete_four_local_arms_validation_partitioned_no_test"
                if contract.production
                else "complete_fixture_non_result_no_test"
            ),
            "scope": {
                "experiment": "LOCAL-Q1",
                "mask_only_known_orientation": True,
                "arms": [name.value for name in _LOCAL_ARMS],
                "fused_executed": False,
                "historical_test_accessed": final_phase_authority[
                    "historical_test_accessed"
                ],
                "sealed_real_test_accessed": final_phase_authority[
                    "sealed_real_test_accessed"
                ],
                "provider_phase_authority_sha256": (
                    final_phase_authority["content_sha256"]
                ),
                "provider_safety_metadata_authority_sha256": safety_authority[
                    "content_sha256"
                ],
            },
            "run_fingerprint_sha256": run_fingerprint,
            "run_binding": run_binding,
            "contract": dict(contract.portable_dict()),
            "batch_plan": {
                "file_sha256": frozen_plan.canonical_file_sha256,
                "content_sha256": frozen_plan.content_sha256,
                "phase_commitment_sha256": _content_sha256(
                    frozen_plan.receipt["phases"]
                ),
                "validation_assignment_commitment_sha256": _content_sha256(
                    frozen_plan.receipt["validation_assignment"]
                ),
            },
            "provider_contract": checked_provider.contract.to_dict(),
            "provider_runtime": provider_receipt,
            "provider_scope_attestation": dict(scope_attestation),
            "provider_planned_safety_metadata_authority": dict(safety_authority),
            "provider_planned_phase_authority": dict(final_phase_authority),
            "phase_access": dict(phase_access),
            "backend_contract": backend.contract.to_dict(),
            "arm_results": arm_rows,
            "fairness": {
                "same_initialization_seed": True,
                "same_initial_model_state_sha256": common_initial_state,
                "same_frozen_train_batches_and_order": True,
                "same_optimizer_steps": len(
                    _plan_phase_receipt(frozen_plan, _TRAIN_PHASE)["batches"]
                )
                * contract.epochs,
                "same_validation_partition_all_arms": True,
                "same_record_sequence_across_representations": True,
                "prepared_input_and_candidate_hash_equal_within_representation": True,
                "generic_short_ablation_reordering_used": False,
            },
            "validation_use_policy": {
                "all_phase_supervision_blind_safety_preflight_before_backend": True,
                "epoch_selection_phase": _SELECT_PHASE,
                "calibration_phase": _CALIBRATION_PHASE,
                "report_phase": _REPORT_PHASE,
                "calibration_population_lookup_count": 1,
                "report_population_lookup_count": 1,
                "calibration_prediction_pass_count_per_arm": 1,
                "report_prediction_pass_count_per_arm": 1,
                "all_winner_replays_before_calibration_lookup": True,
                "all_thresholds_frozen_before_report_lookup": True,
                "component_disjoint": True,
                "label_blind_assignment": True,
            },
            "fused_stage": {
                "status": (
                    "not_entered_requires_independently_validated_c0_and_"
                    "dustbin_authorities"
                ),
                "random_or_unvalidated_checkpoint_evidence_allowed": False,
            },
            "portable_privacy": {
                "machine_paths_present": False,
                "row_or_component_ids_present": False,
                "aggregate_commitments_only": True,
            },
        }
        _assert_portable(receipt)
        receipt["content_sha256"] = _self_content_sha256(receipt)
        receipt_path = partial / "local_q1_run_receipt.json"
        _atomic_json(receipt_path, receipt)
        os.replace(partial, destination)
        final_checkpoints = {
            arm: tuple(destination / path.name for path in paths)
            for arm, paths in checkpoint_paths.items()
        }
        final_authorities = {
            arm: destination / path.name for arm, path in authority_paths.items()
        }
        return LocalQ1RunArtifacts(
            run_directory=destination,
            receipt_path=destination / receipt_path.name,
            checkpoint_paths=MappingProxyType(final_checkpoints),
            authority_receipt_paths=MappingProxyType(final_authorities),
            receipt=MappingProxyType(receipt),
        )
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _strict_authority_json(path: Path) -> Tuple[Mapping[str, Any], str]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise LocalQ1RunnerError("checkpoint authority receipt is missing")
    payload = target.read_bytes()
    file_sha = hashlib.sha256(payload).hexdigest()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalQ1RunnerError("checkpoint authority is not strict JSON") from exc
    if not isinstance(value, Mapping) or _canonical_json(value) != payload:
        raise LocalQ1RunnerError("checkpoint authority is not canonical JSON")
    return value, file_sha


def _validated_serialized_safety_authority(
    value: Any, *, expected_content_sha256: str
) -> None:
    root_keys = {
        "authority",
        "metadata_schema_version",
        "provider_version",
        "provenance_safety_scan",
        "phase_read_order",
        "planned_safety_metadata_read_count",
        "planned_safety_metadata_batch_call_count",
        "provider_safety_record_count",
        "phases",
        "supervision_fields_read",
        "raw_component_ids_present",
        "sealed_real_test_accessed",
        "historical_test_accessed",
        "train_validation_component_overlap_count",
        "validation_component_overlap_count",
        "validation_record_ordinal_overlap_count",
        "validation_record_ordinals_exhaustive",
        "completed_before_backend_session_or_output",
        "content_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != root_keys
        or value.get("authority")
        != (
            "provider.planned_safety_metadata_exact_frozen_phase_batches_"
            "supervision_blind_pre_session"
        )
        or value.get("metadata_schema_version")
        != LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION
        or value.get("provider_version") != LOCAL_Q1_BATCH_PROVIDER_VERSION
        or value.get("provenance_safety_scan")
        != _portable(dict(local_q1_provenance_safety_policy()))
        or value.get("phase_read_order") != list(_PHASES)
        or value.get("planned_safety_metadata_read_count")
        != {phase: 1 for phase in _PHASES}
        or value.get("supervision_fields_read") != []
        or value.get("raw_component_ids_present") is not False
        or value.get("sealed_real_test_accessed") is not False
        or value.get("historical_test_accessed") is not False
        or value.get("train_validation_component_overlap_count") != 0
        or value.get("validation_component_overlap_count") != 0
        or value.get("validation_record_ordinal_overlap_count") != 0
        or value.get("validation_record_ordinals_exhaustive") is not True
        or value.get("completed_before_backend_session_or_output") is not True
        or not hmac.compare_digest(
            str(value.get("content_sha256")), expected_content_sha256
        )
        or not hmac.compare_digest(_self_content_sha256(value), expected_content_sha256)
    ):
        raise LocalQ1RunnerError("checkpoint provider safety authority changed")
    batch_calls = value.get("planned_safety_metadata_batch_call_count")
    phases = value.get("phases")
    if (
        not isinstance(batch_calls, Mapping)
        or set(batch_calls) != set(_PHASES)
        or any(
            type(batch_calls[phase]) is not int or batch_calls[phase] <= 0  # noqa: E721
            for phase in _PHASES
        )
        or not isinstance(phases, Mapping)
        or set(phases) != set(_PHASES)
    ):
        raise LocalQ1RunnerError("checkpoint provider safety authority changed")
    expected_phase_keys = {
        "phase",
        "batch_count",
        "record_count",
        "component_count",
        "metadata_content_sequence_sha256",
        "population_ordinal_sequence_sha256",
        "component_set_sha256",
        "sealed_or_missing_marker_count",
        "historical_test_marker_count",
        "split_violation_count",
        "content_sha256",
    }
    total_records = 0
    for phase in _PHASES:
        row = phases[phase]
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_phase_keys
            or row.get("phase") != phase
            or type(row.get("batch_count")) is not int  # noqa: E721
            or row["batch_count"] != batch_calls[phase]
            or type(row.get("record_count")) is not int  # noqa: E721
            or row["record_count"] <= 0
            or type(row.get("component_count")) is not int  # noqa: E721
            or row["component_count"] <= 0
            or row.get("sealed_or_missing_marker_count") != 0
            or row.get("historical_test_marker_count") != 0
            or row.get("split_violation_count") != 0
            or not hmac.compare_digest(
                _self_content_sha256(row), str(row.get("content_sha256"))
            )
        ):
            raise LocalQ1RunnerError("checkpoint provider safety evidence changed")
        for name in (
            "metadata_content_sequence_sha256",
            "population_ordinal_sequence_sha256",
            "component_set_sha256",
            "content_sha256",
        ):
            _require_sha256(row[name], "provider safety phase " + name)
        total_records += row["record_count"]
    if (
        type(value.get("provider_safety_record_count")) is not int  # noqa: E721
        or value["provider_safety_record_count"] != total_records
    ):
        raise LocalQ1RunnerError("checkpoint provider safety record count changed")


def _validated_serialized_phase_authority(
    value: Any, *, expected_content_sha256: str
) -> str:
    root_keys = {
        "authority",
        "provider_scope_attestation_kind",
        "provider_scope_attestation_sha256",
        "provider_safety_metadata_authority",
        "provider_safety_metadata_authority_sha256",
        "provider_planned_record_count",
        "phase_read_order",
        "planned_records_read_count",
        "planned_records_batch_call_count",
        "phases",
        "sealed_real_test_accessed",
        "historical_test_accessed",
        "train_validation_component_overlap_count",
        "validation_component_overlap_count",
        "validation_record_ordinal_overlap_count",
        "validation_record_ordinals_exhaustive",
        "content_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != root_keys
        or value.get("authority")
        != "provider.planned_records_lazy_exact_frozen_phase_batches"
        or value.get("provider_scope_attestation_kind")
        != "externally_verified_provider_cache_build_scope"
        or value.get("phase_read_order") != list(_PHASES)
        or value.get("planned_records_read_count") != {phase: 1 for phase in _PHASES}
        or value.get("sealed_real_test_accessed") is not False
        or value.get("historical_test_accessed") is not False
        or value.get("train_validation_component_overlap_count") != 0
        or value.get("validation_component_overlap_count") != 0
        or value.get("validation_record_ordinal_overlap_count") != 0
        or value.get("validation_record_ordinals_exhaustive") is not True
        or not hmac.compare_digest(
            str(value.get("content_sha256")), expected_content_sha256
        )
        or not hmac.compare_digest(_self_content_sha256(value), expected_content_sha256)
    ):
        raise LocalQ1RunnerError("checkpoint provider phase authority changed")
    attestation_sha256 = _require_sha256(
        value["provider_scope_attestation_sha256"],
        "provider scope attestation SHA-256",
    )
    safety_sha256 = _require_sha256(
        value["provider_safety_metadata_authority_sha256"],
        "provider safety metadata authority SHA-256",
    )
    _validated_serialized_safety_authority(
        value["provider_safety_metadata_authority"],
        expected_content_sha256=safety_sha256,
    )
    batch_calls = value.get("planned_records_batch_call_count")
    phases = value.get("phases")
    if (
        not isinstance(batch_calls, Mapping)
        or set(batch_calls) != set(_PHASES)
        or any(
            type(batch_calls[phase]) is not int or batch_calls[phase] <= 0  # noqa: E721
            for phase in _PHASES
        )
        or not isinstance(phases, Mapping)
        or set(phases) != set(_PHASES)
    ):
        raise LocalQ1RunnerError("checkpoint provider phase authority changed")
    expected_phase_keys = {
        "phase",
        "record_count",
        "component_count",
        "record_sequence_sha256",
        "population_ordinal_sequence_sha256",
        "component_set_sha256",
        "sealed_or_missing_marker_count",
        "historical_test_marker_count",
        "split_violation_count",
        "sealed_real_test_accessed",
        "historical_test_accessed",
        "content_sha256",
    }
    safety_phases = value["provider_safety_metadata_authority"]["phases"]
    total_records = 0
    for phase in _PHASES:
        row = phases[phase]
        safety_row = safety_phases[phase]
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_phase_keys
            or row.get("phase") != phase
            or type(row.get("record_count")) is not int  # noqa: E721
            or row["record_count"] <= 0
            or type(row.get("component_count")) is not int  # noqa: E721
            or row["component_count"] <= 0
            or row.get("sealed_or_missing_marker_count") != 0
            or row.get("historical_test_marker_count") != 0
            or row.get("split_violation_count") != 0
            or row.get("sealed_real_test_accessed") is not False
            or row.get("historical_test_accessed") is not False
            or any(
                row.get(name) != safety_row.get(name)
                for name in (
                    "record_count",
                    "component_count",
                    "population_ordinal_sequence_sha256",
                    "component_set_sha256",
                    "sealed_or_missing_marker_count",
                    "historical_test_marker_count",
                    "split_violation_count",
                )
            )
            or not hmac.compare_digest(
                _self_content_sha256(row), str(row.get("content_sha256"))
            )
        ):
            raise LocalQ1RunnerError("checkpoint provider phase evidence changed")
        for name in (
            "record_sequence_sha256",
            "population_ordinal_sequence_sha256",
            "component_set_sha256",
            "content_sha256",
        ):
            _require_sha256(row[name], "provider phase " + name)
        total_records += row["record_count"]
    if (
        type(value.get("provider_planned_record_count")) is not int  # noqa: E721
        or value["provider_planned_record_count"] != total_records
    ):
        raise LocalQ1RunnerError("checkpoint provider phase record count changed")
    return attestation_sha256


def load_validated_local_checkpoint_binding(
    *,
    authority_receipt_path: Path,
    checkpoint_path: Path,
    expected_authority_file_sha256: str,
    expected_authority_content_sha256: str,
    expected_batch_plan_file_sha256: str,
    expected_batch_plan_content_sha256: str,
    require_dustbin: bool = False,
) -> TrustedCheckpointBinding:
    """Open one externally hash-pinned local winner authority.

    The returned binding remains path-local and is consumed by the backend,
    which performs the restricted checkpoint deserialization.  A self-signed
    receipt is insufficient: both receipt hashes are mandatory external input.
    """

    for name, value in (
        ("authority file SHA-256", expected_authority_file_sha256),
        ("authority content SHA-256", expected_authority_content_sha256),
        ("batch-plan file SHA-256", expected_batch_plan_file_sha256),
        ("batch-plan content SHA-256", expected_batch_plan_content_sha256),
    ):
        _require_sha256(value, name)
    receipt, observed_file_sha = _strict_authority_json(authority_receipt_path)
    if not hmac.compare_digest(observed_file_sha, expected_authority_file_sha256):
        raise LocalQ1RunnerError("checkpoint authority external file hash mismatch")
    if set(receipt) != {
        "schema_version",
        "status",
        "authority_kind",
        "result_eligible",
        "arm",
        "model_scope",
        "checkpoint",
        "checkpoint_config",
        "semantic_projection",
        "semantic_projection_sha256",
        "provider_phase_authority",
        "run_binding",
        "validation_authority",
        "scope",
        "content_sha256",
    }:
        raise LocalQ1RunnerError("checkpoint authority root fields changed")
    if (
        receipt["schema_version"] != LOCAL_Q1_CHECKPOINT_AUTHORITY_SCHEMA_VERSION
        or receipt["status"] != LOCAL_Q1_AUTHORITY_STATUS
        or receipt["authority_kind"] != LOCAL_Q1_AUTHORITY_KIND
        or receipt["result_eligible"] is not True
    ):
        raise LocalQ1RunnerError("checkpoint authority is not result-eligible")
    if not hmac.compare_digest(
        _self_content_sha256(receipt), expected_authority_content_sha256
    ) or not hmac.compare_digest(
        str(receipt["content_sha256"]), expected_authority_content_sha256
    ):
        raise LocalQ1RunnerError("checkpoint authority content hash mismatch")
    _assert_portable(receipt)
    arm = receipt["arm"]
    if arm not in {name.value for name in _LOCAL_ARMS}:
        raise LocalQ1RunnerError("checkpoint authority arm is unsupported")
    if require_dustbin and arm != AblationArmName.LOCAL_DUSTBIN_SINKHORN.value:
        raise LocalQ1RunnerError("fused stage requires dustbin winner authority")
    if receipt["model_scope"] != CheckpointModelScope.PAIRWISE_MODEL.value:
        raise LocalQ1RunnerError("checkpoint authority model scope changed")
    run_binding = receipt["run_binding"]
    if (
        not isinstance(run_binding, Mapping)
        or set(run_binding)
        != {
            "run_fingerprint_sha256",
            "batch_plan_file_sha256",
            "batch_plan_content_sha256",
        }
        or run_binding.get("batch_plan_file_sha256") != expected_batch_plan_file_sha256
        or run_binding.get("batch_plan_content_sha256")
        != expected_batch_plan_content_sha256
    ):
        raise LocalQ1RunnerError("checkpoint authority batch-plan lock changed")
    _require_sha256(run_binding["run_fingerprint_sha256"], "run fingerprint")
    scope = receipt["scope"]
    if not isinstance(scope, Mapping) or set(scope) != {
        "synthetic_train_validation_only",
        "sealed_real_test_accessed",
        "historical_test_accessed",
        "provider_phase_authority_sha256",
        "provider_safety_metadata_authority_sha256",
        "threshold_fit_phase",
        "final_metric_phase",
    }:
        raise LocalQ1RunnerError("checkpoint authority scope changed")
    phase_authority_sha256 = _require_sha256(
        scope["provider_phase_authority_sha256"],
        "provider phase authority SHA-256",
    )
    safety_authority_sha256 = _require_sha256(
        scope["provider_safety_metadata_authority_sha256"],
        "provider safety metadata authority SHA-256",
    )
    if (
        scope["synthetic_train_validation_only"] is not True
        or scope["sealed_real_test_accessed"] is not False
        or scope["historical_test_accessed"] is not False
        or scope["threshold_fit_phase"] != _CALIBRATION_PHASE
        or scope["final_metric_phase"] != _REPORT_PHASE
    ):
        raise LocalQ1RunnerError("checkpoint authority scope changed")
    provider_attestation_sha256 = _validated_serialized_phase_authority(
        receipt["provider_phase_authority"],
        expected_content_sha256=phase_authority_sha256,
    )
    if not hmac.compare_digest(
        safety_authority_sha256,
        receipt["provider_phase_authority"][
            "provider_safety_metadata_authority_sha256"
        ],
    ):
        raise LocalQ1RunnerError("checkpoint provider safety binding changed")
    validation = receipt["validation_authority"]
    if (
        not isinstance(validation, Mapping)
        or set(validation)
        != {
            "selection_policy",
            "winner_epoch",
            "restricted_fresh_session_reload",
            "selection_replay_match",
            "selection_report_sha256",
            "calibration_prediction_pass_count",
            "calibration_report_sha256",
            "threshold_content_sha256",
            "report_prediction_pass_count",
            "report_sha256",
        }
        or validation.get("selection_policy") != _SELECTION_POLICY
        or validation.get("restricted_fresh_session_reload") is not True
        or validation.get("selection_replay_match") is not True
        or validation.get("calibration_prediction_pass_count") != 1
        or validation.get("report_prediction_pass_count") != 1
    ):
        raise LocalQ1RunnerError("checkpoint validation authority is incomplete")
    for name in (
        "selection_report_sha256",
        "calibration_report_sha256",
        "threshold_content_sha256",
        "report_sha256",
    ):
        _require_sha256(validation[name], "checkpoint validation " + name)
    claim = receipt["checkpoint"]
    if not isinstance(claim, Mapping) or set(claim) != {
        "epoch",
        "file_sha256",
        "canonical_content_sha256",
        "config_sha256",
        "model_state_sha256",
        "optimizer_state_sha256",
    }:
        raise LocalQ1RunnerError("checkpoint authority claim changed")
    if (
        type(claim["epoch"]) is not int  # noqa: E721
        or claim["epoch"] <= 0
        or validation["winner_epoch"] != claim["epoch"]
    ):
        raise LocalQ1RunnerError("checkpoint authority winner epoch changed")
    for name in (
        "file_sha256",
        "canonical_content_sha256",
        "config_sha256",
        "model_state_sha256",
        "optimizer_state_sha256",
    ):
        _require_sha256(claim[name], "checkpoint claim " + name)
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise LocalQ1RunnerError("authority checkpoint is missing")
    observed_checkpoint_sha = _sha256_file(checkpoint)
    if not hmac.compare_digest(observed_checkpoint_sha, claim["file_sha256"]):
        raise LocalQ1RunnerError("authority checkpoint file hash mismatch")
    config = receipt["checkpoint_config"]
    if not isinstance(config, Mapping) or not hmac.compare_digest(
        canonical_config_hash(config), claim["config_sha256"]
    ):
        raise LocalQ1RunnerError("authority checkpoint config hash mismatch")
    if (
        set(config)
        != {
            "schema_version",
            "model_config",
            "optimizer_config",
            "aggregation_config",
            "arm",
            "runner_contract",
            "batch_plan",
            "provider_contract",
            "backend_contract",
            "run_fingerprint_sha256",
            "scope",
        }
        or config.get("schema_version") != LOCAL_Q1_CHECKPOINT_CONFIG_SCHEMA_VERSION
    ):
        raise LocalQ1RunnerError("authority checkpoint config schema changed")
    runner_contract = config.get("runner_contract")
    config_plan = config.get("batch_plan")
    provider_contract = config.get("provider_contract")
    backend_contract = config.get("backend_contract")
    config_scope = config.get("scope")
    runner_contract_keys = {
        "schema_version",
        "epochs",
        "initialization_seed",
        "batch_plan_file_sha256",
        "batch_plan_content_sha256",
        "min_training_valid_fraction",
        "min_validation_valid_fraction",
        "production",
        "arms",
        "phase_policy",
        "winner_selection_policy",
        "fused_in_this_stage",
    }
    expected_phase_policy = {
        "pre_session_safety_metadata_reads": list(_PHASES),
        "pre_session_safety_metadata_supervision_fields_read": [],
        "provenance_safety_scan": _portable(dict(local_q1_provenance_safety_policy())),
        "epoch_selection_reads": [_SELECT_PHASE],
        "threshold_fitting_reads": [_CALIBRATION_PHASE],
        "final_reporting_reads": [_REPORT_PHASE],
        "validation_partitions_component_disjoint": True,
        "validation_assignment_label_blind": True,
        "all_winner_replays_before_calibration_population_lookup": True,
        "all_thresholds_frozen_before_report_population_lookup": True,
        "production_external_validation_records_executed": False,
    }
    if (
        not isinstance(runner_contract, Mapping)
        or set(runner_contract) != runner_contract_keys
        or runner_contract.get("schema_version")
        != LOCAL_Q1_RUNNER_CONTRACT_SCHEMA_VERSION
        or type(runner_contract.get("epochs")) is not int  # noqa: E721
        or runner_contract["epochs"] <= 0
        or type(runner_contract.get("initialization_seed")) is not int  # noqa: E721
        or runner_contract["initialization_seed"] < 0
        or runner_contract.get("batch_plan_file_sha256")
        != expected_batch_plan_file_sha256
        or runner_contract.get("batch_plan_content_sha256")
        != expected_batch_plan_content_sha256
        or runner_contract.get("production") is not True
        or runner_contract.get("arms") != [name.value for name in _LOCAL_ARMS]
        or runner_contract.get("phase_policy") != expected_phase_policy
        or runner_contract.get("winner_selection_policy") != _SELECTION_POLICY
        or runner_contract.get("fused_in_this_stage") is not False
    ):
        raise LocalQ1RunnerError("authority runner contract is not formal local-only")
    for name in (
        "min_training_valid_fraction",
        "min_validation_valid_fraction",
    ):
        fraction = runner_contract[name]
        if (
            type(fraction) not in {int, float}  # noqa: E721
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) <= 1.0
        ):
            raise LocalQ1RunnerError(
                "authority runner contract is not formal local-only"
            )
    if (
        not isinstance(config_plan, Mapping)
        or set(config_plan)
        != {
            "file_sha256",
            "content_sha256",
            "phase_commitment_sha256",
            "validation_assignment_commitment_sha256",
        }
        or config_plan.get("file_sha256") != expected_batch_plan_file_sha256
        or config_plan.get("content_sha256") != expected_batch_plan_content_sha256
    ):
        raise LocalQ1RunnerError("authority checkpoint plan binding changed")
    _require_sha256(
        config_plan["phase_commitment_sha256"], "checkpoint phase commitment"
    )
    _require_sha256(
        config_plan["validation_assignment_commitment_sha256"],
        "checkpoint validation assignment commitment",
    )
    if not isinstance(provider_contract, Mapping) or (
        provider_contract.get("provider_version") != LOCAL_Q1_BATCH_PROVIDER_VERSION
        or provider_contract.get("coarse_only_geometry_free") is not True
    ):
        raise LocalQ1RunnerError("authority checkpoint provider identity changed")
    if not isinstance(backend_contract, Mapping):
        raise LocalQ1RunnerError("authority checkpoint backend identity changed")
    expected_backend_contract = {
        "execution_kind": ExecutionKind.SYNTHETIC_TRAIN_VALIDATION.value,
        "backend_version": _FORMAL_BACKEND_VERSION,
        "model_family": _FORMAL_BACKEND_MODEL_FAMILY,
        "device_type": backend_contract.get("device_type"),
        "sealed_real_test_capability": False,
    }
    if backend_contract != expected_backend_contract or backend_contract.get(
        "device_type"
    ) not in {"cpu", "cuda", "mps"}:
        raise LocalQ1RunnerError("authority checkpoint backend identity changed")
    if not isinstance(config_scope, Mapping) or config_scope != {
        "mask_only_known_orientation": True,
        "train_and_validation_only": True,
        "sealed_real_test_accessed": False,
        "historical_test_accessed": False,
        "provider_scope_attestation_sha256": provider_attestation_sha256,
        "provider_safety_metadata_authority_sha256": safety_authority_sha256,
        "fused": False,
    }:
        raise LocalQ1RunnerError("authority checkpoint scope changed")
    if config.get("run_fingerprint_sha256") != run_binding["run_fingerprint_sha256"]:
        raise LocalQ1RunnerError("authority run fingerprint binding changed")

    arm_name = AblationArmName(arm)
    reference_backend = LocalQ1Backend(device="cpu", mode=LocalQ1BackendMode.FORMAL)
    expected_arm = _arm_for(reference_backend, arm_name, production=True)
    if (
        _portable(config.get("arm")) != _portable(expected_arm.to_dict())
        or not hmac.compare_digest(
            canonical_config_hash(config.get("model_config")),
            canonical_config_hash(expected_arm.model_config),
        )
        or not hmac.compare_digest(
            canonical_config_hash(config.get("optimizer_config")),
            canonical_config_hash(expected_arm.optimizer_config),
        )
        or not hmac.compare_digest(
            canonical_config_hash(config.get("aggregation_config")),
            canonical_config_hash(expected_arm.aggregation_config),
        )
    ):
        raise LocalQ1RunnerError("authority checkpoint arm semantics changed")
    model_scope = CheckpointModelScope.PAIRWISE_MODEL
    projection = checkpoint_semantic_projection(config, model_scope)
    semantic_sha = canonical_config_hash(projection)
    if _portable(projection) != receipt[
        "semantic_projection"
    ] or not hmac.compare_digest(semantic_sha, receipt["semantic_projection_sha256"]):
        raise LocalQ1RunnerError("checkpoint semantic projection changed")
    verification_session = LocalQ1Backend.create_session(
        reference_backend,
        expected_arm,
        seed=int(runner_contract["initialization_seed"]),
    )
    loaded = load_trusted_checkpoint(
        checkpoint,
        verification_session.model,
        expected_config=config,
        expected_file_sha256=claim["file_sha256"],
        expected_canonical_content_sha256=claim["canonical_content_sha256"],
        map_location="cpu",
        trusted=True,
    )
    if (
        loaded["epoch"] != claim["epoch"]
        or loaded["model_state_sha256"] != claim["model_state_sha256"]
        or loaded["optimizer_state_sha256"] != claim["optimizer_state_sha256"]
    ):
        raise LocalQ1RunnerError("restricted checkpoint authority replay changed")
    authority_identity = checkpoint_authority_identity_sha256(
        model_scope=model_scope,
        checkpoint_file_sha256=claim["file_sha256"],
        checkpoint_canonical_content_sha256=claim["canonical_content_sha256"],
        checkpoint_config_sha256=claim["config_sha256"],
        semantic_projection_sha256=semantic_sha,
        authority_kind=LOCAL_Q1_AUTHORITY_KIND,
        authority_status=LOCAL_Q1_AUTHORITY_STATUS,
        authority_receipt_file_sha256=expected_authority_file_sha256,
        authority_receipt_content_sha256=expected_authority_content_sha256,
    )
    return TrustedCheckpointBinding(
        path=checkpoint,
        file_sha256=claim["file_sha256"],
        canonical_content_sha256=claim["canonical_content_sha256"],
        config_sha256=claim["config_sha256"],
        expected_config=config,
        model_scope=model_scope,
        semantic_projection_sha256=semantic_sha,
        authority_kind=LOCAL_Q1_AUTHORITY_KIND,
        authority_status=LOCAL_Q1_AUTHORITY_STATUS,
        authority_receipt_file_sha256=expected_authority_file_sha256,
        authority_receipt_content_sha256=expected_authority_content_sha256,
        authority_identity_sha256=authority_identity,
    )


def require_fused_stage_authorities(
    *,
    coarse: Optional[TrustedCheckpointBinding],
    dustbin: Optional[TrustedCheckpointBinding],
) -> FusedCheckpointBindings:
    """Return typed fused evidence only when both independent authorities exist."""

    if coarse is None or dustbin is None:
        raise LocalQ1BackendError(
            "formal fused stage requires validated C0 and dustbin authorities"
        )
    if not isinstance(coarse, TrustedCheckpointBinding) or not isinstance(
        dustbin, TrustedCheckpointBinding
    ):
        raise TypeError("fused authorities must be TrustedCheckpointBinding values")
    return FusedCheckpointBindings(coarse=coarse, local=dustbin)


__all__ = [
    "LOCAL_Q1_CHECKPOINT_AUTHORITY_SCHEMA_VERSION",
    "LOCAL_Q1_RUNNER_CONTRACT_SCHEMA_VERSION",
    "LOCAL_Q1_RUNNER_SCHEMA_VERSION",
    "LocalQ1RunArtifacts",
    "LocalQ1RunnerContract",
    "LocalQ1RunnerError",
    "load_validated_local_checkpoint_binding",
    "require_fused_stage_authorities",
    "run_local_q1",
]
