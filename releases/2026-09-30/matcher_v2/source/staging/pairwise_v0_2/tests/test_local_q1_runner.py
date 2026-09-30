from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import pytest
import torch
from torch import nn

from staging.pairwise_v0_2.models.local_matcher import MatcherMode
from staging.pairwise_v0_2.models.pairwise import PairwiseModelConfig
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.checkpoint import (
    canonical_config_hash,
    canonical_tensor_tree_sha256,
)
from staging.pairwise_v0_2.training.local_q1_backend import (
    LOCAL_Q1_BACKEND_VERSION,
    LocalQ1Backend,
    LocalQ1BackendError,
    LocalQ1BackendMode,
)
from staging.pairwise_v0_2.training.local_q1_provider import (
    LOCAL_Q1_BATCH_PROVIDER_VERSION,
    LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION,
    LOCAL_Q1_PROVENANCE_SAFETY_MAX_DEPTH,
    LOCAL_Q1_PROVENANCE_SAFETY_MAX_NODES,
    LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE,
    FrozenLocalQ1PlannedSafetyMetadata,
    local_q1_provenance_safety_policy,
    scan_local_q1_provenance_safety,
)
from staging.pairwise_v0_2.training.local_q1_runner import (
    LocalQ1RunnerContract,
    LocalQ1RunnerError,
    _checked_backend,
    _prediction_replay_projection,
    load_validated_local_checkpoint_binding,
    require_fused_stage_authorities,
    run_local_q1,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    BackendContract,
    BatchProviderContract,
    ExecutionKind,
    PredictionBatch,
    PreparedAblationBatch,
    TrainBatchResult,
    record_sequence_fingerprint,
)


def _portable(value: Any) -> Any:
    if hasattr(value, "value"):
        return _portable(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _portable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_portable(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _portable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else _canonical(value)
    return hashlib.sha256(payload).hexdigest()


def _component_token(value: str) -> str:
    return hashlib.sha256(
        LOCAL_Q1_VALIDATION_ASSIGNMENT_NAMESPACE.encode("utf-8")
        + b"\0"
        + value.encode("utf-8")
    ).hexdigest()


_ARCHIVE = ArchiveBinding(
    logical_id="fixture://local-q1-runner/masks",
    archive_format="zip",
    sha256="a" * 64,
)


def _record(
    ordinal: int,
    *,
    split: str,
    dataset: str,
    label: bool,
    component: str,
) -> TrainingPairRecord:
    group = "group/{}/{}".format(split, component)
    endpoints = []
    for side in ("a", "b"):
        fragment = "fragment/{}/{}/{}".format(split, ordinal, side)
        endpoints.append(
            MaskMemberRef(
                binding=_ARCHIVE,
                archive_member="masks/{}.png".format(fragment.replace("/", "_")),
                fragment_id=fragment,
                dataset_id=dataset,
                canonical_group_id=group,
                component_id=component,
                split=split,
                threshold_rule="grayscale_uint8_gt_127",
                content_sha256=_sha("content/" + fragment),
            )
        )
    return TrainingPairRecord(
        fragment_a=endpoints[0],
        fragment_b=endpoints[1],
        label=label,
        direction_b_wrt_a="right" if label else None,
        dataset_id=dataset,
        canonical_group_id=group,
        component_id=component,
        split=split,
        canonical_pair_key=tuple(
            sorted((endpoints[0].fragment_id, endpoints[1].fragment_id))
        ),
        label_origin="fixture",
        provenance={
            "fixture_non_result": True,
            "real_dunhuang_sealed_test": False,
        },
    )


def _populations():
    train = tuple(
        _record(
            index,
            split="train",
            dataset="train_fixture",
            label=index % 2 == 0,
            component="train-component-{}".format(index),
        )
        for index in range(4)
    )
    validation = []
    ordinal = 100
    for phase_index in range(3):
        for dataset in ("mm_fixture", "eccv_fixture"):
            for label in (False, True):
                validation.append(
                    _record(
                        ordinal,
                        split="val",
                        dataset=dataset,
                        label=label,
                        component="val-component-{}-{}-{}".format(
                            phase_index, dataset, int(label)
                        ),
                    )
                )
                ordinal += 1
    return train, tuple(validation)


def _entry(ordinals: Sequence[int], ordinal: int = 0):
    return {
        "ordinal": ordinal,
        "record_ordinals": list(ordinals),
    }


def _fixture_plan():
    train, validation = _populations()
    phase_ordinals = {
        "train": tuple(range(4)),
        "validation_select": tuple(range(0, 4)),
        "validation_calibration": tuple(range(4, 8)),
        "validation_report": tuple(range(8, 12)),
    }
    phases = {
        phase: {
            "phase": phase,
            "record_count": len(ordinals),
            "population_ordinal_set_sha256": _sha(sorted(ordinals)),
            "batches": [
                {
                    **_entry(ordinals),
                    "batch_geometry_sha256": _sha(
                        {"fixture_phase": phase, "record_ordinals": list(ordinals)}
                    ),
                }
            ],
        }
        for phase, ordinals in phase_ordinals.items()
    }
    phase_component_sha256 = {}
    for phase, ordinals in phase_ordinals.items():
        source = train if phase == "train" else validation
        phase_component_sha256[phase] = _sha(
            sorted({_component_token(source[index].component_id) for index in ordinals})
        )
    assignment = {
        "status": (
            "frozen_label_blind_component_partition_nonempty_disjoint_exhaustive"
        ),
        "phase_order": [
            "validation_select",
            "validation_calibration",
            "validation_report",
        ],
        "fields_read": ["split", "component_id"],
        "supervision_fields_read": [],
        "assignment_sha256": _sha("fixture-validation-assignment"),
        "phases": {
            phase: {
                "record_count": 4,
                "component_set_sha256": phase_component_sha256[phase],
            }
            for phase in (
                "validation_select",
                "validation_calibration",
                "validation_report",
            )
        },
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
    receipt = {"phases": phases, "validation_assignment": assignment}
    content_sha = _sha(receipt)
    file_sha = _sha({"receipt": receipt, "kind": "fixture-plan-file"})
    return SimpleNamespace(
        receipt=receipt,
        content_sha256=content_sha,
        canonical_file_sha256=file_sha,
    )


_SAFETY_PRIVACY = {
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
    "provenance_safety_scan": _portable(dict(local_q1_provenance_safety_policy())),
}


class _FixtureProvider:
    def __init__(
        self,
        train: Optional[Sequence[TrainingPairRecord]] = None,
        validation: Optional[Sequence[TrainingPairRecord]] = None,
        plan: Any = None,
    ) -> None:
        default_train, default_validation = _populations()
        self.train = tuple(default_train if train is None else train)
        self.validation = tuple(
            default_validation if validation is None else validation
        )
        self.plan = _fixture_plan() if plan is None else plan
        self.contract = BatchProviderContract(
            coarse_preprocess_mode="tight_crop_letterbox",
            coarse_preprocessing_sha256=_sha("fixture-preprocessing"),
            geometry_config_sha256=_sha("fixture-geometry"),
            cache_interface="fixture_read_only_zero_miss_zero_write",
            provider_version="fixture-local-q1-provider/0.1",
        )
        self.calls = []
        self.safety_attempts = []
        self.safety_calls = []
        self.planned_calls = []

    def planned_safety_metadata(self, phase: str, batch_ordinal: int):
        self.safety_attempts.append((phase, batch_ordinal))
        phase_receipt = self.plan.receipt["phases"][phase]
        entry = phase_receipt["batches"][batch_ordinal]
        source = self.train if phase == "train" else self.validation
        expected_split = "train" if phase == "train" else "val"
        phase_records = tuple(
            source[index]
            for batch in phase_receipt["batches"]
            for index in batch["record_ordinals"]
        )
        phase_component_set_sha256 = _sha(
            sorted({_component_token(record.component_id) for record in phase_records})
        )
        rows = []
        for population_ordinal in entry["record_ordinals"]:
            record = source[population_ordinal]
            provenance_markers = scan_local_q1_provenance_safety(record.provenance)
            rows.append(
                {
                    "population_ordinal": population_ordinal,
                    "expected_split": record.split,
                    "component_token_sha256": _component_token(record.component_id),
                    "sealed_real_test_marker_explicit_false": (
                        provenance_markers.top_level_sealed_marker_explicit_false
                    ),
                    "sealed_scope_marker": (provenance_markers.sealed_scope_marker),
                    "historical_test_marker": (
                        provenance_markers.historical_test_marker
                    ),
                }
            )
        receipt = {
            "schema_version": LOCAL_Q1_PLANNED_SAFETY_METADATA_SCHEMA_VERSION,
            "status": (
                "frozen_authoritative_population_safety_metadata_no_materialization"
            ),
            "provider_version": LOCAL_Q1_BATCH_PROVIDER_VERSION,
            "batch_plan_file_sha256": self.plan.canonical_file_sha256,
            "batch_plan_content_sha256": self.plan.content_sha256,
            "validation_assignment_sha256": self.plan.receipt["validation_assignment"][
                "assignment_sha256"
            ],
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
                None
                if phase == "train"
                else self.plan.receipt["validation_assignment"]["phases"][phase][
                    "component_set_sha256"
                ]
            ),
            "record_count": len(rows),
            "population_ordinal_sequence_sha256": _sha(list(entry["record_ordinals"])),
            "batch_component_set_sha256": _sha(
                sorted({row["component_token_sha256"] for row in rows})
            ),
            "records": rows,
            "privacy": dict(_SAFETY_PRIVACY),
        }
        receipt["content_sha256"] = _sha(receipt)
        self.safety_calls.append((phase, batch_ordinal))
        return FrozenLocalQ1PlannedSafetyMetadata(
            receipt=receipt,
            content_sha256=receipt["content_sha256"],
        )

    def planned_records(self, phase: str, batch_ordinal: int):
        entry = self.plan.receipt["phases"][phase]["batches"][batch_ordinal]
        source = self.train if phase == "train" else self.validation
        records = tuple(source[index] for index in entry["record_ordinals"])
        self.planned_calls.append((phase, batch_ordinal))
        return records

    def prepare(
        self,
        records: Sequence[TrainingPairRecord],
        *,
        arm: AblationArm,
        phase: str,
    ) -> PreparedAblationBatch:
        population = tuple(records)
        sequence = record_sequence_fingerprint(population)
        self.calls.append(
            {
                "arm": arm.name.value,
                "phase": phase,
                "sequence_sha256": sequence,
                "component_set": frozenset(
                    record.component_id for record in population
                ),
            }
        )
        return PreparedAblationBatch(
            payload=population,
            sample_count=len(population),
            record_sequence_sha256=sequence,
            prepared_input_sha256=_sha(
                "prepared/{}/{}".format(arm.candidate_representation, sequence)
            ),
            local_candidate_sha256=_sha(
                "candidates/{}/{}".format(arm.candidate_representation, sequence)
            ),
            coarse_preprocessing_sha256=self.contract.coarse_preprocessing_sha256,
            geometry_config_sha256=self.contract.geometry_config_sha256,
            processing_counts={
                "mask_load_count": 2 * len(population),
                "coarse_preprocess_count": 2 * len(population),
                "geometry_build_count": 0,
                "geometry_cache_read_count": 2 * len(population),
                "geometry_cache_write_count": 0,
                "local_candidate_count": 4 * len(population),
            },
            candidate_representation=arm.candidate_representation,
        )

    def portable_receipt(self):
        return {
            "schema_version": "fixture-provider-runtime/0.1",
            "status": "fixture_non_result",
            "prepare_call_count": len(self.calls),
            "phase_call_counts": {
                phase: sum(call["phase"] == phase for call in self.calls)
                for phase in (
                    "train",
                    "validation_select",
                    "validation_calibration",
                    "validation_report",
                )
            },
        }


class _FixtureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.training_epoch = nn.Parameter(torch.zeros(()))


class _FixtureSession:
    def __init__(self, arm: AblationArm, seed: int, session_log: list) -> None:
        torch.manual_seed(seed)
        self.arm = arm
        # Mirrors production diagnostics such as elapsed time and memory: a
        # freshly reloaded session has a different runtime-only value even when
        # its checkpoint predictions are identical.
        self.runtime_session_ordinal = len(session_log)
        self.model = _FixtureModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.initial_model_state_sha256 = canonical_tensor_tree_sha256(
            self.model.state_dict()
        )
        session_log.append((arm.name.value, seed, self.initial_model_state_sha256))

    def train_batch(self, batch: PreparedAblationBatch) -> TrainBatchResult:
        with torch.no_grad():
            self.model.training_epoch.add_(1.0)
        return TrainBatchResult(
            loss=1.0 / float(self.model.training_epoch.item()),
            valid_count=batch.sample_count,
            diagnostics={
                "matcher_mode": self.arm.matcher_mode,
                "fixture_non_result": True,
                "runtime_session_ordinal": self.runtime_session_ordinal,
            },
        )

    def predict_batch(self, batch: PreparedAblationBatch, *, evidence):
        del evidence
        learned = float(self.model.training_epoch.item()) >= 2.0
        probability = []
        for record in batch.payload:
            if learned:
                probability.append(0.9 if record.label else 0.1)
            else:
                probability.append(0.4 if record.label else 0.6)
        return PredictionBatch(
            probability=torch.tensor(probability, dtype=torch.float32),
            valid=torch.ones(len(probability), dtype=torch.bool),
            diagnostics={
                "matcher_mode": self.arm.matcher_mode,
                "fixture_non_result": True,
                "runtime_session_ordinal": self.runtime_session_ordinal,
            },
        )


class _FixtureBackend:
    def __init__(self, session_log: list) -> None:
        self.contract = BackendContract(
            execution_kind=ExecutionKind.PROVIDER_FREE_DRY_RUN,
            backend_version="fixture-local-q1-backend/0.1",
            model_family="fixture-local-four-arm",
            device_type="cpu",
        )
        self.model_template = PairwiseModelConfig()
        self.optimizer_config = {
            "name": "SGD",
            "lr": 0.1,
            "fixture_non_result": True,
        }
        pooling = self.model_template.arc_pooling
        self.aggregation_config = {
            "arc_pooling": {
                "mode": pooling.mode.value,
                "temperature": pooling.temperature,
                "top_k": pooling.top_k,
            },
            "direction_pooling": "log_mean_exp",
            "candidate_population": "same_frozen_LOCAL_Q1_provider_payload",
        }
        self._session_log = session_log

    def model_config_for(self, name: AblationArmName):
        matcher = {
            AblationArmName.LOCAL_DUAL_SOFTMAX: MatcherMode.DUAL_SOFTMAX.value,
            AblationArmName.LOCAL_DUSTBIN_SINKHORN: (
                MatcherMode.DUSTBIN_SINKHORN.value
            ),
            AblationArmName.KEYPOINT_DUAL_SOFTMAX: MatcherMode.DUAL_SOFTMAX.value,
            AblationArmName.KEYPOINT_DUSTBIN_SINKHORN: (
                MatcherMode.DUSTBIN_SINKHORN.value
            ),
        }[name]
        architecture = replace(
            self.model_template,
            local=replace(self.model_template.local, matcher_mode=matcher),
        ).to_dict()
        return {
            "schema_version": "fixture-local-model-contract/0.1",
            "architecture": architecture,
            "candidate_representation": (
                "contour_keypoint"
                if name
                in {
                    AblationArmName.KEYPOINT_DUAL_SOFTMAX,
                    AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
                }
                else "multirun_sliding_window"
            ),
            "initialization": {
                "kind": "runner_common_seed_random_local_initialization",
                "result_eligible": False,
            },
            "training": {
                "trainable_components": ["local_model"],
                "frozen_components": ["coarse_model", "fusion"],
                "score_source": "local",
                "step_config": {"fixture_non_result": True},
                "step_config_sha256": canonical_config_hash(
                    {"fixture_non_result": True}
                ),
            },
            "provider_tensor_contract": {
                "payload_type": "fixture_records",
                "fixture_non_result": True,
            },
        }

    def create_session(self, arm: AblationArm, *, seed: int):
        return _FixtureSession(arm, seed, self._session_log)


def _run_case(tmp_path: Path):
    train, validation = _populations()
    plan = _fixture_plan()
    contract = LocalQ1RunnerContract(
        epochs=2,
        initialization_seed=260829,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        min_training_valid_fraction=1.0,
        min_validation_valid_fraction=1.0,
        production=False,
    )
    provider = _FixtureProvider(train, validation, plan)
    session_log = []
    artifacts = run_local_q1(
        contract=contract,
        plan=plan,
        train_records=train,
        validation_records=validation,
        provider=provider,
        backend_factory=lambda: _FixtureBackend(session_log),
        output_root=tmp_path / "outputs",
    )
    return artifacts, provider, session_log, contract, plan


def test_runner_uses_exact_plan_order_and_three_validation_roles(tmp_path: Path):
    artifacts, provider, session_log, contract, _plan = _run_case(tmp_path)

    assert artifacts.receipt["status"] == "complete_fixture_non_result_no_test"
    assert artifacts.receipt["scope"]["fused_executed"] is False
    assert artifacts.receipt["scope"]["sealed_real_test_accessed"] is False
    scope = artifacts.receipt["provider_planned_phase_authority"]
    assert scope["authority"] == (
        "provider.planned_records_lazy_exact_frozen_phase_batches"
    )
    assert scope["provider_scope_attestation_kind"] == (
        "fixture_non_result_no_external_scope_authority"
    )
    assert len(scope["provider_scope_attestation_sha256"]) == 64
    for phase in (
        "train",
        "validation_select",
        "validation_calibration",
        "validation_report",
    ):
        assert scope["phases"][phase]["sealed_or_missing_marker_count"] == 0
        assert scope["phases"][phase]["historical_test_marker_count"] == 0
    assert scope["train_validation_component_overlap_count"] == 0
    assert (
        artifacts.receipt["scope"]["provider_phase_authority_sha256"]
        == scope["content_sha256"]
    )
    assert provider.planned_calls == [
        ("train", 0),
        ("validation_select", 0),
        ("validation_calibration", 0),
        ("validation_report", 0),
    ]
    assert provider.safety_calls == [
        ("train", 0),
        ("validation_select", 0),
        ("validation_calibration", 0),
        ("validation_report", 0),
    ]
    safety = artifacts.receipt["provider_planned_safety_metadata_authority"]
    assert safety["supervision_fields_read"] == []
    assert safety["provenance_safety_scan"] == _portable(
        dict(local_q1_provenance_safety_policy())
    )
    assert safety["completed_before_backend_session_or_output"] is True
    assert safety["train_validation_component_overlap_count"] == 0
    assert safety["validation_component_overlap_count"] == 0
    assert (
        scope["provider_safety_metadata_authority_sha256"] == safety["content_sha256"]
    )
    for authority_path in artifacts.authority_receipt_paths.values():
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        assert (
            authority["scope"]["sealed_real_test_accessed"]
            is scope["sealed_real_test_accessed"]
        )
        assert (
            authority["scope"]["historical_test_accessed"]
            is scope["historical_test_accessed"]
        )
        assert (
            authority["provider_phase_authority"]["content_sha256"]
            == scope["content_sha256"]
        )
        assert (
            authority["checkpoint_config"]["scope"]["provider_scope_attestation_sha256"]
            == artifacts.receipt["provider_scope_attestation"]["content_sha256"]
        )
    phase_access = artifacts.receipt["phase_access"]
    assert phase_access["planned_safety_metadata_read_count"] == {
        "train": 1,
        "validation_select": 1,
        "validation_calibration": 1,
        "validation_report": 1,
    }
    assert phase_access["planned_safety_metadata_batch_call_count"] == {
        "train": 1,
        "validation_select": 1,
        "validation_calibration": 1,
        "validation_report": 1,
    }
    assert phase_access["planned_records_read_count"] == {
        "train": 1,
        "validation_select": 1,
        "validation_calibration": 1,
        "validation_report": 1,
    }
    assert phase_access["planned_records_batch_call_count"] == {
        "train": 1,
        "validation_select": 1,
        "validation_calibration": 1,
        "validation_report": 1,
    }
    assert phase_access["prepare_call_count"] == {
        "train": 8,
        "validation_select": 12,
        "validation_calibration": 4,
        "validation_report": 4,
    }
    assert phase_access["predict_call_count"] == {
        "train": 0,
        "validation_select": 12,
        "validation_calibration": 4,
        "validation_report": 4,
    }
    events = phase_access["event_log"]
    backend_index = events.index("backend_created")
    calibration_index = events.index("planned_records_read:validation_calibration")
    report_index = events.index("planned_records_read:validation_report")
    assert events.index("planned_records_read:train") < backend_index
    assert events.index("planned_records_read:validation_select") < backend_index
    assert max(
        events.index("planned_safety_metadata_read:" + phase)
        for phase in (
            "train",
            "validation_select",
            "validation_calibration",
            "validation_report",
        )
    ) < events.index("planned_records_read:train")
    assert backend_index < min(
        index
        for index, event in enumerate(events)
        if event.startswith("training_session_created:")
    )
    assert max(
        index
        for index, event in enumerate(events)
        if event.startswith("epoch_selection_complete:")
    ) < events.index("all_winners_selection_reproduced")
    assert (
        max(
            index
            for index, event in enumerate(events)
            if event.startswith("winner_selection_reproduced:")
        )
        < calibration_index
    )
    assert (
        max(
            index
            for index, event in enumerate(events)
            if event.startswith("threshold_frozen:")
        )
        < report_index
    )
    assert (
        artifacts.receipt["fairness"]["generic_short_ablation_reordering_used"] is False
    )
    assert artifacts.receipt["fairness"]["same_optimizer_steps"] == 2
    assert len(session_log) == 8  # four training sessions + four winner reloads
    assert {seed for _arm, seed, _state in session_log} == {
        contract.initialization_seed
    }
    assert len({state for _arm, _seed, state in session_log}) == 1

    counts = {
        phase: sum(call["phase"] == phase for call in provider.calls)
        for phase in (
            "train",
            "validation_select",
            "validation_calibration",
            "validation_report",
        )
    }
    assert counts == {
        "train": 8,
        "validation_select": 12,
        "validation_calibration": 4,
        "validation_report": 4,
    }
    for arm in (
        AblationArmName.LOCAL_DUAL_SOFTMAX.value,
        AblationArmName.LOCAL_DUSTBIN_SINKHORN.value,
        AblationArmName.KEYPOINT_DUAL_SOFTMAX.value,
        AblationArmName.KEYPOINT_DUSTBIN_SINKHORN.value,
    ):
        sequence = [call["phase"] for call in provider.calls if call["arm"] == arm]
        assert sequence == [
            "train",
            "validation_select",
            "train",
            "validation_select",
            "validation_select",
            "validation_calibration",
            "validation_report",
        ]

    phase_components = {
        phase: set().union(
            *[
                set(call["component_set"])
                for call in provider.calls
                if call["phase"] == phase
            ]
        )
        for phase in (
            "validation_select",
            "validation_calibration",
            "validation_report",
        )
    }
    assert not (
        phase_components["validation_select"]
        & phase_components["validation_calibration"]
    )
    assert not (
        phase_components["validation_select"] & phase_components["validation_report"]
    )
    assert not (
        phase_components["validation_calibration"]
        & phase_components["validation_report"]
    )


def test_each_arm_selects_epoch_two_then_calibrates_and_reports_once(tmp_path: Path):
    artifacts, _provider, _sessions, _contract, _plan = _run_case(tmp_path)

    for row in artifacts.receipt["arm_results"]:
        assert row["winner"]["epoch"] == 2
        assert row["winner"]["restricted_fresh_session_reload"] is True
        assert row["winner"]["selection_replay_match"] is True
        assert len(row["epochs"]) == 2
        assert row["epochs"][0]["validation_select"]["equal_dataset_macro_cluster"][
            "auroc"
        ] == pytest.approx(0.0)
        assert row["epochs"][1]["validation_select"]["equal_dataset_macro_cluster"][
            "auroc"
        ] == pytest.approx(1.0)
        assert row["validation_calibration"]["phase"] == "validation_calibration"
        assert row["threshold"]["source_split"] == "validation_calibration"
        assert row["validation_report"]["phase"] == "validation_report"
        assert row["checkpoint_authority"]["result_eligible"] is False
        authority = json.loads(
            artifacts.authority_receipt_paths[row["arm"]["name"]].read_text(
                encoding="utf-8"
            )
        )
        original_selection = row["epochs"][1]["validation_select"]
        # The fixture deliberately emits a per-session runtime diagnostic, so
        # the full report hashes differ while prediction replay still matches.
        assert authority["validation_authority"][
            "selection_report_sha256"
        ] != _sha(original_selection)

    assert all(len(paths) == 2 for paths in artifacts.checkpoint_paths.values())
    assert all(
        path.is_file()
        for paths in artifacts.checkpoint_paths.values()
        for path in paths
    )
    assert all(path.is_file() for path in artifacts.authority_receipt_paths.values())


def test_prediction_replay_projection_excludes_only_runtime_diagnostics() -> None:
    original = {
        "phase": "validation_select",
        "batch_count": 1,
        "prediction_commitment_sha256": "a" * 64,
        "record_sequence_sha256": "b" * 64,
        "coverage": {"record_count": 4, "valid_count": 4},
        "equal_dataset_macro_cluster": {"auroc": 0.75, "auprc": 0.8},
        "diagnostic_sequence_sha256": "c" * 64,
    }
    replay = copy.deepcopy(original)
    replay["diagnostic_sequence_sha256"] = "d" * 64

    assert _prediction_replay_projection(original) == _prediction_replay_projection(
        replay
    )

    replay["prediction_commitment_sha256"] = "e" * 64
    assert _prediction_replay_projection(original) != _prediction_replay_projection(
        replay
    )


def test_validation_external_audit_and_provider_lookup_are_truly_lazy(
    tmp_path: Path,
) -> None:
    train, validation = _populations()
    plan = _fixture_plan()
    provider = _FixtureProvider(train, validation, plan)
    access_snapshots = []

    class GuardedValidationView(Sequence[TrainingPairRecord]):
        def __len__(self) -> int:
            return len(validation)

        def __getitem__(self, index):
            if isinstance(index, slice):
                raise AssertionError("runner must not slice the validation view")
            access_snapshots.append(
                {
                    "ordinal": index,
                    "planned": tuple(provider.planned_calls),
                    "prepared_phases": tuple(call["phase"] for call in provider.calls),
                }
            )
            return validation[index]

    backend_snapshots = []
    safety_snapshots = []

    def backend_factory():
        backend_snapshots.append(tuple(provider.planned_calls))
        safety_snapshots.append(tuple(provider.safety_calls))
        return _FixtureBackend([])

    contract = LocalQ1RunnerContract(
        epochs=1,
        initialization_seed=12,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=False,
    )
    artifacts = run_local_q1(
        contract=contract,
        plan=plan,
        train_records=train,
        validation_records=GuardedValidationView(),
        provider=provider,
        backend_factory=backend_factory,
        output_root=tmp_path / "lazy",
    )

    assert all(
        tuple(phase for phase, _ordinal in snapshot) == ("train", "validation_select")
        for snapshot in backend_snapshots
    )
    assert all(
        tuple(phase for phase, _ordinal in snapshot)
        == (
            "train",
            "validation_select",
            "validation_calibration",
            "validation_report",
        )
        for snapshot in safety_snapshots
    )
    select_reads = [row for row in access_snapshots if row["ordinal"] < 4]
    calibration_reads = [row for row in access_snapshots if 4 <= row["ordinal"] < 8]
    report_reads = [row for row in access_snapshots if row["ordinal"] >= 8]
    assert len(select_reads) == len(calibration_reads) == len(report_reads) == 4
    assert all(
        tuple(phase for phase, _ordinal in row["planned"])
        == ("train", "validation_select")
        and row["prepared_phases"] == ()
        for row in select_reads
    )
    assert all(
        tuple(phase for phase, _ordinal in row["planned"])
        == ("train", "validation_select", "validation_calibration")
        and row["prepared_phases"].count("validation_select") == 8
        and "validation_calibration" not in row["prepared_phases"]
        for row in calibration_reads
    )
    assert all(
        tuple(phase for phase, _ordinal in row["planned"])
        == (
            "train",
            "validation_select",
            "validation_calibration",
            "validation_report",
        )
        and row["prepared_phases"].count("validation_calibration") == 4
        and "validation_report" not in row["prepared_phases"]
        for row in report_reads
    )
    assert artifacts.receipt["phase_access"]["ordering_proof"] == {
        "all_phase_safety_metadata_before_backend": True,
        "safety_metadata_supervision_fields_read": [],
        "train_and_select_lookup_before_backend": True,
        "no_calibration_lookup_before_all_winner_replays": True,
        "no_report_lookup_before_all_thresholds_frozen": True,
        "calibration_population_lookup_count": 1,
        "report_population_lookup_count": 1,
        "calibration_prediction_pass_count_per_arm": 1,
        "report_prediction_pass_count_per_arm": 1,
    }


def test_formal_entry_requires_real_frozen_plan_before_backend_creation(tmp_path: Path):
    train, validation = _populations()
    plan = _fixture_plan()
    contract = LocalQ1RunnerContract(
        epochs=1,
        initialization_seed=7,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=True,
    )
    provider = _FixtureProvider()
    backend_called = False

    def backend_factory():
        nonlocal backend_called
        backend_called = True
        return _FixtureBackend([])

    with pytest.raises(TypeError, match="FrozenLocalQ1BatchPlan"):
        run_local_q1(
            contract=contract,
            plan=plan,
            train_records=train,
            validation_records=validation,
            provider=provider,
            backend_factory=backend_factory,
            output_root=tmp_path,
        )
    assert backend_called is False
    assert provider.calls == []


def test_plan_hash_and_component_partition_fail_before_model_steps(tmp_path: Path):
    train, validation = _populations()
    plan = _fixture_plan()
    wrong_contract = LocalQ1RunnerContract(
        epochs=1,
        initialization_seed=3,
        batch_plan_file_sha256="f" * 64,
        batch_plan_content_sha256=plan.content_sha256,
        production=False,
    )
    provider = _FixtureProvider()
    with pytest.raises(LocalQ1RunnerError, match="preregistered locks"):
        run_local_q1(
            contract=wrong_contract,
            plan=plan,
            train_records=train,
            validation_records=validation,
            provider=provider,
            backend_factory=lambda: _FixtureBackend([]),
            output_root=tmp_path / "hash",
        )
    assert provider.calls == []

    overlapping = copy.deepcopy(plan.receipt)
    overlapping["phases"]["validation_calibration"]["batches"][0]["record_ordinals"] = [
        0,
        5,
        6,
        7,
    ]
    tainted = SimpleNamespace(
        receipt=overlapping,
        content_sha256=plan.content_sha256,
        canonical_file_sha256=plan.canonical_file_sha256,
    )
    good_contract = LocalQ1RunnerContract(
        epochs=1,
        initialization_seed=3,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=False,
    )
    backend_calls = []
    with pytest.raises(LocalQ1RunnerError, match="safety metadata"):
        run_local_q1(
            contract=good_contract,
            plan=tainted,
            train_records=train,
            validation_records=validation,
            provider=provider,
            backend_factory=lambda: backend_calls.append(True),
            output_root=tmp_path / "overlap",
        )
    assert backend_calls == []
    assert provider.calls == []
    assert provider.planned_calls == []
    assert provider.safety_calls[-1][0] == "validation_calibration"


def test_external_populations_are_audit_only_and_every_field_is_bound(
    tmp_path: Path,
) -> None:
    train, validation = _populations()
    plan = _fixture_plan()
    contract = LocalQ1RunnerContract(
        epochs=1,
        initialization_seed=9,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=False,
    )
    flipped = list(train)
    flipped[0] = replace(flipped[0], label=False, direction_b_wrt_a=None)
    val_component = validation[0].component_id
    cross_component = list(train)
    cross_component[0] = replace(
        cross_component[0],
        fragment_a=replace(cross_component[0].fragment_a, component_id=val_component),
        fragment_b=replace(cross_component[0].fragment_b, component_id=val_component),
        component_id=val_component,
    )
    reordered = tuple(reversed(train))
    provenance_changed = list(train)
    provenance_changed[0] = replace(
        provenance_changed[0],
        provenance={
            **dict(provenance_changed[0].provenance),
            "external_only_marker": True,
        },
    )
    cases = (
        (tuple(flipped), validation, "label"),
        (tuple(cross_component), validation, "fragment_a"),
        (reordered, validation, "record 0 field"),
        (tuple(provenance_changed), validation, "provenance"),
    )
    for index, (supplied_train, supplied_validation, message) in enumerate(cases):
        provider = _FixtureProvider(train, validation, plan)
        backend_calls = []

        def backend_factory():
            backend_calls.append(True)
            return _FixtureBackend([])

        with pytest.raises(LocalQ1RunnerError, match=message):
            run_local_q1(
                contract=contract,
                plan=plan,
                train_records=supplied_train,
                validation_records=supplied_validation,
                provider=provider,
                backend_factory=backend_factory,
                output_root=tmp_path / "external-audit-{}".format(index),
            )
        assert backend_calls == []
        assert provider.calls == []
        assert not (tmp_path / "external-audit-{}".format(index)).exists()


@pytest.mark.parametrize(
    ("provenance_update", "message"),
    (
        ({"real_dunhuang_sealed_test": True}, "sealed"),
        ({"historical_test": True}, "historical-test"),
        ({"source_split": "historical_test"}, "historical-test"),
    ),
)
def test_provider_planned_test_or_sealed_provenance_fails_before_backend(
    tmp_path: Path,
    provenance_update: Mapping[str, Any],
    message: str,
) -> None:
    train, validation = _populations()
    plan = _fixture_plan()
    tainted_train = list(train)
    provenance = dict(tainted_train[0].provenance)
    provenance.update(provenance_update)
    tainted_train[0] = replace(tainted_train[0], provenance=provenance)
    tainted_train = tuple(tainted_train)
    provider = _FixtureProvider(tainted_train, validation, plan)
    contract = LocalQ1RunnerContract(
        epochs=1,
        initialization_seed=10,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=False,
    )
    backend_calls = []
    with pytest.raises(LocalQ1RunnerError, match=message):
        run_local_q1(
            contract=contract,
            plan=plan,
            train_records=tainted_train,
            validation_records=validation,
            provider=provider,
            backend_factory=lambda: backend_calls.append(True),
            output_root=tmp_path / "provider-scope",
        )
    assert backend_calls == []
    assert provider.calls == []
    assert provider.planned_calls == []
    assert provider.safety_calls == [("train", 0)]
    assert not (tmp_path / "provider-scope").exists()


@pytest.mark.parametrize(
    ("validation_index", "expected_last_phase", "provenance_update", "message"),
    (
        (
            4,
            "validation_calibration",
            {"real_dunhuang_sealed_test": True},
            "sealed",
        ),
        (
            8,
            "validation_report",
            {"historical_test": True},
            "historical-test",
        ),
    ),
)
def test_late_phase_scope_fails_in_supervision_blind_pre_session_preflight(
    tmp_path: Path,
    validation_index: int,
    expected_last_phase: str,
    provenance_update: Mapping[str, Any],
    message: str,
) -> None:
    train, validation = _populations()
    plan = _fixture_plan()
    tainted_validation = list(validation)
    provenance = dict(tainted_validation[validation_index].provenance)
    provenance.update(provenance_update)
    tainted_validation[validation_index] = replace(
        tainted_validation[validation_index], provenance=provenance
    )
    tainted_validation = tuple(tainted_validation)
    provider = _FixtureProvider(train, tainted_validation, plan)
    contract = LocalQ1RunnerContract(
        epochs=1,
        initialization_seed=13,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=False,
    )
    backend_calls = []

    def backend_factory():
        backend_calls.append(tuple(provider.planned_calls))
        return _FixtureBackend([])

    with pytest.raises(LocalQ1RunnerError, match=message):
        run_local_q1(
            contract=contract,
            plan=plan,
            train_records=train,
            validation_records=tainted_validation,
            provider=provider,
            backend_factory=backend_factory,
            output_root=tmp_path / expected_last_phase,
        )
    assert backend_calls == []
    assert provider.planned_calls == []
    assert provider.calls == []
    assert provider.safety_calls[-1][0] == expected_last_phase
    assert not (tmp_path / expected_last_phase).exists()


def _recursive_provenance_attack(
    case: str, base: Mapping[str, Any]
) -> Mapping[str, Any]:
    provenance = dict(base)
    if case == "nested_mapping":
        provenance["nested"] = {"historical_test": True}
    elif case == "nested_list":
        provenance["nested"] = [{"Sealed_Holdout": True}]
    elif case == "case_alias":
        provenance["Historical_Test"] = True
    elif case == "compound_alias":
        provenance["historical_eval_test"] = 1
    elif case == "source_split":
        provenance["nested"] = {"source_split": "Historical_Test"}
    elif case == "nested_exact_sealed":
        provenance["nested"] = [{"real_dunhuang_sealed_test": True}]
    elif case == "cycle":
        cycle = {}
        cycle["child"] = cycle
        provenance["nested"] = cycle
    elif case == "too_deep":
        nested: Any = False
        for _ in range(LOCAL_Q1_PROVENANCE_SAFETY_MAX_DEPTH + 2):
            nested = {"child": nested}
        provenance["nested"] = nested
    elif case == "too_many":
        provenance["nested"] = [None] * LOCAL_Q1_PROVENANCE_SAFETY_MAX_NODES
    elif case == "unsupported":
        provenance["nested"] = {"unsupported"}
    else:  # pragma: no cover - parametrization owns the closed set
        raise AssertionError("unknown recursive provenance attack")
    return provenance


@pytest.mark.parametrize(
    ("case", "validation_index", "expected_phase"),
    (
        ("nested_mapping", 4, "validation_calibration"),
        ("nested_list", 8, "validation_report"),
        ("case_alias", 4, "validation_calibration"),
        ("compound_alias", 8, "validation_report"),
        ("source_split", 4, "validation_calibration"),
        ("nested_exact_sealed", 8, "validation_report"),
        ("cycle", 4, "validation_calibration"),
        ("too_deep", 8, "validation_report"),
        ("too_many", 4, "validation_calibration"),
        ("unsupported", 8, "validation_report"),
    ),
)
def test_recursive_provenance_attacks_fail_before_backend_session_output_or_prepare(
    tmp_path: Path,
    case: str,
    validation_index: int,
    expected_phase: str,
) -> None:
    train, validation = _populations()
    plan = _fixture_plan()
    tainted_validation = list(validation)
    target = tainted_validation[validation_index]
    tainted_validation[validation_index] = replace(
        target,
        provenance=_recursive_provenance_attack(case, target.provenance),
    )
    provider = _FixtureProvider(train, tuple(tainted_validation), plan)
    contract = LocalQ1RunnerContract(
        epochs=1,
        initialization_seed=15,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=False,
    )
    backend_calls = []
    session_log = []

    def backend_factory():
        backend_calls.append(True)
        return _FixtureBackend(session_log)

    with pytest.raises(
        LocalQ1RunnerError,
        match="safety metadata",
    ):
        run_local_q1(
            contract=contract,
            plan=plan,
            train_records=train,
            validation_records=tuple(tainted_validation),
            provider=provider,
            backend_factory=backend_factory,
            output_root=tmp_path / case,
        )
    assert backend_calls == []
    assert session_log == []
    assert provider.planned_calls == []
    assert provider.calls == []
    assert provider.safety_attempts[-1][0] == expected_phase
    assert not (tmp_path / case).exists()


@pytest.mark.parametrize(
    ("validation_index", "conflicting_component", "expected_phase"),
    (
        (4, "train", "validation_calibration"),
        (8, "validation_select", "validation_report"),
    ),
)
def test_late_phase_component_overlap_fails_before_backend_session_or_output(
    tmp_path: Path,
    validation_index: int,
    conflicting_component: str,
    expected_phase: str,
) -> None:
    train, validation = _populations()
    plan = _fixture_plan()
    component = (
        train[0].component_id
        if conflicting_component == "train"
        else validation[0].component_id
    )
    tainted_validation = list(validation)
    record = tainted_validation[validation_index]
    tainted_validation[validation_index] = replace(
        record,
        fragment_a=replace(record.fragment_a, component_id=component),
        fragment_b=replace(record.fragment_b, component_id=component),
        component_id=component,
    )
    provider = _FixtureProvider(train, tuple(tainted_validation), plan)
    contract = LocalQ1RunnerContract(
        epochs=1,
        initialization_seed=14,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=False,
    )
    backend_calls = []
    with pytest.raises(LocalQ1RunnerError, match="safety metadata"):
        run_local_q1(
            contract=contract,
            plan=plan,
            train_records=train,
            validation_records=tuple(tainted_validation),
            provider=provider,
            backend_factory=lambda: backend_calls.append(True),
            output_root=tmp_path / expected_phase,
        )
    assert backend_calls == []
    assert provider.planned_calls == []
    assert provider.calls == []
    assert provider.safety_calls[-1][0] == expected_phase
    assert not (tmp_path / expected_phase).exists()


def test_provider_planned_train_component_cannot_enter_validation(
    tmp_path: Path,
) -> None:
    train, validation = _populations()
    plan = _fixture_plan()
    tainted_train = list(train)
    validation_component = validation[0].component_id
    tainted_train[0] = replace(
        tainted_train[0],
        fragment_a=replace(
            tainted_train[0].fragment_a,
            component_id=validation_component,
        ),
        fragment_b=replace(
            tainted_train[0].fragment_b,
            component_id=validation_component,
        ),
        component_id=validation_component,
    )
    tainted_train = tuple(tainted_train)
    provider = _FixtureProvider(tainted_train, validation, plan)
    contract = LocalQ1RunnerContract(
        epochs=1,
        initialization_seed=11,
        batch_plan_file_sha256=plan.canonical_file_sha256,
        batch_plan_content_sha256=plan.content_sha256,
        production=False,
    )
    backend_calls = []
    with pytest.raises(LocalQ1RunnerError, match="components overlap"):
        run_local_q1(
            contract=contract,
            plan=plan,
            train_records=tainted_train,
            validation_records=validation,
            provider=provider,
            backend_factory=lambda: backend_calls.append(True),
            output_root=tmp_path / "provider-component-overlap",
        )
    assert backend_calls == []
    assert provider.calls == []
    assert provider.planned_calls == []
    assert provider.safety_calls == [
        ("train", 0),
        ("validation_select", 0),
        ("validation_calibration", 0),
        ("validation_report", 0),
    ]
    assert not (tmp_path / "provider-component-overlap").exists()


class _ScalarFakeFormalBackend:
    def __init__(self) -> None:
        self.contract = BackendContract(
            execution_kind=ExecutionKind.SYNTHETIC_TRAIN_VALIDATION,
            backend_version=(
                LOCAL_Q1_BACKEND_VERSION + "/" + LocalQ1BackendMode.FORMAL.value
            ),
            model_family="DunhuangPairwiseV02-LOCAL-Q1-five-arm",
            device_type="cpu",
        )
        self.session_create_count = 0

    def create_session(self, *_args, **_kwargs):
        self.session_create_count += 1
        return _FixtureModel()


def test_scalar_fake_formal_backend_fails_before_session_or_mutation() -> None:
    fake = _ScalarFakeFormalBackend()
    with pytest.raises(LocalQ1RunnerError, match="exact LocalQ1Backend"):
        _checked_backend(fake, production=True)
    assert fake.session_create_count == 0

    real = LocalQ1Backend(device="cpu", mode=LocalQ1BackendMode.FORMAL)
    assert _checked_backend(real, production=True) is real


def test_fixture_authority_is_not_promotable_and_tamper_is_rejected(tmp_path: Path):
    artifacts, _provider, _sessions, _contract, plan = _run_case(tmp_path)
    arm = AblationArmName.LOCAL_DUSTBIN_SINKHORN.value
    authority_path = artifacts.authority_receipt_paths[arm]
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority_file_sha = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    checkpoint = artifacts.checkpoint_paths[arm][1]

    with pytest.raises(LocalQ1RunnerError, match="not result-eligible"):
        load_validated_local_checkpoint_binding(
            authority_receipt_path=authority_path,
            checkpoint_path=checkpoint,
            expected_authority_file_sha256=authority_file_sha,
            expected_authority_content_sha256=authority["content_sha256"],
            expected_batch_plan_file_sha256=plan.canonical_file_sha256,
            expected_batch_plan_content_sha256=plan.content_sha256,
            require_dustbin=True,
        )

    with pytest.raises(LocalQ1RunnerError, match="external file hash mismatch"):
        load_validated_local_checkpoint_binding(
            authority_receipt_path=authority_path,
            checkpoint_path=checkpoint,
            expected_authority_file_sha256="0" * 64,
            expected_authority_content_sha256=authority["content_sha256"],
            expected_batch_plan_file_sha256=plan.canonical_file_sha256,
            expected_batch_plan_content_sha256=plan.content_sha256,
            require_dustbin=True,
        )

    resigned = copy.deepcopy(authority)
    resigned["result_eligible"] = True
    resigned.pop("content_sha256")
    resigned["content_sha256"] = _sha(resigned)
    resigned_path = tmp_path / "self-resigned-fixture.authority.json"
    resigned_payload = _canonical(resigned)
    resigned_path.write_bytes(resigned_payload)
    resigned_file_sha = hashlib.sha256(resigned_payload).hexdigest()
    with pytest.raises(
        LocalQ1RunnerError,
        match="provider phase authority changed|runner contract is not formal local-only",
    ):
        load_validated_local_checkpoint_binding(
            authority_receipt_path=resigned_path,
            checkpoint_path=checkpoint,
            expected_authority_file_sha256=resigned_file_sha,
            expected_authority_content_sha256=resigned["content_sha256"],
            expected_batch_plan_file_sha256=plan.canonical_file_sha256,
            expected_batch_plan_content_sha256=plan.content_sha256,
            require_dustbin=True,
        )


def test_fused_stage_is_a_separate_two_authority_fail_closed_entry() -> None:
    with pytest.raises(LocalQ1BackendError, match="C0 and dustbin authorities"):
        require_fused_stage_authorities(coarse=None, dustbin=None)
    with pytest.raises(LocalQ1BackendError, match="C0 and dustbin authorities"):
        require_fused_stage_authorities(coarse=None, dustbin=object())


def test_runner_does_not_import_generic_short_ablation_scheduler() -> None:
    source = (Path(__file__).parents[1] / "training" / "local_q1_runner.py").read_text(
        encoding="utf-8"
    )
    assert "run_short_ablation" not in source
    assert "_epoch_order" not in source
    assert "_select_population" not in source
