from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
import torch
from torch import nn

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    HISTORICAL_TEST_ACCESS_EVIDENCE,
)
from staging.pairwise_v0_2.pairwise_data.sampling import (
    validation_stream_fingerprint,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training import c0_runner as module
from staging.pairwise_v0_2.training.c0_runner import (
    BackendAttestation,
    C0PredictionBatch,
    C0PreparedBatch,
    C0RunnerContract,
    C0RunnerError,
    C0TrainBatchResult,
    FrozenReceiptLock,
    ProviderAttestation,
    SourceFileLock,
    run_c0_n_q1,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(*, dataset: str, split: str, label: bool, index: int) -> TrainingPairRecord:
    binding = (
        MM_CANONICAL_BINDING if dataset == "mm_augmented" else ECCV_CANONICAL_BINDING
    )
    short = "mm" if dataset == "mm_augmented" else "eccv"
    group = "{}/group/{}/{}".format(short, split, index)
    component = "{}/component/{}/{}".format(short, split, index)

    def fragment(side: str, content_offset: int) -> MaskMemberRef:
        member = "fixture/{}/{}/{}/{}/{}.png".format(
            short, split, int(label), index, side
        )
        return MaskMemberRef(
            binding=binding,
            archive_member=member,
            fragment_id=member[:-4],
            dataset_id=dataset,
            canonical_group_id=group,
            component_id=component,
            split=split,
            threshold_rule="binary_brighter_value",
            content_sha256="{:064x}".format(
                10_000
                + (0 if split == "train" else 1_000)
                + (0 if dataset == "mm_augmented" else 100)
                + int(label) * 20
                + index * 2
                + content_offset
            ),
        )

    fragment_a = fragment("a", 0)
    fragment_b = fragment("b", 1)
    return TrainingPairRecord(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        label=label,
        direction_b_wrt_a="right" if label else None,
        dataset_id=dataset,
        canonical_group_id=group,
        component_id=component,
        split=split,
        canonical_pair_key=tuple(
            sorted((fragment_a.fragment_id, fragment_b.fragment_id))
        ),
        label_origin="fixture_historical_explicit",
        provenance={"real_dunhuang_sealed_test": False},
    )


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage_counter = nn.Parameter(torch.zeros(()), requires_grad=False)


class _TinySession:
    def __init__(self, events: list[str], batches_per_epoch: int, purpose: str) -> None:
        self.model = _TinyModel()
        self.optimizer = None
        self.model_config = {"family": "tiny-c0-fixture", "width": 1}
        self.optimizer_config = {"kind": "none"}
        self._events = events
        self._batches_per_epoch = batches_per_epoch
        self._events.append("session:" + purpose)

    def train_batch(self, batch: C0PreparedBatch) -> C0TrainBatchResult:
        with torch.no_grad():
            self.model.stage_counter.add_(1.0)
        self._events.append("train_batch")
        return C0TrainBatchResult(
            loss=1.0 / (1.0 + float(self.model.stage_counter.item())),
            valid_count=batch.sample_count,
        )

    def predict_batch(self, batch: C0PreparedBatch) -> C0PredictionBatch:
        stage = int(self.model.stage_counter.item()) // self._batches_per_epoch
        records: Sequence[TrainingPairRecord] = batch.payload
        if stage <= 1:
            probability = tuple(0.5 for _record_value in records)
        else:
            probability = tuple(0.8 if record.label else 0.2 for record in records)
        self._events.append("predict_batch")
        return C0PredictionBatch(
            probability=probability,
            valid=tuple(True for _record_value in records),
        )


class _TinyBackend:
    def __init__(self, events: list[str], batches_per_epoch: int) -> None:
        self.attestation = BackendAttestation(
            backend_version="tiny-backend/1",
            model_family="tiny-c0-fixture",
            device_type="cpu",
        )
        self._events = events
        self._batches_per_epoch = batches_per_epoch

    def create_session(self, *, seed: int, purpose: str) -> _TinySession:
        assert seed >= 0
        return _TinySession(self._events, self._batches_per_epoch, purpose)


class _TinyProvider:
    def __init__(
        self,
        events: list[str],
        phases: list[str],
        archive_locks_sha256: str,
    ) -> None:
        self.attestation = ProviderAttestation(
            provider_version="tiny-provider/1",
            preprocessing_sha256="a" * 64,
            archive_locks_sha256=archive_locks_sha256,
            archives_verified=True,
        )
        self._events = events
        self._phases = phases

    def prepare(
        self, records: Sequence[TrainingPairRecord], *, phase: str
    ) -> C0PreparedBatch:
        values = tuple(records)
        sequence = module._sequence_commit(values)
        self._phases.append(phase)
        return C0PreparedBatch(
            payload=values,
            sample_count=len(values),
            record_sequence_sha256=sequence,
            prepared_input_sha256=hashlib.sha256(
                ("prepared:" + sequence).encode("ascii")
            ).hexdigest(),
        )


class _Fixture:
    def __init__(self, tmp_path: Path) -> None:
        self.events: list[str] = []
        self.phases: list[str] = []
        self.train = tuple(
            _record(dataset=dataset, split="train", label=label, index=index)
            for dataset in ("mm_augmented", "eccv_1113data")
            for label in (False, True)
            for index in range(2)
        )
        self.validation = tuple(
            _record(dataset=dataset, split="val", label=label, index=index)
            for dataset in ("mm_augmented", "eccv_1113data")
            for label in (False, True)
            for index in range(2)
        )
        validation_fingerprint = validation_stream_fingerprint(self.validation)[
            "sha256"
        ]
        self.source = tmp_path / "locked_source.py"
        self.source.write_text("VALUE = 1\n", encoding="utf-8")
        self.contract = C0RunnerContract(
            epochs=5,
            batch_size=2,
            train_count=8,
            train_per_dataset_label=2,
            validation_count=8,
            validation_fingerprint=validation_fingerprint,
            seed="tiny-260828",
            source_locks=(
                SourceFileLock(
                    logical_id="source://tiny-runner",
                    path=self.source,
                    sha256=_sha256_file(self.source),
                ),
            ),
            production=False,
        )
        archive_locks = {
            "mm_augmented": {
                "format": "zip",
                "sha256": MM_CANONICAL_BINDING.sha256,
            },
            "eccv_1113data": {
                "format": "tar",
                "sha256": ECCV_CANONICAL_BINDING.sha256,
            },
        }
        freeze: dict[str, Any] = {
            "schema_version": "dunhuang-pairwise-c0-n-q1-freeze/0.3",
            "status": "pass_metadata_only_no_model_execution",
            "scope": {
                "experiment": "C0-N-Q1",
                "datasets": ["mm_augmented", "eccv_1113data"],
                "pair_stream_splits_read": ["train", "val"],
                "historical_test_access": dict(HISTORICAL_TEST_ACCESS_EVIDENCE),
                "sealed_real_read": False,
                "mask_pixels_decoded": False,
                "model_executed": False,
            },
            "locks": {"archives": archive_locks},
            "validation": {
                "count": 8,
                "frozen_order_sha256": validation_fingerprint,
                "record_sequence_commitment_sha256": module._ordered_commit(
                    module._record_token(record) for record in self.validation
                ),
            },
            "training": {
                "selection": {
                    "count": 8,
                    "target_per_dataset_label": 2,
                    "max_per_component_label": 32,
                    "record_order_commitment_sha256": module._ordered_commit(
                        module._record_token(record) for record in self.train
                    ),
                    "record_set_commitment_sha256": module._set_commit(
                        module._record_token(record) for record in self.train
                    ),
                }
            },
            "selected_train_vs_full_validation_overlap": {
                "component": 0,
                "member": 0,
                "content_sha256": 0,
            },
        }
        freeze["content_sha256"] = module._content_sha256(freeze)
        self.freeze_path = tmp_path / "freeze.json"
        self.freeze_path.write_text(
            json.dumps(freeze, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        self.freeze_lock = FrozenReceiptLock(
            path=self.freeze_path,
            file_sha256=_sha256_file(self.freeze_path),
            content_sha256=freeze["content_sha256"],
        )
        self.archive_locks_sha256 = module._content_sha256(archive_locks)

    def provider_factory(self) -> _TinyProvider:
        self.events.append("provider_factory")
        return _TinyProvider(self.events, self.phases, self.archive_locks_sha256)

    def backend_factory(self) -> _TinyBackend:
        self.events.append("backend_factory")
        return _TinyBackend(self.events, self.contract.batches_per_epoch)

    def run(
        self,
        output: Path,
        *,
        train: Sequence[TrainingPairRecord] | None = None,
        validation: Sequence[TrainingPairRecord] | None = None,
    ):
        return run_c0_n_q1(
            run_plan=self.contract,
            freeze_receipt=self.freeze_lock,
            selected_train_records=self.train if train is None else train,
            validation_records=(self.validation if validation is None else validation),
            provider_factory=self.provider_factory,
            backend_factory=self.backend_factory,
            output_dir=output,
        )


def test_five_epochs_checkpoints_selector_reload_and_diagnostic_thresholds(
    tmp_path: Path,
) -> None:
    fixture = _Fixture(tmp_path)
    artifacts = fixture.run(tmp_path / "run")
    receipt = artifacts.receipt
    assert receipt["observed"] == {
        "epoch_count": 5,
        "checkpoint_count": 5,
        "batches_per_epoch": 4,
        "total_steps": 20,
        "full_validation_replay_count": 6,
    }
    assert len(receipt["epochs"]) == 5
    assert len(list(artifacts.run_directory.glob("checkpoint-epoch-*.pt"))) == 5
    assert receipt["winner"]["epoch"] == 2
    assert receipt["winner"]["restricted_fresh_session_reload"] is True
    assert receipt["winner"]["full_validation_replay_match"] is True
    environment = receipt["environment"]
    assert set(environment) == {
        "schema_version",
        "python",
        "torch",
        "cuda",
        "cudnn",
        "device",
        "determinism",
        "os",
    }
    assert environment["schema_version"] == (
        "dunhuang-pairwise-c0-runtime-environment/0.4"
    )
    assert environment["device"] == {
        "type": "cpu",
        "index": None,
        "name": None,
        "compute_capability": None,
        "total_memory_bytes": None,
    }
    assert environment["cuda"]["driver"] == {
        "status": "not_applicable",
        "version": None,
    }
    assert environment["determinism"] == {
        "cublas_workspace_config": {
            "status": "not_applicable",
            "value": None,
        }
    }
    assert receipt["environment_sha256"] == module._content_sha256(environment)
    assert receipt["winner"]["environment_sha256"] == receipt["environment_sha256"]
    environment_text = json.dumps(environment, sort_keys=True).casefold()
    for forbidden in ("hostname", "environment_variable", "/users/", "\\users\\"):
        assert forbidden not in environment_text
    assert fixture.events[:3] == [
        "provider_factory",
        "backend_factory",
        "session:train",
    ]
    assert fixture.events.count("backend_factory") == 2
    assert "session:winner_reload" in fixture.events
    assert receipt["thresholds"]["production_threshold"] is None
    assert receipt["thresholds"]["pooled_threshold"] is None
    assert set(receipt["thresholds"]["by_dataset"]) == {
        "mm_augmented",
        "eccv_1113data",
    }
    assert receipt["thresholds"]["status"].endswith("after_winner_selection")
    assert sum(phase.startswith("validation-epoch-") for phase in fixture.phases) == 20
    assert (
        sum(phase.startswith("winner-reload-validation") for phase in fixture.phases)
        == 4
    )
    receipt_text = artifacts.receipt_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in receipt_text
    sidecar = json.loads(artifacts.checkpoint_sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["winner_epoch"] == 2
    assert str(tmp_path) in sidecar["winner_path"]


def test_cuda_environment_records_portable_gpu_and_driver_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SimpleNamespace(
        attestation=BackendAttestation(
            backend_version="fixture-backend/0.1",
            model_family="fixture-model",
            device_type="cuda",
        )
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name="Fixture GPU", total_memory=24_000_000_000),
    )
    monkeypatch.setattr(
        module.torch.cuda, "get_device_capability", lambda index: (8, 9)
    )

    def fake_run(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert argv == [
            "nvidia-smi",
            "--id=2",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 5,
        }
        return SimpleNamespace(stdout="550.54.14\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    environment = module._runtime_environment(backend)
    assert environment["device"] == {
        "type": "cuda",
        "index": 2,
        "name": "Fixture GPU",
        "compute_capability": {"major": 8, "minor": 9},
        "total_memory_bytes": 24_000_000_000,
    }
    assert environment["cuda"]["driver"] == {
        "status": "observed",
        "version": "550.54.14",
    }
    assert environment["determinism"] == {
        "cublas_workspace_config": {
            "status": "validated",
            "value": ":4096:8",
        }
    }


def test_cuda_environment_records_driver_query_failure_without_local_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SimpleNamespace(
        attestation=BackendAttestation(
            backend_version="fixture-backend/0.1",
            model_family="fixture-model",
            device_type="cuda",
        )
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name="Fixture GPU", total_memory=1),
    )
    monkeypatch.setattr(
        module.torch.cuda, "get_device_capability", lambda index: (7, 5)
    )

    def missing_nvidia_smi(argv: list[str], **kwargs: Any) -> None:
        raise FileNotFoundError("local executable detail must not enter receipt")

    monkeypatch.setattr(module.subprocess, "run", missing_nvidia_smi)
    environment = module._runtime_environment(backend)
    assert environment["cuda"]["driver"] == {
        "status": "nvidia_smi_not_found",
        "version": None,
    }
    assert "local executable detail" not in json.dumps(environment)


@pytest.mark.parametrize("configured_value", (None, "", ":8192:8"))
def test_cuda_cublas_config_fails_before_session_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str | None,
) -> None:
    fixture = _Fixture(tmp_path)
    if configured_value is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", configured_value)

    def cuda_backend_factory() -> _TinyBackend:
        fixture.events.append("backend_factory")
        backend = _TinyBackend(fixture.events, fixture.contract.batches_per_epoch)
        backend.attestation = replace(backend.attestation, device_type="cuda")
        return backend

    monkeypatch.setattr(fixture, "backend_factory", cuda_backend_factory)
    output = tmp_path / "run"
    with pytest.raises(C0RunnerError, match="CUBLAS_WORKSPACE_CONFIG"):
        fixture.run(output)
    assert fixture.events == ["provider_factory", "backend_factory"]
    assert not output.exists()


def test_freeze_file_tamper_fails_before_factories(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    fixture.freeze_path.write_text(
        fixture.freeze_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(C0RunnerError, match="file SHA-256 mismatch"):
        fixture.run(tmp_path / "run")
    assert fixture.events == []


def test_source_lock_tamper_fails_before_factories(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    fixture.source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(C0RunnerError, match="source lock mismatch"):
        fixture.run(tmp_path / "run")
    assert fixture.events == []


def test_metadata_commitment_tamper_stops_before_both_factories(
    tmp_path: Path,
) -> None:
    fixture = _Fixture(tmp_path)
    first = fixture.train[0]
    changed_fragment = replace(first.fragment_a, content_sha256="f" * 64)
    tampered = (replace(first, fragment_a=changed_fragment),) + fixture.train[1:]
    with pytest.raises(C0RunnerError, match="commitment changed"):
        fixture.run(tmp_path / "run", train=tampered)
    assert fixture.events == []


def test_sealed_real_marker_is_refused_before_both_factories(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    first = fixture.validation[0]
    refused = replace(first, provenance={"real_dunhuang_sealed_test": True})
    validation = (refused,) + fixture.validation[1:]
    with pytest.raises(C0RunnerError, match="sealed-real exclusion"):
        fixture.run(tmp_path / "run", validation=validation)
    assert fixture.events == []


def test_production_constants_cannot_be_relaxed() -> None:
    with pytest.raises(C0RunnerError, match="production C0 constant changed"):
        C0RunnerContract(epochs=4, source_locks=())


def test_production_final_provider_receipt_binds_exact_access_schedule() -> None:
    contract = C0RunnerContract()
    freeze = {
        "content_sha256": "e" * 64,
        "locks": {
            "archives": {
                "mm_augmented": {
                    "format": "zip",
                    "bytes": 11,
                    "sha256": "a" * 64,
                },
                "eccv_1113data": {
                    "format": "tar",
                    "bytes": 13,
                    "sha256": "b" * 64,
                },
            }
        },
    }
    counters = {
        "batch_count": 6_722,
        "record_count": 1_720_196,
        "fragment_request_count": 3_440_392,
        "memo_eviction_count": 0,
    }
    memo_bounds = {
        "max_batch_size": 256,
        "max_cached_fragments": 131_072,
        "max_cache_bytes": 8 * 1024**3,
    }

    class Provider:
        def final_receipt(self):
            return {
                "freeze_content_sha256": "e" * 64,
                "identity_index_content_sha256": "d" * 64,
                "historical_test_access": dict(HISTORICAL_TEST_ACCESS_EVIDENCE),
                "sealed_real_read": False,
                "geometry_cache_local_sinkhorn_calls": 0,
                "archive_verification": [
                    {
                        "archive_format": row["format"],
                        "expected_sha256": row["sha256"],
                        "observed_sha256": row["sha256"],
                        "byte_count": row["bytes"],
                    }
                    for row in freeze["locks"]["archives"].values()
                ],
                "preprocessing": {
                    "preprocessing_sha256": "c" * 64,
                    "geometry_config_sha256": "f" * 64,
                },
                "memo_bounds": memo_bounds,
                "counters": counters,
            }

    evidence = module._validate_final_provider_evidence(
        Provider(),
        contract=contract,
        freeze=freeze,
        expected_identity_index_content_sha256="d" * 64,
    )
    assert evidence["counters"] == counters
    original_cache_bytes = memo_bounds["max_cache_bytes"]
    memo_bounds["max_cache_bytes"] -= 1
    with pytest.raises(C0RunnerError, match="memo bounds"):
        module._validate_final_provider_evidence(
            Provider(),
            contract=contract,
            freeze=freeze,
            expected_identity_index_content_sha256="d" * 64,
        )
    memo_bounds["max_cache_bytes"] = original_cache_bytes
    counters["memo_eviction_count"] = 1
    with pytest.raises(C0RunnerError, match="memo evicted"):
        module._validate_final_provider_evidence(
            Provider(),
            contract=contract,
            freeze=freeze,
            expected_identity_index_content_sha256="d" * 64,
        )
    counters["memo_eviction_count"] = 0
    counters["record_count"] -= 1
    with pytest.raises(C0RunnerError, match="access counts"):
        module._validate_final_provider_evidence(
            Provider(),
            contract=contract,
            freeze=freeze,
            expected_identity_index_content_sha256="d" * 64,
        )
