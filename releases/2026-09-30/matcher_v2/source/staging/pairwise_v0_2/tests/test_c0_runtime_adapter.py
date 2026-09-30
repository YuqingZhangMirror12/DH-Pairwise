from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import pytest
import torch
from torch import nn

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    HISTORICAL_TEST_ACCESS_EVIDENCE,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MM_CANONICAL_BINDING,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.c0_coarse_backend import (
    C0_COARSE_BACKEND_VERSION,
    C0_COARSE_MODEL_CONFIG,
    C0_COARSE_OPTIMIZER_CONFIG,
    C0CoarseBackend,
)
from staging.pairwise_v0_2.training.c0_coarse_provider import (
    C0_COARSE_PROVIDER_RECEIPT_VERSION,
    C0_COARSE_PROVIDER_VERSION,
    C0CoarseProvider,
)
from staging.pairwise_v0_2.training.c0_runner import (
    C0RunnerError,
    c0_runner_session_seed,
)
from staging.pairwise_v0_2.training.c0_runtime_adapter import (
    C0CoarseBackendAdapter,
    C0CoarseProviderAdapter,
)
from staging.pairwise_v0_2.training.short_ablation import (
    BackendContract,
    BatchProviderContract,
    EvidenceMode,
    ExecutionKind,
    PredictionBatch,
    PreparedAblationBatch,
    TrainBatchResult,
    record_sequence_fingerprint,
)


_FREEZE_SHA = "e" * 64
_INDEX_SHA = "d" * 64
_PREPROCESS_SHA = "c" * 64
_GEOMETRY_SHA = "b" * 64
_ARCHIVE_LOCKS = {
    "mm_augmented": {"format": "zip", "bytes": 11, "sha256": "a" * 64},
    "eccv_1113data": {"format": "tar", "bytes": 13, "sha256": "f" * 64},
}


def _record(index: int) -> TrainingPairRecord:
    group = "fixture/group/{}".format(index)
    component = "fixture/component/{}".format(index)

    def fragment(side: str) -> MaskMemberRef:
        member = "fixture/{}/{}.png".format(index, side)
        return MaskMemberRef(
            binding=MM_CANONICAL_BINDING,
            archive_member=member,
            fragment_id=member[:-4],
            dataset_id="mm_augmented",
            canonical_group_id=group,
            component_id=component,
            split="train",
            threshold_rule="binary_brighter_value",
            content_sha256=hashlib.sha256(member.encode()).hexdigest(),
        )

    a, b = fragment("a"), fragment("b")
    return TrainingPairRecord(
        fragment_a=a,
        fragment_b=b,
        label=bool(index % 2),
        direction_b_wrt_a="right" if index % 2 else None,
        dataset_id="mm_augmented",
        canonical_group_id=group,
        component_id=component,
        split="train",
        canonical_pair_key=tuple(sorted((a.fragment_id, b.fragment_id))),
        label_origin="historical_explicit_csv",
        provenance={"real_dunhuang_sealed_test": False},
    )


class _FakeRealProvider(C0CoarseProvider):
    """No archives: exact real API/type surface for adapter integration."""

    def __init__(self) -> None:
        self.contract = BatchProviderContract(
            coarse_preprocess_mode="tight_crop_letterbox",
            coarse_preprocessing_sha256=_PREPROCESS_SHA,
            geometry_config_sha256=_GEOMETRY_SHA,
            cache_interface="bounded_host_coarse_tensor_lru_no_geometry_cache",
            provider_version=C0_COARSE_PROVIDER_VERSION,
            coarse_only_geometry_free=True,
        )
        self.calls: list[tuple[str, Any]] = []
        self.batch_count = 0
        self.record_count = 0

    def receipt(self) -> Mapping[str, Any]:
        return {
            "schema_version": C0_COARSE_PROVIDER_RECEIPT_VERSION,
            "provider_version": C0_COARSE_PROVIDER_VERSION,
            "status": "archive_preflight_complete",
            "freeze_content_sha256": _FREEZE_SHA,
            "identity_index_content_sha256": _INDEX_SHA,
            "archive_verification": [
                {
                    "logical_id": "canonical://fixture/{}".format(dataset),
                    "archive_format": lock["format"],
                    "expected_sha256": lock["sha256"],
                    "observed_sha256": lock["sha256"],
                    "byte_count": lock["bytes"],
                    "verification_mode": "fixture_bytes",
                }
                for dataset, lock in sorted(_ARCHIVE_LOCKS.items())
            ],
            "preprocessing": {
                "preprocessing_sha256": _PREPROCESS_SHA,
                "geometry_config_sha256": _GEOMETRY_SHA,
            },
            "memo_bounds": {
                "max_batch_size": 256,
                "max_cached_fragments": 4,
                "max_cache_bytes": 1024,
            },
            "counters": {
                "batch_count": self.batch_count,
                "record_count": self.record_count,
                "fragment_request_count": 2 * self.record_count,
                "memo_hit_count": 0,
                "memo_miss_count": 2 * self.record_count,
                "mask_loader_call_count": 2 * self.record_count,
                "archive_decode_count": 2 * self.record_count,
                "loader_cache_hit_count": 0,
                "coarse_preprocess_count": 2 * self.record_count,
                "memo_eviction_count": 0,
                "memo_entry_count": 0,
                "memo_byte_count": 0,
            },
            "geometry_cache_local_sinkhorn_calls": 0,
            "historical_test_access": dict(HISTORICAL_TEST_ACCESS_EVIDENCE),
            "sealed_real_read": False,
        }

    def prepare(self, records, *, arm, phase):
        self.calls.append((phase, arm))
        self.batch_count += 1
        self.record_count += len(records)
        sequence = record_sequence_fingerprint(records)
        return PreparedAblationBatch(
            payload={"records": tuple(records)},
            sample_count=len(records),
            record_sequence_sha256=sequence,
            prepared_input_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
            local_candidate_sha256=None,
            coarse_preprocessing_sha256=_PREPROCESS_SHA,
            geometry_config_sha256=None,
            processing_counts={
                "mask_load_count": 2 * len(records),
                "coarse_preprocess_count": 2 * len(records),
                "geometry_build_count": 0,
                "geometry_cache_read_count": 0,
                "geometry_cache_write_count": 0,
                "local_candidate_count": 0,
            },
        )


class _FakeRealSession:
    def __init__(self) -> None:
        self.model = nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.model_config = dict(C0_COARSE_MODEL_CONFIG)
        self.optimizer_config = dict(C0_COARSE_OPTIMIZER_CONFIG)
        self.events: list[str] = []

    def train_batch(self, batch: PreparedAblationBatch) -> TrainBatchResult:
        self.events.append("train")
        return TrainBatchResult(loss=0.25, valid_count=batch.sample_count)

    def predict_batch(
        self, batch: PreparedAblationBatch, *, evidence: EvidenceMode
    ) -> PredictionBatch:
        assert evidence is EvidenceMode.COARSE
        self.events.append("predict")
        return PredictionBatch(
            probability=torch.full((batch.sample_count,), 0.75),
            valid=torch.ones(batch.sample_count, dtype=torch.bool),
        )


class _FakeRealBackend(C0CoarseBackend):
    def __init__(self) -> None:
        self.contract = BackendContract(
            execution_kind=ExecutionKind.SYNTHETIC_TRAIN_VALIDATION,
            backend_version=C0_COARSE_BACKEND_VERSION,
            model_family="SymmetricCoarseSiamese-C0-N-Q1",
            device_type="cpu",
            sealed_real_test_capability=False,
        )
        self.calls: list[tuple[Any, int]] = []
        self.session = _FakeRealSession()

    def create_session(self, arm, *, seed):
        self.calls.append((arm, seed))
        return self.session


def _provider_adapter(provider: C0CoarseProvider) -> C0CoarseProviderAdapter:
    return C0CoarseProviderAdapter(
        provider,
        archive_locks=_ARCHIVE_LOCKS,
        expected_freeze_content_sha256=_FREEZE_SHA,
        expected_identity_index_content_sha256=_INDEX_SHA,
    )


def test_real_api_phase_mapping_and_provider_output_conversion() -> None:
    real = _FakeRealProvider()
    adapter = _provider_adapter(real)
    records: Sequence[TrainingPairRecord] = (_record(0), _record(1))
    train = adapter.prepare(records, phase="train-epoch-3-batch-7")
    validation = adapter.prepare(records, phase="validation-epoch-4-batch-2")
    reload_batch = adapter.prepare(records, phase="winner-reload-validation-batch-0")
    assert [phase for phase, _arm in real.calls] == [
        "train",
        "validation",
        "validation",
    ]
    assert (
        train.sample_count == validation.sample_count == reload_batch.sample_count == 2
    )
    assert all(call[1].uses_local_geometry is False for call in real.calls)
    assert adapter.final_receipt()["counters"]["batch_count"] == 3
    with pytest.raises(C0RunnerError, match="cannot be mapped"):
        adapter.prepare(records, phase="historical-test")


def test_real_backend_signature_and_result_conversion() -> None:
    provider = _provider_adapter(_FakeRealProvider())
    prepared = provider.prepare((_record(0), _record(1)), phase="train-epoch-1-batch-0")
    real = _FakeRealBackend()
    adapter = C0CoarseBackendAdapter(real)
    session = adapter.create_session(
        seed=c0_runner_session_seed("260828", "train"), purpose="train"
    )
    train = session.train_batch(prepared)
    prediction = session.predict_batch(prepared)
    assert train.loss == pytest.approx(0.25)
    assert train.valid_count == 2
    assert prediction.probability == (0.75, 0.75)
    assert prediction.valid == (True, True)
    assert real.calls[0][1] == 260828
    assert real.session.events == ["train", "predict"]


def test_adapter_rejects_wrong_external_identity_anchor() -> None:
    with pytest.raises(C0RunnerError, match="receipt is unsafe"):
        C0CoarseProviderAdapter(
            _FakeRealProvider(),
            archive_locks=_ARCHIVE_LOCKS,
            expected_freeze_content_sha256=_FREEZE_SHA,
            expected_identity_index_content_sha256="0" * 64,
        )
