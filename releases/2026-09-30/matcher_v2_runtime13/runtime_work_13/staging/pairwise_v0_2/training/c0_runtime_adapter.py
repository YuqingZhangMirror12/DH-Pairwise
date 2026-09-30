"""Strict adapters from frozen coarse C0 runtime APIs to ``c0_runner``.

The historical provider/backend intentionally implement the generic
short-ablation protocols.  The dedicated C0 runner has smaller typed
protocols.  These wrappers translate signatures and return values without
weakening either side's coarse-only, archive, geometry, or sealed-test guards.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Mapping, Sequence

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    HISTORICAL_TEST_ACCESS_EVIDENCE,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.training.c0_coarse_backend import (
    C0_COARSE_BACKEND_VERSION,
    C0_COARSE_MODEL_CONFIG,
    C0_COARSE_OPTIMIZER_CONFIG,
    C0_COARSE_SEED,
    C0CoarseBackend,
)
from staging.pairwise_v0_2.training.c0_coarse_provider import (
    C0_COARSE_PROVIDER_RECEIPT_VERSION,
    C0_COARSE_PROVIDER_VERSION,
    C0CoarseProvider,
)
from staging.pairwise_v0_2.training.c0_runner import (
    BackendAttestation,
    C0PredictionBatch,
    C0PreparedBatch,
    C0RunnerError,
    C0TrainBatchResult,
    ProviderAttestation,
    c0_record_sequence_commitment,
    c0_runner_session_seed,
)
from staging.pairwise_v0_2.training.checkpoint import canonical_config_hash
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    BatchProviderContract,
    EvidenceMode,
    ExecutionKind,
    PredictionBatch,
    PreparedAblationBatch,
    TrainBatchResult,
    record_sequence_fingerprint,
)


ADAPTER_VERSION = "dunhuang-pairwise-c0-runtime-adapter/0.3"
_CACHE_INTERFACE = "bounded_host_coarse_tensor_lru_no_geometry_cache"


def _sha256_json(value: Any) -> str:
    import json

    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def c0_coarse_arm() -> AblationArm:
    """Construct the sole arm accepted by the real C0 provider/backend."""

    return AblationArm(
        name=AblationArmName.COARSE_ONLY,
        evidence=EvidenceMode.COARSE,
        matcher_mode=None,
        model_config=C0_COARSE_MODEL_CONFIG,
        optimizer_config=C0_COARSE_OPTIMIZER_CONFIG,
        aggregation_config={
            "evidence": "coarse",
            "matcher_mode": None,
            "score": "symmetric_coarse_probability",
        },
        arc_pooling=None,
    )


def _runner_phase_to_provider(phase: str) -> str:
    if phase.startswith("train-epoch-"):
        return "train"
    if phase.startswith("validation-epoch-") or phase.startswith(
        "winner-reload-validation"
    ):
        return "validation"
    raise C0RunnerError("runner phase cannot be mapped to C0 provider")


def _validate_provider_contract(contract: Any) -> BatchProviderContract:
    if not isinstance(contract, BatchProviderContract):
        raise C0RunnerError("real C0 provider lacks BatchProviderContract")
    if (
        contract.provider_version != C0_COARSE_PROVIDER_VERSION
        or contract.coarse_preprocess_mode != "tight_crop_letterbox"
        or contract.cache_interface != _CACHE_INTERFACE
        or contract.coarse_only_geometry_free is not True
    ):
        raise C0RunnerError("real C0 provider contract changed")
    return contract


def _validate_provider_receipt(
    provider: C0CoarseProvider,
    *,
    archive_locks: Mapping[str, Mapping[str, Any]],
    expected_freeze_content_sha256: str,
    expected_identity_index_content_sha256: str,
    require_unused: bool,
) -> Mapping[str, Any]:
    receipt = provider.receipt()
    if not isinstance(receipt, Mapping):
        raise C0RunnerError("real C0 provider receipt is missing")
    if (
        receipt.get("schema_version") != C0_COARSE_PROVIDER_RECEIPT_VERSION
        or receipt.get("provider_version") != C0_COARSE_PROVIDER_VERSION
        or receipt.get("status") != "archive_preflight_complete"
        or receipt.get("freeze_content_sha256") != expected_freeze_content_sha256
        or receipt.get("identity_index_content_sha256")
        != expected_identity_index_content_sha256
        or "historical_test_read" in receipt
        or receipt.get("historical_test_access")
        != dict(HISTORICAL_TEST_ACCESS_EVIDENCE)
        or receipt.get("sealed_real_read") is not False
        or receipt.get("geometry_cache_local_sinkhorn_calls") != 0
    ):
        raise C0RunnerError("real C0 provider receipt is unsafe")
    counters = receipt.get("counters")
    if not isinstance(counters, Mapping):
        raise C0RunnerError("real provider receipt lacks counters")
    if require_unused and any(
        counters.get(name) != 0
        for name in (
            "batch_count",
            "record_count",
            "fragment_request_count",
            "mask_loader_call_count",
            "archive_decode_count",
            "coarse_preprocess_count",
        )
    ):
        raise C0RunnerError("real provider was used before adapter attestation")
    verification = receipt.get("archive_verification")
    if not isinstance(verification, list) or len(verification) != 2:
        raise C0RunnerError("real provider lacks two archive verifications")
    expected = sorted(
        (str(lock.get("sha256")), int(lock.get("bytes", -1)))
        for lock in archive_locks.values()
    )
    observed = sorted(
        (str(item.get("observed_sha256")), int(item.get("byte_count", -1)))
        for item in verification
        if isinstance(item, Mapping)
        and item.get("expected_sha256") == item.get("observed_sha256")
    )
    if observed != expected:
        raise C0RunnerError("provider archive verification differs from freeze")
    return receipt


class C0CoarseProviderAdapter:
    """Expose an archive-attested ``C0CoarseProvider`` to the runner."""

    def __init__(
        self,
        provider: C0CoarseProvider,
        *,
        archive_locks: Mapping[str, Mapping[str, Any]],
        expected_freeze_content_sha256: str,
        expected_identity_index_content_sha256: str,
    ) -> None:
        if not isinstance(provider, C0CoarseProvider):
            raise TypeError("provider must be C0CoarseProvider")
        contract = _validate_provider_contract(provider.contract)
        receipt = _validate_provider_receipt(
            provider,
            archive_locks=archive_locks,
            expected_freeze_content_sha256=expected_freeze_content_sha256,
            expected_identity_index_content_sha256=(
                expected_identity_index_content_sha256
            ),
            require_unused=True,
        )
        preprocessing = receipt.get("preprocessing")
        if (
            not isinstance(preprocessing, Mapping)
            or preprocessing.get("preprocessing_sha256")
            != contract.coarse_preprocessing_sha256
            or preprocessing.get("geometry_config_sha256")
            != contract.geometry_config_sha256
        ):
            raise C0RunnerError("provider preprocessing receipt changed")
        self._provider = provider
        self._contract = contract
        self._archive_locks = archive_locks
        self._expected_freeze_content_sha256 = expected_freeze_content_sha256
        self._expected_identity_index_content_sha256 = (
            expected_identity_index_content_sha256
        )
        self._arm = c0_coarse_arm()
        self.attestation = ProviderAttestation(
            provider_version="{}+{}".format(
                C0_COARSE_PROVIDER_VERSION, ADAPTER_VERSION
            ),
            preprocessing_sha256=contract.coarse_preprocessing_sha256,
            archive_locks_sha256=_sha256_json(archive_locks),
            archives_verified=True,
            identity_index_content_sha256=(expected_identity_index_content_sha256),
            mask_pixels_loaded_during_attestation=False,
            sealed_real_test_capability=False,
        )

    def prepare(
        self, records: Sequence[TrainingPairRecord], *, phase: str
    ) -> C0PreparedBatch:
        provider_phase = _runner_phase_to_provider(phase)
        prepared = self._provider.prepare(records, arm=self._arm, phase=provider_phase)
        if not isinstance(prepared, PreparedAblationBatch):
            raise C0RunnerError("real provider returned the wrong batch type")
        expected_real_sequence = record_sequence_fingerprint(records)
        if not hmac.compare_digest(
            prepared.record_sequence_sha256, expected_real_sequence
        ):
            raise C0RunnerError("real provider record commitment changed")
        if (
            prepared.local_candidate_sha256 is not None
            or prepared.geometry_config_sha256 is not None
            or any(
                prepared.processing_counts.get(name) != 0
                for name in (
                    "geometry_build_count",
                    "geometry_cache_read_count",
                    "geometry_cache_write_count",
                    "local_candidate_count",
                )
            )
        ):
            raise C0RunnerError("real provider exposed local/geometry evidence")
        return C0PreparedBatch(
            payload=prepared,
            sample_count=prepared.sample_count,
            record_sequence_sha256=c0_record_sequence_commitment(records),
            prepared_input_sha256=prepared.prepared_input_sha256,
        )

    def final_receipt(self) -> Mapping[str, Any]:
        """Read and revalidate the real provider's cumulative evidence."""

        receipt = _validate_provider_receipt(
            self._provider,
            archive_locks=self._archive_locks,
            expected_freeze_content_sha256=self._expected_freeze_content_sha256,
            expected_identity_index_content_sha256=(
                self._expected_identity_index_content_sha256
            ),
            require_unused=False,
        )
        preprocessing = receipt.get("preprocessing")
        if (
            not isinstance(preprocessing, Mapping)
            or preprocessing.get("preprocessing_sha256")
            != self._contract.coarse_preprocessing_sha256
            or preprocessing.get("geometry_config_sha256")
            != self._contract.geometry_config_sha256
        ):
            raise C0RunnerError("provider final preprocessing receipt changed")
        return receipt


def _validate_backend_contract(backend: C0CoarseBackend) -> None:
    contract = backend.contract
    if (
        contract.execution_kind is not ExecutionKind.SYNTHETIC_TRAIN_VALIDATION
        or contract.backend_version != C0_COARSE_BACKEND_VERSION
        or contract.model_family != "SymmetricCoarseSiamese-C0-N-Q1"
        or contract.device_type not in {"cpu", "cuda", "mps"}
        or contract.sealed_real_test_capability is not False
    ):
        raise C0RunnerError("real C0 backend contract changed")


class _C0CoarseSessionAdapter:
    def __init__(self, session: Any, arm: AblationArm) -> None:
        self._session = session
        self._arm = arm
        self.model = session.model
        self.optimizer = session.optimizer
        self.model_config = dict(session.model_config)
        self.optimizer_config = dict(session.optimizer_config)
        if canonical_config_hash(self.model_config) != canonical_config_hash(
            C0_COARSE_MODEL_CONFIG
        ) or canonical_config_hash(self.optimizer_config) != canonical_config_hash(
            C0_COARSE_OPTIMIZER_CONFIG
        ):
            raise C0RunnerError("real C0 session config changed")

    @staticmethod
    def _unwrap(batch: C0PreparedBatch) -> PreparedAblationBatch:
        if not isinstance(batch, C0PreparedBatch) or not isinstance(
            batch.payload, PreparedAblationBatch
        ):
            raise C0RunnerError("adapter batch does not contain real provider output")
        prepared = batch.payload
        if (
            prepared.sample_count != batch.sample_count
            or prepared.prepared_input_sha256 != batch.prepared_input_sha256
        ):
            raise C0RunnerError("adapter batch wrapper changed")
        return prepared

    def train_batch(self, batch: C0PreparedBatch) -> C0TrainBatchResult:
        result = self._session.train_batch(self._unwrap(batch))
        if not isinstance(result, TrainBatchResult):
            raise C0RunnerError("real backend returned the wrong train result")
        return C0TrainBatchResult(loss=result.loss, valid_count=result.valid_count)

    def predict_batch(self, batch: C0PreparedBatch) -> C0PredictionBatch:
        prediction = self._session.predict_batch(
            self._unwrap(batch), evidence=EvidenceMode.COARSE
        )
        if not isinstance(prediction, PredictionBatch):
            raise C0RunnerError("real backend returned the wrong prediction type")
        return C0PredictionBatch(
            probability=tuple(
                float(value) for value in prediction.probability.detach().cpu().tolist()
            ),
            valid=tuple(
                bool(value) for value in prediction.valid.detach().cpu().tolist()
            ),
        )


class C0CoarseBackendAdapter:
    """Expose a guarded ``C0CoarseBackend`` to the dedicated runner."""

    def __init__(
        self, backend: C0CoarseBackend, *, runner_seed: str = "260828"
    ) -> None:
        if not isinstance(backend, C0CoarseBackend):
            raise TypeError("backend must be C0CoarseBackend")
        _validate_backend_contract(backend)
        if runner_seed != "260828":
            raise C0RunnerError("real backend adapter requires runner seed 260828")
        self._backend = backend
        self._runner_seed = runner_seed
        self._arm = c0_coarse_arm()
        self.attestation = BackendAttestation(
            backend_version="{}+{}".format(C0_COARSE_BACKEND_VERSION, ADAPTER_VERSION),
            model_family=backend.contract.model_family,
            device_type=backend.contract.device_type,
            deterministic_reload=True,
            sealed_real_test_capability=False,
        )

    def create_session(self, *, seed: int, purpose: str) -> _C0CoarseSessionAdapter:
        expected = c0_runner_session_seed(self._runner_seed, purpose)
        if seed != expected or purpose not in {"train", "winner_reload"}:
            raise C0RunnerError("runner session seed/purpose changed")
        session = self._backend.create_session(self._arm, seed=C0_COARSE_SEED)
        return _C0CoarseSessionAdapter(session, self._arm)


__all__ = [
    "ADAPTER_VERSION",
    "C0CoarseBackendAdapter",
    "C0CoarseProviderAdapter",
    "c0_coarse_arm",
]
