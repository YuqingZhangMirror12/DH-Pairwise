from __future__ import annotations

import inspect
import math
from dataclasses import replace

import pytest
import torch

from staging.pairwise_v0_2.models.coarse import SymmetricCoarseSiamese
from staging.pairwise_v0_2.models.local_matcher import MatcherMode
from staging.pairwise_v0_2.models.pairwise import ArcPoolingConfig
from staging.pairwise_v0_2.training.c0_coarse_backend import (
    C0_COARSE_MODEL_CONFIG,
    C0_COARSE_OPTIMIZER_CONFIG,
    C0_COARSE_SEED,
    C0CoarseBackend,
    C0CoarseBackendError,
    C0CoarsePayload,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    EvidenceMode,
    ExecutionKind,
    PreparedAblationBatch,
)


def _arm(*, model_config=None, optimizer_config=None) -> AblationArm:
    return AblationArm(
        name=AblationArmName.COARSE_ONLY,
        evidence=EvidenceMode.COARSE,
        matcher_mode=None,
        model_config=(C0_COARSE_MODEL_CONFIG if model_config is None else model_config),
        optimizer_config=(
            C0_COARSE_OPTIMIZER_CONFIG if optimizer_config is None else optimizer_config
        ),
        aggregation_config={"evidence": "coarse", "matcher_mode": None},
        arc_pooling=None,
    )


def _payload(batch_size: int = 2) -> C0CoarsePayload:
    coarse_a = torch.zeros(batch_size, 1, 128, 128, dtype=torch.float32)
    coarse_b = torch.zeros_like(coarse_a)
    for index in range(batch_size):
        coarse_a[index, 0, 16 + index : 80, 24:88] = 1.0
        coarse_b[index, 0, 24:88, 32 + index : 96] = 1.0
    labels = torch.tensor(
        [float(index % 2) for index in range(batch_size)], dtype=torch.float32
    )
    return C0CoarsePayload.from_tensors(coarse_a, coarse_b, labels)


def _batch(payload: C0CoarsePayload | object | None = None) -> PreparedAblationBatch:
    actual = _payload() if payload is None else payload
    prepared_sha = (
        actual.payload_content_sha256
        if isinstance(actual, C0CoarsePayload)
        else "d" * 64
    )
    return PreparedAblationBatch(
        payload=actual,
        sample_count=(
            int(actual.labels.shape[0]) if isinstance(actual, C0CoarsePayload) else 2
        ),
        record_sequence_sha256="a" * 64,
        prepared_input_sha256=prepared_sha,
        local_candidate_sha256=None,
        coarse_preprocessing_sha256="b" * 64,
        geometry_config_sha256=None,
        processing_counts={
            "mask_load_count": 4,
            "coarse_preprocess_count": 4,
            "geometry_build_count": 0,
            "geometry_cache_read_count": 0,
            "geometry_cache_write_count": 0,
            "local_candidate_count": 0,
        },
    )


@pytest.fixture(scope="module")
def session():
    return C0CoarseBackend("cpu").create_session(_arm(), seed=C0_COARSE_SEED)


def test_backend_freezes_exact_model_optimizer_and_safety_contract(session):
    backend = C0CoarseBackend("cpu")
    assert backend.contract.execution_kind is ExecutionKind.SYNTHETIC_TRAIN_VALIDATION
    assert backend.contract.sealed_real_test_capability is False
    assert backend.contract.device_type == "cpu"

    assert isinstance(session.model, SymmetricCoarseSiamese)
    assert session.model_config == dict(C0_COARSE_MODEL_CONFIG)
    assert session.optimizer_config == dict(C0_COARSE_OPTIMIZER_CONFIG)
    assert isinstance(session.optimizer, torch.optim.AdamW)
    assert isinstance(session.loss_function, torch.nn.BCEWithLogitsLoss)
    assert session.optimizer.defaults["lr"] == pytest.approx(3e-4)
    assert session.optimizer.defaults["weight_decay"] == pytest.approx(1e-4)
    assert session.optimizer.defaults["betas"] == (0.9, 0.999)
    assert session.optimizer.defaults["eps"] == pytest.approx(1e-8)
    assert session.optimizer.defaults["amsgrad"] is False
    assert session.optimizer.defaults["foreach"] is False
    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
        assert session.optimizer.defaults.get("fused") is None

    assert session.determinism_config["seed"] == 260828
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_train_step_has_full_validity_and_finite_loss_logits_and_gradients(session):
    result = session.train_batch(_batch())
    assert math.isfinite(result.loss)
    assert result.valid_count == 2
    assert math.isfinite(result.diagnostics["gradient_l2_norm"])
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in session.model.parameters()
    )
    assert result.diagnostics["coarse_forward_count"] == 1
    for key in (
        "local_forward_count",
        "sinkhorn_call_count",
        "geometry_build_count",
        "geometry_cache_read_count",
        "geometry_cache_write_count",
        "local_candidate_count",
    ):
        assert result.diagnostics[key] == 0


def test_predict_uses_eval_inference_mode_and_checks_ab_swap(session):
    gradient_states = []
    handle = session.model.register_forward_pre_hook(
        lambda _module, _inputs: gradient_states.append(torch.is_grad_enabled())
    )
    try:
        prediction = session.predict_batch(_batch(), evidence=EvidenceMode.COARSE)
    finally:
        handle.remove()

    assert session.model.training is False
    assert gradient_states == [False, False]
    assert prediction.probability.requires_grad is False
    assert prediction.valid.tolist() == [True, True]
    assert torch.isfinite(prediction.probability).all().item()
    assert prediction.diagnostics["ab_swap_logit_max_abs_error"] <= 1e-6
    assert prediction.diagnostics["ab_swap_probability_max_abs_error"] <= 1e-6
    assert prediction.diagnostics["coarse_forward_count"] == 2
    assert prediction.diagnostics["local_forward_count"] == 0
    assert prediction.diagnostics["sinkhorn_call_count"] == 0


def test_invalid_mask_sample_fails_closed(session):
    payload = _payload()
    payload.coarse_a[0].zero_()
    changed = C0CoarsePayload.from_tensors(
        payload.coarse_a, payload.coarse_b, payload.labels
    )
    with pytest.raises(C0CoarseBackendError, match="invalid coarse sample"):
        session.train_batch(_batch(changed))


def test_payload_hash_is_recomputed_and_wrapper_hash_is_bound(session):
    payload = _payload()
    lied = replace(payload, payload_content_sha256="0" * 64)
    with pytest.raises(C0CoarseBackendError, match="digest does not match"):
        session.train_batch(replace(_batch(lied), prepared_input_sha256="0" * 64))

    with pytest.raises(C0CoarseBackendError, match="prepared input digest"):
        session.train_batch(replace(_batch(payload), prepared_input_sha256="1" * 64))

    mutable = _payload()
    prepared = _batch(mutable)
    mutable.coarse_b[0, 0, 2, 2] = 1.0
    with pytest.raises(C0CoarseBackendError, match="digest does not match"):
        session.train_batch(prepared)


@pytest.mark.parametrize(
    "labels",
    (
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        torch.tensor([0.0, 0.25], dtype=torch.float32),
        torch.tensor([0.0, float("nan")], dtype=torch.float32),
    ),
)
def test_wrong_label_contract_fails(session, labels):
    original = _payload()
    payload = C0CoarsePayload.from_tensors(original.coarse_a, original.coarse_b, labels)
    with pytest.raises(C0CoarseBackendError, match="labels"):
        session.train_batch(_batch(payload))


def test_wrong_payload_cardinality_evidence_and_local_processing_fail(session):
    with pytest.raises(C0CoarseBackendError, match="payload type"):
        session.train_batch(_batch(object()))

    with pytest.raises(C0CoarseBackendError, match="cardinality"):
        session.train_batch(replace(_batch(), sample_count=1))

    with pytest.raises(C0CoarseBackendError, match="coarse evidence"):
        session.predict_batch(_batch(), evidence=EvidenceMode.LOCAL)

    with pytest.raises(C0CoarseBackendError, match="local candidates"):
        session.train_batch(replace(_batch(), local_candidate_sha256="c" * 64))

    with pytest.raises(C0CoarseBackendError, match="exposed geometry"):
        session.train_batch(replace(_batch(), geometry_config_sha256="c" * 64))

    counts = dict(_batch().processing_counts)
    counts["geometry_cache_read_count"] = 1
    with pytest.raises(C0CoarseBackendError, match="geometry/cache/local"):
        session.train_batch(replace(_batch(), processing_counts=counts))


def test_backend_rejects_wrong_arm_configs_and_seed_before_model_creation():
    backend = C0CoarseBackend("cpu")
    with pytest.raises(C0CoarseBackendError, match="model config"):
        backend.create_session(
            _arm(model_config={**dict(C0_COARSE_MODEL_CONFIG), "hidden_dim": 95}),
            seed=C0_COARSE_SEED,
        )
    with pytest.raises(C0CoarseBackendError, match="optimizer config"):
        backend.create_session(
            _arm(
                optimizer_config={
                    **dict(C0_COARSE_OPTIMIZER_CONFIG),
                    "lr": 1e-3,
                }
            ),
            seed=C0_COARSE_SEED,
        )
    local_arm = AblationArm(
        name=AblationArmName.LOCAL_DUAL_SOFTMAX,
        evidence=EvidenceMode.LOCAL,
        matcher_mode=MatcherMode.DUAL_SOFTMAX.value,
        model_config={
            "matcher_mode": MatcherMode.DUAL_SOFTMAX.value,
            "arc_pooling": {
                "mode": "log_mean_exp",
                "temperature": 0.25,
                "top_k": 3,
            },
        },
        optimizer_config={"name": "AdamW"},
        aggregation_config={"evidence": "local"},
        arc_pooling=ArcPoolingConfig(),
    )
    with pytest.raises(C0CoarseBackendError, match="only coarse-only"):
        backend.create_session(local_arm, seed=C0_COARSE_SEED)
    with pytest.raises(C0CoarseBackendError, match="seed 260828"):
        backend.create_session(_arm(), seed=7)
