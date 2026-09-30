import torch

from staging.pairwise_v0_2.models.coarse import CoarseOutput
from staging.pairwise_v0_2.training.cascade import (
    CascadeMode,
    LocalCascadeScores,
    execute_coarse_local_cascade,
)
from staging.pairwise_v0_2.training.thresholds import CoarseGateArtifact


def _coarse(probability, valid):
    probability = torch.tensor(probability, dtype=torch.float32)
    valid = torch.tensor(valid, dtype=torch.bool)
    embedding = torch.zeros((len(probability), 3), dtype=torch.float32)
    return CoarseOutput(
        logit=torch.logit(probability),
        probability=probability,
        embedding_a=embedding,
        embedding_b=embedding.clone(),
        valid_problem=valid,
    )


def _gate(threshold=0.2):
    return CoarseGateArtifact(
        threshold=threshold,
        target_recall=0.99,
        achieved_recall=1.0,
        negative_rejection_rate=0.5,
        validation_sample_count=20,
        validation_positive_count=10,
        validation_negative_count=10,
        source_split="validation",
        checkpoint_id="checkpoint-1",
        config_hash="config-1",
    )


def test_all_coarse_rejections_make_zero_geometry_and_local_calls():
    calls = {"geometry": 0, "local": 0}

    def geometry(indices):
        calls["geometry"] += 1
        return indices

    def local(payload, indices, coarse):
        calls["local"] += 1
        raise AssertionError("local scorer must not run for rejected negatives")

    result = execute_coarse_local_cascade(
        _coarse([0.05, 0.1, 0.19], [True, True, True]),
        geometry,
        local,
        gate_artifact=_gate(),
        checkpoint_id="checkpoint-1",
        config_hash="config-1",
    )
    assert calls == {"geometry": 0, "local": 0}
    assert result.gate.reject.tolist() == [True, True, True]
    assert result.local_evaluated.tolist() == [False, False, False]
    assert result.decision_valid.tolist() == [True, True, True]
    assert result.receipt.geometry_callback_count == 0
    assert result.receipt.local_callback_count == 0


def test_invalid_coarse_fails_open_and_remains_review_only():
    selected_seen = []

    def geometry(indices):
        selected_seen.append(indices)
        return "geometry"

    def local(payload, indices, coarse):
        assert payload == "geometry"
        return LocalCascadeScores(
            probability=torch.tensor([0.9]),
            valid=torch.tensor([True]),
            requires_review=torch.tensor([False]),
        )

    result = execute_coarse_local_cascade(
        _coarse([0.05, 0.1], [True, False]),
        geometry,
        local,
        gate_artifact=_gate(),
        checkpoint_id="checkpoint-1",
        config_hash="config-1",
    )
    assert selected_seen == [(1,)]
    assert result.gate.reject.tolist() == [True, False]
    assert result.gate.pass_to_local.tolist() == [False, True]
    assert result.local_evaluated.tolist() == [False, True]
    assert torch.allclose(result.final_probability, torch.tensor([0.05, 0.9]))
    assert result.requires_review.tolist() == [False, True]
    assert result.decision_valid.tolist() == [True, False]


def test_training_bypass_is_explicit_in_portable_receipt():
    selected_seen = []

    def geometry(indices):
        selected_seen.append(indices)
        return indices

    def local(payload, indices, coarse):
        return LocalCascadeScores(
            probability=torch.tensor([0.2, 0.8]),
            valid=torch.tensor([True, True]),
            requires_review=torch.tensor([False, False]),
        )

    result = execute_coarse_local_cascade(
        _coarse([0.01, 0.02], [True, True]),
        geometry,
        local,
        mode=CascadeMode.TRAINING,
    )
    assert selected_seen == [(0, 1)]
    assert result.receipt.training_gate_bypassed
    assert result.receipt.gate_policy == (
        "bypassed_for_training_all_samples_to_local"
    )
    assert result.receipt.pass_to_local_count == 2
    assert result.receipt.coarse_reject_count == 0
