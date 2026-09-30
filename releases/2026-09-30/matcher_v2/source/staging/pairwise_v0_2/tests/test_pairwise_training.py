from pathlib import Path

import pytest
import torch

from staging.pairwise_v0_2.models import (
    CoarseModelConfig,
    DunhuangPairwiseV02,
    LocalModelConfig,
    PairwiseModelConfig,
    aggregate_direction_candidates,
)
from staging.pairwise_v0_2.training import (
    PairwiseBatch,
    PairwiseLossConfig,
    apply_high_recall_gate,
    binary_metrics,
    canonical_config_hash,
    directional_candidate_loss,
    eval_directional_step,
    eval_step,
    fit_high_recall_gate,
    load_trusted_checkpoint,
    monotonic_score_loss,
    save_checkpoint,
    swap_directional_model_inputs,
    train_directional_step,
    train_step,
)


def _config() -> PairwiseModelConfig:
    return PairwiseModelConfig(
        coarse=CoarseModelConfig(widths=(4, 8), embedding_dim=8, hidden_dim=8),
        local=LocalModelConfig(
            input_channels=3,
            feature_dim=8,
            num_heads=2,
            ff_dim=16,
            matcher_mode="dual_softmax",
        ),
        fusion_hidden_dim=8,
    )


def _batch() -> PairwiseBatch:
    generator = torch.Generator().manual_seed(17)
    return PairwiseBatch(
        coarse_a=torch.rand((2, 1, 16, 16), generator=generator),
        coarse_b=torch.rand((2, 1, 16, 16), generator=generator),
        local_a=torch.rand((2, 3, 3, 8, 8), generator=generator),
        local_b=torch.rand((2, 4, 3, 8, 8), generator=generator),
        token_mask_a=torch.tensor([[True, True, True], [True, True, False]]),
        token_mask_b=torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        ),
        label=torch.tensor([True, False]),
    )


def _directional_inputs(batch: PairwiseBatch):
    sample_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    return {
        "coarse_a": batch.coarse_a,
        "coarse_b": batch.coarse_b,
        "local_a": batch.local_a.index_select(0, sample_index),
        "local_b": batch.local_b.index_select(0, sample_index),
        "token_mask_a": batch.token_mask_a.index_select(0, sample_index),
        "token_mask_b": batch.token_mask_b.index_select(0, sample_index),
        "sample_index": sample_index,
        "direction_index": torch.tensor([0, 0, 2, 1, 3], dtype=torch.long),
        "candidate_valid": torch.tensor([True, False, True, True, True]),
    }


def test_batch_requires_explicit_bool_labels_and_swaps_correspondence() -> None:
    value = _batch()
    with pytest.raises(TypeError, match="label"):
        PairwiseBatch(
            value.coarse_a,
            value.coarse_b,
            value.local_a,
            value.local_b,
            value.token_mask_a,
            value.token_mask_b,
            torch.tensor([1.0, 0.0]),
        )
    correspondence = torch.zeros((2, 3, 4))
    correspondence_mask = torch.ones((2, 3, 4), dtype=torch.bool)
    with_correspondence = PairwiseBatch(
        value.coarse_a,
        value.coarse_b,
        value.local_a,
        value.local_b,
        value.token_mask_a,
        value.token_mask_b,
        value.label,
        correspondence,
        correspondence_mask,
    )
    assert with_correspondence.swapped().correspondence.shape == (2, 4, 3)


def test_small_cpu_train_and_eval_step_update_parameters() -> None:
    model = DunhuangPairwiseV02(_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.fusion[0].weight.detach().clone()
    result = train_step(
        model,
        _batch(),
        optimizer,
        loss_config=PairwiseLossConfig(swap_weight=0.1),
    )
    assert torch.isfinite(result.loss.total)
    assert result.loss.valid_pair_count == 2
    assert result.gradient_norm >= 0.0
    assert not torch.equal(before, model.fusion[0].weight.detach())
    evaluation = eval_step(model, _batch())
    assert torch.isfinite(evaluation.loss.total)
    assert not model.training


def test_finite_nonconverged_sinkhorn_trains_but_decision_fails_closed() -> None:
    config = PairwiseModelConfig(
        coarse=CoarseModelConfig(widths=(4, 8), embedding_dim=8, hidden_dim=8),
        local=LocalModelConfig(
            input_channels=3,
            feature_dim=8,
            num_heads=2,
            ff_dim=16,
            matcher_mode="dustbin_sinkhorn",
            matcher_temperature=0.25,
            sinkhorn_iterations=1,
            sinkhorn_tolerance=0.0,
            sinkhorn_training_policy="finite_with_residual",
        ),
        fusion_hidden_dim=8,
    )
    model = DunhuangPairwiseV02(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.local_model.local_head[0].weight.detach().clone()
    result = train_step(model, _batch(), optimizer, compute_swapped=False)

    assert result.output.local.finite_problem.tolist() == [True, True]
    assert result.output.local.training_valid.tolist() == [True, True]
    assert result.output.decision_valid.tolist() == [False, False]
    assert result.loss.valid_pair_count == 2
    assert result.loss.sinkhorn_residual.item() > 0.0
    assert result.transport_diagnostics.nonconverged_finite_count == 2
    assert result.transport_diagnostics.decision_valid_count == 0
    assert not torch.equal(before, model.local_model.local_head[0].weight.detach())


def test_ragged_directional_train_step_matches_geometry_tensor_contract() -> None:
    model = DunhuangPairwiseV02(_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _batch()
    sample_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    direction_index = torch.tensor([0, 0, 2, 1, 3], dtype=torch.long)
    model_inputs = {
        "coarse_a": batch.coarse_a,
        "coarse_b": batch.coarse_b,
        "local_a": batch.local_a.index_select(0, sample_index),
        "local_b": batch.local_b.index_select(0, sample_index),
        "token_mask_a": batch.token_mask_a.index_select(0, sample_index),
        "token_mask_b": batch.token_mask_b.index_select(0, sample_index),
        "sample_index": sample_index,
        "direction_index": direction_index,
        "candidate_valid": torch.tensor([True, False, True, True, True]),
    }
    before = model.fusion[0].weight.detach().clone()
    result = train_directional_step(
        model,
        model_inputs,
        batch.label,
        torch.tensor([-1, -1], dtype=torch.long),
        torch.tensor([False, False]),
        optimizer,
    )
    assert torch.isfinite(result.total_loss)
    assert result.output.direction_output.pair_valid.tolist() == [True, True]
    assert not torch.equal(before, model.fusion[0].weight.detach())
    evaluated = eval_directional_step(
        model,
        model_inputs,
        batch.label,
        torch.tensor([-1, -1], dtype=torch.long),
        torch.tensor([False, False]),
    )
    assert torch.isfinite(evaluated.total_loss)
    assert evaluated.gradient_norm is None


def test_local_only_directional_training_excludes_coarse_and_fusion_gradients() -> None:
    model = DunhuangPairwiseV02(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    batch = _batch()
    model_inputs = _directional_inputs(batch)
    coarse_before = {
        name: value.detach().clone()
        for name, value in model.coarse_model.named_parameters()
    }
    fusion_before = {
        name: value.detach().clone() for name, value in model.fusion.named_parameters()
    }

    result = train_directional_step(
        model,
        model_inputs,
        batch.label,
        torch.tensor([0, -1], dtype=torch.long),
        torch.tensor([True, False]),
        optimizer,
        score_source="local",
        # Deliberately leave the non-zero backwards-compatible coarse weight:
        # local-only semantics must disable the auxiliary, not rely on callers.
        coarse_loss_weight=0.25,
    )

    assert result.output.score_source == "local"
    assert result.output.coarse_output is None
    assert result.output.arc_pairwise_output is not None
    assert result.coarse_loss.item() == 0.0
    torch.testing.assert_close(
        result.output.arc_logits,
        result.output.arc_pairwise_output.local_logit,
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.local_model.parameters()
    )
    assert all(parameter.grad is None for parameter in model.coarse_model.parameters())
    assert all(parameter.grad is None for parameter in model.fusion.parameters())
    for name, value in model.coarse_model.named_parameters():
        torch.testing.assert_close(value, coarse_before[name], rtol=0.0, atol=0.0)
    for name, value in model.fusion.named_parameters():
        torch.testing.assert_close(value, fusion_before[name], rtol=0.0, atol=0.0)


def test_local_only_sinkhorn_keeps_training_residual_and_decision_fail_closed() -> None:
    config = PairwiseModelConfig(
        coarse=CoarseModelConfig(widths=(4, 8), embedding_dim=8, hidden_dim=8),
        local=LocalModelConfig(
            input_channels=3,
            feature_dim=8,
            num_heads=2,
            ff_dim=16,
            matcher_mode="dustbin_sinkhorn",
            matcher_temperature=0.25,
            sinkhorn_iterations=1,
            sinkhorn_tolerance=0.0,
            sinkhorn_training_policy="finite_with_residual",
        ),
        fusion_hidden_dim=8,
    )
    model = DunhuangPairwiseV02(config)
    batch = _batch()
    result = eval_directional_step(
        model,
        _directional_inputs(batch),
        batch.label,
        torch.tensor([-1, -1], dtype=torch.long),
        torch.tensor([False, False]),
        score_source="local",
    )

    assert result.output.arc_pairwise_output is not None
    arc = result.output.arc_pairwise_output
    assert arc.local.finite_problem.all().item()
    assert arc.local.training_valid.all().item()
    assert not arc.local.decision_valid.any().item()
    assert result.output.arc_training_valid.tolist() == [True, False, True, True, True]
    assert not result.output.arc_valid.any().item()
    assert result.output.training_direction_output.pair_valid.tolist() == [True, True]
    assert result.output.direction_output.pair_valid.tolist() == [False, False]
    assert result.sinkhorn_residual_loss.item() > 0.0
    assert result.transport_diagnostics is not None
    assert result.transport_diagnostics.finite_problem_count == 5
    assert result.transport_diagnostics.nonconverged_finite_count == 5
    assert result.transport_diagnostics.decision_valid_count == 0
    assert result.coarse_loss.item() == 0.0


@pytest.mark.parametrize("score_source", ["fused", "local"])
def test_ragged_directional_swap_ablation_preserves_inverse_slots(
    score_source: str,
) -> None:
    model = DunhuangPairwiseV02(_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _batch()
    sample_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    direction_index = torch.tensor([0, 1, 2, 1, 3], dtype=torch.long)
    model_inputs = {
        "coarse_a": batch.coarse_a,
        "coarse_b": batch.coarse_b,
        "local_a": batch.local_a.index_select(0, sample_index),
        "local_b": batch.local_b.index_select(0, sample_index),
        "token_mask_a": batch.token_mask_a.index_select(0, sample_index),
        "token_mask_b": batch.token_mask_b.index_select(0, sample_index),
        "sample_index": sample_index,
        "direction_index": direction_index,
        "candidate_valid": torch.ones(5, dtype=torch.bool),
    }
    swapped = swap_directional_model_inputs(model_inputs)
    assert swapped["direction_index"].tolist() == [1, 0, 3, 0, 2]
    result = train_directional_step(
        model,
        model_inputs,
        batch.label,
        torch.tensor([-1, -1], dtype=torch.long),
        torch.tensor([False, False]),
        optimizer,
        compute_swapped=True,
        score_source=score_source,
    )
    assert result.swapped_output is not None
    assert result.output.score_source == score_source
    assert result.swapped_output.score_source == score_source
    assert torch.isfinite(result.swap_loss)
    assert result.swap_loss.item() < 1e-8


def test_monotonic_and_unknown_positive_direction_losses() -> None:
    clean = torch.tensor([0.8, 0.4], requires_grad=True)
    degraded = torch.tensor([0.9, 0.7], requires_grad=True)
    loss = monotonic_score_loss(clean, degraded, torch.tensor([True, False]))
    assert loss.item() == pytest.approx(0.1)

    output = aggregate_direction_candidates(
        torch.tensor([[0.0, 2.0, -1.0, -2.0], [-1.0, -1.0, -1.0, -1.0]]),
        torch.ones((2, 4), dtype=torch.bool),
    )
    # First positive has unknown direction: pair MIL only.  Negative still
    # supervises all four directions as negative.
    directional = directional_candidate_loss(
        output,
        torch.tensor([True, False]),
        torch.tensor([-1, -1], dtype=torch.long),
        direction_target_valid=torch.tensor([False, False]),
    )
    assert torch.isfinite(directional)


def test_high_recall_gate_is_validation_only_and_invalid_scores_fail_open() -> None:
    probability = torch.tensor([0.10, 0.20, 0.60, 0.70, 0.80, 0.90])
    label = torch.tensor([False, False, True, True, True, False])
    valid = torch.ones(6, dtype=torch.bool)
    with pytest.raises(ValueError, match="validation only"):
        fit_high_recall_gate(
            probability,
            label,
            valid,
            checkpoint_id="ckpt",
            config_hash="cfg",
            source_split="test",
        )
    artifact = fit_high_recall_gate(
        probability,
        label,
        valid,
        checkpoint_id="ckpt",
        config_hash="cfg",
        target_recall=1.0,
    )
    assert artifact.threshold == pytest.approx(0.6)
    decision = apply_high_recall_gate(
        torch.tensor([0.1, float("nan"), 0.8]),
        torch.tensor([True, True, True]),
        artifact,
        checkpoint_id="ckpt",
        config_hash="cfg",
    )
    assert decision.reject.tolist() == [True, False, False]
    assert decision.pass_to_local.tolist() == [False, True, True]
    assert decision.requires_review.tolist() == [False, True, False]
    with pytest.raises(ValueError, match="does not match"):
        apply_high_recall_gate(
            probability,
            valid,
            artifact,
            checkpoint_id="different",
            config_hash="cfg",
        )


def test_metrics_and_config_bound_checkpoint_round_trip(tmp_path: Path) -> None:
    metrics = binary_metrics(
        torch.tensor([0.1, 0.2, 0.8, 0.9]),
        torch.tensor([False, False, True, True]),
        torch.ones(4, dtype=torch.bool),
    )
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)

    config = _config()
    model = DunhuangPairwiseV02(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    receipt = save_checkpoint(
        path,
        model,
        config=config,
        epoch=3,
        optimizer=optimizer,
        metrics={"loss": 0.4},
        provenance={"manifest_sha256": "abc"},
    )
    assert receipt.config_hash == canonical_config_hash(config)
    restored = DunhuangPairwiseV02(config)
    metadata = load_trusted_checkpoint(
        path,
        restored,
        expected_config=config,
        expected_file_sha256=receipt.file_sha256,
        trusted=True,
    )
    assert metadata["epoch"] == 3
    assert metadata["file_sha256"] == receipt.file_sha256
    with pytest.raises(ValueError, match="trusted=True"):
        load_trusted_checkpoint(
            path,
            restored,
            expected_config=config,
            expected_file_sha256=receipt.file_sha256,
        )
    changed = PairwiseModelConfig(
        coarse=config.coarse,
        local=config.local,
        fusion_hidden_dim=9,
    )
    with pytest.raises(ValueError, match="does not match"):
        load_trusted_checkpoint(
            path,
            restored,
            expected_config=changed,
            expected_file_sha256=receipt.file_sha256,
            trusted=True,
        )
    with pytest.raises(ValueError, match="sensitive key"):
        save_checkpoint(
            tmp_path / "secret.pt",
            model,
            config=config,
            epoch=0,
            provenance={"password": "must-not-be-written"},
        )
