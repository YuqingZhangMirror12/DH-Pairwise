import numpy as np
import pytest
import torch

from staging.pairwise_v0_2.models import (
    ArcPoolingConfig,
    ArcPoolingMode,
    CoarseModelConfig,
    DunhuangPairwiseV02,
    LocalModelConfig,
    PairwiseModelConfig,
    SymmetricCoarseSiamese,
    aggregate_direction_candidates,
    aggregate_flat_arc_candidates,
)
from staging.pairwise_v0_2.training.geometry_batch import (
    GeometryBatchConfig,
    preprocess_coarse_mask,
)


def _config(mode: str, **local_overrides: object) -> PairwiseModelConfig:
    local = {
        "input_channels": 3,
        "feature_dim": 8,
        "num_heads": 2,
        "ff_dim": 16,
        "matcher_mode": mode,
        "sinkhorn_iterations": 100,
        "sinkhorn_tolerance": 1e-3,
    }
    local.update(local_overrides)
    return PairwiseModelConfig(
        coarse=CoarseModelConfig(widths=(4, 8), embedding_dim=8, hidden_dim=8),
        local=LocalModelConfig(**local),
        fusion_hidden_dim=8,
    )


def _inputs(batch: int = 2):
    generator = torch.Generator().manual_seed(13)
    coarse_a = torch.rand((batch, 1, 16, 16), generator=generator)
    coarse_b = torch.rand((batch, 1, 16, 16), generator=generator)
    local_a = torch.rand((batch, 3, 3, 8, 8), generator=generator)
    local_b = torch.rand((batch, 4, 3, 8, 8), generator=generator)
    mask_a = torch.tensor([[True, True, True], [True, False, False]])[:batch]
    mask_b = torch.tensor([[True, True, True, True], [True, True, False, False]])[
        :batch
    ]
    return coarse_a, coarse_b, local_a, local_b, mask_a, mask_b


@pytest.mark.parametrize("mode", ["dual_softmax", "dustbin_sinkhorn"])
def test_model_is_swap_symmetric_and_padded_transport_is_zero(mode: str) -> None:
    model = DunhuangPairwiseV02(_config(mode)).eval()
    inputs = _inputs()
    forward = model(*inputs)
    swapped = model(inputs[1], inputs[0], inputs[3], inputs[2], inputs[5], inputs[4])
    assert forward.decision_valid.tolist() == [True, True]
    assert torch.allclose(forward.coarse_logit, swapped.coarse_logit, atol=1e-6)
    assert torch.allclose(forward.local_logit, swapped.local_logit, atol=1e-6)
    assert torch.allclose(forward.fused_logit, swapped.fused_logit, atol=1e-6)
    assert torch.allclose(
        forward.local.assignment,
        swapped.local.assignment.transpose(1, 2),
        atol=1e-5,
    )
    assert torch.count_nonzero(forward.local.assignment[1, 1:]) == 0
    assert torch.count_nonzero(forward.local.assignment[1, :, 2:]) == 0
    assert torch.count_nonzero(forward.local.unmatched_a[1, 1:]) == 0
    assert torch.count_nonzero(forward.local.unmatched_b[1, 2:]) == 0


def test_canonical_coarse_preprocessing_is_strict_model_and_swap_compatible() -> None:
    overshoot_shape = np.zeros((3, 3), dtype=bool)
    overshoot_shape[1, 1] = True
    normal_shape = np.zeros((40, 52), dtype=bool)
    normal_shape[8:32, 14:38] = True
    config = GeometryBatchConfig()
    coarse_a = preprocess_coarse_mask(overshoot_shape, config).unsqueeze(0)
    coarse_b = preprocess_coarse_mask(normal_shape, config).unsqueeze(0)
    model = SymmetricCoarseSiamese(widths=(4,), embedding_dim=8, hidden_dim=8).eval()

    with torch.inference_mode():
        forward = model(coarse_a, coarse_b)
        swapped = model(coarse_b, coarse_a)

    assert forward.valid_problem.tolist() == [True]
    assert swapped.valid_problem.tolist() == [True]
    torch.testing.assert_close(forward.logit, swapped.logit, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(
        forward.probability, swapped.probability, rtol=0.0, atol=1e-7
    )

    # The model contract remains exact and was not relaxed to absorb the bug.
    noncanonical = coarse_a.clone()
    noncanonical[0, 0, 64, 64] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    with torch.inference_mode():
        assert model(noncanonical, coarse_b).valid_problem.tolist() == [False]


def test_nonfinite_valid_patch_abstains_without_poisoning_output() -> None:
    model = DunhuangPairwiseV02(_config("dustbin_sinkhorn")).eval()
    inputs = list(_inputs())
    inputs[2][0, 0, 0, 0, 0] = torch.nan
    output = model(*inputs)
    assert output.local_valid.tolist() == [False, True]
    assert output.requires_review.tolist() == [True, False]
    assert torch.isfinite(output.fused_probability).all()
    assert torch.count_nonzero(output.local.assignment[0]) == 0


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        -0.01,
        1.01,
    ],
)
def test_invalid_coarse_mask_is_neutral_and_review_only(invalid_value: float) -> None:
    model = DunhuangPairwiseV02(_config("dual_softmax")).eval()
    inputs = list(_inputs())
    inputs[0][0, 0, 0, 0] = invalid_value
    output = model(*inputs)

    assert output.coarse.valid_problem.tolist() == [False, True]
    assert output.coarse.logit[0].item() == 0.0
    assert output.coarse.probability[0].item() == 0.5
    assert torch.count_nonzero(output.coarse.embedding_a[0]) == 0
    assert torch.count_nonzero(output.coarse.embedding_b[0]) == 0
    assert output.requires_review[0].item()
    assert torch.isfinite(output.fused_probability).all()


@pytest.mark.parametrize("fill_value", [0.0, 1.0])
def test_constant_coarse_mask_is_neutral_and_review_only(fill_value: float) -> None:
    model = DunhuangPairwiseV02(_config("dual_softmax")).eval()
    inputs = list(_inputs())
    inputs[1][0].fill_(fill_value)
    output = model(*inputs)

    assert output.coarse.valid_problem.tolist() == [False, True]
    assert output.coarse.logit[0].item() == 0.0
    assert output.coarse.probability[0].item() == 0.5
    assert output.requires_review[0].item()


def test_nonconverged_sinkhorn_fails_closed() -> None:
    model = DunhuangPairwiseV02(
        _config(
            "dustbin_sinkhorn",
            sinkhorn_iterations=1,
            sinkhorn_tolerance=0.0,
            require_sinkhorn_convergence=True,
        )
    ).eval()
    output = model(*_inputs())
    assert not output.local.transport_converged.any().item()
    assert not output.local_valid.any().item()
    assert output.requires_review.all().item()


def test_direction_and_ragged_arc_mil_are_explicit_and_differentiable() -> None:
    direction_logits = torch.tensor(
        [[-1.0, 2.0, 0.0, -2.0], [float("nan"), 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    direction_valid = torch.tensor(
        [[True, True, True, True], [False, False, False, False]]
    )
    result = aggregate_direction_candidates(direction_logits, direction_valid)
    assert result.best_direction_name == ("b_right_of_a", None)
    assert result.pair_valid.tolist() == [True, False]
    assert torch.isfinite(result.pair_probability).all()

    arcs = torch.tensor([-2.0, 1.0, 2.0, -1.0, 0.5], requires_grad=True)
    hierarchical = aggregate_flat_arc_candidates(
        arcs,
        torch.ones(5, dtype=torch.bool),
        torch.tensor([0, 0, 0, 1, 1]),
        torch.tensor([0, 0, 2, 1, 3]),
        batch_size=2,
    )
    direction = hierarchical.direction_output
    assert direction.candidate_valid.tolist() == [
        [True, False, True, False],
        [False, True, False, True],
    ]
    direction.pair_logit.sum().backward()
    assert torch.isfinite(arcs.grad).all()
    assert float(arcs.grad.abs().sum()) > 0.0


def _pooled_arc_logit(values, config):
    count = len(values)
    output = aggregate_flat_arc_candidates(
        torch.tensor(values, dtype=torch.float64),
        torch.ones(count, dtype=torch.bool),
        torch.zeros(count, dtype=torch.long),
        torch.zeros(count, dtype=torch.long),
        batch_size=1,
        arc_pooling=config,
    )
    return output.direction_output.candidate_logits[0, 0]


def test_arc_pooling_ablations_make_candidate_count_bias_explicit() -> None:
    configs = (
        ArcPoolingConfig(ArcPoolingMode.LOG_MEAN_EXP, temperature=0.25),
        ArcPoolingConfig(ArcPoolingMode.MAX),
        ArcPoolingConfig(ArcPoolingMode.TOP_K_MEAN, top_k=2),
    )
    for config in configs:
        single = _pooled_arc_logit([1.75], config)
        identical = _pooled_arc_logit([1.75] * 7, config)
        torch.testing.assert_close(single, identical, rtol=0.0, atol=1e-12)

    base = [3.0, 2.0]
    low_tail = base + [-100.0] * 20
    lme = configs[0]
    maximum = configs[1]
    top_k = configs[2]
    assert _pooled_arc_logit(low_tail, lme) < _pooled_arc_logit(base, lme)
    torch.testing.assert_close(
        _pooled_arc_logit(low_tail, maximum),
        _pooled_arc_logit(base, maximum),
    )
    torch.testing.assert_close(
        _pooled_arc_logit(low_tail, top_k), _pooled_arc_logit(base, top_k)
    )
    # TOP_K is bounded against low-tail count, but repeated high candidates can
    # still change it.  This is an ablation to validate, not a selected winner.
    assert _pooled_arc_logit([3.0, 3.0, 2.0], top_k) > _pooled_arc_logit(base, top_k)


def test_forward_flat_candidates_accepts_arbitrary_arc_count() -> None:
    model = DunhuangPairwiseV02(_config("dual_softmax")).eval()
    coarse_a, coarse_b, local_a, local_b, mask_a, mask_b = _inputs()
    # Five arc candidates distributed over two samples and four directions.
    sample = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    direction = torch.tensor([0, 0, 2, 1, 3], dtype=torch.long)
    local_a = local_a.index_select(0, sample)
    local_b = local_b.index_select(0, sample)
    mask_a = mask_a.index_select(0, sample)
    mask_b = mask_b.index_select(0, sample)
    result = model.forward_flat_candidates(
        coarse_a,
        coarse_b,
        local_a,
        local_b,
        mask_a,
        mask_b,
        sample,
        direction,
        torch.tensor([True, False, True, True, True]),
    )
    assert result.arc_logits.shape == (5,)
    assert result.direction_output.pair_valid.tolist() == [True, True]
    assert result.arc_valid.tolist() == [True, False, True, True, True]
    assert result.coarse_output is not None
    assert result.arc_pairwise_output is not None
    assert result.score_source == "fused"


def test_local_only_ragged_scores_ignore_coarse_and_bypass_fusion() -> None:
    model = DunhuangPairwiseV02(_config("dual_softmax")).eval()
    coarse_a, coarse_b, local_a, local_b, mask_a, mask_b = _inputs()
    sample = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    direction = torch.tensor([0, 0, 2, 1, 3], dtype=torch.long)
    candidate_valid = torch.tensor([True, False, True, True, True])
    arguments = (
        local_a.index_select(0, sample),
        local_b.index_select(0, sample),
        mask_a.index_select(0, sample),
        mask_b.index_select(0, sample),
        sample,
        direction,
        candidate_valid,
    )
    calls = {"coarse": 0, "fusion": 0}

    def record_coarse(*_args: object) -> None:
        calls["coarse"] += 1

    def record_fusion(*_args: object) -> None:
        calls["fusion"] += 1

    handles = (
        model.coarse_model.register_forward_hook(record_coarse),
        model.fusion.register_forward_hook(record_fusion),
    )
    try:
        first = model.forward_flat_candidates(
            coarse_a, coarse_b, *arguments, score_source="local"
        )
        # These values are invalid for the coarse model by design.  A strict
        # local-only arm must neither inspect them nor change its output.
        changed_a = torch.full_like(coarse_a, float("nan"))
        changed_b = torch.full_like(coarse_b, -7.0)
        second = model.forward_flat_candidates(
            changed_a, changed_b, *arguments, score_source="local"
        )
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"coarse": 0, "fusion": 0}
    assert first.score_source == "local"
    assert first.coarse_output is None
    assert first.arc_pairwise_output is not None
    assert first.arc_pairwise_output.score_source == "local"
    torch.testing.assert_close(first.arc_logits, first.arc_pairwise_output.local_logit)
    torch.testing.assert_close(
        first.arc_pairwise_output.fused_logit,
        first.arc_pairwise_output.local_logit,
    )
    assert torch.equal(
        first.arc_valid,
        first.arc_pairwise_output.local.decision_valid & candidate_valid,
    )
    assert torch.equal(
        first.arc_training_valid,
        first.arc_pairwise_output.local.training_valid & candidate_valid,
    )
    torch.testing.assert_close(first.arc_logits, second.arc_logits)
    torch.testing.assert_close(
        first.direction_output.pair_logit, second.direction_output.pair_logit
    )


def test_local_only_dual_and_sinkhorn_arms_share_everything_before_assignment() -> None:
    dual_config = _config("dual_softmax")
    sinkhorn_config = _config("dustbin_sinkhorn")
    dual_dict = dual_config.to_dict()
    sinkhorn_dict = sinkhorn_config.to_dict()
    dual_dict["local"]["matcher_mode"] = "dustbin_sinkhorn"
    assert dual_dict == sinkhorn_dict

    torch.manual_seed(41)
    dual = DunhuangPairwiseV02(dual_config).eval()
    sinkhorn = DunhuangPairwiseV02(sinkhorn_config).eval()
    sinkhorn.load_state_dict(dual.state_dict())

    coarse_a, coarse_b, local_a, local_b, mask_a, mask_b = _inputs()
    sample = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    direction = torch.tensor([0, 0, 2, 1, 3], dtype=torch.long)
    arguments = (
        coarse_a,
        coarse_b,
        local_a.index_select(0, sample),
        local_b.index_select(0, sample),
        mask_a.index_select(0, sample),
        mask_b.index_select(0, sample),
        sample,
        direction,
    )
    dual_output = dual.forward_flat_candidates(*arguments, score_source="local")
    sinkhorn_output = sinkhorn.forward_flat_candidates(*arguments, score_source="local")
    assert dual_output.arc_pairwise_output is not None
    assert sinkhorn_output.arc_pairwise_output is not None
    dual_local = dual_output.arc_pairwise_output.local
    sinkhorn_local = sinkhorn_output.arc_pairwise_output.local

    # Shared weights, encoders, context, and affinity are byte-for-byte the
    # same.  The explicit matcher mode is the first algorithmic fork.
    torch.testing.assert_close(
        dual_local.token_features_a, sinkhorn_local.token_features_a
    )
    torch.testing.assert_close(
        dual_local.token_features_b, sinkhorn_local.token_features_b
    )
    torch.testing.assert_close(dual_local.affinity, sinkhorn_local.affinity)
    assert not torch.allclose(dual_local.assignment, sinkhorn_local.assignment)
    torch.testing.assert_close(dual_output.arc_logits, dual_local.logit)
    torch.testing.assert_close(sinkhorn_output.arc_logits, sinkhorn_local.logit)


def test_candidate_score_source_rejects_implicit_fallbacks() -> None:
    model = DunhuangPairwiseV02(_config("dual_softmax")).eval()
    coarse_a, coarse_b, local_a, local_b, mask_a, mask_b = _inputs()
    sample = torch.tensor([0, 1], dtype=torch.long)
    direction = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="score_source"):
        model.forward_flat_candidates(
            coarse_a,
            coarse_b,
            local_a,
            local_b,
            mask_a,
            mask_b,
            sample,
            direction,
            score_source="automatic",
        )


def test_forward_flat_candidates_handles_no_local_as_review_not_rejection() -> None:
    model = DunhuangPairwiseV02(_config("dual_softmax")).eval()
    coarse_a, coarse_b, _, _, _, _ = _inputs()
    result = model.forward_flat_candidates(
        coarse_a,
        coarse_b,
        torch.empty((0, 1, 3, 8, 8)),
        torch.empty((0, 1, 3, 8, 8)),
        torch.empty((0, 1), dtype=torch.bool),
        torch.empty((0, 1), dtype=torch.bool),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
    )
    assert result.coarse_output is not None
    assert result.coarse_output.valid_problem.tolist() == [True, True]
    assert result.direction_output.pair_valid.tolist() == [False, False]
    assert result.direction_output.pair_probability.tolist() == [0.5, 0.5]
    assert result.arc_pairwise_output is None


def test_invalid_direction_names_and_indices_fail_closed() -> None:
    with pytest.raises(ValueError, match="direction_names"):
        PairwiseModelConfig(direction_names=("left", "left", "above", "below"))
    with pytest.raises(ValueError, match="direction_index"):
        aggregate_flat_arc_candidates(
            torch.zeros(1),
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([4], dtype=torch.long),
            batch_size=1,
        )
