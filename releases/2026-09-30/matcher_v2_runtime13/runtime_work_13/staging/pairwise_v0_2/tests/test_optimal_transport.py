import pytest
import torch

from staging.pairwise_v0_2.models.optimal_transport import dustbin_sinkhorn


def _assert_unit_real_marginals(result, mask_a, mask_b, atol=2e-4):
    row_mass = result.real_transport.sum(dim=2) + result.dustbin_col
    col_mass = result.real_transport.sum(dim=1) + result.dustbin_row
    torch.testing.assert_close(
        row_mass,
        mask_a.to(row_mass.dtype),
        rtol=0.0,
        atol=atol,
    )
    torch.testing.assert_close(
        col_mass,
        mask_b.to(col_mass.dtype),
        rtol=0.0,
        atol=atol,
    )


def test_balanced_dustbin_marginals_and_diagnostics() -> None:
    affinity = torch.zeros((1, 2, 3), dtype=torch.float64)
    mask_a = torch.ones((1, 2), dtype=torch.bool)
    mask_b = torch.ones((1, 3), dtype=torch.bool)

    result = dustbin_sinkhorn(
        affinity,
        mask_a,
        mask_b,
        num_iterations=200,
        tolerance=1e-8,
    )

    assert result.real_transport.shape == (1, 2, 3)
    assert result.dustbin_row.shape == (1, 3)
    assert result.dustbin_col.shape == (1, 2)
    assert result.dustbin_corner.shape == (1,)
    _assert_unit_real_marginals(result, mask_a, mask_b, atol=1e-8)
    torch.testing.assert_close(
        result.dustbin_row.sum(1) + result.dustbin_corner,
        torch.tensor([3.0], dtype=torch.float64),
        rtol=0.0,
        atol=1e-8,
    )
    torch.testing.assert_close(
        result.dustbin_col.sum(1) + result.dustbin_corner,
        torch.tensor([2.0], dtype=torch.float64),
        rtol=0.0,
        atol=1e-8,
    )
    assert result.diagnostics.valid_problem.tolist() == [True]
    assert result.diagnostics.converged.tolist() == [True]
    assert result.diagnostics.iteration_count.tolist() == [200]
    assert result.diagnostics.matched_mass.item() > 0.0


def test_variable_valid_lengths_zero_padded_tokens() -> None:
    torch.manual_seed(7)
    affinity = torch.randn((2, 4, 5), dtype=torch.float64)
    mask_a = torch.tensor([[True, True, True, False], [True, False, True, False]])
    mask_b = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, True, False],
        ]
    )

    result = dustbin_sinkhorn(
        affinity,
        mask_a,
        mask_b,
        dustbin_score=-0.25,
        num_iterations=200,
    )

    _assert_unit_real_marginals(result, mask_a, mask_b)
    assert torch.count_nonzero(result.real_transport[0, 3]) == 0
    assert torch.count_nonzero(result.real_transport[0, :, 2:]) == 0
    assert torch.count_nonzero(result.real_transport[1, 1]) == 0
    assert torch.count_nonzero(result.real_transport[1, 3]) == 0
    assert torch.count_nonzero(result.real_transport[1, :, 4]) == 0
    assert result.diagnostics.valid_a_count.tolist() == [3, 2]
    assert result.diagnostics.valid_b_count.tolist() == [2, 4]
    assert result.diagnostics.valid_problem.tolist() == [True, True]


def test_nonfinite_values_in_padded_cells_are_ignored() -> None:
    affinity = torch.randn((1, 3, 4), dtype=torch.float64)
    mask_a = torch.tensor([[True, True, False]])
    mask_b = torch.tensor([[True, False, True, False]])
    affinity[0, 2, :] = torch.nan
    affinity[0, :, 1] = torch.inf
    affinity[0, :, 3] = torch.nan

    result = dustbin_sinkhorn(affinity, mask_a, mask_b, num_iterations=200)

    assert result.diagnostics.input_is_usable.tolist() == [True]
    assert result.diagnostics.valid_problem.tolist() == [True]
    assert result.diagnostics.finite_output.tolist() == [True]
    _assert_unit_real_marginals(result, mask_a, mask_b)


def test_swapping_sequences_transposes_augmented_solution() -> None:
    torch.manual_seed(11)
    affinity = torch.randn((2, 4, 3), dtype=torch.float64)
    mask_a = torch.tensor([[True, True, False, True], [True, False, True, False]])
    mask_b = torch.tensor([[True, False, True], [True, True, True]])
    kwargs = {
        "dustbin_score": torch.tensor(-0.4, dtype=torch.float64),
        "temperature": 0.7,
        "num_iterations": 250,
        "tolerance": 1e-7,
    }

    forward = dustbin_sinkhorn(affinity, mask_a, mask_b, **kwargs)
    swapped = dustbin_sinkhorn(affinity.transpose(1, 2), mask_b, mask_a, **kwargs)

    torch.testing.assert_close(
        forward.real_transport,
        swapped.real_transport.transpose(1, 2),
        rtol=1e-7,
        atol=1e-8,
    )
    torch.testing.assert_close(
        forward.dustbin_col, swapped.dustbin_row, rtol=1e-7, atol=1e-8
    )
    torch.testing.assert_close(
        forward.dustbin_row, swapped.dustbin_col, rtol=1e-7, atol=1e-8
    )
    torch.testing.assert_close(
        forward.dustbin_corner,
        swapped.dustbin_corner,
        rtol=1e-7,
        atol=1e-8,
    )


def test_zero_length_side_and_nonfinite_sample_fail_closed_in_batch() -> None:
    affinity = torch.zeros((4, 2, 3), dtype=torch.float64)
    affinity[3, 0, 0] = torch.nan
    mask_a = torch.tensor(
        [
            [True, True],
            [False, False],
            [False, False],
            [True, True],
        ]
    )
    mask_b = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [False, False, False],
            [True, True, True],
        ]
    )

    result = dustbin_sinkhorn(affinity, mask_a, mask_b)

    assert result.diagnostics.valid_problem.tolist() == [True, False, False, False]
    assert result.diagnostics.input_is_usable.tolist() == [True, True, True, False]
    assert result.diagnostics.converged.tolist()[1:] == [False, False, False]
    assert result.diagnostics.iteration_count.tolist()[1:] == [0, 0, 0]
    assert result.diagnostics.finite_output.tolist() == [True, True, True, True]
    assert torch.count_nonzero(result.real_transport[1:]) == 0
    assert torch.count_nonzero(result.dustbin_row[1:]) == 0
    assert torch.count_nonzero(result.dustbin_col[1:]) == 0
    assert torch.count_nonzero(result.dustbin_corner[1:]) == 0


def test_extreme_logits_and_forbidden_pairs_remain_finite() -> None:
    affinity = torch.tensor(
        [[[1.0e4, -1.0e4, -torch.inf], [-1.0e4, 1.0e4, -torch.inf]]],
        dtype=torch.float64,
    )

    result = dustbin_sinkhorn(
        affinity,
        dustbin_score=-100.0,
        temperature=0.1,
        num_iterations=300,
    )

    assert result.diagnostics.valid_problem.tolist() == [True]
    assert result.diagnostics.finite_output.tolist() == [True]
    assert torch.isfinite(result.real_transport).all()
    assert torch.isfinite(result.dustbin_row).all()
    assert torch.isfinite(result.dustbin_col).all()
    assert result.real_transport[0, 0, 2].item() == 0.0
    assert result.real_transport[0, 1, 2].item() == 0.0


def test_gradients_reach_affinity_and_learnable_dustbin() -> None:
    affinity = torch.tensor(
        [[[1.0, -0.2, 0.3], [0.1, 0.8, -0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    dustbin_score = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)
    result = dustbin_sinkhorn(
        affinity,
        dustbin_score=dustbin_score,
        temperature=0.8,
        num_iterations=150,
    )
    weights = torch.tensor([[[1.0, -0.3, 0.7], [-0.4, 0.8, 0.2]]], dtype=torch.float64)
    loss = (
        (result.real_transport * weights).sum()
        + 0.17 * result.dustbin_col[0, 0]
        - 0.11 * result.dustbin_row[0, 2]
    )

    loss.backward()

    assert affinity.grad is not None
    assert torch.isfinite(affinity.grad).all()
    assert torch.count_nonzero(affinity.grad) > 0
    assert dustbin_score.grad is not None
    assert torch.isfinite(dustbin_score.grad)
    assert dustbin_score.grad.abs().item() > 0.0


def test_masked_backward_is_finite_and_padding_has_zero_gradient() -> None:
    torch.manual_seed(19)
    affinity = torch.randn((2, 4, 5), dtype=torch.float64, requires_grad=True)
    mask_a = torch.tensor([[True, False, True, False], [True, True, True, False]])
    mask_b = torch.tensor(
        [
            [True, True, False, True, False],
            [True, False, True, False, True],
        ]
    )
    dustbin_score = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    result = dustbin_sinkhorn(
        affinity,
        mask_a,
        mask_b,
        dustbin_score=dustbin_score,
        num_iterations=150,
    )
    weights = torch.randn_like(result.real_transport)

    (result.real_transport * weights).sum().backward()

    assert affinity.grad is not None
    assert torch.isfinite(affinity.grad).all()
    assert dustbin_score.grad is not None
    assert torch.isfinite(dustbin_score.grad)
    active_cells = mask_a[:, :, None] & mask_b[:, None, :]
    assert torch.count_nonzero(affinity.grad[active_cells]) > 0
    assert torch.count_nonzero(affinity.grad[~active_cells]) == 0


def test_stable_default_profile_long_bounded_sequence_forward_backward() -> None:
    """The model's 0.25/100 baseline remains healthy on ragged long arcs."""

    generator = torch.Generator().manual_seed(77)
    affinity = (
        torch.rand((1, 128, 192), generator=generator) * 2.0 - 1.0
    ).requires_grad_()
    mask_a = torch.ones((1, 128), dtype=torch.bool)
    mask_b = torch.ones((1, 192), dtype=torch.bool)
    mask_a[:, 113:] = False
    mask_b[:, 157:] = False
    result = dustbin_sinkhorn(
        affinity,
        mask_a,
        mask_b,
        temperature=0.25,
        num_iterations=100,
        tolerance=1e-3,
    )

    assert result.diagnostics.converged.tolist() == [True]
    assert result.diagnostics.row_residual_max.item() <= 1e-3
    assert result.diagnostics.col_residual_max.item() <= 1e-3
    weight = torch.linspace(-1.0, 1.0, 128 * 192).reshape(1, 128, 192)
    loss = (result.real_transport * weight).sum() + 0.1 * result.dustbin_col.sum()
    loss.backward()
    assert affinity.grad is not None
    assert torch.isfinite(affinity.grad).all()
    assert torch.count_nonzero(affinity.grad) > 0
    active = mask_a[:, :, None] & mask_b[:, None, :]
    assert torch.count_nonzero(affinity.grad[~active]) == 0


@pytest.mark.parametrize(
    "affinity, error_type",
    [
        (torch.zeros((2, 3)), ValueError),
        (torch.zeros((1, 2, 3), dtype=torch.int64), TypeError),
        (torch.zeros((0, 2, 3)), ValueError),
    ],
)
def test_invalid_affinity_contract(affinity, error_type) -> None:
    with pytest.raises(error_type):
        dustbin_sinkhorn(affinity)


def test_invalid_mask_contract() -> None:
    affinity = torch.zeros((1, 2, 3))
    with pytest.raises(TypeError, match="torch.bool"):
        dustbin_sinkhorn(affinity, torch.ones((1, 2)), None)
    with pytest.raises(ValueError, match="shape"):
        dustbin_sinkhorn(
            affinity,
            torch.ones((1, 3), dtype=torch.bool),
            None,
        )
