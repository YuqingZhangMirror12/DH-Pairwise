from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from staging.pairwise_v0_2.models.frozen_transport_matrix_readout import (
    FrozenTransportMatrixReadout,
)
from staging.pairwise_v0_2.training.frozen_transport_matrix_adapter import (
    frozen_transport_matrix_fit_step,
)


def _permutation_plan(permutation: tuple[int, ...]) -> torch.Tensor:
    plan = torch.zeros((len(permutation), len(permutation)), dtype=torch.float32)
    for row, column in enumerate(permutation):
        plan[row, column] = 1.0
    return plan


def _full_inputs(plans: torch.Tensor):
    count, length_a, length_b = plans.shape
    return {
        "assignment": plans,
        "affinity": plans * 0.75,
        "unmatched_a": torch.zeros((count, length_a), dtype=plans.dtype),
        "unmatched_b": torch.zeros((count, length_b), dtype=plans.dtype),
        "token_mask_a": torch.ones((count, length_a), dtype=torch.bool),
        "token_mask_b": torch.ones((count, length_b), dtype=torch.bool),
        "correspondence_mask": torch.ones(
            (count, length_a, length_b), dtype=torch.bool
        ),
        "base_arc_logit": torch.zeros((count,), dtype=plans.dtype),
    }


def test_three_layer_head_scores_coherent_streaks_above_same_mass_shuffle() -> None:
    torch.manual_seed(17)
    diagonal = _permutation_plan(tuple(range(8)))
    anti_diagonal = _permutation_plan(tuple(reversed(range(8))))
    # Every adjacent row moves by more than one column; the inverse
    # permutation has the same property, so neither symmetric view contains a
    # spurious one-pixel diagonal run.
    shuffled = _permutation_plan((0, 2, 4, 6, 1, 3, 5, 7))
    plans = torch.stack((diagonal, anti_diagonal, shuffled))
    head = FrozenTransportMatrixReadout(
        grid_size=8, widths=(5, 7, 9), embedding_dim=11
    ).eval()

    output = head(**_full_inputs(plans))
    assert sum(isinstance(layer, nn.Conv2d) for layer in head.matrix_cnn) == 3
    assert output.continuity[0].item() == pytest.approx(7.0 / 8.0)
    assert output.continuity[1].item() == pytest.approx(7.0 / 8.0)
    assert output.continuity[2].item() == pytest.approx(0.0)
    assert output.arc_logit[0] > output.arc_logit[2] + 0.5
    assert output.arc_logit[1] > output.arc_logit[2] + 0.5


def _multi_scale_inputs():
    torch.manual_seed(23)
    count, length_a, length_b = 2, 5, 7
    mask_a = torch.tensor(
        [[True, True, True, True, False], [True, True, True, False, False]]
    )
    mask_b = torch.tensor(
        [
            [True, True, True, True, True, True, False],
            [True, True, True, True, True, False, False],
        ]
    )
    scale_a = torch.tensor([[0, 0, 1, 1, -1], [0, 0, 1, -1, -1]])
    scale_b = torch.tensor([[0, 0, 0, 1, 1, 1, -1], [0, 0, 1, 1, 1, -1, -1]])
    allowed = (
        (scale_a[:, :, None] == scale_b[:, None, :])
        & mask_a[:, :, None]
        & mask_b[:, None, :]
    )
    return {
        "assignment": torch.rand((count, length_a, length_b)) * allowed,
        "affinity": (2.0 * torch.rand((count, length_a, length_b)) - 1.0) * allowed,
        "unmatched_a": torch.rand((count, length_a)) * mask_a,
        "unmatched_b": torch.rand((count, length_b)) * mask_b,
        "token_mask_a": mask_a,
        "token_mask_b": mask_b,
        "correspondence_mask": allowed,
        "base_arc_logit": torch.tensor((0.2, -0.4)),
    }


def test_swap_parity_with_unequal_lengths_and_multiple_scale_blocks() -> None:
    head = FrozenTransportMatrixReadout(
        grid_size=10, widths=(4, 6, 8), embedding_dim=10
    ).eval()
    inputs = _multi_scale_inputs()
    forward = head(**inputs)
    swapped = head(
        inputs["assignment"].transpose(1, 2),
        inputs["affinity"].transpose(1, 2),
        inputs["unmatched_b"],
        inputs["unmatched_a"],
        inputs["token_mask_b"],
        inputs["token_mask_a"],
        inputs["correspondence_mask"].transpose(1, 2),
        base_arc_logit=inputs["base_arc_logit"],
    )
    torch.testing.assert_close(forward.arc_logit, swapped.arc_logit)
    torch.testing.assert_close(forward.correction, swapped.correction)
    torch.testing.assert_close(forward.embedding, swapped.embedding)
    torch.testing.assert_close(forward.continuity, swapped.continuity)
    assert torch.equal(forward.transport_valid, swapped.transport_valid)


def test_padding_and_forbidden_cross_scale_values_cannot_change_score() -> None:
    head = FrozenTransportMatrixReadout(
        grid_size=12, widths=(4, 6, 8), embedding_dim=9
    ).eval()
    inputs = {name: value[:1].clone() for name, value in _multi_scale_inputs().items()}
    reference = head(**inputs)

    # Disabled cross-scale cells are explicitly not model evidence.
    corrupted = {name: value.clone() for name, value in inputs.items()}
    forbidden = ~corrupted["correspondence_mask"]
    corrupted["assignment"][forbidden] = 1000.0
    corrupted["affinity"][forbidden] = -1000.0
    ignored = head(**corrupted)
    torch.testing.assert_close(reference.arc_logit, ignored.arc_logit)

    padded_assignment = torch.full((1, 9, 10), float("nan"))
    padded_affinity = torch.full((1, 9, 10), float("inf"))
    padded_unmatched_a = torch.full((1, 9), float("nan"))
    padded_unmatched_b = torch.full((1, 10), float("inf"))
    padded_mask_a = torch.zeros((1, 9), dtype=torch.bool)
    padded_mask_b = torch.zeros((1, 10), dtype=torch.bool)
    padded_allowed = torch.zeros((1, 9, 10), dtype=torch.bool)
    length_a = inputs["assignment"].shape[1]
    length_b = inputs["assignment"].shape[2]
    padded_assignment[:, :length_a, :length_b] = inputs["assignment"]
    padded_affinity[:, :length_a, :length_b] = inputs["affinity"]
    padded_unmatched_a[:, :length_a] = inputs["unmatched_a"]
    padded_unmatched_b[:, :length_b] = inputs["unmatched_b"]
    padded_mask_a[:, :length_a] = inputs["token_mask_a"]
    padded_mask_b[:, :length_b] = inputs["token_mask_b"]
    padded_allowed[:, :length_a, :length_b] = inputs["correspondence_mask"]
    padded = head(
        padded_assignment,
        padded_affinity,
        padded_unmatched_a,
        padded_unmatched_b,
        padded_mask_a,
        padded_mask_b,
        padded_allowed,
        base_arc_logit=inputs["base_arc_logit"],
    )
    torch.testing.assert_close(reference.arc_logit, padded.arc_logit)
    torch.testing.assert_close(reference.embedding, padded.embedding)
    torch.testing.assert_close(reference.continuity, padded.continuity)


def test_gradient_reaches_matrix_evidence_and_every_trainable_layer() -> None:
    torch.manual_seed(31)
    assignment = torch.rand((2, 6, 6), requires_grad=True)
    affinity = torch.rand((2, 6, 6), requires_grad=True)
    unmatched_a = torch.rand((2, 6), requires_grad=True)
    unmatched_b = torch.rand((2, 6), requires_grad=True)
    inputs = _full_inputs(assignment)
    inputs.update(
        {
            "affinity": affinity,
            "unmatched_a": unmatched_a,
            "unmatched_b": unmatched_b,
        }
    )
    head = FrozenTransportMatrixReadout(grid_size=8, widths=(4, 6, 8), embedding_dim=10)
    output = head(**inputs)
    output.arc_logit.sum().backward()

    for value in (assignment, affinity, unmatched_a, unmatched_b):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all().item()
        assert value.grad.abs().sum().item() > 0.0
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all().item()


def test_minimal_fit_adapter_uses_targets_only_after_target_free_forward() -> None:
    signature = inspect.signature(FrozenTransportMatrixReadout.forward)
    assert all(
        "target" not in name and "direction" not in name
        for name in signature.parameters
    )

    diagonal = _permutation_plan(tuple(range(6)))
    shuffled = _permutation_plan((0, 2, 4, 1, 3, 5))
    plans = torch.stack((diagonal,) * 4 + (shuffled,) * 4)
    inputs = _full_inputs(plans)
    head = FrozenTransportMatrixReadout(grid_size=8, widths=(4, 6, 8), embedding_dim=10)
    result = frozen_transport_matrix_fit_step(
        head,
        **inputs,
        arc_decision_valid=torch.ones(8, dtype=torch.bool),
        arc_training_valid=torch.ones(8, dtype=torch.bool),
        sample_index=torch.tensor((0, 0, 0, 0, 1, 1, 1, 1)),
        direction_index=torch.tensor((0, 1, 2, 3, 0, 1, 2, 3)),
        geometry_valid=torch.ones(2, dtype=torch.bool),
        pair_label=torch.tensor((True, False)),
        direction_target=torch.tensor((0, -1)),
        direction_target_valid=torch.tensor((True, False)),
    )
    assert result.loss.ndim == 0
    assert torch.isfinite(result.loss).item()
    assert result.hierarchy.direction_output.candidate_logits.shape == (2, 4)
    result.loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in head.parameters()
    )
