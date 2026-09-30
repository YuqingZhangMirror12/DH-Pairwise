from __future__ import annotations

import copy

import pytest
import torch

import staging.pairwise_v0_2.models.local_matcher as local_matcher_module
import staging.pairwise_v0_2.models.optimal_transport as transport_module
from staging.pairwise_v0_2.models.local_matcher import (
    MatcherMode,
    OrderedLocalMatcher,
    SinkhornTrainingPolicy,
)


def _matcher(mode: MatcherMode) -> OrderedLocalMatcher:
    return OrderedLocalMatcher(
        input_channels=3,
        feature_dim=16,
        num_heads=4,
        ff_dim=24,
        matcher_mode=mode,
        matcher_temperature=0.25,
        sinkhorn_iterations=8,
        sinkhorn_tolerance=1.0,
        require_sinkhorn_convergence=True,
        sinkhorn_training_policy=SinkhornTrainingPolicy.FINITE_WITH_RESIDUAL,
        dropout=0.0,
    )


def _inputs(*, height: int = 8, width: int = 8):
    generator = torch.Generator().manual_seed(20260830)
    patches_a = torch.randn((2, 5, 3, height, width), generator=generator)
    patches_b = torch.randn((2, 4, 3, height, width), generator=generator)
    mask_a = torch.tensor(
        [[True, True, True, True, False], [True, True, True, False, False]]
    )
    mask_b = torch.tensor([[True, True, True, False], [True, True, True, True]])
    correspondence = mask_a[:, :, None] & mask_b[:, None, :]
    return patches_a, patches_b, mask_a, mask_b, correspondence


def _loss(output) -> torch.Tensor:
    assignment_weight = torch.linspace(
        -0.3,
        0.4,
        output.assignment.numel(),
        dtype=output.assignment.dtype,
        device=output.assignment.device,
    ).reshape_as(output.assignment)
    return (
        output.logit.sum()
        + 0.03 * (output.assignment * assignment_weight).sum()
        + 0.02 * output.unmatched_a.sum()
        - 0.01 * output.unmatched_b.sum()
        + 0.01 * output.token_features_a.square().mean()
        + 0.01 * output.token_features_b.square().mean()
    )


def _forward_backward(model: OrderedLocalMatcher, inputs):
    output = model(*inputs)
    loss = _loss(output)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output, loss.detach(), gradients


@pytest.mark.parametrize(
    "mode", (MatcherMode.DUAL_SOFTMAX, MatcherMode.DUSTBIN_SINKHORN)
)
def test_training_checkpoint_preserves_forward_and_parameter_gradients(mode) -> None:
    torch.manual_seed(14)
    template = _matcher(mode)
    checkpointed = copy.deepcopy(template).train()
    uncheckpointed = copy.deepcopy(template).eval()
    inputs = _inputs()

    observed, observed_loss, observed_gradients = _forward_backward(
        checkpointed, inputs
    )
    expected, expected_loss, expected_gradients = _forward_backward(
        uncheckpointed, inputs
    )

    for name in (
        "logit",
        "affinity",
        "assignment",
        "unmatched_a",
        "unmatched_b",
        "token_features_a",
        "token_features_b",
        "row_residual_max",
        "col_residual_max",
    ):
        torch.testing.assert_close(
            getattr(observed, name), getattr(expected, name), rtol=0.0, atol=0.0
        )
    torch.testing.assert_close(observed_loss, expected_loss, rtol=0.0, atol=0.0)
    assert observed_gradients.keys() == expected_gradients.keys()
    for name in observed_gradients:
        torch.testing.assert_close(
            observed_gradients[name],
            expected_gradients[name],
            rtol=0.0,
            atol=0.0,
        )


def test_checkpoint_is_non_reentrant_training_only(monkeypatch) -> None:
    patch_calls = []
    sinkhorn_calls = []
    original_patch_checkpoint = local_matcher_module.activation_checkpoint
    original_sinkhorn_checkpoint = transport_module.activation_checkpoint

    def record_patch(function, *args, **kwargs):
        patch_calls.append(dict(kwargs))
        return original_patch_checkpoint(function, *args, **kwargs)

    def record_sinkhorn(function, *args, **kwargs):
        sinkhorn_calls.append(dict(kwargs))
        return original_sinkhorn_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(local_matcher_module, "activation_checkpoint", record_patch)
    monkeypatch.setattr(transport_module, "activation_checkpoint", record_sinkhorn)
    matcher = _matcher(MatcherMode.DUSTBIN_SINKHORN)
    inputs = _inputs()

    matcher.train()
    matcher(*inputs)
    assert len(patch_calls) == 2
    assert len(sinkhorn_calls) == 1
    assert all(call["use_reentrant"] is False for call in patch_calls + sinkhorn_calls)
    assert all(
        call["preserve_rng_state"] is False for call in patch_calls + sinkhorn_calls
    )

    patch_calls.clear()
    sinkhorn_calls.clear()
    matcher.eval()
    matcher(*inputs)
    assert patch_calls == []
    assert sinkhorn_calls == []

    matcher.train()
    with torch.no_grad():
        matcher(*inputs)
    assert patch_calls == []
    assert sinkhorn_calls == []

    # Dual-softmax already fits the established runtime envelope and keeps its
    # original forward/backward path without recomputation overhead.
    _matcher(MatcherMode.DUAL_SOFTMAX).train()(*inputs)
    assert patch_calls == []
    assert sinkhorn_calls == []


def _logical_saved_tensor_bytes(model: OrderedLocalMatcher, inputs) -> int:
    saved_bytes = 0

    def pack(value: torch.Tensor) -> torch.Tensor:
        nonlocal saved_bytes
        saved_bytes += value.numel() * value.element_size()
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = model(*inputs)
        loss = _loss(output)
    loss.backward()
    return saved_bytes


def test_checkpoint_substantially_reduces_logical_saved_tensor_bytes() -> None:
    torch.manual_seed(18)
    template = _matcher(MatcherMode.DUSTBIN_SINKHORN)
    inputs = _inputs(height=16, width=16)

    uncheckpointed = _logical_saved_tensor_bytes(copy.deepcopy(template).eval(), inputs)
    checkpointed = _logical_saved_tensor_bytes(copy.deepcopy(template).train(), inputs)

    assert checkpointed < uncheckpointed // 5
