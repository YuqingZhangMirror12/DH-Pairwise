import inspect

import torch

from staging.pairwise_v0_2.models.rachel_n512 import (
    RachelN512Config,
    RachelN512Pairwise,
    TransportSequenceHead,
)
from staging.pairwise_v0_2.training.rachel_n512_loss import (
    RachelN512LossConfig,
    compute_rachel_n512_loss,
)


def _config():
    return RachelN512Config(
        canvas_size=32,
        coarse_size=16,
        contour_cap=8,
        window_sizes_px=(4.0, 8.0),
        patch_size=8,
        feature_dim=16,
        num_heads=4,
        landmark_count=4,
        context_layers=1,
        evidence_dim=8,
        sinkhorn_iterations=20,
        sinkhorn_tolerance=1e-2,
        activation_checkpointing=False,
    )


def _inputs():
    first = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
    second = torch.zeros_like(first)
    first[:, :, 8:24, 4:12] = 1.0
    second[0, :, 8:24, 6:14] = 1.0
    second[1, :, 5:19, 18:27] = 1.0
    points_a = torch.tensor(
        [
            [8.0, 4.0],
            [8.0, 8.0],
            [8.0, 11.0],
            [12.0, 11.0],
            [23.0, 11.0],
            [23.0, 8.0],
            [23.0, 4.0],
            [12.0, 4.0],
        ]
    )
    points_b_positive = points_a + torch.tensor([0.0, 2.0])
    points_b_negative = torch.tensor(
        [
            [5.0, 18.0],
            [5.0, 22.0],
            [5.0, 26.0],
            [10.0, 26.0],
            [18.0, 26.0],
            [18.0, 22.0],
            [18.0, 18.0],
            [10.0, 18.0],
        ]
    )
    points_a = points_a[None].repeat(2, 1, 1)
    points_b = torch.stack((points_b_positive, points_b_negative))
    valid = torch.ones((2, 8), dtype=torch.bool)
    return first, second, points_a, points_b, valid


def test_forward_is_target_blind_finite_and_uses_full_contour():
    signature = inspect.signature(RachelN512Pairwise.forward)
    assert not any(
        word in name
        for name in signature.parameters
        for word in ("label", "target", "translation_gt", "direction")
    )
    model = RachelN512Pairwise(_config()).eval()
    mask_a, mask_b, points_a, points_b, valid = _inputs()
    with torch.no_grad():
        output = model(mask_a, mask_b, points_a, points_b, valid, valid)
    assert output.assignment.shape == (2, 8, 8)
    assert output.unmatched_a.shape == (2, 8)
    assert output.translation_hat_rc.shape == (2, 2)
    assert output.training_valid.all().item()
    assert torch.isfinite(output.fused_logit).all().item()
    patches = model.patch_sampler(mask_a, points_a, valid)
    assert patches.shape == (2, 8, 2, 1, 8, 8)
    assert set(torch.unique(patches).tolist()).issubset({0.0, 1.0})


def test_pair_score_is_swap_invariant_and_translation_is_antisymmetric():
    torch.manual_seed(7)
    model = RachelN512Pairwise(_config()).eval()
    mask_a, mask_b, points_a, points_b, valid = _inputs()
    with torch.no_grad():
        forward = model(mask_a, mask_b, points_a, points_b, valid, valid)
        swapped = model(mask_b, mask_a, points_b, points_a, valid, valid)
    torch.testing.assert_close(
        forward.fused_logit, swapped.fused_logit, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        forward.local_logit, swapped.local_logit, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        forward.assignment,
        swapped.assignment.transpose(1, 2),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        forward.translation_hat_rc,
        -swapped.translation_hat_rc,
        atol=2e-4,
        rtol=2e-4,
    )


def test_translation_consensus_suppresses_large_displacement_outliers():
    model = RachelN512Pairwise(_config())
    # Five matches support one translation while two lower-mass matches are
    # far-away outliers.  The robust estimator must improve on the raw mean.
    points_a = torch.tensor(
        [
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [4.0, 0.0],
                [6.0, 0.0],
                [8.0, 0.0],
                [10.0, 0.0],
                [12.0, 0.0],
            ]
        ]
    )
    expected = torch.tensor([[3.0, 5.0]])
    points_b = points_a + expected[:, None, :]
    points_b = points_b.clone()
    points_b[0, 5] = points_a[0, 5] + torch.tensor([22.0, -17.0])
    points_b[0, 6] = points_a[0, 6] + torch.tensor([-18.0, 24.0])
    assignment = torch.zeros((1, 7, 7), requires_grad=True)
    assignment.data[0, torch.arange(7), torch.arange(7)] = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.35, 0.35]
    )
    raw_displacement = points_b - points_a
    raw_mean = (assignment.diagonal(dim1=1, dim2=2)[:, :, None] * raw_displacement).sum(
        dim=1
    ) / assignment.diagonal(dim1=1, dim2=2).sum(dim=1)[:, None]
    robust, dispersion, _ = model._translation(assignment, points_a, points_b)
    assert torch.linalg.vector_norm(robust - expected) < torch.linalg.vector_norm(
        raw_mean - expected
    )
    assert torch.isfinite(dispersion).all().item()
    (robust.square().sum() + dispersion.sum()).backward()
    assert assignment.grad is not None
    assert torch.isfinite(assignment.grad).all().item()
    assert assignment.grad.abs().sum().item() > 0.0


def test_structured_evidence_distinguishes_contiguous_from_scattered_support():
    config = _config()
    head = TransportSequenceHead(config)
    affinity = torch.zeros((1, 8, 8))
    contiguous = torch.zeros_like(affinity)
    scattered = torch.zeros_like(affinity)
    contiguous[0, torch.arange(4), torch.arange(4)] = 1.0
    indices = torch.tensor([0, 2, 4, 6])
    scattered[0, indices, indices] = 1.0
    valid = torch.ones((1, 8), dtype=torch.bool)
    unmatched_contiguous = 1.0 - contiguous.sum(dim=2)
    unmatched_scattered = 1.0 - scattered.sum(dim=2)
    _, evidence_contiguous = head(
        affinity,
        contiguous,
        unmatched_contiguous,
        unmatched_contiguous,
        valid,
        valid,
        torch.zeros(1),
        32,
    )
    _, evidence_scattered = head(
        affinity,
        scattered,
        unmatched_scattered,
        unmatched_scattered,
        valid,
        valid,
        torch.zeros(1),
        32,
    )
    assert evidence_contiguous[0, 5] > evidence_scattered[0, 5]


def test_compact_partial_assignment_and_translation_loss_backward():
    torch.manual_seed(9)
    model = RachelN512Pairwise(_config()).train()
    mask_a, mask_b, points_a, points_b, valid = _inputs()
    output = model(mask_a, mask_b, points_a, points_b, valid, valid)
    target_a = torch.full((2, 8), -1, dtype=torch.long)
    target_b = torch.full((2, 8), -1, dtype=torch.long)
    target_a[0] = torch.arange(8)
    target_b[0] = torch.arange(8)
    labels = torch.tensor([1.0, 0.0])
    translation_target = torch.tensor([[0.0, 2.0], [0.0, 0.0]])
    translation_valid = torch.tensor([True, False])
    loss = compute_rachel_n512_loss(
        output,
        labels,
        target_a,
        target_b,
        translation_target,
        translation_valid,
    )
    assert loss.valid_pair_count == 2
    assert loss.supervised_match_count == 8
    assert loss.supervised_dustbin_a_count == 8
    assert loss.supervised_dustbin_b_count == 8
    assert loss.translation_count == 1
    assert torch.isfinite(loss.total).item()
    loss.total.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all().item() for value in gradients)
    assert any(value.abs().sum().item() > 0.0 for value in gradients)


def test_formal_training_fast_loss_matches_checked_loss_without_cpu_counts():
    torch.manual_seed(11)
    model = RachelN512Pairwise(_config()).train()
    mask_a, mask_b, points_a, points_b, valid = _inputs()
    output = model(mask_a, mask_b, points_a, points_b, valid, valid)
    target_a = torch.full((2, 8), -1, dtype=torch.long)
    target_b = torch.full((2, 8), -1, dtype=torch.long)
    target_a[0] = torch.arange(8)
    target_b[0] = torch.arange(8)
    labels = torch.tensor([1.0, 0.0])
    translation_target = torch.tensor([[0.0, 2.0], [0.0, 0.0]])
    translation_valid = torch.tensor([True, False])
    checked = compute_rachel_n512_loss(
        output,
        labels,
        target_a,
        target_b,
        translation_target,
        translation_valid,
    )
    fast = compute_rachel_n512_loss(
        output,
        labels,
        target_a,
        target_b,
        translation_target,
        translation_valid,
        RachelN512LossConfig(
            validate_runtime_targets=False,
            collect_cpu_diagnostics=False,
        ),
    )
    torch.testing.assert_close(fast.total, checked.total)
    torch.testing.assert_close(fast.assignment_nll, checked.assignment_nll)
    torch.testing.assert_close(
        fast.translation_smooth_l1, checked.translation_smooth_l1
    )
    assert fast.valid_pair_count == -1
    assert fast.supervised_match_count == -1
