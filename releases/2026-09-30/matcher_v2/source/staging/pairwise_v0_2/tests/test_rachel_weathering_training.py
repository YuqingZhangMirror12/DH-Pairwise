"""E1 loss contract: clean parity, gap preservation and assignment gradients."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from staging.pairwise_v0_2.training.rachel_n512_loss import (
    RachelN512LossConfig, compute_rachel_n512_loss,
)
from staging.pairwise_v0_2.training.rachel_weathering_training import compute_weathering_loss


def fixture():
    assignment = torch.full((3, 4, 4), .1, requires_grad=True)
    valid = torch.ones(3, dtype=torch.bool)
    output = SimpleNamespace(
        assignment=assignment, unmatched_a=torch.full((3, 4), .6, requires_grad=True),
        unmatched_b=torch.full((3, 4), .6, requires_grad=True), training_valid=valid,
        fused_logit=torch.zeros(3, requires_grad=True), coarse_logit=torch.zeros(3, requires_grad=True),
        local_logit=torch.zeros(3, requires_grad=True), coarse=SimpleNamespace(valid_problem=valid),
        translation_hat_rc=torch.tensor([[5., 7.], [6., 9.], [4., 5.]], requires_grad=True),
        transport=SimpleNamespace(diagnostics=SimpleNamespace(
            row_residual_max=torch.zeros(3), col_residual_max=torch.zeros(3))))
    targets = torch.full((3, 4), -1, dtype=torch.long)
    targets[:2, :2] = torch.arange(2)
    args = (torch.tensor([1., 1., 0.]), targets, targets.clone(),
            torch.tensor([[1., 2.], [3., 4.], [0., 0.]]), torch.tensor([True, True, False]))
    return output, args


def test_clean_loss_exact_parity():
    output, args = fixture()
    old = compute_rachel_n512_loss(output, *args)
    new = compute_weathering_loss(output, *args, pose_supervision_enabled=args[-1])
    for field in old.__dataclass_fields__:
        left, right = getattr(old, field), getattr(new, field)
        assert torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right


def test_changed_positive_retains_assignment_but_not_zero_gap_pose_gradient():
    output, args = fixture()
    loss = compute_weathering_loss(output, *args, pose_supervision_enabled=torch.tensor([True, False, False]))
    assert loss.translation_count == 1
    assert loss.supervised_match_count == 4
    loss.total.backward()
    assert output.translation_hat_rc.grad[0].abs().sum() > 0
    assert output.translation_hat_rc.grad[1:].abs().sum() == 0
    assert output.assignment.grad[1].abs().sum() > 0
    assert output.fused_logit.grad[1].abs() > 0
    assert torch.isfinite(output.assignment.grad).all()


def test_all_changed_positives_keep_only_original_non_pose_terms():
    output, args = fixture()
    config = RachelN512LossConfig()
    old_no_pose = compute_rachel_n512_loss(output, *args, config=replace(config, translation_weight=0.))
    new = compute_weathering_loss(output, *args, pose_supervision_enabled=torch.zeros(3, dtype=torch.bool))
    torch.testing.assert_close(old_no_pose.total, new.total, atol=0, rtol=0)
    assert new.translation_count == 0
    assert new.translation_smooth_l1 == 0


def test_sidecar_cannot_enable_negative_pose_or_remove_required_correspondence():
    output, args = fixture()
    with pytest.raises(ValueError, match="negative"):
        compute_weathering_loss(output, *args, pose_supervision_enabled=torch.ones(3, dtype=torch.bool))
    bad = list(args)
    bad[1] = torch.full_like(args[1], -1)
    bad[2] = torch.full_like(args[2], -1)
    with pytest.raises(ValueError, match="no compact correspondence"):
        compute_weathering_loss(output, *bad, pose_supervision_enabled=torch.zeros(3, dtype=torch.bool))


def test_translation_valid_is_not_rewritten_and_original_gt_is_preserved():
    output, args = fixture()
    gt, valid = args[-2].clone(), args[-1].clone()
    compute_weathering_loss(output, *args, pose_supervision_enabled=torch.tensor([False, True, False]))
    assert torch.equal(args[-2], gt)
    assert torch.equal(args[-1], valid)
