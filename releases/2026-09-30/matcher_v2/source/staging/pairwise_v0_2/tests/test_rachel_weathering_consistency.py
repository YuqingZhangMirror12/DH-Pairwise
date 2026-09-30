"""Focused CPU E2 contracts; no source dataset reads, remote access or training."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from staging.pairwise_v0_2.pairwise_data import rachel_weathered_dataset as weather
from staging.pairwise_v0_2.tests.test_rachel_weathered_dataset import sample
from staging.pairwise_v0_2.tests.test_rachel_weathering_training import fixture
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from staging.pairwise_v0_2.training.rachel_weathering_training import compute_weathering_loss
from staging.pairwise_v0_2.training.rachel_weathering_consistency import (
    E2Dataset, E2LossConfig, E2Sample, aligned_assignment_kl, clean_teacher_forward,
    collate_e2_pairs, compute_e2_loss, make_e2_loader, source_corrected_gap_loss, train_e2_epoch,
)


def test_e2_exact_e1_student_one_decode_and_ancestry_sidecars(tmp_path):
    clean = sample()

    class CountingBase:
        calls = 0

        def __len__(self):
            return 1

        def __getitem__(self, index):
            self.calls += 1
            return clean

    base = CountingBase()
    kwargs = dict(seed=260909, epoch=1, cache_dir=tmp_path, clean_probability=0., mild_probability=1.)
    expected, report = weather.RachelWeatheredDataset([clean], **kwargs)[0]
    dataset = E2Dataset(base, **kwargs)
    actual = dataset[0]
    assert base.calls == 1 and actual.clean is clean and actual.report == report
    assert report["changed_pair"]
    for name, value in vars(expected).items():
        other = getattr(actual.student, name)
        assert np.array_equal(value, other) if isinstance(value, np.ndarray) else value == other
    wrapped = collate_e2_pairs([actual])
    np.testing.assert_array_equal(wrapped.batch.translation_a_to_b_rc, wrapped.clean_batch.translation_a_to_b_rc)
    assert not wrapped.pose_supervision_enabled[0]
    assert not hasattr(wrapped.batch, "ancestor_a")  # No new student-model input.
    for side in "ab":
        ancestor = getattr(actual, "ancestor_" + side)
        assert len(np.unique(ancestor[ancestor >= 0])) == np.count_nonzero(ancestor >= 0)
    dataset.set_epoch(2)
    assert dataset.epoch == 2


def test_e2_fallback_uses_clean_identity_not_attempted_corrupt_mapping():
    clean = sample()
    dataset = E2Dataset([clean], clean_probability=0., mild_probability=1.)
    with patch.object(weather, "inherit_pair_targets", side_effect=lambda sample, a, b:
                      (np.full(len(a.points), -2), np.full(len(b.points), -2))):
        result = dataset[0]
    assert result.report["fallback_reason"] and not result.report["changed_pair"]
    assert result.student is result.clean is clean
    np.testing.assert_array_equal(result.ancestor_a, np.arange(len(clean.points_rc_a)))
    bad = replace(result, ancestor_a=np.zeros_like(result.ancestor_a))
    with pytest.raises(ValueError, match="injective"):
        collate_e2_pairs([bad])


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.dropout = torch.nn.Dropout(.1)
        self.calls = []

    def forward(self, mask_a, mask_b, points_a, points_b, valid_a, valid_b):
        self.calls.append((self.training, torch.is_grad_enabled()))
        b, n = points_a.shape[:2]
        p = self.weight.sigmoid()
        eye = torch.eye(n, device=points_a.device)[None].expand(b, n, n)
        assignment = eye * p + (1 - eye) * (1 - p) / n
        valid = torch.ones(b, dtype=torch.bool, device=points_a.device)
        logits = self.weight.expand(b)
        return SimpleNamespace(assignment=assignment, unmatched_a=(1-p).expand(b, n),
            unmatched_b=(1-p).expand(b, n), training_valid=valid, fused_logit=logits,
            coarse_logit=logits, local_logit=logits, coarse=SimpleNamespace(valid_problem=valid),
            translation_hat_rc=self.weight.expand(b, 2),
            transport=SimpleNamespace(diagnostics=SimpleNamespace(
                row_residual_max=torch.zeros(b), col_residual_max=torch.zeros(b))))


def test_teacher_stopgrad_mode_restoration_and_error_restoration():
    model = TinyModel().train()
    model.dropout.eval()  # Mixed module flags must be preserved exactly.
    inputs = (torch.zeros(1, 1, 4, 4),) * 2 + (torch.zeros(1, 4, 2),) * 2 + (torch.ones(1, 4, dtype=torch.bool),) * 2
    teacher = clean_teacher_forward(model, inputs)
    assert model.calls == [(False, False)]
    assert model.training and not model.dropout.training
    for value in vars(teacher).values():
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad and value.grad_fn is None
    assert model.weight.grad is None
    with patch.object(model, "forward", side_effect=RuntimeError("fixture")):
        with pytest.raises(RuntimeError, match="fixture"):
            clean_teacher_forward(model, inputs)
    assert model.training and not model.dropout.training


def test_assignment_source_permutation_ignores_unrepresented_mass_and_detaches_teacher():
    teacher_matrix = torch.tensor([[.6, .1, .05], [.1, .5, .1], [.05, .1, .6]], requires_grad=True)
    # source A [2,0], source B [1,2]: teacher's omitted columns enter OTHER.
    aa, ab = torch.tensor([2, 0]), torch.tensor([1, 2])
    student_matrix = teacher_matrix.detach()[aa][:, ab].clone().requires_grad_()
    ua = 1 - student_matrix.detach().sum(1)
    ub = 1 - student_matrix.detach().sum(0)
    student = SimpleNamespace(assignment=student_matrix, unmatched_a=ua, unmatched_b=ub)
    teacher = SimpleNamespace(assignment=teacher_matrix, unmatched_a=1-teacher_matrix.sum(1),
                              unmatched_b=1-teacher_matrix.sum(0))
    zero, count = aligned_assignment_kl(student, teacher, aa, ab)
    assert float(zero) < 1e-6 and count == 4
    changed = SimpleNamespace(assignment=(student_matrix * .5), unmatched_a=ua, unmatched_b=ub)
    loss, _ = aligned_assignment_kl(changed, teacher, aa, ab)
    assert float(loss) > 0
    loss.backward()
    assert student_matrix.grad.abs().sum() > 0 and teacher_matrix.grad is None
    ignored, count = aligned_assignment_kl(student, teacher, torch.full_like(aa, -1), ab)
    assert ignored == 0 and count == 0


def test_true_eroded_edge_gap_is_tolerated_but_wrong_candidates_have_gradient():
    clean_a = torch.tensor([[0., 0.], [0., 30.]])
    gt = torch.tensor([10., 5.])
    clean_b = clean_a + gt
    # Artificial inward displacements leave a 6px boundary gap at original GT.
    new_a, new_b = clean_a + torch.tensor([3., 0.]), clean_b + torch.tensor([-3., 0.])
    ids = torch.arange(2)
    config = E2LossConfig(gap_tolerance_px=0.)
    correct = torch.eye(2, requires_grad=True)
    loss, detail = source_corrected_gap_loss(correct, ids, ids, ids, ids,
        new_a, new_b, clean_a, clean_b, gt, config=config)
    assert loss == 0 and detail["raw_gap_sum_px"] == 12 and detail["corrected_gap_sum_px"] == 0
    wrong = torch.tensor([[.5, .5], [.5, .5]], requires_grad=True)
    bad, _ = source_corrected_gap_loss(wrong, ids, ids, ids, ids,
        new_a, new_b, clean_a, clean_b, gt, config=config)
    assert float(bad) > 0
    bad.backward()
    assert wrong.grad[0, 1] > 0 and wrong.grad[0, 0] < 0
    # Loss depends on source correspondence, not the magnitude of known gap.
    bigger, _ = source_corrected_gap_loss(wrong, ids, ids, ids, ids,
        new_a + 10, new_b - 10, clean_a, clean_b, gt, config=config)
    torch.testing.assert_close(bigger, bad, atol=0, rtol=0)


def _loss_sidecars(args, changed):
    b, n = args[1].shape
    points = torch.tensor([[0., 0.], [0., 30.], [30., 30.], [30., 0.]])[None].repeat(b, 1, 1)
    points_b = points + args[-2][:, None]
    return dict(pose_supervision_enabled=args[-1] & ~changed, changed_pair=changed,
        ancestor_a=torch.arange(n)[None].repeat(b, 1), ancestor_b=torch.arange(n)[None].repeat(b, 1),
        student_points_a=points + 2, student_points_b=points_b - 2,
        clean_points_a=points, clean_points_b=points_b)


def test_clean_loss_exact_e1_parity_and_changed_negative_has_no_gap_supervision():
    output, args = fixture()
    clean = compute_e2_loss(output, None, *args, **_loss_sidecars(args, torch.zeros(3, dtype=torch.bool)))
    e1 = compute_weathering_loss(output, *args, pose_supervision_enabled=args[-1])
    assert torch.equal(clean.total, e1.total)
    assert clean.pair_consistency == clean.assignment_consistency == clean.gap_transport == 0
    # Pair and assignment consistency may supervise negatives; geometry may not.
    teacher, _ = fixture()
    teacher.fused_logit = torch.ones(3, requires_grad=True)
    changed = compute_e2_loss(output, teacher, *args,
        **_loss_sidecars(args, torch.tensor([False, False, True])))
    assert changed.pair_consistency > 0 and changed.gap_transport == 0
    assert changed.diagnostics["gap_pair_count"] == 0
    changed.total.backward()
    assert teacher.fused_logit.grad is None and teacher.assignment.grad is None


def test_e2_changed_positive_keeps_gt_and_raw_pose_mask_but_adds_assignment_gradient():
    output, args = fixture()
    teacher, _ = fixture()
    before = args[-2].clone()
    result = compute_e2_loss(output, teacher, *args,
        **_loss_sidecars(args, torch.tensor([True, True, False])))
    assert result.diagnostics["gap_pair_count"] == 2 and result.gap_transport > 0
    result.total.backward()
    assert output.translation_hat_rc.grad.abs().sum() == 0
    assert output.assignment.grad[:2].abs().sum() > 0
    assert torch.equal(before, args[-2]) and teacher.assignment.grad is None


def test_train_loop_microbatch_exposure_update_parity_and_cpu_loader():
    source = sample()
    # Small token fixture, real mask shape retained; no dense weathering needed.
    points = np.array([[10, 10], [10, 40], [40, 40], [40, 10]], np.float32)
    source = replace(source, points_rc_a=points, points_rc_b=points + [9., -7.],
        contour_valid_a=np.ones(4, bool), contour_valid_b=np.ones(4, bool),
        target_a=np.array([0, 1, -1, -1]), target_b=np.array([0, 1, -1, -1]))
    report = dict(changed_pair=True, changed_a=True, changed_b=False,
        pose_supervision_enabled=False, inherited_match_count=2, effective_supervised_match_count=2,
        ignored_token_count=0, fallback_reason=None, side_a=dict(tier="mild", applied=True,
        removed_fraction=.01), side_b=dict(tier="clean", applied=False))
    student = replace(source, points_rc_a=points + 2)
    item = E2Sample(student, report, source, np.arange(4), np.arange(4))
    loader = make_e2_loader([item] * 3, range(3), batch_size=1, num_workers=0, seed=42, contour_cap=4)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    args = SimpleNamespace(effective_batch_size=2, batch_size=1, log_every=999, output=None)
    result = train_e2_epoch(model, loader, optimizer, RachelN512LossConfig(), torch.device("cpu"), args, 0, "e2")
    assert result["samples"] == 3 and result["optimizer_updates"] == 2
    assert result["weathering"]["counts"]["pair_exposures"] == 3
    assert result["e2"]["counts"]["gap_pair_count"] == 3
    assert result["e2"]["losses"]["gap_transport"] > 0
    assert model.calls == [(False, False), (True, True)] * 3
    assert result["peak_allocated_gpu_bytes"] == 0


def test_e2_config_rejects_bad_scales_and_weights():
    with pytest.raises(ValueError):
        E2LossConfig(gap_scale_px=0)
    with pytest.raises(ValueError):
        E2LossConfig(assignment_consistency_weight=-1)
