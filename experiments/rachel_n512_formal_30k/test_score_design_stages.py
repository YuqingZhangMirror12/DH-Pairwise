"""Independent CPU helper tests; these are tiny synthetic steps, not training runs."""
from contextlib import contextmanager
from dataclasses import replace
import inspect
import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.rachel_n512_formal_30k import score_design_stages as stages
from staging.pairwise_v0_2.models.rachel_candidate_score import CandidateScoreConfig, build_score_model
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from staging.pairwise_v0_2.training.rachel_weathering_training import compute_weathering_loss


@contextmanager
def raises(exception, message=".*"):
    with unittest.TestCase().assertRaisesRegex(exception, message):
        yield


def model(architecture="candidate_dual"):
    torch.manual_seed(28)
    config = RachelN512Config(canvas_size=80, coarse_size=32, contour_cap=16, patch_size=8,
        feature_dim=16, num_heads=4, landmark_count=4, context_layers=1, evidence_dim=8,
        sinkhorn_iterations=40, activation_checkpointing=False)
    return build_score_model(config, architecture,
        CandidateScoreConfig(hidden_dim=8, max_additional_modes=2))


def inputs():
    a, b = torch.zeros(1, 1, 80, 80), torch.zeros(1, 1, 80, 80)
    a[:, :, 15:57, 12:43] = 1
    b[:, :, 10:61, 28:63] = 1
    pa = torch.tensor([[[15., 12.], [15., 27.], [15., 42.], [35., 42.],
                        [56., 42.], [56., 27.], [56., 12.], [35., 12.]]])
    pb = torch.tensor([[[10., 28.], [10., 45.], [10., 62.], [35., 62.],
                        [60., 62.], [60., 45.], [60., 28.], [35., 28.]]])
    return a, b, pa, pb, torch.ones(1, 8, dtype=torch.bool), torch.ones(1, 8, dtype=torch.bool)


def targets():
    return (torch.ones(1), torch.arange(8)[None], torch.arange(8)[None],
            torch.zeros(1, 2), torch.ones(1, dtype=torch.bool))


def loss(spec, output, *, changed=False):
    return stages.compute_phase_loss(spec, output, *targets(),
        pose_supervision_enabled=torch.tensor([not changed]))


def snapshots(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def same(snapshot, module):
    return all(torch.equal(v, module.state_dict()[k]) for k, v in snapshot.items())


def receipt_after_tiny_synthetic_step(net, initial):
    # Counters are fixture values to unit-test the binding, not claimed epochs.
    return stages.matcher_pretraining_receipt(net, initial_matcher_sha256=initial,
        completed_epochs=12, pair_exposures=288000, optimizer_updates=18000,
        train_manifest_sha256="a" * 64, run_identity_sha256="b" * 64)


def configure_c(net, receipt):
    spec = stages.phase(13, architecture=net.architecture)
    stages.configure_phase(net, spec, receipt=receipt,
        expected_train_manifest_sha256="a" * 64, expected_run_identity_sha256="b" * 64)
    return spec


def test_registered_budget_and_explicit_loss_changes():
    assert [stages.phase(e).name for e in range(1, 21)] == ["M"] * 12 + ["C"] * 8
    assert all(stages.phase(e, schedule="joint").name == "J" for e in range(1, 21))
    staged, joint = stages.stage_protocol(), stages.stage_protocol(schedule="joint")
    assert staged["total_pair_exposures"] == joint["total_pair_exposures"] == 480000
    assert staged["total_optimizer_updates"] == joint["total_optimizer_updates"] == 30000
    assert (staged["matcher_update_exposures"], staged["classifier_update_exposures"]) == (288000, 192000)
    assert staged["evaluation"] == joint["evaluation"]
    evaluation = staged["evaluation"]
    assert evaluation["schema_version"] == "rachel-score-design-s3-comparison/1"
    assert evaluation["primary"]["epoch"] == 20 and evaluation["primary"]["selection"] == "fixed_epoch"
    assert evaluation["auxiliary"]["eligible_epochs"] == list(range(13, 21))
    assert set(evaluation["auxiliary"]["selection_rules"]) == {"max_f1", "recall95"}
    assert not evaluation["live_s0_s2_budget_freeze_compatible"]
    assert not evaluation["live_winner_from_epochs5_through20_allowed"]
    assert evaluation["requires_dedicated_evaluator_schema"]
    config = RachelN512LossConfig()
    m, mp = stages.phase_loss_profile(stages.phase(1, architecture="candidate_dual"), config)
    c, cp = stages.phase_loss_profile(stages.phase(13, architecture="candidate_dual"), config)
    j, jp = stages.phase_loss_profile(stages.phase(1, schedule="joint", architecture="candidate_dual"), config)
    assert m.fused_pair_weight == m.local_pair_weight == mp["candidate_correctness_weight"] == 0
    assert m.coarse_pair_weight == config.coarse_pair_weight
    assert m.assignment_weight == config.assignment_weight
    assert c.assignment_weight == c.translation_weight == c.sinkhorn_residual_weight == 0
    assert c.fused_pair_weight == 1 and c.local_pair_weight == .5 and c.coarse_pair_weight == .25
    assert cp["coarse_pair_bce_is_constant"] and cp["candidate_correctness_weight"] == .5
    assert j == config and jp["candidate_correctness_weight"] == .5
    with raises(ValueError): stages.phase(1, budget=30)
    with raises(ValueError): stages.phase(21)


def test_m_bypasses_candidate_and_updates_only_matcher():
    net = model()
    spec = stages.phase(1, architecture=net.architecture)
    stages.configure_phase(net, spec)
    optimizer = torch.optim.AdamW(stages.optimizer_parameter_groups(net), lr=1e-3, weight_decay=1e-4)
    frozen = {"score": snapshots(net.score_head), "fusion": snapshots(net.base_model.fusion),
              "old": snapshots(net.base_model.local_head)}
    before = stages.matcher_state_digest(net)
    with patch.object(net.score_head, "forward", side_effect=AssertionError("M ran candidate head")):
        output = stages.forward_for_phase(net, spec, *inputs())
    result = loss(spec, output)
    result.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.base_model.primal.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.base_model.coarse.parameters())
    assert all(p.grad is None for p in net.score_head.parameters())
    assert all(p.grad is None for p in net.base_model.fusion.parameters())
    assert all(p.grad is None for p in net.base_model.local_head.parameters())
    optimizer.step()
    assert stages.matcher_state_digest(net) != before
    assert same(frozen["score"], net.score_head) and same(frozen["fusion"], net.base_model.fusion)
    assert same(frozen["old"], net.base_model.local_head)
    assert result.counts["r_valid"] == 0 and result.counts["assignment_supervised_matches"] == 8
    assert not net.score_head.training and not net.base_model.fusion.training


def test_c_freezes_parameters_bn_dropout_and_retains_optimizer_groups():
    net = model()
    # Future BN/dropout components must remain frozen, even though the current
    # matcher primarily uses other normalization. This probe belongs to matcher.
    net.base_model.freeze_probe = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(.9))
    m = stages.phase(1, architecture=net.architecture)
    stages.configure_phase(net, m)
    initial = stages.matcher_state_digest(net)
    optimizer = torch.optim.AdamW(stages.optimizer_parameter_groups(net), lr=1e-3, weight_decay=.01)
    groups_before = [[id(p) for p in group["params"]] for group in optimizer.param_groups]
    loss(m, stages.forward_for_phase(net, m, *inputs())).total.backward()
    optimizer.step()
    receipt = receipt_after_tiny_synthetic_step(net, initial)
    c = configure_c(net, receipt)
    # configure_phase cleared every stale gradient at the transition.
    assert all(p.grad is None for p in net.parameters())
    frozen_digest = stages.matcher_state_digest(net)
    score_before, fusion_before = snapshots(net.score_head), snapshots(net.base_model.fusion)
    net.train()  # Simulate an enclosing trainer's otherwise dangerous call.
    out = stages.forward_for_phase(net, c, *inputs())
    assert not net.base_model.training and net.base_model.fusion.training and net.score_head.training
    probe = torch.randn(5, 4)
    before_buffers = snapshots(net.base_model.freeze_probe)
    torch.testing.assert_close(net.base_model.freeze_probe(probe), net.base_model.freeze_probe(probe), rtol=0, atol=0)
    assert same(before_buffers, net.base_model.freeze_probe)
    result = loss(c, out, changed=True)
    result.total.backward()
    assert all(p.grad is None for n, p in net.named_parameters() if stages._family(n) == "matcher")
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.score_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.base_model.fusion.parameters())
    optimizer.step()
    assert stages.matcher_state_digest(net) == frozen_digest
    assert not same(score_before, net.score_head) and not same(fusion_before, net.base_model.fusion)
    assert groups_before == [[id(p) for p in group["params"]] for group in optimizer.param_groups]
    assert result.counts["shift_supervised"] == result.counts["assignment_supervised_matches"] == 0
    # Reuse the receipt after a classifier update: matcher state still matches.
    configure_c(net, receipt)


def test_c_rejects_random_matcher_wrong_budget_and_wrong_state_receipt():
    net = model()
    initial = stages.matcher_state_digest(net)
    with raises(ValueError, "unchanged"):
        receipt_after_tiny_synthetic_step(net, initial)
    with raises(ValueError):
        stages.configure_phase(net, stages.phase(13, architecture=net.architecture))
    with torch.no_grad(): net.base_model.primal.weight.add_(.01)
    receipt = receipt_after_tiny_synthetic_step(net, initial)
    with raises(ValueError, "budget"):
        configure_c(net, dict(receipt, pair_exposures=0))
    with torch.no_grad(): net.base_model.primal.weight.add_(.01)
    with raises(ValueError, "state"):
        configure_c(net, receipt)
    with raises(ValueError, "boundary"):
        stages.configure_phase(net, stages.phase(1, architecture=net.architecture), at_optimizer_boundary=False)


def test_joint_loss_is_original_plus_only_dual_r_and_pair_label_is_not_r_label():
    for architecture in ("candidate_pair", "candidate_dual"):
        net = model(architecture)
        j = stages.phase(1, schedule="joint", architecture=architecture)
        stages.configure_phase(net, j)
        output = stages.forward_for_phase(net, j, *inputs())
        # Force a valid but wrong candidate for a genuinely positive pair.
        r = torch.zeros(1, 2, requires_grad=True)
        details = dict(output.score_details, candidate_logits=r,
            translations_rc=torch.full((1, 2, 2), 500.), valid=torch.ones(1, 2, dtype=torch.bool))
        output = replace(output, score_details=details)
        labels, ta, tb, gt, valid = targets()
        labels_before = labels.clone()
        result = stages.compute_phase_loss(j, output, labels, ta, tb, gt, valid,
            pose_supervision_enabled=torch.zeros(1, dtype=torch.bool))
        original = compute_weathering_loss(output, labels, ta, tb, gt, valid,
            pose_supervision_enabled=torch.zeros(1, dtype=torch.bool))
        expected = original.total + .5 * result.candidate_correctness_bce if architecture == "candidate_dual" else original.total
        torch.testing.assert_close(result.total, expected, rtol=0, atol=0)
        assert torch.equal(labels, labels_before) and result.counts["pair_positive"] == 1
        assert result.counts["r_positive"] == 0 and result.counts["r_negative"] == 2
        assert result.counts["shift_supervised"] == 0  # Changed GT still supervises both R0 candidates.
        output.local_logit.retain_grad()
        result.base_loss.local_pair_bce.backward(retain_graph=True)
        assert output.local_logit.grad.item() < 0  # Positive pair pushes local probability UP.
        result.candidate_correctness_bce.backward()
        assert (r.grad > 0).all()  # Wrong poses push R probability DOWN, separately.
    assert not any("target" in p or "label" in p for p in inspect.signature(stages.forward_for_phase).parameters)
    with raises(ValueError, "TRAIN-only"):
        stages.compute_phase_loss(j, output, *targets(), pose_supervision_enabled=torch.ones(1, dtype=torch.bool), split="val")


if __name__ == "__main__":
    torch.set_num_threads(1)
    for test in (test_registered_budget_and_explicit_loss_changes,
                 test_m_bypasses_candidate_and_updates_only_matcher,
                 test_c_freezes_parameters_bn_dropout_and_retains_optimizer_groups,
                 test_c_rejects_random_matcher_wrong_budget_and_wrong_state_receipt,
                 test_joint_loss_is_original_plus_only_dual_r_and_pair_label_is_not_r_label):
        test()
        print(test.__name__ + ": PASS")
