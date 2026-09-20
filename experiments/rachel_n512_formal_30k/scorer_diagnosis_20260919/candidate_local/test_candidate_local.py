"""CPU synthetic canaries only; no real images/checkpoints or remote connection."""
from copy import deepcopy
from dataclasses import asdict
import inspect
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local import model as local
from experiments.rachel_n512_formal_30k import train_score_decoupled as trainer
from experiments.rachel_n512_formal_30k.test_train_score_decoupled import identity, tiny_loader
from experiments.rachel_n512_formal_30k.test_score_design_stages import inputs
from experiments.rachel_n512_formal_30k.test_train_score_staged import tensor_tree_equal
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import continue_classifier as continuation
from staging.pairwise_v0_2.models.rachel_decoupled_score import CrossAttentionPairHead, build_decoupled_score_model
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config


def source_model():
    config = RachelN512Config(canvas_size=80, coarse_size=32, contour_cap=16, patch_size=8,
        feature_dim=16, num_heads=4, landmark_count=4, context_layers=1, evidence_dim=8,
        sinkhorn_iterations=40, activation_checkpointing=False)
    return build_decoupled_score_model(config, "cross_attention", phase="classifier",
                                      model_options={"cross_attention_depth": 2})


def source_checkpoint():
    torch.manual_seed(42)
    model = source_model()
    origin = identity(model)
    origin["model_options"] = {"cross_attention_depth": 2}
    origin["matcher_checkpoint_sha256"] = "a" * 64
    optimizer = trainer.create_optimizer(model)
    for name, p in model.score_head.named_parameters():
        if name != "no_evidence_logit":
            p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        state["step"].fill_(12000)
    for group in optimizer.param_groups:
        group["lr"] = 2e-5
    receipt = trainer.matcher_receipt(model, origin)
    return trainer.checkpoint_payload(model, optimizer, identity=origin,
        loss_config=trainer.RachelN512LossConfig(), completed=80, receipt=receipt,
        winners={}, role="synthetic fixture only")


def geometry():
    a = torch.tensor([[[0., 0.], [30., 0.], [60., 0.], [100., 100.]]])
    b = torch.tensor([[[10., 5.], [40., 5.], [70., 5.], [260., 210.], [300., 250.]]])
    q = torch.zeros(1, 4, 5)
    q[0, 0, 0], q[0, 1, 1], q[0, 2, 2], q[0, 3, 3] = .9, .8, .7, .1
    va, vb = torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 5, dtype=torch.bool)
    return q, a, b, va, vb


def head():
    torch.manual_seed(51)
    return CrossAttentionPairHead(16, 4, 2).eval()


class CandidateLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_decoder_configuration_and_final_peak_membership(self):
        from experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint import TOP2_CONFIG
        self.assertEqual(asdict(local.DECODER_CONFIG), asdict(TOP2_CONFIG))
        q, a, b, va, vb = geometry()
        got = local.select_predicted_inliers(q, a, b, va, vb)
        ref = local.estimate_translation_layout(a[0], b[0], q[0], va[0], vb[0], config=TOP2_CONFIG)
        self.assertTrue(np.array_equal(got.candidate_indices[0, :ref.candidate_count], ref.candidate_indices))
        self.assertTrue(np.array_equal(got.candidate_inliers[0, :ref.candidate_count], ref.inlier_mask))
        self.assertTrue(torch.equal(got.mask_a, torch.tensor([[True, True, True, False]])))
        self.assertTrue(torch.equal(got.mask_b, torch.tensor([[True, True, True, False, False]])))
        self.assertTrue(torch.equal(got.translation_a_to_b_rc, torch.tensor([[10., 5.]])))

    def test_duplicate_endpoints_not_duplicate_votes(self):
        a = torch.tensor([[[0., 0.], [30., 0.]]])
        b = torch.tensor([[[10., 5.], [11., 5.], [40., 5.]]])
        q = torch.tensor([[[.9, .8, 0.], [0., 0., .7]]])
        selected = local.select_predicted_inliers(q, a, b, torch.ones(1, 2, dtype=torch.bool),
                                                  torch.ones(1, 3, dtype=torch.bool))
        self.assertEqual(int(selected.candidate_inliers.sum()), 3)
        self.assertEqual(int(selected.mask_a.sum()), 2)
        self.assertEqual(int(selected.mask_b.sum()), 3)

    def test_no_gt_or_label_argument_or_override(self):
        self.assertEqual(set(inspect.signature(local.select_predicted_inliers).parameters),
            {"assignment", "points_a_rc", "points_b_rc", "valid_a", "valid_b"})
        for field in ("gt_translation", "label", "real_label", "candidate_override"):
            with self.assertRaises(TypeError):
                local.select_predicted_inliers(*geometry(), **{field: torch.zeros(1, 2)})
        q, a, b, va, vb = geometry()
        q.requires_grad_()
        a.requires_grad_()
        got = local.select_predicted_inliers(q, a, b, va, vb)
        self.assertFalse(got.translation_a_to_b_rc.requires_grad)
        self.assertFalse(got.mask_a.requires_grad)

    def test_same_init_and_initial_logits_exactly_equal_source(self):
        source = head()
        state = deepcopy(source.state_dict())
        before_rng = torch.get_rng_state().clone()
        c1 = local.CandidateResidualHead(source, "predicted_inliers").eval()
        c2 = local.CandidateResidualHead(source, "all_valid_control").eval()
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(tensor_tree_equal(c1.state_dict(), c2.state_dict()))
        self.assertTrue(tensor_tree_equal(source.state_dict(), state))
        q, a, b, va, vb = geometry()
        selected = local.select_predicted_inliers(q, a, b, va, vb)
        fa, fb = torch.randn(1, 4, 16), torch.randn(1, 5, 16)
        expected = source(fa, fb, va, vb)
        for branch in (c1, c2):
            result = branch(fa, fb, va, vb, selected)
            self.assertTrue(torch.equal(result.fused_logit, expected))
            self.assertEqual(float(result.local_delta), 0.)

    def test_c1_c2_only_differ_by_residual_mask(self):
        q, a, b, va, vb = geometry()
        selected = local.select_predicted_inliers(q, a, b, va, vb)
        c1 = local.CandidateResidualHead(head(), "predicted_inliers").eval()
        c2 = local.CandidateResidualHead(head(), "all_valid_control").eval()
        with torch.no_grad():
            c1.local_head.classifier[-1].weight.fill_(.1)
            c2.local_head.load_state_dict(c1.local_head.state_dict())
        fa, fb = torch.randn(1, 4, 16), torch.randn(1, 5, 16)
        o1, o2 = c1(fa, fb, va, vb, selected), c2(fa, fb, va, vb, selected)
        self.assertTrue(torch.equal(o1.global_logit, o2.global_logit))
        self.assertTrue(torch.equal(o1.local_eligible, o2.local_eligible))
        expected = c1.local_head(fa, fb, selected.mask_a, selected.mask_b)
        self.assertTrue(torch.equal(expected, o1.local_delta))
        self.assertTrue(torch.equal(c2.local_head(fa, fb, va, vb), o2.local_delta))
        self.assertFalse(torch.allclose(o1.local_delta, o2.local_delta))

    def test_no_candidate_falls_back_even_with_large_positive_residual_bias(self):
        q, a, b, va, vb = geometry()
        selected = local.select_predicted_inliers(torch.zeros_like(q), a, b, va, vb)
        fa, fb = torch.randn(1, 4, 16), torch.randn(1, 5, 16)
        for mode in local.MODES:
            network = local.CandidateResidualHead(head(), mode)
            with torch.no_grad():
                network.local_head.classifier[-1].bias.fill_(100.)
                network.local_head.no_evidence_logit.fill_(100.)
            scores = network(fa, fb, va, vb, selected)
            self.assertFalse(bool(scores.local_eligible))
            self.assertEqual(float(scores.local_delta), 0.)
            self.assertTrue(torch.equal(scores.fused_logit, scores.global_logit))

    def test_valid_high_inlier_layout_does_not_imply_positive(self):
        q, a, b, va, vb = geometry()
        selected = local.select_predicted_inliers(q, a, b, va, vb)
        self.assertTrue(bool(selected.layout_valid))
        source = head()
        with torch.no_grad():
            source.classifier[-1].weight.zero_()
            source.classifier[-1].bias.fill_(-10.)
        network = local.CandidateResidualHead(source)
        scores = network(torch.randn(1, 4, 16), torch.randn(1, 5, 16), va, vb, selected)
        self.assertLess(float(scores.fused_logit.sigmoid()), .001)

    def test_a_b_exchange_selection_and_scores(self):
        q, a, b, va, vb = geometry()
        ab = local.select_predicted_inliers(q, a, b, va, vb)
        ba = local.select_predicted_inliers(q.transpose(1, 2), b, a, vb, va)
        self.assertTrue(torch.equal(ab.mask_a, ba.mask_b))
        self.assertTrue(torch.equal(ab.mask_b, ba.mask_a))
        self.assertTrue(torch.equal(ab.translation_a_to_b_rc, -ba.translation_a_to_b_rc))
        fa, fb = torch.randn(1, 4, 16), torch.randn(1, 5, 16)
        for mode in local.MODES:
            network = local.CandidateResidualHead(head(), mode).eval()
            with torch.no_grad():
                network.local_head.classifier[-1].weight.fill_(.07)
            first = network(fa, fb, va, vb, ab).fused_logit
            second = network(fb, fa, vb, va, ba).fused_logit
            self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))

    def test_batch_and_arbitrary_padding_do_not_enter_attention(self):
        q, a, b, va, vb = geometry()
        fa, fb = torch.randn(1, 4, 16), torch.randn(1, 5, 16)
        ia, ib = torch.tensor([0, 2, 3, 6]), torch.tensor([1, 2, 4, 5, 7])
        pad_q, pad_a, pad_b = torch.full((2, 7, 8), float("nan")), torch.full((2, 7, 2), float("nan")), torch.full((2, 8, 2), float("nan"))
        pad_fa, pad_fb = torch.full((2, 7, 16), float("nan")), torch.full((2, 8, 16), float("nan"))
        pad_va, pad_vb = torch.zeros((2, 7), dtype=torch.bool), torch.zeros((2, 8), dtype=torch.bool)
        for i in range(2):
            pad_q[i][ia[:, None], ib[None, :]] = q[0]
            pad_a[i, ia], pad_b[i, ib] = a[0], b[0]
            pad_fa[i, ia], pad_fb[i, ib] = fa[0], fb[0]
            pad_va[i, ia], pad_vb[i, ib] = True, True
        original = local.select_predicted_inliers(q, a, b, va, vb)
        padded = local.select_predicted_inliers(pad_q, pad_a, pad_b, pad_va, pad_vb)
        network = local.CandidateResidualHead(head()).eval()
        with torch.no_grad():
            network.local_head.classifier[-1].weight.fill_(.03)
        expected = network(fa, fb, va, vb, original).fused_logit
        got = network(pad_fa, pad_fb, pad_va, pad_vb, padded).fused_logit
        self.assertTrue(torch.allclose(got, expected.expand(2), atol=1e-6, rtol=1e-6))
        self.assertFalse(bool(torch.any(padded.mask_a & ~pad_va)))
        self.assertFalse(bool(torch.any(padded.mask_b & ~pad_vb)))

    def test_six_input_wrapper_exact_baseline_and_frozen_matcher_gradients(self):
        torch.manual_seed(31)
        source = source_model().eval()
        before = trainer.state_digest(source)
        baseline = source(*inputs())
        network = local.FrozenCandidateLocalModel(source).train()
        initial = network(*inputs())
        self.assertTrue(torch.equal(initial.fused_logit, baseline.fused_logit))
        self.assertTrue(torch.equal(initial.assignment, baseline.assignment))
        loss = F.binary_cross_entropy_with_logits(initial.fused_logit, torch.zeros_like(initial.fused_logit))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in network.base_model.parameters()))
        self.assertTrue(any(p.grad is not None for p in network.score_head.global_head.parameters()))
        if initial.score_details["local_eligible"].any():
            self.assertGreater(float(network.score_head.local_head.classifier[-1].weight.grad.abs().sum()), 0.)
        self.assertFalse(network.base_model.training)
        self.assertEqual(trainer.state_digest(source), before)
        with self.assertRaises(ValueError):
            network.set_phase("matcher")

    def test_zero_projection_learns_then_all_local_layers_receive_gradients(self):
        q, a, b, va, vb = geometry()
        selected = local.select_predicted_inliers(q, a, b, va, vb)
        network = local.CandidateResidualHead(head()).train()
        optimizer = torch.optim.AdamW(network.parameters(), lr=2e-5)
        fa, fb = torch.randn(1, 4, 16), torch.randn(1, 5, 16)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = network(fa, fb, va, vb, selected)
            F.binary_cross_entropy_with_logits(output.fused_logit, torch.zeros(1)).backward()
            if step == 0:
                self.assertEqual(float(network.local_head.cross_attention.in_proj_weight.grad.abs().sum()), 0.)
            else:
                self.assertGreater(float(network.local_head.cross_attention.in_proj_weight.grad.abs().sum()), 0.)
            optimizer.step()

    def test_optimizer_preserves_global_moments_adds_only_one_cold_group(self):
        checkpoint = source_checkpoint()
        saved = deepcopy(checkpoint)
        rng = torch.get_rng_state().clone()
        network = local.build_from_s6_epoch20(checkpoint)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        optimizer, receipt = local.restore_source_optimizer(checkpoint, network)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        result = optimizer.state_dict()
        self.assertTrue(tensor_tree_equal(result["param_groups"][:2], checkpoint["optimizer_state_dict"]["param_groups"]))
        self.assertTrue(tensor_tree_equal(result["state"], checkpoint["optimizer_state_dict"]["state"]))
        self.assertEqual(result["param_groups"][2]["phase_family"], "candidate_local")
        self.assertTrue(all(p not in optimizer.state for p in network.score_head.local_head.parameters()))
        self.assertFalse(receipt["optimizer_reset"])
        self.assertTrue(tensor_tree_equal(saved["model_state_dict"], checkpoint["model_state_dict"]))

    def test_original_samplewise_trainer_reuses_forward_and_loss_without_matcher_updates(self):
        import tempfile
        from types import SimpleNamespace
        checkpoint = source_checkpoint()
        network = local.build_from_s6_epoch20(checkpoint)
        optimizer, _ = local.restore_source_optimizer(checkpoint, network)
        before = trainer.state_digest(network.base_model)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(microbatch=1, physical_microbatch=16, effective_batch=16,
                                   runtime_effective_batch=None, log_every=1000, output=directory)
            report = continuation.train_segment(network, tiny_loader(16, 16), optimizer,
                trainer.RachelN512LossConfig(), torch.device("cpu"), args, 21)
        self.assertEqual((report["samples"], report["optimizer_updates"]), (16, 1))
        self.assertEqual(report["loss_components"]["total"], report["loss_components"]["fused_pair_bce"])
        self.assertEqual(trainer.state_digest(network.base_model), before)
        self.assertTrue(all(int(optimizer.state[p]["step"]) == 12001 for p in network.score_head.global_head.parameters() if p in optimizer.state))
        self.assertTrue(all(int(optimizer.state[p]["step"]) == 1 for p in network.score_head.local_head.parameters() if p in optimizer.state))


if __name__ == "__main__":
    unittest.main()
