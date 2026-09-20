"""Focused synthetic CPU tests; no checkpoint, real data, training or network."""
from dataclasses import replace
import inspect
import unittest

import torch
from torch.nn import functional as F

from . import model as m


def fixture():
    pa = torch.tensor([[[0., 0.], [30., 0.], [60., 0.], [100., 100.], [0., 0.]]])
    pb = torch.tensor([[[10., 5.], [40., 5.], [70., 5.], [260., 210.], [300., 250.], [0., 0.]]])
    va = torch.tensor([[True, True, True, True, False]])
    vb = torch.tensor([[True, True, True, True, True, False]])
    q = torch.zeros(1, 5, 6)
    q[0, 0, 0], q[0, 1, 1], q[0, 2, 2], q[0, 3, 3] = .9, .8, .7, .1
    s = m.select_predicted_inliers(q, pa, pb, va, vb)
    gen = torch.Generator().manual_seed(49)
    a, b = torch.randn(1, 5, 8, generator=gen), torch.randn(1, 6, 8, generator=gen)
    weights = q[torch.arange(1)[:, None], s.candidate_indices[..., 0].clamp_min(0), s.candidate_indices[..., 1].clamp_min(0)]
    weights = torch.where(s.candidate_valid, weights, torch.zeros_like(weights))
    return a, b, va, vb, s, dict(candidate_weights=weights, points_a_rc=pa, points_b_rc=pb), q


def network(arm):
    return m.make_fresh_scorer(arm, seed=42, feature_dim=8, num_heads=2).eval()


class MatchedOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_initialization_and_capacity_control(self):
        before = torch.get_rng_state().clone()
        arms = [m.make_fresh_scorer(arm) for arm in m.ARMS]
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for arm in arms[1:]:
            for key, value in arms[0].head.state_dict().items():
                self.assertTrue(torch.equal(value, arm.head.state_dict()[key]), key)
        self.assertEqual(arms[0].metadata()["parameters"], arms[1].metadata()["parameters"])
        self.assertEqual(arms[0].metadata()["parameters"]["total"], 178979)
        self.assertEqual(arms[2].metadata()["parameters"]["additional"], 18720)
        self.assertEqual(arms[2].metadata()["parameters"]["total"], 197699)

    def test_uses_exact_production_selector_no_label_or_gt_api(self):
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local.model import select_predicted_inliers
        self.assertIs(m.select_predicted_inliers, select_predicted_inliers)
        a, b, va, vb, s, meta, q = fixture()
        for arm in m.ARMS:
            net = network(arm)
            supplied = net(a, b, va, vb, s, **meta)
            derived = net(a, b, va, vb, assignment=q, points_a_rc=meta["points_a_rc"], points_b_rc=meta["points_b_rc"])
            torch.testing.assert_close(supplied.logit, derived.logit)
            for forbidden in ("label", "gt_translation", "layout_correct"):
                self.assertNotIn(forbidden, inspect.signature(net.forward).parameters)
                with self.assertRaises(TypeError):
                    net(a, b, va, vb, s, **{forbidden: True})

    def test_endpoint_mask_only_and_all_tokens_reference(self):
        a, b, va, vb, s, meta, _ = fixture()
        all_head, local = network("all_tokens"), network("matched_tokens")
        torch.testing.assert_close(all_head(a, b, va, vb, s).logit, all_head.head(a, b, va, vb))
        torch.testing.assert_close(local(a, b, va, vb, s).logit, local.head(a, b, s.mask_a, s.mask_b))
        changed_a, changed_b = a.clone(), b.clone()
        changed_a[~s.mask_a] = 900.
        changed_b[~s.mask_b] = -900.
        for arm in ("matched_tokens", "matched_edges"):
            net = network(arm)
            torch.testing.assert_close(net(a, b, va, vb, s, **meta).logit,
                net(changed_a, changed_b, va, vb, s, **meta).logit)

    def test_padding_does_not_enter_any_arm(self):
        a, b, va, vb, s, meta, _ = fixture()
        aa, bb = a.clone(), b.clone()
        aa[~va], bb[~vb] = float("nan"), float("nan")
        for arm in m.ARMS:
            net = network(arm)
            torch.testing.assert_close(net(a, b, va, vb, s, **meta).logit, net(aa, bb, va, vb, s, **meta).logit)

    def test_safe_indices_masks_and_candidate_padding(self):
        a, b, va, vb, s, meta, _ = fixture()
        for index in (-1, 5, 4):  # negative, out-of-range, padded token
            indices = s.candidate_indices.clone()
            indices[0, 0, 0] = index
            with self.assertRaises(ValueError):
                network("matched_tokens")(a, b, va, vb, replace(s, candidate_indices=indices))
        wrong = s.mask_a.clone(); wrong[0, 3] = True
        with self.assertRaises(ValueError):
            network("matched_tokens")(a, b, va, vb, replace(s, mask_a=wrong))
        wrong_indices = s.candidate_indices.clone(); wrong_indices[0, 511] = 0
        with self.assertRaises(ValueError):
            network("matched_edges")(a, b, va, vb, replace(s, candidate_indices=wrong_indices), **meta)

    def test_repeated_endpoint_deduplicated_but_distinct_edges_preserved(self):
        pa = torch.tensor([[[0., 0.], [30., 0.]]])
        pb = torch.tensor([[[10., 5.], [11., 5.], [40., 5.]]])
        q = torch.tensor([[[.9, .8, 0.], [0., 0., .7]]])
        va, vb = torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)
        s = m.select_predicted_inliers(q, pa, pb, va, vb)
        out = network("matched_edges")(torch.randn(1, 2, 8), torch.randn(1, 3, 8), va, vb, s,
            assignment=q, points_a_rc=pa, points_b_rc=pb)
        self.assertEqual(int(out.selected_token_count_a), 2)
        self.assertEqual(int(out.selected_token_count_b), 3)
        self.assertEqual(int(out.inlier_edge_count), 3)
        indices = s.candidate_indices.clone(); indices[:, 1] = indices[:, 0]
        with self.assertRaises(ValueError):
            m.validate_selection(replace(s, candidate_indices=indices), va, vb)

    def test_swap_invariance_including_real_selector(self):
        a, b, va, vb, s, meta, q = fixture()
        reverse = m.select_predicted_inliers(q.transpose(1, 2), meta["points_b_rc"], meta["points_a_rc"], vb, va)
        for arm in m.ARMS:
            net = network(arm)
            ab = net(a, b, va, vb, s, **meta).logit
            ba = net(b, a, vb, va, reverse, assignment=q.transpose(1, 2),
                points_a_rc=meta["points_b_rc"], points_b_rc=meta["points_a_rc"]).logit
            torch.testing.assert_close(ab, ba, atol=2e-6, rtol=2e-5)

    def test_edge_metadata_cache_agrees_and_is_detached(self):
        a, b, va, vb, s, meta, q = fixture()
        q.requires_grad_(); meta["points_a_rc"].requires_grad_()
        derived = m.edge_metadata(s, va, vb, assignment=q,
            points_a_rc=meta["points_a_rc"], points_b_rc=meta["points_b_rc"])
        self.assertFalse(derived.requires_grad)
        self.assertTrue(torch.equal(derived[..., 0][s.candidate_inliers], meta["candidate_weights"][s.candidate_inliers]))
        self.assertEqual(float(derived[..., 1].sum()), 0.)
        net = network("matched_edges")
        cached = net(a, b, va, vb, s, candidate_weights=meta["candidate_weights"], edge_residual_norm=derived[..., 1])
        torch.testing.assert_close(cached.logit, net(a, b, va, vb, s, **meta).logit)
        with self.assertRaises(ValueError):
            net(a, b, va, vb, s)
        with self.assertRaises(ValueError):
            net(a, b, va, vb, s, assignment=q, candidate_weights=meta["candidate_weights"]+1,
                points_a_rc=meta["points_a_rc"], points_b_rc=meta["points_b_rc"])

    def test_no_candidate_fallback_trainable_not_global_rescue(self):
        a, b, va, vb, _, meta, q = fixture()
        s = m.select_predicted_inliers(torch.zeros_like(q), meta["points_a_rc"], meta["points_b_rc"], va, vb)
        for arm in ("matched_tokens", "matched_edges"):
            net = network(arm)
            with torch.no_grad(): net.head.no_evidence_logit.fill_(-2.)
            out = net(a, b, va, vb, s, candidate_weights=torch.zeros_like(s.candidate_valid, dtype=a.dtype),
                points_a_rc=meta["points_a_rc"], points_b_rc=meta["points_b_rc"])
            self.assertTrue(bool(out.used_fallback)); self.assertFalse(bool(out.has_decoded_candidate))
            self.assertFalse(bool(out.has_raw_candidates)); self.assertEqual(float(out.logit), -2.)
            F.binary_cross_entropy_with_logits(out.logit, torch.ones(1)).backward()
            self.assertNotEqual(float(net.head.no_evidence_logit.grad), 0.)
        self.assertFalse(bool(network("all_tokens")(a, b, va, vb, s).used_fallback))

    def test_valid_candidate_negative_not_forced_positive(self):
        a, b, va, vb, s, meta, _ = fixture()
        for arm in m.ARMS:
            net = network(arm)
            with torch.no_grad():
                net.head.classifier[-1].weight.zero_(); net.head.classifier[-1].bias.fill_(-10.)
            out = net(a, b, va, vb, s, **meta)
            self.assertTrue(bool(out.has_decoded_candidate))
            self.assertLess(float(out.logit.sigmoid()), .001)

    def test_gradients_only_selected_features_and_trainable_heads(self):
        for arm in m.ARMS:
            a, b, va, vb, s, meta, _ = fixture()
            a.requires_grad_(); b.requires_grad_()
            net = network(arm)
            F.binary_cross_entropy_with_logits(net(a, b, va, vb, s, **meta).logit, torch.zeros(1)).backward()
            self.assertTrue(torch.isfinite(a.grad).all())
            self.assertGreater(float(a.grad.abs().sum()), 0.)
            if arm != "all_tokens":
                self.assertEqual(float(a.grad[~s.mask_a].abs().sum()), 0.)
                self.assertEqual(float(b.grad[~s.mask_b].abs().sum()), 0.)
            self.assertGreater(float(net.head.cross_attention.in_proj_weight.grad.abs().sum()), 0.)
            if arm == "matched_edges":
                self.assertGreater(float(net.mate_projection.weight.grad.abs().sum()), 0.)
                self.assertGreater(float(net.edge_projection.weight.grad.abs().sum()), 0.)

    def test_edge_pairing_not_just_two_endpoint_sets(self):
        a, b, va, vb, s, meta, _ = fixture()
        active_ids = torch.nonzero(s.candidate_inliers[0]).flatten()
        changed_indices = s.candidate_indices.clone()
        changed_indices[0, active_ids, 1] = changed_indices[0, active_ids.roll(1), 1]
        changed = replace(s, candidate_indices=changed_indices)
        # Keep metadata fixed to isolate paired feature construction, not geometry.
        kwargs = dict(candidate_weights=meta["candidate_weights"], edge_residual_norm=torch.zeros_like(meta["candidate_weights"]))
        net = network("matched_edges")
        self.assertFalse(torch.allclose(net(a, b, va, vb, s, **kwargs).logit,
            net(a, b, va, vb, changed, **kwargs).logit, atol=1e-7, rtol=1e-7))
        local = network("matched_tokens")
        torch.testing.assert_close(local(a, b, va, vb, s).logit, local(a, b, va, vb, changed).logit)

    def test_batch16_matches_individual_with_mixed_no_candidate(self):
        a, b, va, vb, s, meta, q = fixture()
        q16 = q.repeat(16, 1, 1); q16[1] = 0.
        pa, pb = meta["points_a_rc"].repeat(16, 1, 1), meta["points_b_rc"].repeat(16, 1, 1)
        va16, vb16 = va.repeat(16, 1), vb.repeat(16, 1)
        sel = m.select_predicted_inliers(q16, pa, pb, va16, vb16)
        for arm in m.ARMS:
            net = network(arm)
            out = net(a.repeat(16, 1, 1), b.repeat(16, 1, 1), va16, vb16, sel,
                assignment=q16, points_a_rc=pa, points_b_rc=pb)
            self.assertEqual(out.logit.shape, (16,))
            torch.testing.assert_close(out.logit[0], net(a, b, va, vb, s, **meta).logit[0])
            self.assertFalse(bool(out.has_decoded_candidate[1]))
            self.assertEqual(bool(out.used_fallback[1]), arm != "all_tokens")


if __name__ == "__main__":
    unittest.main()
