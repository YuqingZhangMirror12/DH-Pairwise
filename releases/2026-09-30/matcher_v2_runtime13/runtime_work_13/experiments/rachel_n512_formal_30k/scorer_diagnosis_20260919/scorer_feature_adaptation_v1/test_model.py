"""Small CPU-only synthetic tests; no checkpoints, real data or training run."""
from dataclasses import fields
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from staging.pairwise_v0_2.models.rachel_decoupled_score import build_decoupled_score_model
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, RachelN512Pairwise
from staging.pairwise_v0_2.models.translation_layout import (
    TranslationLayoutConfig, estimate_translation_layout,
)
from ..pair_grid_readout.model import PairGridHead
from .model import build_g0_g1, pair_bce_with_same_anchor_ranking


class FeatureAdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(270928)
        config = RachelN512Config(canvas_size=64, coarse_size=32, contour_cap=8,
            feature_dim=16, landmark_count=4, context_layers=1, sinkhorn_iterations=30,
            window_sizes_px=(7., 16., 32., 64.), activation_checkpointing=False)
        self.source = build_decoupled_score_model(config, "cross_attention", phase="classifier",
            model_options={"cross_attention_depth": 4})
        self.g0, self.g1 = build_g0_g1(self.source)
        mask = torch.zeros(2, 1, 64, 64)
        mask[0, :, 12:40, 12:40] = 1
        mask[1, :, 10:48, 12:35] = 1
        points = torch.tensor([[[12., 12.], [12., 25.], [12., 39.], [25., 39.],
            [39., 39.], [39., 25.], [39., 12.], [25., 12.]]]).repeat(2, 1, 1)
        valid = torch.ones(2, 8, dtype=torch.bool)
        self.inputs = (mask, mask.flip(-1), points, points.flip(1), valid, valid)

    def test_identical_initialization_no_aliases_and_fresh_d2_head(self):
        a, b = self.g0.state_dict(), self.g1.state_dict()
        self.assertEqual(a.keys(), b.keys())
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), name)
            if a[name].numel():
                self.assertNotEqual(a[name].data_ptr(), b[name].data_ptr(), name)
        for name in ("patch_sampler", "patch_encoder", "scale_gate", "context"):
            orig = getattr(self.source.base_model, name).state_dict()
            copied = getattr(self.g1.scorer_stem, name).state_dict()
            for key in orig:
                self.assertTrue(torch.equal(orig[key], copied[key]), name + key)
                self.assertNotEqual(orig[key].data_ptr(), copied[key].data_ptr(), name + key)
        self.assertEqual(self.g1.score_head.decoder.depth, 2)
        self.assertEqual(self.g1.score_head.decoder.cross_attention.num_heads, 4)
        self.assertEqual(self.g1.score_head.tau, 15.)
        self.assertFalse(torch.equal(self.source.score_head.cross_attention.in_proj_weight,
            self.g1.score_head.decoder.cross_attention.in_proj_weight))
        self.assertFalse(any(hasattr(self.g1.scorer_stem, k)
            for k in ("primal", "dual", "coarse", "fusion", "local_head")))
        self.assertIs(self.g1.scorer_stem._encode_patches.__func__, RachelN512Pairwise._encode_patches)
        out0, out1 = (m.scorer_forward(*self.inputs) for m in (self.g0, self.g1))
        self.assertTrue(torch.equal(out0.raw_similarity, out1.raw_similarity))
        self.assertTrue(torch.equal(out0.calibrated_logit, out1.calibrated_logit))

    def test_stem_reproduces_source_context_before_adaptation(self):
        with torch.no_grad():
            original = self.source.base_model(*self.inputs)
            a, b = self.g1.scorer_stem(*self.inputs)
        self.assertTrue(torch.equal(original.token_features_a, a))
        self.assertTrue(torch.equal(original.token_features_b, b))

    def test_gradient_boundary_g0_vs_g1(self):
        for model in (self.g0, self.g1):
            model.train()
            self.assertFalse(model.base_model.training)
            self.assertEqual(model.scorer_stem.training, model.feature_trainable)
            with patch.object(model.base_model, "forward", side_effect=AssertionError("Matcher called")):
                output = model.scorer_forward(*self.inputs)
            F.binary_cross_entropy_with_logits(output.calibrated_logit, torch.ones(2)).backward()
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.base_model.parameters()))
            for name in ("patch_encoder", "scale_gate", "context"):
                parameters = list(getattr(model.scorer_stem, name).parameters())
                if model.feature_trainable:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters), name)
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters), name)
                else:
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in parameters), name)
            for parameter in (model.score_head.raw_scale, model.score_head.bias,
                              model.score_head.decoder.cross_attention.in_proj_weight):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.source.parameters()))

    def test_raw_affine_matches_existing_pair_grid_and_rejects_empty(self):
        with torch.no_grad():
            a, b = self.g1.scorer_stem(*self.inputs)
        head = self.g1.score_head
        with torch.no_grad():
            head.raw_scale.fill_(2.4)
            head.bias.fill_(-.8)
        out = head.forward_scores(a, b, *self.inputs[-2:])
        direct = PairGridHead.forward(head, a, b, *self.inputs[-2:])
        self.assertTrue(torch.equal(direct, out.calibrated_logit))
        torch.testing.assert_close(out.calibrated_logit, head.scale * out.raw_similarity + head.bias)
        previous_raw = out.raw_similarity.detach().clone()
        with torch.no_grad():
            head.bias.add_(5)
        self.assertTrue(torch.equal(previous_raw, head.forward_scores(a, b, *self.inputs[-2:]).raw_similarity))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            head(a, b, self.inputs[-2] & False, self.inputs[-1])

    def test_frozen_matcher_and_production_layout_unchanged_after_scorer_step(self):
        model = self.g1.train()
        inputs = tuple(t[:1] for t in self.inputs)
        with torch.no_grad():
            original = self.source.base_model(*inputs)
        matcher_before = {k: t.clone() for k, t in model.base_model.state_dict().items()}
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01)
        optimizer.zero_grad()
        model.scorer_forward(*inputs).calibrated_logit.sum().backward()
        optimizer.step()  # A single synthetic gradient check, not a training job.
        result = model(*inputs)
        for name in ("assignment", "affinity", "translation_hat_rc", "translation_hat_xy_cartesian",
                     "translation_dispersion_px", "matched_mass", "token_features_a", "token_features_b"):
            self.assertTrue(torch.equal(getattr(original, name), getattr(result, name)), name)
        for name, t in model.base_model.state_dict().items():
            self.assertTrue(torch.equal(matcher_before[name], t), name)
        config = TranslationLayoutConfig(correspondence_mode="topk_union", top_k=2,
            max_candidates=512, min_inliers=3, inlier_radius_px=10.)
        layouts = [estimate_translation_layout(inputs[2][0].numpy(), inputs[3][0].numpy(),
            x.assignment[0].detach().numpy(), inputs[4][0].numpy(), inputs[5][0].numpy(), config=config)
            for x in (original, result)]
        for field in fields(layouts[0]):
            a, b = getattr(layouts[0], field.name), getattr(layouts[1], field.name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                self.assertEqual(a, b, field.name)
        self.assertTrue(model.metadata()["adaptation_not_reproduction"])
        self.assertFalse(model.metadata()["colleague_window_auxiliary_used"])

    def test_both_modes_stay_frozen_where_required(self):
        for model in (self.g0, self.g1):
            model.eval()
            self.assertFalse(model.scorer_stem.training)
            model.train()
            self.assertFalse(model.base_model.training)
            self.assertTrue(model.score_head.training)
            with self.assertRaises(ValueError):
                model.set_phase("matcher")

    def test_matcher_phase_source_supported_without_mutation(self):
        self.source.set_phase("matcher")
        before = {name: (p.detach().clone(), p.requires_grad)
                  for name, p in self.source.named_parameters()}
        was_training = self.source.training
        models = build_g0_g1(self.source)
        self.assertEqual(self.source.phase, "matcher")
        self.assertEqual(self.source.training, was_training)
        for name, p in self.source.named_parameters():
            self.assertTrue(torch.equal(p, before[name][0]))
            self.assertEqual(p.requires_grad, before[name][1])
        for model in models:
            self.assertTrue(all(not p.requires_grad for p in model.base_model.parameters()))
            self.assertFalse(model.base_model.training)
            self.assertEqual(model.metadata()["source_model"]["phase"], "matcher")


class SameAnchorLossTests(unittest.TestCase):
    def setUp(self):
        self.raw = torch.tensor([.6, .5, .2, .9, .8, .1], requires_grad=True)
        self.logit = torch.tensor([.2, -.1, -.4, .8, -.2, 1.], requires_grad=True)
        self.labels = torch.tensor([1., 0., 0., 1., 0., 1.])
        self.edges = torch.tensor([[0, 1], [0, 2], [3, 4]], dtype=torch.long)
        self.ids = ["a", "a", "a", "b", "b", "c"]

    def loss(self, **kwargs):
        options = dict(known_negative_pairs=self.edges, anchor_ids=self.ids, same_anchor_confirmed=True)
        options.update(kwargs)
        return pair_bce_with_same_anchor_ranking(self.raw, self.logit, self.labels, **options)

    def test_formula_hardest_only_and_unlisted_positive_skipped(self):
        out = self.loss()
        expected_rank = ((.15 - .6 + .5) + (.15 - .9 + .8)) / 2
        self.assertAlmostEqual(out.ranking.item(), expected_rank, places=6)
        expected_bce = F.binary_cross_entropy_with_logits(self.logit, self.labels)
        torch.testing.assert_close(out.pair_bce, expected_bce)
        torch.testing.assert_close(out.total, expected_bce + .3 * out.ranking)
        self.assertEqual((out.ranked_positive_count, out.skipped_positive_count), (2, 1))
        out.ranking.backward()
        torch.testing.assert_close(self.raw.grad, torch.tensor([-.5, .5, 0., -.5, .5, 0.]))
        self.assertIsNone(self.logit.grad)  # Ranking does not use calibrated logits/sigmoid.

    def test_no_negative_is_differentiable_zero_and_keeps_all_bce(self):
        out = self.loss(known_negative_pairs=None, anchor_ids=None, same_anchor_confirmed=False)
        self.assertEqual(out.ranking.item(), 0.)
        self.assertEqual((out.ranked_positive_count, out.skipped_positive_count), (0, 3))
        self.assertTrue(torch.equal(out.total, out.pair_bce))
        out.total.backward()
        self.assertTrue(torch.equal(self.raw.grad, torch.zeros_like(self.raw)))
        self.assertTrue(torch.all(self.logit.grad != 0))

    def test_pure_rank_is_common_translation_invariant(self):
        shifted = pair_bce_with_same_anchor_ranking(self.raw + 8., self.logit, self.labels,
            known_negative_pairs=self.edges, anchor_ids=self.ids, same_anchor_confirmed=True)
        torch.testing.assert_close(self.loss().ranking, shifted.ranking, atol=1e-6, rtol=0)

    def test_requires_known_same_anchor_edges_not_all_batch_negatives(self):
        failures = [dict(same_anchor_confirmed=False), dict(anchor_ids=None),
            dict(anchor_ids=["a", "other", "a", "b", "b", "c"]),
            dict(known_negative_pairs=torch.tensor([[0, 3]])),  # positive is not negative
            dict(known_negative_pairs=torch.tensor([[0, 4]])),  # cross-anchor negative
            dict(known_negative_pairs=torch.tensor([[0, 1], [0, 1]])),
            dict(known_negative_pairs=torch.tensor([[0, 99]])),
            dict(known_negative_pairs=self.edges.float())]
        for kwargs in failures:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.loss(**kwargs)
        # Same-anchor IDs do not invent an edge: no edges means no ranking.
        out = self.loss(known_negative_pairs=torch.empty((0, 2), dtype=torch.long))
        self.assertEqual(out.ranked_positive_count, 0)


if __name__ == "__main__":
    unittest.main()
