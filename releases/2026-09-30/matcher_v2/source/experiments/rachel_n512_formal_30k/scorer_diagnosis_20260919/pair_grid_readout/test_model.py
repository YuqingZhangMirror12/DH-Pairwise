"""Bounded numerical tests; no training data, model checkpoint or GPU required."""
import math
import unittest

import torch
import torch.nn.functional as F

from staging.pairwise_v0_2.models.rachel_decoupled_score import CrossAttentionPairHead, build_decoupled_score_model
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from .model import PairGridHead, FrozenPairGridModel, cosine_grid_lme


class GridReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(270920)
        self.source = CrossAttentionPairHead(16, 4, depth=2)
        self.head = PairGridHead(self.source)
        self.a, self.b = torch.randn(2, 7, 16), torch.randn(2, 9, 16)
        self.va, self.vb = torch.ones(2, 7, dtype=torch.bool), torch.ones(2, 9, dtype=torch.bool)

    def test_decoder_exactly_retained(self):
        expected = self.source._decode_pair(self.a[0], self.b[0])
        actual = self.head.decoder(self.a[0], self.b[0])
        for x, y in zip(expected, actual):
            self.assertTrue(torch.equal(x, y))
        names = [name for name, _ in self.head.named_parameters()]
        self.assertFalse(any("pool_gate" in n or "classifier" in n for n in names))

    def test_matches_direct_formula(self):
        a, b = self.head.decoder(self.a[0], self.b[0])
        grid = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T
        expected = torch.log(torch.exp(15 * grid.double()).mean()) / 15
        actual = self.head(self.a[:1], self.b[:1], self.va[:1], self.vb[:1])[0]
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_symmetry_and_permutation(self):
        original = self.head(self.a, self.b, self.va, self.vb)
        torch.testing.assert_close(original, self.head(self.b, self.a, self.vb, self.va))
        torch.testing.assert_close(original, self.head(self.a.flip(1), self.b.roll(3, 1), self.va, self.vb))

    def test_nan_padding_is_excluded(self):
        original = self.head(self.a, self.b, self.va, self.vb)
        a = F.pad(self.a, (0, 0, 0, 3), value=float("nan"))
        b = F.pad(self.b, (0, 0, 0, 2), value=float("nan"))
        torch.testing.assert_close(original, self.head(a, b, F.pad(self.va, (0, 3)), F.pad(self.vb, (0, 2))))

    def test_invalid_inputs_and_empty_fallback(self):
        with self.assertRaises(ValueError):
            self.head(self.a, self.b, self.va.float(), self.vb)
        a = self.a.clone(); a[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.head(a, self.b, self.va, self.vb)
        result = self.head(a, self.b, self.va & False, self.vb)
        self.assertTrue(torch.equal(result, torch.zeros(2)))

    def test_duplication_invariance(self):
        original = self.head(self.a, self.b, self.va, self.vb)
        repeated = self.head(self.a.repeat_interleave(2, 1), self.b.repeat_interleave(2, 1),
                             self.va.repeat_interleave(2, 1), self.vb.repeat_interleave(2, 1))
        torch.testing.assert_close(original, repeated)

    def test_unrelated_tokens_still_dilute_lme(self):
        a = torch.tensor([[1., 0.]])
        b = torch.tensor([[1., 0.], [0., 1.]])
        full = cosine_grid_lme(a, b)
        self.assertLess(full.item(), cosine_grid_lme(a, b[:1]).item())
        self.assertAlmostEqual(full.item(), 1 - math.log(2)/15, places=6)

    def test_bce_gradients_and_state_roundtrip(self):
        a, b = self.a.requires_grad_(), self.b.requires_grad_()
        score = self.head(a, b, self.va, self.vb)
        F.binary_cross_entropy_with_logits(score, torch.tensor([1., 0.])).backward()
        for grad in (a.grad, b.grad, self.head.raw_scale.grad, self.head.bias.grad,
                     self.head.decoder.cross_attention.in_proj_weight.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        clone = PairGridHead(self.source)
        clone.load_state_dict(self.head.state_dict(), strict=True)
        self.assertTrue(torch.equal(score, clone(a, b, self.va, self.vb)))
        # deepcopy: training this head cannot change the old model.
        self.assertTrue(all(p.grad is None for p in self.source.parameters()))

    def test_wrapper_freezes_matcher_and_preserves_transport(self):
        config = RachelN512Config(canvas_size=64, coarse_size=32, contour_cap=8,
            feature_dim=16, landmark_count=4, context_layers=1, sinkhorn_iterations=30,
            activation_checkpointing=False)
        source = build_decoupled_score_model(config, "cross_attention", phase="classifier",
                                             model_options={"cross_attention_depth": 2})
        model = FrozenPairGridModel(source).train()
        mask = torch.zeros(1, 1, 64, 64); mask[:, :, 12:40, 12:40] = 1
        points = torch.tensor([[[12., 12.], [12., 25.], [12., 39.], [25., 39.],
                                [39., 39.], [39., 25.], [39., 12.], [25., 12.]]])
        valid = torch.ones(1, 8, dtype=torch.bool)
        inputs = (mask, mask.flip(-1), points, points.flip(1), valid, valid)
        with torch.no_grad():
            original = source.base_model(*inputs)
        result = model(*inputs)
        for name in ("assignment", "affinity", "translation_hat_rc", "token_features_a"):
            self.assertTrue(torch.equal(getattr(original, name), getattr(result, name)))
        result.fused_logit.sum().backward()
        self.assertFalse(model.base_model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.base_model.parameters()))
        self.assertIsNotNone(model.score_head.bias.grad)


if __name__ == "__main__":
    unittest.main()
