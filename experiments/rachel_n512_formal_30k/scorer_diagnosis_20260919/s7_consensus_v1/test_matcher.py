from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, RachelN512Pairwise
from . import matcher


def inputs():
    a = torch.zeros(1, 1, 32, 32)
    b = a.clone()
    a[:, :, 4:25, 3:15] = 1
    b[:, :, 7:27, 17:29] = 1
    pa = torch.tensor([[[4., 3.], [4., 14.], [24., 14.], [24., 3.], [0., 0.], [0., 0.]]])
    pb = torch.tensor([[[7., 17.], [7., 28.], [26., 28.], [26., 17.], [0., 0.], [0., 0.]]])
    valid = torch.tensor([[True, True, True, True, False, False]])
    return a, b, pa, pb, valid, valid.clone()


class MatcherAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(482)
        cfg = RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
            window_sizes_px=(7., 16., 32., 64.), context_layers=2, landmark_count=2,
            activation_checkpointing=False)
        self.base = RachelN512Pairwise(cfg).eval()

    def test_forward_parity_and_exactly_one_sinkhorn(self):
        args = inputs()
        with torch.no_grad():
            old = self.base(*args)
        adapter = matcher.S7MatcherAdapter(self.base)
        with patch.object(matcher, 'dustbin_sinkhorn', wraps=matcher.dustbin_sinkhorn) as sink:
            new = adapter(*args)
        self.assertEqual(sink.call_count, 1)
        for name, a, b in (('S', old.affinity, new.affinity), ('Q', old.assignment, new.assignment),
            ('H_a', old.token_features_a, new.context_a), ('H_b', old.token_features_b, new.context_b),
            ('U_a', old.unmatched_a, new.unmatched_a), ('U_b', old.unmatched_b, new.unmatched_b)):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)

    def test_legacy_classifier_and_decoder_are_never_called(self):
        adapter = matcher.S7MatcherAdapter(self.base)
        with ExitStack() as stack:
            for obj, name in ((self.base.coarse, 'forward'), (self.base.local_head, 'forward'),
                              (self.base.fusion, 'forward'), (self.base, '_translation')):
                stack.enter_context(patch.object(obj, name, side_effect=AssertionError(name)))
            adapter(*inputs())

    def test_freeze_is_preserved_by_outer_train(self):
        adapter = matcher.S7MatcherAdapter(self.base)
        adapter.train()
        self.assertFalse(adapter.base.training)
        self.assertFalse(any(p.requires_grad for p in adapter.parameters()))
        new = adapter(*inputs())
        self.assertFalse(new.assignment.requires_grad)
        self.assertFalse(new.context_a.requires_grad)

    def test_scratch_gradients_reach_matcher_not_unused_classifiers(self):
        adapter = matcher.S7MatcherAdapter(self.base, frozen=False).train()
        output = adapter(*inputs())
        loss = -output.assignment[:, 0, 1].clamp_min(1e-20).log().sum()
        loss.backward()
        self.assertGreater(float(self.base.primal.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.base.dual.weight.grad.abs().sum()), 0)
        self.assertTrue(any(p.grad is not None for p in self.base.patch_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in self.base.coarse.parameters()))
        self.assertNotEqual(self.base.primal.weight.data_ptr(), self.base.dual.weight.data_ptr())

    def test_invalid_padding_nan_is_not_content(self):
        adapter = matcher.S7MatcherAdapter(self.base)
        args = inputs()
        expected = adapter(*args)
        args = list(args)
        args[2] = args[2].clone()
        args[3] = args[3].clone()
        args[2][~args[4]] = float('nan')
        args[3][~args[5]] = float('nan')
        actual = adapter(*args)
        torch.testing.assert_close(expected.assignment, actual.assignment, rtol=0, atol=0)
        torch.testing.assert_close(expected.context_a, actual.context_a, rtol=0, atol=0)

    def test_endpoint_mass_with_dustbin(self):
        output = matcher.S7MatcherAdapter(self.base)(*inputs())
        qa = output.assignment.sum(2) + output.unmatched_a
        qb = output.assignment.sum(1) + output.unmatched_b
        torch.testing.assert_close(qa, output.valid_a.float(), atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(qb, output.valid_b.float(), atol=2e-4, rtol=2e-4)
        self.assertEqual(output.assignment.dtype, torch.float32)

    def test_local_descriptor_is_exposed_without_second_encoding(self):
        adapter = matcher.S7MatcherAdapter(self.base)
        with patch.object(self.base, '_encode_patches', wraps=self.base._encode_patches) as encode:
            output = adapter(*inputs())
        self.assertEqual(encode.call_count, 2)  # Once per fragment, not per pass.
        self.assertEqual(output.local_a.shape, output.context_a.shape)


if __name__ == '__main__':
    unittest.main()
