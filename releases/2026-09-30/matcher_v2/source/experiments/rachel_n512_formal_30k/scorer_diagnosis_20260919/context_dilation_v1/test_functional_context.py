import copy
import unittest
import torch
from staging.pairwise_v0_2.models.rachel_n512 import CyclicLandmarkContext, RachelN512Config
try:
    from .functional_context import context_forward, circular_conv
except ImportError:
    from functional_context import context_forward, circular_conv


class FunctionalContextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(260919)
        self.context = CyclicLandmarkContext(RachelN512Config(feature_dim=8, num_heads=2,
            landmark_count=8, context_layers=2)).eval().requires_grad_(False)
        a, b = torch.randn(2, 37, 8), torch.randn(2, 53, 8)
        va, vb = torch.ones(2, 37, dtype=torch.bool), torch.ones(2, 53, dtype=torch.bool)
        va[0, -3:], vb[1, -5:] = False, False
        self.inputs = a, b, va, vb, torch.rand(2, 37, 2) * 799, torch.rand(2, 53, 2) * 799, 800

    def test_dilation1_reproduces_original_full_context(self):
        with torch.inference_mode():
            actual = context_forward(self.context, *self.inputs, dilation=1)
            expected = self.context(*self.inputs)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_dilation2_matches_independent_copied_conv_configuration(self):
        alternative = copy.deepcopy(self.context)
        for block in alternative.blocks:
            for i in (0, 3):
                old = block[i]
                replacement = torch.nn.Conv1d(old.in_channels, old.out_channels, old.kernel_size,
                    stride=old.stride, padding=old.kernel_size[0] - 1, dilation=2,
                    groups=old.groups, bias=old.bias is not None, padding_mode="circular")
                replacement.load_state_dict(old.state_dict())
                block[i] = replacement.eval().requires_grad_(False)
        with torch.inference_mode():
            actual = context_forward(self.context, *self.inputs, dilation=2)
            expected = alternative(*self.inputs)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_shapes_valid_mask_finiteness_and_original_unchanged(self):
        state = {k: v.clone() for k, v in self.context.state_dict().items()}
        configs = [(m.padding, m.dilation) for m in self.context.modules() if isinstance(m, torch.nn.Conv1d)]
        with torch.inference_mode():
            actual = context_forward(self.context, *self.inputs, dilation=2)
        for value, original, valid in zip(actual, self.inputs[:2], self.inputs[2:4]):
            self.assertEqual(value.shape, original.shape)
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue((value[~valid] == 0).all())
        for k, v in self.context.state_dict().items():
            self.assertTrue(torch.equal(v, state[k]))
        self.assertEqual(configs, [(m.padding, m.dilation) for m in self.context.modules() if isinstance(m, torch.nn.Conv1d)])

    def test_reject_unregistered_dilation(self):
        with self.assertRaises(ValueError):
            circular_conv(self.context.blocks[0][0], torch.zeros(1, 8, 32), 3)


if __name__ == "__main__":
    unittest.main()
