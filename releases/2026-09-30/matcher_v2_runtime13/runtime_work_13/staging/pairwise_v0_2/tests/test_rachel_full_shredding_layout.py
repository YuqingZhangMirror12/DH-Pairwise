"""A known seam exercises the real ShreddingNet morphology and pose solver."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from staging.pairwise_v0_2.baselines.rachel_shreddingnet_benchmark import ReleaseRecipe
from staging.pairwise_v0_2.models.rachel_full_shredding_layout import RachelFullShreddingLayout
from staging.pairwise_v0_2.models.rachel_pairwise_layout import RachelPairwiseLayout
from staging.pairwise_v0_2.models.shredding_layout_head import ShreddingLayoutHead
from staging.pairwise_v0_2.models.translation_layout import TranslationLayoutConfig


class FixedPairEvidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0
        self.output = SimpleNamespace(
            assignment=torch.eye(32).flip(1)[None].repeat(2, 1, 1),
            affinity=torch.eye(32).flip(1)[None].repeat(2, 1, 1),
            coarse_probability=torch.tensor([0.001, 0.8]),
            local_probability=torch.tensor([0.002, 0.9]),
            fused_probability=torch.tensor([0.001, 0.95]),
            translation_dispersion_px=torch.tensor([30.0, 40.0]),
        )

    def forward(self, *inputs):
        self.calls += 1
        return self.output


class KnownAntidiagonalMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, mask_a, mask_b, points_a, points_b, valid_a, valid_b):
        return torch.eye(32, device=points_a.device).flip(1)[None].repeat(len(points_a), 1, 1)


class FullShreddingLayoutTest(unittest.TestCase):
    def test_fixed_fine_layout_preserves_full_scores_and_places_known_seam(self):
        pair = FixedPairEvidence()
        full = RachelPairwiseLayout(pair, {"full_top2_mode": TranslationLayoutConfig(
            correspondence_mode="topk_union", top_k=2)})
        head = ShreddingLayoutHead(KnownAntidiagonalMatcher(), ReleaseRecipe(), amp=False)
        model = RachelFullShreddingLayout(full, head)
        index = torch.arange(32, dtype=torch.float32)
        points_a = torch.stack((index * 2, index * 3), dim=1)[None].repeat(2, 1, 1)
        expected = torch.tensor([[3.0, -5.0], [-4.0, 8.0]])
        points_b = points_a.flip(1) + expected[:, None]
        masks = torch.zeros((2, 1, 8, 8))
        valid = torch.ones((2, 32), dtype=torch.bool)
        output = model(masks, masks, points_a, points_b, valid, valid,
                       return_correspondence=True)
        self.assertEqual(pair.calls, 1)
        self.assertIs(output.pair_output, pair.output)
        for name in ("coarse_probability", "local_probability", "fused_probability"):
            self.assertIs(getattr(output, name), getattr(pair.output, name))
        self.assertEqual(set(output.layouts), {"full_top2_mode", "full_with_shred_matching_layout"})
        self.assertTrue(output.primary_layout.valid.all())
        self.assertTrue(output.primary_layout.computed.all())
        torch.testing.assert_close(output.primary_layout.t_a_to_b_rc, expected.double())
        torch.testing.assert_close(points_b.double() + output.primary_layout.offset_b_in_a_rc[:, None],
                                   points_a.flip(1).double())
        self.assertGreaterEqual(output.primary_layout.diagnostics[0]["inlier_count"], 16)
        self.assertIsNotNone(output.primary_layout.head_output.correspondences)
        torch.testing.assert_close(pair.output.translation_dispersion_px, torch.tensor([30.0, 40.0]))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
