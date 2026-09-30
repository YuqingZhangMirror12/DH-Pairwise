import unittest
import torch
from staging.pairwise_v0_2.models.rachel_decoupled_score import CrossAttentionPairHead
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.pooling_factor_probe import factor_probe


class PoolingFactorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        self.a, self.b = torch.randn(7, 12), torch.randn(11, 12)
        self.ma = torch.tensor([1, 0, 1, 0, 0, 0, 1], dtype=torch.bool)
        self.mb = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool)

    def test_depths_and_no_parameter_mutation(self):
        for depth in (1, 2, 4):
            with self.subTest(depth=depth):
                head = CrossAttentionPairHead(12, 3, depth=depth).eval().requires_grad_(False)
                before = {k: v.clone() for k, v in head.state_dict().items()}
                out = factor_probe(head, self.a, self.b, self.ma, self.mb, repeats=2)
                self.assertLess(out['factor_replay_logit_error'], 1e-6)
                self.assertEqual(len(out['interventions']), 12)
                self.assertTrue(all(torch.equal(v, head.state_dict()[k]) for k, v in before.items()))

    def test_all_tokens_are_exact_full_forward(self):
        head = CrossAttentionPairHead(12, 3, depth=2).eval()
        out = factor_probe(head, self.a, self.b, torch.ones(7,dtype=torch.bool), torch.ones(11,dtype=torch.bool), repeats=2)
        self.assertTrue(all(abs(r['delta_logit']) < 1e-6 for r in out['interventions']))

    def test_pool_does_not_rerun_attention(self):
        head = CrossAttentionPairHead(12, 3, depth=2).eval()
        calls = []
        hook = head.cross_attention.register_forward_hook(lambda *_: calls.append(1))
        try:
            factor_probe(head, self.a, self.b, self.ma, self.mb, repeats=2)
        finally:
            hook.remove()
        # Original forward 2, decoded-full 2, three subset forwards x2.
        self.assertEqual(len(calls), 10)

    def test_empty_inlier_no_fabricated_pool(self):
        head = CrossAttentionPairHead(12, 3).eval()
        out = factor_probe(head, self.a, self.b, torch.zeros(7,dtype=torch.bool), self.mb, repeats=2)
        self.assertTrue(out['no_inlier_evidence'])
        self.assertEqual(out['interventions'], [])


if __name__ == '__main__':
    unittest.main()
