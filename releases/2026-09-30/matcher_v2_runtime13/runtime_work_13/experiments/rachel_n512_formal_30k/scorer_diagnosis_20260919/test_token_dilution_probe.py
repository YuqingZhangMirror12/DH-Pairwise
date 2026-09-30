import unittest
import numpy as np
import torch

from staging.pairwise_v0_2.models.rachel_decoupled_score import CrossAttentionPairHead
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.token_dilution_probe import (
    compact_membership, intervention_masks, run_head_probe,
)


class DilutionTests(unittest.TestCase):
    def test_noncontiguous_original_ids(self):
        np.testing.assert_array_equal(compact_membership([0, 3, 6, 8], [3, 8]), [0, 1, 0, 1])
        with self.assertRaises(ValueError):
            compact_membership([0, 3, 6, 8], [2])

    def test_nested_context_preserves_seam(self):
        ma, mb = np.arange(16) < 3, np.arange(10) < 2
        specs = intervention_masks(ma, mb, repeats=4)
        for rep in range(4):
            doses = [s for s in specs if s["kind"] == "retain_inliers_add_context" and s["repeat"] == rep]
            for before, after in zip(doses, doses[1:]):
                for old, new in zip(before["masks"], after["masks"]):
                    self.assertTrue(np.all(~old | new))
            for spec in doses:
                for member, mask in zip([ma, mb], spec["masks"]):
                    self.assertTrue(mask[member].all())
        for spec in specs:
            if spec["kind"] == "remove_random_same_count":
                self.assertEqual([int((~m).sum()) for m in spec["masks"]], [3, 2])

    def test_seed_replay(self):
        ma, mb = np.arange(20) < 4, np.arange(12) < 5
        a, b = intervention_masks(ma, mb, seed=9), intervention_masks(ma, mb, seed=9)
        for aa, bb in zip(a, b):
            for am, bm in zip(aa["masks"], bb["masks"]):
                np.testing.assert_array_equal(am, bm)

    def test_real_attention_forward_and_no_mutation(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        head = CrossAttentionPairHead(8, 2, depth=2).eval()
        state = {k: v.clone() for k, v in head.state_dict().items()}
        a, b = torch.randn(16, 8), torch.randn(12, 8)
        result = run_head_probe(head, a, b, np.arange(16) < 3, np.arange(12) < 2, repeats=2)
        self.assertLess(result["permutation_logit_error"], 1e-6)
        full = [r for r in result["interventions"] if r.get("noninlier_fraction") == 1][0]
        self.assertEqual(full["delta_logit"], 0.)
        self.assertEqual((full["count_a"], full["count_b"]), (16, 12))
        for key, value in head.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]))
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_empty_inlier_case_is_explicit(self):
        head = CrossAttentionPairHead(8, 2).eval()
        result = run_head_probe(head, torch.randn(4, 8), torch.randn(3, 8),
                                np.zeros(4, bool), np.zeros(3, bool), repeats=2)
        local = [r for r in result["interventions"] if r.get("noninlier_fraction") == 0][0]
        self.assertTrue(local["no_evidence"])
        self.assertEqual(local["logit"], float(head.no_evidence_logit))


if __name__ == "__main__":
    unittest.main()
