from dataclasses import replace
import unittest
import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.pose_consensus import PoseConsensusBuilder
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_consensus import make_cloud
from .audit import audit_pair, audit_proposals


class MergeAuditTests(unittest.TestCase):
    def setUp(self):
        self.builder = PoseConsensusBuilder(CompatibilityConfig(.5, 1., 1., 1., 1., 9., 1.))

    def test_identical_clear_hypotheses_pass_pre_overlap_gates(self):
        cloud = make_cloud([5])
        p = self.builder.build_from_cloud(cloud, torch.tensor([[5., 0.]]))
        r = audit_pair(self.builder, p.cloud, p.hypotheses[0], p.hypotheses[0])
        self.assertEqual(r['first_veto'], 'passes_audited_gates')
        self.assertTrue(r['equal_input_edge_sets'])

    def test_existing_alternatives_can_veto_identical_hypotheses(self):
        cloud = make_cloud([0], per_group=4)
        cloud = replace(cloud, ids=torch.tensor([[0, 0], [0, 1], [1, 2], [2, 3]]),
            displacement=torch.tensor([[-2., 0.], [2., 0.], [0., 0.], [0., 0.]]),
            spacing_a=torch.ones(4), spacing_b=torch.ones(4))
        p = self.builder.build_from_cloud(cloud, torch.zeros((2, 2)))
        self.assertEqual(len(p.clusters), 2)  # Reproduce, not endorse, the bound implementation.
        r = audit_pair(self.builder, p.cloud, p.hypotheses[0], p.hypotheses[1])
        self.assertEqual(r['first_veto'], 'same_endpoint')
        self.assertEqual(r['exclusive_pairs'], 1)
        self.assertEqual(r['newly_introduced_exclusive_pairs'], 0)
        self.assertEqual(r['union_kernel_rejected_edges'], 0)
        self.assertTrue(r['equal_input_edge_sets'])
        self.assertTrue(r['first_conflict']['already_within_input_hypothesis'])

    def test_far_modes_report_distance_before_other_gates(self):
        p = self.builder.build_from_cloud(make_cloud([0, 200]), torch.tensor([[0., 0.], [200., 0.]]))
        r = audit_pair(self.builder, p.cloud, p.hypotheses[0], p.hypotheses[1])
        self.assertEqual(r['first_veto'], 'fitted_centers_far')
        self.assertNotIn('exclusive_pairs', r)

    def test_audit_does_not_mutate_proposals(self):
        p = self.builder.build_from_cloud(make_cloud([1, 3]), torch.tensor([[1., 0.], [3., 0.]]))
        before = [x.translation.clone() for x in p.hypotheses]
        ids = p.cloud.ids.clone()
        r = audit_proposals(self.builder, p)
        self.assertEqual(len(r['identical_candidate_pairs']), 2)
        self.assertTrue(torch.equal(ids, p.cloud.ids))
        self.assertTrue(all(torch.equal(x.translation, y) for x, y in zip(p.hypotheses, before)))


if __name__ == '__main__':
    unittest.main()
