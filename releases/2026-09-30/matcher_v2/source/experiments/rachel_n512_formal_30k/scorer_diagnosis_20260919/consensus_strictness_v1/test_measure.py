import copy
from dataclasses import replace
import math
import unittest

import torch
import measure
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_evidence import fixture
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import recall_full_q


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.geometry = measure.CompatibilityConfig(.5,.23085256082452021,.2842015546641426,
            .27137619747785585,1.,9.,.5878077664079844)
        self.builder = measure.PoseConsensusBuilder(self.geometry)
        self.pair = fixture()

    def test_observer_leaves_actual_builder_unchanged(self):
        direct = self.builder(self.pair)
        seen, all_c = measure.observe_builder(self.builder,self.pair)
        self.assertEqual(len(direct.clusters),len(seen.clusters))
        self.assertGreaterEqual(len(all_c),len(seen.clusters))
        self.assertEqual(direct.merge_trace,seen.merge_trace)
        for a,b in zip(direct.clusters,seen.clusters):
            torch.testing.assert_close(a.translation,b.translation,rtol=0,atol=0)
            self.assertTrue(torch.equal(a.edge_ids,b.edge_ids))

    def test_q_readout_matches_original_recall(self):
        p,all_c=measure.observe_builder(self.builder,self.pair)
        row=measure.measure_pair(self.builder,self.pair,p,all_c)
        for c,record in zip(all_c,row['clusters']):
            recall=recall_full_q(self.pair,c.translation,self.geometry)
            self.assertAlmostEqual(float(recall.weights.sum()),record['full_q_weighted_mass'],places=7)

    def test_bandwidth_is_half_width_not_isotropic_radius(self):
        p,all_c=measure.observe_builder(self.builder,self.pair)
        row=measure.measure_pair(self.builder,self.pair,p,all_c)
        expected=math.sqrt(2)*3*self.builder._local_scale(p.cloud)
        self.assertAlmostEqual(row['common_center_radius_px'],expected)
        self.assertEqual(row['normal_material_offset_range_px'],[0,9.])

    def test_no_gt_population_stays_unknown(self):
        p,all_c=measure.observe_builder(self.builder,self.pair)
        row=measure.measure_pair(self.builder,self.pair,p,all_c)
        out=measure.attach_label(row,dict(pair_id='x',label=1),None,set())
        self.assertIsNone(out['gt_diagnostic'])
        self.assertFalse(out['gt_known'])
        self.assertTrue(all(c['gt_error_px'] is None for c in out['clusters']))

    def test_correct_endpoint_coverage_uses_unique_endpoints(self):
        row=dict(hypotheses=[dict(index=0,translation_rc=[0,0],edge_ids=[[0,0],[0,1]]),
                            dict(index=1,translation_rc=[5,0],edge_ids=[[1,2]])],
            clusters=[dict(index=0,translation_rc=[0,0],hypothesis_ids=[0],retained=True,
                support_mass_px=1,edge_ids=[[0,0],[0,1]],original_union_ids=[[0,0],[0,1]],
                full_q_significant_ids=[[0,0],[0,1]])])
        out=measure.attach_label(row,dict(pair_id='x',label=1),[0,0],set())
        d=out['gt_diagnostic'];self.assertEqual(d['correct_union_endpoints_a'],2)
        self.assertEqual(d['largest_correct_compatible_coverage']['endpoints_a'],.5)
        self.assertAlmostEqual(d['largest_correct_compatible_coverage']['edges'],2/3)
        self.assertEqual(d['correct_hypotheses_dropped'],1)

    def test_three_bad_gt_excluded_not_deleted(self):
        p,all_c=measure.observe_builder(self.builder,self.pair)
        row=measure.measure_pair(self.builder,self.pair,p,all_c)
        out=measure.attach_label(row,dict(pair_id='excluded',label=1),[0,0],{'excluded'})
        self.assertTrue(out['gt_excluded']);self.assertTrue(out['gt_known'])

    def test_actual_scale_synthetic_is_only_measurement(self):
        result=measure.synthetic(self.builder)
        self.assertEqual(len(result),15)
        self.assertTrue(all(x['arc_distance_change_fixed_seeds_invariant'] for x in result))
        self.assertLess(max(x['common_center_radius_px'] for x in result),7)


if __name__=='__main__':
    unittest.main()
