from copy import deepcopy
import unittest

from . import summarize_candidate_pose_alignment as p


def row():
    return dict(pair_id='case',label=True,decision_valid=True,classification={'fused':.9},
        target_translation_rc=[0.,0.],layouts={'full_top2_mode':dict(valid=True,translation_rc=[0.,0.],translation_l2_px=0.)},
        candidate_details=dict(group_logits=[3.,2.,99.],group_ranks=[1,2,3],
            group_present=[True,True,True],group_eligible=[True,True,False],
            group_translation_rc=[[100.,0.],[0.,0.],[0.,0.]],selected_group_rank=1,
            has_selected_candidate_group=True,used_fallback=False,selected_group_translation_rc=[100.,0.]))


class PoseTests(unittest.TestCase):
    thresholds={'max_f1':.8,'recall_99':.4}

    def test_gt_best_not_deployment_and_ineligible_largest_logit_excluded(self):
        r=row();before=deepcopy(r);d=p.inspect_row(r,'test',self.thresholds)
        self.assertEqual(r,before)
        self.assertEqual(d['selected_group_rank'],1)
        self.assertFalse(d['gt_diagnostic']['highest_scorer_layout20'])
        self.assertTrue(d['gt_diagnostic']['production_layout20'])
        self.assertEqual(d['gt_diagnostic']['gt_best_eligible_group_rank'],2)
        self.assertEqual(d['gt_diagnostic']['gt_best_eligible_scorer_order_1based'],2)
        a=p.aggregate([d],self.thresholds)
        self.assertEqual(a['positive_gt_pose']['hypothetical_loss'],1)
        self.assertEqual(a['operating_points']['max_f1']['deployed_production_layout20_accepted'],1)
        self.assertEqual(a['operating_points']['max_f1']['hypothetical_highest_scorer_layout20_accepted'],0)

    def test_negative_and_ood_never_receive_pose_gt_claim(self):
        r=row();r['label']=False
        d=p.inspect_row(r,'test',self.thresholds);self.assertIsNone(d['gt_diagnostic'])
        r['label']=True
        d=p.inspect_row(r,'ood',self.thresholds);self.assertIsNone(d['gt_diagnostic'])
        a=p.aggregate([d],self.thresholds)
        self.assertNotIn('positive_gt_pose',a)
        self.assertNotIn('deployed_production_layout20_accepted',a['operating_points']['max_f1'])

    def test_gt_best_never_uses_ineligible_perfect_pose(self):
        r=row();r['candidate_details']['group_translation_rc'][1]=[30.,0.]
        d=p.inspect_row(r,'test',self.thresholds)['gt_diagnostic']
        self.assertEqual(d['gt_best_eligible_group_rank'],2)
        self.assertEqual(d['gt_best_eligible_error_l2_px'],30.)
        self.assertFalse(d['gt_best_eligible_layout20_coverage'])

    def test_missing_gt_is_missing_not_failed(self):
        r=row();r['target_translation_rc']=None
        self.assertIsNone(p.inspect_row(r,'test',self.thresholds)['gt_diagnostic'])

    def test_noeligible_fallback_is_not_oracle_pose(self):
        r=row();d=r['candidate_details']
        d.update(group_eligible=[False]*3,group_logits=[None]*3,selected_group_rank=0,
            has_selected_candidate_group=False,used_fallback=True,selected_group_translation_rc=[None,None])
        a=p.inspect_row(r,'test',self.thresholds)
        self.assertIsNone(a['highest_scorer_translation_rc'])
        self.assertFalse(a['gt_diagnostic']['gt_best_eligible_layout20_coverage'])
        self.assertTrue(a['gt_diagnostic']['loss_not_deployed'])

    def test_changing_gt_does_not_select_different_scorer_group(self):
        r=row();a=p.inspect_row(r,'test',self.thresholds)
        r['target_translation_rc']=[100.,0.];r['layouts']['full_top2_mode']['translation_l2_px']=100.
        b=p.inspect_row(r,'test',self.thresholds)
        self.assertEqual(a['selected_group_rank'],b['selected_group_rank'])
        self.assertNotEqual(a['gt_diagnostic']['gt_best_eligible_group_rank'],b['gt_diagnostic']['gt_best_eligible_group_rank'])

    def test_tied_logits_first_slot_saved_selection_must_match(self):
        r=row();r['candidate_details']['group_logits']=[3.,3.,None]
        self.assertEqual(p.inspect_row(r,'test',self.thresholds)['selected_slot'],0)
        r['candidate_details']['selected_group_rank']=2
        with self.assertRaisesRegex(ValueError,'argmax'):p.inspect_row(r,'test',self.thresholds)

    def test_real_negative_cohorts_and_decision_gate(self):
        r=row();r.update(label=False,strict_member=True,review_status=None,decision_valid=False)
        a=p.inspect_row(r,'real',self.thresholds)
        self.assertEqual(a['cohort'],'real_original_negative39')
        self.assertFalse(any(a['accepted'].values()))
        r['strict_member']=False
        self.assertEqual(p.inspect_row(r,'real',self.thresholds)['cohort'],'real_constructed_negative469')

    def test_translation_sign_and_reported_error_checked(self):
        r=row();r['layouts']['full_top2_mode']['translation_l2_px']=100
        with self.assertRaisesRegex(ValueError,'GT error disagrees'):p.inspect_row(r,'test',self.thresholds)


if __name__=='__main__':unittest.main()
