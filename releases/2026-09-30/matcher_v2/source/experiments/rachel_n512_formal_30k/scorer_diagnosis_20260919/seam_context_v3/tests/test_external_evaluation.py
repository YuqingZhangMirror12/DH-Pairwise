import unittest
import numpy as np
from ..evaluate_external import input_batches, summarize


class ExternalEvaluationTests(unittest.TestCase):
    def test_cached_inputs_keep_endpoint_order_and_mask_channel(self):
        masks=np.zeros((2,800,800),bool);masks[0,17,22]=1;masks[1,43,52]=1
        arrays=dict(packed_masks=np.packbits(masks,-1),points=np.zeros((2,512,2),np.float32),valid=np.ones((2,512),bool))
        pairs=[dict(pair_id='pair',fragment_a_id='b',fragment_b_id='a',label=True,gt='must not enter batch')]
        _,batch=next(input_batches(dict(fragment_ids=['a','b'],pairs=pairs),arrays,8))
        self.assertEqual(batch['mask_a'].shape,(1,1,800,800))
        self.assertTrue(batch['mask_a'][0,0,43,52]);self.assertTrue(batch['mask_b'][0,0,17,22])
        self.assertEqual(len(batch),6)

    def test_unknown_pose_not_reported_as_zero_accuracy(self):
        rows=[dict(label=y,gt_known=False,score=s,numeric_valid=True,has_candidate=True,layout20=False,coverage8=False,q0_coverage8=False) for y,s in [(True,.7),(False,.2)]]
        m=summarize(rows,.47)
        self.assertEqual(m['pair_f1'],1.)
        for key in ('joint_f1','layout20','candidate_coverage','wrong_pose_accepted'):
            self.assertIsNone(m[key])

    def test_layout_and_acceptance_are_separate(self):
        rows=[dict(label=True,gt_known=True,score=.2,numeric_valid=True,has_candidate=True,layout20=True,coverage8=True,q0_coverage8=True,refined_coverage8=True),
              dict(label=False,gt_known=False,score=.1,numeric_valid=True,has_candidate=True,layout20=False,coverage8=False,q0_coverage8=False,refined_coverage8=False)]
        m=summarize(rows,.47)
        self.assertEqual(m['pair_fn'],1);self.assertEqual(m['layout_correct_count'],1)
        self.assertEqual(m['correct_layout_rejected'],1)
        self.assertEqual(sum(m['failure_groups'].values()),1)


if __name__=='__main__':unittest.main()
