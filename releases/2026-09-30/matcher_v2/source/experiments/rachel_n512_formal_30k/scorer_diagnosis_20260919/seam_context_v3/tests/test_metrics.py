import unittest
import torch
from ..validate import metrics,select_threshold
from ..config import Config
from ..seam_verifier import material_evidence


def row(label,score,layout):
    return dict(label=label,score=score,gt_known=label,numeric_valid=True,
                has_candidate=True,layout20=layout,coverage8=layout,q0_coverage8=False)


class Tests(unittest.TestCase):
    def test_wrong_pose_is_joint_fp_and_fn_not_pair_false_label(self):
        rows=[row(True,.8,True),row(True,.8,False),row(True,.1,True),row(False,.8,False)]
        m=metrics(rows,.3)
        self.assertEqual((m['joint_tp'],m['joint_fp'],m['joint_fn']),(1,2,2))
        self.assertEqual((m['pair_tp'],m['pair_fp'],m['pair_fn']),(2,1,1))

    def test_threshold_bounds_and_no_candidate(self):
        a=row(True,.99,True);a['has_candidate']=False
        b=row(False,.05,False)
        m=metrics([a,b],.3);self.assertEqual(m['joint_fn'],1)
        threshold,_=select_threshold({'clean':[a,b],'hard':[a,b]},Config())
        self.assertAlmostEqual(threshold,.3)

    def test_overlap_is_not_hidden_by_canvas_clipping(self):
        a=torch.zeros(1,1,64,64);a[:,:,16:48,16:48]=1
        t=torch.tensor([[0.,0.],[200.,0.],[-200.,0.]])
        g=material_evidence(a,a,t)
        self.assertAlmostEqual(float(g[0,0]),1.)
        self.assertEqual(float(g[1:].abs().sum()),0.)


if __name__=='__main__':unittest.main()
