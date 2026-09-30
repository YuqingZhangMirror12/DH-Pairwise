import unittest
import numpy as np
from .ranking import auc, cut_at_fp, best


class RankAuditTests(unittest.TestCase):
    def test_auc_ties(self):
        self.assertEqual(auc([1,0,1,0],[1,1,1,1]),.5)

    def test_auc_perfect_and_reverse(self):
        self.assertEqual(auc([1,1,0,0],[4,3,2,1]),1.)
        self.assertEqual(auc([1,1,0,0],[1,2,3,4]),0.)

    def test_cut_rejects_boundary_tie(self):
        scores=[.8,.7,.7,.2]
        threshold=cut_at_fp(scores,2)
        self.assertEqual(sum(s>=threshold for s in scores),1)
        self.assertEqual(threshold,np.nextafter(.7,np.inf))

    def test_cut_exact_budget(self):
        scores=[.8,.6,.4,.2]
        self.assertEqual(sum(s>=cut_at_fp(scores,2) for s in scores),2)

    def test_winner_stable_and_not_assumed(self):
        r={'candidates':[{'cluster_id':0,'m':2,'s':.1},{'cluster_id':1,'m':1,'s':.8}]}
        self.assertEqual(best(r,'m')['cluster_id'],0)
        self.assertEqual(best(r,'s')['cluster_id'],1)
        r['candidates'][1]['s']=.1
        self.assertEqual(best(r,'s')['cluster_id'],0)


if __name__=='__main__': unittest.main()
