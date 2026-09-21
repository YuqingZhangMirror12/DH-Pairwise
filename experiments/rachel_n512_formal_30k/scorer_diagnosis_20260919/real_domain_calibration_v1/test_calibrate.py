import unittest
from .calibrate import metrics, select_threshold
from .prepare import assign_folds, manuscript, negative_pairs


class CalibrationTests(unittest.TestCase):
    def test_max_f1_and_ties(self):
        s=select_threshold([True,True,False,False],[.7,.2,.6,.1],[True]*4,"max_f1")
        self.assertEqual(s["threshold"],.2)
        self.assertAlmostEqual(s["calibration_metrics"]["f1"],.8)

    def test_recall_target_and_invalid(self):
        s=select_threshold([True,True,False],[.7,.2,.1],[True]*3,"recall_95")
        self.assertEqual(s["threshold"],.2)
        s=select_threshold([True,True,False],[.7,.2,.1],[True,False,True],"recall_95")
        self.assertFalse(s["target_met"])
        self.assertEqual(s["calibration_metrics"]["recall"],.5)

    def test_tied_auc(self):
        m=metrics([True,False,True,False],[True]*4,[.5,.5,.5,.5])
        self.assertEqual(m["auroc"],.5)

    def test_manuscript_aliases(self):
        self.assertEqual(manuscript("m0010_recto"),manuscript("m0010_verso"))
        self.assertEqual(manuscript("m0032b_seite1"),manuscript("m0032b_seite2"))
        self.assertEqual(manuscript("chu6064versototal"),"chu6064")

    def test_negative_sampler_never_crosses_fold(self):
        groups={f"g{i}/f{j}":f"g{i}" for i in range(20) for j in range(2)}
        retained=[dict(pair_id=f"p{i}",fragment_a_id=f"g{i}/f0",fragment_b_id=f"g{i}/f1",label=True) for i in range(20)]
        folds=assign_folds(groups,retained,123)
        for r in retained:r["fold"]=folds[groups[r["fragment_a_id"]]]
        negatives=negative_pairs("ood",groups,folds,retained,20,124)
        self.assertEqual(len(negatives),20)
        self.assertEqual(len({r["pair_id"] for r in negatives}),20)
        for r in negatives:
            a,b=groups[r["fragment_a_id"]],groups[r["fragment_b_id"]]
            self.assertNotEqual(a,b)
            self.assertEqual(folds[a],folds[b])
            self.assertEqual(r["fold"],folds[a])


if __name__=="__main__":unittest.main()
