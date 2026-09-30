import unittest
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.seam_context_v3.extend_independent_evidence import revised_metric_rows


class EvidenceTests(unittest.TestCase):
    def example(self):
        row=dict(dataset='敦煌',model_id='independent_features',policy='原五折阈值冻结',
            threshold_values=[.2,.3],threshold_median=.3,
            accuracy=.8,precision=.7,recall=.6,f1=.64,auroc=.85,tp=6,fp=2,fn=4,tn=8,
            n=20,positive=10,negative=10,layout_correct_total=7,layout_correct_accepted=5)
        return dict(protocol=dict(thresholds_refit=False,original_predictions_modified=False),rows=[row])

    def test_values_and_variable_denominators_preserved(self):
        r=revised_metric_rows(self.example())[0]
        self.assertEqual(r['positive'],10)
        self.assertEqual(r['layout'],7)
        self.assertEqual(r['joint'],5)
        self.assertEqual(r['thresholds'],[.2,.3])
        self.assertEqual(r['policy'],'真实五折')

    def test_no_silent_recalibration(self):
        x=self.example();x['protocol']['thresholds_refit']=True
        with self.assertRaises(ValueError):revised_metric_rows(x)


if __name__=='__main__':unittest.main()
