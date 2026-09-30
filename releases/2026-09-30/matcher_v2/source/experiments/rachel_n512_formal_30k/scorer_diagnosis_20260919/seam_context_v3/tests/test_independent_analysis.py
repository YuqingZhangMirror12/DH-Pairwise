import copy
import tempfile
from pathlib import Path
import unittest
from ..analyze_independent import paired,proposal_differences,reference_tables
from ...bounded_real_calibration_v2.common import save,sha


class IndependentAnalysisTests(unittest.TestCase):
    def test_pair_decision_is_not_successful_layout(self):
        meta=dict(split='real',pairs=[dict(pair_id='a',label=True),dict(pair_id='b',label=False)])
        left=[dict(pair_id='a',accepted=True),dict(pair_id='b',accepted=False)]
        right=[dict(pair_id='a',accepted=False),dict(pair_id='b',accepted=False)]
        result=paired(meta,left,right,{'a':False},{'a':True})
        self.assertEqual(result['classification'],dict(left_correct__right_wrong=1,left_correct__right_correct=1))
        self.assertEqual(result['correct_layout_accepted'],dict(left_fail__right_fail=1))
        meta['split']='ood';self.assertNotIn('correct_layout_accepted',paired(meta,left,right,{},{}))

    def test_refinement_changes_do_not_change_proposal_comparison(self):
        common=dict(pair_id='a',q1_mass=.8)
        left=[dict(**common,candidates=[dict(proposal_translation_rc=[2.,3.],translation_rc=[4.,5.])])]
        right=[dict(**common,candidates=[dict(proposal_translation_rc=[2.,3.],translation_rc=[6.,7.])])]
        self.assertEqual(proposal_differences(left,right)['max_aligned_proposal_translation_difference_px'],0.)

    def test_proposal_comparison_cannot_silently_truncate(self):
        with self.assertRaisesRegex(ValueError,'population length'):
            proposal_differences([dict(pair_id='a')],[])

    def test_s7_hard_extension_preserves_historical_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';hard=root/'hard';base.mkdir();hard.mkdir()
            names=('dunhuang_cv','turufan')
            comparison=dict(splits={n:dict(models={'v3_B22':dict(score=.5)}) for n in names})
            oof={n:{'v3_B22':dict(bounded_max_f1=[dict(pair_id='a',accepted=True)])} for n in names}
            save(base/'comparison.json',comparison);save(base/'oof_predictions.json',oof)
            extended=copy.deepcopy(comparison);extended_oof=copy.deepcopy(oof)
            extended['s7_hard_extension']=dict(prior_sha256=sha(base/'comparison.json'))
            for n in names:
                extended['splits'][n]['models']['S7_H']=dict(score=.6)
                extended_oof[n]['S7_H']=dict(bounded_max_f1=[dict(pair_id='a',accepted=False)])
            save(hard/'comparison.json',extended);save(hard/'oof_predictions.json',extended_oof)
            result,pred=reference_tables(base,hard)
            self.assertEqual(result,extended);self.assertEqual(pred,extended_oof)
            extended_oof[names[0]]['v3_B22']['bounded_max_f1'][0]['accepted']=False
            save(hard/'oof_predictions.json',extended_oof)
            with self.assertRaisesRegex(ValueError,'historical out-of-fold'):
                reference_tables(base,hard)


if __name__=='__main__':unittest.main()
