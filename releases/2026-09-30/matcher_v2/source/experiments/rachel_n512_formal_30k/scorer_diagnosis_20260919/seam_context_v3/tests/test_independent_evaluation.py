import copy
import unittest
from ..evaluate_independent import validate_selection


def fixture(epoch=6):
    chosen=dict(epoch=epoch,key=[.9,.9,.9],threshold=.47)
    binding=dict(base_sha256='base',arm='independent_features',epochs=16,effective_batch=32)
    cp=dict(stage='independent_scorer_adaptation',arm='independent_features',binding=binding,
        selection_state=chosen,epoch=epoch+1,offset=0,updates=epoch*750)
    selected=dict(**chosen,binding=copy.deepcopy(binding),report=dict(threshold=.47))
    status=dict(status='training_complete',stop_reason='prespecified16epoch_budget_completed',
        last_epoch=16,updates=12000,exposures=384000,matcher_unchanged=True,best=copy.deepcopy(chosen))
    return cp,selected,status


class IndependentEvaluationTests(unittest.TestCase):
    def test_selected_epoch_not_last_is_valid(self):
        validate_selection(*fixture(6),'base')

    def test_real_evaluation_cannot_run_while_training(self):
        cp,selection,status=fixture(0);status['status']='training'
        with self.assertRaisesRegex(ValueError,'training not complete'):
            validate_selection(cp,selection,status,'base')
        validate_selection(cp,selection,status,'base',initial_parity=True)

    def test_parity_exception_cannot_evaluate_adapted_model(self):
        with self.assertRaisesRegex(ValueError,'only the original epoch0'):
            validate_selection(*fixture(2),'base',initial_parity=True)

    def test_threshold_must_match_cal_report(self):
        cp,selection,status=fixture();selection['report']['threshold']=.3
        with self.assertRaisesRegex(ValueError,'threshold not bound'):
            validate_selection(cp,selection,status,'base')

    def test_matcher_and_budget_completion_required(self):
        for name,value in [('matcher_unchanged',False),('updates',11000)]:
            cp,selection,status=fixture();status[name]=value
            with self.assertRaises(ValueError):validate_selection(cp,selection,status,'base')

    def test_cross_arm_and_base_mismatch_rejected(self):
        cp,selection,status=fixture();cp['arm']='frozen_features'
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            validate_selection(cp,selection,status,'base')
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            validate_selection(*fixture(),'otherbase')


if __name__=='__main__':unittest.main()
