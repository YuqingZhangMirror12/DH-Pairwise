from copy import deepcopy
import unittest
from consensus_binary_eval_adapter.test_contracts import fixture as binary_fixture, curve, report
from .contracts import validate_terminal, verify_selection_curve


def fixture(choice='real'):
    cp,s,c,t,config=binary_fixture(choice)
    cp['binding'].update(arm='scratch_aggressive',stage='scorer',fixed_matcher={'synthetic':'new random run'})
    t.update(arm='scratch_aggressive',stages=['matcher','scorer'],selected_matcher=cp['binding']['fixed_matcher'])
    return cp,s,c,t,config


class ContractsTests(unittest.TestCase):
    def test_both_selected_checkpoints_have_exact_two_stage_identity(self):
        for choice in ('sim','real'):
            cp,s,c,t,config=fixture(choice)
            self.assertEqual(validate_terminal(cp,s,c,t,config,choice),cp['binding'])
            verify_selection_curve(curve(),s)

    def test_v14_e32_only_or_joint_checkpoint_cannot_be_substituted(self):
        for case in ('arm','stage','stages','origin','schema'):
            cp,s,c,t,config=fixture()
            if case=='arm':cp['binding']['arm']='scratch_fixed'
            elif case=='stage':cp['stage']='matcher'
            elif case=='stages':t['stages']=['scorer']
            elif case=='origin':t['selected_matcher']={'synthetic':'E32'}
            else:config['schema']='binary-cluster-scorer/1'
            with self.assertRaises(ValueError):validate_terminal(cp,s,c,t,config,'real')

    def test_updated_matcher_bad_budget_test_or_migration_rejected(self):
        for key,value in [('matcher_unchanged',False),('selection_on_real',False),('test_used',True),
                          ('actual_epochs',50),('updates',0),('migration_origin',{'old':True})]:
            cp,s,c,t,config=fixture();s[key]=value;c[key]=value
            with self.assertRaises(ValueError):validate_terminal(cp,s,c,t,config,'real')

    def test_stale_completion_real_epochzero_and_changed_threshold(self):
        for case in ('status','epoch','threshold'):
            cp,s,c,t,config=fixture()
            if case=='status':t['status']='running'
            elif case=='epoch':cp['epoch']=0
            else:cp['thresholds']=dict(dunhuang_cv=.19,turufan=.4)
            with self.assertRaises(ValueError):validate_terminal(cp,s,c,t,config,'real')


if __name__=='__main__':unittest.main()
