from copy import deepcopy
import os
import unittest
from .contracts import *
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.train import TrainingConfig,canonical_record

VARIANT=os.environ.get('BINARY_VERIFY_VARIANT','patch')
def report(key=(.7,.8,.9)):
    return dict(status='development_selection',key=list(key),test_used=False,real_used=True,gradients_used=False,
                development_evaluation=True,thresholds=dict(dunhuang_cv=.3,turufan=.4),elapsed_seconds=2.)
def fixture(choice='real'):
    config=canonical_record(TrainingConfig(scorer_variant=VARIANT).record())
    b=dict(arm='scratch_fixed',formal_training=True,preflight_steps=0,config=config)
    metrics=report() if choice=='real' else dict(key=[.8,.9],threshold=.3)
    cp=dict(binding=b,stage='scorer',epoch=2,metrics=metrics)
    cp.update({'thresholds':metrics['thresholds']} if choice=='real' else {'threshold':.3})
    s=dict(status='selected',binding=b,actual_epochs=16,updates=12000,exposures=384000,
        stop_reason=STOPS[0],selection_on_real=True,test_used=False,matcher_unchanged=True,migration_origin=None,
        best=dict(epoch=2,key=[.8,.9],threshold=.3),best_real=dict(epoch=2,key=report()['key'],thresholds=report()['thresholds']))
    c=dict(s,status='stage_complete');t=dict(status='training_complete',arm='scratch_fixed',stages=['scorer'],binding=b,last_stage=c)
    return cp,s,c,t,config
def curve():
    return [dict(epoch=e,updates=e*750,exposures=e*24000,key=[.8,.9] if e else [.1,.2],threshold=.3,
                 real_development=report() if e else report((1.,1.,1.))) for e in range(0,18,2)]

class ContractTests(unittest.TestCase):
    def test_terminal_both_selections(self):
        for choice in ('sim','real'):
            cp,s,c,t,config=fixture(choice)
            self.assertEqual(validate_terminal(cp,s,c,t,config,VARIANT,choice),cp['binding'])
    def test_reject_wrong_variant_joint_or_running(self):
        cp,s,c,t,config=fixture()
        for other in ('invalid','stats' if VARIANT=='patch' else 'patch'):
            with self.assertRaises(ValueError):validate_terminal(cp,s,c,t,config,other,'real')
        for target,key,value in [(t,'status','running'),(cp['binding'],'arm','scratch_joint'),(cp,'epoch',0)]:
            before=deepcopy(target);target[key]=value
            with self.assertRaises(ValueError):validate_terminal(cp,s,c,t,config,VARIANT,'real')
            target.clear();target.update(before)
    def test_terminal_rejects_changed_matcher_test_selection_or_budget(self):
        for key,value in [('matcher_unchanged',False),('test_used',True),('selection_on_real',False),
                           ('updates',0),('actual_epochs',50),('migration_origin',{'unapproved':True})]:
            cp,s,c,t,config=fixture();s[key]=value;c[key]=value
            with self.assertRaises(ValueError):validate_terminal(cp,s,c,t,config,VARIANT,'real')
    def test_complete_and_terminal_must_match(self):
        cp,s,c,t,config=fixture();c['unreported']=1
        with self.assertRaises(ValueError):validate_terminal(cp,s,c,t,config,VARIANT,'real')
    def test_curve_earlier_ties_and_no_real_epoch0(self):
        _,s,_,_,_=fixture();verify_selection_curve(curve(),s)
        bad=curve();bad[2]['real_development']['key'][0]=.71
        with self.assertRaises(ValueError):verify_selection_curve(bad,s)
        for bad in (curve()[:-1],curve()[::-1],curve()+[curve()[0]]):
            with self.assertRaises(ValueError):verify_selection_curve(bad,s)
    def test_curve_refuses_test_or_gradient_real_reports(self):
        _,s,_,_,_=fixture()
        for field in ('test_used','gradients_used'):
            bad=curve();bad[1]['real_development'][field]=True
            with self.assertRaises(ValueError):verify_selection_curve(bad,s)
    def test_threshold_protocol_and_timing_identity(self):
        for x in (.2,.3,.8):validate_threshold(x)
        for x in (.19,.305,.81,float('nan'),True):
            with self.assertRaises(ValueError):validate_threshold(x)
        self.assertEqual(metric_identity(dict(key=[1],elapsed_seconds=2,real_development={'elapsed_seconds':3,'key':[2]})),
                         dict(key=[1],real_development={'key':[2]}))
    def test_exact_real_role_population(self):
        plan={'datasets':{'dunhuang_cv':{'excluded_gt_pair_ids':['bad'],'roles':{
            'real_cal':{'pair_ids':['a']},'real_select':{'pair_ids':['b']},'real_test':{'pair_ids':['c']}}}}}
        rows=[dict(pair_id=i) for i in ('a','b','c','bad')]
        self.assertEqual(partition_real(rows,plan,'dunhuang_cv')['real_test'],[rows[2]])
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):partition_real(bad,plan,'dunhuang_cv')

