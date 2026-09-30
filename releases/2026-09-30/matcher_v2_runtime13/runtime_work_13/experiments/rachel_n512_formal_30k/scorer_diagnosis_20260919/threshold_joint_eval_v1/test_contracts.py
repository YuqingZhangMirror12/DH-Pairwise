from copy import deepcopy
import unittest

import torch

from .contracts import (validate_budget, validate_threshold, choose_real_best, validate_archive,
                        matcher_change, validate_joint_terminal, partition_real, metric_identity)


def report(key=(.7, .8, .9)):
    return dict(status='development_selection', key=list(key), test_used=False, real_used=True,
        gradients_used=False, development_evaluation=True,
        thresholds=dict(dunhuang_cv=.3, turufan=.4), elapsed_seconds=2.)


def terminal_fixture(choice='real'):
    config={'fixture': True}
    binding=dict(arm='scratch_joint', formal_training=True, preflight_steps=0, config=config)
    metrics=report() if choice=='real' else dict(key=[.8,.9], threshold=.3)
    cp=dict(binding=binding, stage='scorer', epoch=2, metrics=metrics)
    if choice=='real':cp['thresholds']=metrics['thresholds']
    else:cp['threshold']=.3
    selection=dict(status='selected', binding=binding, actual_epochs=16, updates=12000, exposures=384000,
        stop_reason='simulation_plateau_after_lr_reductions', selection_on_real=True, test_used=False,
        matcher_unchanged=False, development_evaluation=True,
        matcher_change=dict(changed_tensors=3,unused_changed=[]),
        best=dict(epoch=2,key=[.8,.9],threshold=.3),
        best_real=dict(epoch=2,key=report()['key'],thresholds=report()['thresholds']))
    complete=dict(selection,status='stage_complete')
    terminal=dict(status='training_complete',arm='scratch_joint',stages=['scorer'],binding=binding,last_stage=complete)
    return cp,selection,complete,terminal,config


class ContractTests(unittest.TestCase):
    def test_registered_budget_and_threshold_only(self):
        _,s,_,_,_=terminal_fixture();self.assertEqual(validate_budget(s),16)
        for k,v in [('actual_epochs',15),('updates',0),('exposures',0),('stop_reason','failure')]:
            with self.assertRaises(ValueError):validate_budget(dict(s,**{k:v}))
        for value in (.2,.3,.8):validate_threshold(value)
        for value in (.19,.81,.305,float('nan'),True):
            with self.assertRaises(ValueError):validate_threshold(value)

    def test_real_winner_excludes_epoch0_and_exact_tie_keeps_earlier(self):
        rows=[dict(epoch=4,real_report=report()),dict(epoch=0,real_report=report((1.,1.,1.))),dict(epoch=2,real_report=report())]
        self.assertEqual(choose_real_best(rows)['epoch'],2)
        rows[0]['real_report']['key'][0]=.71
        self.assertEqual(choose_real_best(rows)['epoch'],4)

    def test_real_selection_cannot_read_test_or_gradients(self):
        for key in ('test_used','gradients_used'):
            r=report();r[key]=True
            with self.assertRaises(ValueError):choose_real_best([dict(epoch=2,real_report=r)])
        for rows in ([],[dict(epoch=0,real_report=report())],
                     [dict(epoch=2,real_report=report())]*2,[dict(epoch=1,real_report=report())]):
            with self.assertRaises(ValueError):choose_real_best(rows)

    def test_archive_ignores_only_two_registered_timing_fields(self):
        metrics=dict(key=[.7],threshold=.3,elapsed_seconds=1.,real_development=report())
        cp=dict(binding={'a':1},stage='scorer',epoch=2,updates=1500,exposures=48000,threshold=.3,metrics=metrics)
        row=dict(deepcopy(metrics),epoch=2,updates=1500,exposures=48000)
        row['elapsed_seconds']=3.;row['real_development']['elapsed_seconds']=4.
        validate_archive(cp,row,cp['binding'],2)
        row['real_development']['key'][0]=.9
        with self.assertRaises(ValueError):validate_archive(cp,row,cp['binding'],2)
        self.assertIn('elapsed_seconds',metrics['real_development'])
        self.assertEqual(metric_identity({'other':{'elapsed_seconds':1}}),{'other':{'elapsed_seconds':1}})

    def test_matcher_joint_vs_frozen_distinguished(self):
        a={'matcher.active':torch.tensor([1.]),'matcher.base.coarse.w':torch.tensor([2.]),'head.x':torch.tensor([3.])}
        b=deepcopy(a);b['matcher.active']+=.01
        self.assertEqual(matcher_change(a,b,expect_updated=True)['changed_tensors'],1)
        with self.assertRaises(ValueError):matcher_change(a,b,expect_updated=False)
        with self.assertRaises(ValueError):matcher_change(a,a,expect_updated=True)
        matcher_change(a,a,expect_updated=False)
        b['matcher.base.coarse.w']+=1
        with self.assertRaises(ValueError):matcher_change(a,b,expect_updated=True)

    def test_matcher_nonfinite_or_missing_fails(self):
        a={'matcher.x':torch.tensor([1.])}
        for b in ({},{'matcher.x':torch.tensor([float('nan')])},{'matcher.x':torch.tensor([1.,2.])}):
            with self.assertRaises(ValueError):matcher_change(a,b,expect_updated=None)

    def test_joint_terminal_checks_both_selection_kinds(self):
        for kind in ('sim','real'):
            cp,s,c,t,config=terminal_fixture(kind)
            self.assertEqual(validate_joint_terminal(cp,s,c,t,config,kind),cp['binding'])
            cp['epoch']=4
            with self.assertRaises(ValueError):validate_joint_terminal(cp,s,c,t,config,kind)

    def test_joint_cannot_masquerade_as_frozen_or_failed(self):
        for field,value in [('matcher_unchanged',True),('test_used',True),('selection_on_real',False),('actual_epochs',0)]:
            cp,s,c,t,config=terminal_fixture()
            s[field]=value;c[field]=value
            with self.assertRaises(ValueError):validate_joint_terminal(cp,s,c,t,config,'real')
        cp,s,c,t,config=terminal_fixture();t['status']='running'
        with self.assertRaises(ValueError):validate_joint_terminal(cp,s,c,t,config,'real')

    def test_partition_exact_ids_and_explicit_context(self):
        plan={'datasets':{'dunhuang_cv':{'excluded_gt_pair_ids':['bad'],'roles':{
            'real_cal':{'pair_ids':['a']},'real_select':{'pair_ids':['b']},'real_test':{'pair_ids':['c']}}}}}
        rows=[dict(pair_id=i) for i in ('a','b','c','bad')]
        groups=partition_real(rows,plan,'dunhuang_cv')
        self.assertEqual(groups['real_test'],[rows[2]])
        self.assertEqual(len(groups['gt_corrected_800_development_context']),3)
        for invalid in (rows[:-1],rows+[rows[0]],rows+[dict(pair_id='extra')]):
            with self.assertRaises(ValueError):partition_real(invalid,plan,'dunhuang_cv')


if __name__=='__main__':unittest.main()
