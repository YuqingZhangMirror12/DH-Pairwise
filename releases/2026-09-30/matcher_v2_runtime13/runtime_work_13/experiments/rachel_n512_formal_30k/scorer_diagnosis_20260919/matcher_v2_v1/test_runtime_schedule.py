"""Actual CPU optimizer/resume tests; not evidence of CUDA/DDP admission."""
import copy
from dataclasses import replace
import random
import unittest

import numpy as np
import torch

from ..curriculum_training_v1 import test_exposure as fixtures
from ..curriculum_training_v1.exposure import STAGES,build_ledger,digest,learning_rate_at
from ..curriculum_training_v1.runtime_plan import lock_record
from ..curriculum_training_v1.test_validation_adapter import setup_plan
from ..curriculum_training_v1.test_training_core import Wrapper
from ..curriculum_training_v1.training_core import Topology,run_updates
from ..curriculum_training_v1.checkpoint_io import tree_sha
from .additive_exposure import build_additive_ledger
from .test_additive_exposure import extras
from .runtime_schedule import compile_plan


def fixture(arm='B3'):
    ledger,plan=setup_plan()
    if arm=='B2':return ledger,plan,compile_plan(plan,ledger,arm)
    added=build_additive_ledger(ledger,extras(),((0,3,1),(3,7,3),(7,12,2),(12,16,2)),seed=ledger.seed)
    return ledger,plan,compile_plan(plan,ledger,arm,additive=added,admitted_data_sha256=digest('combined fixture'))


def cpu_train(resume=None,stop=None):
    torch.set_num_threads(1);torch.manual_seed(731);random.seed(811);np.random.seed(231)
    original,_,(ledger,plan,metadata)=fixture()
    model=Wrapper();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    dataset=[(torch.tensor([i/30.,(i%5)/5.,1.]),torch.tensor(float(row.label))) for i,row in enumerate(ledger.catalog)]
    seen=[]
    def evaluate(update):
        random.random();np.random.random();torch.rand(5)
        return {'original_updates':ledger.base_completed(update)}
    def capture(row):seen.append(dict(row,**{k:v for k,v in ledger.counts(row['completed_updates']).items() if k!='completed_updates'}))
    state=run_updates(model,optimizer,dataset,ledger,'curriculum',Topology(0,1,2,2),'cpu',
        plan.learning_rate_knots,plan.validation_updates,evaluate=evaluate,resume=resume,
        binding=dict(schedule=metadata,explicit_cpu_fixture=True),on_update=capture,stop_after=stop)
    return copy.deepcopy(state),seen


class ScheduleTests(unittest.TestCase):
    def test_B2_identity_no_added_updates(self):
        base,original,(actual,plan,meta)=fixture('B2')
        self.assertEqual(plan,original)
        self.assertEqual(actual.sequence('curriculum'),base.sequence('curriculum'))
        self.assertEqual(meta['added_straight_updates'],0)

    def test_B1_B3_same_ledger_schedule_other_than_arm(self):
        a=fixture('B1')[2];b=fixture('B3')[2]
        self.assertEqual(a[:2],b[:2])
        self.assertEqual({k:v for k,v in a[2].items() if k!='arm'},{k:v for k,v in b[2].items() if k!='arm'})

    def test_each_actual_update_obeys_original_lr_clock(self):
        _,original,(ledger,plan,meta)=fixture()
        for i in range(ledger.total_updates):
            self.assertEqual(learning_rate_at(i,plan.learning_rate_knots),
                             learning_rate_at(ledger.base_completed(i),original.learning_rate_knots))
        self.assertEqual([ledger.base_completed(i) for i in plan.validation_updates],list(original.validation_updates))
        self.assertEqual(sum(ledger.stage_updates),ledger.total_updates)

    def test_full_registered_24k_plan_compiles_to31667(self):
        base=build_ledger(fixtures.samples((2,2,2)),dict(zip(STAGES,(15000,6000,3000))),4,effective_batch=4)
        _,template=setup_plan()
        record=copy.deepcopy(template.record)
        record.update(ledger_sha256=base.sha256,total_updates=24000,stage_updates=dict(zip(STAGES,base.stage_updates)),
            seed=4,validation_updates=list(range(0,24001,1500)),learning_rate_knots=[[0,1e-4],[15000,5e-5],[21000,2.5e-5]])
        original=lock_record(record,base)
        added=build_additive_ledger(base,extras(),((0,1500,167),(1500,15000,4500),(15000,21000,2000),(21000,24000,1000)),seed=4)
        runtime,plan,meta=compile_plan(original,base,'B3',additive=added,admitted_data_sha256=digest('fixture'))
        self.assertEqual(runtime.total_updates,31667)
        self.assertEqual(runtime.stage_updates,(19667,8000,4000))
        self.assertEqual(plan.record['learning_rate_knots'],[[0,1e-4],[19667,5e-5],[27667,2.5e-5]])
        self.assertEqual(runtime.base_completed(31667),24000)

    def test_reject_different_seed_mutated_sequences_or_missing_admission(self):
        base,plan=setup_plan();added=build_additive_ledger(base,extras(),((0,16,8),),seed=base.seed)
        for wrong in (replace(added,seed=9),replace(added,curriculum=added.curriculum[::-1]),replace(added,sha256='a'*64)):
            with self.assertRaises(ValueError):compile_plan(plan,base,'B1',additive=wrong,admitted_data_sha256=digest('fixture'))
        with self.assertRaises(ValueError):compile_plan(plan,base,'B1',additive=added)
        with self.assertRaises(ValueError):compile_plan(plan,base,'B2',additive=added)
        with self.assertRaises(ValueError):compile_plan(plan,base,'B4')

    def test_actual_AdamW_resume_across_insertions_and_original_stage_ends(self):
        full,trace=cpu_train()
        for at in (1,4,10,11,12,17,18,22):
            partial,_=cpu_train(stop=at)
            resumed,tail=cpu_train(resume=partial)
            self.assertEqual(tree_sha(full),tree_sha(resumed),msg=f'CPU resume differs at actual update {at}')
            self.assertEqual(tail,trace[at:])
        self.assertEqual(trace[-1]['original_updates'],16)
        self.assertEqual(trace[-1]['added_straight_updates'],8)


if __name__=='__main__':unittest.main()
