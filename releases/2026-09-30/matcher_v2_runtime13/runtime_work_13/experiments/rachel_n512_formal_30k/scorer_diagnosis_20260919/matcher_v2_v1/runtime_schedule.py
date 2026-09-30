"""Compile original-clock scheduling into an unchanged actual-update trainer.

The existing FP32/AdamW/checkpoint loop can consume this ledger. Its actual
update cursor stays authoritative; LR knots and observations are translated
once, then locked. No monkey patch or change to running legacy code is needed.
"""
from bisect import bisect_left
from dataclasses import dataclass
import json

from ..curriculum_training_v1.exposure import STAGES, digest, integer, is_sha, learning_rate_at
from ..curriculum_training_v1.runtime_plan import lock_record
from .additive_exposure import AdditiveLedger, build_additive_ledger


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class RuntimeLedger:
    ledger: object
    stage_updates: tuple
    original_clock: tuple

    @property
    def catalog(self):return self.ledger.catalog
    @property
    def effective_batch(self):return self.ledger.effective_batch
    @property
    def seed(self):return self.ledger.seed
    @property
    def sha256(self):return self.ledger.sha256
    @property
    def total_updates(self):return self.ledger.total_updates

    def sequence(self,order):return self.ledger.sequence(order)

    def base_completed(self,completed):
        integer(completed,'completed update')
        require(completed<=self.total_updates,'completed cursor exceeds budget')
        return self.original_clock[completed]

    def counts(self,completed):
        original=self.base_completed(completed)
        return dict(completed_updates=completed,original_updates=original,
                    added_straight_updates=completed-original,
                    original_exposures=original*self.effective_batch,
                    added_straight_exposures=(completed-original)*self.effective_batch)


def compile_plan(base_plan, base_ledger, arm, *, additive=None, admitted_data_sha256=None):
    require(arm in ('B1','B2','B3'),'only registered B1/B2/B3 arms may compile')
    require(lock_record(base_plan.record,base_ledger).sha256==base_plan.sha256,
            'original locked plan/ledger changed')
    if arm in ('B1','B3'):
        require(isinstance(additive,AdditiveLedger),'data-augmented arm needs its complete added ledger')
        require(additive.base_sha256==base_ledger.sha256 and additive.base_catalog_size==len(base_ledger.catalog),
                'additive ledger belongs to a different original plan')
        require(additive.seed==base_ledger.seed,'keep original training RNG seed; do not alter it with the extra stream')
        require(additive.catalog[:len(base_ledger.catalog)]==base_ledger.catalog,'original catalog changed')
        rebuilt=build_additive_ledger(base_ledger,additive.catalog[len(base_ledger.catalog):],additive.windows,seed=additive.seed)
        require(rebuilt==additive,'additive ledger was mutated or not reproducibly derived')
        require(is_sha(admitted_data_sha256),'combined dataset admission binding required')
        actual=additive
        original_clock=tuple(additive.base_completed(i) for i in range(additive.total_updates+1))
    else:
        require(additive is None and admitted_data_sha256 is None,'B2 must keep the original data and budget')
        actual=base_ledger
        original_clock=tuple(range(base_ledger.total_updates+1))
    require(original_clock[0]==0 and original_clock[-1]==base_ledger.total_updates,
            'original update endpoints differ')
    require(all(b-a in (0,1) for a,b in zip(original_clock,original_clock[1:])),
            'original update clock skipped or went backward')

    def actual_cursor(original):
        at=bisect_left(original_clock,original)
        require(at<len(original_clock) and original_clock[at]==original,'unrepresentable original cursor')
        return at

    stage_ends=[];completed=0
    for count in base_ledger.stage_updates:
        completed+=count;stage_ends.append(actual_cursor(completed))
    actual_stages=tuple(b-a for a,b in zip([0]+stage_ends,stage_ends))
    runtime_ledger=RuntimeLedger(actual,actual_stages,original_clock)
    record=json.loads(json.dumps(base_plan.record))
    knots=[[actual_cursor(at),value] for at,value in base_plan.learning_rate_knots]
    marks=[actual_cursor(at) for at in base_plan.validation_updates]
    record.update(ledger_sha256=actual.sha256,total_updates=actual.total_updates,
                  stage_updates=dict(zip(STAGES,actual_stages)),learning_rate_knots=knots,validation_updates=marks)
    if arm in ('B1','B3'):
        record['data_admission_sha256']=admitted_data_sha256
    plan=lock_record(record,runtime_ledger)
    # The existing training core looks up LR by the actual optimizer cursor.
    # Prove translated knots equal the intended original-clock LR at EVERY step.
    for step in range(actual.total_updates):
        require(learning_rate_at(step,knots)==learning_rate_at(original_clock[step],base_plan.learning_rate_knots),
                'compiled LR diverges from the original-data clock')
    metadata=dict(schema='matcher-v2-runtime-schedule/1',arm=arm,module=plan.record['module'],
        base_plan_sha256=base_plan.sha256,base_ledger_sha256=base_ledger.sha256,
        runtime_plan_sha256=plan.sha256,runtime_ledger_sha256=actual.sha256,
        original_updates=base_ledger.total_updates,actual_updates=actual.total_updates,
        added_straight_updates=actual.total_updates-base_ledger.total_updates,
        original_stage_updates=list(base_ledger.stage_updates),actual_stage_updates=list(actual_stages),
        original_learning_rate_knots=[list(v) for v in base_plan.learning_rate_knots],
        actual_learning_rate_knots=knots,original_validation_updates=list(base_plan.validation_updates),
        actual_validation_updates=marks,original_clock_sha256=digest(original_clock),
        lr_clock='completed original updates',checkpoint_clock='all completed optimizer updates',
        source_geometry_unchanged=True,original_exposures_and_order_preserved=True,
        gpu_started=False)
    return runtime_ledger,plan,metadata
