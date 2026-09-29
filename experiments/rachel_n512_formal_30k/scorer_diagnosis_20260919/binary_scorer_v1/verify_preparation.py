"""CPU-only explicit test scope and complete prepared-source fingerprint."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import unittest


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();source=Path(a.source).resolve();out=Path(a.out)
    sys.path.insert(0,str(source));base='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'
    tests=[base+'binary_scorer_v1.test_binary']+[base+'s7_consensus_v1.'+x for x in
        ('test_threshold','test_frozen_start','test_validation_protocol')]
    tests += [base+'s7_consensus_v1.test_evaluation_checkpoint.EvaluationCheckpointTests.'+x for x in
        ('test_immutable_copy_and_identical_replay','test_timing_only_change_keeps_original_bytes',
         'test_different_weights_or_selection_cannot_overwrite','test_nested_development_timing_replay_preserves_archive',
         'test_nested_development_metric_change_is_not_timing')]
    start=time.time();suite=unittest.TestLoader().loadTestsFromNames(tests)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    module=importlib.import_module(base+'binary_scorer_v1.head')
    counts={v:sum(p.numel() for p in module.BinaryClusterHead(v).parameters()) for v in ('patch','stats')}
    record=dict(status='cpu_tests_passed' if result.wasSuccessful() else 'cpu_tests_failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),scope=tests,
        excluded_old_mergefix_fixture='test_repair_policy_is_part_of_training_binding: removed policy not part of threshold source',
        seconds=time.time()-start,head_parameters=counts,gpu_preflight=False,formal_training_started=False,
        source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')})
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('preserve previous verification receipt')
    out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='source_sha256'}))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
