"""CPU-only admission, stage bridge, actual network gradients and engine tests."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import unittest


def inventory(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();source=Path(a.source).resolve();out=Path(a.out)
    if out.exists():raise ValueError('preserve previous preparation receipt')
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(source))
    base='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'
    before=inventory(source);start=time.time()
    tests=[base+'aggressive_binary_v1.'+s for s in ('test_admission','test_runtime','test_engine','test_launch_training')]
    tests += [base+'binary_scorer_v1.test_binary',base+'s7_consensus_v1.test_training_contracts',
              base+'s7_consensus_v1.test_matcher',base+'s7_consensus_v1.test_threshold']
    suite=unittest.TestLoader().loadTestsFromNames(tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    after=inventory(source)
    module=importlib.import_module(base+'aggressive_binary_v1.runtime')
    passed=result.wasSuccessful() and before==after and not result.skipped
    receipt=dict(schema='aggressive-binary-cpu-preparation/1',status='passed' if passed else 'failed',
        tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),
        scope=tests,source_sha256=after,source_unchanged=before==after,seconds=time.time()-start,
        head_parameters=sum(p.numel() for p in module.fresh_patch_head(26092406).parameters()),
        full_generation_authorized=False,formal_training_started=False,gpu_preflight=False,
        engine_test_scope='real CPU optimizer steps; mock data/validation/distribution/CUDA; not DDP proof',
        actual_network_scope='96D Matcher backward and Patch Scorer backward; stage freeze/import checked',
        human_approval_scope='synthetic receipts only; no real approval file created')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('scope','source_sha256')}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
