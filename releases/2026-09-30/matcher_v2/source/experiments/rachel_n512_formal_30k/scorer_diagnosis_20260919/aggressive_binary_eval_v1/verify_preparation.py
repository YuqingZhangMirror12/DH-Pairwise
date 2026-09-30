"""Source-bound synthetic CPU evaluation tests. No real inputs/checkpoints."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from entry import bootstrap,inventory,sha


def worker(a):
    os.environ['BINARY_VERIFY_VARIANT']='patch'
    training,_=bootstrap(a.source,a.common_source,a.binary_source)
    names=['consensus_aggressive_eval_adapter.'+n for n in
        ('test_contracts','test_loading','test_evaluate','test_population','test_entry')]
    # Same actual Patch MLP, Q/arc aggregation and per-layer numeric replay.
    names+=['consensus_binary_eval_adapter.test_snapshot']
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    r=dict(variant='patch',status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),
        real_inference_performed=False,real_checkpoints_opened=False,gpu_inference_performed=False,
        actual_training_import=str(Path(training.__file__).resolve()))
    Path(a.out).write_text(json.dumps(r,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','common-source','binary-source','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--real-plan');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only verification required')
    if a.worker:return worker(a)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def bindings():return dict(adapter_python_sha256=inventory(Path(__file__).parent),
        binary_python_sha256=inventory(a.binary_source),common_python_sha256=inventory(a.common_source),
        training_source_sha256=inventory(a.source,True))
    before=bindings();dest=out/'patch_cpu.json'
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--source',a.source,'--common-source',a.common_source,
        '--binary-source',a.binary_source,'--out',str(dest)]
    with (out/'patch_cpu.log').open('xb') as log:
        code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy()).returncode
    r=json.loads(dest.read_text()) if dest.exists() else dict(variant='patch',status='failed',tests=0,errors=1,failures=0,skipped=0)
    r['returncode']=code;same=before==bindings();passed=same and code==0 and r['status']=='passed'
    receipt=dict(schema='aggressive-binary-evaluation-preparation/1',status='cpu_preparation_passed' if passed else 'failed',
        **before,source_files_unchanged=same,verified_variants=['patch'] if passed else [],results=[r],
        tests=r['tests'],errors=r['errors'],failures=r['failures'],skipped=r['skipped'],
        real_plan_sha256=sha(a.real_plan),fixed_case_plan_sha256=sha(Path(a.common_source)/'case_plan.json'),
        real_inference_performed=False,real_checkpoints_opened=False,gpu_preflight=False,formal_training_started=False,
        recorded_unix=time.time())
    (out/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','tests','errors','failures','skipped')}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
