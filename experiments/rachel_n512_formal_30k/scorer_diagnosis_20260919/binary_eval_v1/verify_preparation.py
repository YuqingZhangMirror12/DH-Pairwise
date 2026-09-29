"""Fresh-process CPU synthetic evaluation tests; no real checkpoints or inputs."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from entry import bootstrap
from contracts import inventory,sha,read

def worker(a):
    os.environ['BINARY_VERIFY_VARIANT']=a.variant
    training,_=bootstrap(a.source,a.common_source)
    head=training.fresh_head(training.TrainingConfig().head_seed,a.variant)
    expected=34529 if a.variant=='patch' else 3201
    if sum(p.numel() for p in head.parameters())!=expected:raise ValueError('binary head architecture differs')
    names=['consensus_binary_eval_adapter.'+n for n in ('test_contracts','test_snapshot','test_evaluate','test_entry','test_loading')]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    record=dict(variant=a.variant,status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),head_parameters=expected,
        real_inference_performed=False,real_checkpoints_opened=False,gpu_inference_performed=False,
        actual_training_import=str(Path(training.__file__).resolve()))
    Path(a.out).write_text(json.dumps(record,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','common-source','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--real-plan');p.add_argument('--worker',action='store_true');p.add_argument('--variant',choices=('patch','stats'))
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only verification')
    if a.worker:return worker(a)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def bindings():return dict(adapter_python_sha256=inventory(Path(__file__).parent),common_python_sha256=inventory(a.common_source),
                               training_source_sha256=inventory(a.source,True))
    before=bindings();results=[]
    for variant in ('patch','stats'):
        dest=out/(variant+'_cpu.json')
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--variant',variant,'--source',a.source,
                 '--common-source',a.common_source,'--out',str(dest)]
        with (out/(variant+'_cpu.log')).open('xb') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy()).returncode
        record=read(dest) if dest.exists() else dict(variant=variant,status='failed',tests=0,errors=1,failures=0,skipped=0)
        results.append(dict(record,returncode=code))
    same=bindings()==before;passed=same and all(r['status']=='passed' and r['returncode']==0 for r in results)
    receipt=dict(schema='binary-evaluation-preparation/1',status='cpu_preparation_passed' if passed else 'failed',**before,
        source_files_unchanged=same,verified_variants=['patch','stats'] if passed else [],
        tests=sum(r['tests'] for r in results),errors=sum(r['errors'] for r in results),failures=sum(r['failures'] for r in results),
        skipped=sum(r['skipped'] for r in results),real_plan_sha256=sha(a.real_plan),
        fixed_case_plan_sha256=sha(Path(a.common_source)/'case_plan.json'),
        real_inference_performed=False,real_checkpoints_opened=False,gpu_preflight=False,formal_training_started=False,
        results=results,recorded_unix=time.time())
    (out/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','tests','errors','failures','skipped','verified_variants')}))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
