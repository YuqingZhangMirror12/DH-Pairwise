"""CPU-only queue tests and actual synthetic MLP tracing, both head variants."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from queue_contracts import read,save,sha,hashes

KEYS=('adapter_python_sha256','binary_python_sha256','common_python_sha256','training_source_sha256','real_plan_sha256','fixed_case_plan_sha256')
def worker(args):
    os.environ['BINARY_VERIFY_VARIANT']=args.variant
    sys.path.insert(0,str(Path(args.adapter).resolve()))
    from entry import bootstrap
    bootstrap(args.source,args.common_source,args.binary_source)
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in ('test_queue','test_driver','test_worker'))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    record=dict(variant=args.variant,status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),
        real_inference_performed=False,gpu_tasks_started=False,synthetic_production_size_head_traced_on_cpu=True)
    save(args.out,record)
    if not result.wasSuccessful():raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','common-source','binary-source','adapter','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--evaluation-preparation');p.add_argument('--worker',action='store_true');p.add_argument('--variant',choices=('patch',))
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only preparation')
    if args.worker:return worker(args)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);receipt=read(args.evaluation_preparation)
    if receipt.get('status')!='cpu_preparation_passed' or receipt.get('verified_variants')!=['patch']:
        raise ValueError('completed binary evaluation CPU preparation required')
    def actual():return dict(adapter_python_sha256=hashes(args.adapter),common_python_sha256=hashes(args.common_source),binary_python_sha256=hashes(args.binary_source),
        training_source_sha256={str(p.relative_to(Path(args.source))):sha(p) for p in sorted(Path(args.source).rglob('*.py'))})
    bindings={k:receipt[k] for k in KEYS}
    if any(bindings[k]!=v for k,v in actual().items()):raise ValueError('prepared evaluator/source changed')
    before=hashes(Path(__file__).parent);results=[]
    for variant in ('patch',):
        dest=out/(variant+'_cpu.json')
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--variant',variant,
            '--source',args.source,'--common-source',args.common_source,'--binary-source',args.binary_source,'--adapter',args.adapter,'--out',str(dest)]
        with (out/(variant+'_cpu.log')).open('xb') as stream:
            code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy()).returncode
        result=read(dest) if dest.exists() else dict(variant=variant,status='failed',tests=0,errors=1,failures=0,skipped=0)
        results.append(dict(result,returncode=code))
    same=before==hashes(Path(__file__).parent) and all(bindings[k]==v for k,v in actual().items())
    passed=same and all(r['status']=='passed' and r['returncode']==0 for r in results)
    result=dict(schema='aggressive-binary-evaluation-queue-preparation/1',status='passed' if passed else 'failed',
        verified_variants=['patch'] if passed else [],queue_python_sha256=before,source_bindings=bindings,
        source_files_unchanged=same,tests=sum(r['tests'] for r in results),errors=sum(r['errors'] for r in results),
        failures=sum(r['failures'] for r in results),skipped=sum(r['skipped'] for r in results),results=results,
        gpu_tasks_started=False,real_inference_performed=False,automatic_retries=0,time_unix=time.time())
    save(out/'preparation.json',result);print(json.dumps({k:result[k] for k in ('status','tests','errors','failures','skipped')}))
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()

