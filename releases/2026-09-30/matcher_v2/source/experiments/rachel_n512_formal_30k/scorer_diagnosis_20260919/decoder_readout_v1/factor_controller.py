"""Four bounded CPU workers, isolated frozen controls; automatic retries=0."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path('/root/autodl-tmp/decoder_controls_20260929')
SOURCE=ROOT/'source_01'
AUDIT=Path('/root/autodl-tmp/scorer_internal_diagnosis_20260929')
MODELS=('threshold_scratch_fixed','binary_patch','binary_stats','aggressive_binary_patch')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def identity(pid):
    p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime=int(f[19]),cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
def verify_pilot(model):
    d=ROOT/'pilot_01'/model;done=read(d/'complete.json');protocol=read(d/'protocol.json')
    if (done['status']!='pilot_complete' or done['pairs']!=2 or protocol['script_sha256']!=sha(SOURCE/'run_factor_sweep.py')
        or protocol['decoder_sha256']!=sha(SOURCE/'decoder.py') or done['records_sha256']!=sha(d/'records.jsonl')
        or done['summary_sha256']!=sha(d/'summary.json') or not done['model_unchanged'] or (d/'failure.json').exists()):
        raise ValueError('CPU pilot not complete/source-identical:'+model)
def exact_optimization_proof():
    slow=AUDIT/'factor_pilot_01/binary_patch';fast=ROOT/'pilot_01/binary_patch'
    def stripped(r):
        if isinstance(r,dict):return {k:stripped(v) for k,v in r.items() if k!='seconds'}
        if isinstance(r,list):return [stripped(v) for v in r]
        return r
    a=[stripped(json.loads(x)) for x in (slow/'records.jsonl').read_text().splitlines()]
    b=[stripped(json.loads(x)) for x in (fast/'records.jsonl').read_text().splitlines()]
    if a!=b:raise ValueError('exact mass-bound seed optimization changed actual pilot predictions')
    save(ROOT/'exact_bound_proof.json',dict(status='passed',pairs=len(a),all_five_searches_bit_identical=True,
        old_records_sha256=sha(slow/'records.jsonl'),new_records_sha256=sha(fast/'records.jsonl')))
def run():
    if (ROOT/'controller_status.json').exists():raise ValueError('prior controller exists; do not repeat')
    binding={p.name:sha(p) for p in SOURCE.glob('*.py')};jobs=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        for phase in ('pilot_01','full_development_01'):
            jobs=[]
            for model in MODELS:
                out=ROOT/phase/model
                if phase=='pilot_01' and out.exists():verify_pilot(model);continue
                if out.exists():raise ValueError('preserve prior output; no automatic resume/retry')
                (ROOT/'logs').mkdir(exist_ok=True)
                args=[sys.executable,str(SOURCE/'run_factor_sweep.py'),'--audit-root',str(AUDIT),'--model',model,'--out',str(out)]
                if phase=='pilot_01':args+=['--pilot-limit','2']
                with (ROOT/'logs'/(phase+'_'+model+'.log')).open('xb') as log:
                    child=subprocess.Popen(args,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        start_new_session=True,preexec_fn=lambda:os.nice(10))
                time.sleep(.1);job=dict(model=model,out=str(out),phase=phase,identity=identity(child.pid),command=args)
                jobs.append((child,job));save(ROOT/(phase+'_'+model+'_launch.json'),job)
            save(ROOT/'controller_status.json',dict(status='running_'+phase,workers=[j for _,j in jobs],gpu_used=False,source_sha256=binding,automatic_retries=0))
            codes=[]
            for child,job in jobs:
                code=child.wait();codes.append(code);save(ROOT/(phase+'_'+job['model']+'_exit.json'),dict(returncode=code,job=job))
            if any(codes):raise ValueError('frozen worker failed; preserve outputs and diagnose')
            if {p.name:sha(p) for p in SOURCE.glob('*.py')}!=binding:raise ValueError('source changed while running')
            if phase=='pilot_01':
                for model in MODELS:verify_pilot(model)
                exact_optimization_proof()
        receipts={}
        for model in MODELS:
            d=ROOT/'full_development_01'/model;c=read(d/'complete.json')
            if (c['status']!='complete' or c['pairs']!=2619 or not c['model_unchanged']
                or c['records_sha256']!=sha(d/'records.jsonl') or c['summary_sha256']!=sha(d/'summary.json')):raise ValueError('incomplete full comparison')
            receipts[model]=dict(complete_sha256=sha(d/'complete.json'),**c)
        save(ROOT/'complete.json',dict(status='complete',models=receipts,pairs=10476,gpu_used=False,test_used=False,source_sha256=binding))
        save(ROOT/'controller_status.json',dict(status='complete',pairs=10476,gpu_used=False))
    except BaseException as e:
        save(ROOT/'controller_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),jobs=[j for _,j in jobs],automatic_retry=False));raise
def main():
    p=argparse.ArgumentParser();p.add_argument('--driver',action='store_true');a=p.parse_args()
    if a.driver:return run()
    if (ROOT/'controller_launch.json').exists():raise ValueError('controller already registered')
    with (ROOT/'controller.log').open('xb') as log:
        c=subprocess.Popen([sys.executable,__file__,'--driver'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                           start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    time.sleep(.15);r=identity(c.pid);save(ROOT/'controller_launch.json',r);print(json.dumps(r))
if __name__=='__main__':main()
