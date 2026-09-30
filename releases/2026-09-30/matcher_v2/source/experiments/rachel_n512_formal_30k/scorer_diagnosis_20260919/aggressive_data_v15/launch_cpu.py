"""One bounded CPU data job, exact launch identity, zero retries, no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--probe',type=int,default=0)
    p.add_argument('--reuse-committed-groups')
    p.add_argument('--supplement-previous')
    a=p.parse_args();source=Path(a.source).resolve();out=Path(a.out).resolve()
    launch=out.with_name(out.name+'_launch.json')
    if out.exists() or launch.exists():raise ValueError('output/launch exists; no duplicate job or automatic restart')
    package='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_data_v15'
    command=[sys.executable,'-u','-m',package+'.run','--baseline',a.baseline,'--out',str(out),'--workers','4']
    if a.probe:command+=['--pilot-groups',str(a.probe)]
    if a.reuse_committed_groups:command+=['--reuse-committed-groups',a.reuse_committed_groups]
    if a.supplement_previous:
        if a.probe or a.reuse_committed_groups:raise ValueError('one explicit mode')
        command[3]=package+'.supplement'
        command+=['--previous',a.supplement_previous]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(source))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_name(out.name+'.log').open('x') as log:
        child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,starttime=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
        command=command,cwd=str(source),time_unix=time.time(),cpu_workers=4,gpu_used=False,
        retry_count=0,training_started=False,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    launch.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

if __name__=='__main__':main()
