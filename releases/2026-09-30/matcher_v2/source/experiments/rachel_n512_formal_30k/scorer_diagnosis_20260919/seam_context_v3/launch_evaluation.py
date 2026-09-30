"""Launch one frozen evaluation per free GPU; never launches training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .prepare import read,save


def main(a):
    root=Path(a.root);out=Path(a.out);devices=[int(x) for x in a.gpus.split(',')]
    if len(devices)!=4 or len(set(devices))!=4:raise ValueError('four distinct GPUs required')
    if read(root/'training/status.json')['status']!='training_complete':raise ValueError('training is not complete')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('GPU compute processes exist; do not interfere')
    out.mkdir(parents=True,exist_ok=False)
    jobs=[];handles=[]
    module=__package__+'.evaluate_external'
    for split,gpu in zip(('dunhuang','dunhuang_cv','turufan','sim_test'),devices):
        cmd=[sys.executable,'-u','-m',module,'--checkpoint',str(root/'training/best_joint.pt'),
             '--selection',str(root/'training/B_selection.json'),'--split',split,'--out',str(out/split),'--microbatch','8']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        log=(out/(split+'.log')).open('x')
        proc=subprocess.Popen(cmd,cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();handles.append(proc)
        jobs.append(dict(split=split,gpu=gpu,pid=proc.pid,command=cmd))
    save(out/'launch.json',dict(pid=os.getpid(),jobs=jobs,started=time.time(),training=False))
    while True:
        states=[dict(job,returncode=proc.poll()) for job,proc in zip(jobs,handles)]
        done=all(x['returncode'] is not None for x in states)
        save(out/'driver_status.json',dict(status=('complete' if all(x['returncode']==0 for x in states) else 'failed') if done else 'running',jobs=states,training=False,updated=time.time()))
        if done:break
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--gpus',default='0,1,2,3')
    main(p.parse_args())
