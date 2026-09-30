"""Start the four required real evaluations only after both arms finish."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .prepare import read,save
from .evaluate_independent import ARMS,load_winner


def run(a):
    root,out=Path(a.root),Path(a.out);devices=[int(x) for x in a.gpus.split(',')]
    if len(devices)!=4 or len(set(devices))!=4:raise ValueError('four distinct evaluation GPUs required')
    check=read(root/'evaluation_adapter_parity.json')
    if not check.get('passed') or not check.get('initial_epoch_only'):raise ValueError('initial adapter parity required')
    for arm in ARMS:
        model,_,_=load_winner(root/arm,a.base);del model
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    gpu_ids={int(line.split(',')[0]):line.split(',')[1].strip() for line in gpu_rows}
    if any(i not in gpu_ids for i in devices):raise ValueError('requested GPU unavailable')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
    wanted={gpu_ids[i] for i in devices}
    if any(line.split(',')[0].strip() in wanted for line in active):raise RuntimeError('evaluation GPUs busy; do not interfere')
    out.mkdir(parents=True,exist_ok=False);jobs=[];processes=[]
    plan=[(arm,split) for arm in ARMS for split in ('dunhuang_cv','turufan')]
    for (arm,split),gpu in zip(plan,devices):
        dest=out/arm/split;dest.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,'-u','-m',__package__+'.evaluate_independent','--run',str(root/arm),'--base',a.base,
            '--split',split,'--out',str(dest),'--microbatch','8']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/f'{arm}_{split}.log').open('x') as log:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(p);jobs.append(dict(arm=arm,split=split,gpu=gpu,pid=p.pid,command=cmd))
    save(out/'launch.json',dict(pid=os.getpid(),jobs=jobs,started=time.time(),training=False))
    while True:
        states=[dict(job,returncode=p.poll()) for job,p in zip(jobs,processes)]
        done=all(x['returncode'] is not None for x in states)
        save(out/'driver_status.json',dict(status=('complete' if all(x['returncode']==0 for x in states) else 'failed') if done else 'running',jobs=states,updated=time.time()))
        if done:break
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','out','base'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpus',default='0,1,2,3');run(p.parse_args())
