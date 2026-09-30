"""Finite four-GPU inference queue; no model training or external scheduling."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from common import read,save


def run(root):
    root=Path(root);registry=read(root/'registry.json');here=Path(__file__).resolve().parent
    jobs=queue.Queue();state={};mutex=threading.Lock()
    for key,reg in registry.items():
        if 'reuse_predictions' not in reg:
            done=root/'predictions'/key/'status.json'
            if done.exists() and read(done).get('status')=='complete':
                state[key]=dict(status='complete',reused=True)
            else:jobs.put(key);state[key]=dict(status='pending')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    gpus=[s.split(',')[1].strip() for s in raw.splitlines() if s.split(',')[0].strip() in ('0','1','2','3')]
    if len(gpus)!=4:raise ValueError('expected four assigned GPUs')
    def update(key,**kw):
        with mutex:state[key].update(kw);save(root/'inference_queue.json',state)
    def worker(gpu):
        while True:
            try:key=jobs.get_nowait()
            except queue.Empty:return
            reg=registry[key]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=reg['pythonpath'],
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            cmd=[sys.executable,str(here/'infer.py'),'--root',str(root),'--key',key,'--gpu',gpu]
            with (root/'logs'/(key+'.log')).open('a') as log:
                child=subprocess.Popen(cmd,cwd=reg['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
                update(key,status='running',gpu=gpu,pid=child.pid,started=time.time())
                code=child.wait()
            update(key,status='complete' if code==0 else 'failed',exit_code=code,finished=time.time())
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,gpus))
    code=subprocess.call([sys.executable,str(here/'calibrate.py'),'--root',str(root)])
    failed=[k for k,v in state.items() if v['status']!='complete']
    save(root/'driver_status.json',dict(status='complete' if not failed and code==0 else 'needs_attention',
        failed=failed,calibration_exit_code=code,training=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        root=Path(a.root)
        if (root/'driver_launch.json').exists():raise ValueError('already launched')
        with (root/'logs/driver.log').open('a') as log:
            child=subprocess.Popen([sys.executable,__file__,'--root',str(root)],stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
        save(root/'driver_launch.json',dict(pid=child.pid,started=time.time()))
        print(dict(status='launched',pid=child.pid))
    else:run(a.root)
