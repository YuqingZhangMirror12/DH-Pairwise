"""Finite two-GPU job. Preparation -> same-budget training -> final evaluation."""
import argparse
import os
import subprocess
import sys
import time
from support import *
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.local_evidence_v2.runtime import lease

def work(root,shard):
    root=Path(root);arm=ARMS[shard];code=Path(__file__).parent;out=root/f'worker_{shard}.json'
    (root/'logs').mkdir(exist_ok=True)
    def stage(name,args):
        cmd=[sys.executable,str(code/(name+'.py')),*args]
        log=root/'logs'/f'{arm}_{name}.log'
        with log.open('a') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            save(out,dict(status='running',arm=arm,stage=name,pid=child.pid,gpu=GPUS[shard],log=str(log),command=cmd))
            status=child.wait()
        if status:raise RuntimeError(f'{name} exited {status}; see {log}')
    stage('precompute',['--root',str(root),'--shard',str(shard),'--batch-size','8'])
    start=time.time()
    while True:
        other=root/'cache'/f'shard_{1-shard}_status.json'
        peer=root/f'worker_{1-shard}.json'
        if other.exists() and read(other).get('status')=='complete':break
        if peer.exists() and read(peer).get('status')=='failed':raise RuntimeError('peer precompute failed')
        if time.time()-start>6*3600:raise TimeoutError('peer preparation timeout')
        save(out,dict(status='waiting_for_shared_cache',arm=arm,gpu=GPUS[shard]));time.sleep(15)
    stage('train',['--root',str(root),'--arm',arm]+(['--resume'] if (root/'training'/arm/'last.pt').exists() else []))
    stage('evaluate',['--root',str(root),'--arm',arm])
    save(out,dict(status='complete',arm=arm,stage='train_val_test_real_ood_complete',gpu=GPUS[shard]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--shard',type=int,choices=(0,1),required=True);a=p.parse_args()
    try:
        with lease(GPUS[a.shard],Path(a.root)/'gpu_locks'):work(a.root,a.shard)
    except BaseException as e:
        save(Path(a.root)/f'worker_{a.shard}.json',dict(status='failed',error=repr(e),arm=ARMS[a.shard]));raise
