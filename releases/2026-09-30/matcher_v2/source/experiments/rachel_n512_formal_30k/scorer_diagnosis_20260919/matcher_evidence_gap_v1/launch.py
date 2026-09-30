"""One CPU-only diagnostic per output; no automatic retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent;name='full' if a.full else 'pilot'
    if (root/(name+'_launch.json')).exists() or (root/name).exists():raise SystemExit('Already registered: do not duplicate')
    if a.full:
        done=json.loads((root/'pilot/complete.json').read_text())
        assert done['status']=='complete' and done['pairs']==8 and not (root/'pilot/failure.json').exists()
    command=[sys.executable,str(root/'run.py'),'--phase1','/root/autodl-tmp/consensus_strictness_20260925/results_phase1',
        '--prior','/root/autodl-tmp/consensus_threshold_20260925/replay_results',
        '--helper','/root/autodl-tmp/consensus_threshold_20260925/replay.py','--out',str(root/name),'--workers','4' if a.full else '2']
    if not a.full:command+=['--limit','4']
    env=os.environ.copy();env.update(PYTHONPATH='/root/autodl-tmp/consensus_threshold_20260925/training_source_01',
        CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/(name+'.log')).open('xb') as log:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    proc=Path('/proc')/str(process.pid)
    record=dict(pid=process.pid,starttime=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
        command=command,started_unix=time.time(),gpu_used=False,training=False,
        hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')})
    (root/(name+'_launch.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
