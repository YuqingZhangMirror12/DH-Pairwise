"""Launch an isolated CPU measurement; no training process/source changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--task', choices=['raw','compare'], required=True)
p.add_argument('--workers', type=int, default=16)
args=p.parse_args()
root=Path(__file__).resolve().parent
out=root/('raw_results' if args.task=='raw' else 'simple_results')
record=root/(args.task+'_launch.json')
if out.exists() or record.exists():
    raise SystemExit('existing run: inspect it, never automatically repeat')
env=os.environ.copy()
env.update(PYTHONPATH='/root/autodl-tmp/s7_consensus_layered_v14_mergefix_20260925/source',
    PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
command=[sys.executable,str(root/('raw_measure.py' if args.task=='raw' else 'compare.py')),
    '--phase1','/root/autodl-tmp/consensus_strictness_20260925/results_phase1','--out',str(out),'--workers',str(args.workers)]
with (root/(args.task+'.log')).open('xb') as f:
    proc=subprocess.Popen(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
path=Path('/proc')/str(proc.pid)
r=dict(pid=proc.pid,starttime=int(path.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]),
    cmdline=path.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(),command=command,cpu_workers=args.workers)
record.write_text(json.dumps(r,indent=2)); print(json.dumps(r))
