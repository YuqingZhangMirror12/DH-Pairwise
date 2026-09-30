"""Run the two authorized frozen evaluations only after S7-H has completed."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import torch
from .evaluate import check_selection
from ..seam_context_v3.prepare import read,save


def run(a):
    root=Path(a.root).resolve();training=root/'training';out=root/'evaluation'
    selection=read(training/'scorer_selection.json')
    cp=torch.load(training/'best_scorer.pt',map_location='cpu',weights_only=False)
    check_selection(read(training/'status.json'),selection,cp);del cp
    gpus=[int(v) for v in a.gpus.split(',')]
    if len(gpus)!=2 or len(set(gpus))!=2:raise ValueError('two distinct evaluation GPUs required')
    available={int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
    if not set(gpus)<=available.keys() or any(available[g]>700 for g in gpus):raise ValueError('assigned GPU busy or unavailable')
    out.mkdir(exist_ok=False);launches=[]
    for split,gpu in zip(('dunhuang_cv','turufan'),gpus):
        command=[sys.executable,'-u','-m',__package__+'.evaluate','--training',str(training),
            '--split',split,'--out',str(out/split)]
        with (out/(split+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=root/'source',env=dict(os.environ,
                PYTHONPATH=str(root/'source'),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launches.append(dict(split=split,gpu=gpu,pid=process.pid,command=command,
            proc_starttime=Path(f'/proc/{process.pid}/stat').read_text().split()[21]))
    save(out/'launch.json',dict(launched_at=datetime.now(timezone.utc).isoformat(),jobs=launches,training_complete=True))
    print(launches,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--gpus',default='0,1');run(p.parse_args())
