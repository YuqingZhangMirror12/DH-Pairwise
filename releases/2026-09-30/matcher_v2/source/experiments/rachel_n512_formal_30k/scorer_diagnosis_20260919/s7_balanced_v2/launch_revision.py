"""Detach exactly one named data export; never restart existing output."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from ..s7_compound_v1.materialize import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--groups',type=int,default=400)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=26092343)
    p.add_argument('--pilot',default='pilot_distribution_003')
    p.add_argument('--approval')
    p.add_argument('--reuse-completed-from')
    p.add_argument('--profile',default=str(Path(__file__).with_name('distribution_profile_v3.json').resolve()))
    a=p.parse_args();root=Path(a.root).resolve()
    launch=root/(a.output+'_launch.json')
    if launch.exists() or (root/a.output).exists():raise ValueError('already launched; inspect existing job')
    command=[sys.executable,'-u','-m',__package__+'.revision_pipeline','--root',str(root),
        '--profile',str(Path(a.profile).resolve()),
        '--output',a.output,'--groups',str(a.groups),'--workers',str(a.workers),'--seed',str(a.seed),
        '--pilot',a.pilot]
    if a.reuse_completed_from:command.extend(['--reuse-completed-from',str(Path(a.reuse_completed_from).resolve())])
    if a.approval:command.extend(['--approval',str(Path(a.approval).resolve())])
    with (root/(a.output+'_pipeline.log')).open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    starttime=int((Path('/proc')/str(child.pid)/'stat').read_text().split(') ',1)[1].split()[19])
    record=dict(pid=child.pid,starttime=starttime,command=command,cwd=str(Path.cwd()),training_started=False)
    save_json(launch,record);print(record)


if __name__=='__main__':main()
