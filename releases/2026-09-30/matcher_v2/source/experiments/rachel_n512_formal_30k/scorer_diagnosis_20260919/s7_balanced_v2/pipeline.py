"""Run only the approved data export after a passed real-loader pilot."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from ..s7_compound_v1.materialize import read,save_json


def wait_pilot(root,parent_pid,parent_starttime):
    """Observe the existing pilot, never start or restart a second producer."""
    validation=root/'pilot_001/validation.json'
    if parent_pid is None:
        return
    if parent_starttime is None:
        raise ValueError('pilot process identity required')
    save_json(root/'pipeline_status.json',dict(status='waiting_for_pilot',stage='pilot',
        parent_pid=os.getpid(),pilot_parent_pid=parent_pid,training_started=False))
    deadline=time.monotonic()+7200
    try:
        while True:
            if validation.exists():
                if read(validation)['status']!='passed':raise ValueError('pilot validation did not pass')
                return
            status=root/'pilot_001/status.json'
            if status.exists() and read(status)['status']=='failed':
                raise RuntimeError('pilot generation failed; no formal data started')
            process=Path('/proc')/str(parent_pid)
            if (not process.exists() or
                    int((process/'stat').read_text().split(') ',1)[1].split()[19])!=parent_starttime):
                raise RuntimeError('pilot/validator parent exited without passed validation')
            if time.monotonic()>deadline:
                raise TimeoutError('pilot wait timed out; producer was not stopped or restarted')
            time.sleep(10)
    except BaseException as exc:
        save_json(root/'pipeline_status.json',dict(status='failed',stage='pilot',error=repr(exc),
            parent_pid=os.getpid(),training_started=False))
        raise


def run(root,workers,pilot_parent_pid=None,pilot_parent_starttime=None):
    root=Path(root).resolve()
    wait_pilot(root,pilot_parent_pid,pilot_parent_starttime)
    if read(root/'pilot_001/validation.json')['status']!='passed':
        raise ValueError('pilot must pass before full production')
    if not (root/'sources_v2.json').is_file():
        raise ValueError('source plan missing')
    out=root/'train24k_v2'
    if out.exists():
        raise ValueError('formal output already exists; inspect rather than overwrite')
    out.mkdir()
    snapshot=out/'generator_snapshot'
    shutil.copytree(Path(__file__).parent,snapshot/'s7_balanced_v2',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(Path(__file__).parent.parent/'s7_compound_v1',snapshot/'s7_compound_v1',
                    ignore=shutil.ignore_patterns('__pycache__'))
    start=time.time()
    prefix='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.'
    steps=[('generate',[sys.executable,'-u','-m',prefix+'materialize',
            '--sources',str(root/'sources_v2.json'),'--out',str(out),'--workers',str(workers)]),
           ('validate',[sys.executable,'-u','-m',prefix+'validate','--root',str(out)]),
           ('describe',[sys.executable,'-u','-m',prefix+'describe','--root',str(out)])]
    os.nice(10)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        for stage,command in steps:
            with (root/(stage+'.log')).open('a') as stream:
                child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
                record=dict(status='running',stage=stage,parent_pid=os.getpid(),child_pid=child.pid,
                            child_command=command,started_at=start,training_started=False)
                save_json(root/'pipeline_status.json',record)
                code=child.wait()
                if code:
                    raise RuntimeError(f'{stage} exited with code {code}')
        save_json(root/'pipeline_status.json',dict(status='complete',stage='ready',parent_pid=os.getpid(),
            root=str(out),elapsed_seconds=time.time()-start,training_started=False))
    except BaseException as exc:
        save_json(root/'pipeline_status.json',dict(status='failed',stage=stage,error=repr(exc),
            parent_pid=os.getpid(),elapsed_seconds=time.time()-start,training_started=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--pilot-parent-pid',type=int);p.add_argument('--pilot-parent-starttime',type=int)
    a=p.parse_args();run(a.root,a.workers,a.pilot_parent_pid,a.pilot_parent_starttime)
