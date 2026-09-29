"""Detached data-only gates -> six thousand audited heldout instances.

No GPU processes, training, retries, data rewrites or recipe changes. A failed
fold stops escalation to full generation; all prior evidence remains intact.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..seam_context_v3.prepare import read,save,sha

MODULE='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.heldout_v14'


def identity(pid):
    p=Path('/proc')/str(pid)
    return dict(pid=pid,starttime=int((p/'stat').read_text().split(') ',1)[1].split()[19]),
                cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())


def run(root,workers):
    plan=read(root/'sources.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.time()
    try:
        for gate in (True,False):
            stage='gates' if gate else 'full'
            jobs={};logs=[]
            for si,split in enumerate(('cal','select','test')):
                output=root/('gates' if gate else 'full')/split
                if output.exists():
                    raise ValueError('previous output exists; inspect it rather than retry')
                command=[sys.executable,'-u','-m',MODULE,'materialize','--sources',str(root/'sources.json'),
                    '--split',split,'--out',str(output),'--workers',str(min(4,workers) if gate else workers),
                    '--seed',str(plan['seed']+1000+si)]
                if gate:
                    command.append('--gate')
                log=(root/(stage+'_'+split+'.log')).open('x');logs.append(log)
                child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                jobs[split]=(child,dict(**identity(child.pid),command=command,output=str(output)))
            while True:
                status={}
                for split,(child,launch) in jobs.items():
                    output=Path(launch['output']);code=child.poll()
                    record=read(output/'status.json') if (output/'status.json').exists() else {}
                    failed=(output/'failure.json').exists() or (code is not None and code!=0)
                    status[split]=dict(**launch,returncode=code,status=record,failed=failed)
                alive=any(r['returncode'] is None for r in status.values())
                failed=any(r['failed'] for r in status.values())
                save(root/'pipeline_status.json',dict(status='running_with_failure' if failed else 'running',
                    stage=stage,jobs=status,updated_unix=time.time(),training_started=False))
                if not alive:
                    break
                time.sleep(15)
            for log in logs:
                log.close()
            if failed or any(r['status'].get('status')!='complete' for r in status.values()):
                raise ValueError(stage+' failed; no automatic restart or escalation')
        subprocess.run([sys.executable,'-u','-m',MODULE,'finish','--sources',str(root/'sources.json'),
                        '--out',str(root/'full')],env=env,check=True)
        save(root/'pipeline_status.json',dict(status='complete',stage='data_ready',pairs=6000,
            elapsed_seconds=time.time()-started,training_started=False))
    except BaseException as exc:
        save(root/'pipeline_status.json',dict(status='failed',stage=locals().get('stage','setup'),
            error=repr(exc),elapsed_seconds=time.time()-started,training_started=False))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--driver',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve()
    if not 1<=a.workers<=16:
        raise ValueError('workers is per fold, maximum16')
    if a.driver:
        run(root,a.workers);return
    if not (root/'sources.json').exists() or (root/'launch.json').exists() or (root/'pipeline_status.json').exists():
        raise ValueError('complete source plan required; no repeat launcher')
    command=[sys.executable,'-u','-m',__spec__.name,'--root',str(root),'--workers',str(a.workers),'--driver']
    with (root/'pipeline.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    result=dict(**identity(child.pid),command=command,cwd=str(Path.cwd()),
        source_plan_sha256=sha(root/'sources.json'),training_started=False)
    save(root/'launch.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
