"""One explicit detached CPU pipeline launch; never restarts automatically."""
import argparse,fcntl,os,subprocess,sys,time
from pathlib import Path
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_compound_v1.materialize import save_json,read

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pilot-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    root=a.out.resolve();config=read(root/'config.json')
    lock=(root/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=root/'detached_launch.json'
    if prior.exists() and not a.resume:raise ValueError('already launched; inspect exact identity before explicit resume')
    if prior.exists():
        r=read(prior);proc=Path('/proc')/str(r['pid'])
        if proc.exists() and int((proc/'stat').read_text().split(') ')[-1].split()[19])==r['starttime']:
            raise ValueError('same pipeline still alive; no duplicate')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    command=[sys.executable,'-m','experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_data_full_v17.pipeline','--out',str(root)]
    if a.pilot_only:command.append('--pilot-only')
    if a.resume:command.append('--resume')
    with (root/'pipeline.log').open('ab') as log:
        job=subprocess.Popen(command,cwd=config['source'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    proc=Path('/proc')/str(job.pid)
    r=dict(pid=job.pid,starttime=int((proc/'stat').read_text().split(') ')[-1].split()[19]),
        command=command,started_unix=time.time(),automatic_restarts=0,gpu_used=False)
    save_json(root/('detached_resume_'+str(job.pid)+'.json' if a.resume else 'detached_launch.json'),r)
    print(r)
if __name__=='__main__':main()
