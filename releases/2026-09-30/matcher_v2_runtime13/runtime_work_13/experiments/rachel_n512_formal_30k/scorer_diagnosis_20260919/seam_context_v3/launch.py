"""Explicit post-approval launch. Does not rebuild data or bypass a failed test."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from .prepare import read,save,sha


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--preflight',required=True)
    p.add_argument('--gpu-tests',required=True);p.add_argument('--distributed-check',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--resume-migration')
    p.add_argument('--gpus',default='0,1');args=p.parse_args()
    gpus=[int(x) for x in args.gpus.split(',')];world=len(gpus)
    if not gpus or len(set(gpus))!=world:raise RuntimeError('GPU IDs must be nonempty and distinct')
    root=Path(args.root).resolve();data=root/'data';out=root/'training'
    if read(data/'protocol.json')['status']!='ready':raise RuntimeError('data not ready')
    if not read(args.preflight)['passed'] or not read(args.gpu_tests)['passed']:raise RuntimeError('required checks failed')
    evidence=read(args.preflight)
    distributed=read(args.distributed_check)
    if not distributed.get('passed'):raise RuntimeError('distributed synchronization check failed')
    if evidence['intended_world_size']!=world or any(e['world_size']!=world for e in distributed['events']):
        raise RuntimeError('distributed check topology differs from launch')
    if any(event['microbatch']!=evidence['microbatch'][event['stage']] for event in distributed['events']):
        raise RuntimeError('distributed check used different physical batches')
    hashes={x.name:sha(x) for x in Path(__file__).parent.glob('*.py') if x.name not in ('launch.py','gpu_tests.py')}
    hashes['beam_kernel.cpp']=sha(Path(__file__).with_name('beam_kernel.cpp'))
    if evidence.get('source_hashes')!=hashes:
        raise RuntimeError('implementation changed after GPU preflight; revalidate first')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    for line in gpu.splitlines():
        index,used=map(int,line.split(','))
        if index in gpus and used>700:raise RuntimeError('assigned GPU busy; do not preempt unrelated process')
    available={int(line.split(',')[0]) for line in gpu.splitlines()}
    if not set(gpus)<=available:raise RuntimeError('requested GPU missing')
    out.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
        '-m',__package__+'.train','--data',str(data),'--out',str(out),'--preflight',args.preflight]
    if args.resume:command+=['--resume']
    if args.resume_migration:
        if not args.resume:raise RuntimeError('migration is only allowed with --resume')
        command+=['--resume-migration',str(Path(args.resume_migration).resolve())]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (out/'training.log').open('a') as log:
        process=subprocess.Popen(command,cwd=root/'source',env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(out/'launch.json',dict(pid=process.pid,command=command,gpus=gpus,effective_batch=32,
        source_hashes=evidence['source_hashes'],pretrained_source=None,resume_migration=args.resume_migration))
    print('Launched pid',process.pid,flush=True)
