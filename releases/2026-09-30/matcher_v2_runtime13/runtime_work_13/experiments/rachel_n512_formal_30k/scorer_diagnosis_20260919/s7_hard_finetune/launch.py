"""Launch the approved S7-H job, preserving the historical model source paths."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from ..seam_context_v3.prepare import read, save

SOURCE = Path('/root/autodl-tmp/s7_hard_finetune_20260923/source')
OLD = Path('/root/autodl-tmp/rachel_score_design_20260913_001/scorer_diagnosis_20260919')
PYTHONPATH = str(SOURCE)
MATCHER = '/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt'
HEAD = str(OLD/'priority_s7_matched_g_v1/training/matched_tokens/head_epoch_016.pt')


def run(a):
    root = Path(a.root).resolve(); out = root/'training'
    gpus = [int(v) for v in a.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or not gpus:
        raise ValueError('GPU IDs must be distinct and nonempty')
    test = read(out/f'preflight_mb{a.microbatch}.json')
    if not test.get('passed') or test['world'] != len(gpus):
        raise ValueError('same-batch, same-world preflight required')
    if (out/'pause.request').exists():
        raise ValueError('explicit pause request is still present')
    if (out/'last.pt').exists() and not a.resume:
        raise ValueError('existing checkpoints require an explicit resume')
    if (out/'launch.json').exists():
        prior = read(out/'launch.json')
        stat = Path(f'/proc/{prior["pid"]}/stat')
        if stat.exists() and stat.read_text().split()[21] == prior.get('proc_starttime'):
            raise ValueError('recorded training parent is still running')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    available = {int(line.split(',')[0]): int(line.split(',')[1]) for line in gpu.splitlines()}
    if not set(gpus) <= available.keys() or any(available[g] > 700 for g in gpus):
        raise ValueError('assigned GPU unavailable or occupied; do not preempt')
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        f'--nproc_per_node={len(gpus)}', '-m', __package__+'.train',
        '--matcher', MATCHER, '--head', HEAD, '--data', str(root/'data'),
        '--out', str(out), '--microbatch', str(a.microbatch), '--workers', '2']
    if a.resume:
        command.append('--resume')
    environment = dict(os.environ, PYTHONPATH=PYTHONPATH, CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (out/'training.log').open('a') as log:
        process = subprocess.Popen(command, cwd=SOURCE, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = dict(pid=process.pid, proc_starttime=Path(f'/proc/{process.pid}/stat').read_text().split()[21],
        launched_at=datetime.now(timezone.utc).isoformat(), command=command, source=str(SOURCE),
        pythonpath=PYTHONPATH, gpus=gpus, microbatch=a.microbatch,
        effective_batch=a.microbatch*len(gpus), resume=a.resume, f_i_remain_paused=True)
    save(out/'launch.json',launch)
    print(launch,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--microbatch',type=int,default=16)
    p.add_argument('--resume',action='store_true');run(p.parse_args())
