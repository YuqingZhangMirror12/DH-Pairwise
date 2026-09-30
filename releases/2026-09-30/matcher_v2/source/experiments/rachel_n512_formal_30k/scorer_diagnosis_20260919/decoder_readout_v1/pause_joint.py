"""Explicit user-authorized checkpoint-preserving pause, identity checked.

Does not inject code into the bound trainer. It preserves the last atomic
AdamW/RNG checkpoint; any later in-memory work will be replayed, not claimed saved.
"""
import argparse
import os
from pathlib import Path
import shutil
import signal
import time
import torch
from launch_priority import JOINT, identity, same_process, read, save, sha, inventory, free


def alive(record):
    try:return same_process(identity(record['pid']),record) and identity(record['pid'])['state']!='Z'
    except FileNotFoundError:return False


def snapshot():
    launch=read(JOINT/'formal_launch_scratch_joint.json');controller=launch['controller'];job=launch['job']
    for r in (controller,job):
        if not same_process(identity(r['pid']),r):raise ValueError('formal process identity changed')
    if (str(JOINT) not in job['cmdline'] or 'torch.distributed.run' not in job['cmdline']
        or not str((JOINT/'source').resolve())==str((Path('/proc')/str(job['pid'])/'cwd').resolve())):
        raise ValueError('wrong joint process/cwd')
    if job.get('gpus')!='3,4':raise ValueError('joint assigned GPUs changed')
    ranks=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:r=identity(int(p.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if r['parent']==job['pid']:
            if '.train ' not in r['cmdline'] or str(JOINT/'formal_scratch_joint') not in r['cmdline']:
                raise ValueError('unexpected child under assigned torchrun')
            ranks.append(r)
    if len(ranks)!=2:raise ValueError('expected exactly two identified joint ranks')
    formal=JOINT/'formal_scratch_joint';stage=formal/'scorer'
    if any((JOINT/p).exists() for p in ('recovery_complete.json','failure_scratch_joint_controller.json')) or any((formal/p).exists() for p in ('training_complete.json','failure.json')):
        raise ValueError('joint is terminal/failed; do not signal')
    cp=torch.load(stage/'last.pt',map_location='cpu',weights_only=False)
    if (cp['stage']!='scorer' or cp['world_size']!=2 or len(cp['rng'])!=2
        or not cp.get('optimizer',{}).get('state') or len(cp['optimizer']['param_groups'])!=2
        or cp['binding']!=read(formal/'CONFIG.json') or cp['updates']<=0):
        raise ValueError('no complete joint optimizer/RNG checkpoint')
    return launch,ranks,cp,read(stage/'status.json')


def pause(apply=False):
    launch,ranks,cp,status=snapshot()
    out=JOINT/'pooling_priority_pause_20260929'
    if out.exists():raise ValueError('pause already attempted; inspect instead of redoing')
    report=dict(status='ready_to_pause',checkpoint_updates=cp['updates'],checkpoint_epoch=cp['epoch'],
                checkpoint_offset=cp['offset'],last_reported_updates=status.get('updates'),ranks=ranks,
                controller=launch['controller'],training=launch['job'])
    if not apply:return report
    out.mkdir()
    save(out/'before.json',report)
    # Stop only the controller first so a planned SIGTERM is not misclassified
    # as a training crash by its child.wait(). It does not own a GPU context.
    for r in [launch['controller'],launch['job']]:
        if not alive(r):raise ValueError('identity/lifecycle changed before planned signal')
        os.kill(r['pid'],signal.SIGTERM)
        if r is launch['controller']:
            deadline=time.monotonic()+10
            while alive(r) and time.monotonic()<deadline:time.sleep(.1)
            if alive(r):raise ValueError('controller did not stop; trainer preserved')
    deadline=time.monotonic()+45
    while any(alive(r) for r in [launch['job']]+ranks) and time.monotonic()<deadline:time.sleep(.25)
    if any(alive(r) for r in [launch['job']]+ranks):raise ValueError('planned SIGTERM did not release ranks; no forced kill')
    stage=JOINT/'formal_scratch_joint/scorer';path=stage/'last.pt';saved=out/'resume_full_state.pt'
    shutil.copy2(path,saved);digest=sha(path)
    if sha(saved)!=digest:raise ValueError('atomic checkpoint copy differs')
    cp=torch.load(saved,map_location='cpu',weights_only=False)
    for key in ('model','optimizer','rng','plateau','curve','epoch','offset','updates','exposures','binding'):
        if key not in cp:raise ValueError('missing complete resume field:'+key)
    if len(cp['rng'])!=2 or len(cp['optimizer']['param_groups'])!=2:raise ValueError('joint full state incomplete')
    for gpu in ('3','4'):free(gpu)
    failure=JOINT/'formal_scratch_joint/failure.json'
    if failure.exists():raise ValueError('rank wrote a failure during planned pause; preserve and diagnose before launching')
    shutil.copy2(JOINT/'formal_launch_scratch_joint.json',out/'formal_launch_before.json')
    shutil.copy2(JOINT/'driver_scratch_joint.json',out/'driver_before.json')
    final=dict(status='paused_at_verified_checkpoint',reason='user prioritized fresh pooling comparisons',
        checkpoint=str(path),checkpoint_copy=str(saved),checkpoint_sha256=digest,binding=cp['binding'],
        epoch=cp['epoch'],offset=cp['offset'],updates=cp['updates'],exposures=cp['exposures'],
        optimizer_parameter_groups=len(cp['optimizer']['param_groups']),rng_rank_count=len(cp['rng']),
        learning_rates=[g['lr'] for g in cp['optimizer']['param_groups']],
        last_reported_updates_before_pause=status.get('updates'),
        later_unsaved_updates_must_be_replayed=True,maximum_checkpoint_interval_updates=100,
        source_sha256=inventory(JOINT/'source'),training_command=launch['job']['command'],
        old_controller=launch['controller'],old_job=launch['job'],old_ranks=ranks,
        planned_sigterm=True,gpus_released=['3','4'],automatic_restart=False,paused_unix=time.time())
    save(out/'complete.json',final)
    save(JOINT/'driver_scratch_joint.json',dict(status='paused_by_user_priority',pause=str(out/'complete.json'),
        checkpoint_updates=cp['updates'],gpu_indices='3,4',old_controller=launch['controller'],resume_after='both pooling controls and required frozen evaluations'))
    return {k:final[k] for k in ('status','epoch','offset','updates','exposures','learning_rates','gpus_released','checkpoint_sha256')}


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
    print(json.dumps(pause(a.apply)))
