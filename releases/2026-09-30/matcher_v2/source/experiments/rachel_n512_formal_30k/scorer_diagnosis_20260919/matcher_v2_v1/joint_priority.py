"""Explicit, one-shot user-authorized joint pause -> v2 gate -> joint resume.

Never changes a running source or starts v2 formal training. The joint's last
atomic full checkpoint is preserved; unsaved steps are replayed. Even a failed
v2 gate resumes the known joint checkpoint once (no automatic retry).
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time

import torch

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.model_adapter import require

JOINT = Path('/root/autodl-tmp/s7_consensus_threshold_joint_e32_20260927/recovery_20260929_01')
GROUP = Path('/root/autodl-tmp/s7_pooling_controls_e32_20260929')
OLD_CONTROLLER = Path('/root/autodl-tmp/pooling_priority_20260929/runtime_01/launch_priority.py')


def read(path):return json.loads(Path(path).read_text())


def identity(pid):
    path = Path('/proc')/str(pid); fields = (path/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid, starttime=int(fields[19]), state=fields[0], parent=int(fields[1]),
        cmdline=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())


def same(a,b):return all(a[k] == b[k] for k in ('pid','starttime','cmdline'))


def alive(record):
    try:current=identity(record['pid']);return same(current,record) and current['state']!='Z'
    except FileNotFoundError:return False


def source_inventory(root):return {str(p.relative_to(root)):file_sha(p) for p in root.rglob('*.py')}


def check_controller(record):
    if str(OLD_CONTROLLER) in record['cmdline'] and '--group' in record['cmdline']:
        return
    words=shlex.split(record['cmdline'])
    require(__package__+'.joint_priority' in words and '--source' in words and '--out-new' in words,
            'unrecognized joint continuation controller')
    source=Path(words[words.index('--source')+1]).resolve();previous=Path(words[words.index('--out-new')+1]).resolve()
    scope=Path('/root/autodl-tmp/matcher_v2_20260930')
    require(source.is_relative_to(scope) and previous.is_relative_to(scope), 'controller outside registered v2 scope')
    launch=read(previous/'controller_launch.json');resumed=read(previous/'joint_resume_launch.json')
    require(same(record,launch['controller']) and same(record,resumed['controller'])
        and file_sha(source/'source_binding.json')==launch['source_binding_sha256'],
        'previous authorized continuation identity/source changed')


def check_checkpoint(cp, binding):
    for key in ('model','optimizer','rng','plateau','curve','epoch','offset','updates','exposures','binding'):
        require(key in cp, 'incomplete saved joint state: '+key)
    require(cp['stage']=='scorer' and cp['world_size']==2 and len(cp['rng'])==2
        and cp['optimizer']['state'] and len(cp['optimizer']['param_groups'])==2
        and cp['binding']==binding and cp['updates']>0, 'invalid joint checkpoint')


def snapshot():
    launch=read(JOINT/'formal_launch_scratch_joint.json'); controller=launch['controller'];job=launch['job']
    require(alive(controller) and alive(job), 'current joint controller/job identity changed')
    check_controller(controller)
    require(job['gpus']=='3,4' and str(JOINT/'formal_scratch_joint') in job['cmdline']
        and 'torch.distributed.run' in job['cmdline'] and '.s7_consensus_v1.train ' in job['cmdline']
        and (Path('/proc')/str(job['pid'])/'cwd').resolve()==JOINT/'source', 'wrong joint trainer')
    for path in (JOINT/'recovery_complete.json',JOINT/'failure_scratch_joint_controller.json',
                 JOINT/'formal_scratch_joint/training_complete.json',JOINT/'formal_scratch_joint/failure.json',
                 GROUP/'queue_failure.json'):
        require(not path.exists(), 'terminal/failure exists: '+str(path))
    ranks=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:r=identity(int(p.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if r['parent']==job['pid']:
            require('.s7_consensus_v1.train ' in r['cmdline'] and str(JOINT/'formal_scratch_joint') in r['cmdline'],
                    'unexpected child beneath torchrun')
            ranks.append(r)
    require(len(ranks)==2, 'expected exactly two training ranks')
    path=JOINT/'formal_scratch_joint/scorer/last.pt'
    cp=torch.load(path,map_location='cpu',weights_only=False)
    check_checkpoint(cp,read(JOINT/'formal_scratch_joint/CONFIG.json'))
    return launch,ranks,cp,read(JOINT/'formal_scratch_joint/scorer/status.json')


def pause(out, free):
    launch,ranks,cp,status=snapshot()
    write_json(out/'before_pause.json',dict(launch=launch,ranks=ranks,status=status,checkpoint_updates=cp['updates']))
    # Controller owns no GPU and is waiting on its child. End it first so a
    # planned pause is not reported as an unexpected nonzero training return.
    for record in (launch['controller'],launch['job']):
        require(alive(record),'process changed before planned signal')
        os.kill(record['pid'],signal.SIGTERM)
        if record is launch['controller']:
            deadline=time.monotonic()+10
            while alive(record) and time.monotonic()<deadline:time.sleep(.1)
            require(not alive(record),'controller did not stop; trainer preserved')
    deadline=time.monotonic()+45
    while any(alive(r) for r in [launch['job']]+ranks) and time.monotonic()<deadline:time.sleep(.25)
    require(not any(alive(r) for r in [launch['job']]+ranks),'planned SIGTERM did not release ranks; never force-kill')
    path=JOINT/'formal_scratch_joint/scorer/last.pt'; saved=out/'joint_resume_full_state.pt'
    shutil.copy2(path,saved); digest=file_sha(path)
    require(file_sha(saved)==digest,'checkpoint copy changed')
    cp=torch.load(saved,map_location='cpu',weights_only=False)
    check_checkpoint(cp,read(JOINT/'formal_scratch_joint/CONFIG.json'))
    for gpu in ('3','4'):free(gpu)
    require(not (JOINT/'formal_scratch_joint/failure.json').exists(),'joint wrote a failure; preserve for inspection')
    command=list(launch['job']['command'])
    if '--resume' not in command:command.append('--resume')
    receipt=dict(status='paused_at_verified_checkpoint', checkpoint=str(path), checkpoint_copy=str(saved),
        checkpoint_sha256=digest,binding=cp['binding'],updates=cp['updates'],exposures=cp['exposures'],
        epoch=cp['epoch'],offset=cp['offset'],rng_rank_count=2,optimizer_parameter_groups=2,
        learning_rates=[g['lr'] for g in cp['optimizer']['param_groups']],training_command=command,
        last_reported_updates_before_pause=status.get('updates'),later_unsaved_updates_must_be_replayed=True,
        source_sha256=source_inventory(JOINT/'source'),old_launch=launch,old_ranks=ranks,
        reason='user authorized two GPUs for Matcher v2 feasibility',planned_sigterm=True,gpus_released=['3','4'])
    write_json(out/'pause_complete.json',receipt)
    write_json(JOINT/'driver_scratch_joint.json',dict(status='paused_by_user_v2_verification',pause=str(out/'pause_complete.json'),
        checkpoint_updates=cp['updates'],gpu_indices='3,4',resume_after='bounded v2 gate, even if gate fails'),replace=True)
    return receipt


def resume(out,pause,old):
    for gpu in ('3','4'):old.free(gpu)
    require(file_sha(pause['checkpoint'])==pause['checkpoint_sha256']
        and source_inventory(JOINT/'source')==pause['source_sha256'],'joint checkpoint/source changed during v2 test')
    cp=torch.load(pause['checkpoint'],map_location='cpu',weights_only=False);check_checkpoint(cp,pause['binding'])
    with (out/'joint_resume.log').open('xb') as log:
        child=subprocess.Popen(pause['training_command'],cwd=JOINT/'source',stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1'))
    time.sleep(.2);job=dict(**identity(child.pid),command=pause['training_command'],gpus='3,4',
                           phase='formal_resume_v2_gate',root=str(JOINT/'formal_scratch_joint'))
    controller=identity(os.getpid())
    launch=dict(controller=controller,job=job,checkpoint_sha256=pause['checkpoint_sha256'],updates=cp['updates'])
    write_json(out/'joint_resume_launch.json',launch)
    write_json(JOINT/'formal_launch_scratch_joint.json',launch,replace=True)
    write_json(JOINT/'driver_scratch_joint.json',dict(status='running_formal',controller=controller,job=job,
        gpu_indices='3,4',v2_verification_pause_resumed=True),replace=True)
    code=child.wait();write_json(out/'joint_resume_return.json',dict(returncode=code,job=job))
    write_json(JOINT/'scratch_joint_formal_exit.json',dict(returncode=code,job=job,finished_unix=time.time()),replace=True)
    require(code==0,'joint continuation failed; no automatic retry')
    formal=JOINT/'formal_scratch_joint';selection=read(formal/'scorer/selection.json');terminal=read(formal/'training_complete.json')
    require(selection['status']=='selected' and terminal['status']=='training_complete'
        and selection['binding']==pause['binding'] and selection['matcher_unchanged'] is False,
        'joint completion does not verify')
    for name in ('best_joint','best_real'):
        require(file_sha(formal/'scorer'/(name+'.pt'))==selection[name+'_sha256'],'joint selected model changed')
    sys.path.insert(0,'/root/autodl-tmp/consensus_threshold_joint_20260927/recovery_source_01/evaluation_queue')
    import run_queued as queue
    common=queue.load(queue.QUEUE/'source/common.py','v2_priority_joint_eval_io')
    write_json(JOINT/'driver_scratch_joint.json',dict(status='running_frozen_evaluation',controller=controller,gpu_indices='3,4'),replace=True)
    evaluated=queue.evaluate_all(common,'3,4',JOINT/'postprocess_joint_01')
    write_json(JOINT/'recovery_complete.json',dict(status='complete',training_terminal_sha256=file_sha(formal/'training_complete.json'),
        evaluation_terminal_sha256=file_sha(JOINT/'postprocess_joint_01/evaluation_complete.json'),gpus='3,4',automatic_restarts=0,
        authorized_pause_resume=True))
    write_json(JOINT/'driver_scratch_joint.json',dict(status='complete',controller=controller,selection=selection,evaluation=evaluated),replace=True)
    write_json(GROUP/'priority_queue_complete.json',dict(status='complete',joint_resumed_and_evaluated=True,
        v2_verification_pause=str(out/'pause_complete.json')))
    write_json(out/'joint_continuation_complete.json',dict(status='training_and_evaluation_complete'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','spec','preparation','out-new'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--old-controller-sha256',required=True)
    args=p.parse_args();out=args.out_new.resolve()
    require(not out.exists(),'new explicit one-shot priority root required')
    require(file_sha(OLD_CONTROLLER)==args.old_controller_sha256,'joint supervising source differs')
    spec=importlib.util.spec_from_file_location('_verified_old_joint_controller',OLD_CONTROLLER)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    # Fail before any signal unless preparation/spec validation is successful.
    from .runtime_inputs import load_inputs
    values=load_inputs(read(args.spec));require(values['source']==args.source.resolve(),'bound v2 source differs')
    prep=read(args.preparation)
    require(prep['status']=='passed' and prep['errors']==prep['failures']==prep['skipped']==0
        and prep['binding_sha256']==file_sha(args.source/'source_binding.json') and prep['cuda_initialized'] is False,
        'complete same-source CPU proof required before pausing')
    snapshot();out.mkdir(parents=True)
    write_json(out/'controller_launch.json',dict(controller=identity(os.getpid()),authorization='2026-09-30 user request',
        source_binding_sha256=file_sha(args.source/'source_binding.json'),automatic_retry=False))
    paused=pause(out,old.free)
    try:
        command=[sys.executable,'-m',__package__+'.launcher','--spec',str(args.spec),'--preparation',str(args.preparation),
            '--out',str(out/'v2_gate'),'--python',sys.executable,'--gpus','3','4','--gate-only']
        with (out/'v2_gate_controller.log').open('xb') as log:
            result=subprocess.run(command,cwd=args.source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        write_json(out/'v2_gate_return.json',dict(returncode=result.returncode,command=command))
    finally:
        # A feasibility failure never silently strands the previously running experiment.
        resume(out,paused,old)


if __name__=='__main__':main()
