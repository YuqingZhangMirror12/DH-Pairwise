"""Two authorized fresh pooling controls, then explicit joint continuation.

No blind restart, no queue based on free memory alone, no source hot-editing.
All GPU processes belong to an independently registered single-GPU lane.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

PACKAGE='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1'
ORIGINAL=Path('/root/autodl-tmp/s7_consensus_layered_v14_mergefix_20260925')
ROOT=Path('/root/autodl-tmp/s7_pooling_controls_e32_20260929')
PREPARED=Path('/root/autodl-tmp/pooling_priority_20260929/runtime_01')
JOINT=Path('/root/autodl-tmp/s7_consensus_threshold_joint_e32_20260927/recovery_20260929_01')
REFERENCE='/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt'
E32=ORIGINAL/'formal_scratch/matcher/best_joint.pt'
E32_SHA='80cac47d5bc5340df35a7a7c36ab4a3580a9eea8c99adf797ff5744cd2068b17'
GPUS={'patch_mean':'3','patch_sum':'4'}

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory(p):
    p=Path(p);return {str(x.relative_to(p)):sha(x) for x in sorted(p.rglob('*.py'))}
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('x') as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def identity(pid):
    p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime=int(f[19]),state=f[0],parent=int(f[1]),
                cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
def same_process(a,b):return all(a[k]==b[k] for k in ('pid','starttime','cmdline'))
def free(gpu):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=20)
    ids={r.split(',')[0].strip():r.split(',')[1].strip() for r in raw.splitlines()}
    used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=20)
    if gpu not in ids or ids[gpu] in used.splitlines():raise ValueError('assigned GPU not free; no preemption by this controller')
    return ids[gpu]

def verify_preparation():
    r=read(PREPARED/'preparation_remote.json')
    if (r['status']!='cpu_preparation_passed' or r['verified_variants']!=list(GPUS)
        or r['errors'] or r['failures'] or r['skipped'] or r['tests']<64
        or r['training_source_sha256']!=inventory(PREPARED/'source')
        or r['adapter_python_sha256']!=inventory(PREPARED/'pooling_eval_v1')
        or r['common_python_sha256']!=inventory(PREPARED/'s7_consensus_eval_v14')):
        raise ValueError('bound CPU/evaluation preparation changed')
    external=read(PREPARED/'controller_preparation.json')
    if external['status']!='passed' or external['sha256']!=sha(__file__):
        raise ValueError('controller tests/source not verified')
    if sha(E32)!=E32_SHA:raise ValueError('confirmed frozen E32 differs')
    return r

def prepare_formal():
    r=verify_preparation();ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'preparation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (ROOT/'source').exists():shutil.copytree(PREPARED/'source',ROOT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        if inventory(ROOT/'source')!=r['training_source_sha256']:raise ValueError('formal source differs')
        for rel in ('data_contract.json','geometry_calibration_v2/geometry_calibration.json'):
            dst=ROOT/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():shutil.copy2(ORIGINAL/rel,dst)
            if sha(dst)!=sha(ORIGINAL/rel):raise ValueError('data or geometry changed')
        for name in ('real_split.json','preparation_remote.json'):
            if not (ROOT/name).exists():shutil.copy2(PREPARED/name,ROOT/name)
            if sha(ROOT/name)!=sha(PREPARED/name):raise ValueError('source roles/preparation changed')
    pause=read(JOINT/'pooling_priority_pause_20260929/complete.json')
    if pause.get('status')!='paused_at_verified_checkpoint' or pause.get('gpus_released')!=['3','4']:
        raise ValueError('explicit verified joint pause required')
    if sha(pause['checkpoint_copy'])!=pause['checkpoint_sha256']:
        raise ValueError('paused full state changed')
    return pause

def command(variant,out,gate=False,resume=False):
    args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
        '-m',PACKAGE+'.train','--arm','scratch_fixed','--scorer-variant',variant,
        '--out',str(out),'--checkpoint',REFERENCE,'--contract',str(ROOT/'data_contract.json'),
        '--calibration',str(ROOT/'geometry_calibration_v2/geometry_calibration.json'),
        '--real-split',str(ROOT/'real_split.json'),'--frozen-matcher-state',str(E32),'--frozen-matcher-sha256',E32_SHA]
    if gate:args+=['--preflight-steps','12']
    if resume:args+=['--resume']
    return args

def validate_gate(r,binding,replay=False):
    expected=dict(status='passed',formal_training=False,updated_weights_discarded=True,arm='scratch_fixed',
        stage='scorer',updates=12,exposures=384,world_size=1,microbatch=32,accumulate=1,effective_batch=32,
        matcher_unchanged=True,head_updated=True)
    if any(r.get(k)!=v for k,v in expected.items()) or r.get('binding')!=binding:
        raise ValueError('new head actual GPU gate differs')
    if len(r.get('model_state_hashes',[]))!=1 or r['initial_head_sha256']==r['final_head_sha256']:
        raise ValueError('gate did not prove an actual head update')
    if replay and r.get('resume_matches_uninterrupted') is not True:raise ValueError('update1 replay failed')

def terminal(variant):
    formal=ROOT/('formal_'+variant);s=read(formal/'scorer/selection.json');t=read(formal/'training_complete.json')
    if (s['status']!='selected' or t['status']!='training_complete' or t['last_stage']!=read(formal/'scorer/complete.json')
        or s['matcher_unchanged'] is not True or not s['stop_reason'] or s['test_used'] is not False
        or s['binding']!=t['binding'] or any((formal/p).exists() for p in ('failure.json','scorer/failure.json'))):
        raise ValueError('formal terminal inconsistent')
    for f in ('best_joint','best_real'):
        if sha(formal/'scorer'/(f+'.pt'))!=s[f+'_sha256']:raise ValueError('selected weights changed')
    return s

def lane(variant):
    prepare_formal();gpu=GPUS[variant];lock=(ROOT/(variant+'.lock')).open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);controller=identity(os.getpid());current=None
    def publish(status,**kw):save(ROOT/('driver_'+variant+'.json'),dict(status=status,controller=controller,gpu=gpu,updated_unix=time.time(),automatic_retries=0,**kw))
    def execute(args,phase,cwd):
        nonlocal current
        uuid=free(gpu);logs=ROOT/'logs';logs.mkdir(exist_ok=True)
        with (logs/(variant+'_'+phase+'.log')).open('xb') as log:
            child=subprocess.Popen(args,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1'))
        time.sleep(.15);current=dict(**identity(child.pid),command=args,gpu=gpu,gpu_uuid=uuid,phase=phase,started_unix=time.time())
        save(ROOT/(variant+'_'+phase+'_launch.json'),dict(controller=controller,job=current));publish('running_'+phase,job=current)
        code=child.wait();save(ROOT/(variant+'_'+phase+'_exit.json'),dict(returncode=code,job=current,finished_unix=time.time()))
        if code:raise RuntimeError(phase+' failed; no retry: '+str(code))
        current=None
    try:
        sys.path.insert(0,str(ROOT/'source'));train=importlib.import_module(PACKAGE+'.train')
        cfg=train.TrainingConfig(scorer_variant=variant)
        args=SimpleNamespace(arm='scratch_fixed',scorer_variant=variant,checkpoint=REFERENCE,
            contract=str(ROOT/'data_contract.json'),calibration=str(ROOT/'geometry_calibration_v2/geometry_calibration.json'),
            frozen_matcher_state=str(E32),frozen_matcher_sha256=E32_SHA,real_split=str(ROOT/'real_split.json'),preflight_steps=12)
        binding=train.make_binding(args,cfg,read(ROOT/'data_contract.json'));gate=ROOT/('gate_'+variant)
        formal=ROOT/('formal_'+variant)
        if gate.exists() or formal.exists():raise ValueError('prior attempt exists; inspect, never restart automatically')
        gate.mkdir();execute(command(variant,gate,True),'gate',ROOT/'source')
        a=read(gate/'scorer/preflight.json');validate_gate(a,binding)
        execute(command(variant,gate,True,True),'gate_resume',ROOT/'source')
        b=read(gate/'scorer/preflight_resumed.json');validate_gate(b,binding,True)
        if a['model_state_hashes']!=b['model_state_hashes']:raise ValueError('gate replay hash differs')
        save(ROOT/(variant+'_gpu_gate.json'),dict(status='passed',original=a,replay=b,short_weights_discarded=True))
        formal.mkdir();execute(command(variant,formal),'formal',ROOT/'source');selection=terminal(variant)
        jobs=[]
        for choice in ('sim','real'):
            for split in ('sim_test_v14','dunhuang_cv','turufan'):
                out=ROOT/'evaluation'/variant/(choice+'_'+split)
                ev=[sys.executable,str(PREPARED/'pooling_eval_v1/entry.py'),'--root',str(ROOT),
                    '--variant',variant,'--selection',choice,'--split',split,'--reference',REFERENCE,
                    '--out',str(out),'--preparation',str(PREPARED/'preparation_remote.json'),
                    '--common-source',str(PREPARED/'s7_consensus_eval_v14'),'--real-plan',str(ROOT/'real_split.json'),
                    '--case-plan',str(PREPARED/'s7_consensus_eval_v14/case_plan.json')]
                execute(ev,'eval_'+choice+'_'+split,PREPARED)
                frozen=read(out/'prediction_complete.json');summary=read(out/'summary.json')
                if (frozen['status']!='all_predictions_frozen' or not frozen['model_state_unchanged']
                    or frozen['sha256']!=sha(out/'pair_predictions.jsonl') or summary['status']!='complete'
                    or read(out/'status.json')['status']!='complete' or (out/'failure.json').exists()):
                    raise ValueError('frozen population incomplete')
                jobs.append(dict(out=str(out),selection=choice,split=split,returncode=0,predictions_sha256=frozen['sha256']))
        save(ROOT/(variant+'_complete.json'),dict(status='training_and_evaluation_complete',selection=selection,jobs=jobs,gpu=gpu))
        publish('complete',jobs=jobs)
    except BaseException as e:
        save(ROOT/(variant+'_failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),job=current,automatic_retry=False));publish('failed');raise
    finally:lock.close()

def resume_joint(pause):
    """One authorized continuation, not re-running a gate or a fresh experiment."""
    import torch
    for gpu in ('3','4'):free(gpu)
    directory=JOINT/'pooling_priority_resume_20260929';directory.mkdir(exist_ok=False)
    checkpoint=Path(pause['checkpoint']);cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if sha(checkpoint)!=pause['checkpoint_sha256'] or cp['binding']!=pause['binding']:
        raise ValueError('paused joint full state changed')
    if inventory(JOINT/'source')!=pause['source_sha256']:raise ValueError('joint source changed while paused')
    if any((JOINT/p).exists() for p in ('failure_scratch_joint_controller.json','recovery_complete.json','formal_scratch_joint/failure.json')):
        raise ValueError('joint lifecycle changed; inspect instead of resuming')
    command=list(pause['training_command'])+['--resume']
    with (directory/'training.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=JOINT/'source',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1'))
    time.sleep(.2);job=dict(**identity(child.pid),command=command,gpus='3,4',phase='formal_resume',root=str(JOINT/'formal_scratch_joint'))
    controller=identity(os.getpid());save(directory/'launch.json',dict(controller=controller,job=job,checkpoint_sha256=pause['checkpoint_sha256'],updates=cp['updates']))
    save(JOINT/'formal_launch_scratch_joint.json',dict(controller=controller,job=job))
    save(JOINT/'driver_scratch_joint.json',dict(status='running_formal',controller=controller,job=job,gpu_indices='3,4',priority_pause_resumed=True))
    code=child.wait();save(JOINT/'scratch_joint_formal_exit.json',dict(returncode=code,job=job,finished_unix=time.time()))
    if code:raise RuntimeError('joint explicit continuation failed; no retry')
    formal=JOINT/'formal_scratch_joint';s=read(formal/'scorer/selection.json');t=read(formal/'training_complete.json')
    if (s['status']!='selected' or t['status']!='training_complete' or s['binding']!=pause['binding']
        or s['matcher_unchanged'] is not False or any(sha(formal/'scorer'/(n+'.pt'))!=s[n+'_sha256'] for n in ('best_joint','best_real'))):
        raise ValueError('joint resumed terminal did not verify')
    queue_path=Path('/root/autodl-tmp/consensus_threshold_joint_20260927/recovery_source_01/evaluation_queue')
    sys.path.insert(0,str(queue_path));queue=importlib.import_module('run_queued')
    common=queue.load(queue.QUEUE/'source/common.py','priority_joint_evaluation_io')
    save(JOINT/'driver_scratch_joint.json',dict(status='running_frozen_evaluation',controller=controller,gpu_indices='3,4'))
    evaluated=queue.evaluate_all(common,'3,4',JOINT/'postprocess_joint_01')
    save(JOINT/'recovery_complete.json',dict(status='complete',training_terminal_sha256=sha(formal/'training_complete.json'),
        evaluation_terminal_sha256=sha(JOINT/'postprocess_joint_01/evaluation_complete.json'),gpus='3,4',automatic_restarts=0,
        authorized_pause_resume=True))
    save(JOINT/'driver_scratch_joint.json',dict(status='complete',controller=controller,selection=s,evaluation=evaluated))
    save(directory/'complete.json',dict(status='continued_to_training_and_evaluation_completion',checkpoint_sha256=pause['checkpoint_sha256']))

def group():
    pause=prepare_formal();lock=(ROOT/'priority_group.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        children=[]
        for v,gpu in GPUS.items():
            free(gpu)
            with (ROOT/(v+'_controller.log')).open('xb') as log:
                c=subprocess.Popen([sys.executable,__file__,'--lane',v],stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            time.sleep(.2);save(ROOT/('controller_launch_'+v+'.json'),identity(c.pid));children.append((v,c))
        for v,c in children:
            code=c.wait();save(ROOT/(v+'_controller_exit.json'),dict(returncode=code))
        if any(read(ROOT/(v+'_controller_exit.json'))['returncode'] for v in GPUS):raise ValueError('priority lane failed; preserve GPU reservation')
        for v in GPUS:
            if read(ROOT/(v+'_complete.json'))['status']!='training_and_evaluation_complete':raise ValueError('priority work incomplete')
        save(ROOT/'pooling_complete.json',dict(status='both_new_heads_and_frozen_evaluations_complete',variants=list(GPUS)))
        resume_joint(pause)
        save(ROOT/'priority_queue_complete.json',dict(status='complete',joint_resumed_and_evaluated=True))
    except BaseException as e:
        save(ROOT/'queue_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:lock.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=list(GPUS));p.add_argument('--group',action='store_true');a=p.parse_args()
    if a.lane:return lane(a.lane)
    if a.group:return group()
    prepare_formal()
    if (ROOT/'controller_launch.json').exists():raise ValueError('group already registered')
    for gpu in GPUS.values():free(gpu)
    with (ROOT/'controller.log').open('xb') as log:
        c=subprocess.Popen([sys.executable,__file__,'--group'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    time.sleep(.2);record=identity(c.pid);save(ROOT/'controller_launch.json',record);print(json.dumps(record))

if __name__=='__main__':main()
