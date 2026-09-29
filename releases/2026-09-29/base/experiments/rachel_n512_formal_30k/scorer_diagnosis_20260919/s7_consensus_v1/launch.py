"""One detached, non-restarting controller for two isolated two-GPU arms."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import sys
import time
import traceback

from .preflight_matcher import digest
from .config import TrainingConfig
from .train import make_binding, save_json


MODULE='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.train'
REFERENCE='/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt'
PREFLIGHT_STEPS=12


def process_identity(pid):
    root=Path('/proc')/str(pid)
    stat=(root/'stat').read_text()
    after=stat[stat.rfind(')')+2:].split()
    return dict(pid=pid,starttime=int(after[19]),cmdline=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())


def validate_gate_receipt(gate,arm,stage,expected_binding):
    """Do not reuse a successful gate from a different data recipe or source."""
    if (gate.get('status')!='passed' or not gate.get('resume_matches_uninterrupted')
            or gate.get('formal_training') is not False
            or gate.get('updated_weights_discarded') is not True
            or gate.get('arm')!=arm or gate.get('stage')!=stage
            or gate.get('world_size')!=2 or gate.get('microbatch')!=8
            or gate.get('accumulate')!=2 or gate.get('effective_batch')!=32
            or gate.get('updates')!=PREFLIGHT_STEPS
            or gate.get('exposures')!=32*PREFLIGHT_STEPS
            or len(gate.get('model_state_hashes',[]))!=2
            or len(set(gate['model_state_hashes']))!=1
            or (arm=='m12' and gate.get('matcher_unchanged') is not True)):
        raise ValueError('distributed forward/backward/resume gate failed: '+arm)
    if gate.get('binding')!=expected_binding:
        raise ValueError('gate data/calibration/reference/config/source binding differs: '+arm)


def validate_gates(root):
    contract_path=root/'data_contract.json'
    calibration_path=root/'geometry_calibration_v2'/'geometry_calibration.json'
    contract=json.loads(contract_path.read_text())
    calibration=json.loads(calibration_path.read_text())
    migration_path=root/'migration_plan.json'
    migration_plan=str(migration_path) if migration_path.exists() else None
    if (calibration.get('status')!='complete'
            or calibration.get('schema')!='s7-consensus-train-geometry/2'
            or calibration.get('contract_sha256')!=digest(contract_path)):
        raise ValueError('geometry calibration must be rebuilt from the registered TRAIN contract')
    for arm,stage in (('m12','scorer'),('scratch','matcher')):
        gate_name='ddp_'+arm+('_mergefix_01' if migration_plan else '_conservative_01')
        path=root/'preflight'/gate_name/stage/'preflight_resumed.json'
        gate=json.loads(path.read_text())
        args=SimpleNamespace(arm=arm,checkpoint=REFERENCE,contract=str(contract_path),
            calibration=str(calibration_path),preflight_steps=PREFLIGHT_STEPS,migration_plan=migration_plan)
        expected=make_binding(args,TrainingConfig(),contract)
        validate_gate_receipt(gate,arm,stage,expected)
        if arm=='scratch' and migration_plan:
            from .migration import load_migration
            plan,_,saved=load_migration(migration_plan,expected,TrainingConfig())
            origin=gate.get('migration_origin') or {}
            if (origin.get('checkpoint')!=plan['resume'] or origin.get('retained_updates')!=saved['updates']
                    or gate.get('total_updates')!=saved['updates']+PREFLIGHT_STEPS):
                raise ValueError('scratch gate did not use the actual imported optimizer/RNG checkpoint')
    space=shutil.disk_usage(root)
    if space.free<20*2**30:
        raise ValueError('insufficient headroom for bound proposal caches/checkpoints')
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    available={int(line.split(',')[0]):line.split(',')[1].strip() for line in query.splitlines() if line.strip()}
    if not {0,1,2,3}<=set(available):
        raise ValueError('four registered GPUs not available; do not silently change batch topology')
    usage=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    occupied={line.split(',')[0].strip() for line in usage.splitlines() if line.strip()}
    if occupied & {available[i] for i in (0,1,2,3)}:
        raise ValueError('registered GPUs already have compute processes')


def driver(root):
    lock=(root/'controller.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(__file__).resolve().parents[4]
    jobs={};streams=[]
    try:
        validate_gates(root)
        for arm,devices in (('m12','0,1'),('scratch','2,3')):
            output=root/('formal_'+arm)
            if output.exists():
                raise ValueError('preserve existing formal experiment; no implicit restart: '+str(output))
        for arm,devices in (('m12','0,1'),('scratch','2,3')):
            output=root/('formal_'+arm);output.mkdir()
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                '-m',MODULE,'--arm',arm,'--out',str(output),'--checkpoint',REFERENCE,
                '--contract',str(root/'data_contract.json'),
                '--calibration',str(root/'geometry_calibration_v2'/'geometry_calibration.json')]
            if (root/'migration_plan.json').exists():
                command.extend(['--migration-plan',str(root/'migration_plan.json')])
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2')
            log=(output/'train.log').open('ab');streams.append(log)
            child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
            launch=dict(**process_identity(child.pid),arm=arm,gpus=devices,command=command,
                cwd=str(source),started_unix=time.time(),microbatch=8,accumulate=2,effective_batch=32,
                restart_policy='never',root=str(output))
            save_json(output/'launch.json',launch);jobs[arm]=(child,launch)
        save_json(root/'formal_launch.json',dict(controller=process_identity(os.getpid()),
            jobs={k:v[1] for k,v in jobs.items()},started_unix=time.time()))
        while True:
            states={}
            for arm,(child,launch) in jobs.items():
                code=child.poll();output=Path(launch['root']);complete=output/'training_complete.json'
                verified=bool(code==0 and complete.exists() and json.loads(complete.read_text()).get('status')=='training_complete')
                states[arm]=dict(**launch,returncode=code,training_complete_verified=verified,
                    status='running' if code is None else ('training_complete' if verified else 'failed'))
            alive=any(v['returncode'] is None for v in states.values())
            failed=any(v['status']=='failed' for v in states.values())
            status=('running_with_failure' if failed else 'running') if alive else ('failed' if failed else 'training_complete_evaluation_pending')
            save_json(root/'driver_status.json',dict(status=status,jobs=states,updated_unix=time.time(),
                real_evaluation_started=False,automatic_restarts=0))
            if not alive:
                break
            time.sleep(20)
    except Exception as error:
        save_json(root/'driver_failure.json',dict(status='failed',type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),time_unix=time.time(),
            child_pids={k:v[0].pid for k,v in jobs.items()},children_not_automatically_stopped=True))
        raise
    finally:
        for log in streams:
            log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--driver',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve()
    if args.driver:
        driver(root);return
    if (root/'controller_launch.json').exists() or (root/'formal_launch.json').exists():
        raise ValueError('controller already registered; inspect before any explicit recovery')
    validate_gates(root)
    command=[sys.executable,'-m',__spec__.name,'--root',str(root),'--driver']
    with (root/'controller.log').open('ab') as stream:
        child=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[4],stdout=stream,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    receipt=dict(**process_identity(child.pid),command=command,root=str(root),started_unix=time.time())
    save_json(root/'controller_launch.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
