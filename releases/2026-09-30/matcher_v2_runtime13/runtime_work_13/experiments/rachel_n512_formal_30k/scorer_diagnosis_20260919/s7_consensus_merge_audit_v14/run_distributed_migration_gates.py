"""Disposable two-arm GPU gates, including explicit update1 checkpoint replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def run(root):
    root=Path(root);source=root/'source';sys.path.insert(0,str(source))
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.launch import REFERENCE,MODULE,process_identity,validate_gates
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.migration import load_migration
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.train import make_binding
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.config import TrainingConfig
    from types import SimpleNamespace
    status=root/'preflight'/'driver_status.json'
    if status.exists():raise ValueError('preflight controller already registered; no implicit retry')
    for arm in ('m12','scratch'):
        if (root/'preflight'/('ddp_'+arm+'_mergefix_01')).exists():raise ValueError('preserve earlier gate')
    args=SimpleNamespace(arm='scratch',checkpoint=REFERENCE,contract=str(root/'data_contract.json'),
        calibration=str(root/'geometry_calibration_v2/geometry_calibration.json'),
        migration_plan=str(root/'migration_plan.json'),preflight_steps=12)
    binding=make_binding(args,TrainingConfig(),json.loads(Path(args.contract).read_text()))
    load_migration(args.migration_plan,binding,TrainingConfig())
    info=dict(controller=process_identity(os.getpid()),root=str(root),formal_training=False)
    try:
        write(status,dict(status='cpu_tests',**info))
        command=[sys.executable,'-m','unittest','discover','-s',
            'experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1','-t','.','-p','test_*.py']
        with (root/'preflight'/'cpu_tests.log').open('xb') as log:
            subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        for resumed in (False,True):
            jobs={};streams=[]
            for arm,devices in (('m12','0,1'),('scratch','2,3')):
                output=root/'preflight'/('ddp_'+arm+'_mergefix_01');output.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                    '-m',MODULE,'--arm',arm,'--out',str(output),'--checkpoint',REFERENCE,
                    '--contract',args.contract,'--calibration',args.calibration,
                    '--migration-plan',args.migration_plan,'--preflight-steps','12']
                if resumed:command.append('--resume')
                log=(output/('resume.log' if resumed else 'gate.log')).open('xb');streams.append(log)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
                jobs[arm]=(child,dict(**process_identity(child.pid),gpus=devices,command=command))
            while True:
                states={arm:dict(**receipt,returncode=child.poll()) for arm,(child,receipt) in jobs.items()}
                write(status,dict(status='resume_gate' if resumed else 'gpu_gate',**info,jobs=states))
                if all(row['returncode'] is not None for row in states.values()):break
                time.sleep(5)
            for stream in streams:stream.close()
            if any(row['returncode']!=0 for row in states.values()):raise RuntimeError('distributed gate failed; inspect retained logs')
        validate_gates(root)
        write(status,dict(status='passed',**info,gate_weights_discarded=True,ready_for_explicit_formal_launch=True))
    except Exception as exc:
        write(root/'preflight'/'failure.json',dict(status='failed',**info,error=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
