"""Unchanged training launcher, externally repaired terminal evaluation.

An adopted pipeline must have completed training successfully and failed only
on the known pre-prediction identity bug. It is not retrained or relabelled as
an old success. The new, distinct completion receipt preserves both histories.
"""
import argparse
import os
from pathlib import Path
import time
import traceback

from common import (HERE, PACKAGE, api, read, sha, bound, save, require, check_preparation,
                    source_map, environment, ensure_evaluation_binding)
import controller


def train_command(args):
    return [str(args.python), '-m', PACKAGE+'.matcher_v2_v1.launcher', '--spec',str(args.spec),
        '--preparation',str(args.preparation),'--out',str(args.out/'training'),
        '--python',str(args.python),'--gpus',*map(str,args.gpus)]


def eval_command(args, training_root):
    values = [str(args.python),str(HERE/'controller.py')]
    for name in ('spec','preparation','canonical_straight','case_plan','python','repair_preparation'):
        values += ['--'+name.replace('_','-'),str(getattr(args,name))]
    values += ['--controller-root',str(training_root),'--out',str(args.out/'evaluation'),
               '--gpus',*map(str,args.gpus)]
    if args.adopt_pipeline:
        values += ['--reuse-evaluation',str(args.adopt_pipeline/'evaluation')]
    return values


def verify_complete(root):
    root = Path(root); require(not (root/'failure.json').exists(), 'pipeline failure precedes completion')
    made = read(root/'pipeline_launch.json'); actual = read(root/'complete.json')
    args = argparse.Namespace(**made['arguments'])
    for key,value in vars(args).copy().items():
        if key != 'gpus' and value is not None: setattr(args,key,Path(value))
    require(args.out == root, 'pipeline output identity differs')
    check_preparation(args.repair_preparation)
    inputs = api('matcher_v2_v1.runtime_inputs').load_inputs(read(args.spec))
    api('matcher_v2_v1.evaluate').check_preparation(args.preparation,inputs['source'])
    original = api('matcher_v2_v1.pipeline')
    train_root = (args.adopt_pipeline or root)/'training'
    require(made['source_files'] == source_map() and made['execution'] == bound(args.spec)
            and made['repair_preparation'] == bound(args.repair_preparation), 'tested pipeline changed')
    subprocesses = {'evaluation':original.verify_child(root,'evaluation',eval_command(args,train_root))}
    if not args.adopt_pipeline:
        subprocesses['training'] = original.verify_child(root,'training',train_command(args))
    evaluated = controller.verify_complete(root/'evaluation')
    for choice in dict.fromkeys(j['selection'] for j in api('matcher_v2_v1.evaluation_controller').jobs(read(args.spec)['module'])):
        api('matcher_v2_v1.terminal').verified_export(train_root,args.spec,inputs['plan'],choice)
    expected = dict(schema='matcher-v2-terminal-repaired-pipeline/1',status='complete',arm=read(args.spec)['arm'],
        module=read(args.spec)['module'],execution=bound(args.spec),subprocesses=subprocesses,
        effective_training_controller_root=str(train_root),training_complete=bound(train_root/'controller_complete.json'),
        evaluation_complete=bound(root/'evaluation/evaluation_complete.json'),
        adopted_original_pipeline=str(args.adopt_pipeline) if args.adopt_pipeline else None,
        pipeline_launch_sha256=sha(root/'pipeline_launch.json'),mandatory_frozen_evaluation_complete=True,
        automatic_retry=False,report_delivery_still_required=True,gpu_release_requires_current_idle_check=True)
    require({k:v for k,v in actual.items() if k!='completed_unix'} == expected, 'pipeline completion evidence differs')
    return actual


def run(args):
    for key,value in vars(args).copy().items():
        if key != 'gpus' and value is not None: setattr(args,key,Path(value).resolve())
    check_preparation(args.repair_preparation)
    inputs = api('matcher_v2_v1.runtime_inputs').load_inputs(read(args.spec)); source = inputs['source']
    api('matcher_v2_v1.evaluate').check_preparation(args.preparation,source)
    if read(args.spec)['module'] != 'matcher': ensure_evaluation_binding(source)
    launcher = api('curriculum_training_v1.launcher'); original = api('matcher_v2_v1.pipeline')
    require(args.python.is_file() and not args.out.exists(), 'exclusive pipeline/runtime required')
    world = read(args.spec)['topology']['world_size']
    require(len(args.gpus)==world, 'unchanged explicit topology required')
    train_root = (args.adopt_pipeline or args.out)/'training'
    if args.adopt_pipeline:
        # Inspect all actual returns and every successful original artifact
        # BEFORE taking ownership of any device or creating the repair root.
        module = read(args.spec)['module']; final = api('matcher_v2_v1.evaluation_controller')
        selected = {c:final.verified_export(train_root,args.spec,inputs['plan'],c)[1]
                    for c in dict.fromkeys(j['selection'] for j in final.jobs(module))}
        plan = final.freeze_plan(args.spec,bound(args.canonical_straight),args.case_plan,source)
        old_args = argparse.Namespace(**vars(args),controller_root=train_root,reuse_evaluation=args.adopt_pipeline/'evaluation')
        controller.inspect_old(old_args,inputs,selected,plan)
    devices = launcher.check_free(args.gpus,world)
    args.out.mkdir(parents=True)
    save(args.out/'pipeline_launch.json',dict(controller=launcher.identity(os.getpid()),
        execution=bound(args.spec),source_files=source_map(),repair_preparation=bound(args.repair_preparation),
        gpu_devices=devices,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        automatic_retry=False,training_code_unchanged=True,started_unix=time.time()))
    try:
        env = environment(source,args.gpus)
        if not args.adopt_pipeline:
            launcher.execute(args.out,'training',train_command(args),env,source)
            original.verify_child(args.out,'training',train_command(args))
        launcher.check_free(args.gpus,world)
        launcher.execute(args.out,'evaluation',eval_command(args,train_root),env,source)
        subprocesses = {'evaluation':original.verify_child(args.out,'evaluation',eval_command(args,train_root))}
        if not args.adopt_pipeline: subprocesses['training'] = original.verify_child(args.out,'training',train_command(args))
        controller.verify_complete(args.out/'evaluation')
        record = dict(schema='matcher-v2-terminal-repaired-pipeline/1',status='complete',arm=read(args.spec)['arm'],
            module=read(args.spec)['module'],execution=bound(args.spec),subprocesses=subprocesses,
            effective_training_controller_root=str(train_root),training_complete=bound(train_root/'controller_complete.json'),
            evaluation_complete=bound(args.out/'evaluation/evaluation_complete.json'),
            adopted_original_pipeline=str(args.adopt_pipeline) if args.adopt_pipeline else None,
            pipeline_launch_sha256=sha(args.out/'pipeline_launch.json'),mandatory_frozen_evaluation_complete=True,
            automatic_retry=False,report_delivery_still_required=True,gpu_release_requires_current_idle_check=True,
            completed_unix=time.time())
        save(args.out/'complete.json',record); verify_complete(args.out)
        api('curriculum_training_v1.checkpoint_io').write_json(args.out/'driver_status.json',record,replace=True)
        return record
    except BaseException as error:
        save(args.out/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('spec','preparation','canonical-straight','case-plan','out','python','repair-preparation'):
        p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--adopt-pipeline',type=Path)
    p.add_argument('--gpus',type=int,nargs='+',required=True)
    run(p.parse_args())


if __name__=='__main__': main()
