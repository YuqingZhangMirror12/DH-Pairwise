"""Experiment3 one-shot queue adapter; does not modify the live dispatcher."""
import argparse
import importlib.util
from pathlib import Path
import time
import traceback

PREPARED=Path('/root/autodl-tmp/aggressive_binary_20260927')
FORMAL=Path('/root/autodl-tmp/s7_aggressive_binary_v17_20260928')
QUEUE=Path('/root/autodl-tmp/scorer_queue_20260928')
TASK='aggressive_scratch'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def check_dispatched(common):
    # The dispatcher persists each launch before starting another task, but its
    # combined driver_status is written at the end of the loop. Bind launches
    # and real process identities, avoiding a stale-status race on three free lanes.
    for name in ('binary_patch','binary_stats'):
        files=list((QUEUE/'launches').glob('*_'+name+'.json'))
        if len(files)!=1:raise ValueError('higher-priority lightweight dispatch missing or duplicated')
        record=common.read(files[0]);work=Path(record['work'])
        if record.get('operation')!=name or (work/'failure.json').exists():
            raise ValueError('higher-priority lightweight experiment failed')
        complete=(work/'complete.json').exists() and common.read(work/'complete.json').get('status')=='complete'
        if not complete and not common.live(record['identity']):
            raise ValueError('higher-priority lightweight process not live or complete')
    if (QUEUE/'launches/joint_e32.json').exists():
        raise ValueError('joint fine-tune cannot precede the registered new-data experiment')


def verified_evaluation(result):
    wanted={(c,s) for c in ('sim','real') for s in ('sim_test_aggressive','dunhuang_cv','turufan')}
    jobs=result.get('jobs',[])
    if (result.get('status')!='complete' or not result.get('all_six_populations_verified')
            or result.get('fixed_case_evaluations')!=22 or len(jobs)!=6
            or {(j.get('selection_kind'),j.get('split')) for j in jobs}!=wanted
            or any(j.get('status')!='complete' or j.get('returncode')!=0
                   or j.get('verified',{}).get('status')!='passed' for j in jobs)):
        raise ValueError('all six experiment3 frozen evaluations and fixed cases must verify')


def run(args,common):
    work=Path(args.work)
    if work!=QUEUE/'work'/TASK:raise ValueError('registered queue work directory required')
    admission=common.read(QUEUE/'admissions'/(TASK+'.json'))
    if admission.get('task')!=TASK or admission.get('status')!='ready':raise ValueError('experiment3 not admitted')
    for file,digest in admission['files_sha256'].items():
        if common.sha(file)!=digest:raise ValueError('registered queue/controller/preparation changed')
    check_dispatched(common)
    if not common.free(args.gpus):raise ValueError('assigned lane is occupied; no interruption')
    work.mkdir(parents=True,exist_ok=False)
    try:
        launcher=PREPARED/'training_source_03/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/aggressive_binary_v1/launch_training.py'
        common.launch_wait([common.PYTHON,str(launcher),'--root',str(FORMAL),'--gpus',args.gpus,
            '--queue-release',args.release],work/'training_launch.log',common.env())
        controller=common.read(FORMAL/'controller_launch.json')['controller']
        common.wait_terminal(FORMAL/'driver.json',controller,[FORMAL/'controller_failure.json'],
            'training_complete_evaluation_pending')
        if not common.free(args.gpus):raise ValueError('completed training has not released its devices')
        common.launch_wait([common.PYTHON,str(PREPARED/'aggressive_binary_eval_queue_v1/launch_evaluation.py'),
            '--root',str(FORMAL),'--prepared',str(PREPARED),'--variant','patch','--gpus',args.gpus],
            work/'evaluation_launch.log',common.env())
        out=FORMAL/'evaluation_patch_01'
        identity=common.read(out/'controller_launch.json')['controller']
        result=common.wait_terminal(out/'evaluation_complete.json',identity,
            [out/'failure.json',out/'controller_failure.json',out/'launch_failure.json'],'complete')
        verified_evaluation(result)
        complete=dict(status='complete',operation=TASK,gpus=args.gpus,
            training_root=str(FORMAL/'formal_scratch_aggressive'),
            training_terminal_sha256=common.sha(FORMAL/'formal_scratch_aggressive/training_complete.json'),
            evaluation_complete=str(out/'evaluation_complete.json'),evaluation_sha256=common.sha(out/'evaluation_complete.json'),
            completed_unix=time.time(),automatic_retries=0)
        common.save(work/'complete.json',complete)
    except BaseException as error:
        common.save(work/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retries=0))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('gpus','release','work'):p.add_argument('--'+name,required=True)
    args=p.parse_args();common=load(QUEUE/'source/common.py','night_queue_io')
    run(args,common)


if __name__=='__main__':main()
