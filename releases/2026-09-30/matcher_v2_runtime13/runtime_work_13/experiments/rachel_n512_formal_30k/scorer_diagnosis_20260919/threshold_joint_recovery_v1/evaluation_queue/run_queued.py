"""Last-priority E32 joint job plus its frozen-control and final evaluation.

The server owns this bounded lifecycle. It never polls via SSH, signals existing
jobs, changes training source, retries, or moves a higher-priority experiment.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import time
import traceback
from contracts import (PREPARED,POST,FORMAL,CONTROL,QUEUE,REFERENCE,REL,SPLITS,TASKS,
    read,save,sha,inventory,evaluator_command,same_model,validate_trace)

TASK='joint_e32'
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def check_priority(common):
    for task in ('binary_patch','binary_stats','aggressive_scratch'):
        pattern='*_'+task+'.json' if task.startswith('binary_') else task+'.json'
        launches=list((QUEUE/'launches').glob(pattern))
        if len(launches)!=1:raise ValueError('missing/duplicate higher-priority launch: '+task)
        record=read(launches[0]);work=Path(record['work'])
        if record.get('operation')!=task or (work/'failure.json').exists():raise ValueError('higher-priority failure')
        done=(work/'complete.json').exists() and read(work/'complete.json').get('status')=='complete'
        if not done and not common.live(record['identity']):raise ValueError('higher-priority process not live or complete')

def validate_admission():
    record=read(QUEUE/'admissions'/(TASK+'.json'))
    if record.get('task')!=TASK or record.get('status')!='ready':raise ValueError('joint task not admitted')
    for path,digest in record['files_sha256'].items():
        if sha(path)!=digest:raise ValueError('bound task/preparation/source changed')
    return record

def prior_release(path):
    # Reuse the existing, tested per-lane completion/hash checks without changing
    # the 116-test joint training controller's immutable admission implementation.
    control=load('/root/autodl-tmp/binary_scorer_20260927/controller_source_03/launch_training.py','release_validator')
    release=control.validate_queue_release(path)
    if release['schema']!='binary-released-lane/2':raise ValueError('current parallel lane proof required')
    formal=Path(release['branches'][0]['formal_root']);arm=formal.name.removeprefix('formal_')
    evaluation=formal.parent/('evaluation_'+arm+'_20260928/evaluation_complete.json')
    if sha(evaluation)!=release['evaluation_sha256']:raise ValueError('original released evaluation changed')
    return evaluation

def worker_command(operation,kind,out,extra=()):
    return [str(Path(__file__).with_name('worker.py')),operation,'--kind',kind,'--out',str(out),*extra]

def cpu_worker(common,operation,kind,out,log,extra=()):
    common.launch_wait([common.PYTHON,*worker_command(operation,kind,out,extra)],log,common.env())
    result=read(out)
    if result.get('status')!='passed':raise ValueError('CPU gate/audit failed')
    return result

def verify_frozen_sim(common,out):
    """Retain and verify the earlier three evaluations; do not infer again."""
    formal=CONTROL/'formal_scratch_fixed'
    if not (formal/'training_complete.json').exists():
        launched=read(CONTROL/'formal_launch_scratch_fixed.json')
        common.wait_terminal(formal/'training_complete.json',launched['job'],
            [formal/'failure.json',formal/'scorer/failure.json',CONTROL/'failure_scratch_fixed_controller.json'],'training_complete')
    post=CONTROL/'postprocess_arms_v1'
    old=load(post/'launch_arm_evaluation.py','frozen_control_evaluation_checks')
    old.training_terminal(CONTROL,'scratch_fixed')
    selected=old.gates(CONTROL,post,'scratch_fixed',post/'gates/cpu_scratch_fixed.json',post/'gates/cuda_scratch_fixed.json')
    path=CONTROL/'evaluation_scratch_fixed_20260928/evaluation_complete.json'
    complete=read(path);jobs=complete.get('jobs',[])
    if (complete.get('status')!='complete' or len(jobs)!=3 or {j['split'] for j in jobs}!=set(SPLITS)
            or any(j.get('returncode')!=0 or j.get('status')!='complete' for j in jobs)):
        raise ValueError('completed original frozen SIM evaluation required')
    for job in jobs:old.verify_result(Path(job['out']),'scratch_fixed',job['split'],selected['selected']['scratch_fixed'])
    save(out/'original_frozen_sim.json',dict(status='complete',evaluation_complete=str(path),
        sha256=sha(path),jobs=jobs,reused_without_new_inference=True))

def ready_task(pending,completed,control_ready,reuse_joint):
    for task in pending:
        kind,choice,split=task
        if kind=='frozen' and not control_ready:continue
        if kind=='joint' and choice=='real' and reuse_joint and ('joint','sim',split) not in completed:continue
        return task
    return None

def evaluate_all(common,gpus,out):
    """Two local slots: control reselection overlaps independent joint tests."""
    out=Path(out);out.mkdir(exist_ok=False);(out/'logs').mkdir()
    verify_frozen_sim(common,out)
    joint=cpu_worker(common,'selected','joint',out/'selected_joint.json',out/'logs/selected_joint.log')
    devices=tuple(int(i) for i in gpus.split(','))
    if len(devices)!=2 or len(set(devices))!=2 or not common.free(gpus):raise ValueError('two released devices required')
    for kind,gpu in zip(('joint','frozen'),devices):
        receipt=out/('trace_'+kind+'.json')
        common.launch_wait([common.PYTHON,*worker_command('trace',kind,receipt,('--device','cuda:0'))],
            out/'logs'/('trace_'+kind+'.log'),common.env(gpu))
        validate_trace(read(receipt),kind)
    reuse_joint=same_model(joint['selected']['sim'],joint['selected']['real'])
    plan=dict(schema='threshold-joint-comparison-queue/1',tasks=TASKS,
        joint_same_model_prediction_reuse=reuse_joint,gpus=gpus,queue_sha256=inventory(Path(__file__).parent),
        preparation_sha256=sha(POST/'preparation.json'),selected_joint_sha256=sha(out/'selected_joint.json'),
        original_frozen_sim_reused=True,automatic_retries=0,created_unix=time.time())
    save(out/'plan.json',plan)
    active={};records=[];pending=list(TASKS);completed=set();failures=[];control_ready=False

    def start(command,gpu,name,destination,task=None,reused=False):
        if not common.free(str(gpu)):raise ValueError('assigned evaluation device occupied')
        log=(out/'logs'/(name+'.log')).open('xb')
        proc=subprocess.Popen(command,env=common.env('' if reused else gpu),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        record=dict(name=name,task=task,out=str(destination),gpu=gpu,identity=common.identity(proc.pid),
            status='running',command=command,reused_predictions=reused,started_unix=time.time())
        save(out/(name+'_launch.json'),record);records.append(record);active[gpu]=(proc,log,record)

    start([common.PYTHON,*worker_command('reselect','frozen',out/'control_reselection',('--device','cuda:0'))],
        devices[0],'control_reselection',out/'control_reselection')
    while active or pending:
        for gpu,(proc,log,record) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();record.update(returncode=code,finished_unix=time.time())
            save(out/(record['name']+'_exit.json'),dict(returncode=code,identity=record['identity']))
            try:
                if code:raise RuntimeError('evaluation worker failed; no retry')
                if record['task'] is None:
                    # Loads every archived epoch proof and recomputes the chosen
                    # winner before any control TEST/REAL-TEST forward is allowed.
                    result=cpu_worker(common,'selected','frozen',out/'selected_frozen.json',out/'logs/selected_frozen.log')
                    control_ready=True;record['verified']=result
                else:
                    kind,choice,split=record['task']
                    result=cpu_worker(common,'verify',kind,out/(record['name']+'_verified.json'),
                        out/'logs'/(record['name']+'_verify.log'),('--choice',choice,'--split',split,
                            '--selected-gate',str(out/('selected_'+kind+'.json')),'--job',record['out']))
                    record['verified']=result;completed.add(tuple(record['task']))
                record['status']='complete'
            except Exception as error:
                record.update(status='failed',error=repr(error));failures.append(record['name'])
            del active[gpu]
        if failures:pending.clear()  # Keep existing siblings running to terminal.
        else:
            for gpu in devices:
                if gpu in active:continue
                task=ready_task(pending,completed,control_ready,reuse_joint)
                if task is None:continue
                pending.remove(task);kind,choice,split=task;name='_'.join(task);destination=out/name
                reused=kind=='joint' and choice=='real' and reuse_joint
                if reused:
                    command=[common.PYTHON,*worker_command('reuse',kind,destination,('--choice',choice,'--split',split,
                        '--selected-gate',str(out/'selected_joint.json'),'--source',str(out/('joint_sim_'+split))))]
                else:command=evaluator_command(common.PYTHON,kind,choice,split,destination)
                start(command,gpu,name,destination,task,reused)
        save(out/'driver_status.json',dict(status='running_with_failure' if failures else 'running',jobs=records,
            pending=pending,failures=failures,automatic_retries=0,updated_unix=time.time()))
        if active:time.sleep(10)
        elif pending:raise RuntimeError('unresolved evaluation dependency; no spin/retry')
    if failures or completed!=set(TASKS):raise RuntimeError('joint/control evaluation incomplete')
    result=dict(status='complete',jobs=records,final_evaluations=9,fixed_case_evaluations=33,
        original_frozen_sim_reused=True,all_populations_verified=True,plan_sha256=sha(out/'plan.json'),
        training_modified=False,automatic_retries=0,completed_unix=time.time())
    save(out/'evaluation_complete.json',result);save(out/'driver_status.json',result)
    return result

def run(args,common):
    work=Path(args.work)
    if work!=QUEUE/'work'/TASK:raise ValueError('registered task workspace required')
    validate_admission();check_priority(common)
    release=prior_release(args.release)
    if not common.free(args.gpus):raise ValueError('assigned lane occupied; no interruption')
    work.mkdir(parents=True,exist_ok=False)
    try:
        launcher=PREPARED/'threshold_joint_v1/launch_training.py'
        common.launch_wait([common.PYTHON,str(launcher),'--root',str(FORMAL),'--arm','scratch_joint',
            '--gpus',args.gpus,'--release-evaluation',str(release)],work/'training_launch.log',common.env())
        controller=read(FORMAL/'controller_launch_scratch_joint.json')
        common.wait_terminal(FORMAL/'driver_scratch_joint.json',controller,
            [FORMAL/'failure_scratch_joint_controller.json'],'training_complete_evaluation_pending')
        if not common.free(args.gpus):raise ValueError('trained job still occupies devices')
        result=evaluate_all(common,args.gpus,FORMAL/'postprocess_joint_01')
        save(work/'complete.json',dict(status='complete',operation=TASK,gpus=args.gpus,
            training_terminal_sha256=sha(FORMAL/'formal_scratch_joint/training_complete.json'),
            evaluation_complete=str(FORMAL/'postprocess_joint_01/evaluation_complete.json'),
            evaluation_sha256=sha(FORMAL/'postprocess_joint_01/evaluation_complete.json'),
            final_evaluations=result['final_evaluations'],completed_unix=time.time(),automatic_retries=0))
    except BaseException as error:
        failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retries=0)
        save(work/'failure.json',failure)
        if (FORMAL/'postprocess_joint_01').exists():save(FORMAL/'postprocess_joint_01/failure.json',failure)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('gpus','release','work'):p.add_argument('--'+name,required=True)
    a=p.parse_args();common=load(QUEUE/'source/common.py','night_queue_io');run(a,common)

if __name__=='__main__':main()
