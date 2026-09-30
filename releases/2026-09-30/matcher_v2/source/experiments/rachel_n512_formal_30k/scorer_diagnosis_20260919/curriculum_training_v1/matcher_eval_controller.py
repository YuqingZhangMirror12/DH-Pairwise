"""Evaluate one completed C/M Matcher on explicitly released GPUs.

Two registered exports x four fixed populations. No auto retry, no scanning
for an idle-looking GPU, no Scorer experiment and no model selection here.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback

from .checkpoint_io import file_sha,write_json
from .execution import load_inputs
from .launcher import check_free,identity
from .matcher_entry import check_preparation
from .matcher_population import SPLITS,freeze_plan
from .matcher_run import verify_population
from .matcher_terminal import verified_export
from .model_adapter import require
from .runtime_io import read
from .verify_validation_preparation import bind_baseline


def jobs():
    return [dict(name=choice+'_'+split,selection=choice,split=split)
            for choice in ('sim_best','equal_budget_endpoint') for split in SPLITS]


def command(interpreter,args,population_plan,out,job):
    require(job in jobs(), 'unregistered native evaluation job')
    return [str(interpreter),'-m',__package__+'.matcher_entry','--spec',str(args.spec),
        '--population-plan',str(population_plan),'--preparation',str(args.preparation),
        '--controller-root',str(args.controller_root),'--order',args.order,
        '--selection',job['selection'],'--split',job['split'],'--device','cuda:0','--out',str(out)]


def verify_job(root,job):
    require(job in jobs(), 'unregistered native job completion')
    root=Path(root);path=root/job['name'];returned=read(root/(job['name']+'_return.json'))
    launch=root/(job['name']+'_launch.json')
    require(returned['returncode']==0 and returned['launch_sha256']==file_sha(launch)
            and read(launch)['job']==job, 'job did not successfully return')
    audit=verify_population(path);origin=read(path/'evaluation_complete.json')['provenance']
    require(origin['selection_kind']==job['selection'] and origin['split']==job['split'], 'completed wrong selected model/population')
    require(read(path/'independent_artifact_audit.json')==audit, 'child independent audit missing/different')
    return dict(job=job,return_sha256=file_sha(root/(job['name']+'_return.json')),
        audit=audit,provenance=origin)


def execute_queue(root,devices,environment,build_command,*,free_check=check_free,verify=verify_job):
    """Poll child handles locally; a SSH disconnect never retries this queue."""
    root=Path(root);pending=list(jobs());active={};completed=[];failures=[]
    try:
        while pending or active:
            for gpu in devices:
                if failures or not pending or gpu in active:continue
                free_check([gpu],1)
                job=pending.pop(0);name=job['name'];out=root/name
                require(not out.exists() and not (root/(name+'_launch.json')).exists(), 'preserve prior evaluation attempt')
                values=build_command(job,out);log=(root/(name+'.log')).open('xb')
                child=subprocess.Popen(values,cwd=environment['PYTHONPATH'],
                    env=dict(environment,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:process=identity(child.pid)
                except FileNotFoundError:process=dict(pid=child.pid,already_exited=True)
                launch=dict(job=job,command=values,process=process,gpu=gpu,automatic_retry=False,started_unix=time.time())
                write_json(root/(name+'_launch.json'),launch)
                active[gpu]=dict(child=child,log=log,job=job,start=time.time())
            for gpu,state in list(active.items()):
                code=state['child'].poll()
                if code is None:continue
                state['log'].close();job=state['job'];name=job['name']
                write_json(root/(name+'_return.json'),dict(returncode=code,
                    launch_sha256=file_sha(root/(name+'_launch.json')),elapsed_seconds=time.time()-state['start'],automatic_retry=False))
                del active[gpu]
                if code!=0:failures.append(dict(job=name,returncode=code))
                else:
                    try:completed.append(verify(root,job))
                    except Exception as error:failures.append(dict(job=name,error=repr(error)))
            write_json(root/'driver_status.json',dict(status='draining_after_failure' if failures else 'running',
                active={str(gpu):dict(job=s['job']['name'],pid=s['child'].pid) for gpu,s in active.items()},
                pending=[j['name'] for j in pending],completed=len(completed),failures=failures),replace=True)
            if failures and not active:break
            if active:time.sleep(2)
        require(not failures and not pending and len(completed)==len(jobs()), 'native evaluation queue incomplete: '+repr(failures))
        return completed
    finally:
        # Unexpected controller errors do not silently terminate children or
        # reacquire their GPUs. Persist handles for a subsequent diagnosis.
        if active:
            write_json(root/'live_children_after_controller_error.json',
                {str(gpu):dict(pid=s['child'].pid,job=s['job']['name']) for gpu,s in active.items()})
            for state in active.values():state['log'].close()


def run(args):
    for key in ('spec','preparation','controller_root','case_plan','out','python'):
        setattr(args,key,Path(getattr(args,key)).resolve())
    check_preparation(args.preparation);inputs=load_inputs(read(args.spec))
    require(args.order in ('curriculum','mixed') and inputs['plan'].record['module']=='matcher', 'native Matcher only')
    require(len(args.gpus) in (1,2), 'one or two explicitly released evaluation GPUs required')
    require(Path(args.python).is_absolute() and Path(args.python).is_file(), 'explicit installed Python required')
    # No GPU may be acquired on the strength of a process disappearing.
    for choice in ('sim_best','equal_budget_endpoint'):
        verified_export(args.controller_root,args.spec,inputs['plan'],args.order,choice)
    bind_baseline(inputs['baseline']);population=freeze_plan(args.spec,args.case_plan,inputs['baseline'])
    root=Path(args.out).resolve();require(not root.exists(), 'fresh native evaluation controller root required')
    device_info=check_free(args.gpus,len(args.gpus));root.mkdir(parents=True)
    population_path=root/'population_plan.json';write_json(population_path,population)
    package_root=Path(__file__).resolve().parents[len(__package__.split('.'))]
    environment=dict(os.environ,PYTHONPATH=str(package_root),CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    write_json(root/'controller_launch.json',dict(controller=identity(os.getpid()),order=args.order,
        gpu_devices=device_info,training_controller_root=str(args.controller_root),
        population_plan_sha256=file_sha(population_path),preparation_sha256=file_sha(args.preparation),
        automatic_retry=False,jobs=jobs()))
    try:
        results=execute_queue(root,args.gpus,environment,
            lambda job,out:command(args.python,args,population_path,out,job))
        complete=dict(status='complete',schema='curriculum-native-controller-complete/1',order=args.order,
            jobs=results,job_count=len(results),population_plan_sha256=file_sha(population_path),
            frozen_model_evaluations=2,automatic_retry=False,scorer_used=False)
        write_json(root/'evaluation_complete.json',complete)
        write_json(root/'driver_status.json',dict(status='complete',completed=len(results),active={},pending=[]),replace=True)
        return complete
    except BaseException as error:
        write_json(root/'controller_failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('spec','preparation','controller-root','case-plan','out','python'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--order',choices=('curriculum','mixed'),required=True)
    parser.add_argument('--gpus',type=int,nargs='+',required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
