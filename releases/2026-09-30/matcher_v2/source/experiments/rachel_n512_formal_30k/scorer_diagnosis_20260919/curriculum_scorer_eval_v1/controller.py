"""Six frozen evaluations for one completed curriculum lightweight Scorer.

Accept explicit released GPUs; never search/preempt, refit, or retry. The
equal-budget endpoint remains available but is not an extra automatic job.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.execution import load_inputs
from ..curriculum_training_v1.launcher import check_free, identity
from ..curriculum_training_v1.model_adapter import require
from ..curriculum_training_v1.runtime_io import read
from ..curriculum_training_v1.verify_validation_preparation import bind_baseline
from .entry import check_preparation, bind_evaluation
from .population import SPLITS, freeze_plan
from .terminal import MODULES, verified_export


def jobs():
    return [dict(name=choice+'_'+split,selection=choice,split=split)
            for choice in ('sim_best','real_best') for split in SPLITS]


def command(interpreter,args,population_plan,out,job):
    require(job in jobs(), 'unregistered curriculum Scorer evaluation job')
    return [str(interpreter),'-m',__package__+'.entry','--spec',str(args.spec),
        '--population-plan',str(population_plan),'--preparation',str(args.preparation),
        '--controller-root',str(args.controller_root),'--common-source',str(args.common_source),
        '--binary-source',str(args.binary_source),'--selection',job['selection'],
        '--split',job['split'],'--device','cuda:0','--out',str(out)]


def verify_job(root,job):
    from .audit import verify_population
    require(job in jobs(), 'unregistered Scorer completion')
    root = Path(root).resolve(); out = root/job['name']
    returned = read(root/(job['name']+'_return.json')); launch_path = root/(job['name']+'_launch.json')
    launch = read(launch_path)
    require(returned['returncode'] == 0 and returned['launch_sha256'] == file_sha(launch_path)
            and launch['job'] == job, 'job success/launch binding differs')
    values = launch['command']
    for flag, value in (('--selection',job['selection']),('--split',job['split']),('--out',str(out))):
        require(values.count(flag) == 1 and values[values.index(flag)+1] == value, 'job command differs: '+flag)
    audit = verify_population(out); origin = read(out/'evaluation_complete.json')['provenance']
    controller = read(root/'controller_launch.json')
    require(origin['selection_kind'] == job['selection'] and origin['split'] == job['split']
            and origin['checkpoint_sha256'] == controller['selected_models'][job['selection']]['checkpoint_sha256']
            and origin['population_plan_sha256'] == controller['population_plan_sha256']
            and origin['preparation_sha256'] == controller['preparation_sha256'],
            'completed job model/population/source differs')
    require(read(out/'independent_artifact_audit.json') == audit, 'child independent audit missing/different')
    return dict(job=job,return_sha256=file_sha(root/(job['name']+'_return.json')),audit=audit,provenance=origin)


def execute_queue(root,devices,environment,build_command,*,free_check=check_free,verify=verify_job,registered_jobs=None):
    registered_jobs = jobs() if registered_jobs is None else registered_jobs
    require(registered_jobs and len({j['name'] for j in registered_jobs}) == len(registered_jobs), 'unique nonempty job registry required')
    root = Path(root); pending = list(registered_jobs); active = {}; completed = []; failures = []
    try:
        while pending or active:
            for gpu in devices:
                if failures or not pending or gpu in active: continue
                free_check([gpu],1)
                job = pending.pop(0); name = job['name']; out = root/name
                require(not out.exists() and not (root/(name+'_launch.json')).exists(), 'preserve prior job attempt')
                values = build_command(job,out); log = (root/(name+'.log')).open('xb')
                child = subprocess.Popen(values,cwd=environment['PYTHONPATH'],
                    env=dict(environment,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True)
                try: process = identity(child.pid)
                except FileNotFoundError: process = dict(pid=child.pid,already_exited=True)
                write_json(root/(name+'_launch.json'),dict(job=job,command=values,process=process,
                    gpu=gpu,automatic_retry=False,started_unix=time.time()))
                active[gpu] = dict(child=child,log=log,job=job,start=time.time())
            for gpu,state in list(active.items()):
                code = state['child'].poll()
                if code is None: continue
                state['log'].close(); job = state['job']; name = job['name']; del active[gpu]
                write_json(root/(name+'_return.json'),dict(returncode=code,
                    launch_sha256=file_sha(root/(name+'_launch.json')),elapsed_seconds=time.time()-state['start'],
                    automatic_retry=False))
                if code != 0: failures.append(dict(job=name,returncode=code))
                else:
                    try: completed.append(verify(root,job))
                    except Exception as error: failures.append(dict(job=name,error=repr(error)))
            write_json(root/'driver_status.json',dict(status='draining_after_failure' if failures else 'running',
                active={str(gpu):dict(job=s['job']['name'],pid=s['child'].pid) for gpu,s in active.items()},
                pending=[j['name'] for j in pending],completed=len(completed),failures=failures),replace=True)
            if failures and not active: break
            if active: time.sleep(2)
        require(not failures and not pending and len(completed) == len(registered_jobs),
                'Scorer evaluation queue incomplete: '+repr(failures))
        return completed
    finally:
        # Keep remaining child handles for diagnosis; do not cancel another
        # live job or implicitly re-use its card after a controller failure.
        if active:
            write_json(root/'live_children_after_controller_error.json',
                {str(gpu):dict(pid=s['child'].pid,job=s['job']['name']) for gpu,s in active.items()})
            for state in active.values(): state['log'].close()


def run(args):
    for key in ('spec','preparation','controller_root','case_plan','out','python','common_source','binary_source'):
        setattr(args,key,Path(getattr(args,key)).resolve())
    inputs = load_inputs(read(args.spec))
    require(inputs['plan'].record['module'] in MODULES, 'only the two curriculum heads are registered')
    require(len(args.gpus) in (1,2), 'one or two explicitly released evaluation GPUs required')
    require(args.python.is_file(), 'explicit installed Python required')
    check_preparation(args.preparation,args.common_source,args.binary_source,inputs['baseline'])
    selected = {choice:verified_export(args.controller_root,args.spec,inputs['plan'],choice)[1]
                for choice in ('sim_best','real_best')}
    bind_baseline(inputs['baseline']); bind_evaluation(args.common_source,args.binary_source)
    population = freeze_plan(args.spec,args.case_plan,inputs['baseline'])
    root = args.out; require(not root.exists(), 'fresh controller root required; no implicit restart')
    devices = check_free(args.gpus,len(args.gpus)); root.mkdir(parents=True)
    population_path = root/'population_plan.json'; write_json(population_path,population)
    package_root = Path(__file__).resolve().parents[len(__package__.split('.'))]
    environment = dict(os.environ,PYTHONPATH=str(package_root),CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    write_json(root/'controller_launch.json',dict(controller=identity(os.getpid()),module=inputs['plan'].record['module'],
        gpu_devices=devices,training_controller_root=str(args.controller_root),selected_models=selected,
        population_plan_sha256=file_sha(population_path),preparation_sha256=file_sha(args.preparation),
        automatic_retry=False,jobs=jobs()))
    try:
        results = execute_queue(root,args.gpus,environment,
            lambda job,out:command(args.python,args,population_path,out,job))
        complete = dict(status='complete',schema='curriculum-scorer-controller-complete/1',jobs=results,
            job_count=len(results),frozen_model_evaluations=2,population_plan_sha256=file_sha(population_path),
            module=inputs['plan'].record['module'],automatic_retry=False,endpoint_automatically_evaluated=False)
        write_json(root/'evaluation_complete.json',complete)
        write_json(root/'driver_status.json',dict(status='complete',completed=len(results),active={},pending=[]),replace=True)
        return complete
    except BaseException as error:
        write_json(root/'controller_failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
            automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('spec','preparation','controller-root','case-plan','out','python','common-source','binary-source'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--gpus',type=int,nargs='+',required=True)
    run(parser.parse_args())


if __name__=='__main__': main()
