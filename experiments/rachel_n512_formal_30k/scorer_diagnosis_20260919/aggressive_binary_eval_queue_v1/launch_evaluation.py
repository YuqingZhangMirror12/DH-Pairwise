"""One-shot binary frozen evaluation on two explicitly released GPUs.

One trained head: SIM-best/REAL-best x TEST/Dunhuang/Turufan. Never trains,
modifies a bound source, refits a threshold, retries, or interrupts other jobs.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from queue_contracts import (read,save,sha,hashes,identity,TASKS,FORMAL_ROOT,PREPARED,
    evaluator_command,validate_selected_gate,validate_gpu_gate)

def gpu_inventory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader,nounits'],text=True,timeout=15)
    result={}
    for row in rows.splitlines():
        index,uuid,name=[s.strip() for s in row.split(',',2)];result[int(index)]=dict(uuid=uuid,name=name)
    return result
def gpu_indices(value):
    indices=tuple(int(x) for x in value.split(','))
    if len(indices)!=2 or len(set(indices))!=2 or min(indices)<0:raise ValueError('two distinct released GPUs required')
    return indices
def assert_free(indices,inventory):
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=15)
    occupied={row.split(',')[-1].strip() for row in rows.splitlines() if row.strip()}
    if any(i not in inventory or inventory[i]['uuid'] in occupied for i in indices):
        raise ValueError('requested GPU occupied/unavailable; no interruption allowed')
def environment(gpu=None):
    return dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),PYTHONDONTWRITEBYTECODE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
def verify_preparation(prepared):
    receipt=read(Path(prepared)/'queue_preparation_remote.json')
    if (receipt.get('schema')!='aggressive-binary-evaluation-queue-preparation/1' or receipt.get('status')!='passed'
            or receipt.get('verified_variants')!=['patch'] or receipt.get('errors')!=0
            or receipt.get('failures')!=0 or receipt.get('skipped')!=0 or receipt.get('tests',0)<=0
            or receipt.get('source_files_unchanged') is not True or receipt.get('gpu_tasks_started') is not False
            or receipt.get('real_inference_performed') is not False
            or receipt.get('queue_python_sha256')!=hashes(Path(__file__).parent)):
        raise ValueError('both binary queue CPU preparations required')
    evaluation=read(Path(prepared)/'evaluation_preparation_remote.json')
    for key in ('adapter_python_sha256','binary_python_sha256','common_python_sha256','training_source_sha256','real_plan_sha256','fixed_case_plan_sha256'):
        if receipt.get('source_bindings',{}).get(key)!=evaluation.get(key):raise ValueError('queue/evaluator binding differs')
    return receipt
def worker_command(args,operation,out,extra=()):
    return [sys.executable,str(Path(__file__).with_name('worker.py')),operation,'--root',args.root,
            '--prepared',args.prepared,'--variant',args.variant,'--out',str(out),*extra]
def execute_worker(args,operation,out,log,gpu=None,extra=()):
    command=worker_command(args,operation,out,extra)
    with Path(log).open('xb') as stream:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=environment(gpu))
        launch=dict(command=command,identity=identity(child.pid),gpu=gpu,started_unix=time.time())
        save(Path(out).with_name(Path(out).stem+'_launch.json'),launch)
        code=child.wait()
    save(Path(out).with_name(Path(out).stem+'_exit.json'),dict(returncode=code,identity=launch['identity'],finished_unix=time.time()))
    if code!=0:raise RuntimeError(operation+' failed; no automatic retry')
    return read(out)

def driver(args):
    root=Path(args.root);out=Path(args.out);indices=gpu_indices(args.gpus)
    with (out/'controller.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_preparation(args.prepared);plan=read(out/'plan.json')
        if plan['queue_python_sha256']!=hashes(Path(__file__).parent):raise ValueError('registered queue source changed')
        selected=validate_selected_gate(read(out/'selected_models.json'),args.variant,root)
        inventory=gpu_inventory()
        if {str(i):inventory.get(i) for i in indices}!=plan['gpu_inventory']:raise ValueError('GPU inventory changed')
        assert_free(indices,inventory)
        # This is an actual disposable CUDA no-op recording gate, not training.
        gate=execute_worker(args,'trace',out/'gpu_trace_gate.json',out/'logs/gpu_trace_gate.log',
                            gpu=indices[0],extra=('--device','cuda:0'))
        validate_gpu_gate(gate,args.variant,selected['source_bindings'])
        pending=list(TASKS);active={};jobs=[];failures=[];started=time.time()
        while pending or active:
            for gpu,(process,log,record) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close();record.update(returncode=code,finished_unix=time.time())
                save(out/(record['task']+'_exit.json'),dict(returncode=code,identity=record['identity'],finished_unix=time.time()))
                try:
                    if code!=0:raise RuntimeError('frozen evaluator returned '+str(code))
                    proof=execute_worker(args,'verify-job',out/(record['task']+'_verified.json'),
                        out/'logs'/(record['task']+'_verify.log'),extra=('--selection',record['selection_kind'],
                            '--split',record['split'],'--job',record['out'],'--selected-gate',str(out/'selected_models.json')))
                    if proof['status']!='passed':raise ValueError('frozen outputs did not verify')
                    record.update(status='complete',verified=proof)
                except Exception as error:
                    record.update(status='failed',error=repr(error));failures.append(dict(task=record['task'],error=repr(error)))
                del active[gpu]
            if failures:pending.clear()  # Existing siblings finish, no new job/retry.
            else:
                for gpu in indices:
                    if gpu in active or not pending:continue
                    assert_free((gpu,),inventory)
                    choice,split=pending.pop(0);task=choice+'_'+split;destination=out/task
                    if destination.exists():raise FileExistsError('preserve existing frozen evaluation '+str(destination))
                    command=evaluator_command(sys.executable,root,args.prepared,destination,args.variant,choice,split)
                    log=(out/'logs'/(task+'.log')).open('xb')
                    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=environment(gpu))
                    record=dict(task=task,variant=args.variant,selection_kind=choice,split=split,gpu=gpu,
                        out=str(destination),command=command,identity=identity(process.pid),status='running',started_unix=time.time())
                    active[gpu]=(process,log,record);jobs.append(record);save(out/(task+'_launch.json'),record)
            save(out/'driver_status.json',dict(status='running_with_failure' if failures else 'running',
                controller=identity(os.getpid()),jobs=jobs,pending=pending,failures=failures,
                automatic_retries=0,training_modified=False,updated_unix=time.time()))
            if active:time.sleep(10)
        if failures:
            save(out/'failure.json',dict(status='failed',failures=failures,jobs=jobs,automatic_retries=0))
            raise RuntimeError('evaluation incomplete; failed outputs preserved')
        if len(jobs)!=6 or {(r['selection_kind'],r['split']) for r in jobs}!=set(TASKS):
            raise ValueError('six unique completed jobs required')
        validate_selected_gate(selected,args.variant,root)
        complete=dict(status='complete',variant=args.variant,jobs=jobs,plan_sha256=sha(out/'plan.json'),
            all_six_populations_verified=True,fixed_case_evaluations=22,
            training_modified=False,automatic_retries=0,elapsed_seconds=time.time()-started,finished_unix=time.time())
        save(out/'driver_status.json',complete);save(out/'evaluation_complete.json',complete)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--prepared',default=str(PREPARED))
    p.add_argument('--variant',choices=('patch',),required=True);p.add_argument('--gpus',required=True)
    p.add_argument('--driver',action='store_true');args=p.parse_args()
    if Path(args.root).resolve()!=FORMAL_ROOT or Path(args.prepared).resolve()!=PREPARED:
        raise ValueError('only registered binary formal/preparation roots allowed')
    args.out=str(FORMAL_ROOT/('evaluation_'+args.variant+'_02'));out=Path(args.out)
    if args.driver:
        try:driver(args)
        except BaseException as error:
            save(out/'controller_failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
                pid=os.getpid(),automatic_retries=0,training_modified=False,time_unix=time.time()))
            raise
        return
    verify_preparation(args.prepared);indices=gpu_indices(args.gpus)
    formal=FORMAL_ROOT/('formal_scratch_aggressive')
    if (read(formal/'training_complete.json').get('status')!='training_complete'
            or any((formal/f).exists() for f in ('failure.json','failure_matcher.json','failure_scorer.json','matcher/failure.json','scorer/failure.json'))):
        raise ValueError('head must actually finish before frozen evaluation')
    out.mkdir(exist_ok=False);(out/'logs').mkdir()
    try:
        selected=execute_worker(args,'selected',out/'selected_models.json',out/'logs/selected_models.log')
        validate_selected_gate(selected,args.variant,FORMAL_ROOT)
        inventory=gpu_inventory();assert_free(indices,inventory)
        save(out/'plan.json',dict(schema='aggressive-binary-frozen-evaluation-queue/1',variant=args.variant,tasks=TASKS,
            root=args.root,prepared=args.prepared,selected=selected['selected'],queue_python_sha256=hashes(Path(__file__).parent),
            gpu_inventory={str(i):inventory[i] for i in indices},source_bindings=selected['source_bindings'],
            queue_preparation_sha256=sha(PREPARED/'queue_preparation_remote.json'),automatic_retries=0,
            training_modified=False,created_unix=time.time()))
        command=[sys.executable,str(Path(__file__).resolve()),'--root',args.root,'--prepared',args.prepared,
                 '--variant',args.variant,'--gpus',args.gpus,'--driver']
        with (out/'logs/controller.log').open('xb') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=environment(),start_new_session=True)
        receipt=dict(status='dispatched_not_claimed_complete',controller=identity(child.pid),command=command,
                     variant=args.variant,out=str(out),created_unix=time.time())
        save(out/'controller_launch.json',receipt);print(receipt)
    except BaseException as error:
        save(out/'launch_failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
            automatic_retries=0,training_modified=False,time_unix=time.time()));raise

if __name__=='__main__':main()
