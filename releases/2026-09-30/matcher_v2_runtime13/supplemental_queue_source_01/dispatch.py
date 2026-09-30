"""Bounded local CPU scheduler after B1--B3 terminal training/evaluation.

No GPU probing, allocation, training, preemption, source edits or automatic retry.
Up to2 low-priority single-threaded children; B3 has priority among ready jobs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import time
import traceback

import contracts as c
import jobs

OUT=c.ROOT/'supplemental_queue_01'


def identity(pid):
    root=Path('/proc')/str(pid);stat=(root/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime=int(stat[19]),state=stat[0],
        cmdline=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())


def verify_preparation(path):
    proof=c.read(path)
    c.require(proof['status']=='passed' and proof['tests']>=20
        and proof['errors']==proof['failures']==proof['skipped']==0 and proof['source_unchanged'] is True
        and proof['source_sha256']==c.own_code() and proof['dependencies_sha256']==c.dependency_code(c.ROOT)
        and proof['cuda_initialized'] is False and proof['runtime_preflight']['status']=='passed'
        and proof['runtime_preflight']['gpu_probed'] is False,'exact source-bound remote CPU preparation required')
    return proof


def unchanged(proof):
    c.require(c.own_code()==proof['source_sha256'] and c.dependency_code(c.ROOT)==proof['dependencies_sha256'],
              'companion source changed after preparation')
    c.check_static()


def execute(job,root,ready,proof,*,popen=subprocess.Popen,verify=jobs.verify_job):
    work=root/job['id'];work.mkdir()
    try:
        unchanged(proof)
        c.require(c.bound(ready['execution']['path'])==ready['execution']
                  and c.bound(ready['pipeline_complete']['path'])==ready['pipeline_complete'],
                  'verified predecessor artifacts changed')
        c.require(c.upstream_failure(job['arm'],job['module']) is None,'upstream failed after terminal verification')
        out=work/'evaluation';values=jobs.command(job,out,ready);started=time.time()
        with (work/'child.log').open('xb') as stream:
            child=popen(values,env=c.cpu_environment(),cwd=str(c.ROOT),stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT)
            c.save(work/'launch.json',dict(job=job,command=values,process=identity(child.pid),
                upstream=ready,started_unix=started,cuda_visible_devices='',automatic_retry=False))
            code=child.wait()
        c.save(work/'return.json',dict(returncode=code,launch_sha256=c.sha(work/'launch.json'),
            elapsed_seconds=time.time()-started,automatic_retry=False))
        c.require(code==0,'supplemental child failed; preserve output, no retry')
        unchanged(proof);verified=verify(job,out,ready)
        result=dict(status='complete',job=job,actual_return=c.bound(work/'return.json'),verified=verified,
            upstream=ready,gpu_used=False,training_changed=False,completed_unix=time.time())
        c.save(work/'complete.json',result)
        return result
    except BaseException as error:
        c.save(work/'failure.json',dict(status='failed',job=job,error=repr(error),
            traceback=traceback.format_exc(),automatic_retry=False,other_jobs_not_signalled=True))
        raise


def choose_ready(pending,ready,capacity):
    c.require(0<=capacity<=2,'at most two CPU children')
    return [job for job in pending if (job['arm'],job['module']) in ready][:capacity]


def run(preparation):
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','supplemental controller must hide CUDA')
    proof=verify_preparation(preparation);unchanged(proof)
    c.require(not OUT.exists(),'exclusive supplemental queue; no duplicate/restart')
    runtime=c.load_runtime();planned=jobs.registry();OUT.mkdir()
    c.save(OUT/'launch.json',dict(controller=identity(os.getpid()),preparation=c.bound(preparation),
        source_sha256=proof['source_sha256'],dependencies_sha256=proof['dependencies_sha256'],
        jobs=planned,cpu_workers=2,tick_seconds=60,nice=os.getpriority(os.PRIO_PROCESS,0),
        gpu_used=False,training_changed=False,automatic_retry=False,started_unix=time.time()))
    pending=list(planned);active={};completed=[];failures=[];ready={};seen_failures=set()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            while pending or active:
                # Only small terminal/failure files while waiting, not training/GPU polls.
                for key in dict.fromkeys((j['arm'],j['module']) for j in pending):
                    if key in ready or key in seen_failures:continue
                    arm,module=key
                    try:
                        c.require(c.upstream_failure(arm,module) is None,'predecessor failed; no automatic retry')
                        _,root=c.paths(arm,module)
                        if (root/'complete.json').exists():
                            value=c.terminal_ready(arm,module,runtime)
                            c.require(value is not None,'terminal receipt vanished during verification')
                            ready[key]=value;c.save(OUT/(arm.lower()+'_'+module+'_ready.json'),value)
                    except BaseException as error:
                        seen_failures.add(key);detail=dict(arm=arm,module=module,error=repr(error),
                            traceback=traceback.format_exc(),upstream=c.upstream_failure(arm,module),automatic_retry=False)
                        c.save(OUT/(arm.lower()+'_'+module+'_upstream_failure.json'),detail)
                        removed=[j for j in pending if (j['arm'],j['module'])==key]
                        failures.extend(dict(job=j,upstream_failure=detail) for j in removed)
                        pending=[j for j in pending if j not in removed]
                for name,(future,job) in list(active.items()):
                    if not future.done():continue
                    del active[name]
                    try:completed.append(future.result())
                    except BaseException as error:failures.append(dict(job=job,error=repr(error),automatic_retry=False))
                for job in choose_ready(pending,ready,2-len(active)):
                    pending.remove(job)
                    active[job['id']]=(pool.submit(execute,job,OUT,ready[(job['arm'],job['module'])],proof),job)
                c.save(OUT/'status.json',dict(status='running_with_failure' if failures else 'waiting_or_running',
                    pending=[j['id'] for j in pending],active=list(active),completed=[r['job']['id'] for r in completed],
                    failures=failures,cpu_workers=2,gpu_used=False,training_changed=False),replace=True)
                if pending or active:time.sleep(60)
        c.require(not failures and len(completed)==len(planned),'supplemental queue has preserved failures')
        unchanged(proof)
        c.save(OUT/'complete.json',dict(status='complete',results=completed,job_count=len(planned),
            training_changed=False,gpu_used=False,report_delivery_still_required=True,automatic_retry=False))
    except BaseException as error:
        c.save(OUT/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
            automatic_retry=False,training_changed=False,other_jobs_not_signalled=True))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preparation',type=Path,required=True)
    args=parser.parse_args();os.nice(10);run(args.preparation)


if __name__=='__main__':main()
