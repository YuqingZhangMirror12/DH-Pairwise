"""One-shot 32-worker CPU handoff; no training, cleanup or geometry changes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

try:
    from . import heldout_reduce_plan as p, heldout_parallel_plan as scheduling
    from .heldout_adopt import check_admission
    from .heldout_reduce_run import run_role
except ImportError:
    import heldout_reduce_plan as p, heldout_parallel_plan as scheduling
    from heldout_adopt import check_admission
    from heldout_reduce_run import run_role

BASE=Path('/root/autodl-tmp/model_selection_v2_20261002')


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def resources():
    cpu=Path('/sys/fs/cgroup/cpu.max').read_text().strip().split()
    quota=float(cpu[0])/int(cpu[1]) if cpu[0]!='max' else len(os.sched_getaffinity(0))
    memory=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    used=int(Path('/sys/fs/cgroup/memory.current').read_text())
    available=int(memory)-used if memory!='max' else 0
    fs=os.statvfs(BASE);free=fs.f_bavail*fs.f_frsize
    if quota<48 or len(os.sched_getaffinity(0))<48 or available<64*1024**3 or free<20*1024**3:
        raise ValueError('not enough admitted CPU/memory/disk headroom for 32 workers')
    return dict(cpu_quota=quota,affinity_count=len(os.sched_getaffinity(0)),cgroup_memory_available=available,
                disk_available=free,workers=32,threads_per_worker=1,nice=10,cuda_visible_devices='')


def merge(root,launch,role):
    projection=read(launch['reduction_plan']['path']);generation=read(projection['original_generation_plan']['path'])
    tasks={(t['stage'],t['slot']):t for t in p.chosen_tasks(projection,generation,role)}
    components=[];refs=[]
    for shard in launch['shards']:
        if shard['role']!=role:continue
        path=root/'shard_receipts'/shard['name']/'complete.json';value=read(path)
        if value['shard']!=shard or value['status']!='complete' or value['admitted_pairs']!=90:
            raise ValueError('incomplete shard; full role must not be claimed')
        components.append(value);refs.append(p.ref(path))
    records=[r for value in components for r in value['records']]
    scheduling.validate_merged(records,tasks)
    records.sort(key=lambda r:(r['generator'],tasks[r['stage'],r['baseline_slot']]['quota_slot'],r['stage'],r['baseline_ordinal']))
    baseline={};bound={}
    for value in components:
        for k,v in value['base_audit'].items():
            if k in baseline and baseline[k]!=v:raise ValueError('baseline admission has multiple owners')
            baseline[k]=v
        for k,v in value['frozen_source_sha256'].items():
            if k in bound and bound[k]!=v:raise ValueError('worker native sources differ')
            bound[k]=v
    done=dict(schema='mixed-sim-heldout-reduced-role-build/2',role=role,status='complete',pilot_only=False,
        planned_quotas=720,admitted_pairs=1440,records=records,base_audit=baseline,
        failures=[f for value in components for f in value['failures']],shards=refs,
        reduction_plan=launch['reduction_plan'],original_generation_plan=projection['original_generation_plan'],
        frozen_source_sha256=bound,gpu_used=False,model_outputs_used=False,reserve_count=12,
        reused_groups=sum(value['reused_groups'] for value in components),
        cross_shard_duplicate_inputs=0,selection_does_not_depend_on_worker_completion_order=True)
    save(root/'build_receipts'/role/'complete.json',done)
    return p.ref(root/'build_receipts'/role/'complete.json')


def worker(root):
    launch=read(root/'controller_launch.json');began=time.time()
    try:
        with ProcessPoolExecutor(max_workers=32) as pool:
            jobs={pool.submit(run_role,(str(root),s['role'],s)):s for s in launch['shards']}
            for future in as_completed(jobs):
                try:result=future.result()
                except BaseException as exc:
                    save(root/'parallel_worker_failure.json',dict(status='failed',shard=jobs[future],error=repr(exc),
                         traceback=traceback.format_exc(),time_unix=time.time(),automatic_retry=False))
                    raise
                print(json.dumps(dict(shard=jobs[future]['name'],result=result)),flush=True)
        if any(read(root/'shard_receipts'/s['name']/'complete.json')['status']!='complete' for s in launch['shards']):
            save(root/'full_shortfall.json',dict(status='shortfall',missing_quota_is_not_complete=True))
            return 2
        roles={role:merge(root,launch,role) for role in p.ROLES}
        save(root/'parallel_worker_complete.json',dict(status='parallel_curriculum_worker_complete',roles=roles,
             started_unix=began,finished_unix=time.time(),rows=2880,workers=32))
        return 0
    except BaseException as exc:
        path=root/'parallel_worker_failure.json'
        if not path.exists():save(path,dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),
                                           time_unix=time.time(),automatic_retry=False))
        raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--worker',action='store_true')
    args=ap.parse_args();root=args.root.resolve();old=BASE/'curriculum_build_04'
    if root!=BASE/'curriculum_build_05':raise ValueError('dedicated parallel output root required')
    if args.worker:return worker(root)
    if root.exists():raise ValueError('parallel build already exists; never duplicate')
    resource=resources()
    transition=BASE/'parallel_transition_01/retired.json';retired=read(transition)
    if retired['status']!='retired_for_user_requested_cpu_parallelism':raise ValueError('prior role runners not retired')
    paused=read(transition.with_name('paused.json'))
    for item in paused['processes']:
        proc=Path('/proc')/str(item['pid'])
        if proc.exists():
            stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
            if int(stat[19])==item['starttime'] and stat[0]!='Z':raise ValueError('old exact process still active')
    previous=read(old/'controller_launch.json')
    projection=read(previous['reduction_plan']['path'])
    if p.ref(previous['reduction_plan']['path'])!=previous['reduction_plan']:raise ValueError('reduction quota changed')
    generation=read(projection['original_generation_plan']['path'])
    if p.ref(projection['original_generation_plan']['path'])!=projection['original_generation_plan']:raise ValueError('original plan changed')
    sources=read(generation['source_plan_path']);p.validate(projection,generation,sources)
    shards=scheduling.make_shards(projection);scheduling.validate_shards(shards,projection,generation)
    check_admission(read(previous['original_pilot_admission']['path']))
    source_dir=Path(__file__).resolve().parent
    files=('heldout_parallel_build.py','heldout_parallel_plan.py','heldout_reduce_plan.py','heldout_reduce_run.py',
           'heldout_run.py','heldout_augment.py','heldout_plan.py','heldout_extend_plan.py','heldout_adopt.py')
    bindings=dict(previous['bindings']);bindings.update({str(source_dir/n):p.sha(source_dir/n) for n in files})
    for path in (transition,transition.with_name('paused.json'),old/'controller_launch.json'):
        bindings[str(path)]=p.sha(path)
    for path,sha in bindings.items():
        if p.sha(path)!=sha:raise ValueError('bound input changed: '+path)
    root.mkdir();save(root/'reduction_plan.json',projection)
    if p.sha(root/'reduction_plan.json')!=previous['reduction_plan']['sha256']:raise ValueError('projection changed')
    os.environ['PYTHONPATH']=previous['frozen_runtime']
    launch=dict(previous,schema='mixed-heldout-parallel-controller/1',pid=os.getpid(),started_unix=time.time(),
        bindings=bindings,reduction_plan=p.ref(root/'reduction_plan.json'),
        previous_reduced_build=str(old),reuse_roots=[str(BASE/('curriculum_build_'+n)) for n in ('04','03','02')],
        workers=32,threads_per_worker=1,shards=shards,resource_admission=resource,parallel_admission_checked=True,
        disjoint_quota_write_ownership=True,worker_completion_order_does_not_select_samples=True,
        uncommitted_inflight_work_may_repeat=True,committed_pixels_regenerated=False)
    save(root/'controller_launch.json',launch)
    command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--worker']
    began=time.time();result=subprocess.run(command,cwd=root,env=os.environ.copy())
    save(root/'full_actual_return.json',dict(command=command,returncode=result.returncode,bindings=bindings,
                                          started_unix=began,finished_unix=time.time()))
    if result.returncode:return result.returncode
    completed=read(root/'parallel_worker_complete.json')
    for path,sha in bindings.items():
        if p.sha(path)!=sha:raise ValueError('bound source changed during parallel build')
    save(root/'controller_complete.json',dict(status='reduced_curriculum_complete_pending_combined_release_audit',
        finished_unix=time.time(),bindings=bindings,desired_curriculum_pairs_per_role=1440,
        desired_total_pairs_per_role=1600,roles=completed['roles'],strict_subsets=launch['strict_subsets'],
        no_training=True,no_model_selection=True,no_gpu=True,old_outputs_changed=False,parallel_workers=32))
    return 0


if __name__=='__main__':raise SystemExit(main())
