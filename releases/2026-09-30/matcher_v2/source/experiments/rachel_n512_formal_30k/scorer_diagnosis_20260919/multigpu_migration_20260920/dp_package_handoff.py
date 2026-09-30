"""Explicit one-shot edge_multi DP handoff; never execute on import.

The old package worker is stopped under dispatch.lock, its leaf is interrupted,
and the new owner is committed before ONLY the old worker PID is retired.
Existing runtime receipts are immutable history, not rewritten as successes.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

try:
    from . import dynamic_pool as pool, transfer_registry as registry, lane_runner as lane
except ImportError:
    import dynamic_pool as pool
    import transfer_registry as registry
    import lane_runner as lane

SCHEMA = 'edge-multi-dp-handoff/1'
PACKAGE = 'edge_multi'
GPU_INDICES = (2, 0, 3, 6)  # Explicit spec may substitute GPU1 for GPU0; never GPU4/5.


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def option(command, name):
    if command.count(name) != 1:
        raise ValueError('require one '+name)
    index = command.index(name)
    if index+1 == len(command):
        raise ValueError('option value missing: '+name)
    return command[index+1]


def source_check(spec):
    if not spec.get('source_sha256') or spec['command'][1] not in spec['source_sha256']:
        raise ValueError('replacement runtime source SHA required')
    for path, expected in spec['source_sha256'].items():
        if not Path(path).is_absolute() or digest(path) != expected:
            raise ValueError('replacement source changed: '+path)


def require_available_ownership(plan, records, uuids):
    """No reserved GPU may be stolen; original unfinished leaves must be gated."""
    other_reserved = {gpu for name, exp in records['experiments'].items()
                      if name != PACKAGE and (exp['status']=='running' or exp.get('reservation_retained'))
                      for gpu in [exp.get('assigned_gpu')]+exp.get('reserved_gpu_uuids', [])}
    if other_reserved.intersection(uuids):
        raise ValueError('another package owns one of the requested DP GPUs')
    for uuid in uuids[1:]:
        original = next(x for x in plan['lanes'] if x['gpu_uuid']==uuid)
        status = lane.read(Path(plan['output_root'])/original['name']/'status.json')
        completed = status.get('completed_stages')
        if (type(completed) is not int or not 0 <= completed <= len(original['stages'])
                or status.get('status') not in ('complete','running','waiting_artifact')):
            raise ValueError('original GPU lane is not safely complete/delegated: '+original['name'])
        if (len(status.get('stages', [])) < completed or any(
                r.get('status')!='complete' or r.get('returncode')!=0
                for r in status['stages'][:completed])):
            raise ValueError('completed original lane prefix lacks successful receipts')
        if status['status']=='complete' and (completed!=len(original['stages']) or status.get('active_pid') is not None):
            raise ValueError('inconsistent completed lane')
        for stage in original['stages'][completed:]:
            if not stage.get('gpu', True):
                continue
            key = str((Path(plan['output_root'])/original['name']/('runtime_'+stage['name']+'.json')).resolve())
            if key not in records['stages']:
                raise ValueError('unfinished original GPU stage is not delegated: '+stage['name'])
            owner, row, delegated = registry._stage_spec(records, key)
            if (owner['status'] not in ('queued','running','complete') or row['status']=='failed'
                    or delegated['command']!=stage['command']):
                raise ValueError('original GPU delegation is not safe: '+stage['name'])


def prepare(plan_path, root, spec):
    plan, root = lane.read(plan_path), Path(root).resolve()
    if root != (Path(plan['output_root'])/'dynamic_pool').resolve():
        raise ValueError('must use the existing pool ownership registry')
    records = registry.read_registry(root)
    exp = records['experiments'][PACKAGE]
    stages = exp['stages']
    first = stages[0]
    key = pool.receipt_key(first)
    row = records['stages'][key]
    inventory = {x['index']: x['uuid'] for x in plan['inventory']}
    indices = tuple(spec.get('gpu_indices', GPU_INDICES))
    if indices not in ((2,0,3,6), (2,1,3,6)):
        raise ValueError('allowed explicit physical GPU order is [2,0,3,6] or [2,1,3,6]')
    uuids = [inventory[i] for i in indices]
    if (exp['status'] != 'running' or row['status'] != 'running'
            or first['name'] != 'edge_multi_C16_train' or len(stages) != 7
            or exp.get('active_stage') != first['name'] or exp['assigned_gpu'] != uuids[0]
            or exp.get('dp_handoff') or row.get('attempt_history')):
        raise ValueError('only the original running edge_multi training package may be handed off once')
    if (exp.get('active_pid') != row.get('pid') or exp.get('active_startticks') != row.get('startticks')
            or exp.get('worker_pid') != row.get('worker_pid') or exp.get('worker_startticks') != row.get('worker_startticks')):
        raise ValueError('package/leaf ownership identities disagree')
    command = list(spec.get('command', []))
    if (spec.get('schema') != SCHEMA or len(command) < 3 or command[0] != first['command'][0]
            or command.count('--') != 1 or spec.get('cwd') != first['cwd']
            or spec.get('env') != first.get('env', {})):
        raise ValueError('replacement must preserve Python, cwd and explicit original environment')
    separator = command.index('--')
    outer, arguments = command[:separator], command[separator+1:]
    original = first['command']
    if original[1] != '-m' or original[2] not in ('matched_only.train',
            'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.train'):
        raise ValueError('unexpected original training entrypoint')
    expected_arguments = original[3:] + ([] if '--resume' in original[3:] else ['--resume'])
    if (arguments != expected_arguments or option(outer, '--mode') != 'resume'
            or option(outer, '--module') not in ('matched_only.train', original[2])
            or option(outer, '--gpus') != ','.join(uuids)
            or Path(option(outer, '--lock-root')).resolve() != (Path(plan['output_root'])/'gpu_locks').resolve()):
        raise ValueError('DP runtime args, four physical GPUs or unchanged resume CLI differ')
    runtime_receipt = Path(option(outer, '--receipt')).resolve()
    if runtime_receipt.exists() or str(runtime_receipt) == key:
        raise ValueError('new DP runtime receipt must be unused, not the old receipt')
    # GPU6 remains safe only while all future GPU leaves are owned by the existing
    # delegated M20 package; its original wrapper waits before taking any GPU lease.
    m20 = records['experiments'].get('M20_all_tokens')
    if not m20 or m20['status'] not in ('queued', 'running'):
        raise ValueError('registered M20 delegated package required')
    for stage in m20['stages']:
        if pool.receipt_key(stage) not in records['stages']:
            raise ValueError('M20 GPU leaf is not delegated')
    require_available_ownership(plan, records, uuids)
    source_check(spec)
    output = Path(option(arguments, '--output')).resolve()
    if not output.is_dir() or not (output/'last.pt').is_file():
        raise ValueError('existing output and atomic last.pt required')
    return dict(plan=plan, root=root, experiment=deepcopy(exp), first=deepcopy(first),
                first_key=key, old_row=deepcopy(row), uuids=uuids, output=output,
                runtime_receipt=runtime_receipt, spec=deepcopy(spec))


class Runtime:
    inspect = staticmethod(pool.identity)
    pid = staticmethod(os.getpid)
    signal = staticmethod(os.kill)
    signal_group = staticmethod(os.killpg)
    popen = staticmethod(subprocess.Popen)
    sleep = staticmethod(time.sleep)
    monotonic = staticmethod(time.monotonic)
    foreign = staticmethod(pool.device.foreign_pids)


def exact_process(rt, pid, ticks, *, alive=True):
    value = rt.inspect(pid)
    if value is not None and value['startticks'] != ticks:
        raise RuntimeError('PID identity reused: '+str(pid))
    if alive and (value is None or value['state'] in ('Z','X')):
        raise RuntimeError('required owner/leaf is no longer alive: '+str(pid))
    return value


def wait_for(rt, pid, ticks, predicate, timeout):
    end = rt.monotonic()+timeout
    while True:
        value = exact_process(rt, pid, ticks, alive=False)
        if predicate(value):
            return value
        if rt.monotonic() >= end:
            raise TimeoutError('bounded process transition timed out: '+str(pid))
        rt.sleep(.1)


def require_old_owner(records, context):
    old = context['experiment']
    exp = records['experiments'][PACKAGE]
    row = records['stages'][context['first_key']]
    keys = ('worker_pid','worker_startticks','active_pid','active_startticks','active_stage','assigned_gpu')
    if (exp['status'] != 'running' or row['status'] != 'running'
            or any(exp.get(k) != old.get(k) for k in keys) or exp['stages'] != old['stages']):
        raise RuntimeError('old ownership/stage changed; refusing handoff')
    return exp, row


class Leases:
    """Parent-owned flock descriptions inherited by the single DP child."""
    def __init__(self, root, rt):
        self.root, self.rt, self.handles = Path(root), rt, {}

    def acquire(self, uuid):
        self.root.mkdir(parents=True, exist_ok=True)
        handle = (self.root/(uuid+'.lock')).open('a+')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            foreign = self.rt.foreign(uuid)
            if foreign:
                raise RuntimeError('DP GPU has a foreign compute process: '+str((uuid,foreign)))
        except BaseException:
            handle.close()
            raise
        self.handles[uuid] = handle

    def descriptor_records(self, uuids):
        return [dict(uuid=u, path=str((self.root/(u+'.lock')).resolve()), fd=self.handles[u].fileno()) for u in uuids]

    def close(self):
        # Do not LOCK_UN inherited file descriptions: a surviving DP child must
        # retain its lease even if this parent is terminated unexpectedly.
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def execute(plan_path, root, spec, *, runtime=None, interrupt_timeout=120):
    rt = runtime or Runtime()
    context = prepare(plan_path, root, spec)
    root, plan, old, first = context['root'], context['plan'], context['experiment'], context['first']
    own = exact_process(rt, rt.pid(), rt.inspect(rt.pid())['startticks'])
    handoff_root = root/'experiments'/PACKAGE/'dp_handoff_v1'
    handoff_root.mkdir(parents=True, exist_ok=False)
    receipt = handoff_root/'handoff.json'
    state = dict(schema=SCHEMA,status='preparing',pid=own['pid'],startticks=own['startticks'],
                 old_worker=dict(pid=old['worker_pid'],startticks=old['worker_startticks']),
                 old_leaf=dict(pid=old['active_pid'],startticks=old['active_startticks']),
                 gpu_uuids=context['uuids'],old_runtime_receipt=option(context['old_row']['command'],'--receipt'),
                 new_runtime_receipt=str(context['runtime_receipt']),started_at_unix=time.time())
    lane.save(receipt,state)
    stopped = interrupted = adopted = False
    child = None
    child_identity = None
    leases = Leases(Path(plan['output_root'])/'gpu_locks',rt)
    try:
        with registry.locked_registry(root) as records:
            exp,row = require_old_owner(records,context)
            require_available_ownership(plan, records, context['uuids'])
            # Coordinator chooses/reserves/spawns under this same lock. Taking
            # leases outside it could race an already chosen M20 worker and
            # cause that worker's nonblocking lease to fail before our abort.
            for uuid in sorted(context['uuids'][1:]):
                leases.acquire(uuid)
            exact_process(rt,old['worker_pid'],old['worker_startticks'])
            exact_process(rt,old['active_pid'],old['active_startticks'])
            # Holding the lock proves the worker cannot be stopped while it owns it.
            rt.signal(old['worker_pid'],signal.SIGSTOP)
            stopped = True
        wait_for(rt,old['worker_pid'],old['worker_startticks'],lambda p:p is not None and p['state'] in ('T','t'),5)
        exact_process(rt,old['active_pid'],old['active_startticks'])
        rt.signal(old['active_pid'],signal.SIGINT)
        interrupted = True
        exit_observation = wait_for(rt,old['active_pid'],old['active_startticks'],
                                    lambda p:p is None or p['state'] in ('Z','X'),interrupt_timeout)
        # Only after the old leaf exits can last.pt be considered a stable commit.
        checkpoint = context['output']/'last.pt'
        checkpoint_record = dict(path=str(checkpoint),sha256=digest(checkpoint),bytes=checkpoint.stat().st_size,
            progress_authority='last.pt model/optimizer/RNG; status.json may lag an atomic commit')
        leases.acquire(context['uuids'][0])
        source_check(spec)
        with registry.locked_registry(root) as records:
            exp,row = require_old_owner(records,context)
            exact_process(rt,old['worker_pid'],old['worker_startticks'])
            prior = deepcopy(row)
            prior.update(status='interrupted_for_dp_handoff',interruption_signal='SIGINT',
                         exit_observation=exit_observation,finished_at_unix=time.time())
            row['attempt_history'] = [prior]
            row.update(worker_pid=own['pid'],worker_startticks=own['startticks'],
                       pid=None,startticks=None,active_attempt='dp_resume',handoff_receipt=str(receipt))
            exp.update(worker_pid=own['pid'],worker_startticks=own['startticks'],active_pid=None,
                       active_startticks=None,reserved_gpu_uuids=context['uuids'],
                       dp_handoff=dict(receipt=str(receipt),checkpoint=checkpoint_record,status='adopted'))
        adopted = True  # Persistent ownership changes BEFORE retiring the stopped parent.
        observed = exact_process(rt,old['worker_pid'],old['worker_startticks'],alive=False)
        if observed is not None and observed['state'] not in ('Z','X'):
            if observed['state'] not in ('T','t'):
                raise RuntimeError('old worker unexpectedly resumed; no broad signal allowed')
            rt.signal(old['worker_pid'],signal.SIGKILL)  # ONLY this PID, never its process group.
        state.update(status='adopted',checkpoint=checkpoint_record,old_leaf_exit_observation=exit_observation)
        lane.save(receipt,state)
        fd_records = leases.descriptor_records(context['uuids'])
        env = dict(os.environ,**spec['env'])
        env.update(CUDA_VISIBLE_DEVICES=','.join(context['uuids']),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',
                   RACHEL_DP_LEASE_FDS_JSON=json.dumps(fd_records),
                   RACHEL_DP_RESUME_CHECKPOINT_SHA256=checkpoint_record['sha256'],
                   RACHEL_DP_HANDOFF_RECEIPT=str(receipt))
        source_check(spec)
        with (handoff_root/'dp_training.log').open('x') as log, registry.locked_registry(root) as records:
            exp,row = records['experiments'][PACKAGE],records['stages'][context['first_key']]
            if exp['worker_pid'] != own['pid'] or exp['worker_startticks'] != own['startticks']:
                raise RuntimeError('new package ownership changed')
            child=rt.popen(spec['command'],cwd=spec['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True,pass_fds=tuple(r['fd'] for r in fd_records))
            child_identity=rt.inspect(child.pid)
            if child_identity is None:
                raise RuntimeError('DP child identity unavailable')
            row.update(pid=child.pid,startticks=child_identity['startticks'],command=spec['command'],
                       log=str(handoff_root/'dp_training.log'),runtime_receipt=str(context['runtime_receipt']))
            exp.update(active_pid=child.pid,active_startticks=child_identity['startticks'])
        code=child.wait();child=None
        if code:
            raise RuntimeError('DP resume attempt failed: '+str(code))
        lane.require_subset(lane.read(context['runtime_receipt']),dict(status='complete'))
        pool.check_completion(first)  # True C16 completion; a segment checkpoint is NOT stage completion.
        with registry.locked_registry(root) as records:
            row=records['stages'][context['first_key']]
            row.update(status='complete',returncode=0,completion=first['completion'],finished_at_unix=time.time())
            records['experiments'][PACKAGE].update(completed_stages=1,active_pid=None,active_startticks=None,
                active_stage=None,reserved_gpu_uuids=[context['uuids'][0]])
        leases.close()  # Remaining six endpoint wrappers use their original one-GPU lease.
        for number,stage in enumerate(old['stages'][1:],2):
            lane.validate_stage(stage)
            for proof in stage.get('prerequisites',[]):
                lane.require_subset(lane.read(proof['path']),proof['expect'])
            command=pool.wrapped(stage,context['uuids'][0],root,PACKAGE,plan)
            env=dict(os.environ,**stage.get('env',{}))
            env.update(RACHEL_TRANSFER_WORKER=PACKAGE,CUDA_VISIBLE_DEVICES=context['uuids'][0],
                       PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log_path=root/'experiments'/PACKAGE/('%03d_%s.log'%(number,stage['name']))
            with log_path.open('x') as log,registry.locked_registry(root) as records:
                row=records['stages'][pool.receipt_key(stage)]
                if row['status']!='queued':raise RuntimeError('endpoint already dispatched; no retry')
                child=rt.popen(command,cwd=stage['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                child_identity=rt.inspect(child.pid)
                if child_identity is None:raise RuntimeError('endpoint child identity unavailable')
                row.update(status='running',pid=child.pid,startticks=child_identity['startticks'],
                    worker_pid=own['pid'],worker_startticks=own['startticks'],command=command,log=str(log_path))
                records['experiments'][PACKAGE].update(active_stage=stage['name'],active_pid=child.pid,active_startticks=child_identity['startticks'])
            code=child.wait();child=None
            if code:raise RuntimeError('original endpoint failed: '+stage['name'])
            pool.check_completion(stage)
            with registry.locked_registry(root) as records:
                records['stages'][pool.receipt_key(stage)].update(status='complete',returncode=0,
                    completion=stage['completion'],finished_at_unix=time.time())
                records['experiments'][PACKAGE].update(completed_stages=number,active_pid=None,
                                                       active_startticks=None,active_stage=None)
        with registry.locked_registry(root) as records:
            exp=records['experiments'][PACKAGE]
            exp.update(status='complete',finished_at_unix=time.time(),reservation_retained=False)
            exp['dp_handoff']['status']='complete'
        state.update(status='complete',completed_stages=7,finished_at_unix=time.time())
        lane.save(receipt,state)
        return state
    except BaseException as error:
        # Do not resume the stopped old worker after interrupting its child: its
        # old except path would mark the whole delegated package failed.
        if stopped and not interrupted and not adopted:
            if exact_process(rt,old['worker_pid'],old['worker_startticks'],alive=False):
                rt.signal(old['worker_pid'],signal.SIGCONT)
        if adopted:
            with registry.locked_registry(root) as records:
                pool.fail_package(records,PACKAGE,repr(error),retain=child is not None)
        state.update(status='failed',error=repr(error),adopted=adopted,
                     manual_recovery_required=bool(interrupted and not adopted or child is not None),
                     surviving_child_pid=None if child is None else child.pid,finished_at_unix=time.time())
        lane.save(receipt,state)
        # As in the original scheduler, observer failure never kills a live
        # independent leaf. Its inherited lock FDs remain held until it exits.
        raise
    finally:
        leases.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--spec',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--interrupt-timeout',type=float,default=120.)
    args=p.parse_args()
    if not 0 < args.interrupt_timeout <= 600:raise ValueError('bounded interrupt timeout 0..600s required')
    spec=lane.read(args.spec)
    if not args.execute:
        c=prepare(args.plan,args.root,spec)
        print(json.dumps(dict(status='ready_for_explicit_handoff',package=PACKAGE,gpu_uuids=c['uuids'],
            old_worker_pid=c['experiment']['worker_pid'],old_leaf_pid=c['experiment']['active_pid'],
            new_runtime_receipt=str(c['runtime_receipt']))))
    else:
        if os.name!='posix' or not Path('/proc/self/stat').exists():raise RuntimeError('execute requires Linux')
        print(json.dumps(execute(args.plan,args.root,spec,interrupt_timeout=args.interrupt_timeout)))


if __name__=='__main__':
    main()
