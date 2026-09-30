"""Finite transferred experiment pool; original modules, outputs and math stay fixed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from . import transfer_registry as registry, lane_runner as lane, device_runtime as device
except ImportError:
    import transfer_registry as registry
    import lane_runner as lane
    import device_runtime as device


def identity(pid):
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), state=fields[0], startticks=int(fields[19]))
    except FileNotFoundError:
        return None


def receipt_key(stage):
    return str(Path(stage['original_runtime_receipt']).resolve())


def require_alive(pid, startticks, inspect=identity):
    value = inspect(pid)
    if value is not None and value['startticks'] != startticks:
        raise RuntimeError('PID reused: '+str(pid))
    return value is not None and value['state'] not in ('Z', 'X')


def check_completion(stage):
    lane.require_subset(lane.read(stage['completion']), stage['completion_expect'])
    for proof in stage.get('additional_completions', []):
        lane.require_subset(lane.read(proof['path']), proof['expect'])


def prerequisites_ready(package, records, *, read=lane.read, plan=None):
    """Internal proofs must come from earlier leaves; only external proofs gate dispatch."""
    produced = set()
    for stage in package['stages']:
        for proof in stage.get('prerequisites', []):
            if proof['path'] in produced:
                continue
            actual = None
            try:
                actual = read(proof['path'])
                lane.require_subset(actual, proof['expect'])
            except (OSError, ValueError, KeyError):
                for exp in records.get('experiments', {}).values():
                    for source in exp['stages']:
                        paths = [source['completion']] + [p['path'] for p in source.get('additional_completions', [])]
                        row = records.get('stages', {}).get(receipt_key(source), {})
                        if proof['path'] in paths and (exp.get('status') == 'failed' or row.get('status') == 'failed'):
                            raise RuntimeError('transferred prerequisite failed: '+proof['path'])
                producer = proof.get('producer_lane')
                if producer and plan:
                    try:
                        producer_status = read(str(Path(plan['output_root'])/producer/'status.json'))
                    except FileNotFoundError:
                        producer_status = {}
                    if producer_status.get('status') == 'failed':
                        raise RuntimeError('producer lane failed: '+producer)
                if isinstance(actual, dict) and actual.get('status') in ('complete', 'training_complete', 'smoke_complete', 'failed', 'error'):
                    raise RuntimeError('terminal prerequisite does not satisfy contract: '+proof['path'])
                return False
        produced.add(stage['completion'])
        produced.update(p['path'] for p in stage.get('additional_completions', []))
    return True


def gpu_idle(uuid, lock_root):
    """Short coordinator availability probe; no parent holds a lease during work."""
    path = Path(lock_root)/(uuid+'.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        try:
            return not device.foreign_pids(uuid)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def eligible_gpus(plan, records, *, read=lane.read, idle=gpu_idle):
    reserved = {e.get('assigned_gpu') for e in records['experiments'].values()
                if e.get('status') == 'running' or e.get('reservation_retained')}
    available = []
    for original in plan['lanes']:
        uuid = original['gpu_uuid']
        if uuid in reserved:
            continue
        try:
            status = read(str(Path(plan['output_root'])/original['name']/'status.json'))
            lane.require_subset(status, dict(status='complete', active_pid=None,
                                             completed_stages=len(original['stages'])))
            if len(status.get('stages', [])) != len(original['stages']) or any(
                    s.get('status') != 'complete' or s.get('returncode') != 0 for s in status['stages']):
                continue
        except (OSError, ValueError, KeyError):
            continue
        if idle(uuid, Path(plan['output_root'])/'gpu_locks'):
            available.append(uuid)
    return available


def choose_assignments(plan, records, **kwargs):
    queued = sorted((e for e in records['experiments'].values() if e['status'] == 'queued'),
                    key=lambda e: (e.get('priority', 100), e['name']))
    ready = [e for e in queued if prerequisites_ready(e, records, read=kwargs.get('read', lane.read), plan=plan)]
    return [(e['name'], uuid) for e, uuid in zip(ready, eligible_gpus(plan, records, **kwargs))]


def fail_package(records, name, error, *, retain=False):
    experiment = records['experiments'][name]
    experiment.update(status='failed', error=str(error), finished_at_unix=time.time(),
                      reservation_retained=retain)
    for stage in experiment['stages']:
        record = records['stages'][receipt_key(stage)]
        if record.get('status') != 'complete':
            record.update(status='failed', error=str(error), finished_at_unix=time.time())


def wrapped(stage, uuid, root, name, plan):
    command = list(stage['command'])
    if not stage.get('gpu', True):
        return command
    lane.validate_stage(stage)
    if command[1] != '-m':
        raise ValueError('GPU stage must retain original python -m command')
    receipt = Path(root)/'experiments'/name/('runtime_'+stage['name']+'.json')
    if receipt.resolve() == Path(stage['original_runtime_receipt']).resolve():
        raise ValueError('transfer worker may not use original gate receipt')
    return [command[0], plan['device_wrapper'], '--gpu-uuid', uuid,
            '--lock-root', str(Path(plan['output_root'])/'gpu_locks'), '--receipt', str(receipt),
            '--module', command[2], '--']+command[3:]


class Runtime:
    inspect = staticmethod(identity)
    popen = staticmethod(subprocess.Popen)
    sleep = staticmethod(time.sleep)


def worker(plan_path, root, name, uuid, *, runtime=None):
    runtime = runtime or Runtime()
    plan, root = lane.read(plan_path), Path(root)
    own = runtime.inspect(os.getpid())
    if own is None:
        raise RuntimeError('worker process identity unavailable')
    with registry.locked_registry(root) as records:
        exp = records['experiments'][name]
        if (exp['status'] != 'running' or exp['assigned_gpu'] != uuid
                or exp.get('worker_pid') != own['pid'] or exp.get('worker_startticks') != own['startticks']):
            raise RuntimeError('worker does not own this atomically reserved experiment')
        stages = exp['stages']
    output = root/'experiments'/name
    output.mkdir(parents=True, exist_ok=True)
    child = None
    try:
        for number, stage in enumerate(stages):
            lane.validate_stage(stage)
            for proof in stage.get('prerequisites', []):
                lane.require_subset(lane.read(proof['path']), proof['expect'])
            env = dict(os.environ, **stage.get('env', {}))
            env.update(RACHEL_TRANSFER_WORKER=name, CUDA_VISIBLE_DEVICES=uuid if stage.get('gpu', True) else '',
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            command = wrapped(stage, uuid, root, name, plan)
            log_path = output/('%03d_%s.log' % (number+1, stage['name']))
            with log_path.open('x') as log, registry.locked_registry(root) as records:
                row = records['stages'][receipt_key(stage)]
                if row.get('status') != 'queued':
                    raise RuntimeError('stage already dispatched; no retry: '+stage['name'])
                child = runtime.popen(command, cwd=stage['cwd'], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
                observed = runtime.inspect(child.pid)
                if observed is None:
                    raise RuntimeError('new leaf identity unavailable')
                row.update(status='running', pid=child.pid, startticks=observed['startticks'],
                           worker_pid=own['pid'], worker_startticks=own['startticks'],
                           command=command, log=str(log_path), started_at_unix=time.time())
                records['experiments'][name].update(active_stage=stage['name'], active_pid=child.pid,
                                                    active_startticks=observed['startticks'])
            code = child.wait()
            child = None
            with registry.locked_registry(root) as records:
                records['stages'][receipt_key(stage)].update(returncode=code)
                records['experiments'][name].update(active_pid=None, active_startticks=None)
            if code:
                raise RuntimeError('leaf failed with returncode %s: %s' % (code, stage['name']))
            check_completion(stage)
            with registry.locked_registry(root) as records:
                records['stages'][receipt_key(stage)].update(status='complete', returncode=0,
                    finished_at_unix=time.time(), completion=stage['completion'])
                records['experiments'][name].update(completed_stages=number+1, active_pid=None,
                                                    active_startticks=None, active_stage=None)
        with registry.locked_registry(root) as records:
            records['experiments'][name].update(status='complete', finished_at_unix=time.time(), active_pid=None)
    except BaseException as error:
        with registry.locked_registry(root) as records:
            fail_package(records, name, repr(error), retain=child is not None)
        raise  # Never signal/restart an independently running leaf.


def execute(plan_path, root, *, runtime=None, poll_seconds=15):
    runtime = runtime or Runtime()
    plan, root = lane.read(plan_path), Path(root)
    if not 0 < poll_seconds <= 30:
        raise ValueError('poll interval must be positive and <=30s')
    root.mkdir(parents=True, exist_ok=True)
    with (root/'coordinator.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'coordinator_status.json').exists():
            raise ValueError('coordinator already started; no automatic resume')
        own = runtime.inspect(os.getpid())
        if own is None:
            raise RuntimeError('coordinator identity unavailable')
        status = dict(status='running', pid=own['pid'], startticks=own['startticks'], started_at_unix=time.time())
        lane.save(root/'coordinator_status.json', status)
        children = {}
        try:
            while True:
                for pid, child in list(children.items()):
                    if child.poll() is not None:
                        children.pop(pid)
                with registry.locked_registry(root) as records:
                    if not records['experiments']:
                        raise ValueError('no registered experiments')
                    for name, exp in records['experiments'].items():
                        if exp['status'] == 'running':
                            try:
                                if not require_alive(exp['worker_pid'], exp['worker_startticks'], runtime.inspect):
                                    raise RuntimeError('worker exited without final experiment receipt')
                            except (RuntimeError, KeyError) as error:
                                fail_package(records, name, repr(error), retain=True)
                        elif exp['status'] == 'queued':
                            try:
                                prerequisites_ready(exp, records, plan=plan)
                            except RuntimeError as error:
                                fail_package(records, name, repr(error))
                    for name, uuid in choose_assignments(plan, records):
                        exp = records['experiments'][name]
                        exp.update(status='running', assigned_gpu=uuid, dispatched_at_unix=time.time())
                        child = None
                        try:
                            child = runtime.popen([sys.executable, str(Path(__file__).resolve()), '--plan', str(plan_path),
                                '--root', str(root), '--worker', name, '--gpu', uuid], start_new_session=True)
                            observed = runtime.inspect(child.pid)
                            if observed is None:
                                raise RuntimeError('new worker identity unavailable')
                            exp.update(worker_pid=child.pid, worker_startticks=observed['startticks'])
                            children[child.pid] = child
                        except BaseException as error:
                            fail_package(records, name, repr(error), retain=child is not None)
                    terminal = bool(records['experiments']) and all(
                        e['status'] in ('complete', 'failed') for e in records['experiments'].values())
                    counts = {s:sum(e['status']==s for e in records['experiments'].values())
                              for s in ('queued', 'running', 'complete', 'failed')}
                status.update(experiments=counts, updated_at_unix=time.time())
                if terminal:
                    status.update(status='failed' if counts['failed'] else 'complete', finished_at_unix=time.time())
                    lane.save(root/'coordinator_status.json', status)
                    return status
                lane.save(root/'coordinator_status.json', status)
                runtime.sleep(poll_seconds)
        except BaseException as error:
            status.update(status='failed', error=repr(error), finished_at_unix=time.time())
            lane.save(root/'coordinator_status.json', status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--root', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--execute', action='store_true')
    group.add_argument('--worker')
    parser.add_argument('--gpu')
    args = parser.parse_args()
    if args.worker:
        if not args.gpu:
            parser.error('--worker requires --gpu')
        worker(args.plan, args.root, args.worker, args.gpu)
    else:
        execute(args.plan, args.root)


if __name__ == '__main__':
    main()
