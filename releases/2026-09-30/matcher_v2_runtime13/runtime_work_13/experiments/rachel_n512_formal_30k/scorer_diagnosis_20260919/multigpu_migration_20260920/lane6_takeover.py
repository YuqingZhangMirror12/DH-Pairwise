"""Adopt one existing lane6 GPU leaf and advance its three independent M20 CPU leaves.

External scheduling only: original commands, sources, outputs and budgets are unchanged.
No signals, retries, model imports, or automatic takeover of another running owner.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

try:
    from . import lane_runner as lane
    from .dynamic_pool import identity
except ImportError:
    import lane_runner as lane
    from dynamic_pool import identity

CPU_NAMES = ('M20_hard_SIMVAL6000', 'M20_train_cache', 'M20_val_cache')
WRAPPER_SCHEMA = 'rachel-assigned-physical-gpu-runtime/1'


class Runtime:
    inspect = staticmethod(identity)
    popen = staticmethod(subprocess.Popen)
    sleep = staticmethod(time.sleep)


def alive_exact(pid, ticks, inspect=identity):
    value = inspect(pid)
    if value is not None and value['startticks'] != ticks:
        raise RuntimeError('adopted PID reused: '+str(pid))
    return value is not None and value['state'] not in ('Z', 'X')


def proofs(stage):
    lane.require_subset(lane.read(stage['completion']), stage['completion_expect'])
    for proof in stage.get('additional_completions', []):
        lane.require_subset(lane.read(proof['path']), proof['expect'])


def wrapper_proof(plan, selected, stage, pid, *, complete=False):
    path = Path(plan['output_root'])/selected['name']/('runtime_'+stage['name']+'.json')
    value = lane.read(path)
    expected = dict(schema=WRAPPER_SCHEMA, pid=pid, module=stage['command'][2],
                    arguments=stage['command'][3:], gpu={'uuid': selected['gpu_uuid']})
    if complete:
        expected['status'] = 'complete'
    lane.require_subset(value, expected)
    if value.get('status') not in ('starting', 'running', 'complete'):
        raise ValueError('adopted wrapper is failed/unknown')
    return str(path)


def preflight(plan, state, pid, ticks, *, runtime=None, require_old_terminal=False):
    runtime = runtime or Runtime()
    selected = next(v for v in plan['lanes'] if v['name'] == 'lane6')
    stages, index = selected['stages'], state['completed_stages']
    if (state.get('lane') != 'lane6' or state.get('status') != 'running'
            or len(state['stages']) != index+1 or not 0 <= index < len(stages)):
        raise ValueError('requires one active leaf after preserved completed prefix')
    active = stages[index]
    if (active['name'] != 'M16_all_tokens_C16_train' or not active.get('gpu')
            or state.get('active_name') != active['name'] or state.get('active_pid') != pid):
        raise ValueError('only the explicitly identified M16 C16 training leaf may be adopted')
    for specification, record in zip(stages[:index], state['stages'][:index]):
        lane.require_subset(record, dict(name=specification['name'], status='complete', returncode=0))
    lane.require_subset(state['stages'][index], dict(name=active['name'], pid=pid, status='running',
        command=lane.wrapped_command(active, selected, plan), cwd=active['cwd']))
    old = runtime.inspect(state['pid'])
    if require_old_terminal and old is not None and old['state'] not in ('Z', 'X'):
        raise ValueError('old lane parent remains alive; only its external owner may terminate it')
    live = alive_exact(pid, ticks, runtime.inspect)
    wrapper_proof(plan, selected, active, pid, complete=not live)
    if not live:
        proofs(active)
    training = next(s for s in stages if s['name'] == 'S7_M13_M20_train')
    proofs(training)
    cpu = [next(s for s in stages if s['name'] == name) for name in CPU_NAMES]
    for stage in cpu:
        lane.validate_stage(stage)
        if stage.get('gpu') is not False or stages.index(stage) <= index:
            raise ValueError('CPU preparation must be an untouched future CPU leaf')
        for proof in stage.get('prerequisites', []):
            lane.require_subset(lane.read(proof['path']), proof['expect'])
        if Path(stage['output']).exists() or Path(stage['completion']).exists():
            raise FileExistsError('CPU preparation already started: '+stage['name'])
        flag = '--checkpoint' if '--checkpoint' in stage['command'] else '--matcher-checkpoint'
        checkpoint = Path(stage['command'][stage['command'].index(flag)+1])
        if checkpoint.name != 'epoch_020.pt' or not checkpoint.is_file():
            raise ValueError('completed M20 checkpoint must exist')
    root = Path(plan['output_root'])/'lane6'
    if (root/'takeover').exists() or (root/'status.pre_takeover.json').exists():
        raise FileExistsError('takeover already attempted; no implicit resume')
    return selected, index, cpu


def env_for(stage, selected):
    env = dict(os.environ, **stage.get('env', {}))
    env.update(CUDA_VISIBLE_DEVICES=selected['gpu_uuid'] if stage.get('gpu', True) else '',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


class CPUPreparation:
    def __init__(self, stages, root, selected, runtime):
        self.stages, self.root, self.selected, self.runtime = stages, root, selected, runtime
        self.jobs, self.errors = {}, []

    def save(self, name, value):
        lane.save(self.root/(name+'.json'), value)

    def start(self):
        for stage in self.stages:
            name = stage['name']
            record = dict(name=name, command=stage['command'], cwd=stage['cwd'], status='starting',
                          completion=stage['completion'], started_at_unix=time.time())
            child = None
            try:
                for proof in stage.get('prerequisites', []):
                    lane.require_subset(lane.read(proof['path']), proof['expect'])
                with (self.root/(name+'.log')).open('x') as log:
                    child = self.runtime.popen(stage['command'], cwd=stage['cwd'],
                        env=env_for(stage, self.selected), stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                observed = self.runtime.inspect(child.pid)
                record.update(status='running', pid=child.pid,
                              startticks=observed['startticks'] if observed else None)
                self.jobs[name] = (stage, child, record)
                self.save(name, record)
            except Exception as error:
                record.update(status='failed', error=repr(error))
                self.save(name, record)
                self.errors.append(name+': '+repr(error))
                break  # No retry; existing jobs remain monitored and are never killed.

    def poll(self):
        for name, (stage, child, record) in self.jobs.items():
            if record['status'] != 'running':
                continue
            code = child.poll()
            if code is None:
                continue
            record.update(returncode=code, finished_at_unix=time.time())
            try:
                if code != 0:
                    raise RuntimeError('CPU leaf exited '+str(code))
                proofs(stage)
                record['status'] = 'complete'
            except Exception as error:
                record.update(status='failed', error=repr(error))
                self.errors.append(name+': '+repr(error))
            self.save(name, record)

    def finish(self):
        while any(r['status'] == 'running' for _, _, r in self.jobs.values()):
            self.poll()
            if any(r['status'] == 'running' for _, _, r in self.jobs.values()):
                self.runtime.sleep(5)
        if self.errors:
            raise RuntimeError('; '.join(self.errors))


def wait_adopted(plan, selected, stage, pid, ticks, preparations, runtime):
    while alive_exact(pid, ticks, runtime.inspect):
        preparations.poll()  # A CPU failure does not interrupt the adopted GPU job.
        runtime.sleep(5)
    wrapper_proof(plan, selected, stage, pid, complete=True)
    proofs(stage)


def execute(plan_path, adopt_pid, adopt_startticks, *, runtime=None):
    runtime = runtime or Runtime()
    plan = lane.validate_plan(lane.read(plan_path))
    root = Path(plan['output_root'])/'lane6'
    with (root/'lane.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = lane.read(root/'status.json')
        selected, index, cpu = preflight(plan, state, adopt_pid, adopt_startticks,
            runtime=runtime, require_old_terminal=True)
        with (root/'status.pre_takeover.json').open('x') as backup:
            json.dump(state, backup, indent=2)
        extra = root/'takeover'
        extra.mkdir()
        metadata = dict(status='running', pid=os.getpid(), original_parent_pid=state['pid'],
            adopted_pid=adopt_pid, adopted_startticks=adopt_startticks, started_at_unix=time.time(),
            plan=str(Path(plan_path).resolve()), cpu_stages=list(CPU_NAMES))
        lane.save(extra/'protocol.json', metadata)
        state.update(pid=os.getpid(), takeover=str(extra/'protocol.json'))
        lane.save(root/'status.json', state)
        preparations = CPUPreparation(cpu, extra, selected, runtime)
        try:
            preparations.start()
            active = selected['stages'][index]
            wait_adopted(plan, selected, active, adopt_pid, adopt_startticks, preparations, runtime)
            state['stages'][index].update(status='complete', returncode=0, finished_at_unix=time.time(),
                adopted=True, returncode_evidence='terminal process plus complete original wrapper and leaf proofs')
            state.update(completed_stages=index+1, active_pid=None, active_name=None)
            lane.save(root/'status.json', state)
            for number in range(index+1, len(selected['stages'])):
                preparations.poll()
                if preparations.errors:
                    preparations.finish()
                stage = selected['stages'][number]
                if stage['name'] in CPU_NAMES:
                    preparations.finish()
                    _, _, recorded = preparations.jobs[stage['name']]
                    item = deepcopy(recorded)
                    item['prefetched'] = True
                    proofs(stage)
                else:
                    for proof in stage.get('prerequisites', []):
                        while True:
                            value = None
                            try:
                                value = lane.read(proof['path'])
                                lane.require_subset(value, proof['expect'])
                                break
                            except (OSError, ValueError):
                                if isinstance(value, dict) and value.get('status') in ('failed', 'error'):
                                    raise RuntimeError('dependency failed: '+proof['path'])
                                producer = proof.get('producer_lane')
                                if producer:
                                    path = Path(plan['output_root'])/producer/'status.json'
                                    if path.exists() and lane.read(path).get('status') == 'failed':
                                        raise RuntimeError('producer lane failed: '+producer)
                                preparations.poll()
                                if preparations.errors:
                                    preparations.finish()
                                runtime.sleep(5)
                    command = lane.wrapped_command(stage, selected, plan)
                    item = dict(name=stage['name'], command=command, cwd=stage['cwd'], status='running',
                        completion=stage['completion'], started_at_unix=time.time())
                    with (root/('%03d_%s.log' % (number+1, stage['name']))).open('x') as log:
                        child = runtime.popen(command, cwd=stage['cwd'], env=env_for(stage, selected),
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    item['pid'] = child.pid
                    state['stages'].append(item)
                    state.update(status='running', active_pid=child.pid, active_name=stage['name'])
                    lane.save(root/'status.json', state)
                    while child.poll() is None:
                        preparations.poll()
                        runtime.sleep(5)
                    code = child.wait()
                    item.update(returncode=code, finished_at_unix=time.time())
                    if code:
                        raise RuntimeError('leaf exited %s: %s' % (code, stage['name']))
                    proofs(stage)
                    item['status'] = 'complete'
                if stage['name'] in CPU_NAMES:
                    state['stages'].append(item)
                state.update(completed_stages=number+1, active_pid=None, active_name=None)
                lane.save(root/'status.json', state)
            preparations.finish()
            state.update(status='complete', finished_at_unix=time.time())
            metadata.update(status='complete', finished_at_unix=time.time())
        except BaseException as error:
            state.update(status='failed', error=repr(error), finished_at_unix=time.time())
            metadata.update(status='failed', error=repr(error), finished_at_unix=time.time())
            raise
        finally:
            lane.save(root/'status.json', state)
            lane.save(extra/'protocol.json', metadata)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--lane', choices=('lane6',), default='lane6')
    parser.add_argument('--adopt-pid', type=int, required=True)
    parser.add_argument('--adopt-startticks', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = execute(args.plan, args.adopt_pid, args.adopt_startticks)
    else:
        plan = lane.validate_plan(lane.read(args.plan))
        root = Path(plan['output_root'])/'lane6'
        _, index, stages = preflight(plan, lane.read(root/'status.json'), args.adopt_pid, args.adopt_startticks)
        result = dict(status='preflight_only', adopted_stage_index=index,
            cpu_stages=[s['name'] for s in stages], old_parent_must_release_lane_lock=True)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
