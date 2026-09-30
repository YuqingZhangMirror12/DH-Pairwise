"""Finite independent migration lanes. No old host PID dependencies or retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_subset(actual, expected):
    if not isinstance(actual, dict):
        raise ValueError('receipt is not an object')
    for key, value in expected.items():
        if key not in actual:
            raise ValueError('missing receipt field '+key)
        if isinstance(value, dict):
            require_subset(actual[key], value)
        elif actual[key] != value:
            raise ValueError('receipt differs: '+key)


def validate_stage(stage):
    if not stage.get('name') or not isinstance(stage.get('command'), list):
        raise ValueError('stage needs name/command')
    for key in ('cwd', 'completion'):
        if not Path(stage[key]).is_absolute():
            raise ValueError('absolute path required: '+key)
    if not stage.get('completion_expect'):
        raise ValueError('explicit completion fields required')
    if any('after_priority' in x or 'priority_supervisor' in x or 'legacy_tail' in x
           or x.endswith('.queue') or 'after_dependencies' in x for x in stage['command']):
        raise ValueError('only leaf jobs may run in a migration lane')


def wrapped_command(stage, lane, plan):
    command = list(stage['command'])
    if not stage.get('gpu', True):
        return command
    if len(command) < 3 or command[1] != '-m':
        raise ValueError('GPU leaf must be python -m module')
    return [command[0], plan['device_wrapper'], '--gpu-uuid', lane['gpu_uuid'],
            '--lock-root', str(Path(plan['output_root'])/'gpu_locks'),
            '--receipt', str(Path(plan['output_root'])/lane['name']/('runtime_'+stage['name']+'.json')),
            '--module', command[2], '--'] + command[3:]


def validate_plan(plan):
    if plan.get('schema') != 'seven-gpu-migration/1':
        raise ValueError('wrong migration schema')
    lanes = plan['lanes']
    if len(lanes) != 7 or len({x['gpu_uuid'] for x in lanes}) != 7:
        raise ValueError('seven distinct physical GPUs required')
    names = set()
    for lane in lanes:
        if lane['name'] in names:
            raise ValueError('duplicate lane')
        names.add(lane['name'])
        if not lane['stages']:
            raise ValueError('empty lane')
        for stage in lane['stages']:
            validate_stage(stage)
            wrapped_command(stage, lane, plan)
    return plan


def execute(plan_path, lane_name):
    plan = validate_plan(read(plan_path))
    lane = next(x for x in plan['lanes'] if x['name'] == lane_name)
    root = Path(plan['output_root'])/lane_name
    root.mkdir(parents=True, exist_ok=True)
    with (root/'lane.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'status.json').exists():
            raise ValueError('lane already started; no automatic retry/resume')
        state = dict(schema=plan['schema'], lane=lane_name, gpu_uuid=lane['gpu_uuid'],
                     pid=os.getpid(), status='running', started_at_unix=time.time(),
                     completed_stages=0, stages=[], active_pid=None)
        save(root/'status.json', state)
        try:
            for index, stage in enumerate(lane['stages']):
                for requirement in stage.get('prerequisites', []):
                    state.update(status='waiting_artifact', waiting_for=requirement['path'])
                    save(root/'status.json', state)
                    while True:
                        path = Path(requirement['path'])
                        if path.exists():
                            value = None
                            try:
                                value = read(path)
                                require_subset(value, requirement['expect'])
                                break
                            except (ValueError, json.JSONDecodeError):
                                if isinstance(value, dict) and value.get('status') in ('failed', 'error'):
                                    raise RuntimeError('dependency failed: '+str(path))
                        producer = requirement.get('producer_lane')
                        if producer:
                            report = Path(plan['output_root'])/producer/'status.json'
                            if report.exists() and read(report).get('status') == 'failed':
                                raise RuntimeError('producer lane failed: '+producer)
                        time.sleep(15)
                env = dict(os.environ)
                env.update(stage.get('env', {}))
                env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1')
                if stage.get('gpu', True):
                    env['CUDA_VISIBLE_DEVICES'] = lane['gpu_uuid']
                else:
                    env['CUDA_VISIBLE_DEVICES'] = ''
                log_path = root/('%03d_%s.log' % (index+1, stage['name']))
                item = dict(name=stage['name'], command=wrapped_command(stage,lane,plan),
                            cwd=stage['cwd'], log=str(log_path), status='running',
                            started_at_unix=time.time(), completion=stage['completion'])
                with log_path.open('x') as log:
                    child = subprocess.Popen(item['command'], cwd=stage['cwd'], env=env,
                                             stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    item['pid'] = child.pid
                    state['stages'].append(item)
                    state.update(status='running', active_pid=child.pid, active_name=stage['name'])
                    state.pop('waiting_for', None)
                    save(root/'status.json', state)
                    code = child.wait()
                item.update(returncode=code, finished_at_unix=time.time())
                if code:
                    raise RuntimeError('stage exited %s: %s' % (code,stage['name']))
                require_subset(read(stage['completion']), stage['completion_expect'])
                for proof in stage.get('additional_completions', []):
                    require_subset(read(proof['path']), proof['expect'])
                item['status'] = 'complete'
                state.update(completed_stages=index+1, active_pid=None, active_name=None)
                save(root/'status.json', state)
            state.update(status='complete', finished_at_unix=time.time())
            save(root/'status.json', state)
        except BaseException as exc:
            state.update(status='failed', error=repr(exc), finished_at_unix=time.time())
            save(root/'status.json', state)
            # Observer interruption never kills/restarts an independently running child.
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--lane')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    plan=validate_plan(read(args.plan))
    if args.execute:
        if not args.lane: p.error('--lane required for execution')
        execute(args.plan,args.lane)
    else:
        print(json.dumps({x['name']:len(x['stages']) for x in plan['lanes']}))


if __name__=='__main__':
    main()
