"""Finite CPU-only follow-up to a specifically identified live feature writer.

Wait without touching that process, then derive TRAIN/VAL candidate stages.
Never acquire GPU resources, dispatch training, retry a failed writer, or alter
the existing GPU queue. A stale complete receipt alone is not sufficient.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def process_identity(pid):
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(state=fields[0], startticks=int(fields[19]))


def dependency_state(receipt, process, pid, startticks):
    if receipt.get('pid') != pid or receipt.get('gpu_jobs_started') is not False:
        raise RuntimeError('preparation receipt identity changed')
    if process is not None and process['startticks'] != startticks:
        raise RuntimeError('preparation PID reused; do not attach to replacement')
    if receipt.get('status') == 'failed':
        raise RuntimeError('source preparation failed; no restart requested')
    if process is not None and process['state'] != 'Z':
        return 'waiting'
    if receipt.get('status') != 'complete' or receipt.get('completed_splits') != ['train', 'val']:
        raise RuntimeError('preparation process ended without complete TRAIN/VAL')
    return 'ready'


def execute(args):
    root = Path(args.output).resolve()
    status_path = root / 'status.json'
    if status_path.exists():
        raise ValueError('follow-up already started; no implicit retry or replacement')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU follow-up requires CUDA_VISIBLE_DEVICES empty')
    if not 0 < args.wait_hours <= 48 or args.pid < 1 or args.startticks < 1:
        raise ValueError('invalid dependency identity or finite wait bound')
    root.mkdir(parents=True, exist_ok=True)
    source = Path(args.base_root).resolve(strict=True)
    state = dict(status='waiting_feature_cache', pid=os.getpid(), started_at=time.time(),
        base_root=str(source), dependency_pid=args.pid, dependency_startticks=args.startticks,
        gpu_jobs_started=False, automatic_retry=False, completed_splits=[], stages=[])
    save(status_path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    deadline = time.monotonic() + args.wait_hours * 3600
    try:
        while True:
            receipt = json.loads((source / 'status.json').read_text())
            if dependency_state(receipt, process_identity(args.pid), args.pid, args.startticks) == 'ready':
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('feature writer still live at observation deadline; NOT restarted')
            time.sleep(30)
        for split in ('train', 'val'):
            output = root / split
            command = [sys.executable, '-u', '-m', 'matched_only.stage_cache', '--base-cache',
                str(source / split), '--split', split, '--output', str(output)]
            stage = dict(split=split, output=str(output), command=command, status='running')
            with (root / (split + '.log')).open('x') as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, env=env)
                stage.update(pid=child.pid, started_at=time.time())
                state['stages'].append(stage)
                state.update(status='preparing_stages', active_split=split, active_pid=child.pid)
                save(status_path, state)
                code = child.wait()
            if code:
                raise RuntimeError('%s candidate-stage cache exited %s' % (split, code))
            result = json.loads((output / 'protocol.json').read_text())
            if (result.get('status') != 'complete' or result.get('formal_training_eligible') is not True
                    or result.get('completed_pairs') != {'train': 24000, 'val': 3000}[split]
                    or result.get('gt_used') is not False or result.get('labels_used') is not False):
                raise RuntimeError('candidate-stage cache is incomplete or not target-blind')
            stage.update(status='complete', finished_at=time.time())
            state['completed_splits'].append(split)
            state.update(active_pid=None)
            save(status_path, state)
        state.update(status='complete', active_split=None)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['finished_at'] = time.time()
        save(status_path, state)
    return state


def main(args):
    if not args.detach:
        os.nice(10)
        return execute(args)
    # Require actual liveness at registration; a receipt alone cannot prove it.
    source = Path(args.base_root).resolve(strict=True)
    process = process_identity(args.pid)
    if process is None or process['state'] == 'Z' or process['startticks'] != args.startticks:
        raise RuntimeError('expected feature-cache writer is not live now')
    dependency_state(json.loads((source / 'status.json').read_text()), process, args.pid, args.startticks)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--base-root', str(source),
        '--output', str(root), '--pid', str(args.pid), '--startticks', str(args.startticks),
        '--wait-hours', str(args.wait_hours)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (root / 'supervisor.log').open('x') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, env=env, start_new_session=True)
    save(root / 'launch.json', dict(pid=child.pid, command=command,
        dependency_pid=args.pid, dependency_startticks=args.startticks, device='cpu'))
    print(json.dumps(dict(pid=child.pid, output=str(root))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--startticks', type=int, required=True)
    parser.add_argument('--wait-hours', type=float, default=48.)
    parser.add_argument('--detach', action='store_true')
    main(parser.parse_args())
