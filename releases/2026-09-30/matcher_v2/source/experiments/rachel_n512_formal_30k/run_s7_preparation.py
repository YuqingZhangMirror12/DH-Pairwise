"""Finite CPU preparation job; detached launch leaves a concrete PID receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.rachel_n512_formal_30k.materialize_s7_training import write_json


def process_ticks(pid):
    return Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()[19]


def run(args):
    root = Path(args.output_root)
    source = Path(__file__).resolve().parents[2]
    pool = root / 'gen5_pool'
    data = root / 'data'
    status_path = root / 'preparation_status.json'
    commands = [
        ('gen5_pool', [sys.executable, '-u', '-m',
            'experiments.rachel_n512_formal_30k.build_s7_gen5_pool',
            '--dataset-root', args.dataset_root, '--output-root', str(pool),
            '--pairs-per-class', '1200', '--workers', str(args.workers), '--seed', str(args.seed)]),
        ('fixed_24k', [sys.executable, '-u', '-m',
            'experiments.rachel_n512_formal_30k.materialize_s7_training',
            '--dataset-root', args.dataset_root, '--output-root', str(data),
            '--reference-manifest', args.reference_manifest, '--outline-bank', args.outline_bank,
            '--gen5-manifest', str(pool / 'train_gen5_partition.json'),
            '--workers', str(args.workers), '--seed', str(args.seed)])]
    state = dict(status='running', producer_pid=os.getpid(),
                 producer_start_ticks=process_ticks(os.getpid()), stages=[], started_at=time.time())
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONPATH=str(source))
    write_json(status_path, state)
    try:
        for name, command in commands:
            with (root / (name + '.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=source, env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                stage = dict(name=name, command=command, pid=child.pid,
                             start_ticks=process_ticks(child.pid), status='running')
                state['stages'].append(stage)
                while child.poll() is None:
                    state.update(current_stage=name, updated_at=time.time())
                    progress = (pool if name == 'gen5_pool' else data) / 'status.json'
                    if progress.exists():
                        try:
                            state['data_progress'] = json.loads(progress.read_text())
                        except json.JSONDecodeError:
                            pass
                    write_json(status_path, state)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                stage.update(status='complete' if child.returncode == 0 else 'failed',
                             returncode=child.returncode, finished_at=time.time())
                write_json(status_path, state)
                if child.returncode:
                    raise RuntimeError('%s failed; see %s.log' % (name, name))
        ready = json.loads((data / 'status.json').read_text())
        if ready.get('status') != 'complete' or ready.get('sample_count') != 24000 or ready.get('positive_count') != 12000:
            raise RuntimeError('fixed TRAIN producer did not publish complete balanced data')
        state.update(status='complete', sample_count=24000, positive_count=12000, negative_count=12000,
                     manifest=ready['manifest'], finished_at=time.time())
        write_json(status_path, state)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_at=time.time())
        write_json(status_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root', 'reference-manifest', 'outline-bank', 'output-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=260915)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.launch == args.run or not 1 <= args.workers <= 4:
        parser.error('choose exactly one of --launch/--run; CPU workers1..4')
    root = Path(args.output_root).resolve()
    args.output_root = str(root)
    if args.run:
        run(args)
        return
    root.mkdir(parents=True, exist_ok=True)
    marker = root / 'launch_config.json'
    with marker.open('x') as stream:
        json.dump(vars(args), stream, indent=2)
    command = [sys.executable, '-u', '-m', 'experiments.rachel_n512_formal_30k.run_s7_preparation']
    for key in ('dataset_root', 'reference_manifest', 'outline_bank', 'output_root', 'workers', 'seed'):
        command.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
    command.append('--run')
    with (root / 'producer.log').open('x') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=child.pid, start_ticks=process_ticks(child.pid), command=command,
                   status_path=str(root / 'preparation_status.json'))
    write_json(root / 'producer_launch.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
