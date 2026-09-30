"""Finite, detached CPU preparation of the same S7 TRAIN24K / VAL3K cache."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def run(args):
    root = args.output.resolve()
    if args.detach:
        root.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [sys.executable, str(Path(__file__).resolve()), '--output', str(root),
                   '--workers', str(args.workers)]
        with (root/'supervisor.log').open('x') as log:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(root/'launch.json', dict(pid=child.pid, command=command, device='cpu',
                                     workers=args.workers, launched_at=time.time()))
        print(json.dumps(dict(pid=child.pid, output=str(root))))
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not root.is_dir():
        raise ValueError('requires detached CPU launch with existing output root')
    os.nice(10)
    status = dict(status='running', pid=os.getpid(), started_at=time.time(),
                  completed_splits=[], gpu_jobs_started=False)
    try:
        for split in ('train', 'val'):
            command = [sys.executable, '-u', '-m', 'matched_only.cache', '--split', split,
                       '--output', str(root/split), '--workers', str(args.workers)]
            with (root/(split+'.log')).open('x') as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                status.update(active_split=split, active_pid=child.pid)
                save(root/'status.json', status)
                code = child.wait()
            if code:
                raise RuntimeError('%s cache exited %s' % (split, code))
            protocol = json.loads((root/split/'protocol.json').read_text())
            if (protocol['status'] != 'complete' or not protocol['formal_training_eligible']
                    or protocol['completed_pairs'] != {'train': 24000, 'val': 3000}[split]):
                raise ValueError('incomplete formal cache: '+split)
            status['completed_splits'].append(split)
            save(root/'status.json', status)
        status.update(status='complete', completed_at=time.time(), active_pid=None)
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        save(root/'status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument('--detach', action='store_true')
    run(parser.parse_args())
