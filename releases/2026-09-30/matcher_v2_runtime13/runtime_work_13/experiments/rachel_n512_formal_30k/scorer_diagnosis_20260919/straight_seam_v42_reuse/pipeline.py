"""Bounded CPU-only reference reproduction: 30 review pairs then 900 SELECT pairs.

No training, no TEST, no retries, no mutation of reference sources or old runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def commands(args, out):
    source = Path(__file__).resolve().parent
    common = [sys.executable, str(source/'generate.py'), '--reference-dir', str(args.reference_dir),
              '--source-admission', str(args.source_admission), '--real-audit', str(args.real_audit),
              '--workers', '2']
    audit = [sys.executable, str(source/'audit_review.py'), '--metrics-code', str(args.metrics_code),
             '--official-code', str(args.official_code)]
    return [
        ('review_generate', common + ['--mode', 'review30', '--seed', '26093082', '--output-new', str(out/'review30')]),
        ('review_audit_render', audit + ['--root', str(out/'review30'), '--output-new', str(out/'review30_html'), '--render']),
        ('select_generate', common + ['--mode', 'select900', '--seed', '26093083', '--output-new', str(out/'select900')]),
        ('select_audit', audit + ['--root', str(out/'select900'), '--output-new', str(out/'select900_audit')]),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir', 'source-admission', 'real-audit', 'metrics-code', 'official-code', 'output-new'):
        p.add_argument('--'+name, required=True, type=Path)
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = args.output_new.resolve(); out.mkdir(parents=True, exist_ok=False)
    steps = commands(args, out)
    frozen = {}
    for directory in (Path(__file__).resolve().parent, args.reference_dir):
        for path in directory.glob('*.py'):
            frozen[str(path)] = digest(path)
    for path in (args.source_admission, args.real_audit, args.metrics_code):
        frozen[str(path)] = digest(path)
    launch = dict(pid=os.getpid(), pgid=os.getpgrp(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
                  started_unix=time.time(), commands=steps, file_hashes=frozen, gpu=False,
                  automatic_retries=0, training_admitted=False, seed_policy='independent from published reference',
                  output_root=str(out))
    write(out/'launch.json', launch)
    completed = []
    for name, command in steps:
        if any(digest(path) != expected for path, expected in frozen.items()):
            write(out/'failure.json', dict(stage=name, reason='frozen input changed', completed=completed)); return 2
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        started = time.time()
        with (out/(name+'.log')).open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            write(out/'status.json', dict(stage=name, child_pid=child.pid, started_unix=started,
                                         completed=completed, training_admitted=False))
            rc = child.wait()
        receipt = dict(stage=name, returncode=rc, seconds=time.time()-started, finished_unix=time.time())
        write(out/(name+'_return.json'), receipt)
        if rc:
            write(out/'failure.json', dict(**receipt, completed=completed)); return rc
        completed.append(receipt)
    write(out/'complete.json', dict(status='complete', completed=completed, finished_unix=time.time(),
                                   files_unchanged=all(digest(path)==expected for path,expected in frozen.items()),
                                   training_admitted=False, gpu=False))
    write(out/'status.json', dict(stage='complete', completed=completed, training_admitted=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
