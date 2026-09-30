"""One-shot bounded CPU-only replay launcher; never starts model training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    name = 'pilot' if args.pilot else 'replay'
    out = root / (name + '_results')
    launch = root / (name + '_launch.json')
    if out.exists() or launch.exists():
        raise SystemExit('Existing output/launch: inspect, do not repeat automatically')
    prepared = json.loads((root / 'training_preparation_remote.json').read_text())
    assert prepared['status'] == 'cpu_preparation_passed'
    relative = Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1')
    for filename, expected in prepared['source_package_sha256'].items():
        actual = hashlib.sha256((root / 'training_source_01' / relative / filename).read_bytes()).hexdigest()
        assert actual == expected, filename
    if not args.pilot:
        complete = json.loads((root / 'pilot_results' / 'complete.json').read_text())
        assert complete['status'] == 'complete' and complete['structural_verification_passed']
        assert not (root / 'pilot_results' / 'failure.json').exists()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / 'training_source_01'), PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, str(root / 'replay.py'), '--phase1',
               '/root/autodl-tmp/consensus_strictness_20260925/results_phase1',
               '--out', str(out), '--workers', '2' if args.pilot else '4']
    if args.pilot:
        command += ['--limit', '3']
    with (root / (name + '.log')).open('xb') as log:
        proc = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    process = Path('/proc') / str(proc.pid)
    record = dict(pid=proc.pid, starttime=int((process / 'stat').read_text().rsplit(')', 1)[1].split()[19]),
                  cmdline=(process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(), command=command,
                  gpu_used=False, model_training=False, threshold=16, threshold_semantics='full diameter')
    launch.write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
