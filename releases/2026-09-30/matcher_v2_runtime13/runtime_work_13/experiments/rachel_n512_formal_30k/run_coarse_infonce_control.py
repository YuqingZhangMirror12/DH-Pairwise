"""Independent CPU queue: TRAIN/VAL features -> InfoNCE fit -> frozen TEST/REAL."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


CHECKPOINT = '/root/autodl-tmp/rachel_ablation_v3_20260907_001/confirmation_candidate_20260908_001/seed260909/training/multiscale7_16_32_64/winner.pt'
DATASET = '/root/autodl-tmp/dataset_rachel_pairwise_n512_v1'
PREPARED = '/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared'


def plan(root):
    root = Path(root).resolve()
    experiment = root / 'coarse_infonce'
    features = experiment / 'features'
    freeze = experiment / 'training/validation_freeze.json'
    commands = []
    for split in ('train', 'val'):
        commands.append(('export_' + split, [sys.executable, '-m',
            'experiments.rachel_n512_formal_30k.export_coarse_retrieval_features',
            '--checkpoint', CHECKPOINT, '--dataset', DATASET, '--split', split,
            '--output', str(features / split), '--cpu-threads', '2', '--batch-size', '32']))
    commands.append(('fit', [sys.executable, '-m',
        'experiments.rachel_n512_formal_30k.train_coarse_infonce_retrieval', 'fit',
        '--train-features', str(features / 'train'), '--val-features', str(features / 'val'),
        '--output', str(experiment / 'training')]))
    for split in ('test', 'real'):
        commands.append(('export_' + split, [sys.executable, '-m',
            'experiments.rachel_n512_formal_30k.export_coarse_retrieval_features',
            '--checkpoint', CHECKPOINT, '--dataset', DATASET, '--prepared-cache', PREPARED,
            '--split', split, '--output', str(features / split), '--evaluation-freeze', str(freeze),
            '--cpu-threads', '2', '--batch-size', '32']))
        commands.append(('evaluate_' + split, [sys.executable, '-m',
            'experiments.rachel_n512_formal_30k.train_coarse_infonce_retrieval', 'evaluate',
            '--features', str(features / split), '--freeze', str(freeze),
            '--output', str(experiment / 'evaluation' / split)]))
    return commands


def run(root):
    root = Path(root).resolve()
    experiment = root / 'coarse_infonce'
    experiment.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, command in plan(root):
        row = dict(name=name, command=command, status='running', started_unix=time.time())
        rows.append(row)
        with (experiment / (name + '.log')).open('x') as stream:
            child = subprocess.Popen(command, cwd=root / 'source', stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            (experiment / 'steps.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
            code = child.wait()
        row.update(status='complete' if code == 0 else 'failed', exit_code=code, ended_unix=time.time())
        (experiment / 'steps.json').write_text(json.dumps(rows, indent=2))
        if code:
            raise RuntimeError('CPU InfoNCE control failed: ' + name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    run(parser.parse_args().root)
