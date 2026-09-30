"""Predeclared transport-only and external assignment controls after Stage1."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time


def completed(path):
    while True:
        if path.exists():
            try:
                state = json.loads(path.read_text())
            except json.JSONDecodeError:
                time.sleep(5)
                continue
            if state['status'] == 'complete':
                return
            if state['status'] == 'failed':
                raise RuntimeError('upstream failed: ' + str(path))
        time.sleep(10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    completed(root / 'stage1_state.json')
    completed(root / 'assignment_val_state.json')
    commands = []
    # All six external geometry controls keep the already frozen VAL rule.
    for split in ('test', 'real'):
        command = [sys.executable, '-m', 'experiments.rachel_n512_formal_30k.evaluate_cached_assignment_ablation',
            '--cache', str(root / 'cache' / split), '--output', str(root / 'assignment' / split), '--workers', '8',
            '--freeze', str(root / 'assignment/val/validation_freeze.json')]
        if split == 'real':
            command += ['--real-gt', '/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json']
        commands.append(('assignment_' + split, command))
    commands.append(('matrix_transport_train', [sys.executable, '-m', 'experiments.rachel_n512_formal_30k.train_matrix_pair_head',
        '--train-manifest', str(root / 'cache/train/manifest.json'), '--val-manifest', str(root / 'cache/val/manifest.json'),
        '--output-root', str(root / 'heads/matrix_transport_only'), '--device', 'cuda:0',
        '--epochs', '10', '--batch-size', '8', '--transport-only']))
    for split in ('test', 'real'):
        commands.append(('matrix_transport_' + split, [sys.executable, '-m', 'experiments.rachel_n512_formal_30k.train_matrix_pair_head',
            '--head', str(root / 'heads/matrix_transport_only/head.pt'),
            '--evaluate-manifest', str(root / 'cache' / split / 'manifest.json'),
            '--output-root', str(root / 'heads' / ('matrix_transport_only_' + split)), '--device', 'cuda:0', '--batch-size', '8']))
    rows = []
    for name, command in commands:
        row = dict(name=name, command=command, status='running', started_unix=time.time())
        rows.append(row)
        with (root / (name + '.log')).open('x') as out:
            child = subprocess.Popen(command, cwd=root / 'source', stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            (root / 'stage1_followup_steps.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
            code = child.wait()
        row.update(status='complete' if code == 0 else 'failed', exit_code=code, ended_unix=time.time())
        (root / 'stage1_followup_steps.json').write_text(json.dumps(rows, indent=2))
        if code:
            raise RuntimeError('experiment failed: ' + name)


if __name__ == '__main__':
    main()
