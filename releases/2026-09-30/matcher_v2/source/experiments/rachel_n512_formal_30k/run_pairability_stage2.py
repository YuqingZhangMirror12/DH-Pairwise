"""Run the three predeclared data arms after GPU and data dependencies finish."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

from experiments.rachel_n512_formal_30k.run_pairability_stage1_followup import completed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    completed(root / 'stage1_followup_state.json')
    completed(root / 'data/data_build_state.json')
    data_root = root / 'data/composite_v1'
    data_summary = json.loads((data_root / 'summary.json').read_text())
    if data_summary['train60k']['total'] != 60000 or data_summary['matched24k']['total'] != 24000:
        raise ValueError('completed dataset does not satisfy the predeclared population sizes')
    checkpoint = '/root/autodl-tmp/rachel_ablation_v3_20260907_001/confirmation_candidate_20260908_001/seed260909/training/multiscale7_16_32_64/winner.pt'
    dataset = '/root/autodl-tmp/dataset_rachel_pairwise_n512_v1'
    arms = [('original24k', None), ('matched24k', data_root / 'train_matched24k.json'),
            ('realism60k', data_root / 'train_60k.json')]
    commands = []
    for name, manifest in arms:
        command = [sys.executable, '-m', 'experiments.rachel_n512_formal_30k.train_realism_data_ablation',
            '--checkpoint', checkpoint, '--dataset', dataset, '--output', str(root / 'realism_training' / name),
            '--arm', name, '--device', 'cuda:0']
        if manifest:
            command += ['--train-manifest', str(manifest)]
        commands.append(('train_' + name, command))
    # Freeze all three independent VAL selections before opening external sets.
    for name, _ in arms:
        for split in ('test', 'real'):
            commands.append(('eval_' + name + '_' + split, [sys.executable, '-m',
                'experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint',
                '--training-run', str(root / 'realism_training' / name), '--split', split,
                '--dataset', dataset, '--output', str(root / 'realism_evaluation' / name / split),
                '--device', 'cuda:0']))
    rows = []
    for name, command in commands:
        row = dict(name=name, command=command, status='running', started_unix=time.time())
        rows.append(row)
        with (root / (name + '.log')).open('x') as out:
            child = subprocess.Popen(command, cwd=root / 'source', stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            (root / 'stage2_steps.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
            code = child.wait()
        row.update(status='complete' if code == 0 else 'failed', exit_code=code, ended_unix=time.time())
        (root / 'stage2_steps.json').write_text(json.dumps(rows, indent=2))
        if code:
            raise RuntimeError('data ablation failed: ' + name)


if __name__ == '__main__':
    main()
