"""Persist an ordered frozen-matcher experiment across SSH disconnections."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    root = Path(a.root).resolve()
    source = root / 'source'
    state_path = root / 'stage1_state.json'
    if not a.worker:
        if state_path.exists():
            raise RuntimeError('state exists: inspect live PID before another attempt')
        with (root / 'stage1_controller.log').open('x') as out:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--worker'],
                cwd=source, start_new_session=True, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        print(json.dumps(dict(controller_pid=child.pid, log=str(root / 'stage1_controller.log'))), flush=True)
        return
    state = dict(status='running', controller_pid=os.getpid(), started_unix=time.time(), stages=[])
    checkpoint = '/root/autodl-tmp/rachel_ablation_v3_20260907_001/confirmation_candidate_20260908_001/seed260909/training/multiscale7_16_32_64/winner.pt'
    commands = []
    for split in ('val', 'train'):
        commands.append(('cache_' + split, [sys.executable, '-m', 'experiments.rachel_n512_formal_30k.export_matrix_pair_cache',
            '--checkpoint', checkpoint, '--split', split, '--output', str(root / 'cache' / split)]))
    commands.append(('matrix_head_train', [sys.executable, '-m', 'experiments.rachel_n512_formal_30k.train_matrix_pair_head',
        '--train-manifest', str(root / 'cache/train/manifest.json'), '--val-manifest', str(root / 'cache/val/manifest.json'),
        '--output-root', str(root / 'heads/matrix_soft'), '--device', 'cuda:0', '--epochs', '10', '--batch-size', '8']))
    # External populations are opened only after the head and VAL thresholds freeze.
    for split in ('test', 'real'):
        commands.append(('cache_' + split, [sys.executable, '-m', 'experiments.rachel_n512_formal_30k.export_matrix_pair_cache',
            '--checkpoint', checkpoint, '--split', split, '--output', str(root / 'cache' / split)]))
        commands.append(('matrix_head_' + split, [sys.executable, '-m', 'experiments.rachel_n512_formal_30k.train_matrix_pair_head',
            '--head', str(root / 'heads/matrix_soft/head.pt'), '--evaluate-manifest', str(root / 'cache' / split / 'manifest.json'),
            '--output-root', str(root / 'heads' / ('matrix_soft_' + split)), '--device', 'cuda:0', '--batch-size', '8']))
    try:
        for name, command in commands:
            record = dict(name=name, command=command, started_unix=time.time(), status='running', log=str(root / (name + '.log')))
            state['stages'].append(record)
            with Path(record['log']).open('x') as out:
                child = subprocess.Popen(command, cwd=source, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
                record['pid'] = child.pid
                state_path.write_text(json.dumps(state, indent=2))
                print(json.dumps(record), flush=True)
                code = child.wait()
            record.update(status='complete' if code == 0 else 'failed', exit_code=code, ended_unix=time.time())
            state_path.write_text(json.dumps(state, indent=2))
            if code:
                raise RuntimeError(name + ' failed: inspect ' + record['log'])
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['updated_unix'] = time.time()
        state_path.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
