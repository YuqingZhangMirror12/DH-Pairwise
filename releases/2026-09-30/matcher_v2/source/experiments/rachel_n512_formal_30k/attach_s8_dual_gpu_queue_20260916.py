"""Append authorized S8 after S6 and preserve old tails behind an S8/S7 join."""
import copy
import json
import os
from pathlib import Path
import subprocess
import time

R = Path('/root/autodl-tmp/rachel_score_design_20260913_001')
N = R / 's6_s7_20260915'
NEW = N / 'dual_gpu_20260916'
OLD = N / 'priority_after_s5/queues'
S8 = R / 's8_step2048_attention_20260916'
PYTHON = '/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python'
GPU = 'GPU-c1b7c89a-0519-b37f-6885-f046419eee28'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def live(pid, marker):
    try:
        return marker.encode() in Path('/proc', str(pid), 'cmdline').read_bytes()
    except FileNotFoundError:
        return False


def dependency(name, row):
    config = Path(row['config'])
    return dict(name=name, pid=row['pid'], marker=str(config),
                status_path=str(config.parent / 'queue_state.json'))


def launch(name, config_path):
    config = read(config_path)
    root = Path(config['root'])
    assert not (root / 'queue_state.json').exists(), name
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
                       PYTHONPATH=config['source'], PYTHONUNBUFFERED='1')
    with (root / 'parent.log').open('a') as log:
        child = subprocess.Popen([PYTHON, '-u', '-m',
            'experiments.rachel_n512_formal_30k.run_recall_benchmark_queue',
            '--config', str(config_path)], cwd=config['source'], env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    for _ in range(100):
        if child.poll() is not None:
            raise RuntimeError(f'{name} queue exited: {child.returncode}')
        path = root / 'queue_state.json'
        state = read(path) if path.exists() else {}
        if state.get('pid') == child.pid and state.get('status') in ('running', 'waiting_for_dependency'):
            break
        time.sleep(.2)
    else:
        raise RuntimeError(f'{name} registration not observed; inspect before retry')
    ticks = Path('/proc', str(child.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
    return dict(name=name, pid=child.pid, start_ticks=ticks, config=str(config_path),
                gpu_uuid=GPU, status=state['status'], active_stage=state.get('active_stage'))


def main():
    receipt_path = NEW / 's8_and_tail_receipt.json'
    assert not receipt_path.exists(), 'One-shot attachment already attempted; inspect receipt'
    rows = {r['name']: r for r in read(NEW / 'launch_receipt.json')['queues']}
    for name in ['S6', 'S7']:
        row = rows[name]
        state = read(Path(row['config']).parent / 'queue_state.json')
        assert live(row['pid'], row['config']) or state['status'] == 'complete', name
    config_path = S8 / 'queue/config.json'
    config = read(config_path)
    assert not (config_path.parent / 'queue_state.json').exists()
    assert not config['dependencies']
    assert all(not Path(s['completion_path']).exists() for s in config['stages'])
    config['dependencies'] = [dependency('S6_all_complete', rows['S6'])]
    config['deployment'] = dict(user_authorized=True, name='S8',
        order='After S6 depth2/depth4 and their evaluations; runs on GPU0 alongside S7 if still running',
        gpu_uuid=GPU, depth_selection='Depth2 predeclared, never selected on REAL/OOD')
    save(config_path, config)
    receipt = dict(status='attaching', at=time.time(), queues=[],
                   dependencies='GPU0: S6 -> S8; GPU1: S7; old tails wait for both S8 and S7')
    save(receipt_path, receipt)
    s8 = launch('S8', config_path)
    receipt['queues'].append(s8)
    save(receipt_path, receipt)
    previous = s8
    for number in range(1, 4):
        old_path = OLD / f'original_tail_{number}' / 'config.json'
        old = read(old_path)
        state = read(old_path.parent / 'queue_state.json')
        assert all(s['status'] == 'queued' for s in state['stages'])
        assert not live(state['pid'], str(old_path))
        new = copy.deepcopy(old)
        root = NEW / 'queues' / f'original_tail_{number}'
        root.mkdir()
        new['root'] = str(root)
        if number == 1:
            new['dependencies'] = [d for d in old['dependencies'] if d['name'] != 'new_S3_S4_S5_complete']
            new['dependencies'] += [dependency('S8_complete', s8), dependency('S7_complete', rows['S7'])]
        else:
            replace_name = 'continuation_to020_with_all_frozen_evaluations' if number == 2 else 'after020_Tstage_to050'
            new['dependencies'] = [dependency(d['name'], previous) if d['name'] == replace_name else d
                                   for d in old['dependencies']]
        assert new['stages'] == old['stages']
        save(root / 'config.json', new)
        previous = launch(f'tail{number}', root / 'config.json')
        previous['supersedes_unstarted_queue'] = str(old_path)
        receipt['queues'].append(previous)
        save(receipt_path, receipt)
    receipt['status'] = 'S8_and_all_original_tails_registered'
    save(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
