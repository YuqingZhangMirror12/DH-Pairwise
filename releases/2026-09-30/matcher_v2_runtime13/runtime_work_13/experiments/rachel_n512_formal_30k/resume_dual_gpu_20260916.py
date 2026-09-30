"""One-shot user-authorized S6 resume and independent S7 launch after reboot."""
import copy
import json
import os
from pathlib import Path
import subprocess
import time

R = Path('/root/autodl-tmp/rachel_score_design_20260913_001')
N = R / 's6_s7_20260915'
OLD = N / 'priority_after_s5/queues'
NEW = N / 'dual_gpu_20260916'
PYTHON = '/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python'
GPUS = ['GPU-c1b7c89a-0519-b37f-6885-f046419eee28',
        'GPU-09b5433b-5369-6551-6bed-6538fcd7b858']


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


def launch(name, config_path, gpu, resume):
    cfg = read(config_path)
    root = Path(cfg['root'])
    command = [PYTHON, '-u', '-m',
               'experiments.rachel_n512_formal_30k.run_recall_benchmark_queue',
               '--config', str(config_path)]
    if resume:
        command.append('--resume')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       PYTHONPATH=cfg['source'], PYTHONUNBUFFERED='1')
    with (root / 'dual_gpu_parent.log').open('a') as log:
        child = subprocess.Popen(command, cwd=cfg['source'], env=environment,
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):
        if child.poll() is not None:
            raise RuntimeError(f'{name} queue exited: {child.returncode}')
        state_path = root / 'queue_state.json'
        state = read(state_path) if state_path.exists() else {}
        if state.get('pid') == child.pid and state.get('status') in ('running', 'waiting_for_dependency'):
            break
        time.sleep(.2)
    else:
        raise RuntimeError(f'{name} queue registration not observed; do not blindly relaunch')
    ticks = Path('/proc', str(child.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
    return dict(name=name, pid=child.pid, start_ticks=ticks, gpu_uuid=gpu,
                config=str(config_path), status=state['status'],
                active_stage=state.get('active_stage'), active_child=state.get('active_child'))


def main():
    if NEW.exists():
        raise RuntimeError('one-shot launch directory already exists; inspect receipt before recovery')
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader'], text=True).strip()
    if compute:
        raise RuntimeError('GPU compute process already running; inspect before launching')
    visible = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                      '--format=csv,noheader'], text=True).splitlines()
    assert set(GPUS).issubset(set(visible)), visible
    s6path = OLD / 's6_depth_2_4/config.json'
    s6 = read(s6path)
    s6state = read(s6path.parent / 'queue_state.json')
    assert s6state['config'] == s6
    assert s6state['active_stage'] == 's6_depth2_C13_C20'
    assert read(R / 'attention_depth_20260915/s4_cross_attention_depth2/training/status.json')['status'] == 'interrupted'
    for name in ['s6_depth_2_4', 's7_augmented_full24', 'original_tail_1', 'original_tail_2', 'original_tail_3']:
        cp = OLD / name / 'config.json'
        state = read(cp.parent / 'queue_state.json')
        assert not live(state['pid'], str(cp)), name
        child = state.get('active_child')
        assert not child or not live(child['pid'], child['marker']), name
    s5dependency = s6['dependencies'][0]
    assert read(s5dependency['status_path'])['status'] == 'complete'
    old_s7path = OLD / 's7_augmented_full24/config.json'
    old_s7 = read(old_s7path)
    old_s7state = read(old_s7path.parent / 'queue_state.json')
    assert all(s['status'] == 'queued' for s in old_s7state['stages'])
    assert all(not Path(s['completion_path']).exists() for s in old_s7['stages'])
    NEW.mkdir()
    s7root = NEW / 'queues/s7_gpu1'
    s7root.mkdir(parents=True)
    s7 = copy.deepcopy(old_s7)
    s7['root'] = str(s7root)
    s7['dependencies'] = [copy.deepcopy(s5dependency)]
    # Only scheduling changes: all training/evaluation commands and outputs stay intact.
    assert s7['stages'] == old_s7['stages']
    save(s7root / 'config.json', s7)
    receipt = dict(status='launching', authorized_by='User 2026-09-16: restart complete; run S6 and S7 on separate GPUs',
                   at=time.time(), old_s7_queue_superseded=str(old_s7path),
                   s6_resume_exposure=438000, queues=[],
                   original_tails='Preserved paused until S5-2048 attention extension is attached after S6; must join S6/S7 before running')
    save(NEW / 'launch_receipt.json', receipt)
    for name, path, gpu, resume in [('S6', s6path, GPUS[0], True),
                                    ('S7', s7root / 'config.json', GPUS[1], False)]:
        receipt['queues'].append(launch(name, path, gpu, resume))
        save(NEW / 'launch_receipt.json', receipt)
    receipt['status'] = 'dual_gpu_queues_registered'
    save(NEW / 'launch_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
