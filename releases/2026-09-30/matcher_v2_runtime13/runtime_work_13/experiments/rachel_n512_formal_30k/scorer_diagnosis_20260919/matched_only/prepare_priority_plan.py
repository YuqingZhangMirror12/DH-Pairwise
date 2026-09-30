"""Prepare an explicit priority plan; preparation never signals or trains.

The separate --wait-cache command is a finite CPU-only prerequisite stage.
Legacy continuations remain separate, using their original sealed commands.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from . import priority_supervisor as supervisor

R = Path('/root/autodl-tmp/rachel_score_design_20260913_001')
D = R / 'scorer_diagnosis_20260919'
SOURCE = D / 'matched_only_source_v1'
OLD = D / 'followup_source_v1'
SPECTRAL = D / 'spectral_source_v1'
ADAPT = D / 'feature_adaptation_training_source_v1'
ADAPT_PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.scorer_feature_adaptation_v1.'
PYTHON = '/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python'
ARMS = ('all_tokens', 'matched_tokens', 'edge_seed', 'edge_multi', 'matched_edges')
FEATURES = D / 'matched_only_cache_v1/formal_v1'
CANDIDATES = D / 'matched_only_stage_cache_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def process(pid, startticks):
    value = supervisor.Runtime.process(pid)
    if value is None or value['startticks'] != startticks or value['state'] in ('T', 't', 'Z', 'X'):
        raise RuntimeError('expected running/waiting identity unavailable: %s' % pid)
    return {key: value[key] for key in ('pid', 'startticks', 'argv')}


def environment(legacy=OLD, *, cpu=False):
    env = dict(PATH=os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin'), HOME='/root',
        LANG='C.UTF-8', PYTHONUNBUFFERED='1', PYTHONPATH=str(SOURCE) + ':' + str(legacy),
        CUDA_VISIBLE_DEVICES='' if cpu else '0', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for key in ('LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH'):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def stage(name, command, completion, expect, *, legacy=OLD, cpu=False):
    return dict(name=name, command=command, cwd=str(SOURCE), env=environment(legacy, cpu=cpu),
        completion=str(completion), completion_expect=expect)


def make_stages(root, cache_process):
    root = Path(root)
    out = root / 'cache_ready'
    rows = [stage('wait_S7_feature_and_candidate_cache', [PYTHON, '-m',
        'matched_only.prepare_priority_plan', '--wait-cache', '--output', str(out),
        '--pid', str(cache_process['pid']), '--startticks', str(cache_process['startticks'])],
        out / 'status.json', dict(status='complete', completed_splits=['train', 'val']), cpu=True)]
    for arm in ARMS:
        out = root / 'smokes' / arm
        command = [PYTHON, '-m', 'matched_only.smoke', '--arm', arm,
            '--train-cache', str(FEATURES / 'train'), '--output', str(out)]
        if arm in ('edge_seed', 'edge_multi'):
            command += ['--stage-cache', str(CANDIDATES / 'train')]
        rows.append(stage(arm + '_discard32', command, out / 'results.json', dict(
            status='complete', samples=32, optimizer_updates=2, physical_microbatch=16,
            effective_batch=16, formal_pair_exposures=0, weights_and_optimizer_discarded=True)))
    for arm in ARMS:
        out = root / 'training' / arm
        command = [PYTHON, '-m', 'matched_only.train', '--arm', arm,
            '--train-cache', str(FEATURES / 'train'), '--val-cache', str(FEATURES / 'val'),
            '--output', str(out)]
        if arm in ('edge_seed', 'edge_multi'):
            command += ['--train-stage-cache', str(CANDIDATES / 'train'),
                        '--val-stage-cache', str(CANDIDATES / 'val')]
        rows.append(stage(arm + '_C16_train', command, out / 'status.json', dict(status='complete',
            completed_segments=64, completed_head_epochs=16, classifier_pair_exposures=384000,
            optimizer_updates=24000, matcher_updated=False, real_ood_used=False)))
        # Only the two predeclared fixed budgets. Each endpoint reports both
        # SIMVAL operating points; no extra held-out checkpoint sweep.
        for budget in (16, 8):
            for split in ('test', 'real', 'ood'):
                destination = out / 'evaluation' / ('c%d' % budget) / split
                command = [PYTHON, '-m', 'matched_only.evaluate', '--training-run', str(out),
                    '--head-budget', str(budget), '--selection', 'fixed_epoch', '--split', split,
                    '--output', str(destination), '--device', 'cuda:0', '--batch-size', '1',
                    '--workers', '4', '--dataset', '/root/autodl-tmp/dataset_rachel_pairwise_n512_v1',
                    '--prepared-cache', '/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared',
                    '--ood-prepared', '/root/autodl-tmp/turufan_ood_pairwise_20260912_001/prepared',
                    '--translation-gt-json', '/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json']
                if split == 'real':
                    command += ['--keep-ids', str(R / 'keep_ids.json')]
                rows.append(stage('%s_C%d_%s' % (arm, budget, split), command,
                                  destination / 'protocol.json', dict(status='complete')))
    # Colleague-inspired G0/G1: same fresh CA/grid readout and auxiliary ranking;
    # only G1 may adapt its independent copied feature stem. Matcher stays frozen.
    for arm in ('G0', 'G1'):
        out = root / 'adaptation_smokes' / arm
        command = [PYTHON, '-m', ADAPT_PACKAGE + 'train', '--arm', arm,
            '--train-cache', str(FEATURES / 'train'), '--val-cache', str(FEATURES / 'val'),
            '--availability', str(D / 'scorer_feature_adaptation_v1/availability_v1'),
            '--output', str(out), '--smoke', '32']
        rows.append(stage(arm + '_discard32', command, out / 'smoke.json', dict(
            status='complete_discard_smoke', formal_training_counted=False, model_weights_discarded=True,
            checkpoint_saved=False, training=dict(ordinary_pairs=32, optimizer_updates=2,
                ordinary_physical_batch=16)), legacy=ADAPT))
    for arm in ('G0', 'G1'):
        out = root / 'adaptation_training' / arm
        command = [PYTHON, '-m', ADAPT_PACKAGE + 'train', '--arm', arm,
            '--train-cache', str(FEATURES / 'train'), '--val-cache', str(FEATURES / 'val'),
            '--availability', str(D / 'scorer_feature_adaptation_v1/availability_v1'), '--output', str(out)]
        rows.append(stage(arm + '_C16_train', command, out / 'status.json', dict(status='complete',
            completed_segments=64, completed_head_epochs=16, real_ood_used=False,
            exposures=dict(ordinary_pairs=384000, auxiliary_groups=23520, auxiliary_pairs=48448,
                pair_forwards=432448, no_aux_updates=480, optimizer_updates=24000)), legacy=ADAPT))
        for budget in (16, 8):
            for split in ('test', 'real', 'ood'):
                destination = out / 'evaluation' / ('c%d' % budget) / split
                command = [PYTHON, '-m', ADAPT_PACKAGE + 'evaluate', '--training-run', str(out),
                    '--head-budget', str(budget), '--selection', 'fixed_epoch', '--split', split,
                    '--output', str(destination), '--device', 'cuda:0', '--batch-size', '1',
                    '--workers', '4', '--dataset', '/root/autodl-tmp/dataset_rachel_pairwise_n512_v1',
                    '--prepared-cache', '/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared',
                    '--ood-prepared', '/root/autodl-tmp/turufan_ood_pairwise_20260912_001/prepared',
                    '--translation-gt-json', '/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json']
                if split == 'real':
                    command += ['--keep-ids', str(R / 'keep_ids.json')]
                rows.append(stage('%s_C%d_%s' % (arm, budget, split), command,
                    destination / 'protocol.json', dict(status='complete'), legacy=ADAPT))
    rows.append(stage('summarize_new_seven_arms', [PYTHON, '-m', 'matched_only.summarize_priority',
        '--root', str(root), '--output', str(root / 'summary')], root / 'summary/status.json',
        dict(status='complete', completed_endpoints=42, expected_endpoints=42), cpu=True))
    for phase, legacy, plan_path in (
        ('depth', OLD, D / 'followup_v1/launch_plan.json'),
        ('spectral', SPECTRAL, D / 'spectral_training_v1/gpu_v1/launch_plan.json')):
        out = root / ('retained_' + phase)
        rows.append(stage('retained_' + phase, [PYTHON, '-m', 'matched_only.legacy_tail',
            '--phase', phase, '--plan', str(plan_path), '--plan-sha256', sha(plan_path),
            '--output', str(out), '--handoff-receipt', str(root / 'handoff.json')],
            out / 'status.json', dict(status='complete', completed_stages=22 if phase == 'depth' else 4),
            legacy=legacy))
    return rows


def prepare(root):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError('priority output must be new')
    if sys.executable != PYTHON or Path(__file__).resolve().parent.parent != SOURCE:
        raise ValueError('prepare on the registered remote interpreter/source only')
    followup_plan = D / 'followup_v1/launch_plan.json'
    spectral_plan = D / 'spectral_training_v1/gpu_v1/launch_plan.json'
    queue_path = D / 'followup_v1/candidate_local/queue_status.json'
    current = read(queue_path)
    queue_process = process(119795, 934449976)
    cache_process = process(124783, 935064765)
    expected_queue = [{key: row.get(key) for key in supervisor.LEDGER_KEYS} for row in current['stages']]
    plan = dict(schema=supervisor.SCHEMA, output_root=str(root), predecessor_timeout_s=48 * 3600,
        dispatchers=[
            dict(name='old_followup_dispatcher', process=process(101639, 932414692),
                status_path=str(D / 'followup_v1/supervisor_status.json'),
                status_expect=dict(status='running', pid=101639, completed_stages=2,
                    active_name='candidate_C1_C2_queue', active_pid=119795), expected_children=[119795]),
            dict(name='old_spectral_dispatcher', process=process(103819, 932543651),
                status_path=str(D / 'spectral_training_v1/gpu_v1/after_dependencies_status.json'),
                status_expect=dict(status='waiting_dependencies', pid=103819, completed_stages=0),
                expected_children=[])],
        current_queue=dict(process=queue_process, status_path=str(queue_path),
            schema=current['schema_version'], completed_stages=20, stage_ledger=expected_queue),
        protected_processes=[process(119924, 934450472)],
        stages=make_stages(root, cache_process),
        source_bindings={str(path): sha(path) for path in sorted((SOURCE / 'matched_only').glob('*.py'))
                         if not path.name.startswith('test_')},
        scientific_priority='same S7/M12: all tokens vs endpoints vs seed vs multi-mode vs final edges',
        old_work_retained=True, original_training_untouched=True,
        colleague_G0_G1='same-anchor raw ranking; frozen stem G0 versus independent trainable stem G1')
    code_root = ADAPT / 'experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919'
    for folder in ('matched_only', 'pair_grid_readout', 'scorer_feature_adaptation_v1'):
        for path in sorted((code_root / folder).glob('*.py')):
            if not path.name.startswith('test_'):
                plan['source_bindings'][str(path)] = sha(path)
    if not (code_root / 'scorer_feature_adaptation_v1/evaluate.py').is_file():
        raise ValueError('G0/G1 endpoint adapter must be ready before dispatch is arranged')
    for path in (followup_plan, spectral_plan):
        plan['source_bindings'][str(path)] = sha(path)
    supervisor.validate_plan(plan)
    supervisor.validate_live_handoff(plan, supervisor.Runtime())
    root.mkdir(parents=True, exist_ok=False)
    supervisor.atomic_json(root / 'launch_plan.json', plan)
    return dict(status='prepared_not_executed', stages=len(plan['stages']), plan=str(root / 'launch_plan.json'))


def cache_ready(receipt, actual, pid, startticks):
    if receipt.get('pid') != pid or receipt.get('gpu_jobs_started') is not False:
        raise RuntimeError('candidate-cache writer receipt identity differs')
    if actual is not None and actual['startticks'] != startticks:
        raise RuntimeError('candidate-cache writer PID reused')
    if receipt.get('status') == 'failed':
        raise RuntimeError('candidate-cache preparation failed')
    terminal = actual is None or actual['state'] in ('Z', 'X')
    if terminal:
        if receipt.get('status') != 'complete' or receipt.get('completed_splits') != ['train', 'val']:
            raise RuntimeError('candidate-cache writer exited without complete TRAIN/VAL')
        return True
    return False


def wait_cache(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('cache wait must be CPU-only')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    status = dict(status='waiting', pid=os.getpid(), started_at=time.time())
    supervisor.atomic_json(root / 'status.json', status)
    deadline = time.monotonic() + 48 * 3600
    try:
        while not cache_ready(read(CANDIDATES / 'status.json'), supervisor.Runtime.process(args.pid), args.pid, args.startticks):
            if time.monotonic() >= deadline:
                raise TimeoutError('cache prerequisite wait expired; no process restarted')
            time.sleep(30)
        for split, size in (('train', 24000), ('val', 3000)):
            for location in (FEATURES, CANDIDATES):
                row = read(location / split / 'protocol.json')
                if row.get('status') != 'complete' or row.get('completed_pairs') != size:
                    raise RuntimeError('incomplete %s %s cache' % (location, split))
        status.update(status='complete', completed_splits=['train', 'val'])
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status['finished_at'] = time.time()
        supervisor.atomic_json(root / 'status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait-cache', action='store_true')
    parser.add_argument('--pid', type=int)
    parser.add_argument('--startticks', type=int)
    args = parser.parse_args()
    if args.wait_cache:
        wait_cache(args)
    else:
        print(json.dumps(prepare(args.output)))
