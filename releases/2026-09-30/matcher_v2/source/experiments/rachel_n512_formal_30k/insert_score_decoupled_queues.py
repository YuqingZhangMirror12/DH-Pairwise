"""Create and launch the user-requested S3/S4/S5 insertion plus unchanged tail.

Requires the scoped hold receipt. Copies the three old observer configurations
to NEW queue roots, changing dependencies only; never overwrites an experiment
or starts an already active training trajectory again.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.rachel_n512_formal_30k.prepare_score_budget_queue import (
    PYTHON, DATASET, METADATA_CHECKPOINT)

PREFIX = 'experiments.rachel_n512_formal_30k.'
TRAIN_MANIFEST = '/root/autodl-tmp/rachel_recall_benchmarks_20260911_001/data/train_e1_24k.json'


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def dependency(name, pid, config):
    return dict(name=name, pid=pid, marker=str(config),
                status_path=str(Path(config).parent / 'queue_state.json'))


def stage(name, command, output, receipt='status.json', statuses=('complete',), resume=False):
    value = dict(name=name, command=command, marker=str(output),
                 completion_path=str(Path(output) / receipt), completion_statuses=list(statuses))
    if resume:
        value['resume_arguments'] = ['--resume']
    return value


def launch(config):
    root, source = Path(config['root']), Path(config['source'])
    config_path = root / 'config.json'
    write_new(config_path, config)
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root / 'parent.log').open('x') as output:
        child = subprocess.Popen([PYTHON, '-m', PREFIX + 'run_recall_benchmark_queue',
            '--config', str(config_path)], cwd=source, env=env, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    return child.pid, config_path


def new_configurations(root, first5_pid):
    root = Path(root)
    run = root / 'new_s345_20260914'
    source, data = run / 'source', run / 'step_data'
    density_train = root / 'paired_source_density_full24_v4/data/train_n512.json'
    density_clean = root / 'paired_density_clean_eval_v4/data'
    data_stages = []
    for split in ('train', 'val', 'test'):
        output = data / split
        command = [PYTHON, '-m', PREFIX + 'materialize_step_density', '--split', split,
            '--canonical-root', DATASET, '--output', str(output)]
        if split == 'train':
            command += ['--fixed-e1-manifest', TRAIN_MANIFEST, '--paired512-manifest', str(density_train)]
        else:
            command += ['--fixed512-cache', str(density_clean / (split + '_n512'))]
        data_stages.append(stage('prepare_step_' + split, command, output, resume=True))
    data_config = dict(root=str(root / 'queues/new_s345_data'), source=str(source),
                       dependencies=[], stages=data_stages)
    stages = []
    specifications = (
        ('s3_matrix', 'matrix_cnn', 'original512'),
        ('s4_cross_attention', 'cross_attention', 'original512'),
        ('s5_control512', 'matrix_cnn', 'paired512'),
        ('s5_step3_cap2048', 'matrix_cnn', 'step3'),
    )
    for name, head, sampling in specifications:
        arm = run / name
        command = [PYTHON, '-m', PREFIX + 'train_score_decoupled',
            '--checkpoint', METADATA_CHECKPOINT, '--dataset', DATASET,
            '--train-materialized-manifest', TRAIN_MANIFEST, '--head-kind', head,
            '--sampling', sampling, '--stop-after-epoch', '20', '--microbatch', '1',
            '--effective-batch', '16', '--workers', '4']
        if name == 's4_cross_attention':
            command += ['--matcher-checkpoint', str(run / 's3_matrix/training/epoch_012.pt')]
        if sampling != 'original512':
            command += ['--density-train-manifest', str(data / 'train/manifest.json'),
                        '--clean-val-manifest', str(data / 'val/manifest.json')]
        smoke, training = arm / 'smoke32', arm / 'training'
        smoke_command = command + ['--output', str(smoke), '--smoke', '32']
        if name == 's4_cross_attention':
            smoke_command += ['--smoke-phase', 'classifier']
        stages.append(stage(name + '_smoke32', smoke_command,
                            smoke, 'smoke.json', ('smoke_complete',)))
        stages.append(stage(name + '_M12_C8', command + ['--output', str(training)], training, resume=True))
        for selection in ('fixed_epoch', 'max_f1', 'recall95'):
            for split in ('test', 'real', 'ood'):
                output = arm / 'evaluation' / selection / split
                evaluate = [PYTHON, '-m', PREFIX + 'evaluate_score_decoupled',
                    '--training-run', str(training), '--selection', selection, '--split', split,
                    '--dataset', DATASET, '--output', str(output), '--batch-size', '1', '--workers', '4']
                if split == 'real':
                    evaluate += ['--keep-ids', str(root / 'keep_ids.json')]
                if name == 's4_cross_attention':
                    evaluate += ['--baseline-evaluation', str(run / 's3_matrix/evaluation' / selection / split)]
                elif sampling == 'step3':
                    evaluate += ['--baseline-evaluation', str(run / 's5_control512/evaluation' / selection / split)]
                stages.append(stage(name + '_' + selection + '_' + split, evaluate, output, 'protocol.json'))
    gpu_config = dict(root=str(root / 'queues/new_s345_after_s2'), source=str(source),
        dependencies=[dependency('S2_first5_and_all_frozen_evaluations', first5_pid,
                                root / 'queues/candidates5/config.json')], stages=stages)
    return data_config, gpu_config


def run(root):
    root = Path(root).resolve(strict=True)
    new_root = root / 'new_s345_20260914'
    hold = json.loads((new_root / 'downstream_hold.json').read_text())
    if hold['status'] != 'held_for_user_requested_insertion':
        raise ValueError('requires completed scoped downstream hold')
    if (new_root / 'insertion_launch.json').exists():
        raise RuntimeError('insertion already launched; inspect handles, do not duplicate')
    for old in hold['observers']:
        if (Path('/proc') / str(old['pid']) / 'cmdline').exists():
            if (Path('/proc') / str(old['pid']) / 'cmdline').read_bytes():
                raise RuntimeError('held downstream observer is live')
    source = new_root / 'source'
    for module in ('train_score_decoupled.py', 'evaluate_score_decoupled.py', 'materialize_step_density.py'):
        if not (source / 'experiments/rachel_n512_formal_30k' / module).is_file():
            raise ValueError('missing insertion implementation: ' + module)
    ready = json.loads((new_root / 'deployment_ready.json').read_text())
    if ready.get('status') != 'ready' or not ready.get('cpu_tests_passed'):
        raise ValueError('requires explicit tested deployment receipt')
    first5 = json.loads((root / 'queues/candidates5/queue_state.json').read_text())
    data_config, gpu_config = new_configurations(root, first5['pid'])
    data_pid, data_path = launch(data_config)
    # Dataset preparation starts immediately on CPU. It is an explicit overall
    # prerequisite, avoiding a trainer racing an incomplete step manifest.
    # S3/S4 can start as soon as S2 ends; their inputs are already complete.
    waiting = new_root / 'wait_step_data'
    wait_stage = stage('await_complete_step_data', [PYTHON, '-m',
        PREFIX + 'insert_score_decoupled_queues', '--wait-queue', str(data_path),
        '--expected-pid', str(data_pid), '--output', str(waiting)], waiting)
    at = next(i for i, item in enumerate(gpu_config['stages'])
              if item['name'].startswith('s5_control512'))
    gpu_config['stages'].insert(at, wait_stage)
    gpu_pid, gpu_path = launch(gpu_config)
    record = dict(status='launching', data_queue=dict(pid=data_pid, config=str(data_path)),
                  inserted_queue=dict(pid=gpu_pid, config=str(gpu_path)), preserved_tail=[])
    # Make a recovery receipt even if a later observer launch fails.
    write_new(new_root / 'insertion_launch.json', record)
    previous = dict((item['name'], item['config']) for item in hold['observers'])
    continuation = copy.deepcopy(previous['continuation_to020'])
    continuation['root'] = str(root / 'queues/continuation_to020_after_new_s345')
    continuation['dependencies'].append(dependency('new_S3_S4_S5_complete', gpu_pid, gpu_path))
    # Also require the finite CPU producer to have completed, not merely its
    # manifests being readable; retained data work must not silently fail.
    continuation['dependencies'].append(dependency('new_step_data_complete', data_pid, data_path))
    cp, cpath = launch(continuation)
    record['preserved_tail'].append(dict(previous='continuation_to020', pid=cp, config=str(cpath)))
    (new_root / 'insertion_launch.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    tail = copy.deepcopy(previous['after020_s3_to050'])
    tail['root'] = str(root / 'queues/after020_Tstage_to050_after_new_s345')
    tail['dependencies'][0] = dependency('continuation_to020_with_all_frozen_evaluations', cp, cpath)
    tp, tpath = launch(tail)
    record['preserved_tail'].append(dict(previous='after020_s3_to050', pid=tp, config=str(tpath)))
    (new_root / 'insertion_launch.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    inputs = copy.deepcopy(previous['after050_inputs020_v4'])
    inputs['root'] = str(root / 'queues/after050_inputs020_v4_after_new_s345')
    inputs['dependencies'][0] = dependency('after020_Tstage_to050', tp, tpath)
    ip, ipath = launch(inputs)
    record['preserved_tail'].append(dict(previous='after050_inputs020_v4', pid=ip, config=str(ipath)))
    record.update(status='launched', original_training_and_stages_preserved=True,
                  original_S3_display_name='T-stage (candidate_dual M12+C8)',
                  new_config_count=4, new_gpu_evaluation_count=36)
    # This is this launcher's own receipt, never a training/config mutation.
    (new_root / 'insertion_launch.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False))


def wait_queue(config, pid, output):
    config, output = Path(config), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', dependency=str(config), expected_pid=pid)
    (output / 'status.json').write_text(json.dumps(record))
    while True:
        try:
            cmd = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
        except FileNotFoundError:
            cmd = b''
        alive = str(config).encode() in cmd
        status_path = config.parent / 'queue_state.json'
        current = json.loads(status_path.read_text()) if status_path.exists() else {}
        if current.get('status') == 'complete' and not alive:
            record.update(status='complete', dependency_status='complete')
            (output / 'status.json').write_text(json.dumps(record))
            return
        if current.get('status') == 'failed' or not alive:
            record.update(status='failed', dependency_status=current.get('status'))
            (output / 'status.json').write_text(json.dumps(record))
            raise RuntimeError('step-data producer stopped without completing; no training will consume it')
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    parser.add_argument('--wait-queue')
    parser.add_argument('--expected-pid', type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.wait_queue:
        if not args.expected_pid or not args.output or args.root:
            parser.error('wait mode requires expected-pid/output and no root')
        wait_queue(args.wait_queue, args.expected_pid, args.output)
    elif args.root:
        run(args.root)
    else:
        parser.error('root or wait-queue required')
