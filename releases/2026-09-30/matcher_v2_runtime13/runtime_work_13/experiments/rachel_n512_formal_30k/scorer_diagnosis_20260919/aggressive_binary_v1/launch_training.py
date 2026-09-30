"""External experiment3 controller: random Matcher, then fresh frozen-Matcher head.

This file never starts on import. A released GPU lane, full new data and tested
evaluation adapter are required before either disposable two-GPU gate runs.
"""
import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

BASE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'
PREPARED = Path('/root/autodl-tmp/aggressive_binary_20260927')
FORMAL = Path('/root/autodl-tmp/s7_aggressive_binary_v17_20260928')
DATA = Path('/root/autodl-tmp/aggressive_data_v17_full30k_20260927/dataset_03')
CONTROL = Path('/root/autodl-tmp/binary_scorer_20260927/controller_source_03/launch_training.py')
REFERENCE = '/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt'
PLAN_SHA = '0f8c13bbc4277052ecd6d21e335e97bf58dc8522174e2a11341c3221d50450f6'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in root.rglob('*.py')}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def controller_helpers():
    # This standalone tested controller supplies only device/process/queue I/O;
    # its E32 initialization and training driver are never called.
    spec = importlib.util.spec_from_file_location('binary_lane_io', CONTROL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_gate(receipt, binding, stage, replay=False):
    need(stage in ('matcher', 'scorer'), 'explicit registered stage required')
    expected = dict(status='passed', formal_training=False, updated_weights_discarded=True,
        arm='scratch_aggressive', stage=stage, updates=12, exposures=384, world_size=2,
        microbatch=8, accumulate=2, effective_batch=32,
        matcher_unchanged=stage == 'scorer', head_unchanged=stage == 'matcher',
        matcher_frozen_expected=stage == 'scorer')
    need(all(receipt.get(k) == v for k, v in expected.items()),
         'actual stage update/freeze/batch gate failed')
    hashes = receipt.get('model_state_hashes', [])
    need(len(hashes) == 2 and len(set(hashes)) == 1 and bool(hashes[0]), 'DDP state hashes differ')
    need(receipt.get('binding') == binding, 'stage gate source/data binding changed')
    if replay:
        need(receipt.get('resume_matches_uninterrupted') is True, 'update1-to12 recovery differs')


def validate_cpu(receipt, source):
    need(receipt.get('schema') == 'aggressive-binary-cpu-preparation/1'
         and receipt.get('status') == 'passed' and receipt.get('tests', 0) > 0
         and all(receipt.get(k) == 0 for k in ('errors', 'failures', 'skipped'))
         and receipt.get('source_unchanged') is True and receipt.get('head_parameters') == 34529
         and receipt.get('formal_training_started') is False and receipt.get('gpu_preflight') is False,
         'actual CPU stage/gradient/engine preparation required')
    need(inventory(source) == receipt.get('source_sha256'), 'prepared training source changed')


def terminal(stage_dir, binding, stage):
    d = Path(stage_dir)
    need(not (d / 'failure.json').exists() and not (d.parent / ('failure_' + stage + '.json')).exists(),
         'stage failure precedes terminal receipt')
    selection = read(d / 'selection.json'); complete = read(d / 'complete.json')
    need(selection.get('status') == 'selected' and complete.get('status') == 'stage_complete'
         and selection.get('binding') == binding
         and {k: v for k, v in selection.items() if k != 'status'} ==
             {k: v for k, v in complete.items() if k != 'status'}, 'stage terminal differs')
    c = binding['config']; e = selection['actual_epochs']
    need(type(e) is int and e % 2 == 0 and c['minimum_epochs'] <= e <= c['maximum_epochs']
         and selection['updates'] == e * 750 and selection['exposures'] == e * 24000
         and bool(selection.get('stop_reason')) and selection.get('test_used') is False,
         'stage budget/update/stop identity differs')
    need(sha(d / 'best_joint.pt') == selection.get('best_joint_sha256'), 'SIM selected weight changed')
    if stage == 'matcher':
        need(selection.get('selection_on_real') is False and selection.get('best_real') is None
             and selection['best']['epoch'] > 0, 'random Matcher uses trained SIM selection only')
    else:
        need(selection.get('selection_on_real') is True and selection.get('matcher_unchanged') is True
             and selection.get('best_real', {}).get('epoch', 0) > 0
             and sha(d / 'best_real.pt') == selection.get('best_real_sha256'),
             'fresh Scorer REAL selection/frozen Matcher differs')
    return selection, complete


def prepare_formal(root, helper):
    prepared = PREPARED / 'training_source_03'
    receipt_path = PREPARED / 'cpu_tests_remote_03.json'
    receipt = read(receipt_path); validate_cpu(receipt, prepared)
    need(not (DATA / 'pipeline_failure.json').exists(), 'new data pipeline failed')
    done = read(DATA / 'pipeline_complete.json')
    need(done.get('status') == 'complete' and done.get('pairs') == 30000
         and done.get('data_contract_sha256') == sha(DATA / 'data_contract.json')
         and done.get('geometry_sha256') == sha(DATA / 'geometry_calibration_v2/geometry_calibration.json'),
         'new data, audit and calibration have not completed')
    source = root / 'source'
    if not source.exists():
        shutil.copytree(prepared, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    validate_cpu(receipt, source)
    for source_path, relative in (
        (DATA / 'data_contract.json', 'data_contract.json'),
        (DATA / 'geometry_calibration_v2/geometry_calibration.json', 'geometry_calibration_v2/geometry_calibration.json'),
        (DATA / 'human_approval.json', 'review_approval.json'),
        (PREPARED / 'real_split.json', 'real_split.json'),
        (receipt_path, 'training_preparation_remote.json'),
        (PREPARED / 'evaluation_preparation_remote.json', 'evaluation_preparation_remote.json')):
        target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source_path, target)
        need(sha(target) == sha(source_path), 'formal artifact differs: ' + relative)
    need(sha(root / 'real_split.json') == PLAN_SHA, 'REAL source partition differs')
    spec = importlib.util.spec_from_file_location('aggressive_eval_entry', PREPARED / 'aggressive_binary_eval_v1/entry.py')
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    entry.validate_preparation(SimpleNamespace(root=root,
        preparation=root / 'evaluation_preparation_remote.json',
        common_source=PREPARED / 's7_consensus_eval_v14', binary_source=PREPARED / 'binary_eval_v1',
        real_plan=root / 'real_split.json', case_plan=PREPARED / 's7_consensus_eval_v14/case_plan.json'))
    need(shutil.disk_usage(root).free >= 20 * 2**30, 'training disk reserve insufficient')
    return source, receipt


def arguments(root, stage, steps=0):
    return SimpleNamespace(stage=stage, out=str(root / 'formal_scratch_aggressive'), checkpoint=REFERENCE,
        contract=str(root / 'data_contract.json'), calibration=str(root / 'geometry_calibration_v2/geometry_calibration.json'),
        review_approval=str(root / 'review_approval.json'), real_split=str(root / 'real_split.json'),
        selected_matcher_stage=str(root / 'formal_scratch_aggressive/matcher') if stage == 'scorer' else None,
        preflight_steps=steps, resume=False)


def command(args, output, resume=False):
    result = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        '-m', BASE + 'aggressive_binary_v1.runtime', '--out', str(output)]
    for key in ('stage', 'checkpoint', 'contract', 'calibration', 'review_approval', 'real_split', 'selected_matcher_stage'):
        value = getattr(args, key)
        if value is not None:
            result += ['--' + key.replace('_', '-'), str(value)]
    if args.preflight_steps:
        result += ['--preflight-steps', str(args.preflight_steps)]
    if resume:
        result.append('--resume')
    return result


def driver(args, helper):
    root = Path(args.root).resolve()
    lock = (root / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = helper.identity(os.getpid()); current = None

    def publish(status, **extra):
        helper.save(root / 'driver.json', dict(status=status, controller=controller, job=current,
            gpu_indices=args.gpus, updated_unix=time.time(), automatic_retries=0, **extra))

    def execute(cmd, phase, output, source):
        nonlocal current
        devices = helper.gpu_check(args.gpus)
        log = root / 'logs' / (phase + '.log')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with log.open('xb') as stream:
            child = subprocess.Popen(cmd, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        current = dict(**helper.identity(child.pid), command=cmd, cwd=str(source), root=str(output),
            phase=phase, gpus=args.gpus, inventory=devices, started_unix=time.time(), automatic_retries=0)
        helper.save(root / (phase + '_launch.json'), current)
        while child.poll() is None:
            publish('running_' + phase); time.sleep(10)
        helper.save(root / (phase + '_exit.json'), dict(returncode=child.returncode, job=current, time_unix=time.time()))
        need(child.returncode == 0, phase + ' failed; preserve output, no automatic retry')
        current = None

    try:
        publish('validating_preparation')
        helper.validate_queue_release(args.queue_release)
        source, prepared = prepare_formal(root, helper)
        sys.path.insert(0, str(source))
        runtime = importlib.import_module(BASE + 'aggressive_binary_v1.runtime')
        need(Path(runtime.__file__).resolve() == source / Path(*BASE.rstrip('.').split('.')) / 'aggressive_binary_v1/runtime.py',
             'wrong bound training source imported')
        config = runtime.TrainingConfig(scorer_variant='patch')
        need(config.schema == 'aggressive-binary-training/1' and config.world_size == 2 and config.effective_batch == 32,
             'registered stage topology changed')
        contract = read(root / 'data_contract.json'); formal = root / 'formal_scratch_aggressive'
        need(not formal.exists(), 'never implicitly restart an existing experiment')
        formal.mkdir()
        for stage in ('matcher', 'scorer'):
            a = arguments(root, stage, 12)
            gate_binding = runtime.make_binding(a, config, contract)
            a.preflight_steps = 0
            formal_binding = runtime.make_binding(a, config, contract)
            helper.save(root / (stage + '_plan.json'), dict(status='registered_before_stage_updates',
                gate_binding=gate_binding, formal_binding=formal_binding, gpus=args.gpus,
                predecessor_release_sha256=sha(args.queue_release), cpu_tests=prepared['tests'],
                initialization='random_matcher' if stage == 'matcher' else 'new_selected_matcher_and_new_head'))
            gate = root / 'preflight' / stage
            gate.mkdir(parents=True, exist_ok=False)
            a.preflight_steps = 12
            execute(command(a, gate), stage + '_gate', gate, source)
            first = read(gate / stage / 'preflight.json'); validate_gate(first, gate_binding, stage)
            execute(command(a, gate, True), stage + '_gate_resume', gate, source)
            replay = read(gate / stage / 'preflight_resumed.json'); validate_gate(replay, gate_binding, stage, True)
            need(first['model_state_hashes'] == replay['model_state_hashes'], 'stage recovery tensors differ')
            helper.save(root / (stage + '_gpu_gate.json'), dict(status='passed', stage=stage, updates=12,
                matcher_unchanged=stage == 'scorer', head_unchanged=stage == 'matcher',
                gate_binding=gate_binding, formal_binding=formal_binding, preflight_weights_discarded=True,
                model_state_hashes=replay['model_state_hashes'], resume_matches_uninterrupted=True))
            a.preflight_steps = 0
            need(runtime.make_binding(a, config, contract) == formal_binding, 'formal binding changed after gate')
            execute(command(a, formal), stage + '_formal', formal, source)
            selected, complete = terminal(formal / stage, formal_binding, stage)
            if stage == 'matcher':
                runtime.selected_matcher(formal / stage, formal_binding, config)
                record = read(formal / 'matcher_training_complete.json')
                need(record.get('stage') == complete and record.get('whole_experiment_complete') is False,
                     'Matcher stage is not whole experiment completion')
            else:
                record = read(formal / 'training_complete.json')
                need(record.get('status') == 'training_complete' and record.get('last_stage') == complete
                     and record.get('binding') == formal_binding, 'two-stage terminal differs')
        publish('training_complete_evaluation_pending', whole_training_complete=True)
    except BaseException as error:
        failure = dict(status='failed', error=repr(error), traceback=traceback.format_exc(), job=current,
            automatic_retries=0, recorded_unix=time.time())
        helper.save(root / 'controller_failure.json', failure); publish('failed', failure=failure)
        raise
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'gpus', 'queue-release'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--driver', action='store_true'); a = p.parse_args()
    root = Path(a.root).resolve()
    need(root == FORMAL, 'registered experiment3 root required')
    helper = controller_helpers(); helper.validate_queue_release(a.queue_release); helper.gpu_check(a.gpus)
    root.mkdir(parents=True, exist_ok=True); (root / 'logs').mkdir(exist_ok=True)
    if a.driver:
        driver(a, helper); return
    need(not (root / 'controller_launch.json').exists(), 'controller already registered; inspect, do not restart')
    cmd = [sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--gpus', a.gpus,
        '--queue-release', a.queue_release, '--driver']
    with (root / 'controller.log').open('xb') as stream:
        child = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    record = dict(controller=helper.identity(child.pid), command=cmd, gpus=a.gpus, started_unix=time.time(), automatic_retries=0)
    helper.save(root / 'controller_launch.json', record); print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
