"""Independent per-arm launcher: bound-source check, DDP gate, replay, fresh head.

User authorized using the two new GPUs in parallel on 2026-09-26. This launcher
is not part of the immutable neural-network source and never launches simple,
restarts a failed child, or imports a trained old head/optimizer.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1'
RELATIVE_PACKAGE = Path(*PACKAGE.split('.'))
ORIGINAL = Path('/root/autodl-tmp/s7_consensus_layered_v14_mergefix_20260925')
PREPARED = Path('/root/autodl-tmp/consensus_threshold_20260925')
REFERENCE = '/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttime=int(fields[19]),
                cmdline=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip())


def gpu_check(devices):
    indices = [int(s) for s in devices.split(',')]
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError('Exactly two distinct GPUs are required')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True)
    inventory = {int(parts[0]):dict(uuid=parts[1].strip(), name=parts[2].strip())
                 for parts in (line.split(',') for line in raw.splitlines() if line.strip())}
    if not set(indices) <= set(inventory):
        raise ValueError('Requested GPUs are unavailable')
    used = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    occupied = {line.split(',')[0].strip() for line in used.splitlines() if line.strip()}
    if occupied & {inventory[i]['uuid'] for i in indices}:
        raise ValueError('Requested GPUs are occupied; never interrupt another arm')
    return {i:inventory[i] for i in indices}


def validate_gate(receipt, binding, arm, replay=False):
    expected = dict(status='passed', formal_training=False, updated_weights_discarded=True,
                    arm=arm, stage='scorer', updates=12, exposures=384, world_size=2,
                    microbatch=8, accumulate=2, effective_batch=32, matcher_unchanged=True)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Gate did not satisfy the registered fresh-head protocol')
    hashes = receipt.get('model_state_hashes', [])
    if len(hashes) != 2 or len(set(hashes)) != 1 or not all(hashes):
        raise ValueError('The two GPU model hashes do not agree')
    if receipt.get('binding') != binding:
        raise ValueError('Gate source/data/geometry/Matcher binding differs')
    if replay and receipt.get('resume_matches_uninterrupted') is not True:
        raise ValueError('Update1-to12 recovery was not verified')


def source_preparation(root):
    preparation = json.loads((PREPARED / 'training_preparation_remote.json').read_text())
    if preparation.get('status') != 'cpu_preparation_passed' or preparation.get('tests') != 107:
        raise ValueError('Missing threshold CPU preparation receipt')
    if preparation.get('errors') or preparation.get('failures'):
        raise ValueError('CPU tests failed')
    expected = preparation['source_package_sha256']
    original_source = PREPARED / 'training_source_01'
    actual = {p.name:digest(p) for p in (original_source / RELATIVE_PACKAGE).glob('*.py')}
    if actual != expected:
        raise ValueError('Prepared threshold source changed since 107 CPU tests')
    with (root / 'preparation.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        source = root / 'source'
        if not source.exists():
            shutil.copytree(original_source, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        copied = {p.name:digest(p) for p in (source / RELATIVE_PACKAGE).glob('*.py')}
        if copied != expected:
            raise ValueError('Formal threshold source differs from prepared source')
        for rel in ('data_contract.json', 'geometry_calibration_v2/geometry_calibration.json'):
            destination = root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(ORIGINAL / rel, destination)
            if digest(destination) != digest(ORIGINAL / rel):
                raise ValueError('Original dataset/calibration must not change')
        if not (root / 'training_preparation_remote.json').exists():
            shutil.copy2(PREPARED / 'training_preparation_remote.json', root / 'training_preparation_remote.json')
    if shutil.disk_usage(root).free < 20 * 2**30:
        raise ValueError('Insufficient disk headroom')
    return source, preparation


def driver(args):
    root = Path(args.root).resolve()
    arm = args.arm
    lock = (root / (arm + '_controller.lock')).open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = identity(os.getpid())
    current = None
    status_file = root / ('driver_' + arm + '.json')

    def publish(status, **extra):
        save(status_file, dict(status=status, arm=arm, controller=controller,
            gpu_indices=args.gpus, updated_unix=time.time(), automatic_restarts=0,
            prior_mergefix_interrupted=False, real_evaluation_started=False, **extra))

    def execute(command, phase, output, source):
        nonlocal current
        inventory = gpu_check(args.gpus)
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)
        log = logs / (arm + '_' + phase + '.log')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus,
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2')
        with log.open('xb') as stream:
            child = subprocess.Popen(command, cwd=source, env=env, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        current = dict(**identity(child.pid), command=command, cwd=str(source),
                       root=str(output), gpus=args.gpus, inventory=inventory,
                       phase=phase, log=str(log), started_unix=time.time(), restart_policy='never')
        save(root / (arm + '_' + phase + '_launch.json'), current)
        if phase == 'formal':
            save(output / 'launch.json', current)
            save(root / ('formal_launch_' + arm + '.json'), dict(controller=controller, job=current))
        while child.poll() is None:
            publish('running_' + phase, job=current)
            time.sleep(10)
        code = child.returncode
        save(root / (arm + '_' + phase + '_exit.json'), dict(returncode=code, job=current, time_unix=time.time()))
        if code != 0:
            raise RuntimeError(phase + ' exited with code ' + str(code) + '; no automatic retry')
        current = None

    try:
        publish('validating_preparation')
        gpu_check(args.gpus)
        source, prepared = source_preparation(root)
        sys.path.insert(0, str(source))
        train = importlib.import_module(PACKAGE + '.train')
        config = importlib.import_module(PACKAGE + '.config').TrainingConfig()
        contract = json.loads((root / 'data_contract.json').read_text())
        calibration = root / 'geometry_calibration_v2' / 'geometry_calibration.json'
        geometry = json.loads(calibration.read_text())
        if (geometry.get('status') != 'complete' or geometry.get('schema') != 's7-consensus-train-geometry/2'
                or geometry.get('contract_sha256') != digest(root / 'data_contract.json')):
            raise ValueError('TRAIN-only calibration binding differs')
        if config.world_size != 2 or config.effective_batch != 32:
            raise ValueError('Registered batch topology differs')
        imported = None
        if arm == 'scratch_fixed':
            selected = ORIGINAL / 'formal_scratch' / 'matcher' / 'best_joint.pt'
            selection = json.loads((selected.parent / 'selection.json').read_text())
            complete = json.loads((selected.parent / 'complete.json').read_text())
            if complete.get('status') != 'stage_complete' or digest(selected) != selection['best_joint_sha256']:
                raise ValueError('Selected scratch Matcher is not complete/hash-verified')
            imported = dict(path=str(selected), sha256=digest(selected), selected_epoch=selection['best']['epoch'])
        bound_args = SimpleNamespace(arm=arm, checkpoint=REFERENCE, contract=str(root / 'data_contract.json'),
            calibration=str(calibration), preflight_steps=12,
            frozen_matcher_state=imported['path'] if imported else None,
            frozen_matcher_sha256=imported['sha256'] if imported else None)
        expected_gate = train.make_binding(bound_args, config, contract)
        bound_args.preflight_steps = 0
        expected_formal = train.make_binding(bound_args, config, contract)
        save(root / (arm + '_plan.json'), dict(status='registered_before_gpu_training',
            user_authorized_parallel_on_new_gpus=True, frozen_matcher=imported or 'historical S7 M12',
            old_head_imported=False, old_optimizer_imported=False, head_seed=config.head_seed,
            formal_binding=expected_formal, gate_binding=expected_gate, gpus=args.gpus,
            cpu_tests=prepared['tests'], next_arm_requires_separate_gpu_gate=True))
        gate = root / 'preflight' / ('ddp_' + arm + '_01')
        formal = root / ('formal_' + arm)
        if gate.exists() or formal.exists():
            raise ValueError('Preserve prior output; this launcher does not retry or resume automatically')
        gate.mkdir(parents=True)

        def command(output, steps=0, resume=False):
            value = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                '-m', PACKAGE + '.train', '--arm', arm, '--out', str(output), '--checkpoint', REFERENCE,
                '--contract', str(root / 'data_contract.json'), '--calibration', str(calibration)]
            if imported:
                value += ['--frozen-matcher-state', imported['path'], '--frozen-matcher-sha256', imported['sha256']]
            if steps:
                value += ['--preflight-steps', str(steps)]
            if resume:
                value.append('--resume')
            return value

        execute(command(gate, steps=12), 'gate', gate, source)
        initial = json.loads((gate / 'scorer' / 'preflight.json').read_text())
        validate_gate(initial, expected_gate, arm)
        execute(command(gate, steps=12, resume=True), 'gate_resume', gate, source)
        replay = json.loads((gate / 'scorer' / 'preflight_resumed.json').read_text())
        validate_gate(replay, expected_gate, arm, replay=True)
        if initial['model_state_hashes'] != replay['model_state_hashes']:
            raise ValueError('Replay differs from uninterrupted updated weights')
        if train.make_binding(bound_args, config, contract) != expected_formal:
            raise ValueError('Formal binding changed after gate')
        save(root / (arm + '_gpu_gate.json'), dict(status='passed', arm=arm, updates=12,
            exposures=384, matcher_unchanged=True, resume_matches_uninterrupted=True,
            model_state_hashes=replay['model_state_hashes'], gate_binding=expected_gate,
            formal_binding=expected_formal, preflight_weights_discarded=True, recorded_unix=time.time()))
        formal.mkdir()
        execute(command(formal), 'formal', formal, source)
        complete = json.loads((formal / 'training_complete.json').read_text())
        selection = json.loads((formal / 'scorer' / 'selection.json').read_text())
        if (complete.get('status') != 'training_complete' or complete['binding'] != expected_formal
                or selection['best_joint_sha256'] != digest(formal / 'scorer' / 'best_joint.pt')
                or selection.get('matcher_unchanged') is not True or not selection.get('stop_reason')):
            raise ValueError('Formal completion/selected weights did not verify')
        publish('training_complete_evaluation_pending', selection=selection,
                frozen_test_and_real_evaluation_pending=True)
    except Exception as error:
        failure = dict(status='failed', type=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc(), current_job=current, automatic_restarts=0,
                       time_unix=time.time())
        save(root / ('failure_' + arm + '_controller.json'), failure)
        publish('failed', failure=failure)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--arm', choices=('m12', 'scratch_fixed'), required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--driver', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.driver:
        driver(args)
        return
    receipt = root / ('controller_launch_' + args.arm + '.json')
    if receipt.exists():
        raise ValueError('Controller already registered; inspect it instead of relaunching')
    gpu_check(args.gpus)
    command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root),
               '--arm', args.arm, '--gpus', args.gpus, '--driver']
    with (root / ('controller_' + args.arm + '.log')).open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    record = dict(**identity(child.pid), command=command, started_unix=time.time(),
                  root=str(root), arm=args.arm, gpus=args.gpus, restart_policy='never')
    save(receipt, record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
