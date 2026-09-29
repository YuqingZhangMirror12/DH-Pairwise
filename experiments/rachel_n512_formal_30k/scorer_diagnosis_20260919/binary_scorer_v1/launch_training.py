"""Queued binary Patch/stats heads, same frozen E32, no automatic restart.

External controller only. Requires the prior queue AND frozen evaluations to
finish, a verified final evaluation adapter, free GPUs, then disposable gates.
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
PREPARED = Path('/root/autodl-tmp/binary_scorer_20260927')
FORMAL_ROOT = Path('/root/autodl-tmp/s7_binary_scorer_v1_20260927')
E32_SHA = '80cac47d5bc5340df35a7a7c36ab4a3580a9eea8c99adf797ff5744cd2068b17'
PLAN_SHA = '0f8c13bbc4277052ecd6d21e335e97bf58dc8522174e2a11341c3221d50450f6'
PREDECESSORS = {
    '/root/autodl-tmp/s7_consensus_simple_v1_20260925/formal_m12':3,
    '/root/autodl-tmp/s7_consensus_simple_v1_20260925/formal_scratch_fixed':3,
    '/root/autodl-tmp/s7_consensus_threshold_v1_20260925/formal_scratch_fixed':3,
    '/root/autodl-tmp/s7_consensus_threshold_joint_e32_20260927/formal_scratch_joint':6,
}
REFERENCE = '/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt'
FROZEN_CONTROL = Path('/root/autodl-tmp/s7_consensus_threshold_v1_20260925/formal_scratch_fixed')
PARALLEL_PREDECESSORS = {k:v for k,v in PREDECESSORS.items() if 'joint_e32' not in k}
SCHEDULE_AUTHORIZATION = Path('/root/autodl-tmp/scorer_queue_20260928/authorization.json')


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
    preparation = json.loads((PREPARED / 'cpu_tests_source02_remote.json').read_text())
    if preparation.get('status') != 'cpu_tests_passed' or preparation.get('tests') != 62:
        raise ValueError('Missing binary source02 CPU preparation receipt')
    if preparation.get('errors') or preparation.get('failures'):
        raise ValueError('CPU tests failed')
    expected = preparation['source_sha256']
    original_source = PREPARED / 'training_source_02'
    actual = {str(p.relative_to(original_source)):digest(p) for p in original_source.rglob('*.py')}
    if actual != expected:
        raise ValueError('Prepared binary source changed since62 CPU tests')
    evaluation=json.loads((PREPARED/'evaluation_preparation_remote.json').read_text())
    validate_evaluation_preparation(evaluation)
    for folder,key in ((PREPARED/'binary_eval_v1','adapter_python_sha256'),
                       (PREPARED/'s7_consensus_eval_v14','common_python_sha256')):
        if {p.name:digest(p) for p in folder.glob('*.py')}!=evaluation[key]:
            raise ValueError('evaluation adapter source changed')
    if (evaluation['training_source_sha256']!=expected
            or digest(PREPARED/'real_split.json')!=PLAN_SHA
            or evaluation.get('real_plan_sha256')!=PLAN_SHA
            or digest(PREPARED/'s7_consensus_eval_v14/case_plan.json')!=evaluation.get('fixed_case_plan_sha256')):
        raise ValueError('evaluation training/real split binding differs')
    with (root / 'preparation.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        source = root / 'source'
        if not source.exists():
            shutil.copytree(original_source, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        copied = {str(p.relative_to(source)):digest(p) for p in source.rglob('*.py')}
        if copied != expected:
            raise ValueError('Formal binary source differs from prepared source')
        for rel in ('data_contract.json', 'geometry_calibration_v2/geometry_calibration.json'):
            destination = root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(ORIGINAL / rel, destination)
            if digest(destination) != digest(ORIGINAL / rel):
                raise ValueError('Original dataset/calibration must not change')
        if not (root / 'training_preparation_remote.json').exists():
            shutil.copy2(PREPARED / 'cpu_tests_source02_remote.json', root / 'training_preparation_remote.json')
        for name in ('real_split.json','evaluation_preparation_remote.json'):
            if not (root/name).exists():shutil.copy2(PREPARED/name,root/name)
            if digest(root/name)!=digest(PREPARED/name):raise ValueError('formal receipt differs:'+name)
    if shutil.disk_usage(root).free < 20 * 2**30:
        raise ValueError('Insufficient disk headroom')
    return source, preparation


def validate_evaluation_preparation(receipt):
    if (receipt.get('schema')!='binary-evaluation-preparation/1'
            or receipt.get('status')!='cpu_preparation_passed'
            or receipt.get('errors') or receipt.get('failures')
            or receipt.get('verified_variants')!=['patch','stats']
            or receipt.get('real_inference_performed') is not False
            or receipt.get('real_checkpoints_opened') is not False
            or receipt.get('source_files_unchanged') is not True
            or receipt.get('gpu_preflight') is not False or receipt.get('formal_training_started') is not False
            or receipt.get('skipped')!=0 or not isinstance(receipt.get('tests'),int) or receipt['tests']<=0):
        raise ValueError('both binary final-evaluation adapters must pass before training')
    runs=receipt.get('results',[])
    if ([r.get('variant') for r in runs]!=['patch','stats']
            or any(r.get('status')!='passed' or r.get('returncode')!=0 or r.get('tests',0)<=0
                   or r.get('errors')!=0 or r.get('failures')!=0 or r.get('skipped')!=0 for r in runs)
            or sum(r['tests'] for r in runs)!=receipt['tests']
            or any(not receipt.get(k) for k in ('adapter_python_sha256','common_python_sha256','training_source_sha256'))):
        raise ValueError('actual source-bound CPU runs for both heads are required')


def validate_frozen_real_release(spec,formal=FROZEN_CONTROL):
    """The previously authorized joint comparison also needs frozen REAL-best.

    Its independent reselection MUST NOT be written into the old training's
    selection.json. Verify its separate receipt and three final populations.
    """
    formal=Path(formal);stage=formal/'scorer';path=Path(spec['real_selection'])
    record=json.loads(path.read_text());original=json.loads((stage/'selection.json').read_text())
    if (spec.get('returncode')!=0 or record.get('schema')!='frozen-e32-real-reselection/1'
            or record.get('status')!='complete' or record.get('test_used') is not False
            or record.get('real_used') is not True or record.get('gradients_used') is not False
            or record.get('optimizer_updates')!=0 or record.get('original_training_outputs_unchanged') is not True
            or record.get('original_selection_sha256')!=digest(stage/'selection.json')
            or record.get('terminal_receipt_sha256')!=digest(formal/'training_complete.json')
            or (path.parent/'failure.json').exists()):
        raise ValueError('frozen E32 REAL reselection has not actually completed')
    curve_path=path.parent/'real_curve.json';curve=json.loads(curve_path.read_text())
    if (record.get('curve_sha256')!=digest(curve_path)
            or [r['epoch'] for r in curve]!=list(range(0,original['actual_epochs']+1,2))
            or any(r.get('model_state_unchanged') is not True for r in curve)):
        raise ValueError('frozen REAL selection omitted/changed archived epochs')
    chosen=max(curve[1:],key=lambda r:tuple(r['real_report']['key']))
    if chosen!=record.get('best'):raise ValueError('frozen REAL winner differs')
    checkpoint=stage/f"epoch_{chosen['epoch']:03d}_weights.pt"
    if Path(chosen['checkpoint']).resolve()!=checkpoint.resolve() or chosen['checkpoint_sha256']!=digest(checkpoint):
        raise ValueError('frozen REAL archive changed')
    jobs=spec.get('evaluations',[]);splits=set()
    if len(jobs)!=3:raise ValueError('frozen REAL three final populations required')
    for job in jobs:
        d=Path(job['root']);p=json.loads((d/'protocol.json').read_text());s=json.loads((d/'summary.json').read_text())
        f=json.loads((d/'prediction_complete.json').read_text());status=json.loads((d/'status.json').read_text())
        split=p['split'];threshold=(chosen['synthetic_cal_threshold'] if split=='sim_test_v14'
                                  else chosen['real_report']['thresholds'].get(split))
        if (job.get('returncode')!=0 or status.get('status')!='complete' or s.get('status')!='complete'
                or p.get('status')!='complete' or f.get('status')!='all_predictions_frozen'
                or f.get('model_state_unchanged') is not True or f.get('sha256')!=digest(d/'pair_predictions.jsonl')
                or (d/'failure.json').exists() or Path(p['checkpoint']).resolve()!=checkpoint.resolve()
                or any(r.get('checkpoint_sha256')!=chosen['checkpoint_sha256'] or r.get('selected_epoch')!=chosen['epoch']
                       or r.get('selection_kind')!='real' or r.get('split')!=split or r.get('threshold')!=threshold
                       or r.get('real_selection_sha256')!=digest(path) for r in (p,s,f))):
            raise ValueError('frozen REAL evaluation does not match independent selected archive')
        splits.add(split)
    if splits!={'sim_test_v14','dunhuang_cv','turufan'}:raise ValueError('frozen REAL splits incomplete/duplicated')
    return spec


def validate_queue_release(path,predecessors=None):
    expected=PREDECESSORS if predecessors is None else predecessors
    release=json.loads(Path(path).read_text())
    parallel = release.get('schema')=='binary-released-lane/2'
    if parallel and predecessors is None:
        authorization=json.loads(SCHEDULE_AUTHORIZATION.read_text())
        if (release.get('authorization_sha256')!=digest(SCHEDULE_AUTHORIZATION)
                or authorization.get('status')!='user_authorized'
                or authorization.get('order')!=['binary_patch','binary_stats','aggressive_scratch','joint_e32']
                or authorization.get('keep_existing_training_unchanged') is not True):
            raise ValueError('latest user-authorized parallel schedule not bound')
        branches=release.get('branches',[])
        if len(branches)!=1 or branches[0].get('formal_root') not in PARALLEL_PREDECESSORS:
            raise ValueError('one actually released existing lane required')
        expected={branches[0]['formal_root']:3}
    if release.get('schema') not in ('binary-prior-queue-release/1','binary-released-lane/2') or release.get('status')!='complete':
        raise ValueError('prior queue release not complete')
    rows=release.get('branches',[])
    if len(rows)!=len(expected) or {r['formal_root'] for r in rows}!=set(expected):
        raise ValueError('all previously queued branches, including E32 joint, must finish first')
    for row in rows:
        formal=Path(row['formal_root']);stage=formal/'scorer'
        terminal=json.loads((formal/'training_complete.json').read_text())
        selection=json.loads((stage/'selection.json').read_text())
        stage_complete=json.loads((stage/'complete.json').read_text())
        if (terminal.get('status')!='training_complete' or selection.get('status')!='selected'
                or terminal.get('binding')!=selection.get('binding')
                or stage_complete.get('status')!='stage_complete' or terminal.get('last_stage')!=stage_complete
                or {k:v for k,v in stage_complete.items() if k!='status'}!={k:v for k,v in selection.items() if k!='status'}
                or digest(stage/'best_joint.pt')!=selection.get('best_joint_sha256')
                or any((formal/n).exists() for n in ('failure.json','scorer/failure.json'))):
            raise ValueError('prior branch terminal/hash/failure check failed')
        jobs=row.get('evaluations',[])
        if len(jobs)!=expected[row['formal_root']]:raise ValueError('prior frozen evaluations missing')
        identities=set()
        for job in jobs:
            directory=Path(job['root']);protocol=json.loads((directory/'protocol.json').read_text())
            status=json.loads((directory/'status.json').read_text())
            complete=json.loads((directory/'prediction_complete.json').read_text())
            summary=json.loads((directory/'summary.json').read_text())
            if (job.get('returncode')!=0 or status.get('status')!='complete'
                    or summary.get('status')!='complete' or protocol.get('status')!='complete'
                    or complete.get('status')!='all_predictions_frozen'
                    or complete.get('model_state_unchanged') is not True
                    or complete.get('sha256')!=digest(directory/'pair_predictions.jsonl')
                    or (directory/'failure.json').exists()):raise ValueError('prior frozen evaluation unverified')
            split=protocol['split'];choice=protocol.get('selection_kind','sim')
            if choice not in ('sim','real'):raise ValueError('unknown prior selected model')
            name='best_joint.pt' if choice=='sim' else 'best_real.pt'
            checkpoint=stage/name
            selected=selection['best' if choice=='sim' else 'best_real']
            if (Path(protocol['checkpoint']).resolve()!=checkpoint.resolve()
                    or protocol.get('checkpoint_sha256')!=digest(checkpoint)
                    or protocol['checkpoint_sha256']!=selection.get(name[:-3]+'_sha256')
                    or protocol.get('selected_epoch')!=selected['epoch']
                    or any(r.get('checkpoint_sha256')!=protocol['checkpoint_sha256']
                           or r.get('selected_epoch')!=selected['epoch'] or r.get('split')!=split
                           for r in (complete,summary))):
                raise ValueError('prior evaluation belongs to another checkpoint/epoch/split')
            identities.add((choice,split))
        splits={'sim_test_v14','dunhuang_cv','turufan'}
        choices={'sim','real'} if len(jobs)==6 else {'sim'}
        if identities!={(c,s) for c in choices for s in splits}:raise ValueError('duplicate/missing frozen split or model choice')
    if predecessors is None and not parallel:
        validate_frozen_real_release(release.get('frozen_control_real',{}))
    return release


def driver(args):
    root = Path(args.root).resolve()
    arm = args.variant
    if root!=FORMAL_ROOT:raise ValueError('only the registered binary experiment root is allowed')
    lock = (root / (arm + '_controller.lock')).open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = identity(os.getpid())
    current = None
    status_file = root / ('driver_' + arm + '.json')

    def publish(status, **extra):
        save(status_file, dict(status=status, arm=arm, controller=controller,
            gpu_indices=args.gpus, updated_unix=time.time(), automatic_restarts=0,
            prior_runs_interrupted=False, final_evaluation_started=False, **extra))

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
        validate_queue_release(args.queue_release)
        gpu_check(args.gpus)
        source, prepared = source_preparation(root)
        sys.path.insert(0, str(source))
        train = importlib.import_module(PACKAGE + '.train')
        config = importlib.import_module(PACKAGE + '.config').TrainingConfig(scorer_variant=arm)
        contract = json.loads((root / 'data_contract.json').read_text())
        calibration = root / 'geometry_calibration_v2' / 'geometry_calibration.json'
        geometry = json.loads(calibration.read_text())
        if (geometry.get('status') != 'complete' or geometry.get('schema') != 's7-consensus-train-geometry/2'
                or geometry.get('contract_sha256') != digest(root / 'data_contract.json')):
            raise ValueError('TRAIN-only calibration binding differs')
        if config.world_size != 2 or config.effective_batch != 32:
            raise ValueError('Registered batch topology differs')
        imported = None
        if arm in ('patch','stats'):
            selected = ORIGINAL / 'formal_scratch' / 'matcher' / 'best_joint.pt'
            selection = json.loads((selected.parent / 'selection.json').read_text())
            complete = json.loads((selected.parent / 'complete.json').read_text())
            if (complete.get('status') != 'stage_complete' or digest(selected) != selection['best_joint_sha256']
                    or digest(selected)!=E32_SHA or selection['best']['epoch']!=32):
                raise ValueError('Selected scratch Matcher is not complete/hash-verified')
            imported = dict(path=str(selected), sha256=digest(selected), selected_epoch=selection['best']['epoch'])
        bound_args = SimpleNamespace(arm='scratch_fixed', checkpoint=REFERENCE, contract=str(root / 'data_contract.json'),
            calibration=str(calibration), preflight_steps=12,
            real_split=str(root/'real_split.json'),scorer_variant=arm,
            frozen_matcher_state=imported['path'] if imported else None,
            frozen_matcher_sha256=imported['sha256'] if imported else None)
        expected_gate = train.make_binding(bound_args, config, contract)
        bound_args.preflight_steps = 0
        expected_formal = train.make_binding(bound_args, config, contract)
        save(root / (arm + '_plan.json'), dict(status='registered_before_gpu_training',
            predecessor_release_sha256=digest(args.queue_release), frozen_matcher=imported,
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
                '-m', PACKAGE + '.train', '--arm', 'scratch_fixed', '--scorer-variant',arm,
                '--real-split',str(root/'real_split.json'),'--out', str(output), '--checkpoint', REFERENCE,
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
        validate_gate(initial, expected_gate, 'scratch_fixed')
        execute(command(gate, steps=12, resume=True), 'gate_resume', gate, source)
        replay = json.loads((gate / 'scorer' / 'preflight_resumed.json').read_text())
        validate_gate(replay, expected_gate, 'scratch_fixed', replay=True)
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
                or selection['best_real_sha256'] != digest(formal / 'scorer' / 'best_real.pt')
                or selection.get('selection_on_real') is not True or selection.get('test_used') is not False
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
    parser.add_argument('--variant', choices=('patch', 'stats'), required=True)
    parser.add_argument('--queue-release',required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--driver', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if root!=FORMAL_ROOT:raise ValueError('only the registered binary experiment root is allowed')
    validate_queue_release(args.queue_release)
    root.mkdir(parents=True, exist_ok=True)
    if args.driver:
        driver(args)
        return
    receipt = root / ('controller_launch_' + args.variant + '.json')
    if receipt.exists():
        raise ValueError('Controller already registered; inspect it instead of relaunching')
    gpu_check(args.gpus)
    command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root),
               '--variant', args.variant, '--gpus', args.gpus, '--queue-release',args.queue_release,'--driver']
    with (root / ('controller_' + args.variant + '.log')).open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    record = dict(**identity(child.pid), command=command, started_unix=time.time(),
                  root=str(root), variant=args.variant, gpus=args.gpus, restart_policy='never')
    save(receipt, record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
