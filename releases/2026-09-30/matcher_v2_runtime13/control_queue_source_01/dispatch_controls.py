"""B1/B2 on verified old-lane release; never borrow B3 or joint devices.

Each arm: its locked Matcher gate/training/12 terminal evaluations, then its
own SIM-selected frozen Matcher with two fresh heads in parallel. CPU waits
are local scheduling, not repeated agent SSH status polling.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path('/root/autodl-tmp/matcher_v2_20260930')
SOURCE = ROOT/'runtime_work_13'
PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919'
PYTHON = Path('/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python')
PREPARATION = ROOT/'runtime_work_13_cpu_remote.json'
CANONICAL = ROOT/'strict_admission_checks_02/canonical.json'
BASE = Path('/root/autodl-tmp/curriculum_training_20260928/locked_plan_02/matcher_execution.json')
COMBINED = ROOT/'strict_training_admission_01/combined_admission.json'
CASES = SOURCE/PACKAGE.replace('.', '/')/'s7_consensus_eval_v14/case_plan.json'
PREDECESSOR = Path('/root/autodl-tmp/curriculum_training_20260928/dispatch_source_02')
PRIOR_QUEUE = Path('/root/autodl-tmp/scorer_queue_micro32_20260928')
SIMPLE = Path('/root/autodl-tmp/s7_consensus_simple_v1_20260925')
OUT = ROOT/'b12_control_queue_01'
LANES = (
    dict(arm='B1', gpus=[1, 2], name='simple_m12', kind='simple', root=str(PRIOR_QUEUE),
         resume=str(SIMPLE/'recovery_m12_20260928_8gpu_v2')),
    dict(arm='B2', gpus=[6, 7], name='simple_e32', kind='simple', root=str(PRIOR_QUEUE),
         resume=str(SIMPLE/'recovery_scratch_fixed_20260928_8gpu_v2')),
)
BINDINGS = {
    SOURCE/'source_binding.json': '9558af206eb70de7f9b62689ad4c9baa57587f875cc8fa309c0d3fa5719cde6c',
    PREPARATION: 'f412dcc20e26f7a1316dfdd425e177780a695e3b574bab8835420ff4141b8a4b',
    CANONICAL: 'fae68421f3fc52f958c18aa1bc075337ae401fbe5f5701a72690b7ce41acfa5a',
    BASE: 'b846706db28403a3e0a145daa64c4ceb7e00a034d8032433efa5da5c64e5134f',
    COMBINED: '07eb6da665f88088f872e64eb647ba419d3016f5c159326b7876cec21faed379',
    ROOT/'b1_matcher_locked_01/execution.json': '8f6dcc6a134753c7609ad96cf00a88d9d71e657c5ac00752bcc603302958a69a',
    ROOT/'b2_matcher_locked_01/execution.json': '5f1f277ff7d2eb3944ff25b8085619a88b9d975862776e2408f516050e85dc33',
    ROOT/'controls_plan_audit_01/complete.json': '5b3154a4af6359e93fc3107d88b17a12fda3ccde4327e365498a1dd75f71d565',
    PREDECESSOR/'dispatch_curriculum.py': '5793a3e464d5528934cfb83cd1d5efc70e71fa0bc72aa3df54dc44b096c2b672',
    PREDECESSOR/'experiment_pipeline.py': '5814ee6ba3138e99c4ad751fb4fb93772986a49dfd3bccaa521ee14a85343ecc',
    PREDECESSOR/'lock_training_plan.py': 'e45e26c2c2daef726b06af2dc1761c60eeb1c20fe84cc775beefddb3086d8307',
    ROOT/'residual_priority_hold_01/complete.json': 'ecab1677da263c25fea49fdc7aebe52adaf1c9c81ba6c75f41625d88fd937737',
    Path('/root/autodl-tmp/curriculum_training_20260928/remaining_queue_02/queue_complete.json'):
        '4f3d7aaf992ccfdf4a475bd4198e7b074b9a4d2757cb15b7c7d315bfcc0b6f03',
}
MODULES = ('matcher', 'scorer_patch', 'scorer_stats')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value, *, replace=False):
    path = Path(path); destination = path.with_name(path.name+'.tmp') if replace else path
    with destination.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    if replace:
        os.replace(destination, path)


def bound(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def own_code():
    return {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}


def check_bindings():
    for path, signature in BINDINGS.items():
        require(sha(path) == signature, 'registered input changed: ' + str(path))


def validate_lanes(lanes):
    require(lanes == LANES, 'only registered B1/B2 old-lane assignments allowed')
    devices = [g for lane in lanes for g in lane['gpus']]
    require(sorted(devices) == [1, 2, 6, 7] and len(set(devices)) == 4,
            'B3 GPU0/5 and joint GPU3/4 must remain untouched')


def paths(arm, module):
    require(arm in ('B1', 'B2') and module in MODULES, 'unregistered control task')
    stem = arm.lower()+'_'+module
    return ROOT/(stem+'_locked_01/execution.json'), ROOT/(stem+'_pipeline_01')


def task_args(spec, root, gpus):
    return SimpleNamespace(spec=Path(spec), preparation=PREPARATION, canonical_straight=CANONICAL,
        case_plan=CASES, python=PYTHON, out=Path(root), gpus=list(gpus))


def command(args):
    values = [str(PYTHON), '-m', PACKAGE+'.matcher_v2_v1.pipeline']
    for name in ('spec', 'preparation', 'canonical_straight', 'case_plan', 'python', 'out'):
        values += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    return values+['--gpus', *map(str, args.gpus)]


def task_environment(gpus):
    require(gpus and set(gpus) <= {1, 2, 6, 7}, 'unregistered control devices')
    return dict(os.environ, PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def fail_first(root):
    root = Path(root)
    failures = [root/'failure.json', root/'training/controller_failure.json', root/'evaluation/controller_failure.json']
    failures += list((root/'training/formal').glob('failure_attempt_*.json'))
    require(not any(path.exists() for path in failures), 'task failure precedes completion; no automatic retry')


def verify_pipeline(args, modules):
    pipeline, runtime = modules['pipeline'], modules['runtime_inputs']
    fail_first(args.out)
    inputs = runtime.load_inputs(read(args.spec)); actual = read(args.out/'complete.json')
    expected = pipeline.verify_all(args.out, args, inputs, pipeline.commands(args, args.out, len(args.gpus)))
    require(all(actual.get(k) == v for k, v in expected.items() if k != 'completed_unix'),
            'actual full terminal pipeline verification differs')
    return inputs


def selected_from_terminal(root, inputs):
    returned_path = Path(root)/'training/export_process_return.json'
    returned = read(returned_path)
    require(returned['returncode'] == 0, 'successful Matcher export process required')
    exports = Path(returned['export_root']); terminal = read(exports/'training_complete.json')
    signature = terminal['binding']['common_plan_sha256']
    require(signature == inputs['plan'].sha256, 'selected Matcher belongs to another control plan')
    return dict(export_root=str(exports), process_return=str(returned_path), common_plan_sha256=signature)


def compile_head(arm, module, selected, compile_api):
    spec, root = paths(arm, module)
    require(module in MODULES[1:] and not root.exists(), 'new head pipeline required')
    result = compile_api.compile_execution(BASE, arm, module, spec.parent,
        combined_admission=COMBINED if arm == 'B1' else None, selected=selected)
    require(result['arm'] == arm and result['module'] == module and result['selected_matcher'] == selected,
            'head imported a different arm/Matcher')
    require(result['topology']['world_size'] == 1 and result['topology']['microbatch'] == 32,
            'unchanged single-card effective32 head required')
    return spec, root


def previous_release(lane, predecessor):
    resume = Path(lane['resume'])
    require(not (resume/'failure.json').exists(), 'current predecessor training failed; retain its devices')
    proof = predecessor.old_release(lane)
    if proof is not None:
        require(proof['gpus'] == lane['gpus'] and proof['kind'] == 'simple', 'wrong predecessor GPU release')
        receipt = read(proof['receipt']); arm = 'm12' if lane['name'] == 'simple_m12' else 'scratch_fixed'
        require(receipt['gpus'] == ','.join(map(str, lane['gpus']))
                and receipt['branches'][0]['formal_root'] == str(SIMPLE/('formal_'+arm)),
                'released model is not the assigned old task')
    return proof


def load_modules():
    sys.path.insert(0, str(SOURCE))
    result = {name: importlib.import_module(PACKAGE+'.matcher_v2_v1.'+name)
              for name in ('pipeline', 'runtime_inputs', 'compile_execution')}
    result['launcher'] = importlib.import_module(PACKAGE+'.curriculum_training_v1.launcher')
    for module in result.values():
        require(SOURCE in Path(module.__file__).resolve().parents, 'runtime imported from an unregistered source')
    sys.path.insert(0, str(PREDECESSOR))
    spec = importlib.util.spec_from_file_location('verified_previous_release', PREDECESSOR/'dispatch_curriculum.py')
    previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
    helper = sys.modules['experiment_pipeline']
    require(Path(helper.__file__).resolve() == PREDECESSOR/'experiment_pipeline.py', 'wrong old-release helper')
    result['previous'] = previous
    return result


def runtime_preflight(modules):
    """Check real imports and already-audited plans without allocating CUDA."""
    modules['pipeline'].check_preparation(PREPARATION, SOURCE)
    audit = read(ROOT/'controls_plan_audit_01/complete.json')
    require(audit['status'] == 'passed' and audit['b1_b3_nonarchitecture_plan_identical']
            and audit['b2_b0_plan_identical'], 'completed paired exposure-plan audit required')
    plans = {}
    for arm, updates in (('B1', 31667), ('B2', 24000)):
        spec_path, _ = paths(arm, 'matcher'); spec = read(spec_path)
        require(bound(spec_path) == audit['executions'][arm], 'control spec differs from audited one')
        require(spec['arm'] == arm and spec['module'] == 'matcher' and spec['selected_matcher'] is None,
                'each control Matcher must start randomly in its own arm')
        for field in ('source_binding', 'baseline_composition', 'base_execution', 'runtime_plan', 'runtime_schedule'):
            require(bound(spec[field]['path']) == spec[field], 'control component changed: '+field)
        plan = read(spec['runtime_plan']['path']); topology = spec['topology']
        require(plan['total_updates'] == updates and plan['effective_batch'] == 32
                and plan['precision'] == 'float32' and plan['termination'] == 'fixed_shared_update_budget',
                'audited training budget/precision changed')
        require(topology == modules['runtime_inputs'].registered_topology('matcher', 16),
                'measured two-card micro16/effective32 configuration required')
        require((spec['combined_admission'] is not None) == (arm == 'B1'), 'control data assignment changed')
        plans[arm] = dict(execution=bound(spec_path), total_updates=updates, topology=topology,
                         original_exposures=768000, added_exposures=245344 if arm == 'B1' else 0)
    return dict(status='passed', imports={name:str(Path(module.__file__).resolve())
        for name, module in modules.items()}, plans=plans, nvidia_or_gpu_probe_used=False,
        immutable_bindings={str(p):signature for p, signature in BINDINGS.items()})


def run_task(arm, module, gpus, lane_root, modules, selected=None):
    work = lane_root/module; work.mkdir()
    try:
        check_bindings()
        spec, root = paths(arm, module)
        if module != 'matcher':
            spec, root = compile_head(arm, module, selected, modules['compile_execution'])
        require(not root.exists(), 'control already attempted; preserve it and do not restart')
        args = task_args(spec, root, gpus)
        modules['launcher'].check_free(gpus, len(gpus))
        values = command(args)
        modules['launcher'].execute(work, 'pipeline', values, task_environment(gpus), SOURCE)
        modules['pipeline'].verify_child(work, 'pipeline', values)
        inputs = verify_pipeline(args, modules)
        result = dict(status='complete', arm=arm, module=module, gpus=gpus,
            pipeline_complete=bound(root/'complete.json'), actual_return=bound(work/'pipeline_return.json'))
        if module == 'matcher':
            result['selected'] = selected_from_terminal(root, inputs)
        save(work/'complete.json', result)
        return result
    except BaseException as error:
        save(work/'failure.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


def run_lane(lane, out, modules, *, tick=60):
    require(tick == 60, 'local resource wait interval must remain60 seconds')
    arm = lane['arm']; root = out/arm; root.mkdir()
    try:
        save(root/'waiting.json', dict(status='waiting_for_old_training_and_required_evaluation',
            preceding_lane=lane, gpu_started=False, automatic_retry=False))
        release = None
        while release is None:
            release = previous_release(lane, modules['previous'])
            if release is None:
                time.sleep(tick)
        check_bindings()
        # A completed old population does not imply that another process has
        # not acquired its cards. The registered launcher checks again per phase.
        modules['launcher'].check_free(lane['gpus'], 2)
        save(root/'release_verified.json', dict(previous=release, checked_unix=time.time()))
        save(root/'status.json', dict(status='running_matcher', arm=arm, gpus=lane['gpus']), replace=True)
        matcher = run_task(arm, 'matcher', lane['gpus'], root, modules)
        save(root/'status.json', dict(status='running_own_frozen_matcher_heads', arm=arm, gpus=lane['gpus']), replace=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_task, arm, module, [gpu], root, modules, matcher['selected'])
                       for module, gpu in zip(MODULES[1:], lane['gpus'])]
            heads = [future.result() for future in futures]
        result = dict(status='control_arm_training_and_required_evaluations_complete', arm=arm,
            matcher=matcher, heads=heads, report_delivery_still_required=True, automatic_retry=False,
            completed_unix=time.time())
        save(root/'complete.json', result); save(root/'status.json', result, replace=True)
        return result
    except BaseException as error:
        failure = dict(status='failed', arm=arm, error=repr(error), traceback=traceback.format_exc(),
                       automatic_retry=False, existing_children_not_signalled=True)
        save(root/'failure.json', failure); save(root/'status.json', failure, replace=True)
        raise


def verify_preparation(path):
    receipt = read(path)
    require(receipt['status'] == 'passed' and receipt['tests'] >= 20
            and receipt['errors'] == receipt['failures'] == receipt['skipped'] == 0
            and receipt['source_sha256'] == own_code() and receipt['cuda_initialized'] is False,
            'exact successful CPU control-queue preparation required')
    preflight = receipt.get('runtime_preflight') or {}
    require(preflight.get('status') == 'passed'
            and preflight.get('nvidia_or_gpu_probe_used') is False
            and preflight.get('immutable_bindings') == {str(p):signature for p, signature in BINDINGS.items()},
            'actual remote CPU import/plan preflight required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU scheduler must hide CUDA')
    validate_lanes(LANES); check_bindings(); verify_preparation(args.preparation)
    require(not OUT.exists(), 'registered queue already attempted; no duplicate/restart')
    for lane in LANES:
        for module in MODULES:
            spec, root = paths(lane['arm'], module)
            require(not root.exists() and (module == 'matcher' or not spec.parent.exists()),
                    'control output/heads already exist; do not duplicate')
    modules = load_modules()
    OUT.mkdir()
    lock = (OUT/'exclusive.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(OUT/'launch.json', dict(controller=modules['launcher'].identity(os.getpid()),
        source_sha256=own_code(), preparation=bound(args.preparation),
        bindings={str(p): signature for p, signature in BINDINGS.items()},
        lanes=LANES, gpu_reserved_for_B3=[0, 5], joint_untouched=[3, 4],
        starts_formal_only_after_new_gpu_gate=True, automatic_retry=False, started_unix=time.time()))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_lane, lane, OUT, modules) for lane in LANES]
            results = [future.result() for future in futures]
        save(OUT/'complete.json', dict(status='B1_B2_matchers_heads_and_required_evaluations_complete',
            results=results, report_delivery_still_required=True, automatic_retry=False))
    except BaseException as error:
        save(OUT/'failure.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
            automatic_retry=False, existing_children_not_signalled=True))
        raise


if __name__ == '__main__':
    main()
