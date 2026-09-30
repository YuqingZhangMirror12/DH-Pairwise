"""After verified B3 Matcher terminal evaluation, run its two heads in parallel.

Explicit GPU0/5 only. No preemption, retry, budget edits, live-checkpoint import,
or B1/B2/residual launch. Each child uses the independently tested v2 pipeline.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

ROOT = Path('/root/autodl-tmp/matcher_v2_20260930')
SOURCE = ROOT / 'runtime_work_13'
OUT = ROOT / 'b3_head_queue_01'
MATCHER = ROOT / 'b3_matcher_pipeline_01'
SPEC = ROOT / 'b3_matcher_locked_01/execution.json'
PREPARATION = ROOT / 'runtime_work_13_cpu_remote.json'
CANONICAL = ROOT / 'strict_admission_checks_02/canonical.json'
BASE = Path('/root/autodl-tmp/curriculum_training_20260928/locked_plan_02/matcher_execution.json')
COMBINED = ROOT / 'strict_training_admission_01/combined_admission.json'
PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919'
CASES = SOURCE / PACKAGE.replace('.', '/') / 's7_consensus_eval_v14/case_plan.json'
PYTHON = Path('/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python')
BINDINGS = {
    SOURCE / 'source_binding.json': '9558af206eb70de7f9b62689ad4c9baa57587f875cc8fa309c0d3fa5719cde6c',
    PREPARATION: 'f412dcc20e26f7a1316dfdd425e177780a695e3b574bab8835420ff4141b8a4b',
    SPEC: '4c21cb624e2b9b20e34c03f9535ecd40405437d49e1db384ecc05454eac4d37f',
    COMBINED: '07eb6da665f88088f872e64eb647ba419d3016f5c159326b7876cec21faed379',
    CANONICAL: 'fae68421f3fc52f958c18aa1bc075337ae401fbe5f5701a72690b7ce41acfa5a',
    BASE: 'b846706db28403a3e0a145daa64c4ceb7e00a034d8032433efa5da5c64e5134f',
}
LANES = (('scorer_patch', 0), ('scorer_stats', 5))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path):
    return dict(path=str(path), sha256=sha(path))


def save(path, row):
    with Path(path).open('x') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_bindings():
    for path, expected in BINDINGS.items():
        if sha(path) != expected:
            raise ValueError('Bound input changed: ' + str(path))


def pending_state(root):
    root = Path(root)
    failures = [root / 'failure.json', root / 'training/controller_failure.json',
                root / 'evaluation/controller_failure.json']
    failures += list((root / 'training/formal').glob('failure_attempt_*.json'))
    if any(p.exists() for p in failures):
        raise ValueError('Matcher failure overrides completion; no automatic retry')
    return 'ready_to_verify' if (root / 'complete.json').exists() else 'waiting'


def task_args(spec, root, gpus):
    return SimpleNamespace(spec=Path(spec), preparation=PREPARATION, canonical_straight=CANONICAL,
        case_plan=CASES, python=PYTHON, out=Path(root), gpus=list(gpus))


def child_command(args):
    command = [str(PYTHON), '-m', PACKAGE + '.matcher_v2_v1.pipeline']
    for name in ('spec', 'preparation', 'canonical_straight', 'case_plan', 'python', 'out'):
        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    return command + ['--gpus', *map(str, args.gpus)]


def verify_pipeline(args, pipeline, runtime):
    pending_state(args.out)
    inputs = runtime.load_inputs(read(args.spec))
    actual = read(args.out / 'complete.json')
    expected = pipeline.verify_all(args.out, args, inputs,
                                   pipeline.commands(args, args.out, len(args.gpus)))
    for key, value in expected.items():
        if key != 'completed_unix' and actual.get(key) != value:
            raise ValueError('Pipeline verification differs: ' + key)
    return inputs


def run_lane(item, selected, modules):
    name, gpu = item
    compile_module, pipeline, runtime, launcher = modules
    lane = OUT / name; lane.mkdir()
    try:
        check_bindings()
        locked = ROOT / ('b3_' + name + '_locked_01')
        spec = compile_module.compile_execution(BASE, 'B3', name, locked,
                                                combined_admission=COMBINED, selected=selected)
        assert spec['topology']['world_size'] == 1 and spec['topology']['microbatch'] == 32
        args = task_args(locked / 'execution.json', ROOT / ('b3_' + name + '_pipeline_01'), [gpu])
        launcher.check_free([gpu], 1)
        command = child_command(args)
        environment = dict(os.environ, PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE='1',
            CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
            OMP_NUM_THREADS='2', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        launcher.execute(lane, 'pipeline', command, environment, SOURCE)
        pipeline.verify_child(lane, 'pipeline', command)
        verify_pipeline(args, pipeline, runtime)
        result = dict(status='complete', module=name, gpu=gpu,
                      pipeline_complete=bound(args.out / 'complete.json'),
                      actual_process_return=bound(lane / 'pipeline_return.json'))
        save(lane / 'complete.json', result)
        return result
    except BaseException as error:
        save(lane / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


def main():
    check_bindings()
    modules = [importlib.import_module(PACKAGE + '.matcher_v2_v1.' + name)
               for name in ('compile_execution', 'pipeline', 'runtime_inputs')]
    launcher = importlib.import_module(PACKAGE + '.curriculum_training_v1.launcher')
    modules.append(launcher)
    OUT.mkdir(exist_ok=False)
    save(OUT / 'launch.json', dict(controller=launcher.identity(os.getpid()),
        script_sha256=sha(__file__), bindings={str(k): v for k, v in BINDINGS.items()},
        lanes=[dict(module=k, gpu=g) for k, g in LANES], automatic_retry=False, started_unix=time.time()))
    save(OUT / 'waiting.json', dict(status='waiting_for_B3_matcher_and_required_evaluation',
        matcher_root=str(MATCHER), gpus_allocated=False, automatic_retry=False))
    try:
        deadline = time.monotonic() + 48 * 3600
        while pending_state(MATCHER) != 'ready_to_verify':
            if time.monotonic() > deadline:
                raise TimeoutError('Matcher did not finish within the bounded wait')
            time.sleep(60)
        check_bindings()
        args = task_args(SPEC, MATCHER, [0, 5])
        inputs = verify_pipeline(args, modules[1], modules[2])
        returned_path = MATCHER / 'training/export_process_return.json'
        returned = read(returned_path)
        assert returned['returncode'] == 0
        export_root = Path(returned['export_root'])
        terminal = read(export_root / 'training_complete.json')
        selected = dict(export_root=str(export_root), process_return=str(returned_path),
                        common_plan_sha256=terminal['binding']['common_plan_sha256'])
        assert selected['common_plan_sha256'] == inputs['plan'].sha256
        save(OUT / 'verified_matcher.json', dict(pipeline_complete=bound(MATCHER / 'complete.json'),
                                               selected=selected))
        launcher.check_free([0, 5], 2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item: run_lane(item, selected, modules), LANES))
        save(OUT / 'complete.json', dict(status='B3_both_heads_and_required_evaluation_complete',
            results=results, automatic_retry=False, report_delivery_still_required=True, completed_unix=time.time()))
    except BaseException as error:
        save(OUT / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
