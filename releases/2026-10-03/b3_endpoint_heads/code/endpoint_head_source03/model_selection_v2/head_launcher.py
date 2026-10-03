"""One explicit new-SELECT head; original GPU gate and update machinery.

No waiting, retry, preemption, automatic second head or terminal inference.
Training completion is explicitly evaluation-pending, not paper-ready results.
"""
import argparse
import os
from pathlib import Path
import time
import traceback
import types

from . import head_execution as execution, posthoc_export as ex
from .checkpoint_scan import module, save
from .protocol import digest, require


def command(interpreter, spec, out, world, mode, stop=None, gate=None, resume=False):
    require(type(world) is int and world in (1, 2), 'registered single- or two-GPU head topology required')
    prefix = ([str(interpreter), '-m'] if world == 1 else [str(interpreter), '-m',
        'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=2', '--module'])
    values = prefix + ['model_selection_v2.head_execution', 'train',
              '--spec', str(Path(spec).resolve()), '--out', str(Path(out).resolve()), '--mode', mode]
    if mode == 'gate':
        require(type(stop) is int and stop in (1, 12) and gate is None,
                'discarded gate requires1/12 actual updates')
        values += ['--gate-stop', str(stop)]
    else:
        require(mode == 'formal' and stop is None and gate is not None,
                'formal training requires its actual GPU gate')
        require(not resume, 'formal recovery requires a separate explicit decision; no automatic resume')
        values += ['--gate-receipt', str(Path(gate).resolve())]
    if resume:
        require(stop == 12, 'only the update1-to12 gate replay is registered')
        values.append('--resume')
    return values


def gate_sequence(native_sequence, root, spec, interpreter, run_child, world=1):
    require(isinstance(native_sequence, types.FunctionType)
            and 'command' in native_sequence.__code__.co_names, 'registered native gate sequence required')
    namespace = dict(native_sequence.__globals__, command=command)
    entry = types.FunctionType(native_sequence.__code__, namespace, native_sequence.__name__,
                               native_sequence.__defaults__, native_sequence.__closure__)
    return entry(root, spec, world, interpreter, run_child)


def assigned_gpus(args, spec):
    world = spec.get('topology', {}).get('world_size', 1)
    multiple = getattr(args, 'gpus', None)
    single = getattr(args, 'gpu', None)
    require((multiple is None) != (single is None), 'one explicit GPU assignment required')
    gpus = list(multiple) if multiple is not None else [single]
    require(all(type(g) is int for g in gpus) and len(set(gpus)) == len(gpus)
            and len(gpus) == world, 'GPU assignment must equal the registered rank count')
    if world == 2:
        expected = {'scorer_patch': [0, 1], 'scorer_stats': [2, 3]}
        require(gpus == expected.get(spec['module']), 'Patch must use GPU0/1; Stats must use GPU2/3')
    else:
        require(world == 1 and gpus[0] in (0, 1), 'only explicitly assigned idle GPU0/1 allowed')
    return gpus


def preparation(args, spec, values):
    direct = getattr(args, 'gpu_admission', False)
    prior = getattr(args, 'preparation', None)
    require(direct is True if prior is None else direct is False, 'choose CPU receipt or direct GPU admission')
    if prior is not None:
        value = ex.read(prior)
        check_preparation(value, args.spec, spec, values)
        return value
    from . import head_endpoint
    adoption = values['selected_matcher_adoption']
    head_endpoint.check(adoption)
    require(spec['topology'] == head_endpoint.DUAL_TOPOLOGY
            and adoption.get('head_topology') == spec['topology'], 'direct GPU gate is bound to user dual-head request')
    return None


def assignment_flags(args, spec):
    gpus = assigned_gpus(args, spec)
    return ['--gpu', str(gpus[0])] if len(gpus) == 1 else ['--gpus', *map(str, gpus)]


def preparation_flags(args):
    return (['--preparation', str(Path(args.preparation).resolve())]
            if getattr(args, 'preparation', None) is not None else ['--gpu-admission'])


def check_preparation(prepared, spec_path, spec, values):
    require(prepared.get('schema') == 'mixed-select-head-cpu-preparation/1'
            and prepared.get('status') == 'passed' and prepared.get('execution') == ex.receipt(spec_path)
            and prepared.get('external_python') == spec['external_python']
            and prepared.get('native_inventory') == spec['native_inventory'], 'matching CPU preparation required')
    require(prepared.get('full800_n512') is True and prepared.get('cuda_initialized') is False
            and prepared.get('task3_overlay_applied') is False and prepared.get('gpu_gate_passed') is False
            and prepared.get('training_started') is False and prepared.get('model_forward_calls') == 0
            and prepared.get('optimizer_updates') == 0
            and prepared.get('original_train_ledger_sha256') == values['ledger'].sha256
            and prepared.get('original_total_updates') == values['ledger'].total_updates
            and bool(prepared.get('trainable_parameter_names')), 'CPU receipt cannot stand in for GPU admission')


def run(args):
    spec_path = Path(args.spec).resolve(); spec, values = execution.load_inputs(spec_path)
    root = Path(args.out).resolve()
    require(not root.exists(), 'new controller output required; no automatic restart')
    gpus = assigned_gpus(args, spec); world = len(gpus)
    require(Path(args.python).is_absolute() and Path(args.python).is_file(), 'explicit Python runtime required')
    prepared = preparation(args, spec, values)
    runtime = spec['runtime']
    support = module('curriculum_training_v1.launcher', runtime)
    native = module('matcher_v2_v1.launcher', runtime)
    devices = support.check_free(gpus, world)  # Failure before output creation; never preempt.
    external_root = str(Path(__file__).resolve().parent.parent)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
        CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONPATH=os.pathsep.join((external_root, runtime)),
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
        environment.pop(key, None)
    root.mkdir(parents=True)
    save(root/'controller_launch.json', dict(controller=support.identity(os.getpid()),
        execution=ex.receipt(spec_path), preparation=ex.receipt(args.preparation) if prepared is not None else None,
        admission_mode='cpu_then_gpu' if prepared is not None else 'direct_native_gpu_gate', gpu_devices=devices,
        external_python=spec['external_python'], automatic_retry=False, started_unix=time.time()))
    try:
        def child(phase, cmd):
            require(support.check_free(gpus, world) == devices, 'assigned physical GPUs changed or became busy')
            return support.execute(root, phase, cmd, environment, external_root)
        gate = gate_sequence(native.gate_sequence, root, spec_path, args.python, child, world)
        proof = ex.read(gate)
        if prepared is not None:
            require(proof['formal_binding_sha256'] == prepared['formal_binding_sha256'],
                    'GPU gate and actual CPU construction bindings differ')
        if args.gate_only:
            result = dict(status='gpu_gate_complete_formal_not_started', gate=ex.receipt(gate),
                formal_started=False, actual_formal_updates=0, task3_overlay_applied=False)
            save(root/'gate_only_complete.json', result)
            return 0
        out = root/'formal'
        returned = child('formal', command(args.python, spec_path, out, world, 'formal', gate=gate))
        actual = ex.read(returned)
        require(type(actual.get('returncode')) is int and actual['returncode'] == 0,
                'actual successful formal child return required')
        complete = ex.read(out/'training_complete.json'); binding = complete['binding']
        formal_binding = {k: v for k, v in binding.items() if k != 'run_mode'}
        require(digest(formal_binding) == proof['formal_binding_sha256'], 'formal model binding differs')
        result = native.verify_formal(out, spec_path, values['plan'])
        result = dict(result, schema='mixed-select-head-training-complete/1',
            selected_matcher_adoption=spec['selected_matcher_adoption'],
            validation_contract=spec['validation_contract'], formal_return=ex.receipt(returned),
            gpu_gate=ex.receipt(gate), task3_overlay_applied=False, terminal_evaluation_started=False,
            finished_unix=time.time())
        save(root/'controller_complete.json', result)
        return 0
    except Exception as exc:
        save(root/'controller_failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            automatic_retry=False, time_unix=time.time()))
        raise


def driver(args):
    """One parent-owned actual controller return; no detached success inference."""
    spec_path = Path(args.spec).resolve(); spec, values = execution.load_inputs(spec_path)
    prepared = preparation(args, spec, values)
    gpus = assigned_gpus(args, spec); world = len(gpus)
    require(Path(args.python).is_absolute() and Path(args.python).is_file(), 'explicit Python runtime required')
    root = Path(args.out).resolve()
    require(not root.exists(), 'preserve earlier driver output; no automatic retry')
    support = module('curriculum_training_v1.launcher', spec['runtime'])
    devices = support.check_free(gpus, world)
    external_root = str(Path(__file__).resolve().parent.parent)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
        PYTHONPATH=os.pathsep.join((external_root, spec['runtime'])), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
        environment.pop(key, None)
    command = [str(args.python), '-m', 'model_selection_v2.head_launcher', 'controller',
        '--spec', str(spec_path), '--out', str(root/'training'), '--python', str(args.python)
        ] + preparation_flags(args) + assignment_flags(args, spec)
    if args.gate_only: command.append('--gate-only')
    root.mkdir(parents=True)
    save(root/'driver_identity.json', dict(process=support.identity(os.getpid()),
        gpu_devices=devices, execution=ex.receipt(spec_path), automatic_retry=False, started_unix=time.time()))
    try:
        returned = support.execute(root, 'controller', command, environment, external_root)
        actual = ex.read(returned)
        require(type(actual.get('returncode')) is int and actual['returncode'] == 0,
                'actual successful controller return required')
        filename = 'gate_only_complete.json' if args.gate_only else 'controller_complete.json'
        complete = root/'training'/filename
        require(complete.is_file(), 'zero process return is not training or gate completion')
        save(root/'driver_complete.json', dict(status='gate_only' if args.gate_only else 'training_evaluation_pending',
            controller_complete=ex.receipt(complete), actual_controller_return=ex.receipt(returned),
            controller_launch=ex.receipt(root/'controller_launch.json'),
            execution=ex.receipt(spec_path), training_started=not args.gate_only,
            terminal_evaluation_started=False, finished_unix=time.time()))
        return 0
    except Exception as exc:
        save(root/'driver_failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            automatic_retry=False, time_unix=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('driver', 'controller'))
    for name in ('spec', 'out', 'python'):
        parser.add_argument('--'+name, type=Path, required=True)
    admission = parser.add_mutually_exclusive_group(required=True)
    admission.add_argument('--preparation', type=Path)
    admission.add_argument('--gpu-admission', action='store_true')
    assignment = parser.add_mutually_exclusive_group(required=True)
    assignment.add_argument('--gpu', type=int)
    assignment.add_argument('--gpus', type=int, nargs=2)
    parser.add_argument('--gate-only', action='store_true')
    args = parser.parse_args()
    return driver(args) if args.action == 'driver' else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
