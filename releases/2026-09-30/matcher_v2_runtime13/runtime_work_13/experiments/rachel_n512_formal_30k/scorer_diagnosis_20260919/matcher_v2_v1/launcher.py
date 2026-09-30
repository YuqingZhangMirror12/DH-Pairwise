"""Dedicated v2 controller; explicit idle GPUs only, single attempt, no retries."""
import argparse
import os
from pathlib import Path
import time
import traceback

import torch

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.execution import check_gate, read
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.launcher import check_free, execute, identity, verify_formal as baseline_verify_formal
from ..curriculum_training_v1.model_adapter import require
from ..curriculum_training_v1.runtime_io import checkpoint_at, check_observations
from .gradient_gate import check_gradient_receipt
from .model_runtime import check_export
from .runtime_inputs import load_inputs
from .validation import select_history


def command(interpreter, spec, out, world, mode, stop=None, gate=None, resume=False):
    require(world in (1, 2), 'unregistered GPU topology')
    values = [str(interpreter)]
    values += (['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=2', '--module']
               if world == 2 else ['-m'])
    values += [__package__+'.execution', '--spec', str(Path(spec).resolve()), '--out', str(Path(out).resolve()), '--mode', mode]
    if mode == 'gate':
        require(stop in (1, 12) and gate is None, 'gate requires1 or12 actual updates')
        values += ['--gate-stop', str(stop)]
    else:
        require(mode == 'formal' and stop is None and gate is not None, 'formal requires passed gate')
        values += ['--gate-receipt', str(Path(gate).resolve())]
    if resume:values.append('--resume')
    return values


def verify_formal(out, spec_path, plan):
    out = Path(out)
    require(not list(out.glob('failure_attempt_*.json')), 'unresolved formal failure precedes completion')
    result = baseline_verify_formal(out, spec_path, plan, 'curriculum')
    complete = read(out/'training_complete.json'); binding = complete['binding']
    final, _ = checkpoint_at(out/'checkpoints', plan.record['total_updates'], binding)
    check_observations(final['observations'], binding)
    replay = select_history(plan, final['observations'], plan.record['total_updates'])
    selection = read(Path(result['export_root'])/'selection.json')
    require(all(selection[k] == v for k, v in replay.items()), 'new-arm selection does not replay from full committed history')
    for item in complete['exports'].values():
        saved = torch.load(item['path'], map_location='cpu', weights_only=False)
        check_export(saved)
    return result


def gate_sequence(root, spec_path, world, interpreter, run_child):
    root = Path(root); full = root/'gates/uninterrupted'; resumed = root/'gates/resumed'
    run_child('gate_full12', command(interpreter, spec_path, full, world, 'gate', stop=12))
    run_child('gate_update1', command(interpreter, spec_path, resumed, world, 'gate', stop=1))
    run_child('gate_resume12', command(interpreter, spec_path, resumed, world, 'gate', stop=12, resume=True))
    binding = read(full/'binding.json')
    require(binding['run_mode'] == 'gate' and read(resumed/'binding.json') == binding, 'gate run bindings differ')
    formal = {k: v for k, v in binding.items() if k != 'run_mode'}
    proof = dict(status='passed', formal_binding_sha256=digest(formal))
    names = None
    for label, folder in [('uninterrupted', full), ('resumed_from_update1', resumed)]:
        path = folder/'gate_update12.json'; gradient = folder/'gradients_update12.json'
        proof[label] = dict(path=str(path.resolve()), sha256=file_sha(path))
        proof['gradient_'+label] = dict(path=str(gradient.resolve()), sha256=file_sha(gradient))
        current = read(gradient)['parameter_names']
        require(names is None or names == current, 'gate trained parameter identities differ')
        names = current
        check_gradient_receipt(proof['gradient_'+label], formal, names, world)
    candidate = root/'gate_candidate.json'; write_json(candidate, proof)
    check_gate(candidate, formal)  # Actual complete model/AdamW/rank RNG state, not flags.
    final = root/'gpu_gate.json'; write_json(final, proof)
    return final


def run(args):
    spec_path = args.spec.resolve(); spec = read(spec_path); inputs = load_inputs(spec)
    plan = inputs['plan']; source = inputs['source']; world = spec['topology']['world_size']
    prepared = read(args.preparation)
    require(prepared.get('schema') == 'matcher-v2-runtime-cpu-preparation/1' and prepared.get('status') == 'passed'
            and prepared.get('binding_sha256') == file_sha(source/'source_binding.json')
            and prepared.get('baseline_composition_sha256') == file_sha(source/'baseline_composition.json')
            and prepared.get('errors') == prepared.get('failures') == prepared.get('skipped') == 0
            and prepared.get('cuda_initialized') is False,
            'matching complete CPU preparation required')
    require(args.python.is_absolute() and args.python.is_file(), 'explicit Python runtime required')
    root = args.out.resolve(); require(not root.exists(), 'new output required; no automatic restart')
    devices = check_free(args.gpus, world)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, args.gpus)),
        CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    root.mkdir(parents=True)
    write_json(root/'controller_launch.json', dict(controller=identity(os.getpid()), gpu_devices=devices,
        spec_sha256=file_sha(spec_path), preparation_sha256=file_sha(args.preparation),
        arm=spec['arm'], module=spec['module'], automatic_retry=False, time_unix=time.time()))
    try:
        def run_child(phase, values):
            check_free(args.gpus, world)
            return execute(root, phase, values, environment, source)
        gate = gate_sequence(root, spec_path, world, args.python, run_child)
        if getattr(args, 'gate_only', False):
            result = dict(status='gpu_feasibility_verified', gpu_gate_sha256=file_sha(gate),
                formal_started=False, actual_formal_updates=0, gate_weights_discarded=True,
                automatic_retry=False, time_unix=time.time())
            write_json(root/'gate_only_complete.json', result)
            write_json(root/'driver_status.json', result, replace=True)
            return 0
        out = root/'formal'
        run_child('formal', command(args.python, spec_path, out, world, 'formal', gate=gate))
        result = verify_formal(out, spec_path, plan)
        write_json(root/'export_process_return.json', result)
        write_json(root/'controller_complete.json', dict(result,
            successful_return_sha256=file_sha(root/'export_process_return.json'),
            formal_return_sha256=file_sha(root/'formal_return.json'), gpu_gate_sha256=file_sha(gate)))
        write_json(root/'driver_status.json', result, replace=True)
        return 0
    except Exception as exc:
        failure = dict(status='failed', error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False, time_unix=time.time())
        write_json(root/'controller_failure.json', failure)
        write_json(root/'driver_status.json', failure, replace=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('spec', 'preparation', 'out', 'python'):p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--gate-only', action='store_true',
        help='Verify12 actual updates and update1 replay; never start a formal run.')
    return run(p.parse_args())


if __name__ == '__main__':raise SystemExit(main())
