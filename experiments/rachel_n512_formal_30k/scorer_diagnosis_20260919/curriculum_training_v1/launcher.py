"""Dedicated one-experiment controller. No free-card search or automatic retry.

The queue must first verify prior required evaluation and explicitly assign
idle physical GPU IDs. This controller never interrupts an occupied device.
It does not declare the experiment's frozen evaluation/report complete.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from .checkpoint_io import file_sha, tree_sha, write_json
from .execution import check_gate, load_inputs, read, require
from .exposure import digest


def identity(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
    return dict(pid=pid, starttime=int(fields[19]), state=fields[0], cmdline=command)


def assigned_free_devices(gpus, world, inventory, applications):
    require(len(gpus) == world and len(set(gpus)) == world
            and all(type(g) is int and g >= 0 for g in gpus), 'explicit unique physical GPU assignment required')
    available = {}
    for row in csv.reader(io.StringIO(inventory)):
        if row:
            require(len(row) == 2, 'unexpected GPU inventory format')
            available[int(row[0].strip())] = row[1].strip()
    require(all(g in available for g in gpus), 'assigned GPU missing')
    selected = {available[g] for g in gpus}; active = []
    for row in csv.reader(io.StringIO(applications)):
        if row:
            require(len(row) == 2, 'unexpected GPU application format')
            if row[0].strip() in selected:
                active.append(dict(uuid=row[0].strip(), pid=int(row[1].strip())))
    require(not active, 'assigned GPUs still have compute processes; never preempt')
    return [dict(index=g, uuid=available[g]) for g in gpus]


def check_free(gpus, world):
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    return assigned_free_devices(gpus, world, inventory, applications)


def command(interpreter, spec, out, order, world, mode, stop=None, gate=None, resume=False):
    require(world in (1, 2), 'unregistered launch topology')
    values = [str(interpreter)]
    if world == 2:
        values += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=2', '--module']
    else:
        values += ['-m']
    values += [__package__ + '.execution', '--spec', str(Path(spec).resolve()), '--out', str(Path(out).resolve()),
               '--order', order, '--mode', mode]
    if mode == 'gate':
        require(stop in (1, 12) and gate is None, 'gate must stop after1 or12 actual updates')
        values += ['--gate-stop', str(stop)]
    else:
        require(mode == 'formal' and stop is None and gate is not None, 'formal gate binding required')
        values += ['--gate-receipt', str(Path(gate).resolve())]
    if resume:
        values.append('--resume')
    return values


def verify_formal(out, spec_path, plan, order):
    """A successful process exit alone is never a completed training proof."""
    out = Path(out); complete = read(out / 'training_complete.json'); binding = complete['binding']
    require(not (out / 'failure.json').exists() and complete['status'] == 'training_complete'
            and complete['stop_reason'] == 'fixed_shared_update_budget'
            and complete['completed_updates'] == plan.record['total_updates']
            and complete['completed_exposures'] == plan.record['total_updates'] * plan.record['effective_batch']
            and binding['common_plan_sha256'] == plan.sha256 and binding['common_plan'] == plan.record
            and binding['execution_manifest_sha256'] == file_sha(spec_path)
            and binding['run_mode'] == 'formal' and binding['order'] == order,
            'formal completion/plan identity differs')
    export_root = Path(complete['export_root']).resolve()
    require(export_root.parent == out.resolve(), 'export belongs to another training root')
    exported = read(export_root / 'training_complete.json')
    require(exported == {k:v for k,v in complete.items() if k != 'export_root'}, 'export terminal differs from training terminal')
    require(file_sha(export_root / 'selection.json') == complete['selection_sha256'], 'selection changed after export')
    selection = read(export_root / 'selection.json')
    require(selection['binding'] == binding and selection['exports'] == complete['exports']
            and selection['fixed_budget_reached'] is True and selection['test_used'] is False,
            'export selection policy or model identities differ')
    names = {'sim_best', 'equal_budget_endpoint'}
    if plan.record['module'] != 'matcher':
        names.add('real_best')
    require(set(complete['exports']) == names, 'required selected models missing')
    for name, model in complete['exports'].items():
        path = Path(model['path']).resolve()
        require(path == export_root / (name + '.pt') and file_sha(path) == model['sha256'], 'actual selected model changed')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        require(saved['binding'] == binding and saved['selection_kind'] == name
                and saved['updates'] == model['update'] > 0
                and tree_sha(saved['model']) == model['model_state_sha256'], 'selected model tensor identity differs')
    return dict(status='training_complete_evaluation_pending', returncode=0,
        training_complete_sha256=file_sha(export_root / 'training_complete.json'),
        execution_training_complete_sha256=file_sha(out / 'training_complete.json'),
        export_root=str(export_root), module=plan.record['module'], order=order,
        frozen_evaluation_complete=False, gpu_releasable_by_this_receipt=False)


class ChildFailure(RuntimeError):
    pass


def execute(root, phase, values, environment, package_root):
    """One child, one returned status. No retries or implicit resumes."""
    root = Path(root); launch = root / (phase + '_launch.json'); returned = root / (phase + '_return.json')
    require(not launch.exists() and not returned.exists(), 'phase already attempted; no automatic replay')
    began = time.time()
    with (root / (phase + '.log')).open('xb') as stream:
        child = subprocess.Popen(values, cwd=package_root, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            observed = identity(child.pid)
        except FileNotFoundError:
            observed = dict(pid=child.pid, identity_unavailable_process_already_exited=True)
        record = dict(phase=phase, command=values, process=observed, started_unix=began,
            cuda_visible_devices=environment['CUDA_VISIBLE_DEVICES'], automatic_retry=False)
        write_json(launch, record)
        write_json(root / 'driver_status.json', dict(status='running_' + phase, child=record), replace=True)
        code = child.wait()
    write_json(returned, dict(phase=phase, returncode=code, elapsed_seconds=time.time()-began,
        launch_sha256=file_sha(launch), automatic_retry=False))
    if code != 0:
        raise ChildFailure('%s exited%d; preserve output and inspect before any explicit recovery' % (phase, code))
    return returned


def gate_sequence(root, spec_path, order, world, interpreter, run_child):
    root = Path(root); gates = root / 'gates'; full = gates / 'uninterrupted'; resumed = gates / 'resumed'
    run_child('gate_full12', command(interpreter, spec_path, full, order, world, 'gate', stop=12))
    run_child('gate_update1', command(interpreter, spec_path, resumed, order, world, 'gate', stop=1))
    run_child('gate_resume12', command(interpreter, spec_path, resumed, order, world, 'gate', stop=12, resume=True))
    binding = read(full / 'binding.json'); require(binding['run_mode'] == 'gate', 'not an actual gate binding')
    formal = {k:v for k,v in binding.items() if k != 'run_mode'}
    require(read(resumed / 'binding.json') == binding, 'two gate paths have different bindings')
    proof = dict(status='passed', formal_binding_sha256=digest(formal))
    for name, directory in [('uninterrupted', full), ('resumed_from_update1', resumed)]:
        path = directory / 'gate_update12.json'
        proof[name] = dict(path=str(path.resolve()), sha256=file_sha(path))
    path = root / 'gate_candidate.json'; write_json(path, proof)
    # check_gate reopens actual complete checkpoint sets; flags alone are not
    # accepted. Failed candidate receipts remain historical, not formal gates.
    check_gate(path, formal)
    final = root / 'gpu_gate.json'; write_json(final, proof)
    return final


def run(args):
    require(__package__ == 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.curriculum_training_v1',
            'run the complete isolated package namespace')
    spec_path = Path(args.spec).resolve(); spec = read(spec_path); inputs = load_inputs(spec)
    plan = inputs['plan']; world = spec['topology']['world_size']
    require(args.order == 'curriculum' or (args.order == 'mixed' and plan.record['module'] == 'matcher'), 'unregistered extra head experiment')
    prepared = read(args.preparation)
    source = Path(__file__).resolve().parent
    own = {p.name:file_sha(p) for p in source.glob('*.py')}
    require(prepared['status'] == 'passed' and prepared['source_sha256'] == own,
            'this launcher and implementation must have passed CPU preparation')
    require(Path(args.python).is_absolute() and Path(args.python).is_file(), 'explicit existing Python runtime required')
    root = Path(args.out).resolve(); require(not root.exists(), 'new controller output required; no automatic restart')
    devices = check_free(args.gpus, world)
    package_root = Path(__file__).resolve().parents[len(__package__.split('.'))]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, args.gpus)),
        CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONPATH=str(package_root),
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    root.mkdir(parents=True)
    write_json(root / 'controller_launch.json', dict(controller=identity(os.getpid()), gpu_devices=devices,
        spec_sha256=file_sha(spec_path), preparation_sha256=file_sha(args.preparation),
        order=args.order, module=plan.record['module'], automatic_retry=False, time_unix=time.time()))
    try:
        def run_child(phase, values):
            # Never treat the completion of one child as permission to compete
            # with a different job that has since acquired an assigned card.
            check_free(args.gpus, world)
            return execute(root, phase, values, environment, package_root)
        gate = gate_sequence(root, spec_path, args.order, world, args.python, run_child)
        formal = root / 'formal'
        run_child('formal', command(args.python, spec_path, formal, args.order, world, 'formal', gate=gate))
        result = verify_formal(formal, spec_path, plan, args.order)
        write_json(root / 'export_process_return.json', result)
        write_json(root / 'controller_complete.json', dict(result,
            successful_return_sha256=file_sha(root / 'export_process_return.json'),
            formal_return_sha256=file_sha(root / 'formal_return.json'), gpu_gate_sha256=file_sha(gate)))
        write_json(root / 'driver_status.json', result, replace=True)
        return 0
    except Exception as error:
        failure = dict(status='failed', error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), automatic_retry=False, time_unix=time.time())
        write_json(root / 'controller_failure.json', failure)
        write_json(root / 'driver_status.json', failure, replace=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--order', choices=('curriculum', 'mixed'), required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
