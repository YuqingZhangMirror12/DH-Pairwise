"""A single admitted fresh head: GPU gate, original training, required terminal.

This is a finite implementation pipeline, not a resource watcher. It is started
only for explicitly free GPUs after typed Matcher adoption. User-fixed endpoint
heads may use direct native GPU gates, without repeating CPU model preparation.
Failed phases are retained and never restarted automatically.
"""
import argparse
import os
from pathlib import Path
import time
import traceback

from . import head_execution as execution, head_launcher as training
from . import head_terminal_queue as terminal, posthoc_export as ex
from .checkpoint_scan import module, save
from .protocol import require
from .released_protocol import checked

SCHEMA = 'mixed-select-fresh-head-pipeline/1'


def inputs(args):
    require(Path(args.python).is_absolute() and Path(args.python).is_file(), 'existing original Python invocation path required')
    spec, values = execution.load_inputs(args.spec)
    training.assigned_gpus(args, spec)
    training.preparation(args, spec, values)
    # Admission before an expensive training run: resolve the original heldouts
    # and their identities, without loading models or inferring TEST.
    native_plan = module('matcher_v2_v1.population', spec['runtime']).freeze_plan(
        spec['original_execution']['path'], ex.receipt(args.canonical), args.case_plan, spec['runtime'])
    require(native_plan['pair_counts']['sim_test'] == terminal.population.COUNTS_TEST['sim_test']
            and native_plan['pair_counts']['turufan'] == terminal.population.COUNTS_TEST['turufan']
            and native_plan['pair_counts']['sim_straight_test'] == terminal.population.COUNTS_TEST['sim_straight_test'],
            'unchanged required heldout populations required')
    return spec, native_plan


def commands(args, root):
    spec = ex.read(args.spec); gpus = training.assigned_gpus(args, spec)
    common = ['--spec', str(Path(args.spec).resolve()), '--python', str(args.python)]
    train = ([str(args.python), '-m', 'model_selection_v2.head_launcher', 'driver'] + common
        + training.assignment_flags(args, spec) + training.preparation_flags(args) + ['--out', str(root/'training')])
    evaluate = [str(args.python), '-m', 'model_selection_v2.head_terminal_queue', 'driver'] + common + [
        '--gpu', str(gpus[0]),
        '--driver-root', str(root/'training'), '--canonical', str(Path(args.canonical).resolve()),
        '--case-plan', str(Path(args.case_plan).resolve()), '--out', str(root/'terminal')]
    return {'training': train, 'terminal': evaluate}


def run(args):
    root = Path(args.out).resolve(); require(not root.exists(), 'new pipeline root required; no recovery/retry')
    spec, plan = inputs(args); runtime = spec['runtime']
    gpus = training.assigned_gpus(args, spec)
    support = module('curriculum_training_v1.launcher', runtime); devices = support.check_free(gpus, len(gpus))
    external, env = terminal.environment(runtime, gpus)
    root.mkdir(parents=True); started = time.time()
    request = save(root/'request.json', dict(schema=SCHEMA, execution=ex.receipt(args.spec),
        preparation=ex.receipt(args.preparation) if args.preparation is not None else None,
        admission_mode='cpu_then_gpu' if args.preparation is not None else 'direct_native_gpu_gate',
        canonical=ex.receipt(args.canonical),
        case_plan=ex.receipt(args.case_plan), legacy_population=plan, python=str(args.python), gpus=gpus,
        gpu_devices=devices, process=support.identity(os.getpid()), started_unix=started,
        task3_overlay_applied=False, automatic_retry=False, commands=commands(args, root)))
    try:
        returns = {}
        for phase, command in commands(args, root).items():
            require(support.check_free(gpus, len(gpus)) == devices, 'assigned physical GPUs changed or became busy')
            support.execute(root, phase, command, env, external)
            returns[phase] = terminal.actual_return(root, phase, command, started, gpus)
            if phase == 'training':
                finished = ex.read(root/'training/driver_complete.json')
                require(finished.get('status') == 'training_evaluation_pending'
                        and finished.get('execution') == ex.receipt(args.spec)
                        and finished.get('training_started') is True
                        and finished.get('terminal_evaluation_started') is False,
                        'successful training process is not formal training completion')
                # Terminal admission independently verifies full native budget,
                # selection exports and CPU/GPU gate binding before any forward.
            else:
                finished = ex.read(root/'terminal/complete.json')
                require(finished.get('schema') == terminal.SCHEMA
                        and finished.get('status') == 'terminal_evaluation_complete'
                        and finished.get('execution') == ex.receipt(args.spec)
                        and finished.get('evaluation_complete') is True,
                        'successful terminal process is not full terminal completion')
        save(root/'controller_complete.json', dict(schema=SCHEMA, status='training_and_required_terminal_complete',
            request=request, phases=returns, training=ex.receipt(root/'training/driver_complete.json'),
            terminal=ex.receipt(root/'terminal/complete.json'), task3_overlay_applied=False,
            finished_unix=time.time(), process_success_not_yet_certified=True))
        return 0
    except BaseException as error:
        save(root/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
             automatic_retry=False, time_unix=time.time()))
        raise


def driver(args):
    root = Path(args.out).resolve(); require(not root.exists(), 'new pipeline driver required; no duplicate start')
    spec, _ = inputs(args); runtime = spec['runtime']
    gpus = training.assigned_gpus(args, spec)
    support = module('curriculum_training_v1.launcher', runtime); devices = support.check_free(gpus, len(gpus))
    external, env = terminal.environment(runtime, gpus)
    command = [str(args.python), '-m', 'model_selection_v2.head_pipeline', 'controller']
    for key in ('spec', 'canonical', 'case_plan'):
        command += ['--'+key.replace('_', '-'), str(Path(getattr(args, key)).resolve())]
    command += ['--python', str(args.python), '--out', str(root/'pipeline')]
    command += training.preparation_flags(args) + training.assignment_flags(args, spec)
    root.mkdir(parents=True); started = time.time()
    save(root/'driver_identity.json', dict(process=support.identity(os.getpid()), started_unix=started,
        gpu_devices=devices, execution=ex.receipt(args.spec), automatic_retry=False))
    try:
        support.execute(root, 'controller', command, env, external)
        returned = terminal.actual_return(root, 'controller', command, started, gpus)
        complete_ref = ex.receipt(root/'pipeline/controller_complete.json'); complete = checked(complete_ref)
        request = checked(complete['request'])
        require(not (root/'pipeline/failure.json').exists() and complete.get('schema') == SCHEMA
                and complete.get('status') == 'training_and_required_terminal_complete'
                and request.get('execution') == ex.receipt(args.spec)
                and request.get('commands') == commands(args, root/'pipeline')
                and set(complete['phases']) == {'training', 'terminal'}, 'complete pipeline ancestry required')
        for phase, cmd in commands(args, root/'pipeline').items():
            actual = terminal.actual_return(root/'pipeline', phase, cmd, request['started_unix'], gpus)
            require(actual == complete['phases'][phase], 'actual pipeline phase return differs')
        checked(complete['training']); checked(complete['terminal'])
        save(root/'complete.json', dict(schema=SCHEMA, status='complete', execution=ex.receipt(args.spec),
            controller_complete=complete_ref, **returned, task3_overlay_applied=False,
            training_and_required_terminal_complete=True, finished_unix=time.time()))
        return 0
    except BaseException as error:
        save(root/'driver_failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
             automatic_retry=False, time_unix=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('driver', 'controller'))
    for name in ('spec', 'canonical', 'case-plan', 'python', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    admission = parser.add_mutually_exclusive_group(required=True)
    admission.add_argument('--preparation', type=Path)
    admission.add_argument('--gpu-admission', action='store_true')
    assignment = parser.add_mutually_exclusive_group(required=True)
    assignment.add_argument('--gpu', type=int)
    assignment.add_argument('--gpus', type=int, nargs=2)
    args = parser.parse_args()
    return driver(args) if args.action == 'driver' else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
