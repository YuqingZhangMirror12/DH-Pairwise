"""One registered B-arm task through gate, training and required final reports.

No card search, retries or preemption. A separate authorized dispatcher assigns
released devices; this pipeline certifies release only after child returns and
complete immutable training/evaluation audits.
"""
import argparse
import os
from pathlib import Path
import time
import traceback

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.launcher import check_free, execute, identity
from ..curriculum_training_v1.model_adapter import require
from ..curriculum_training_v1.runtime_io import read
from ..curriculum_scorer_eval_v1.entry import bind_evaluation
from .runtime_inputs import load_inputs
from .evaluate import check_preparation
from .terminal import verified_export
from .evaluation_controller import jobs, verify_job


def commands(args, root, world):
    root = Path(root); common = ['--spec', str(args.spec), '--preparation', str(args.preparation)]
    train = [str(args.python), '-m', __package__+'.launcher', *common,
        '--out', str(root/'training'), '--python', str(args.python), '--gpus', *map(str, args.gpus)]
    evaluate = [str(args.python), '-m', __package__+'.evaluation_controller', *common,
        '--controller-root', str(root/'training'), '--canonical-straight', str(args.canonical_straight),
        '--case-plan', str(args.case_plan), '--out', str(root/'evaluation'),
        '--python', str(args.python), '--gpus', *map(str, args.gpus)]
    require(len(args.gpus) == world and len(set(args.gpus)) == world, 'assigned task topology differs')
    return train, evaluate


def verify_child(root, phase, expected):
    root = Path(root); launch = root/(phase+'_launch.json'); returned = root/(phase+'_return.json')
    made = read(launch); result = read(returned)
    require(made['phase'] == result['phase'] == phase and made['command'] == expected
            and result['returncode'] == 0 and result['launch_sha256'] == file_sha(launch),
            'actual successful bound child return required: '+phase)
    return dict(launch_sha256=file_sha(launch), return_sha256=file_sha(returned))


def verify_all(root, args, inputs, sequence):
    root = Path(root)
    require(not (root/'failure.json').exists()
            and not (root/'evaluation/controller_failure.json').exists(), 'failure precedes completion')
    subprocesses = {name:verify_child(root, name, values) for name, values in zip(('training', 'evaluation'), sequence)}
    module = inputs['plan'].record['module']; planned = jobs(module)
    for choice in dict.fromkeys(j['selection'] for j in planned):
        verified_export(root/'training', args.spec, inputs['plan'], choice)
    evaluated = read(root/'evaluation/evaluation_complete.json')
    require(evaluated['schema'] == 'matcher-v2-evaluation-controller-complete/1' and evaluated['status'] == 'complete'
            and evaluated['module'] == module and evaluated['job_count'] == len(planned)
            and [item['job'] for item in evaluated['jobs']] and {item['job']['name'] for item in evaluated['jobs']}
                == {j['name'] for j in planned}, 'all preregistered terminal jobs required')
    by_name = {item['job']['name']:item for item in evaluated['jobs']}
    require(len(by_name) == len(evaluated['jobs']) == len(planned), 'duplicate/missing evaluation returns')
    for job in planned:require(verify_job(root/'evaluation', job, module) == by_name[job['name']], 'evaluation audit changed')
    return dict(schema='matcher-v2-pipeline-complete/1', status='complete', arm=read(args.spec)['arm'], module=module,
        execution=dict(path=str(args.spec), sha256=file_sha(args.spec)), subprocesses=subprocesses,
        training_complete_sha256=file_sha(root/'training/controller_complete.json'),
        evaluation_complete_sha256=file_sha(root/'evaluation/evaluation_complete.json'),
        mandatory_frozen_evaluation_complete=True, automatic_retry=False, completed_unix=time.time(),
        gpu_release_requires_current_idle_check=True, report_delivery_still_required=True)


def run(args):
    for key in ('spec', 'preparation', 'canonical_straight', 'case_plan', 'python', 'out'):
        setattr(args, key, Path(getattr(args, key)).resolve())
    spec = read(args.spec); inputs = load_inputs(spec); source = inputs['source']
    check_preparation(args.preparation, source)
    require(args.python.is_file() and not args.out.exists(), 'explicit runtime/new task root required')
    world = spec['topology']['world_size']; devices = check_free(args.gpus, world)
    sequence = commands(args, args.out, world)
    if spec['module'] != 'matcher':
        package = Path(__file__).resolve().parent.parent
        bind_evaluation(package/'s7_consensus_eval_v14', package/'binary_eval_v1')
    root = args.out; root.mkdir(parents=True)
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES=','.join(map(str, args.gpus)), CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    write_json(root/'pipeline_launch.json', dict(controller=identity(os.getpid()), gpu_devices=devices,
        execution_sha256=file_sha(args.spec), source_binding_sha256=file_sha(source/'source_binding.json'),
        automatic_retry=False, commands=sequence, started_unix=time.time()))
    try:
        for phase, values in zip(('training', 'evaluation'), sequence):
            check_free(args.gpus, world)
            execute(root, phase, values, environment, source)
            verify_child(root, phase, values)
        result = verify_all(root, args, inputs, sequence)
        write_json(root/'complete.json', result)
        write_json(root/'driver_status.json', result, replace=True)
        return result
    except BaseException as error:
        failure = dict(status='failed', error=repr(error), traceback=traceback.format_exc(), automatic_retry=False)
        write_json(root/'failure.json', failure); write_json(root/'driver_status.json', failure, replace=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('spec', 'preparation', 'canonical-straight', 'case-plan', 'python', 'out'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    run(parser.parse_args())


if __name__ == '__main__':main()
