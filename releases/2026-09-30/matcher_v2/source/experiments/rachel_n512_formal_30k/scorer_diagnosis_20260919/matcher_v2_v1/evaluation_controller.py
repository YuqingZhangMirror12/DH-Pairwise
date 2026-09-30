"""One terminal evaluation pass; explicit GPUs, no selection/refitting/retries."""
import argparse
import importlib
import os
from pathlib import Path
import traceback

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.launcher import check_free, identity
from ..curriculum_training_v1.matcher_run import verify_population as verify_native
from ..curriculum_training_v1.model_adapter import require
from ..curriculum_training_v1.runtime_io import read
from ..curriculum_scorer_eval_v1.controller import execute_queue
from ..curriculum_scorer_eval_v1.entry import bind_evaluation
from .evaluate import check_preparation
from .runtime_inputs import load_inputs, MODULES
from .terminal import verified_export
from .population import SPLITS, freeze_plan


def jobs(module):
    require(module in MODULES, 'unregistered module')
    choices = ('sim_best', 'equal_budget_endpoint') if module == 'matcher' else ('sim_best', 'real_best')
    return [dict(name=choice+'_'+split, selection=choice, split=split) for choice in choices for split in SPLITS]


def command(interpreter, args, population, out, job, module):
    require(job in jobs(module), 'unregistered final evaluation job')
    return [str(interpreter), '-m', __package__+'.evaluate', '--spec', str(args.spec),
        '--population-plan', str(population), '--preparation', str(args.preparation),
        '--controller-root', str(args.controller_root), '--selection', job['selection'],
        '--split', job['split'], '--device', 'cuda:0', '--out', str(out)]


def verify_job(root, job, module):
    require(job in jobs(module), 'unregistered completed evaluation')
    root = Path(root).resolve(); out = root/job['name']; name = job['name']
    launch_path = root/(name+'_launch.json'); launch = read(launch_path)
    returned = read(root/(name+'_return.json')); controller = read(root/'controller_launch.json')
    require(launch['job'] == job and returned['returncode'] == 0
            and returned['launch_sha256'] == file_sha(launch_path), 'successful actual child return required')
    values = launch['command']
    require(__package__+'.evaluate' in values, 'wrong evaluator implementation')
    for flag, value in (('--selection', job['selection']), ('--split', job['split']), ('--out', str(out)),
                        ('--spec', controller['execution']['path']), ('--population-plan', str(root/'population_plan.json'))):
        require(values.count(flag) == 1 and values[values.index(flag)+1] == value, 'evaluation command identity differs')
    if module == 'matcher':audit = verify_native(out)
    else:
        auditor = importlib.import_module(__package__.rsplit('.', 1)[0]+'.curriculum_scorer_eval_v1.audit')
        audit = auditor.verify_population(out)
    origin = read(out/'evaluation_complete.json')['provenance']
    require(origin['selection_kind'] == job['selection'] and origin['split'] == job['split']
            and origin['checkpoint_sha256'] == controller['selected_models'][job['selection']]['checkpoint_sha256']
            and origin['population_plan_sha256'] == controller['population_plan_sha256']
            and origin['preparation_sha256'] == controller['preparation_sha256'], 'model/population identity differs')
    require(read(out/'independent_artifact_audit.json') == audit, 'independent complete artifact audit differs')
    return dict(job=job, return_sha256=file_sha(root/(name+'_return.json')), audit=audit, provenance=origin)


def run(args):
    for key in ('spec', 'preparation', 'controller_root', 'canonical_straight', 'case_plan', 'out', 'python'):
        setattr(args, key, Path(getattr(args, key)).resolve())
    inputs = load_inputs(read(args.spec)); source = inputs['source']; module = inputs['plan'].record['module']
    require(len(args.gpus) in (1, 2) and args.python.is_file(), 'explicit runtime and one/two assigned GPUs required')
    check_preparation(args.preparation, source)
    planned = jobs(module); choices = list(dict.fromkeys(j['selection'] for j in planned))
    selected = {choice:verified_export(args.controller_root, args.spec, inputs['plan'], choice)[1] for choice in choices}
    canonical = dict(path=str(args.canonical_straight), sha256=file_sha(args.canonical_straight))
    plan = freeze_plan(args.spec, canonical, args.case_plan, source)
    root = args.out; require(not root.exists(), 'new evaluation root required; no automatic repeat')
    devices = check_free(args.gpus, len(args.gpus))
    if module != 'matcher':
        package = Path(__file__).resolve().parent.parent
        bind_evaluation(package/'s7_consensus_eval_v14', package/'binary_eval_v1')
    root.mkdir(parents=True); population_path = root/'population_plan.json'; write_json(population_path, plan)
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    write_json(root/'controller_launch.json', dict(controller=identity(os.getpid()), module=module,
        execution=dict(path=str(args.spec), sha256=file_sha(args.spec)), gpu_devices=devices,
        training_controller_root=str(args.controller_root), selected_models=selected,
        population_plan_sha256=file_sha(population_path), preparation_sha256=file_sha(args.preparation),
        automatic_retry=False, jobs=planned, thresholds_refitted=False, final_pass=True))
    try:
        results = execute_queue(root, args.gpus, environment,
            lambda job, out:command(args.python, args, population_path, out, job, module),
            registered_jobs=planned, verify=lambda folder, job:verify_job(folder, job, module))
        receipt = dict(schema='matcher-v2-evaluation-controller-complete/1', status='complete',
            jobs=results, job_count=len(planned), arm=read(args.spec)['arm'], module=module,
            controller_launch_sha256=file_sha(root/'controller_launch.json'),
            population_plan_sha256=file_sha(population_path), automatic_retry=False,
            threshold_fitting=False, test_used_for_selection=False, turufan_used_for_selection=False)
        write_json(root/'evaluation_complete.json', receipt)
        write_json(root/'driver_status.json', dict(status='complete', completed=len(results), active={}, pending=[]), replace=True)
        return receipt
    except BaseException as error:
        write_json(root/'controller_failure.json', dict(status='failed', error=repr(error),
            traceback=traceback.format_exc(), automatic_retry=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('spec', 'preparation', 'controller-root', 'canonical-straight', 'case-plan', 'out', 'python'):
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    run(p.parse_args())


if __name__ == '__main__':main()
