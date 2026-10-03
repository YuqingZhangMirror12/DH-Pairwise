"""One-shot matched-head terminal queue with parent-owned actual returns.

The queue never trains, waits for GPUs, restarts a phase or fits on TEST. Both
selected heads' validation reuse and CAL operating points are sealed before
the first TEST worker. Identical model states share physical predictions, with
their distinct selection/threshold provenance retained in CPU readouts.
"""
import argparse
import os
from pathlib import Path
import time
import traceback

from . import head_execution as execution, head_observations as observations
from . import head_population as population, head_readout as readout, posthoc_export as ex
from .checkpoint_scan import module, save
from .protocol import require
from .released_protocol import checked

SCHEMA = 'mixed-select-head-terminal-queue/1'
CHOICES = ('sim_best', 'real_best')


def schedule(choices):
    require(set(choices) == set(CHOICES), 'both predeclared SIM/REAL head selections required')
    owners = {}; jobs = []; mapping = {}
    for kind in CHOICES:
        origin = choices[kind]['origin']
        require(origin['selection_kind'] == kind and origin['module'] in execution.HEADS,
                'selected head origin differs')
        signature = (origin['module'], origin['model_state_sha256'], origin['matcher_state_sha256'],
                     origin['validation_contract_sha256'])
        owner = owners.setdefault(signature, kind); mapping[kind] = owner
        if owner != kind:
            continue
        for split in population.SPLITS:
            jobs.append(dict(phase=kind+'_'+split, owner=kind, split=split, pairs=population.COUNTS_TEST[split]))
    return jobs, mapping


def prepare(args, root, spec):
    """Called once by the owning controller, before any worker exists."""
    choices = {}
    for kind in CHOICES:
        folder = root/'choices'/kind
        reuse = observations.adopt(args.driver_root, args.spec, kind, folder/'validation')
        complete, audit, rows = population.reuse_artifacts(reuse, args.spec)
        plan = population.freeze(args.spec, reuse, ex.receipt(args.canonical), args.case_plan, spec['runtime'])
        plan_ref = save(folder/'population_plan.json', plan)
        operating = readout.calibration_record(rows, audit['origin'], complete['rows'])
        choices[kind] = dict(origin=audit['origin'], validation_reuse=reuse, population_plan=plan_ref,
                            operating_points=save(folder/'operating_points.json', operating))
    jobs, owners = schedule(choices)
    protocol = dict(schema=SCHEMA, status='sealed_before_test', execution=ex.receipt(args.spec),
        training_driver=str(Path(args.driver_root).resolve()), choices=choices, jobs=jobs,
        prediction_owners=owners, gpu=args.gpu, interpreter=str(args.python),
        sealed_unix=time.time(), test_used_for_selection=False, threshold_refitting=False,
        same_model_prediction_reuse=True, task3_overlay_applied=False, rotation_ensemble=False,
        automatic_retry=False)
    return save(root/'protocol.json', protocol)


def worker_command(protocol, root, job):
    choice = protocol['choices'][job['owner']]
    return [protocol['interpreter'], '-m', 'model_selection_v2.head_evaluate',
        '--spec', protocol['execution']['path'], '--driver-root', protocol['training_driver'],
        '--population-plan', choice['population_plan']['path'], '--out', str(root/'jobs'/job['phase']),
        '--selection-kind', job['owner'], '--split', job['split'], '--device', 'cuda:0']


def actual_return(root, phase, expected_command, not_before, gpu):
    """A stale complete file or zero-looking boolean is not a returned child."""
    launch_path = root/(phase+'_launch.json'); returned_path = root/(phase+'_return.json')
    launch = ex.read(launch_path); result = ex.read(returned_path)
    require(result.get('phase') == launch.get('phase') == phase
            and type(result.get('returncode')) is int and result['returncode'] == 0
            and result.get('launch_sha256') == ex.file_sha(launch_path)
            and launch.get('command') == expected_command
            and type(launch.get('started_unix')) in (int, float) and launch['started_unix'] >= not_before
            and launch.get('cuda_visible_devices') == cuda_assignment(gpu)
            and launch.get('automatic_retry') is False and result.get('automatic_retry') is False,
            'actual successful command-bound child return after lock required')
    return dict(launch=ex.receipt(launch_path), actual_return=ex.receipt(returned_path))


def verify_lock(ref, spec_path):
    protocol = checked(ref)
    require(protocol.get('schema') == SCHEMA and protocol.get('status') == 'sealed_before_test'
            and protocol['execution'] == ex.receipt(spec_path)
            and type(protocol.get('gpu')) is int and protocol['gpu'] in (0, 1, 2, 3)
            and protocol.get('test_used_for_selection') is False and protocol.get('threshold_refitting') is False
            and protocol.get('task3_overlay_applied') is False and protocol.get('rotation_ensemble') is False
            and protocol.get('automatic_retry') is False, 'registered sealed queue required')
    jobs, owners = schedule(protocol['choices'])
    require(protocol['jobs'] == jobs and protocol['prediction_owners'] == owners, 'registered physical job list differs')
    for kind, choice in protocol['choices'].items():
        complete, audit, rows = population.reuse_artifacts(choice['validation_reuse'], spec_path)
        plan = checked(choice['population_plan']); operating = checked(choice['operating_points'])
        require(audit['origin'] == choice['origin'] == plan['origin']
                and plan['validation_reuse'] == choice['validation_reuse']
                and plan['execution'] == protocol['execution']
                and operating == readout.calibration_record(rows, choice['origin'], complete['rows']),
                'choice/adopted predictions/pre-TEST operating points changed')
    return protocol


def verify_job(root, protocol_ref, job, runtime):
    protocol = verify_lock(protocol_ref, checked(protocol_ref)['execution']['path'])
    require(job in protocol['jobs'], 'unregistered worker result')
    returned = actual_return(root, job['phase'], worker_command(protocol, root, job),
                             protocol['sealed_unix'], protocol['gpu'])
    out = root/'jobs'/job['phase']; choice = protocol['choices'][job['owner']]; expected = choice['origin']
    audit_api = module('curriculum_scorer_eval_v1.audit', runtime)
    audit = audit_api.verify_population(out)
    require(audit == ex.read(out/'independent_artifact_audit.json') and audit['pairs'] == job['pairs'],
            'actual worker rows/independent numeric audit differ')
    complete = ex.read(out/'evaluation_complete.json'); origin = complete['provenance']
    keys = ('schema', 'module', 'selection_kind', 'selected_updates', 'model_state_sha256',
            'checkpoint_sha256', 'matcher_state_sha256', 'selected_matcher_adoption_sha256',
            'validation_contract_sha256', 'actual_formal_return', 'actual_controller_return')
    require(all(origin.get(k) == expected[k] for k in keys)
            and origin['population_plan'] == choice['population_plan']
            and origin['validation_reuse'] == choice['validation_reuse']
            and origin['split'] == complete['split'] == job['split']
            and origin['total_pairs'] == complete['pairs'] == job['pairs']
            and origin['matched_fresh_head'] is True and origin['gt_used_for_prediction'] is False
            and origin['task3_overlay_applied'] is False and origin['rotation_ensemble'] is False
            and complete['model_state_unchanged'] is True,
            'actual worker model/population origin differs from the lock')
    require(all(origin['thresholds'][key] == value for key, value in expected['thresholds'].items()),
            'worker changed preselected primary thresholds')
    return dict(**returned, complete=ex.receipt(out/'evaluation_complete.json'),
        artifact_audit=ex.receipt(out/'independent_artifact_audit.json'),
        predictions=ex.receipt(out/'pair_predictions.jsonl'), labeled=ex.receipt(out/'case_diagnostics.jsonl'),
        summary=ex.receipt(out/'summary.json'), pairs=job['pairs'], split=job['split'],
        physical_inference_owner=job['owner'], model_state_sha256=expected['model_state_sha256'])


def finish(root, protocol_ref, proofs, runtime):
    protocol = verify_lock(protocol_ref, checked(protocol_ref)['execution']['path'])
    require(set(proofs) == {j['phase'] for j in protocol['jobs']}, 'not all required actual workers completed')
    native_audit = module('curriculum_scorer_eval_v1.audit', runtime)
    results = {}; references = {}
    for kind in CHOICES:
        choice = protocol['choices'][kind]; origin = choice['origin']
        complete, _, validation = population.reuse_artifacts(choice['validation_reuse'], protocol['execution']['path'])
        operating = checked(choice['operating_points']); plan = checked(choice['population_plan'])
        owner = protocol['prediction_owners'][kind]; fresh = {}; links = {}
        for split in population.SPLITS:
            proof = proofs[owner+'_'+split]
            require(proof['model_state_sha256'] == origin['model_state_sha256'], 'shared predictions are not identical weights')
            checked(proof['complete']); checked(proof['actual_return'])
            require(ex.file_sha(proof['labeled']['path']) == proof['labeled']['sha256'], 'labeled predictions changed')
            fresh[split] = native_audit.read_rows(proof['labeled']['path']); links[split] = proof
        roles = checked(plan['real_split'])
        result = readout.choice_readout(origin, validation, fresh, roles, operating, runtime)
        result['physical_inference_owner'] = owner
        result['same_model_predictions_reused_from_other_choice'] = owner != kind
        result['prediction_evidence'] = links; result['operating_points'] = choice['operating_points']
        results[kind] = save(root/'choices'/kind/'readout.json', result)
        references[kind] = dict(validation=complete['rows'], operating_points=choice['operating_points'])
        require(ex.file_sha(origin['checkpoint']) == origin['checkpoint_sha256'], 'selected model bytes changed during evaluation')
    report = dict(schema=SCHEMA, status='all_required_populations_evaluated', protocol=protocol_ref,
        choices=results, pretest_evidence=references, jobs=proofs, training_performed_by_terminal_queue=False,
        test_used_for_selection=False, task3_overlay_applied=False, actual_workers_returned_zero=True,
        physical_inference_jobs=len(proofs), finished_unix=time.time(),
        process_success_not_yet_certified=True)
    return save(root/'controller_complete.json', report)


def cuda_assignment(gpu):
    values = list(gpu) if isinstance(gpu, (list, tuple)) else [gpu]
    require(len(values) in (1, 2) and len(set(values)) == len(values)
            and all(type(v) is int and v in (0, 1, 2, 3) for v in values), 'invalid explicit GPU assignment')
    return ','.join(map(str, values))


def environment(runtime, gpu):
    external = str(Path(__file__).resolve().parent.parent)
    value = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda_assignment(gpu), PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join((external, runtime)), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'): value.pop(key, None)
    return external, value


def run(args):
    spec = ex.read(args.spec); runtime = spec['runtime']; root = Path(args.out).resolve()
    require(spec.get('schema') == execution.SCHEMA and not root.exists(), 'new matched-head queue output required')
    require(type(args.gpu) is int and args.gpu in (0, 1, 2, 3), 'only explicitly assigned GPU0–3 allowed')
    require(Path(args.python).is_absolute() and Path(args.python).is_file(), 'explicit existing Python required')
    execution.verify_sources(runtime, spec['native_inventory'], spec['external_python'])
    support = module('curriculum_training_v1.launcher', runtime)
    devices = support.check_free([args.gpu], 1)
    external, env = environment(runtime, args.gpu); root.mkdir(parents=True)
    save(root/'controller_identity.json', dict(process=support.identity(os.getpid()), gpu_devices=devices,
         execution=ex.receipt(args.spec), started_unix=time.time(), automatic_retry=False))
    try:
        protocol_ref = prepare(args, root, spec); protocol = verify_lock(protocol_ref, args.spec)
        package = Path(runtime)/'experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919'
        module('curriculum_scorer_eval_v1.entry', runtime).bind_evaluation(package/'s7_consensus_eval_v14', package/'binary_eval_v1')
        proofs = {}
        for job in protocol['jobs']:
            require(support.check_free([args.gpu], 1) == devices, 'assigned physical GPU identity changed')
            support.execute(root, job['phase'], worker_command(protocol, root, job), env, external)
            proofs[job['phase']] = verify_job(root, protocol_ref, job, runtime)
        finish(root, protocol_ref, proofs, runtime)
        return 0
    except BaseException as error:
        save(root/'controller_failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
             automatic_retry=False, time_unix=time.time()))
        raise


def driver(args):
    root = Path(args.out).resolve(); spec = ex.read(args.spec); runtime = spec['runtime']
    require(not root.exists() and spec.get('schema') == execution.SCHEMA, 'new matched-head terminal driver required')
    require(type(args.gpu) is int and args.gpu in (0, 1, 2, 3), 'only explicitly assigned GPU0–3 allowed')
    require(Path(args.python).is_absolute() and Path(args.python).is_file(), 'explicit existing Python required')
    execution.verify_sources(runtime, spec['native_inventory'], spec['external_python'])
    support = module('curriculum_training_v1.launcher', runtime); devices = support.check_free([args.gpu], 1)
    external, env = environment(runtime, args.gpu)
    # Keep the venv interpreter's invocation path: resolving its symlink could
    # silently execute the base interpreter without this environment's deps.
    command = [str(args.python), '-m', 'model_selection_v2.head_terminal_queue', 'controller']
    for key in ('spec', 'driver_root', 'canonical', 'case_plan'):
        command += ['--'+key.replace('_', '-'), str(Path(getattr(args, key)).resolve())]
    command += ['--python', str(args.python), '--out', str(root/'evaluation'), '--gpu', str(args.gpu)]
    root.mkdir(parents=True); began = time.time()
    save(root/'driver_identity.json', dict(process=support.identity(os.getpid()), started_unix=began,
        gpu_devices=devices, execution=ex.receipt(args.spec), automatic_retry=False))
    try:
        support.execute(root, 'controller', command, env, external)
        returned = actual_return(root, 'controller', command, began, args.gpu)
        complete_ref = ex.receipt(root/'evaluation/controller_complete.json'); complete = checked(complete_ref)
        require(not (root/'evaluation/controller_failure.json').exists()
                and complete.get('schema') == SCHEMA and complete.get('status') == 'all_required_populations_evaluated'
                and complete.get('actual_workers_returned_zero') is True, 'actual controller return is not full evaluation proof')
        protocol = verify_lock(complete['protocol'], args.spec)
        require(set(complete['jobs']) == {j['phase'] for j in protocol['jobs']}
                and set(complete['choices']) == set(CHOICES), 'terminal job/readout coverage incomplete')
        for ref in complete['choices'].values(): checked(ref)
        for job in protocol['jobs']:
            actual = actual_return(root/'evaluation', job['phase'], worker_command(protocol, root/'evaluation', job),
                                   protocol['sealed_unix'], protocol['gpu'])
            require(all(complete['jobs'][job['phase']][k] == v for k, v in actual.items()),
                    'controller did not record its actual successful worker return')
        for proof in complete['jobs'].values():
            for key in ('launch', 'actual_return', 'complete', 'artifact_audit', 'summary'): checked(proof[key])
            for key in ('predictions', 'labeled'):
                require(ex.file_sha(proof[key]['path']) == proof[key]['sha256'], 'terminal predictions changed')
        save(root/'complete.json', dict(schema=SCHEMA, status='terminal_evaluation_complete',
            controller_complete=complete_ref, **returned, execution=ex.receipt(args.spec),
            evaluation_complete=True, task3_overlay_applied=False, finished_unix=time.time()))
        return 0
    except BaseException as error:
        save(root/'driver_failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
             automatic_retry=False, time_unix=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('driver', 'controller'))
    for name in ('spec', 'driver-root', 'canonical', 'case-plan', 'out', 'python'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    return driver(args) if args.action == 'driver' else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
