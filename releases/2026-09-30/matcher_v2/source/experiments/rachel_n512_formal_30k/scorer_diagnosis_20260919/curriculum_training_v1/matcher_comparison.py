"""Reopen completed native C/M evaluations and assemble paired comparisons.

No inference, threshold search, experiment dispatch or score-based case
selection. The parent queue must supply each controller's successful process
return; a complete-looking JSON file alone is not that proof.
"""
import argparse
from pathlib import Path

from .checkpoint_io import file_sha, write_json
from .execution import load_inputs
from .matcher_entry import check_preparation
from .matcher_eval_controller import jobs, verify_job
from .matcher_evaluation import compare_orders
from .matcher_population import COUNTS, SPLITS, validate_plan
from .matcher_run import read_jsonl
from .matcher_terminal import verified_export
from .model_adapter import require
from .runtime_io import read
from .verify_validation_preparation import bind_baseline


def return_receipt(root, order, returncode):
    """Parent queue calls this only after waiting on the actual child handle."""
    root = Path(root).resolve()
    require(order in ('curriculum', 'mixed') and type(returncode) is int,
            'actual controller process result required')
    return dict(schema='curriculum-native-controller-return/1', order=order,
        root=str(root), returncode=returncode,
        launch_sha256=file_sha(root/'controller_launch.json'),
        complete_sha256=file_sha(root/'evaluation_complete.json') if returncode == 0 else None,
        automatic_retry=False)


def terminal_origins(training_root, spec_path, inputs, order):
    result = {}
    for choice in ('sim_best', 'equal_budget_endpoint'):
        saved, origin = verified_export(training_root, spec_path, inputs['plan'], order, choice)
        model = saved['binding']['model_spec']
        result[choice] = dict(origin,
            initial_matcher_state_sha256=model['initial_matcher_state_sha256'],
            architecture=model['architecture'], geometry=model['geometry'],
            proposal_revision=model['proposal_revision'],
            baseline_sources_sha256=saved['binding']['common_plan']['baseline_sources_sha256'])
    return result


def verify_controller(root, process_return, order, expected, population_plan, preparation):
    root = Path(root).resolve(); process_return = Path(process_return).resolve()
    for name in ('failure.json', 'controller_failure.json', 'live_children_after_controller_error.json'):
        require(not (root/name).exists(), 'controller failure precedes completion: '+name)
    returned = read(process_return)
    require(returned == return_receipt(root, order, 0), 'successful controller return binding differs')
    plan_path = root/'population_plan.json'; plan_sha = file_sha(plan_path)
    require(read(plan_path) == population_plan, 'C/M population plan differs')
    launch = read(root/'controller_launch.json'); complete = read(root/'evaluation_complete.json')
    require(launch['order'] == order and launch['jobs'] == jobs()
            and launch['population_plan_sha256'] == plan_sha
            and launch['preparation_sha256'] == file_sha(preparation)
            and launch['automatic_retry'] is False, 'evaluation controller launch differs')
    require(complete['schema'] == 'curriculum-native-controller-complete/1'
            and complete['status'] == 'complete' and complete['order'] == order
            and complete['job_count'] == 8 and complete['frozen_model_evaluations'] == 2
            and complete['population_plan_sha256'] == plan_sha
            and complete['scorer_used'] is False and complete['automatic_retry'] is False,
            'native evaluation controller is incomplete')
    state = read(root/'driver_status.json')
    require(state == dict(status='complete', completed=8, active={}, pending=[]),
            'evaluation controller still has live/pending work')
    recorded = {r['job']['name']: r for r in complete['jobs']}
    require(len(complete['jobs']) == len(recorded) == 8 and set(recorded) == {j['name'] for j in jobs()},
            'missing/duplicate evaluation jobs')
    result = {}
    for job in jobs():
        actual = verify_job(root, job)
        require(actual == recorded[job['name']], 'controller summary differs from actual job')
        origin = actual['provenance']; chosen = expected[job['selection']]
        require(all(origin.get(key) == value for key, value in chosen.items()),
                'prediction was not made with the verified selected Matcher')
        require(origin['order'] == order and origin['population_plan_sha256'] == plan_sha
                and origin['preparation_sha256'] == file_sha(preparation)
                and origin['evaluation_source_sha256'] == read(preparation)['source_sha256']
                and origin['model_state_before'] == origin['matcher_state_sha256'],
                'prediction implementation/model binding differs')
        path = root/job['name']; population = read(path/'population.json')
        rows = read_jsonl(path/'case_diagnostics.jsonl')
        require(actual['audit']['pairs'] == len(rows) == COUNTS[job['split']],
                'not the entire registered evaluation population')
        result[job['name']] = dict(rows=rows, population=population, provenance=origin,
                                  receipt=actual, root=str(path))
    return dict(order=order, jobs=result, evaluation_root=str(root),
        controller_complete_sha256=file_sha(root/'evaluation_complete.json'),
        controller_return_sha256=file_sha(process_return), population_plan_sha256=plan_sha)


def assemble(curriculum, mixed):
    require(curriculum['order'] == 'curriculum' and mixed['order'] == 'mixed'
            and curriculum['population_plan_sha256'] == mixed['population_plan_sha256'],
            'paired C/M evaluation proof required')
    expected_names = {j['name'] for j in jobs()}
    require(set(curriculum['jobs']) == set(mixed['jobs']) == expected_names,
            'both complete eight-job evaluations required')
    comparisons = {}
    for job in jobs():
        name = job['name']; a = curriculum['jobs'][name]; b = mixed['jobs'][name]
        require(a['population'] == b['population'], 'paired population or roles differ')
        groups = a['population']['groups']; left = {r['pair_id']: r for r in a['rows']}
        right = {r['pair_id']: r for r in b['rows']}
        paired = {}
        # Compare the complete population first: subgroup filtering must not
        # hide different inputs, missing predictions or inconsistent labels.
        compare_orders(a['rows'], b['rows'], a['provenance'], b['provenance'])
        for group, ids in groups.items():
            value = compare_orders([left[i] for i in ids], [right[i] for i in ids],
                                   a['provenance'], b['provenance'])
            value['paired_changed_case_ids'] = {}
            for metric in ('retained_correct_coverage', 'q_sum_winner_layout20', 'q_arc_winner_layout20'):
                known = [i for i in ids if left[i]['label'] and left[i]['gt_known']]
                value['paired_changed_case_ids'][metric] = dict(
                    curriculum_only=[i for i in known if left[i][metric] is True and right[i][metric] is not True],
                    mixed_only=[i for i in known if right[i][metric] is True and left[i][metric] is not True])
            paired[group] = value
        comparisons[name] = dict(selection_kind=job['selection'], split=job['split'], groups=paired,
            main_group='all' if job['split'].startswith('sim_') else 'real_test',
            fixed_case_ids=a['population']['fixed_diagnostic_ids'],
            evidence_roots=dict(curriculum=a['root'], mixed=b['root']),
            prediction_sha256=dict(curriculum=a['receipt']['audit']['prediction_sha256'],
                                   mixed=b['receipt']['audit']['prediction_sha256']))
    origins = {x['order']: {k: x[k] for k in ('evaluation_root', 'controller_complete_sha256',
                'controller_return_sha256', 'population_plan_sha256')} for x in (curriculum, mixed)}
    return dict(schema='curriculum-native-matcher-comparison/1', status='complete',
        origins=origins, comparisons=comparisons, scorer_used=False,
        classification_accuracy=None, joint_f1=None,
        warning='Native Matcher only. Compare SIM-best to SIM-best and endpoint to endpoint; same training budget is not equal selected update. REAL roles are developmental, and Turufan has no Layout GT. No causal claim about Scorer quality is made.')


def run(args):
    spec_path = Path(args.spec).resolve(); preparation = Path(args.preparation).resolve()
    receipt = check_preparation(preparation)
    require(receipt.get('population_comparison_tested') is True, 'comparison needs its own CPU preparation')
    inputs = load_inputs(read(spec_path)); bind_baseline(inputs['baseline'])
    require(inputs['plan'].record['module'] == 'matcher', 'native Matcher comparison only')
    population = read(Path(args.curriculum_evaluation_root)/'population_plan.json')
    validate_plan(population, spec_path, inputs['baseline'])
    verified = []
    for order in ('curriculum', 'mixed'):
        expected = terminal_origins(getattr(args, order+'_training_root'), spec_path, inputs, order)
        verified.append(verify_controller(getattr(args, order+'_evaluation_root'),
            getattr(args, order+'_process_return'), order, expected, population, preparation))
    result = assemble(*verified); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'matcher_comparison.json', result)
    write_json(out/'comparison_complete.json', dict(status='complete',
        comparison_sha256=file_sha(out/'matcher_comparison.json'), preparation_sha256=file_sha(preparation),
        execution_sha256=file_sha(spec_path), inference_repeated=False, gpu_used=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'preparation', 'out', 'curriculum-training-root', 'mixed-training-root',
                 'curriculum-evaluation-root', 'mixed-evaluation-root',
                 'curriculum-process-return', 'mixed-process-return'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__': main()
