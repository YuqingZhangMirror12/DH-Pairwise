"""Read-only assembly of the three requested experiments' frozen results.

No inference, checkpoint loading, threshold selection, or training control.
The six-job terminal receipt supplies completeness; export_job verifies the
unchanged per-job evidence. Missing experiments remain unimported, never zero.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .export import export_job, read, require, sha


E32_SHA = '80cac47d5bc5340df35a7a7c36ab4a3580a9eea8c99adf797ff5744cd2068b17'
REAL_PLAN_SHA = '0f8c13bbc4277052ecd6d21e335e97bf58dc8522174e2a11341c3221d50450f6'
EXPERIMENTS = {
    'binary_patch': dict(number=1, variant='patch', sim='sim_test_v14', parameters=34529,
                         title='冻结 E32：Patch/Context＋Q/几何 MLP'),
    'binary_stats': dict(number=2, variant='stats', sim='sim_test_v14', parameters=3201,
                         title='冻结 E32：仅 Q/几何 MLP'),
    'aggressive_binary_patch': dict(number=3, variant='patch', sim='sim_test_aggressive', parameters=34529,
                                   title='新数据：从零 Matcher＋Patch/Context 轻头'),
}
REAL_GROUP_COUNTS = {
    'dunhuang_cv': {'all_development_context': 803, 'gt_corrected_800_development_context': 800,
                    'real_cal': 160, 'real_select': 479, 'real_test': 161},
    'turufan': {'all_development_context': 602, 'real_cal': 120, 'real_select': 360, 'real_test': 122},
}


def tasks(experiment):
    return {(choice, split) for choice in ('sim', 'real')
            for split in (EXPERIMENTS[experiment]['sim'], 'dunhuang_cv', 'turufan')}


def validate_selection(selected, experiment):
    spec = EXPERIMENTS[experiment]
    required_splits = {s for _, s in tasks(experiment)}
    for choice, model in selected.items():
        require(model['variant'] == 'binary_' + spec['variant']
                and model.get('experiment_variant', model['variant']) == experiment
                and model['selection_kind'] == choice
                and model['selection_on_test'] is False
                and model['selection_on_real'] is (choice == 'real')
                and model['real_used_for_stopping'] is False
                and model['real_plan_sha256'] == REAL_PLAN_SHA,
                'selected model scope differs')
        require(set(model['thresholds']) == required_splits, 'selected threshold populations differ')
        require(all(type(t) in (float, int) and .2 <= t <= .8 and abs(t * 100 - round(t * 100)) < 1e-8
                    for t in model['thresholds'].values()), 'threshold is outside fixed CAL grid')
        require(type(model['selected_epoch']) is int and type(model['last_epoch']) is int
                and 16 <= model['last_epoch'] <= 48 and model['last_epoch'] % 2 == 0
                and 0 <= model['selected_epoch'] <= model['last_epoch']
                and model['selected_epoch'] % 2 == 0
                and (choice == 'sim' or model['selected_epoch'] > 0), 'selected epoch differs')
        origin = model['matcher_origin']
        if experiment == 'aggressive_binary_patch':
            require(model['matcher_updated_during_scorer_training'] is False
                    and origin['matcher_trained_from_random_in_this_experiment'] is True
                    and origin['epoch_selected_on_real'] is False
                    and origin['selected_epoch'] > 0, 'new-data Matcher is not the selected random-start run')
        else:
            require(model['matcher_updated_during_training'] is False
                    and origin['sha256'] == E32_SHA and origin['selected_epoch'] == 32
                    and origin['matcher_retrained_for_this_head'] is False, 'frozen E32 identity differs')
    stable = ('matcher_origin', 'data_contract_sha256', 'geometry_calibration_sha256',
              'selection_sha256', 'terminal_receipt_sha256', 'last_epoch', 'stop_reason')
    require(all(selected['sim'][key] == selected['real'][key] for key in stable),
            'SIM/REAL choices must come from the same completed training run')
    execution = selected['sim'].get('execution_continuation')
    require(execution == selected['real'].get('execution_continuation'),
            'SIM/REAL execution continuation differs')
    if execution is not None:
        # The evaluator has already inspected the full preserved checkpoint.
        # This consumer checks and retains its data-only continuation receipt;
        # it must not reopen weights or call a metadata SHA a parameter change.
        require(isinstance(execution, dict)
                and execution.get('schema') == 'binary-microbatch-continuation/1'
                and all(type(execution.get(k)) is int and execution[k] >= 0
                        for k in ('updates', 'exposures'))
                and execution['exposures'] == execution['updates'] * 32
                and all(isinstance(execution.get(k), str) and len(execution[k]) == 64
                        and all(c in '0123456789abcdef' for c in execution[k])
                        for k in ('original_checkpoint_sha256', 'resume_origin_sha256')),
                'invalid execution continuation receipt')


def metric_rows(job, model, experiment):
    """Keep each recorded population/policy at its natural grain. No averaging."""
    split = job['split']
    wanted = {'all': 3000} if split.startswith('sim_test_') else REAL_GROUP_COUNTS[split]
    require(set(job['groups']) == set(wanted), 'evaluation summary groups differ')
    rows = []
    for group, policies in job['groups'].items():
        require(set(policies) == {'primary', 'fixed03'}, 'classification policies differ')
        for policy, metrics in policies.items():
            require(metrics['pairs'] == wanted[group], 'reported population count differs')
            threshold = job['threshold'] if policy == 'primary' else .3
            require(metrics['threshold'] == threshold, 'summary policy threshold differs')
            rows.append(dict(experiment=experiment, experiment_number=EXPERIMENTS[experiment]['number'],
                selection_kind=job['selection_kind'], split=split, population=group, policy=policy,
                is_main=group == job['main_group'] and policy == 'primary',
                threshold_origin=('SIM-CAL at selected epoch' if split.startswith('sim_test_')
                                  or job['selection_kind'] == 'sim' else 'source-isolated REAL-CAL')
                                 if policy == 'primary' else 'fixed 0.30, not calibrated',
                evaluation_context='synthetic held-out' if split.startswith('sim_test_') else 'developmental real data',
                checkpoint_sha256=model['checkpoint_sha256'], selected_epoch=model['selected_epoch'],
                actual_head_epochs=model['last_epoch'], matcher_sha256=model['matcher_origin']['sha256'],
                matcher_selected_epoch=model['matcher_origin']['selected_epoch'],
                data_contract_sha256=model['data_contract_sha256'],
                **({'execution_continuation': deepcopy(model['execution_continuation'])}
                   if model.get('execution_continuation') is not None else {}),
                metrics=metrics, source=job['source']))
    return rows


def assemble_evaluation(directory, experiment):
    require(experiment in EXPERIMENTS, 'unregistered experiment')
    root = Path(directory).resolve()
    require(not any((root / name).exists() for name in ('failure.json', 'controller_failure.json', 'launch_failure.json')),
            'evaluation failure precedes stale completion')
    complete = read(root / 'evaluation_complete.json')
    plan = read(root / 'plan.json')
    gate = read(root / 'selected_models.json')
    variant = EXPERIMENTS[experiment]['variant']
    prefix = 'aggressive-binary' if experiment == 'aggressive_binary_patch' else 'binary'
    require(complete.get('status') == 'complete' and complete.get('variant') == variant
            and complete.get('all_six_populations_verified') is True
            and complete.get('fixed_case_evaluations') == 22
            and complete.get('training_modified') is False
            and read(root / 'driver_status.json') == complete
            and complete['plan_sha256'] == sha(root / 'plan.json'), 'six-job terminal receipt required')
    require(plan.get('schema') == prefix + '-frozen-evaluation-queue/1'
            and plan.get('variant') == variant and len(plan.get('tasks', [])) == 6
            and {tuple(t) for t in plan['tasks']} == tasks(experiment), 'registered evaluation plan differs')
    require(gate.get('schema') == prefix + '-frozen-selection-gate/1'
            and gate.get('status') == 'passed' and gate.get('variant') == variant
            and gate.get('trained_checkpoints_inspected') is True
            and gate.get('real_inference_performed') is False
            and set(gate.get('selected', {})) == {'sim', 'real'}
            and gate['selected'] == plan['selected']
            and gate['source_bindings'] == plan['source_bindings'], 'terminal selection gate differs')
    validate_selection(gate['selected'], experiment)
    records = complete['jobs']
    require(len(records) == 6 and {(r['selection_kind'], r['split']) for r in records} == tasks(experiment),
            'six distinct completed jobs required')
    jobs, rows = [], []
    for record in records:
        choice, split = record['selection_kind'], record['split']
        task = choice + '_' + split
        require(record['task'] == task and record['status'] == 'complete' and record['returncode'] == 0
                and read(root / (task + '_exit.json'))['returncode'] == 0, 'job did not exit successfully')
        verified_path = root / (task + '_verified.json')
        require(record['verified'] == read(verified_path), 'job verification differs from terminal record')
        job = export_job(root / task, verified_path)
        model = gate['selected'][choice]
        require(job['experiment'] == experiment and job['selection_kind'] == choice and job['split'] == split
                and job['checkpoint_sha256'] == model['checkpoint_sha256']
                and job['selected_epoch'] == model['selected_epoch']
                and job['threshold'] == model['thresholds'][split], 'job differs from selected model')
        protocol = read(root / task / 'protocol.json')
        require(all(protocol.get(k) == model.get(k) for k in (
            'matcher_origin', 'data_contract_sha256', 'geometry_calibration_sha256', 'real_plan_sha256',
            'selection_sha256', 'terminal_receipt_sha256', 'execution_continuation')),
            'job source lineage differs')
        jobs.append(job)
        rows.extend(metric_rows(job, model, experiment))
    require(sum(len(job['cases']) for job in jobs) == 22, 'fixed case evaluation count differs')
    return dict(schema='binary-report-experiment/1', experiment=experiment,
        status='six_frozen_jobs_imported', selected=gate['selected'], rows=rows, jobs=jobs,
        source=dict(directory=str(root), terminal_sha256=sha(root / 'evaluation_complete.json'),
                    plan_sha256=complete['plan_sha256'], selected_gate_sha256=sha(root / 'selected_models.json')),
        inference_repeated=False, threshold_refitted=False, conclusions_generated=False)


def comparison_bundle(imports):
    require(imports and set(imports) <= set(EXPERIMENTS),
            'one or more distinct registered experiments required')
    results = [assemble_evaluation(imports[name], name) for name in EXPERIMENTS if name in imports]
    lookup = {result['experiment']: result for result in results}
    if 'binary_patch' in lookup and 'binary_stats' in lookup:
        a, b = (lookup[name]['selected']['sim'] for name in ('binary_patch', 'binary_stats'))
        require(all(a[key] == b[key] for key in ('matcher_origin', 'data_contract_sha256', 'geometry_calibration_sha256')),
                'the two E32 comparisons must share the same frozen Matcher and data')
    return dict(schema='binary-report-comparison/1', status='frozen_evidence_assembly_only',
        available_experiments=list(lookup), unavailable_experiments=[name for name in EXPERIMENTS if name not in lookup],
        unavailable_means='not imported; no claim about remote runtime state',
        all_three_frozen_evaluations_imported=set(lookup) == set(EXPERIMENTS),
        experiment_specs=EXPERIMENTS, experiments=results,
        rows=[row for result in results for row in result['rows']],
        caveats=[
            'SIM-selected and REAL-selected checkpoints are different selection policies; do not blend them.',
            'The two frozen E32 heads share v14 data; the new-data run changes both training data and Matcher.',
            'sim_test_v14 and sim_test_aggressive are different populations, not a matched TEST comparison.',
            'Real CAL/SELECT/TEST and full-context groups overlap; do not sum or average their metrics.',
            'Real source-held-out results are developmental, not historically unseen blind testing.',
            'Turufan has no Layout GT; retain null Layout/Joint metrics.',
            'Imported fixed cases repeat across models/checkpoints; 22 case evaluations are not 22 independent pairs.',
            'This import is not the complete comparative report, causal diagnosis, or overall goal completion.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', action='append', required=True, metavar='EXPERIMENT=DIR')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    imports = {}
    for entry in args.evaluation:
        name, path = entry.split('=', 1)
        require(name not in imports, 'duplicate experiment')
        imports[name] = path
    target = args.out.resolve()
    require(all(not target.is_relative_to(Path(p).resolve()) for p in imports.values()),
            'output cannot modify frozen evaluation inputs')
    require(not target.exists(), 'preserve existing assembled results')
    result = comparison_bundle(imports)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
