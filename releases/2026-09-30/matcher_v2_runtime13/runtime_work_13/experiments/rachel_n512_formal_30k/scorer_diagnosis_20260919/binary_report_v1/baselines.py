"""Re-express existing complex-head predictions on the registered REAL roles.

This is offline reporting, not new inference or threshold/epoch selection.
The previous completed analysis binds every consumed source hash. Its five-fold
metrics are NOT substituted for the new REAL-TEST population.
"""
import argparse
import hashlib
import json
from pathlib import Path

from ..binary_eval_v1.contracts import partition_real
from ..consensus_real_analysis_v1.analyze import compact, independently_count, close
from .assemble import REAL_PLAN_SHA, REAL_GROUP_COUNTS
from .export import read, require, sha


METRIC_ALIASES = {
    'n': 'pairs', 'positive': 'positives', 'negative': 'negatives',
    'layout_correct_total': 'layout20_count', 'layout_accuracy': 'layout20',
    'layout_correct_accepted': 'joint_tp',
    'layout_correct_rejected': 'winner_correct_but_rejected',
}
BASELINES = {
    'mergefix_m12': ('mergefix', 'm12', 'directional_full_q'),
    'mergefix_scratch': ('mergefix', 'scratch', 'directional_full_q'),
    'threshold_m12': ('threshold', 'm12', 'exact_union_q'),
    'threshold_scratch_fixed': ('threshold', 'scratch_fixed', 'exact_union_q'),
}
SPLITS = ('sim_test_v14', 'dunhuang_cv', 'turufan')


def digest_object(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def prior_bindings(path):
    summary = read(path)['summary']
    require(summary['schema'] == 'frozen-real-comparison/1'
            and summary['status'] == 'complete'
            and summary['all_source_hashes_unchanged'] is True,
            'completed earlier analysis required')
    result = {}
    maps = [summary['inputs']] + [item['input_sha256'] for item in summary.get('extensions', [])]
    for mapping in maps:
        for source, value in mapping.items():
            key = str(Path(source).resolve())
            require(key not in result or result[key] == value, 'earlier source bindings conflict')
            result[key] = value
    return result


def bound_file(path, bindings, consumed):
    path = Path(path).resolve()
    require(str(path) in bindings, 'file is not bound by completed earlier analysis: ' + path.name)
    value = sha(path)
    require(value == bindings[str(path)], 'previously audited file changed: ' + path.name)
    consumed[str(path)] = value
    return path


def compact_rows(path):
    with Path(path).open() as stream:
        # Keep only report fields, not the entire candidate/feature trace.
        return [compact(json.loads(line)) for line in stream if line.strip()]


def summarize(rows, threshold, layout_available):
    decisions = [r['decision_valid'] and r['score'] >= threshold for r in rows]
    counted = independently_count(rows, decisions, layout_available)
    return dict(threshold=threshold, **{METRIC_ALIASES.get(k, k): v for k, v in counted.items()})


def groups_for(rows, plan, split):
    if split == 'sim_test_v14':
        require(len(rows) == len({r['pair_id'] for r in rows}) == 3000,
                'v14 TEST must contain 3000 distinct record IDs')
        return {'all': rows}
    groups = partition_real(rows, plan, split)
    expected = REAL_GROUP_COUNTS[split]
    require(set(groups) == set(expected), 'real roles differ')
    for role, subset in groups.items():
        require(len(subset) == expected[role], 'real role count differs: ' + role)
        if role.startswith('real_'):
            spec = plan['datasets'][split]['roles'][role]
            require(sum(r['label'] for r in subset) == spec['positive']
                    and all(r['fold'] in spec['folds'] for r in subset),
                    'real role label or original fold differs')
    return groups


def baseline_job(root, record, name, plan, bindings, consumed):
    variant, arm, evidence_mode = BASELINES[name]
    require(record['arm'] == arm and record['status'] == 'complete'
            and record['returncode'] == 0 and record['split'] in SPLITS,
            'baseline job is not a successful registered evaluation')
    split = record['split']
    require(record['task'] == arm + '_' + split, 'baseline job identity differs')
    folder = Path(root) / record['task']
    require(not (folder / 'failure.json').exists(), 'job failure precedes old completion')
    values = {n: read(bound_file(folder / n, bindings, consumed)) for n in (
        'summary.json', 'protocol.json', 'status.json', 'prediction_complete.json')}
    summary, protocol, status, frozen = (values[n] for n in (
        'summary.json', 'protocol.json', 'status.json', 'prediction_complete.json'))
    require(summary['status'] == protocol['status'] == status['status'] == 'complete'
            and frozen['status'] == 'all_predictions_frozen'
            and frozen['model_state_unchanged'] is True
            and record['verified']['model_state_unchanged'] is True, 'not frozen complete outputs')
    identity = dict(variant=variant, arm=arm, split=split, evidence_mode=evidence_mode)
    require(all(all(value.get(k) == v for k, v in identity.items()) for value in (summary, protocol)),
            'wrong complex-head architecture or branch')
    for key in ('checkpoint_sha256', 'selected_epoch', 'last_epoch', 'threshold',
                'data_contract_sha256', 'geometry_calibration_sha256', 'matcher_origin'):
        require(summary[key] == protocol[key], 'baseline provenance differs: ' + key)
    require(protocol['model_selection_on_test_or_real'] is False
            and protocol['threshold_refitted'] is False and protocol['gt_used_for_prediction'] is False,
            'baseline was not frozen SIM-selected inference')
    require(sha(folder / 'summary.json') == record['verified']['summary_sha256'],
            'terminal summary hash differs')
    prediction = bound_file(folder / 'pair_predictions.jsonl', bindings, consumed)
    require(sha(prediction) == record['verified']['predictions_sha256'] == frozen['sha256'],
            'terminal prediction hash differs')
    labelled = bound_file(folder / 'case_diagnostics.jsonl', bindings, consumed)
    rows = compact_rows(labelled)
    require(len(rows) == protocol['total_pairs'] == frozen['pairs'] == record['verified']['pairs'],
            'baseline row counts differ')
    if split != 'sim_test_v14':
        require(protocol['source']['manifest_sha256'] == plan['datasets'][split]['manifest_sha256'],
                'baseline real dataset differs from registered roles')
    groups = groups_for(rows, plan, split)
    output = []
    for population, subset in groups.items():
        # The fingerprint excludes predictions. It identifies precisely the
        # labelled population, so distinct SIM tests cannot look interchangeable.
        cohort_sha = digest_object(sorted((r['pair_id'], r['label'], r['fold']) for r in subset))
        for policy, threshold in (('primary', protocol['threshold']), ('fixed03', .3)):
            result = summarize(subset, threshold, split != 'turufan')
            old_group = ('all' if population in ('all', 'all_development_context') else
                         'gt_corrected_800' if population == 'gt_corrected_800_development_context' else None)
            if old_group:
                old = summary['groups'][old_group][policy]
                for key, value in result.items():
                    if key in old:
                        close(value, old[key], 'original frozen metric ' + key)
            output.append(dict(model=name, architecture='complex_local_support_conflict_head',
                selection_kind='sim', split=split, population=population, policy=policy,
                is_main=population == ('all' if split == 'sim_test_v14' else 'real_test') and policy == 'primary',
                threshold_origin=('SIM-CAL at selected epoch' if policy == 'primary' else 'fixed 0.30, not calibrated'),
                checkpoint_sha256=protocol['checkpoint_sha256'], selected_epoch=protocol['selected_epoch'],
                actual_head_epochs=protocol['last_epoch'], matcher_sha256=protocol['matcher_origin']['sha256'],
                matcher_selected_epoch=protocol['matcher_origin']['selected_epoch'],
                data_contract_sha256=protocol['data_contract_sha256'], cohort_sha256=cohort_sha,
                real_plan_sha256=REAL_PLAN_SHA if split != 'sim_test_v14' else None,
                evaluation_context='synthetic held-out' if split == 'sim_test_v14' else 'developmental real data',
                metrics=result, source=dict(job=str(folder.resolve()),
                    summary_sha256=sha(folder / 'summary.json'), predictions_sha256=sha(prediction),
                    diagnostics_sha256=sha(labelled)),
                scope_note='same frozen checkpoint and threshold; only prespecified population filtering'))
    return output


def build_baselines(evaluations, analysis, real_plan):
    require(bool(evaluations) and set(evaluations) <= set(BASELINES), 'registered baseline imports required')
    require(sha(real_plan) == REAL_PLAN_SHA, 'registered REAL role plan changed')
    plan = read(real_plan)
    require(plan['source_disjoint'] is True and plan['development_evaluation'] is True,
            'source-isolated developmental roles required')
    bindings = prior_bindings(analysis)
    consumed = {str(Path(analysis).resolve()): sha(analysis), str(Path(real_plan).resolve()): sha(real_plan)}
    output, selected = [], {}
    for name, directory in evaluations.items():
        root = Path(directory).resolve()
        require(not any((root / p).exists() for p in ('failure.json', 'controller_failure.json')),
                'evaluation failure precedes completion')
        terminal = read(bound_file(root / 'evaluation_complete.json', bindings, consumed))
        require(terminal['status'] == 'complete' and terminal['training_modified'] is False,
                'baseline evaluation is not terminal')
        arm = BASELINES[name][1]
        records = [r for r in terminal['jobs'] if r['arm'] == arm]
        require(len(records) == 3 and {r['split'] for r in records} == set(SPLITS),
                'all three frozen baseline populations are required')
        before = len(output)
        for record in records:
            output.extend(baseline_job(root, record, name, plan, bindings, consumed))
        own = output[before:]
        for key in ('checkpoint_sha256', 'selected_epoch', 'matcher_sha256', 'data_contract_sha256'):
            require(len({r[key] for r in own}) == 1, 'baseline jobs mix different selected models')
        selected[name] = {key: own[0][key] for key in ('checkpoint_sha256', 'selected_epoch',
                              'matcher_sha256', 'matcher_selected_epoch', 'data_contract_sha256')}
    require(all(sha(path) == value for path, value in consumed.items()), 'input changed while reporting')
    return dict(schema='binary-report-complex-baselines/1', status='baseline_evidence_ready',
        selected=selected, rows=output, input_sha256=consumed,
        real_plan_sha256=REAL_PLAN_SHA, neural_inference_repeated=False,
        epoch_selected=False, threshold_fitted=False, five_fold_metrics_substituted=False,
        old_analysis_modified=False, new_light_head_results_available=False,
        caveats=[
            'All baseline checkpoints and primary thresholds remain SIM-selected.',
            'REAL-TEST contains 161 Dunhuang and 122 Turufan pairs under the registered source roles.',
            'The full real cohorts were historically exposed during development; this is not a new blind test.',
            'Old five-fold calibrated metrics are a different protocol and remain in the unchanged old report.',
            'New light-head REAL-selected checkpoints must be shown separately from these SIM-selected baselines.',
            'Changing the data, Matcher or builder is not a Scorer-only intervention.',
            'No result for an unavailable light-head experiment is inferred or filled with zero.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', action='append', required=True, metavar='MODEL=DIR')
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--real-plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    imports = {}
    for entry in args.evaluation:
        name, path = entry.split('=', 1)
        require(name not in imports, 'duplicate baseline import')
        imports[name] = path
    target = args.out.resolve()
    require(not target.exists() and all(not target.is_relative_to(Path(p).resolve()) for p in imports.values()),
            'output must be new and outside frozen evidence')
    require(target not in (args.analysis.resolve(), args.real_plan.resolve()), 'preserve input files')
    result = build_baselines(imports, args.analysis, args.real_plan)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(target), sha256=sha(target), models=list(result['selected']),
                         rows=len(result['rows']), new_forward_passes=0)))


if __name__ == '__main__':
    main()
