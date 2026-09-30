"""Combine already imported report evidence, never choose a winning model.

Each comparison is keyed by dataset, exact population, checkpoint-selection
policy and classification-threshold policy. Unlike a leaderboard, this does
not rank scores across different populations or claim a controlled intervention.
"""
import argparse
import copy
import json
from pathlib import Path

from .assemble import EXPERIMENTS, REAL_PLAN_SHA
from .export import read, require, sha


def combined_evidence(baselines, binary=None):
    require(baselines.get('schema') == 'binary-report-complex-baselines/1'
            and baselines.get('status') == 'baseline_evidence_ready'
            and baselines.get('real_plan_sha256') == REAL_PLAN_SHA
            and baselines.get('threshold_fitted') is False
            and baselines.get('five_fold_metrics_substituted') is False,
            'compatible frozen baseline evidence required')
    rows = copy.deepcopy(baselines['rows'])
    available, cases = [], []
    if binary is not None:
        require(binary.get('schema') == 'binary-report-comparison/1'
                and binary.get('status') == 'frozen_evidence_assembly_only',
                'import completed light-head evidence before combining')
        available = binary['available_experiments']
        require(set(available) <= set(EXPERIMENTS) and len(available) == len(set(available)),
                'unregistered or duplicated light-head results')
        for result in binary['experiments']:
            require(result['experiment'] in available and result['status'] == 'six_frozen_jobs_imported',
                    'incomplete light-head experiment')
            require(all(m['real_plan_sha256'] == REAL_PLAN_SHA for m in result['selected'].values()),
                    'light-head source roles differ')
            cases.extend(dict(experiment=result['experiment'], selection_kind=j['selection_kind'],
                              split=j['split'], cases=j['cases']) for j in result['jobs'])
        require({r['experiment'] for r in binary['experiments']} == set(available),
                'light-head import inventory differs')
        for source in binary['rows']:
            row = copy.deepcopy(source)
            row['model'] = row.pop('experiment')
            require(row['model'] in available, 'light-head row has no completed import')
            row['architecture'] = 'binary_' + EXPERIMENTS[row['model']]['variant'] + '_mlp'
            row['real_plan_sha256'] = None if row['split'].startswith('sim_test_') else REAL_PLAN_SHA
            rows.append(row)
    keys, populations = set(), {}
    for row in rows:
        require(row['selection_kind'] in ('sim', 'real') and row['policy'] in ('primary', 'fixed03'),
                'five-fold/other policies cannot masquerade as a matched comparison')
        key = (row['model'], row['selection_kind'], row['split'], row['population'], row['policy'])
        require(key not in keys, 'duplicate result grain'); keys.add(key)
        group = (row['selection_kind'], row['split'], row['population'], row['policy'])
        row['comparison_group'] = '|'.join(group)
        counts = tuple(row['metrics'][k] for k in ('pairs', 'positives', 'negatives'))
        require(group not in populations or populations[group] == counts, 'population class counts differ')
        populations[group] = counts
        if row['split'] == 'turufan':
            require(row['metrics']['layout20_count'] is None and row['metrics']['joint_f1'] is None,
                    'Turufan Layout GT is unavailable, not zero')
        if row['model'] == 'aggressive_binary_patch':
            require(row['split'] != 'sim_test_v14', 'new-data TEST must remain distinct from v14 TEST')
    return dict(schema='binary-report-all-models/1', status='evidence_ready_not_final_conclusions',
        rows=rows, binary_case_groups=cases, available_light_experiments=available,
        unavailable_light_experiments=[x for x in EXPERIMENTS if x not in available],
        unavailable_means='not imported; no inference about current remote task state',
        group_keys=['selection_kind', 'split', 'population', 'policy'],
        automatic_ranking=False, neural_inference_repeated=False, thresholds_refitted=False,
        caveats=list(baselines['caveats']) + [
            'Compare only rows sharing comparison_group; overlapping populations are never added.',
            'SIM and REAL checkpoint selection policies remain separate even on the same held-out pairs.',
            'Differences across Matcher origins, builders or training data are not Scorer-only causal effects.',
            'A new-data simulation TEST is not the original v14 simulation TEST.',
            'MLP feature pooling is not Attention; binary_case_groups retain their recorded semantics.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baselines', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    inputs = [p.resolve() for p in (args.baselines, args.binary) if p is not None]
    require(not args.out.exists() and args.out.resolve() not in inputs, 'preserve existing evidence')
    hashes = {str(p): sha(p) for p in inputs}
    result = combined_evidence(read(args.baselines), read(args.binary) if args.binary else None)
    require(all(sha(p) == value for p, value in hashes.items()), 'input changed during import')
    result['input_sha256'] = hashes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(args.out.resolve()), sha256=sha(args.out), rows=len(result['rows']))))


if __name__ == '__main__':
    main()
