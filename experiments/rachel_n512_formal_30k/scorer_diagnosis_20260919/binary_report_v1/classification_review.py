"""Classification arithmetic and explicit deltas from already frozen imports.

Consumes compare.py's verified population/policy groups. Does not infer, train,
select an epoch/threshold, or turn an unavailable result into zero. The main
old-head comparator is the same E32/T16 experiment; no causal attribution to a
single layer is made, since the whole scoring/refinement design also changed.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from .assemble import E32_SHA, REAL_PLAN_SHA
from .export import read, require, sha

LIGHT = ('binary_patch', 'binary_stats')
BASELINE = 'threshold_scratch_fixed'
RATES = ('accuracy', 'precision', 'recall', 'f1', 'false_positive_rate')
GROUP = ('selection_kind', 'split', 'population', 'policy')


def classification_counts(metrics):
    keys = ('pairs', 'positives', 'negatives', 'tp', 'tn', 'fp', 'fn')
    metrics = dict(metrics)
    # The binary evaluator records FP and all negatives but omits redundant TN.
    # Derive exactly the complement; never assume negatives were all correct.
    if 'tn' not in metrics:
        require(all(type(metrics.get(k)) is int for k in ('negatives', 'fp'))
                and 0 <= metrics['fp'] <= metrics['negatives'],
                'integer negatives/FP required to derive TN')
        metrics['tn'] = metrics['negatives'] - metrics['fp']
    require(all(type(metrics.get(k)) is int and metrics[k] >= 0 for k in keys),
            'nonnegative integer confusion counts required')
    values = {k: metrics[k] for k in keys}
    n, positive, negative, tp, tn, fp, fn = (values[k] for k in keys)
    require(n > 0 and positive > 0 and negative > 0 and n == positive + negative
            and positive == tp + fn and negative == tn + fp,
            'classification totals or class denominators disagree')
    result = dict(values, accuracy=(tp + tn) / n,
                  precision=tp / (tp + fp) if tp + fp else 0.,
                  recall=tp / positive,
                  f1=2 * tp / (2 * tp + fp + fn),
                  false_positive_rate=fp / negative,
                  specificity=tn / negative,
                  balanced_accuracy=(tp / positive + tn / negative) / 2)
    for key in RATES:
        if key in metrics:
            old = metrics[key]
            require(type(old) in (float, int) and math.isfinite(old)
                    and math.isclose(old, result[key], rel_tol=1e-10, abs_tol=1e-10),
                    'frozen metric differs from confusion counts: ' + key)
    return result


def policy_group(row):
    require(row.get('selection_kind') in ('sim', 'real')
            and row.get('policy') in ('primary', 'fixed03'),
            'keep SIM/REAL and threshold policies separate')
    group = tuple(row[k] for k in GROUP)
    require(row.get('comparison_group') == '|'.join(group), 'comparison group changed')
    if row['split'] in ('dunhuang_cv', 'turufan'):
        require(row.get('real_plan_sha256') == REAL_PLAN_SHA, 'real source roles changed')
    require(type(row['metrics'].get('threshold')) in (int, float)
            and .2 <= row['metrics']['threshold'] <= .8, 'threshold outside registered range')
    if row['policy'] == 'fixed03':
        require(row['metrics']['threshold'] == .3, 'fixed0.30 policy changed')
    return group


def metric_delta(current, baseline):
    a, b = current['classification'], baseline['classification']
    require(tuple(a[k] for k in ('pairs', 'positives', 'negatives'))
            == tuple(b[k] for k in ('pairs', 'positives', 'negatives')),
            'comparison class populations differ')
    if current.get('cohort_sha256') is not None:
        require(current['cohort_sha256'] == baseline.get('cohort_sha256'),
                'exact comparison cohort changed')
    changes = {}
    for key in RATES + ('balanced_accuracy',):
        difference = a[key] - b[key]
        direction = 'unchanged' if abs(difference) <= 1e-12 else 'higher' if difference > 0 else 'lower'
        changes[key] = dict(percentage_points=100 * difference, direction=direction,
                            higher_is_better=key != 'false_positive_rate')
    return dict(baseline_model=BASELINE, baseline_checkpoint_sha256=baseline['checkpoint_sha256'],
                baseline_epoch=baseline['selected_epoch'], baseline_threshold=baseline['threshold'],
                same_population_and_selection_policy=True,
                same_frozen_matcher=current['matcher_sha256'] == baseline['matcher_sha256'] == E32_SHA,
                rates=changes, additional_correct=(a['tp'] + a['tn']) - (b['tp'] + b['tn']),
                additional_false_positives=a['fp'] - b['fp'],
                additional_false_negatives=a['fn'] - b['fn'],
                statistical_significance_claimed=False, single_layer_causal_claimed=False)


def review(evidence, require_both=False):
    require(evidence.get('schema') == 'binary-report-all-models/1'
            and evidence.get('status') == 'evidence_ready_not_final_conclusions'
            and evidence.get('neural_inference_repeated') is False
            and evidence.get('thresholds_refitted') is False,
            'completed frozen evidence assembly required')
    available = evidence.get('available_light_experiments', [])
    require(len(available) == len(set(available)), 'duplicate available experiment')
    if require_both:
        require(set(LIGHT) <= set(available), 'both completed light-head imports are required')
    source = evidence['rows']; output = []; seen = set(); baseline = {}
    for row in source:
        group = policy_group(row); key = (row['model'],) + group
        require(key not in seen, 'duplicate result row'); seen.add(key)
        if row['model'] in LIGHT:
            require(row['model'] in available and row['matcher_sha256'] == E32_SHA,
                    'light head is unimported or has a different Matcher')
        if row['model'] == BASELINE:
            require(row['matcher_sha256'] == E32_SHA, 'reference E32 Matcher changed')
        if row['split'] == 'turufan':
            require(row['metrics'].get('layout20_count') is None
                    and row['metrics'].get('joint_f1') is None, 'Turufan has no layout GT')
        counts = classification_counts(row['metrics'])
        item = {k: deepcopy(row[k]) for k in ('model', *GROUP, 'checkpoint_sha256',
                 'selected_epoch', 'actual_head_epochs', 'matcher_sha256', 'matcher_selected_epoch',
                 'data_contract_sha256', 'real_plan_sha256', 'comparison_group', 'source')}
        if 'cohort_sha256' in row:
            item['cohort_sha256'] = row['cohort_sha256']
        item.update(threshold=row['metrics']['threshold'], classification=counts,
                    layout_separate={k: row['metrics'].get(k) for k in ('layout20_count', 'layout20', 'joint_f1')})
        output.append(item)
        if row['model'] == BASELINE:
            baseline[group] = item
    require(bool(baseline), 'the completed same-E32 complex baseline is required')
    require({r['model'] for r in output if r['model'] in LIGHT} == set(available) & set(LIGHT),
            'available light-head inventory has no metric rows')
    for item in output:
        if item['model'] not in LIGHT:
            continue
        previous = baseline.get(tuple(item[k] for k in GROUP))
        item['old_head_comparison'] = (dict(status='comparable', **metric_delta(item, previous))
                                      if previous else dict(status='no_same_selection_policy_baseline',
                                          reason='Do not compare a REAL-selected new head with a SIM-selected old head as a matched result.'))
    return dict(schema='binary-classification-review/1',
        status='both_light_heads_available' if set(LIGHT) <= set(available) else 'waiting_for_completed_light_imports',
        available_light_heads=[name for name in LIGHT if name in available],
        unavailable_means='not imported, not a remote runtime status', baseline_model=BASELINE,
        rows=output, neural_inference=False, thresholds_refitted=False,
        definitions=dict(accuracy='(TP+TN)/all pairs', recall='TP/positive pairs',
            precision='TP/(TP+FP), zero if there are no accepted pairs', f1='2TP/(2TP+FP+FN)',
            false_positive_rate='FP/negative pairs', balanced_accuracy='mean of positive recall and negative specificity'),
        caveats=[
            'Compare only the same dataset, source-role population, checkpoint-selection policy and threshold policy.',
            'Primary SIM-CAL thresholds may have different numeric values; fixed0.30 is listed separately.',
            'No default assumption that negative pairs are correct; every TP/TN/FP/FN is retained.',
            'When a frozen evaluator omits TN, derive TN exactly as negatives minus FP; do not change the frozen summary.',
            'A higher accuracy may coexist with lower positive recall or more negative false positives.',
            'Classification accuracy is not layout accuracy or joint F1. Turufan layout metrics remain null.',
            'Full real development context overlaps the held-out source-role subset; do not add or average them.',
            'The historical real data were exposed during development; this is not newly blind evaluation.',
            'Same E32 and T16 do not make the entire head/refinement redesign a single-layer intervention.',
            'Confusion arithmetic is rechecked here; prediction hashes, row identities and terminal states are established by the preceding frozen import.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-both', action='store_true')
    args = parser.parse_args()
    require(not args.out.exists() and args.out.resolve() != args.evidence.resolve(), 'preserve prior evidence')
    before = sha(args.evidence); result = review(read(args.evidence), args.require_both)
    require(sha(args.evidence) == before, 'frozen import changed while reporting')
    result['input_sha256'] = {str(args.evidence.resolve()): before}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], rows=len(result['rows']),
                         available_light_heads=result['available_light_heads'], output_sha256=sha(args.out))))


if __name__ == '__main__':
    main()
