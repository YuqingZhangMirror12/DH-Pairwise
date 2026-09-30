"""Recount fixed predictions after explicit user-confirmed GT exclusions.

This is a post-hoc evaluation population revision, not threshold refitting,
inference, checkpoint selection, or rewriting historical evaluation results.
Human success labels are not substituted for automated Layout20 outcomes.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from ..bounded_real_calibration_v2.common import read, save, sha, metrics

MODELS = ('S7_M12_matched_C16', 'v3_B22', 'S7_H',
          'frozen_features', 'independent_features')


def revise(snapshot, exclusions):
    if exclusions.get('schema') != 'user-confirmed-gt-exclusions-v1':
        raise ValueError('explicit exclusion manifest required')
    queries = snapshot['queries']
    query_name = 'independent_cases' if 'independent_cases' in queries else 'cases'
    cases = queries[query_name]['rows']
    index = {(c['dataset'], c['pair_id']): c for c in cases}
    if len(index) != len(cases):
        raise ValueError('duplicate source pairs')
    excluded = set()
    for e in exclusions['records']:
        key = e['dataset'], e['pair_id']
        if key in excluded or key not in index:
            raise ValueError('duplicate or missing excluded pair')
        c = index[key]
        if (not c['label'] or c['dataset'] != '敦煌'
                or c['case_name'].split(' · ')[0] != e['case_folder']):
            raise ValueError('exclusion must identify the exact positive GT case')
        excluded.add(key)
    included = [c for c in cases if (c['dataset'], c['pair_id']) not in excluded]
    rows = []
    for dataset in ('敦煌', 'Turufan'):
        group = [c for c in included if c['dataset'] == dataset]
        if not group:
            raise ValueError('both evaluation datasets are required')
        for model_id in MODELS:
            available = [model_id in c['models'] for c in group]
            if not any(available):
                continue
            if not all(available):
                raise ValueError('partial model population')
            pred = [c['models'][model_id] for c in group]
            labels = [bool(c['label']) for c in group]
            good = [p['layout20'] for p in pred] if dataset == '敦煌' else None
            if good is not None and any(g is None for g, y in zip(good, labels) if y):
                raise ValueError('missing positive automated Layout GT')
            for policy, key in [('SIM冻结', 'threshold_sim'), ('原五折阈值冻结', 'threshold_cv')]:
                decisions = [bool(p['decision_valid'] and p['score'] >= p[key]) for p in pred]
                if key == 'threshold_cv' and decisions != [p['accepted_cv'] for p in pred]:
                    raise ValueError('stored OOF decision mismatch')
                thresholds = sorted(set(p[key] for p in pred))
                fold_thresholds = []
                for f in sorted(set(c['fold'] for c in group)):
                    values = {c['models'][model_id][key] for c in group if c['fold'] == f}
                    if len(values) != 1:
                        raise ValueError('multiple thresholds within one fold')
                    fold_thresholds.append(next(iter(values)))
                if key == 'threshold_sim' and len(thresholds) != 1:
                    raise ValueError('SIM threshold must remain fixed')
                rows.append(dict(dataset=dataset, model_id=model_id, policy=policy,
                    threshold_values=thresholds,
                    # Per-fold median, not a median over repeated pair thresholds.
                    threshold_median=median(fold_thresholds),
                    **metrics(labels, decisions, [p['score'] for p in pred], good)))
    return dict(protocol=dict(population_revision='post-hoc user-confirmed incorrect GT exclusions',
        thresholds_refit=False, model_selection_on_real=False, original_predictions_modified=False,
        human_layout_labels_used_as_automated_gt=False, original_query=query_name,
        excluded_positive_pairs=len(excluded), original_count=len(cases), included_count=len(included),
        caveat='Thresholds are frozen from the original source-grouped folds, not recalibrated after exclusion. '
               'Turufan still has no objective layout GT; manual judgments remain a separate analysis.'),
        excluded=exclusions['records'], rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--exclusions', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists() or out.resolve() in (Path(a.snapshot).resolve(), Path(a.exclusions).resolve()):
        raise ValueError('use a new output file; preserve historical results')
    result = revise(read(a.snapshot), read(a.exclusions))
    result['sources'] = dict(snapshot_sha256=sha(a.snapshot), exclusions_sha256=sha(a.exclusions),
        executed_at=datetime.now(timezone.utc).isoformat())
    save(out, result)
    print(str(out), len(result['rows']), 'recounted metric rows; no fitting')


if __name__ == '__main__':
    main()
