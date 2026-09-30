"""Paired F/I diagnostic evidence, never a replacement for full test metrics.

Reads only completed frozen probes of the preselected 36 cases. Matcher and
Scorer representations are compared at identical point indices, and changed
winning candidates are explicitly flagged before comparing pooling evidence.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .analyze_probes import stats
from .prepare import read, save, sha

ARMS = ('frozen_features', 'independent_features')


def feature_difference(reference, adapted):
    a, b = np.asarray(reference, dtype=float), np.asarray(adapted, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('feature comparisons require finite aligned token matrices')
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    valid = (an > 1e-12) & (bn > 1e-12)
    cosine = np.sum(a[valid] * b[valid], axis=1) / (an[valid] * bn[valid])
    return dict(tokens=len(a), nonzero_tokens=int(valid.sum()),
        cosine=stats(cosine), norm_ratio=stats(bn[valid] / an[valid]),
        relative_l2=float(np.linalg.norm(b-a) / max(np.linalg.norm(a), 1e-12)),
        max_absolute_difference=float(np.max(np.abs(b-a))) if a.size else None)


def winner(record):
    p = record['prediction']
    return p['candidates'][p['winner_index']] if p['has_candidate'] else None


def pool_summary(pools):
    result = {key: stats([p[key] for p in pools]) for key in (
        'point_count', 'support_points', 'support_mass', 'internal_gap_mass',
        'exterior_context_mass', 'support_arc_span_px', 'attention80_span_px')}
    result['effective_point_fraction'] = stats([
        p['effective_points']/p['point_count'] for p in pools if p['point_count']])
    result['support_mass_relative_to_uniform'] = stats([
        p['support_mass']/(p['support_points']/p['point_count'])
        for p in pools if p['point_count'] and p['support_points']])
    return result


def compare_case(f, i, arrays_f, arrays_i):
    if (f['split'], f['pair_id']) != (i['split'], i['pair_id']):
        raise ValueError('cannot compare different diagnostic cases')
    for key in ('points_a', 'points_b'):
        if not np.array_equal(arrays_f[key], arrays_i[key]):
            raise ValueError('different contour sampling; tokenwise comparison invalid')
    wf, wi = winner(f), winner(i)
    same = bool(wf and wi and np.allclose(wf['proposal_translation_rc'],
        wi['proposal_translation_rc'], atol=.01, rtol=0))
    features = {}
    for side in 'ab':
        for kind in ('raw', 'context'):
            key = f'{kind}_{side}'
            features[f'matcher_{key}'] = feature_difference(arrays_f[key], arrays_i[key])
            # F has no separate Scorer encoder. Its Scorer input is the frozen feature.
            features[f'scorer_{key}'] = feature_difference(arrays_f[key], arrays_i['scorer_'+key])
    return dict(pair_id=f['pair_id'], split=f['split'], reason=f['reason'],
        same_winning_proposal=same, feature_differences=features,
        q1_max_absolute_difference=float(np.max(np.abs(arrays_f['q1']-arrays_i['q1']))),
        f_score=f['prediction']['score'], i_score=i['prediction']['score'],
        f_layout20=f['prediction']['layout20'] if f['prediction']['gt_known'] else None,
        i_layout20=i['prediction']['layout20'] if i['prediction']['gt_known'] else None,
        f_support_edges=wf['edge_count'] if wf else 0,
        i_support_edges=wi['edge_count'] if wi else 0)


def run(a):
    root = Path(a.root)
    expected = read(a.cases)['cases']
    order = [(r['split'], r['pair_id']) for r in expected]
    if len(set(order)) != len(order):
        raise ValueError('duplicate preselected cases')
    records, paths, by_key = {}, {}, {}
    for arm in ARMS:
        path = root / ('probes_'+arm)
        status = read(path/'status.json')
        protocol = read(root/arm/'dunhuang_cv/protocol.json')
        if (status.get('status') != 'complete' or status.get('count') != len(order)
                or status.get('checkpoint_sha256') != protocol['checkpoint_sha256']
                or status.get('model_provenance', {}).get('arm') != arm):
            raise ValueError('probe incomplete or bound to another model')
        records[arm] = read(path/'records.json')
        by_key[arm] = {(r['split'], r['pair_id']): r for r in records[arm]}
        if set(by_key[arm]) != set(order) or len(records[arm]) != len(order):
            raise ValueError('probe population differs from preselected cases')
        paths[arm] = path
    paired = []
    for key in order:
        f, i = (by_key[arm][key] for arm in ARMS)
        for arm, r in zip(ARMS, (f, i)):
            p = paths[arm]/r['raw_array_file']
            if sha(p) != r['raw_array_sha256']:
                raise ValueError('probe array identity changed')
            if r['parity']['score_delta'] > 2e-5 or (r['parity']['translation_delta_px'] or 0) > .01:
                raise ValueError('probe differs from frozen real inference')
        with np.load(paths[ARMS[0]]/f['raw_array_file']) as af, np.load(paths[ARMS[1]]/i['raw_array_file']) as ai:
            paired.append(compare_case(f, i, af, ai))
    summaries = []
    for arm in ARMS:
        groups = defaultdict(list)
        for r in records[arm]:
            groups[r['reason']].append(r)
        for reason, rows in groups.items():
            summaries.append(dict(arm=arm, reason=reason, cases=len(rows),
                pooling=pool_summary([p for r in rows for p in r['pooling']]),
                support_edges=stats([winner(r)['edge_count'] for r in rows if winner(r)]),
                score=stats([r['prediction']['score'] for r in rows]),
                diagnostic_score_deltas={v: stats([
                    r['diagnostic_interventions'][v]-r['prediction']['score']
                    for r in rows if v in r['diagnostic_interventions']]) for v in (
                        'residual_half', 'residual_zero', 'overlap_zero', 'dustbin_zero')}))
    output = dict(protocol=dict(sample='fixed 36 outcome-stratified diagnostic cases',
        representative_population=False, causal_attribution=False, training=False,
        threshold_fitting=False, matched_token_indices=True,
        pooling_caveat='Pooling can change with the winning candidate as well as network weights; see same_winning_proposal.',
        feature_caveat='Cosine/relative L2 measure adaptation, not feature usefulness.'),
        paired=paired, by_stratum=summaries,
        sources=dict(cases_sha256=sha(a.cases), records_sha256={arm:sha(paths[arm]/'records.json') for arm in ARMS}))
    dest = Path(a.out)
    if dest.exists():
        raise ValueError('use a new output file')
    save(dest, output)
    print(json.dumps(dict(cases=len(paired), strata=len(summaries), output=str(dest))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'cases', 'out'):
        parser.add_argument('--'+key, required=True)
    run(parser.parse_args())
