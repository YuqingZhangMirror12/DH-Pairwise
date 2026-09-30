"""Bind completed F/I evidence without altering original annotation cases.

Only a local report snapshot is updated. No inference, fitting, training,
source-label changes, or external publication is performed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median

from .prepare import read, save
from .extend_independent_report import APP_ID


def run(root, project, complete=False):
    root, project = Path(root), Path(project)
    target = project / 'src/data.json'
    old = read(target)
    new = read(root / 'report_with_probes.json')
    assert old['id'] == new['id'] == APP_ID
    oq, nq = old['queries'], new['queries']
    for key in ('cases', 'fragments', 'heatmaps', 's7_hard', 'support', 'gradients', 'simulation'):
        assert oq[key] == nq[key], 'unrelated original evidence changed: ' + key
    assert len(nq['independent_cases']['rows']) == 1405
    assert len(nq['independent_heatmaps']['rows']) == 72
    # Keep every existing outer field/presentation setting, replacing only
    # reviewed evidence introduced by this explicitly requested comparison.
    snapshot = dict(old)
    snapshot['queries'] = dict(oq)
    for key in ('metrics', 'independent_cases', 'independent_features',
                'gt_exclusions', 'metrics_revised', 'independent_heatmaps',
                'independent_mechanism'):
        snapshot['queries'][key] = nq[key]
    cases = nq['independent_cases']['rows']
    exclusions = {(x['dataset'], x['pair_id']) for x in nq['gt_exclusions']['rows']}
    # Show all five thresholds, including repeats, rather than a set of values.
    for row in snapshot['queries']['metrics_revised']['rows']:
        group = [c for c in cases if c['dataset'] == row['dataset']
                 and (c['dataset'], c['pair_id']) not in exclusions]
        key = 'threshold_cv' if row['policy'] == '真实五折' else 'threshold_sim'
        values = []
        for fold in sorted({c['fold'] for c in group}):
            ts = {c['models'][row['model_id']][key] for c in group if c['fold'] == fold}
            assert len(ts) == 1
            values.append(next(iter(ts)))
        assert len(values) == 5
        assert abs(median(values) - row['threshold']) < 1e-10
        row['thresholds'] = values if key == 'threshold_cv' else values[:1]
    mechanism = read(root / 'mechanism_independent.json')
    rows = []
    for arm in ('frozen_features', 'independent_features'):
        for dataset in ('敦煌', 'Turufan'):
            positives = [c for c in cases if c['dataset'] == dataset and c['label']
                         and (c['dataset'], c['pair_id']) not in exclusions]
            rejected = [c['models'][arm] for c in positives if not c['models'][arm]['accepted_cv']]
            correct_rejected = [p for p in rejected if p['layout20'] is True]
            rows.append(dict(arm=arm, dataset=dataset, positives=len(positives),
                rejected=len(rejected), rejectedSupportAtMost8=sum(p['edge_count'] <= 8 for p in rejected),
                correctRejected=len(correct_rejected) if dataset == '敦煌' else None,
                correctRejectedSupportAtMost8=sum(p['edge_count'] <= 8 for p in correct_rejected)
                    if dataset == '敦煌' else None))
    source = dict(label='F/I冻结前向证据与修订总体',
        files=[str(root / 'mechanism_independent.json'), str(root / 'report_with_probes.json')],
        caveats=['36例为固定分层诊断，非总体随机样本；attention不是因果归因。',
                 '支持条数是模型产生的对应数，不是GT真实接缝长度。'])
    snapshot['queries']['independent_support'] = dict(rows=rows, source=source)
    probe_rows = []
    for s in mechanism['by_stratum']:
        if s['reason'] != 'correct_layout_rejected':
            continue
        p = s['pooling']
        probe_rows.append(dict(arm=s['arm'], cases=s['cases'],
            medianScore=s['score']['q10_q25_median_q75_q90'][2],
            **{k: p[k]['q10_q25_median_q75_q90'][2] for k in (
                'support_points', 'support_arc_span_px', 'attention80_span_px',
                'effective_point_fraction', 'support_mass', 'exterior_context_mass',
                'support_mass_relative_to_uniform')}))
    snapshot['queries']['independent_probe_summary'] = dict(rows=probe_rows, source=source)
    snapshot['buildStatus'] = 'complete' if complete else 'updating'
    snapshot['generatedAt'] = datetime.now(timezone.utc).isoformat()
    snapshot.setdefault('report', {})['asOf'] = '2026-09-23'
    save(target, snapshot)
    print(json.dumps(dict(report_id=snapshot['id'], cases=len(cases),
        probe_records=72, original_annotation_cases_unchanged=True,
        support=rows, status=snapshot['buildStatus']), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--complete', action='store_true')
    a = p.parse_args()
    run(a.root, a.project, a.complete)
