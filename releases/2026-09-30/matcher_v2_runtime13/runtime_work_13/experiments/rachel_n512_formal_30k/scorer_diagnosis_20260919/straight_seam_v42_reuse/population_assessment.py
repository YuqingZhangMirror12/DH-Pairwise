"""Assess all independently audited v4.2 SELECT rows without filtering them.

This is a geometry/selection-bias report, not permission to train. In particular,
passing median targets cannot waive a per-pair tail constraint from the spec.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .frozen_diagnostic import sha, save, validate_population


BOUNDS = {
    'M': {'slide_m40': (0., 2.5)},
    'J': {'extent_px': (167.25, 278.75), 'bend_range': (7., 11.),
          'rough_std': (1.17, 1.67), 'gap_frac_over3': (.17, .37),
          'slide_m20': (0., 3.5), 'slide_m40': (0., 5.)},
    'R': {'extent_px': (351., 585.), 'bend_range': (4.1, 8.1),
          'rough_std': (.46, .96), 'gap_frac_over3': (.33, .53),
          'slide_m20': (0., 1.5)},
}
SEAM_FIELDS = ('extent_px', 'bend_range', 'rough_std', 'gap_frac_over3',
               'overlap_per_extent', 'slide_m10', 'slide_m20', 'slide_m40')
SHAPE_FIELDS = ('rough_std_px', 'rough_corr_px', 'bend_amp_px')


def distribution(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError('Missing/nonfinite population measurements')
    return dict(n=len(array), min=float(array.min()) if len(array) else None,
        max=float(array.max()) if len(array) else None,
        mean=float(array.mean()) if len(array) else None,
        p10_p25_p50_p75_p90=np.quantile(array, [.1, .25, .5, .75, .9]).tolist() if len(array) else None)


def trace_parameters(trace):
    if trace is None:
        return None
    wear = trace['wear']
    if len(wear) != 2:
        raise ValueError('Two-sided wear trace required')
    result = dict(wear_sum_px=sum(wear), wear_max_px=max(wear),
                  overlap_active=float(trace['overlap_active']))
    for name in ('gap_coverage_draw', 'overlap_coverage_draw'):
        if trace[name] is not None:
            result[name] = trace[name]
    return result


def parameter_comparison(attempted, survived):
    keys = sorted({k for row in attempted + survived for k in row})
    result = {}
    for key in keys:
        before = distribution([r[key] for r in attempted if key in r])
        after = distribution([r[key] for r in survived if key in r])
        result[key] = dict(attempted=before, survived=after,
            survivor_minus_attempt_mean=(after['mean'] - before['mean'])
            if before['n'] and after['n'] else None)
    return result


def selection_bias(entries):
    """Positive pairs only: one surviving shape/damage trace per accepted pair.

    Rejected pieces before misfit have no damage draws. Their absence is reported
    instead of inventing zero wear or treating all attempts as comparable pairs.
    """
    shape_draws, damage_draws, survived_shape, survived_damage = [], [], [], []
    missing_damage = 0
    rejection_counts = Counter()
    for row in entries:
        rejection_counts.update(row['rejections'])
        attempts = row['piece_attempts']
        if not attempts or any(a['base'] != row['meta']['base'] for a in attempts):
            raise ValueError('Fixed pre-sampled base or attempt trace changed')
        if attempts[-1]['piece_rejection'] is not None:
            raise ValueError('Last piece was not accepted')
        for field in SHAPE_FIELDS:
            if attempts[-1]['shape'][field] != row['meta'][field]:
                raise ValueError('Accepted shape is not the final logged attempt')
        if attempts[-1]['misfit_trace'] != row['accepted_damage_trace']:
            raise ValueError('Accepted damage is not the final logged trace')
        for attempt in attempts:
            shape_draws.append(attempt['shape'])
            damage = trace_parameters(attempt['misfit_trace'])
            if damage is None:
                missing_damage += 1
            else:
                damage_draws.append(damage)
        survived_shape.append({k: row['meta'][k] for k in SHAPE_FIELDS})
        damage = trace_parameters(row['accepted_damage_trace'])
        if damage is None:
            raise ValueError('Accepted positive has no damage trace')
        survived_damage.append(damage)
    return dict(denominator='positive pairs only', accepted_pairs=len(entries),
        piece_attempts=len(shape_draws), attempts_before_damage_sampling=missing_damage,
        tries=distribution([r['tries'] for r in entries]), rejection_counts=dict(rejection_counts),
        surviving_base_counts=dict(Counter(r['meta']['base'] for r in entries)),
        shape=parameter_comparison(shape_draws, survived_shape),
        damage=parameter_comparison(damage_draws, survived_damage),
        warning='Reference redraws shape/damage on rejection. Mean shifts are descriptive, '
                'not a test of zero selection bias; no rows were removed by this report.')


def inspect(manifest, audit, plan, manifest_sha):
    entries = validate_population(manifest, audit, manifest_sha)
    if plan['split'] != 'select' or plan.get('preflight') is not False:
        raise ValueError('Full SELECT plan required, never TEST or preflight')
    checks, groups = [], {}
    for kind in 'MJR':
        rows = [r for r in audit['records'] if r['kind'] == kind and r['positive']]
        positive_entries = [r for r in entries if r['recipe'] == 'straight_' + kind and r['label']]
        metrics = {key: distribution([r['metrics']['seam'][key] for r in rows]) for key in SEAM_FIELDS}
        metrics['rectangularity_min_of_two'] = distribution([r['metrics']['smaller_rectangularity'] for r in rows])
        metrics['healthy_correspondences'] = distribution([r['target_audit']['correspondence_count'] for r in rows])
        for key, (low, high) in BOUNDS[kind].items():
            value = metrics[key]['p10_p25_p50_p75_p90'][2]
            checks.append(dict(kind=kind, metric=key, statistic='median', actual=value,
                               lower=low, upper=high, passed=low <= value <= high))
        m40 = metrics['slide_m40']['p10_p25_p50_p75_p90'][2]
        checks.append(dict(kind=kind, metric='slide_m40_below_curve', statistic='median',
                           actual=m40, upper_exclusive=6.1, passed=m40 < 6.1))
        bend_outliers = [r for r in rows if r['metrics']['seam']['bend_range'] > 15]
        checks.append(dict(kind=kind, metric='all_bend_le15', statistic='all_pairs',
                           failed_count=len(bend_outliers), denominator=len(rows),
                           passed=not bend_outliers, pair_ids=[r['pair_id'] for r in bend_outliers]))
        rect_outliers = []
        if kind in 'JR':
            rect_outliers = [r for r in rows if (
                r['metrics']['smaller_rectangularity'] >= .9 if kind == 'J'
                else r['metrics']['smaller_rectangularity'] < .9)]
            checks.append(dict(kind=kind, metric='all_rectangularity_by_type', statistic='all_pairs',
                interpretation='strict per-pair audit; no silent reinterpretation as median',
                failed_count=len(rect_outliers), denominator=len(rows), passed=not rect_outliers,
                pair_ids=[r['pair_id'] for r in rect_outliers]))
        complementary = sum(r['metrics']['seam']['complementary_150'] for r in rows)
        if kind == 'J':
            checks.append(dict(kind=kind, metric='complementary_150_fraction',
                actual=complementary / len(rows), upper=.3, passed=complementary / len(rows) <= .3))
        expected_bases = {'rachel':100} if kind == 'M' else (
            {'torn_rachel':160, 'margin_fragment':24, 'torn_strip':16} if kind == 'J' else {'strip':150})
        actual_bases = dict(Counter(r['base'] for r in rows))
        checks.append(dict(kind=kind, metric='exact_base_quota', actual=actual_bases,
                           expected=expected_bases, passed=actual_bases == expected_bases))
        checks.append(dict(kind=kind, metric='all_correspondences_ge8',
            passed=metrics['healthy_correspondences']['min'] >= 8))
        groups[kind] = dict(positive=len(rows), measured=metrics,
            bend_over12=sum(r['metrics']['seam']['bend_range'] > 12 for r in rows),
            bend_over15=len(bend_outliers), rectangularity_outliers=len(rect_outliers),
            complementary_150_count=complementary, complementary_150_fraction=complementary / len(rows),
            outliers=[dict(pair_id=r['pair_id'], id=r['id'], base=r['base'], metrics=r['metrics'])
                      for r in {r['pair_id']:r for r in bend_outliers + rect_outliers}.values()],
            selection_bias=selection_bias(positive_entries),
            configured_shape=plan['SEAM'][kind], configured_damage=plan['MIS'][kind])
    return dict(schema='v42-select-population-assessment/1',
        status='passed_geometry_checks' if all(r['passed'] for r in checks) else 'not_admitted',
        checks=checks, groups=groups, changed_or_excluded_rows=0, training_admitted=False,
        calibration_split='select', test_used=False, model_score_used=False,
        caveats=['R overlap is enabled on only about 20% of pairs; wear can remove it. '
                 'An all-pair overlap median of zero must not be misreported as matching a nonzero real median.',
                 'Correspondence p10>=40 is an aspiration, not the per-pair minimum of eight.',
                 'The permitted fraction with bend in (12,15] is unspecified; it is reported, not silently tuned.',
                 'Rectangularity matches the supplied reference: minimum of the two ratios, '
                 'not the ratio of the fragment selected by pixel area.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output-new', type=Path, required=True)
    a = p.parse_args()
    manifest, audit, plan = [json.loads(f.read_text()) for f in (a.manifest, a.audit, a.plan)]
    if manifest['plan_sha256'] != sha(a.plan):
        raise ValueError('Generation plan changed')
    result = inspect(manifest, audit, plan, sha(a.manifest))
    result.update(manifest_sha256=sha(a.manifest), audit_sha256=sha(a.audit),
                  plan_sha256=sha(a.plan), script_sha256=sha(__file__))
    save(a.output_new, result)
    print(json.dumps(dict(status=result['status'], failed_checks=[r for r in result['checks'] if not r['passed']],
                         output=str(a.output_new), sha256=sha(a.output_new))))
    return 0 if result['status'] == 'passed_geometry_checks' else 2


if __name__ == '__main__':
    raise SystemExit(main())
