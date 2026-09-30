"""Post-prediction diagnostics; no model/search calls or training state changes.

D17 compatibility: a 'tie' means affinity >= true-partner affinity - 0.02,
including strictly better competitors, NOT absolute difference <= 0.02.
Both final affinity and ungained context cosine must be supplied so that an
increase in temperature gain alone is not presented as feature discrimination.
"""
from decimal import Decimal, ROUND_FLOOR
import math

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and np.isfinite(values).all(), 'finite metric vector required')
    if not len(values):
        return dict(n=0, mean=None, min=None, p10=None, p25=None, p50=None, p75=None, p90=None, max=None)
    qs = np.percentile(values, (0, 10, 25, 50, 75, 90, 100))
    return dict(n=len(values), mean=float(values.mean()), **dict(zip(
        ('min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max'), map(float, qs))))


def compact_evidence(points_a, points_b, valid_a, valid_b, affinity, context_cosine, q, dustbin_a, dustbin_b):
    """Retain compact and original indices; invalid padding may contain NaN."""
    pa, pb = np.asarray(points_a), np.asarray(points_b)
    va, vb = np.asarray(valid_a), np.asarray(valid_b)
    require(pa.ndim == pb.ndim == 2 and pa.shape[1] == pb.shape[1] == 2, 'Nx2 point arrays required')
    require(va.dtype == vb.dtype == np.bool_ and va.shape == (len(pa),) and vb.shape == (len(pb),),
            'boolean contour validity required')
    ia, ib = np.flatnonzero(va), np.flatnonzero(vb)
    require(np.isfinite(pa[ia]).all() and np.isfinite(pb[ib]).all(), 'valid points contain nonfinite values')
    result = dict(points_a=pa[ia], points_b=pb[ib], original_a=ia, original_b=ib)
    for key, value in (('affinity', affinity), ('context_cosine', context_cosine), ('q', q)):
        matrix = np.asarray(value)
        require(matrix.shape == (len(pa), len(pb)), 'prediction matrix shape differs: ' + key)
        matrix = matrix[np.ix_(ia, ib)]
        require(np.isfinite(matrix).all(), 'nonfinite valid prediction: ' + key)
        require(key != 'q' or (matrix >= 0).all(), 'negative Sinkhorn mass')
        result[key] = matrix
    for side, value, size, ids in (('a', dustbin_a, len(pa), ia), ('b', dustbin_b, len(pb), ib)):
        array = np.asarray(value)
        require(array.shape == (size,) and np.isfinite(array[ids]).all() and (array[ids] >= 0).all(),
                'invalid dustbin mass: ' + side)
        result['dustbin_' + side] = array[ids]
    return result


def _direction(pa, pb, affinity, cosine, q, dustbin, target):
    """Nearest geometric GT partner on the other valid contour, distance <=4px."""
    if not len(pa) or not len(pb):
        return dict(eligible_rows=0, nonzero_mass_rows=0, rows=[], metrics={})
    distances = np.linalg.norm(np.asarray(pb, float)[None] - np.asarray(pa, float)[:, None] - target, axis=2)
    partners = distances.argmin(axis=1)  # Deterministic lowest compact index for an exact tie.
    eligible = np.flatnonzero(distances[np.arange(len(pa)), partners] <= 4.)
    records = []
    for i in eligible:
        j = int(partners[i])
        weights = np.asarray(q[i], dtype=np.float64)
        mass = float(weights.sum())
        live = mass > 0.
        p = weights / mass if live else np.zeros_like(weights)
        top = int(weights.argmax()) if live else None
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum()) if live else None
        record = dict(row=int(i), partner=j, partner_distance_px=float(distances[i, j]),
            matched_row_mass=mass, dustbin_mass=float(dustbin[i]),
            real_mass_share_including_dustbin=mass / (mass + float(dustbin[i])) if mass + dustbin[i] > 0 else None,
            true_partner_q_share=float(p[j]) if live else None,
            d17_reference_q_share_floor1e12=float(weights[j] / max(mass, 1e-12)),
            q_mass_within_gt8_share=float(p[distances[i] <= 8].sum()) if live else None,
            q_top1_within_gt7=bool(distances[i, top] <= 7.) if live else None,
            q_top1_gt_error_px=float(distances[i, top]) if live else None,
            q_row_entropy=entropy, q_effective_partners=float(np.exp(entropy)) if live else None)
        for prefix, matrix in (('final_affinity', affinity), ('context_cosine', cosine)):
            row, true = matrix[i], matrix[i, j]
            record.update({
                prefix + '_d17_ties': int(np.count_nonzero(row >= true - .02)),
                prefix + '_absolute_ties': int(np.count_nonzero(np.abs(row - true) <= .02)),
                prefix + '_true_rank': int(np.count_nonzero(row > true)) + 1,
                prefix + '_true_score': float(true),
            })
        records.append(record)
    fields = [key for key in records[0] if key not in ('row', 'partner')] if records else []
    summaries = {key: distribution([r[key] for r in records if r[key] is not None]) for key in fields}
    return dict(eligible_rows=len(records), nonzero_mass_rows=sum(r['matched_row_mass'] > 0 for r in records),
                rows=records, metrics=summaries)


def diagnose_pair(evidence, gt_a_to_b_rc):
    """GT is joined to already frozen predictions; never adjusts Q or candidates."""
    target = np.asarray(gt_a_to_b_rc, dtype=np.float64)
    require(target.shape == (2,) and np.isfinite(target).all(), 'finite GT A-to-B translation required')
    pa, pb = evidence['points_a'], evidence['points_b']
    directions = {
        'a_to_b': _direction(pa, pb, evidence['affinity'], evidence['context_cosine'], evidence['q'],
                             evidence['dustbin_a'], target),
        'b_to_a': _direction(pb, pa, evidence['affinity'].T, evidence['context_cosine'].T, evidence['q'].T,
                             evidence['dustbin_b'], -target),
    }
    for side, result in directions.items():
        a, b = ('a', 'b') if side == 'a_to_b' else ('b', 'a')
        for row in result['rows']:
            row['original_row'] = int(evidence['original_' + a][row['row']])
            row['original_partner'] = int(evidence['original_' + b][row['partner']])
    return dict(schema='matcher-v2-ridge-pair/1', directions=directions,
        valid_points_a=len(pa), valid_points_b=len(pb), gt_used_for_forward=False,
        gt_used_for_candidate_search=False, q_modified=False,
        definitions=dict(partner='nearest sampled other-contour point under GT; distance <=4px',
            d17_ties='score >= true partner score -0.02, including strictly better points',
            q_share_denominator='all real valid partner columns, excluding dustbin',
            zero_real_mass='null conditional Q metrics, explicitly counted; not a zero-error success',
            no_seam_rows='empty distributions, retained as unmeasurable; never dropped from layout metrics'))


def classify_positive_geometry(bend_range, rectangularity_a, rectangularity_b):
    """Fixed reference J/R/curved definition, independent of model predictions."""
    if bend_range is None:
        return 'unmeasurable'
    values = (bend_range, rectangularity_a, rectangularity_b)
    require(all(math.isfinite(float(v)) for v in values) and all(float(v) >= 0 for v in values),
            'finite nonnegative mask-only geometry required')
    if bend_range > 12.:
        return 'curved'
    return 'R' if min(rectangularity_a, rectangularity_b) >= .9 else 'J'


def summarize_ridge_population(rows):
    """Expose both row-weighted and equal-pair distributions, including misses."""
    require(rows and len({r['pair_id'] for r in rows}) == len(rows), 'complete unique ridge population required')
    require(all(r['role'] in ('real_cal', 'real_select') and r['seam_group'] in ('J', 'R', 'curved', 'unmeasurable')
                and r['diagnostic']['schema'] == 'matcher-v2-ridge-pair/1' for r in rows),
            'only predefined development positives may enter this summary')
    fields = ('final_affinity_d17_ties', 'context_cosine_d17_ties', 'final_affinity_true_rank',
        'context_cosine_true_rank', 'true_partner_q_share', 'q_mass_within_gt8_share',
        'q_top1_within_gt7', 'q_top1_gt_error_px', 'matched_row_mass',
        'real_mass_share_including_dustbin', 'q_row_entropy', 'q_effective_partners')
    output = {}
    for role in ('real_cal', 'real_select'):
        for group in ('all', 'J', 'R', 'curved', 'unmeasurable'):
            pairs = [r for r in rows if r['role'] == role and (group == 'all' or r['seam_group'] == group)]
            result = dict(pair_count=len(pairs), directions={})
            for direction in ('a_to_b', 'b_to_a'):
                per_pair = [r['diagnostic']['directions'][direction] for r in pairs]
                token_rows = [r for p in per_pair for r in p['rows']]
                result['directions'][direction] = dict(
                    eligible_row_count=sum(p['eligible_rows'] for p in per_pair),
                    zero_real_mass_row_count=sum(p['eligible_rows'] - p['nonzero_mass_rows'] for p in per_pair),
                    unmeasurable_pair_count=sum(p['eligible_rows'] == 0 for p in per_pair),
                    row_weighted={k: distribution([r[k] for r in token_rows if r[k] is not None]) for k in fields},
                    equal_pair_medians={k: distribution([p['metrics'][k]['p50'] for p in per_pair
                        if k in p['metrics'] and p['metrics'][k]['p50'] is not None]) for k in fields})
            output[role + '/' + group] = result
    return dict(schema='matcher-v2-ridge-population/1', pairs=len(rows), groups=output,
        test_used=False, real_backpropagation=False, model_effectiveness_not_implied=True)


def _checked_predictions(rows, role):
    require(rows and len({r['pair_id'] for r in rows}) == len(rows), 'nonempty unique prediction population required')
    for row in rows:
        require(row.get('role') == role, 'forbidden calibration/evaluation role')
        require(all(type(row.get(key)) is bool for key in ('label', 'numeric_valid', 'has_candidate')),
                'explicit boolean targets/validity required')
        require(math.isfinite(float(row['score'])) and float(row['score']) >= 0, 'finite nonnegative score required')
    return rows


def accepted(row, threshold):
    return row['numeric_valid'] and row['has_candidate'] and row['score'] >= threshold


def calibrate_negative_budget(calibration_rows, fraction):
    """Fit only CAL negative scores; tied negatives cannot be split artificially.

    Select the lowest nonnegative threshold whose empirical CAL false positives
    are <= floor(target * all CAL negatives). SELECT FPR is NOT guaranteed.
    """
    rows = _checked_predictions(calibration_rows, 'real_cal')
    require(0 <= fraction < 1, 'negative budget must lie in [0,1)')
    negative = [r for r in rows if not r['label']]
    require(negative, 'CAL must have negative examples')
    budget = int((Decimal(str(fraction)) * len(negative)).to_integral_value(rounding=ROUND_FLOOR))
    scores = sorted((float(r['score']) for r in negative if r['numeric_valid'] and r['has_candidate']), reverse=True)
    threshold = float(np.nextafter(scores[budget], np.inf)) if len(scores) > budget else 0.
    fp = sum(accepted(r, threshold) for r in negative)
    require(fp <= budget and math.isfinite(threshold), 'negative calibration budget failed')
    return dict(schema='cal-negative-budget/1', threshold=threshold, requested_fraction=fraction,
        negative_count=len(negative), allowed_false_positives=budget, actual_false_positives=fp,
        actual_fraction=fp / len(negative), selected_on='CAL negatives only', test_used=False,
        select_used=False, tied_scores_split=False, comparison='score >= threshold')


def score_at_calibrated_budget(selection_rows, calibration, positive_groups):
    """One CAL threshold, shared across J/R/curved; negative FPR is global."""
    rows = _checked_predictions(selection_rows, 'real_select')
    require(calibration['schema'] == 'cal-negative-budget/1' and not calibration['select_used']
            and not calibration['test_used'], 'CAL-only frozen threshold required')
    positive_ids = {r['pair_id'] for r in rows if r['label']}
    require(set(positive_groups) == positive_ids, 'mask-only groups must cover every SELECT positive exactly')
    require(all(g in ('J', 'R', 'curved', 'unmeasurable') for g in positive_groups.values()), 'unknown seam category')
    require(all(type(r.get('layout20')) is bool and type(r.get('candidate_coverage')) is bool
                for r in rows if r['label']), 'known Dunhuang positive layout/coverage required')
    t = calibration['threshold']
    positives = [r for r in rows if r['label']]
    negatives = [r for r in rows if not r['label']]
    require(positives and negatives, 'both classes required')
    tp = sum(accepted(r, t) for r in positives)
    fp = sum(accepted(r, t) for r in negatives)
    fn, tn = len(positives) - tp, len(negatives) - fp
    groups = {}
    for kind in ('J', 'R', 'curved', 'unmeasurable'):
        subset = [r for r in positives if positive_groups[r['pair_id']] == kind]
        groups[kind] = dict(positive_count=len(subset), covered=sum(r['candidate_coverage'] for r in subset),
            layout_correct=sum(r['layout20'] for r in subset),
            layout_correct_and_accepted=sum(r['layout20'] and accepted(r, t) for r in subset))
    return dict(schema='select-at-cal-negative-budget/1', threshold=t, calibration=calibration,
        pair_count=len(rows), tp=tp, fp=fp, tn=tn, fn=fn, accuracy=(tp + tn) / len(rows),
        pair_f1=2 * tp / (2 * tp + fp + fn), observed_select_fpr=fp / len(negatives),
        positive_groups=groups, layout_correct=sum(r['layout20'] for r in positives),
        layout_correct_and_accepted=sum(r['layout20'] and accepted(r, t) for r in positives),
        threshold_refitted_on_select=False, group_specific_thresholds=False,
        caveat='CAL target FPR does not guarantee SELECT FPR; J/R/curved are positive-only mask/GT strata')
