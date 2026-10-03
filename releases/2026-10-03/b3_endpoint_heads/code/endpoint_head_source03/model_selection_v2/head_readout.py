"""CPU matched-head summaries and pre-TEST CAL operating points.

The primary threshold remains the native .20-.80 joint-F1 CAL rule. The separate
1% CAL-FPR threshold is a predeclared diagnostic, not a new TEST-selected default
and not a guarantee of 1% FPR after domain shift. Invalid/no-candidate scores use
zero for rank AUC, matching the original terminal scorer convention.
"""
import math
from fractions import Fraction

from .checkpoint_scan import module
from .protocol import canonical, digest, require

SCHEMA = 'mixed-select-head-pretest-operating-points/1'
VALIDATION_GROUPS = {'new_sim_cal', 'new_sim_select', 'dunhuang_real_cal', 'dunhuang_real_select'}


def check_rows(rows):
    require(rows and len({r['pair_id'] for r in rows}) == len(rows)
            and all(isinstance(r['pair_id'], str) and r['pair_id']
                    and all(type(r[key]) is bool for key in
                            ('label', 'has_candidate', 'numeric_valid', 'gt_known', 'layout20', 'candidate_coverage'))
                    and type(r['score']) in (int, float) and math.isfinite(r['score'])
                    and 0 <= r['score'] <= 1 for r in rows), 'unique finite typed prediction rows required')


def accepted(row, threshold):
    return row['numeric_valid'] and row['has_candidate'] and row['score'] >= threshold


def cal_fpr_one_percent(rows):
    require(rows and len({r['pair_id'] for r in rows}) == len(rows)
            and all(type(r['label']) is bool and type(r['has_candidate']) is bool
                    and type(r['numeric_valid']) is bool and type(r['score']) in (int, float)
                    and math.isfinite(r['score']) and 0 <= r['score'] <= 1 for r in rows),
            'unique finite CAL predictions required')
    negatives = [r for r in rows if not r['label']]
    require(negatives and len(negatives) < len(rows), 'both CAL classes required')
    allowed = len(negatives)//100
    active = sorted((r['score'] for r in negatives if r['numeric_valid'] and r['has_candidate']), reverse=True)
    # Lowest representable >=-threshold that excludes the next forbidden score.
    # An entire tied block is rejected, never split by Pair order. If score=1
    # must be excluded, nextafter(1,+inf) explicitly denotes reject-all.
    threshold = 0. if len(active) <= allowed else math.nextafter(float(active[allowed]), math.inf)
    fp = sum(accepted(r, threshold) for r in negatives)
    require(fp <= allowed, 'CAL false-positive constraint violated')
    return dict(rule='lowest_float64_threshold_cal_negative_fpr_le_1pct/1', threshold=threshold,
        calibration_pairs=len(rows), calibration_negatives=len(negatives), allowed_false_positives=allowed,
        actual_false_positives=fp, actual_fpr=fp/len(negatives), requested_fpr=.01,
        tied_scores_split=False, no_candidate_is_always_rejected=True,
        population_pair_ids_sha256=digest(sorted(r['pair_id'] for r in rows)),
        test_used=False, target_domain_fpr_guaranteed=False, affects_primary_threshold=False)


def calibration_record(groups, origin, rows_ref):
    require(set(groups) == VALIDATION_GROUPS
            and origin['selection_kind'] in ('sim_best', 'real_best'), 'fixed selected-head validation groups required')
    sim = cal_fpr_one_percent(groups['new_sim_cal'])
    dun = cal_fpr_one_percent(groups['dunhuang_real_cal']) if origin['selection_kind'] == 'real_best' else sim
    return canonical(dict(schema=SCHEMA, status='frozen_before_test', selection_kind=origin['selection_kind'],
        selected_model_sha256=origin['model_state_sha256'], selected_update=origin['selected_updates'],
        source_rows=rows_ref, primary_thresholds=origin['thresholds'],
        cal_fpr_1pct=dict(simulation=sim, dunhuang=dun, turufan=sim),
        dunhuang_calibration_source='dunhuang_real_cal' if origin['selection_kind'] == 'real_best' else 'new_sim_cal',
        turufan_calibration_source='new_sim_cal', test_used=False, target_domain_scores_seen=False,
        note='CAL FPR constraint only; report actual FPR separately on each target population.'))


def rank_auc(rows):
    """Mann-Whitney probability, including half credit for exact score ties."""
    scores = [(r['score'] if r['numeric_valid'] and r['has_candidate'] else 0., r['label']) for r in rows]
    positives = sum(label for _, label in scores); negatives = len(scores)-positives
    if not positives or not negatives:
        return None
    by_score = {}
    for score, label in scores:
        group = by_score.setdefault(score, [0, 0]); group[0 if label else 1] += 1
    wins = Fraction(0); seen_negatives = 0
    for score in sorted(by_score):
        positive, negative = by_score[score]
        wins += positive*seen_negatives + Fraction(positive*negative, 2)
        seen_negatives += negative
    return float(wins/(positives*negatives))


def summarize(rows, threshold, runtime, *, turufan=False):
    check_rows(rows)
    require(type(threshold) in (int, float) and math.isfinite(threshold) and threshold >= 0,
            'finite nonnegative frozen threshold required')
    if turufan:
        require(all(r['gt_known'] is False and not r['layout20'] for r in rows),
                'Turufan layout GT cannot be invented')
    result = module('s7_consensus_v1.metrics', runtime).summarize(rows, threshold)
    result['auroc'] = rank_auc(rows)
    result['false_positive_rate'] = result['fp']/result['negatives'] if result['negatives'] else None
    if turufan or not result['known_positive_layouts']:
        for key in list(result):
            if key.startswith('joint_') or key in ('layout20', 'layout20_count', 'candidate_coverage',
                    'candidate_coverage_count', 'covered_but_winner_wrong', 'winner_correct_but_rejected',
                    'positive_no_correct_candidate', 'wrong_pose_accepted'):
                result[key] = None
    return result


def summarize_group(rows, domain, origin, operating, runtime):
    require(domain in ('simulation', 'dunhuang', 'turufan')
            and operating.get('schema') == SCHEMA and operating.get('status') == 'frozen_before_test'
            and operating.get('selection_kind') == origin['selection_kind']
            and operating['selected_model_sha256'] == origin['model_state_sha256']
            and operating['selected_update'] == origin['selected_updates']
            and operating['primary_thresholds'] == origin['thresholds']
            and operating['test_used'] is False and operating['target_domain_scores_seen'] is False,
            'frozen pre-TEST operating-point/model identity differs')
    split = {'simulation': 'sim_test', 'dunhuang': 'dunhuang_cv', 'turufan': 'turufan'}[domain]
    metric = lambda threshold: summarize(rows, threshold, runtime, turufan=domain == 'turufan')
    return dict(primary=metric(origin['thresholds'][split]), fixed03=metric(.3),
        cal_fpr_1pct=metric(operating['cal_fpr_1pct'][domain]['threshold']),
        calibration_fpr_is_not_target_fpr=True, pair_ids_sha256=digest(sorted(r['pair_id'] for r in rows)))


def choice_readout(origin, validation, fresh, role_plan, operating, runtime):
    """Do not pool validation+TEST into a new choice or a new threshold."""
    require(set(validation) == VALIDATION_GROUPS
            and set(fresh) == {'sim_test', 'dunhuang_test', 'turufan', 'sim_straight_test'},
            'all required terminal populations must have actual completed predictions')
    groups = {name: summarize_group(rows, 'simulation' if name.startswith('new_sim') else 'dunhuang',
                                    origin, operating, runtime) for name, rows in validation.items()}
    for name, domain in [('sim_test', 'simulation'), ('sim_straight_test', 'simulation'), ('dunhuang_test', 'dunhuang')]:
        groups[name] = summarize_group(fresh[name], domain, origin, operating, runtime)
    dun = role_plan['datasets']['dunhuang_cv']; by_role = {}
    for role, rows in [('real_cal', validation['dunhuang_real_cal']), ('real_select', validation['dunhuang_real_select']),
                       ('real_test', fresh['dunhuang_test'])]:
        ids = [r['pair_id'] for r in rows]
        require(len(ids) == len(set(ids)) == dun['roles'][role]['pairs']
                and set(ids) == set(dun['roles'][role]['pair_ids']), 'Dunhuang role rows differ')
        by_role[role] = rows
    all_dun = [row for rows in by_role.values() for row in rows]
    require(len({r['pair_id'] for r in all_dun}) == len(all_dun)
            == sum(dun['roles'][role]['pairs'] for role in by_role), 'Dunhuang full population duplicates or misses rows')
    groups['dunhuang_full800_development_exposed'] = summarize_group(all_dun, 'dunhuang', origin, operating, runtime)
    all_turu = fresh['turufan']; by_id = {r['pair_id']: r for r in all_turu}
    turu = role_plan['datasets']['turufan']; seen = set(turu['excluded_gt_pair_ids'])
    require(not seen, 'the registered full Turufan population has no GT exclusions')
    for role in ('real_cal', 'real_select', 'real_test'):
        ids = turu['roles'][role]['pair_ids']
        require(len(ids) == len(set(ids)) == turu['roles'][role]['pairs']
                and not seen.intersection(ids) and set(ids) <= set(by_id), 'Turufan fold overlap/missing row')
        seen.update(ids)
        groups['turufan_'+role] = summarize_group([by_id[i] for i in ids], 'turufan', origin, operating, runtime)
    require(seen == set(by_id) and len(by_id) == len(all_turu), 'Turufan full membership differs')
    groups['turufan_full602_development_exposed'] = summarize_group(all_turu, 'turufan', origin, operating, runtime)
    return canonical(dict(origin=origin, groups=groups, primary_test_groups=['dunhuang_test', 'turufan_real_test'],
        validation_predictions_reused=True, full_real_populations_are_not_blind_tests=True,
        real_or_test_used_for_matcher_selection=False, test_used_for_head_selection=False,
        target_domain_threshold_refitted=False, rotation_ensemble=False, task3_overlay_applied=False))
