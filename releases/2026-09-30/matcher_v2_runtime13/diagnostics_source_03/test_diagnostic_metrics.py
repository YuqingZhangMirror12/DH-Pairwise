import copy
import unittest

import numpy as np

from diagnostic_metrics import (compact_evidence, diagnose_pair, distribution,
    classify_positive_geometry, summarize_ridge_population, calibrate_negative_budget,
    score_at_calibrated_budget)


def evidence(q=None, affinity=None, cosine=None):
    points = np.array([[0., 0.], [0., 10.], [0., 20.]])
    q = np.ones((3, 3)) / 9 if q is None else q
    affinity = np.eye(3) * .01 if affinity is None else affinity
    cosine = affinity if cosine is None else cosine
    return compact_evidence(points, points + [6., -8.], np.ones(3, bool), np.ones(3, bool),
        affinity, cosine, q, np.full(3, .1), np.full(3, .1))


def prediction(pid, score, label=False, role='real_cal', eligible=True, correct=False):
    return dict(pair_id=pid, score=score, label=label, role=role, numeric_valid=eligible,
                has_candidate=eligible, layout20=correct, candidate_coverage=correct)


class RidgeMetrics(unittest.TestCase):
    def test_flat_ridge_versus_correct_peak(self):
        flat = diagnose_pair(evidence(), [6., -8.])['directions']['a_to_b']
        peak = diagnose_pair(evidence(q=np.eye(3) / 3), [6., -8.])['directions']['a_to_b']
        self.assertEqual(flat['eligible_rows'], 3)
        self.assertAlmostEqual(flat['metrics']['true_partner_q_share']['p50'], 1/3)
        self.assertEqual(peak['metrics']['true_partner_q_share']['p50'], 1.)
        self.assertEqual(peak['metrics']['q_top1_within_gt7']['mean'], 1.)
        self.assertAlmostEqual(flat['metrics']['q_effective_partners']['p50'], 3.)
        self.assertEqual(peak['metrics']['q_effective_partners']['p50'], 1.)

    def test_d17_counts_better_candidates_not_only_absolute_ties(self):
        affinity = np.array([[.1, .4, -.2], [.4, .1, -.2], [.4, -.2, .1]])
        row = diagnose_pair(evidence(affinity=affinity), [6., -8.])['directions']['a_to_b']['rows'][0]
        self.assertEqual(row['final_affinity_d17_ties'], 2)
        self.assertEqual(row['final_affinity_absolute_ties'], 1)
        self.assertEqual(row['final_affinity_true_rank'], 2)

    def test_gain_alone_does_not_improve_context_ties(self):
        cosine = np.eye(3) * .01
        low = diagnose_pair(evidence(affinity=cosine, cosine=cosine), [6., -8.])
        high = diagnose_pair(evidence(affinity=cosine * 5, cosine=cosine), [6., -8.])
        l, h = [x['directions']['a_to_b']['rows'][0] for x in (low, high)]
        self.assertEqual(l['context_cosine_d17_ties'], h['context_cosine_d17_ties'])
        self.assertGreater(l['final_affinity_d17_ties'], h['final_affinity_d17_ties'])

    def test_zero_mass_is_null_not_perfect_localization(self):
        row = diagnose_pair(evidence(q=np.zeros((3, 3))), [6., -8.])['directions']['a_to_b']
        self.assertEqual(row['nonzero_mass_rows'], 0)
        self.assertIsNone(row['rows'][0]['true_partner_q_share'])
        self.assertIsNone(row['rows'][0]['q_top1_within_gt7'])
        self.assertEqual(row['metrics']['true_partner_q_share']['n'], 0)
        self.assertEqual(row['rows'][0]['real_mass_share_including_dustbin'], 0.)

    def test_no_geometric_seam_retained_as_unmeasurable(self):
        pair = diagnose_pair(evidence(), [100., 100.])
        self.assertEqual(pair['directions']['a_to_b']['eligible_rows'], 0)
        self.assertEqual(pair['directions']['b_to_a']['rows'], [])

    def test_swap_transpose_symmetry_and_no_mutation(self):
        ev = evidence(q=np.array([[.1, .2, .3], [.2, .5, .4], [.7, .3, .1]]))
        before = copy.deepcopy(ev)
        swapped = dict(points_a=ev['points_b'], points_b=ev['points_a'],
            original_a=ev['original_b'], original_b=ev['original_a'],
            affinity=ev['affinity'].T, context_cosine=ev['context_cosine'].T, q=ev['q'].T,
            dustbin_a=ev['dustbin_b'], dustbin_b=ev['dustbin_a'])
        a = diagnose_pair(ev, [6., -8.])
        b = diagnose_pair(swapped, [-6., 8.])
        self.assertEqual(a['directions']['a_to_b'], b['directions']['b_to_a'])
        for key in ev:
            np.testing.assert_array_equal(ev[key], before[key])

    def test_invalid_nan_padding_ignored_valid_nan_rejected(self):
        ev = evidence()
        pa = np.concatenate([ev['points_a'], [[np.nan, np.nan]]])
        pb = np.concatenate([ev['points_b'], [[np.nan, np.nan]]])
        matrix = np.pad(ev['q'], ((0, 1), (0, 1)), constant_values=np.nan)
        valid = np.array([True, True, True, False])
        compact = compact_evidence(pa, pb, valid, valid, matrix, matrix, matrix,
                                    np.array([.1, .1, .1, np.nan]), np.array([.1, .1, .1, np.nan]))
        self.assertEqual(compact['q'].shape, (3, 3))
        valid[:] = True
        with self.assertRaisesRegex(ValueError, 'valid points'):
            compact_evidence(pa, pb, valid, valid, matrix, matrix, matrix, np.ones(4), np.ones(4))

    def test_negative_q_rejected(self):
        with self.assertRaisesRegex(ValueError, 'negative Sinkhorn'):
            evidence(q=-np.eye(3))

    def test_mask_geometry_category_boundaries(self):
        self.assertEqual(classify_positive_geometry(12., .9, .95), 'R')
        self.assertEqual(classify_positive_geometry(12., .899, .95), 'J')
        self.assertEqual(classify_positive_geometry(12.01, .95, .95), 'curved')
        self.assertEqual(classify_positive_geometry(None, .95, .95), 'unmeasurable')

    def test_row_and_pair_weighting_and_missing_cases_explicit(self):
        pair = diagnose_pair(evidence(), [6., -8.])
        absent = diagnose_pair(evidence(), [100., 100.])
        summary = summarize_ridge_population([
            dict(pair_id='one', role='real_select', seam_group='J', diagnostic=pair),
            dict(pair_id='two', role='real_select', seam_group='J', diagnostic=absent)])
        group = summary['groups']['real_select/J']
        self.assertEqual(group['pair_count'], 2)
        self.assertEqual(group['directions']['a_to_b']['unmeasurable_pair_count'], 1)
        self.assertEqual(group['directions']['a_to_b']['row_weighted']['true_partner_q_share']['n'], 3)
        self.assertEqual(group['directions']['a_to_b']['equal_pair_medians']['true_partner_q_share']['n'], 1)

    def test_quantiles_not_only_median(self):
        result = distribution([0, 1, 2, 3, 100])
        self.assertEqual(result['p50'], 2.)
        self.assertEqual(result['p25'], 1.)
        self.assertEqual(result['max'], 100.)


class FixedFalsePositiveBudgets(unittest.TestCase):
    def test_tied_negatives_never_split_to_hit_budget(self):
        cal = [prediction(str(i), .8 if i < 3 else .1) for i in range(100)]
        result = calibrate_negative_budget(cal, .02)
        self.assertEqual(result['allowed_false_positives'], 2)
        self.assertEqual(result['actual_false_positives'], 0)
        self.assertGreater(result['threshold'], .8)

    def test_fraction_floor_exact_and_invalid_rows_in_denominator(self):
        cal = [prediction(str(i), .5, eligible=i < 1) for i in range(100)]
        result = calibrate_negative_budget(cal, .01)
        self.assertEqual(result['actual_false_positives'], 1)
        self.assertEqual(result['negative_count'], 100)
        self.assertEqual(result['threshold'], 0.)

    def test_cal_positives_do_not_fit_threshold(self):
        cal = [prediction(str(i), i / 100) for i in range(100)]
        expected = calibrate_negative_budget(cal, .05)
        result = calibrate_negative_budget(cal + [prediction('positive', 1., True)], .05)
        self.assertEqual(expected, result)

    def test_select_and_test_cannot_enter_calibration(self):
        for role in ('real_select', 'real_test'):
            with self.assertRaisesRegex(ValueError, 'forbidden'):
                calibrate_negative_budget([prediction('x', .2, role=role)], .02)

    def test_select_actual_fpr_can_exceed_cal_target_and_is_reported(self):
        cal = [prediction(str(i), i / 100) for i in range(100)]
        cut = calibrate_negative_budget(cal, .02)
        selected = [prediction('J', .99, True, 'real_select', correct=True),
                    prediction('R', .3, True, 'real_select', correct=True),
                    prediction('C', .99, True, 'real_select', correct=False),
                    prediction('neg', 1., False, 'real_select')]
        result = score_at_calibrated_budget(selected, cut, {'J': 'J', 'R': 'R', 'C': 'curved'})
        self.assertEqual(result['observed_select_fpr'], 1.)
        self.assertEqual(result['layout_correct'], 2)
        self.assertEqual(result['layout_correct_and_accepted'], 1)
        self.assertEqual(result['tp'], 2)
        self.assertEqual(result['positive_groups']['curved']['layout_correct_and_accepted'], 0)

    def test_missing_layout_or_groups_are_not_assumed_correct(self):
        cut = calibrate_negative_budget([prediction('cal', .1)], .02)
        rows = [prediction('pos', .5, True, 'real_select'), prediction('neg', .1, False, 'real_select')]
        with self.assertRaisesRegex(ValueError, 'cover every'):
            score_at_calibrated_budget(rows, cut, {})
        rows[0]['layout20'] = None
        with self.assertRaisesRegex(ValueError, 'known Dunhuang'):
            score_at_calibrated_budget(rows, cut, {'pos': 'J'})


if __name__ == '__main__':
    unittest.main()
