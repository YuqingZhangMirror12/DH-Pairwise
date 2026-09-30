"""Synthetic confusion counts and matched-policy checks, not model results."""
from copy import deepcopy
import unittest

from . import classification_review as c


def counts(tp=40, tn=60, fp=1, fn=21):
    return dict(pairs=tp+tn+fp+fn, positives=tp+fn, negatives=tn+fp,
                tp=tp, tn=tn, fp=fp, fn=fn, threshold=.32,
                layout20=None, layout20_count=None, joint_f1=None)


def row(model=c.BASELINE, selection='sim', policy='primary', metrics=None):
    return dict(model=model, selection_kind=selection, split='turufan', population='real_test',
                policy=policy, comparison_group='|'.join((selection,'turufan','real_test',policy)),
                metrics=counts() if metrics is None else metrics,
                checkpoint_sha256='a'*64, matcher_sha256=c.E32_SHA, selected_epoch=22,
                actual_head_epochs=28, matcher_selected_epoch=32, data_contract_sha256='b'*64,
                real_plan_sha256=c.REAL_PLAN_SHA, source={})


def evidence(rows=None, available=None):
    return dict(schema='binary-report-all-models/1', status='evidence_ready_not_final_conclusions',
                neural_inference_repeated=False, thresholds_refitted=False,
                available_light_experiments=[] if available is None else available,
                rows=[row()] if rows is None else rows)


class ClassificationReviewTests(unittest.TestCase):
    def test_omitted_redundant_tn_is_derived(self):
        data = counts(); del data['tn']; before = deepcopy(data)
        result = c.classification_counts(data)
        self.assertEqual(result['tn'], 60)
        self.assertAlmostEqual(result['accuracy'], 100/122)
        self.assertEqual(data, before)

    def test_invalid_complement_cannot_be_derived(self):
        for negative, fp in ((1, 2), (True, 0), (3, 1.), (3, -1)):
            data = counts(); del data['tn']; data.update(negatives=negative, fp=fp)
            with self.assertRaises(ValueError): c.classification_counts(data)

    def test_explicit_null_tn_is_not_silently_replaced(self):
        data = counts(); data['tn'] = None
        with self.assertRaises(ValueError): c.classification_counts(data)

    def test_confusion_arithmetic(self):
        result = c.classification_counts(counts())
        self.assertAlmostEqual(result['accuracy'], 100/122)
        self.assertAlmostEqual(result['precision'], 40/41)
        self.assertAlmostEqual(result['recall'], 40/61)
        self.assertAlmostEqual(result['false_positive_rate'], 1/61)
        self.assertAlmostEqual(result['f1'], 80/102)

    def test_no_accepted_pairs_not_perfect_precision(self):
        result = c.classification_counts(counts(0,61,0,61))
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['precision'], 0)
        self.assertEqual(result['f1'], 0)

    def test_no_negative_assumption(self):
        result = c.classification_counts(counts(61,0,61,0))
        self.assertEqual(result['false_positive_rate'], 1)
        self.assertEqual(result['accuracy'], .5)

    def test_bad_denominators_fail(self):
        for key in ('pairs','positives','negatives'):
            value = counts(); value[key] += 1
            with self.assertRaises(ValueError): c.classification_counts(value)

    def test_float_or_bool_counts_fail(self):
        for value in (True,1.,-1):
            data = counts(); data['fp'] = value
            with self.assertRaises(ValueError): c.classification_counts(data)

    def test_inconsistent_or_nonfinite_rate_fails(self):
        for value in (.9,None,float('nan'),float('inf')):
            data = counts(); data['accuracy'] = value
            with self.assertRaises(ValueError): c.classification_counts(data)

    def test_accuracy_is_not_joint_f1(self):
        data = counts(); data['accuracy'] = 100/122; data['joint_f1'] = .1
        self.assertAlmostEqual(c.classification_counts(data)['accuracy'], 100/122)

    def test_missing_light_not_zero_result(self):
        result = c.review(evidence())
        self.assertEqual(result['status'], 'waiting_for_completed_light_imports')
        self.assertEqual(result['available_light_heads'], [])
        self.assertEqual(len(result['rows']), 1)

    def test_final_delivery_requires_both(self):
        with self.assertRaisesRegex(ValueError,'both completed'): c.review(evidence(), True)
        data = evidence([row(),row('binary_patch')], ['binary_patch'])
        with self.assertRaisesRegex(ValueError,'both completed'): c.review(data, True)

    def test_tradeoff_retained_not_blanket_win(self):
        data = evidence([row(),row('binary_patch',metrics=counts(50,57,4,11)),row('binary_stats')],list(c.LIGHT))
        result = c.review(data, True)
        delta = result['rows'][1]['old_head_comparison']
        self.assertAlmostEqual(delta['rates']['accuracy']['percentage_points'], 100*7/122)
        self.assertEqual(delta['additional_false_positives'], 3)
        self.assertEqual(delta['additional_false_negatives'], -10)
        self.assertEqual(delta['rates']['precision']['direction'], 'lower')
        self.assertFalse(delta['single_layer_causal_claimed'])

    def test_same_metrics_zero_change(self):
        data = evidence([row(),row('binary_patch')], ['binary_patch'])
        delta = c.review(data)['rows'][1]['old_head_comparison']
        self.assertTrue(all(v['direction']=='unchanged' for v in delta['rates'].values()))

    def test_real_selection_not_compared_with_sim_baseline(self):
        data = evidence([row(),row('binary_patch',selection='real')], ['binary_patch'])
        delta = c.review(data)['rows'][1]['old_head_comparison']
        self.assertEqual(delta['status'], 'no_same_selection_policy_baseline')
        self.assertNotIn('rates', delta)

    def test_fixed_threshold_not_mixed_with_primary(self):
        item = row('binary_patch',policy='fixed03'); item['metrics']['threshold'] = .3
        result = c.review(evidence([row(),item], ['binary_patch']))
        self.assertEqual(result['rows'][1]['old_head_comparison']['status'], 'no_same_selection_policy_baseline')

    def test_bad_fixed_threshold_rejected(self):
        item = row('binary_patch',policy='fixed03')
        with self.assertRaisesRegex(ValueError,'fixed0.30'): c.review(evidence([row(),item],['binary_patch']))

    def test_wrong_matcher_cannot_claim_control(self):
        item = row('binary_patch'); item['matcher_sha256'] = 'd'*64
        with self.assertRaisesRegex(ValueError,'different Matcher'): c.review(evidence([row(),item],['binary_patch']))

    def test_role_plan_mismatch_rejected(self):
        item = row(); item['real_plan_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError,'source roles'): c.review(evidence([item]))

    def test_turufan_missing_gt_not_zero(self):
        item = row(); item['metrics']['layout20_count'] = 0
        with self.assertRaisesRegex(ValueError,'no layout GT'): c.review(evidence([item]))

    def test_class_population_mismatch_rejected(self):
        item = row('binary_patch',metrics=counts(40,60,1,20))
        with self.assertRaisesRegex(ValueError,'populations differ'): c.review(evidence([row(),item],['binary_patch']))

    def test_same_size_different_recorded_cohort_rejected(self):
        old = row(); old['cohort_sha256'] = 'a'*64
        item = row('binary_patch'); item['cohort_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError,'cohort changed'): c.review(evidence([old,item],['binary_patch']))

    def test_duplicate_result_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate result'): c.review(evidence([row(),row()]))

    def test_sources_unchanged(self):
        data = evidence([row(),row('binary_patch')],['binary_patch']); before = deepcopy(data)
        c.review(data); self.assertEqual(data,before)


if __name__ == '__main__': unittest.main()
