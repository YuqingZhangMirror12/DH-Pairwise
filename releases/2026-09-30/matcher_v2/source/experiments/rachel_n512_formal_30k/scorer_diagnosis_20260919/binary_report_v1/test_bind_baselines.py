"""Report-binding tests use synthetic values, never claim model performance."""
import copy
import unittest

from .assemble import REAL_PLAN_SHA
from .bind_baselines import LABELS, reviewed_rows, assert_append_only


def fixture():
    rows = []
    for model in LABELS:
        for split in ('dunhuang_cv', 'turufan'):
            n, pos, neg = (161, 59, 102) if split == 'dunhuang_cv' else (122, 61, 61)
            for policy in ('primary', 'fixed03'):
                rows.append(dict(model=model, split=split, population='real_test',
                    selection_kind='sim', policy=policy, real_plan_sha256=REAL_PLAN_SHA,
                    source={'synthetic_test': True}, metrics=dict(pairs=n, positives=pos,
                        negatives=neg, layout20_count=20 if split == 'dunhuang_cv' else None,
                        joint_f1=.4 if split == 'dunhuang_cv' else None, threshold=.3)))
    return dict(schema='binary-report-all-models/1', neural_inference_repeated=False,
                thresholds_refitted=False, rows=rows)


class BindingTests(unittest.TestCase):
    def test_all_rows_preserve_source_and_do_not_mutate(self):
        item = fixture(); original = copy.deepcopy(item)
        result = reviewed_rows(item)
        self.assertEqual(len(result), 16)
        self.assertEqual(item, original)
        self.assertTrue(all(r['source']['synthetic_test'] for r in result))

    def test_no_other_population_is_silently_pooled(self):
        item = fixture(); extra = copy.deepcopy(item['rows'][0]); extra['population'] = 'real_select'
        item['rows'].append(extra)
        self.assertEqual(len(reviewed_rows(item)), 16)

    def test_old_three_model_report_still_supported(self):
        item = fixture()
        item['rows'] = [r for r in item['rows'] if r['model'] != 'threshold_scratch_fixed']
        self.assertEqual(len(reviewed_rows(item)), 12)

    def test_extension_preserves_every_existing_row(self):
        rows = reviewed_rows(fixture())
        old = [r for r in rows if r['model'] != 'threshold_scratch_fixed']
        assert_append_only(old, rows)
        changed = copy.deepcopy(rows); changed[0]['metrics']['threshold'] = .7
        with self.assertRaisesRegex(ValueError, 'changed or removed'):
            assert_append_only(old, changed)
        with self.assertRaisesRegex(ValueError, 'changed or removed'):
            assert_append_only(old, rows[1:])

    def test_missing_or_duplicate_rows_fail(self):
        for drop in (True, False):
            item = fixture()
            if drop: item['rows'].pop()
            else: item['rows'][-1] = copy.deepcopy(item['rows'][0])
            with self.assertRaises((ValueError, AssertionError, RuntimeError)):
                reviewed_rows(item)

    def test_wrong_roles_or_selection_fail(self):
        for key, value in [('real_plan_sha256', 'changed'), ('selection_kind', 'real')]:
            item = fixture(); item['rows'][0][key] = value
            with self.assertRaises((ValueError, AssertionError, RuntimeError)):
                reviewed_rows(item)

    def test_class_counts_cannot_match_only_total(self):
        item = fixture(); item['rows'][0]['metrics'].update(positives=58, negatives=103)
        with self.assertRaises((ValueError, AssertionError, RuntimeError)):
            reviewed_rows(item)

    def test_turufan_unavailable_is_not_zero(self):
        item = fixture(); item['rows'][2]['metrics']['layout20_count'] = 0
        with self.assertRaises((ValueError, AssertionError, RuntimeError)):
            reviewed_rows(item)


if __name__ == '__main__':
    unittest.main()
