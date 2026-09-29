import copy
import unittest

from .bind_aggressive_results import (append_queries, identity, MODEL, QUERIES,
                                     METRICS_QUERY, CASES_QUERY, POS_QUERY, CF_QUERY, REPORT_ID)


def row(query, model):
    r = dict(model=model, selection_kind='sim', split='dunhuang_cv',
             population='real_test', policy='primary', intervention='actual',
             case_key=model+'|sim|dunhuang_cv|1', score=.3)
    return r


class AppendV17Tests(unittest.TestCase):
    def setUp(self):
        self.snapshot = dict(id=REPORT_ID, buildStatus='complete', queries={
            q: dict(rows=[row(q, 'binary_patch')], source={'old': 'preserve'}) for q in QUERIES})
        self.snapshot['queries']['human_history'] = {'rows': [{'success': True}]}
        self.new = {q: [row(q, MODEL)] for q in QUERIES}

    def test_append_does_not_modify_existing_rows_or_sources(self):
        original = copy.deepcopy(self.snapshot)
        out = append_queries(self.snapshot, self.new, {'new': True})
        self.assertEqual(original, self.snapshot)
        self.assertEqual(out['queries']['human_history'], original['queries']['human_history'])
        for q in QUERIES:
            self.assertEqual(out['queries'][q]['rows'][0], original['queries'][q]['rows'][0])
            self.assertEqual(out['queries'][q]['source']['old'], 'preserve')
            self.assertEqual(len(out['queries'][q]['rows']), 2)
        self.assertEqual(out['buildStatus'], 'updating')

    def test_second_binding_rejected(self):
        once = append_queries(self.snapshot, self.new, {})
        with self.assertRaises(ValueError): append_queries(once, self.new, {})

    def test_wrong_report_rejected(self):
        self.snapshot['id'] = 'another-report'
        with self.assertRaises(ValueError): append_queries(self.snapshot, self.new, {})

    def test_incomplete_query_set_rejected(self):
        del self.new[POS_QUERY]
        with self.assertRaises(ValueError): append_queries(self.snapshot, self.new, {})

    def test_duplicate_new_rows_rejected(self):
        self.new[CASES_QUERY] *= 2
        with self.assertRaises(ValueError): append_queries(self.snapshot, self.new, {})

    def test_other_model_cannot_be_overwritten(self):
        self.new[METRICS_QUERY][0]['model'] = 'binary_stats'
        with self.assertRaises(ValueError): append_queries(self.snapshot, self.new, {})

    def test_population_is_part_of_metric_identity(self):
        a = row(METRICS_QUERY, MODEL); b = dict(a, population='gt_corrected_800_development_context')
        self.assertNotEqual(identity(METRICS_QUERY, a), identity(METRICS_QUERY, b))

    def test_intervention_is_part_of_cf_identity(self):
        a = row(CF_QUERY, MODEL); b = dict(a, intervention='no_conflict')
        self.assertNotEqual(identity(CF_QUERY, a), identity(CF_QUERY, b))


if __name__ == '__main__': unittest.main()
