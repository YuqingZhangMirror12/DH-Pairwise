import unittest
from ..prepare_independent_probes import prepare


class IndependentProbeCohort(unittest.TestCase):
    def setUp(self):
        self.original = dict(selection='fixed historical sample', cases=[
            dict(split='dunhuang', pair_id='a', reason='historical'),
            dict(split='turufan', pair_id='b', reason='rejected positive')])
        self.populations = {'dunhuang_cv': {'a'}, 'turufan': {'b'}}

    def test_alias_preserves_ids_and_reasons(self):
        result = prepare(self.original, self.populations, set())
        self.assertEqual(result['cases'][0],
                         dict(split='dunhuang_cv', pair_id='a', reason='historical'))
        self.assertEqual(result['cases'][1], self.original['cases'][1])
        self.assertEqual(self.original['cases'][0]['split'], 'dunhuang')

    def test_invalid_gt_is_explicitly_omitted(self):
        result = prepare(self.original, self.populations, {'a'})
        self.assertEqual([r['pair_id'] for r in result['cases']], ['b'])
        self.assertEqual(result['preparation']['omitted_invalid_gt'][0]['pair_id'], 'a')

    def test_missing_case_or_alias_duplicate_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare(self.original, {'dunhuang_cv': set(), 'turufan': {'b'}}, set())
        duplicate = dict(self.original, cases=self.original['cases'] +
                         [dict(split='dunhuang_cv', pair_id='a', reason='other')])
        with self.assertRaises(ValueError):
            prepare(duplicate, self.populations, set())


if __name__ == '__main__':
    unittest.main()
