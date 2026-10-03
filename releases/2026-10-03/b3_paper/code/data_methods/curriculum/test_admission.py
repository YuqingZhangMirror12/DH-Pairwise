import unittest
from copy import deepcopy
from .admission import restrict


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(tasks={'v17.5': [dict(slot=1, recipe='wave', base_keys=['p', 'n']),
                                         dict(slot=2, recipe='clean', base_keys=['q', 'm'])], 'v18': []},
                         recipe_group_quotas={'v17.5': {'wave': 1, 'clean': 1}}, targets={'v17.5': 4})
        self.rows = [dict(version='v17.5', slot=t['slot'], recipe=t['recipe'], base_keys=t['base_keys'],
                         eligible=i == 1, original_measurement={'common_over_smaller_perimeter': .19 if i == 0 else .20})
                     for i, t in enumerate(self.plan['tasks']['v17.5'])]

    def test_filter_does_not_change_recipe_quota_or_parent_plan(self):
        saved = deepcopy(self.plan); result = restrict(self.plan, self.rows)
        self.assertEqual([t['slot'] for t in result['tasks']['v17.5']], [2])
        self.assertEqual(result['recipe_group_quotas'], saved['recipe_group_quotas'])
        self.assertEqual(self.plan, saved)

    def test_incomplete_or_duplicate_census_rejected(self):
        for rows in [self.rows[:1], self.rows + [self.rows[0]]]:
            with self.assertRaises(ValueError):
                restrict(self.plan, rows)

    def test_wrong_identity_or_recipe_rejected(self):
        for key, value in [('base_keys', ['wrong', 'n']), ('recipe', 'gaps')]:
            rows = deepcopy(self.rows); rows[0][key] = value
            with self.assertRaises(ValueError):
                restrict(self.plan, rows)

    def test_cannot_override_measured_eligibility(self):
        rows = deepcopy(self.rows); rows[0]['eligible'] = True
        with self.assertRaises(ValueError):
            restrict(self.plan, rows)

    def test_nonfinite_or_untyped_flag_rejected(self):
        for value in [float('nan'), float('inf')]:
            rows = deepcopy(self.rows); rows[0]['original_measurement']['common_over_smaller_perimeter'] = value
            with self.assertRaises(ValueError):
                restrict(self.plan, rows)
        rows = deepcopy(self.rows); rows[0]['eligible'] = 0
        with self.assertRaises(ValueError):
            restrict(self.plan, rows)


if __name__ == '__main__':
    unittest.main()
