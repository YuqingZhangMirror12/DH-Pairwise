import copy
import unittest

from freeze_development_groups import development_members


def fixture():
    meta = dict(pairs=[], fragment_source_group={})
    for pid, fold in (('cal', 1), ('select', 2), ('test', 0)):
        meta['pairs'].append(dict(pair_id=pid, fold=fold, label=True,
                                 fragment_a_id=pid+'a', fragment_b_id=pid+'b'))
        meta['fragment_source_group'].update({pid+'a': pid, pid+'b': pid})
    plan = dict(role_folds={'real_cal': [1], 'real_select': [2, 3, 4], 'real_test': [0]},
        datasets=dict(dunhuang_cv=dict(excluded_gt_pair_ids=[], roles={
            'real_'+r: dict(pair_ids=[r]) for r in ('cal', 'select', 'test')})))
    return meta, plan


class FrozenRoles(unittest.TestCase):
    def test_only_cal_and_select(self):
        self.assertEqual([r['pair_id'] for r in development_members(*fixture())], ['cal', 'select'])

    def test_test_pair_relabelled_as_cal_is_rejected(self):
        meta, plan = fixture()
        plan['datasets']['dunhuang_cv']['roles']['real_cal']['pair_ids'].append('test')
        with self.assertRaisesRegex(ValueError, 'membership changed'):
            development_members(meta, plan)

    def test_source_group_leakage_rejected(self):
        meta, plan = fixture()
        meta['fragment_source_group']['selecta'] = 'cal'
        with self.assertRaisesRegex(ValueError, 'overlap by source'):
            development_members(meta, plan)

    def test_duplicate_pair_rejected(self):
        meta, plan = fixture()
        meta['pairs'].append(copy.deepcopy(meta['pairs'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate real pair'):
            development_members(meta, plan)

    def test_user_excluded_gt_not_reintroduced(self):
        meta, plan = fixture()
        plan['datasets']['dunhuang_cv']['excluded_gt_pair_ids'] = ['cal']
        plan['datasets']['dunhuang_cv']['roles']['real_cal']['pair_ids'] = []
        self.assertEqual([r['pair_id'] for r in development_members(meta, plan)], ['select'])


if __name__ == '__main__':
    unittest.main()
