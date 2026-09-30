import unittest
import numpy as np
from .source_search import SourceSearch


def entry(pid, stratum='gen5_partition_positive'):
    return dict(pair_id=pid, source_stratum=stratum)


class SourceSearchTests(unittest.TestCase):
    def make(self, n=20, budget=2, enabled=True):
        pool=[entry(str(i)) for i in range(n)]
        return SourceSearch(pool[0], pool, np.random.default_rng(51), budget, enabled)

    def test_one_feasible_source_is_not_missed_by_small_damage_budget(self):
        search=self.make()
        seen=[]
        while True:
            item=search.draw()
            self.assertIsNotNone(item)
            _,source=item
            if source['pair_id']=='19':
                break
            seen.append(source['pair_id'])
            search.rejected(source, static_length=True)
        self.assertEqual(len(seen),len(set(seen)))
        self.assertEqual(search.damage_attempts,0)
        self.assertLessEqual(search.draws,20)

    def test_all_sources_infeasible_stops_after_unique_pool(self):
        search=self.make()
        while (item:=search.draw()) is not None:
            search.rejected(item[1],static_length=True)
        self.assertEqual(search.draws,20)
        self.assertEqual(search.damage_attempts,0)

    def test_damage_failures_keep_original_budget(self):
        search=self.make()
        for _ in range(2):
            _,source=search.draw();search.rejected(source)
        self.assertIsNone(search.draw())
        self.assertEqual(search.draws,2)

    def test_disabled_preserves_legacy_sampling_and_budget(self):
        search=self.make(budget=10,enabled=False)
        rng=np.random.default_rng(51)
        expected=['0']*4+[str(rng.integers(20)) for _ in range(6)]
        for value in expected:
            _,source=search.draw();self.assertEqual(source['pair_id'],value)
            search.rejected(source,static_length=True)
        self.assertIsNone(search.draw())
        self.assertEqual(search.excluded,set())

    def test_pool_multiplicity_retained_for_unexcluded_sources(self):
        search=SourceSearch(entry('x'),[entry('x'),entry('x'),entry('y'),entry('y'),entry('z')],
                            np.random.default_rng(1),4,True)
        _,source=search.draw();search.rejected(source,static_length=True)
        self.assertEqual([e['pair_id'] for e in search.pool],['y','y','z'])

    def test_cannot_change_structural_quota(self):
        with self.assertRaisesRegex(ValueError,'stratum'):
            SourceSearch(entry('x'),[entry('y','native_positive')],np.random.default_rng(1),4,True)


if __name__=='__main__':unittest.main()
