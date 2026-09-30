import unittest
from collections import Counter

from .heldout_v14 import split_sources, schedule, cross_negatives
from ..seam_context_v3.prepare import sources


class HeldoutPlanTests(unittest.TestCase):
    def test_test_never_enters_cal_select_and_alias_family_cannot_cross(self):
        result=split_sources({'a','b','c','d','seen'}, {'a','e','f','seen'}, {'seen'}, {'a':8,'b':7}, 31)
        self.assertEqual(result['test'], {'e','f'})
        self.assertEqual(result['cal']|result['select'], {'a','b','c','d'})
        self.assertFalse(result['cal']&result['select'])
        self.assertFalse(set.union(*result.values())&{'seen'})

    def test_noninteger_subtype_counts_are_explicit_without_extra_samples(self):
        for n in (750,1500):
            value=schedule(n,{},27)
            self.assertEqual(len(value['recipes']), n)
            self.assertEqual(len(value['mirrors']), n)
            self.assertEqual(sum(value['integer_macro_counts'].values()), n)
            self.assertLessEqual(abs(sum(value['partial'])-.25*n),1)
            self.assertLessEqual(abs(sum(m!=0 for m in value['mirrors'])-.15*n),1)
            counts=Counter(value['partial_modes'])
            self.assertLessEqual(abs(counts['end']-counts['middle']),1)
            for r,flag,mode in zip(value['recipes'],value['partial'],value['partial_modes']):
                self.assertEqual(flag,r=='partial')
                self.assertEqual(mode is not None,flag)
            self.assertEqual(value,schedule(n,{},27))

    def test_cross_pairs_unique_canonical_family_and_original_scale_matched(self):
        fragments={}
        for name in ('a_recto','a_verso','b','c','train'):
            for i in range(4):
                token=f'{name}/{i}'
                fragments[token]=dict(fragment_token=token,split_unit_id=name+'.png',
                    foreground_area=100+i*10,bbox_aspect_ratio=1+i*.1)
        result=cross_negatives(fragments,{'a','b','c'},30,84,'val')
        keys={tuple(sorted(r['fragment_'+s]['fragment_token'] for s in 'ab')) for r in result}
        self.assertEqual(len(keys),30)
        self.assertTrue(all(len(sources(r))==2 and sources(r)<={'a','b','c'} for r in result))
        self.assertTrue(all(max(r['scale_match'].values())<=2 for r in result))
        self.assertTrue(all(r['label'] is False and r['correspondence_path'] is None for r in result))

    def test_cross_pool_failure_not_duplicate_padding(self):
        pool={'a':dict(fragment_token='a',split_unit_id='a.png',foreground_area=10,bbox_aspect_ratio=1),
              'b':dict(fragment_token='b',split_unit_id='b.png',foreground_area=100,bbox_aspect_ratio=1)}
        with self.assertRaises(ValueError):
            cross_negatives(pool,{'a','b'},1,6,'val')


if __name__=='__main__':
    unittest.main()
