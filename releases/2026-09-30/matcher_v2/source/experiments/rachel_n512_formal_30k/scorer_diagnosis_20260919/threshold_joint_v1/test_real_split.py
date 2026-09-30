import unittest
from .prepare_real_split import partition


class RealSplitTest(unittest.TestCase):
    def meta(self):
        rows=[];sources={}
        for fold in range(5):
            for y in (False,True):
                a=f'{fold}-{y}-a';b=f'{fold}-{y}-b'
                sources[a]=f'source-{fold}';sources[b]=f'source-{fold}'
                rows.append(dict(pair_id=f'{fold}-{y}',fragment_a_id=a,fragment_b_id=b,fold=fold,label=y))
        return dict(pairs=rows,fragment_source_group=sources)
    def test_source_disjoint_partition(self):
        roles=partition(self.meta(),set())
        self.assertEqual({k:v['pairs'] for k,v in roles.items()},dict(real_cal=2,real_select=6,real_test=2))
    def test_reject_same_source_in_test_and_selection(self):
        meta=self.meta();meta['fragment_source_group']['2-True-a']='source-0'
        with self.assertRaisesRegex(ValueError,'leakage'):partition(meta,set())
    def test_duplicate_or_unknown_exclusion_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing'):partition(self.meta(),{'absent'})
        meta=self.meta();meta['pairs'].append(meta['pairs'][0])
        with self.assertRaisesRegex(ValueError,'duplicate'):partition(meta,set())


if __name__=='__main__':unittest.main()
