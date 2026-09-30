import unittest
import numpy as np
from .data import select_effective_groups,mirror_schedule


class HardSelection(unittest.TestCase):
    def group(self,key,recipe,changed=(True,True)):
        return [dict(s7_group_index=key,s7_recipe=recipe,changed_pair=flag,label=label) for label,flag in zip((1,0),changed)]

    def test_remove_uncoupled_fallback_and_clean(self):
        entries=self.group(0,'local')+self.group(1,'seam_gaps',(True,False))+self.group(2,'reference_e1')+self.group(3,'partial_curve')
        selected=select_effective_groups(entries)
        self.assertEqual(len(selected),4)
        self.assertEqual(sum(e['label'] for e in selected),2)
        self.assertEqual({e['s7_group_index'] for e in selected},{0,3})

    def test_missing_negative_is_not_accepted(self):
        with self.assertRaises(ValueError):select_effective_groups(self.group(0,'local')[:1])

    def test_mirrors_balanced_by_group(self):
        groups=np.repeat(np.arange(3916),2)
        codes=mirror_schedule(3916,.3,260923,1)[groups]
        self.assertTrue(np.array_equal(codes[::2],codes[1::2]))
        self.assertEqual(int((codes==1).sum()),1174)
        self.assertEqual(int((codes==2).sum()),1174)
        self.assertTrue(np.array_equal(codes,mirror_schedule(3916,.3,260923,1)[groups]))


if __name__=='__main__':unittest.main()
