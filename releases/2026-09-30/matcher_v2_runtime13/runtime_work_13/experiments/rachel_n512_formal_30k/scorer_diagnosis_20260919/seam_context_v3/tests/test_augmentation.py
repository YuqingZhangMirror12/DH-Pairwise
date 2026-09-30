import unittest
from dataclasses import replace
import numpy as np
import torch
from ..augmentation import paired_mirror,mirror_schedule
from ..valid_contour import compact,remap_target
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample


def sample():
    p=np.array([[2,3],[3,6],[5,4],[np.nan,np.nan]],np.float32)
    t=np.array([3,4],np.float32);q=p+t
    valid=np.array([1,1,1,0],bool);target=np.array([0,1,2,-2],np.int64)
    ma=np.zeros((1,20,30),np.float32);mb=np.zeros((1,24,34),np.float32)
    for mask,points in ((ma,p),(mb,q)):
        xy=points[valid].astype(int);mask[0,xy[:,0],xy[:,1]]=1
    return RachelPairSample('fixture','a','b',ma,mb,np.zeros((1,8,8),np.float32),
        np.zeros((1,8,8),np.float32),p,q,valid,valid.copy(),target,target.copy(),np.float32(1),
        t,np.array([4,-3],np.float32),np.bool_(True))


class Tests(unittest.TestCase):
    def test_paired_mirror_coordinates_gt_and_identity(self):
        source=sample()
        for axis,coord in (('horizontal',1),('vertical',0)):
            out=paired_mirror(source,axis)
            np.testing.assert_array_equal(out.target_a,source.target_a)
            np.testing.assert_array_equal(out.target_b,source.target_b)
            ids=np.flatnonzero(out.target_a>=0)
            delta=out.points_rc_b[out.target_a[ids]]-out.points_rc_a[ids]
            np.testing.assert_allclose(delta,np.tile(out.translation_a_to_b_rc,(len(ids),1)))
            np.testing.assert_allclose(out.translation_a_to_b_xy_cartesian,
                [out.translation_a_to_b_rc[1],-out.translation_a_to_b_rc[0]])
            for side in 'ab':
                valid=getattr(out,'contour_valid_'+side)
                points=getattr(out,'points_rc_'+side)[valid].astype(int)
                self.assertTrue((getattr(out,'mask_'+side)[0,points[:,0],points[:,1]]==1).all())
            restored=paired_mirror(out,axis)
            for name in source.__dataclass_fields__:
                before,after=getattr(source,name),getattr(restored,name)
                if isinstance(before,np.ndarray):np.testing.assert_allclose(before,after,equal_nan=True)
                else:self.assertEqual(before,after)

    def test_mirror_canonical_reindex_keeps_correspondence(self):
        for axis in ('horizontal','vertical'):
            out=paired_mirror(sample(),axis)
            ga=compact(torch.tensor(out.points_rc_a)[None],torch.tensor(out.contour_valid_a)[None])
            gb=compact(torch.tensor(out.points_rc_b)[None],torch.tensor(out.contour_valid_b)[None])
            target=remap_target(torch.tensor(out.target_a)[None],ga,gb)[0]
            ids=torch.nonzero(target>=0).flatten()
            delta=gb.points[0,target[ids]]-ga.points[0,ids]
            torch.testing.assert_close(delta,torch.tensor(out.translation_a_to_b_rc).expand(len(ids),-1))

    def test_negative_label_and_deterministic_schedule(self):
        s=replace(sample(),label=np.float32(0),translation_valid=np.bool_(False),
            target_a=np.full(4,-1,np.int64),target_b=np.full(4,-1,np.int64),
            translation_a_to_b_rc=np.zeros(2,np.float32),translation_a_to_b_xy_cartesian=np.zeros(2,np.float32))
        for axis in ('horizontal','vertical'):
            out=paired_mirror(s,axis)
            self.assertEqual(out.label,0);self.assertFalse(out.translation_valid)
            np.testing.assert_array_equal(out.translation_a_to_b_rc,np.zeros(2))
            np.testing.assert_array_equal(out.target_a,s.target_a)
        schedule=mirror_schedule(24000,.1,260921,1)
        np.testing.assert_array_equal(np.bincount(schedule),[21600,1200,1200])
        np.testing.assert_array_equal(schedule,mirror_schedule(24000,.1,260921,1))
        self.assertFalse(np.array_equal(schedule,mirror_schedule(24000,.1,260921,2)))
        self.assertFalse(mirror_schedule(24000,0.,260921,1).any())


if __name__=='__main__':unittest.main()
