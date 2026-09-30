import unittest

import numpy as np
import torch

from . import geometry as g


class Sampler(torch.nn.Module):
    """Test double retaining the production halfspan construction, no project imports."""
    def __init__(self):
        super().__init__()
        self.window_sizes_px = (7.,16.,32.,64.)
        self.patch_size = 16
        self.canvas_size = 800
        grids = []
        for w in self.window_sizes_px:
            axis = torch.linspace(-(w-1.)/2., (w-1.)/2., 16)
            rr,cc = torch.meshgrid(axis,axis,indexing="ij")
            grids.append(torch.stack((rr,cc),dim=-1))
        self.register_buffer("offsets_rc",torch.stack(grids))


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self):
        mask = torch.zeros(1,1,800,800)
        mask[:,:,10:391,80:690] = 1
        points = torch.tensor([[[10.,80.],[390.,689.],[200.,400.],[-50.,-100.]]])
        valid = torch.tensor([[True,True,True,False]])
        return mask,points,valid

    def test_identity_exact_no_alias_or_mutation(self):
        mask,points,valid = self.inputs()
        result = g.downscale_fragment(mask,points,valid,1.)
        for original,after in zip((mask,points,valid),(result.mask,result.points_rc,result.valid)):
            self.assertTrue(torch.equal(original,after))
            self.assertNotEqual(original.data_ptr(),after.data_ptr())
        result.mask.zero_();result.points_rc.zero_();result.valid.zero_()
        self.assertGreater(mask.sum(),0)
        self.assertEqual(points[0,0,0],10.)
        self.assertEqual(valid.sum(),3)

    def test_pair_common_center_retains_order_count_and_padding(self):
        mask,points,valid = self.inputs()
        shifted = points.clone();shifted[valid] += torch.tensor([20.,30.])
        for scale in (.5,.75):
            pair = g.downscale_pair(mask,mask,points,shifted,valid,valid,scale)
            expected = 399.5+scale*(points[valid]-399.5)
            self.assertTrue(torch.equal(pair.a.points_rc[valid],expected))
            self.assertTrue(torch.equal(pair.b.points_rc[valid]-pair.a.points_rc[valid],
                torch.tensor([20.,30.]).expand(3,2)*scale))
            self.assertTrue(torch.equal(pair.a.points_rc[~valid],points[~valid]))
            self.assertTrue(torch.equal(pair.a.valid,valid))
            self.assertEqual(pair.a.points_rc.shape,points.shape)
            self.assertEqual(len(pair.inputs),6)
            self.assertFalse(pair.a.diagnostics["point_count_changed"])
            self.assertFalse(pair.a.diagnostics["contour_reextracted"])

    def test_nearest_binary_and_outside_support_zero(self):
        points = torch.tensor([[[0.,0.],[799.,799.]]])
        valid = torch.ones(1,2,dtype=torch.bool)
        for scale in (.5,.75):
            mask = torch.ones(1,1,800,800,dtype=torch.bool)
            result = g.downscale_fragment(mask,points,valid,scale)
            self.assertEqual(result.mask.dtype,torch.bool)
            self.assertTrue(torch.all((result.mask==0)|(result.mask==1)))
            inverse=399.5+(torch.arange(800)-399.5)/scale
            outside=(inverse < -.5)|(inverse > 799.5)
            self.assertFalse(result.mask[0,0,outside,:].any())
            self.assertFalse(result.mask[0,0,:,outside].any())
            self.assertEqual(result.diagnostics["pixel_count_before"],[640000])
            self.assertEqual(result.diagnostics["pixel_count_after"],[int(result.mask.sum())])

    def test_exact_span_compensation_and_source_sampler_untouched(self):
        sampler = Sampler().eval()
        original = sampler.offsets_rc.clone()
        sizes = sampler.window_sizes_px
        for scale in (1.,.5,.75):
            clone = g.cloned_sampler(sampler,scale)
            self.assertFalse(clone.training)
            self.assertEqual(clone.patch_size,16)
            self.assertEqual(clone.canvas_size,800)
            self.assertTrue(torch.equal(clone.offsets_rc,original*scale))
            for old,new in zip(sizes,clone.window_sizes_px):
                self.assertEqual((new-1)/2.,scale*(old-1)/2.)
            p=torch.tensor([200.,500.]);c=399.5
            torch.testing.assert_close(c+scale*(p-c)+clone.offsets_rc,
                c+scale*(p+original-c),rtol=0,atol=4e-5)
            clone.offsets_rc.zero_()
            self.assertTrue(torch.equal(sampler.offsets_rc,original))
            self.assertEqual(sampler.window_sizes_px,sizes)
        self.assertEqual(g.corrected_window_sizes((7,16,32,64),.5),(4.,8.5,16.5,32.5))

    def test_numpy_inputs_and_empty_mask_diagnostic(self):
        result=g.downscale_fragment(np.zeros((1,1,800,800),dtype=np.uint8),
            np.zeros((1,3,2),dtype=np.float32),np.zeros((1,3),dtype=bool),np.float64(.75))
        self.assertEqual(result.mask.dtype,torch.uint8)
        self.assertEqual(result.diagnostics["observed_pixel_count_ratio"],[None])
        self.assertTrue(torch.equal(result.points_rc,torch.zeros(1,3,2)))

    def test_invalid_scale_and_nonbinary_input_rejected(self):
        inputs=self.inputs()
        for scale in (0.,-1.,1.001,float("nan"),float("inf"),True,"0.5"):
            with self.subTest(scale=scale),self.assertRaises(ValueError):
                g.downscale_fragment(*inputs,scale)
            with self.assertRaises(ValueError):g.corrected_window_sizes((7,16),scale)
        mask,points,valid=inputs
        mask[0,0,5,5]=.1
        with self.assertRaisesRegex(ValueError,"binary"):
            g.downscale_fragment(mask,points,valid,.5)


if __name__=="__main__":
    unittest.main()
