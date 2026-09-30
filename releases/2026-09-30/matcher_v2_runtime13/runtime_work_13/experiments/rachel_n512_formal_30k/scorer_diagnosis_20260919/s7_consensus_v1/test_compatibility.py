import unittest
import torch

from .compatibility import CompatibilityConfig, damage_compatibility


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        # Test constants are analytic fixtures, not deployed calibration.
        self.config = CompatibilityConfig(.5, .5, .5, .5, 1., 30.)

    def get(self, values, normal_a=None, normal_b=None):
        r = torch.as_tensor(values, dtype=torch.float64)
        a = torch.tensor([0.,1.]).expand_as(r) if normal_a is None else normal_a
        b = -a if normal_b is None else normal_b
        one = torch.ones(r.shape[:-1])
        return damage_compatibility(r,a,b,one,one,one*2,one*2,self.config)

    def test_gap_not_tangential_slide_or_penetration(self):
        out = self.get([[0,20],[20,0],[0,-20],[0,40]])
        self.assertEqual(float(out.kernel[0]), 1.)
        self.assertLess(float(out.localization_kernel[0]), 1e-20)
        self.assertTrue((out.kernel[1:] < 1e-20).all())
        torch.testing.assert_close(out.material_offset_rc[2], torch.zeros(2,dtype=torch.float64))

    def test_finite_bound(self):
        out = self.get([[0,30],[0,31],[0,32]])
        torch.testing.assert_close(out.kernel, torch.tensor([1.,.6065306597,.1353352832],dtype=torch.float64))
        self.assertLessEqual(float(out.material_offset_rc.norm(dim=-1).max()),30.)

    def test_unreliable_has_no_arbitrary_signed_compensation(self):
        out = self.get([[0,20]], normal_a=torch.tensor([[1.,0.]]), normal_b=torch.tensor([[1.,0.]]))
        self.assertFalse(out.reliable_normal.any())
        self.assertLess(float(out.kernel[0]),1e-20)
        self.assertEqual(float(out.material_offset_rc.abs().sum()),0.)

    def test_swap_invariance(self):
        r = torch.tensor([[1.,15.]],dtype=torch.float64)
        a = self.get(r)
        b = self.get(-r, normal_a=torch.tensor([[0.,-1.]]), normal_b=torch.tensor([[0.,1.]]))
        torch.testing.assert_close(a.kernel,b.kernel)
        torch.testing.assert_close(a.material_offset_rc,-b.material_offset_rc)

    def test_broadcast_and_pose_gradient(self):
        pose = torch.tensor([0.,0.], requires_grad=True)
        out = self.get(torch.tensor([[1.,10.],[3.,8.]],requires_grad=True)-pose)
        out.kernel.sum().backward()
        self.assertTrue(torch.isfinite(pose.grad).all())
        self.assertGreater(float(pose.grad.abs().sum()),0.)

    def test_reject_invalid_config(self):
        with self.assertRaises(ValueError):
            CompatibilityConfig(.5,.5,.5,.5,1.,30.,damage_normal_lower_px=-20)
        with self.assertRaises(ValueError):
            CompatibilityConfig.from_calibration({'status':'running'})


if __name__ == '__main__':
    unittest.main()
