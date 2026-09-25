import unittest

import torch
from torch.nn import functional as F

from .losses import supervised_mass_mean


class SupervisionNumericsTests(unittest.TestCase):
    def test_regular_weighted_mean_output_and_gradient_parity(self):
        values=torch.tensor([.2,1.3,4.],requires_grad=True)
        weight=torch.tensor([.03,.1,.7])
        old=(values*weight).sum()/weight.sum()
        new=supervised_mass_mean(values,weight)
        torch.testing.assert_close(new,old)
        torch.testing.assert_close(torch.autograd.grad(new,values)[0],
                                   torch.autograd.grad(old,values)[0])

    def test_subnormal_sparse_bce_keeps_finite_analytic_gradient(self):
        logits=torch.tensor([-2.,.4,1.],requires_grad=True)
        weight=torch.tensor([0.,1e-42,0.])
        loss=F.binary_cross_entropy_with_logits(logits,torch.zeros(3),reduction='none')
        result=supervised_mass_mean(loss,weight)
        result.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(result,loss[1])
        torch.testing.assert_close(logits.grad,torch.tensor([0.,float(logits[1].sigmoid()),0.]))

    def test_subnormal_local_ce_retains_weight_ratios(self):
        logits=torch.tensor([[1.,2.,3.],[2.,1.,0.]],requires_grad=True)
        weight=torch.tensor([1e-42,3e-42])
        loss=F.cross_entropy(logits,torch.tensor([0,2]),reduction='none')
        result=supervised_mass_mean(loss,weight)
        reference=(loss.double()*weight.double()).sum()/weight.double().sum()
        torch.testing.assert_close(result.double(),reference,rtol=1e-6,atol=1e-7)
        result.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_zero_mass_produces_zero_finite_loss_and_gradient(self):
        values=torch.tensor([1.,2.,3.],requires_grad=True)
        result=supervised_mass_mean(values,torch.zeros(3))
        result.backward()
        self.assertEqual(float(result),0.)
        torch.testing.assert_close(values.grad,torch.zeros(3))


if __name__=='__main__':
    unittest.main()
