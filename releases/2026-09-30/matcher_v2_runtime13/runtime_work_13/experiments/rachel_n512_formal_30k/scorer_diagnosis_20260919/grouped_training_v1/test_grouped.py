import unittest
import torch
from support import group_loss,ARMS

class LossTest(unittest.TestCase):
    def test_bce_weight_balance(self):
        z=torch.zeros(8);y=torch.tensor([1.,0,0,0,0,1,0,0]);v=torch.ones(8,dtype=torch.bool)
        loss,bce,rank=group_loss(z,y,v,ARMS[0])
        self.assertAlmostEqual(loss.item(),float(torch.log(torch.tensor(2.))),places=6)
        self.assertAlmostEqual(rank.item(),float(torch.log(torch.tensor(4.))),places=6)
    def test_positive_negative_gradients(self):
        z=torch.zeros(4,requires_grad=True);y=torch.tensor([0.,1,0,0]);v=torch.ones(4,dtype=torch.bool)
        loss,_,_=group_loss(z,y,v,ARMS[1]);loss.backward()
        self.assertLess(z.grad[1],0);self.assertTrue(torch.all(z.grad[[0,2,3]]>0))
    def test_group_permutation(self):
        z=torch.tensor([2.,-.5,.1,-3.]);y=torch.tensor([1.,0,0,0]);v=torch.ones(4,dtype=torch.bool)
        p=torch.tensor([3,2,0,1])
        self.assertAlmostEqual(group_loss(z,y,v,ARMS[1])[0].item(),group_loss(z[p],y[p],v[p],ARMS[1])[0].item(),places=6)
    def test_invalid_group_has_no_rank_gradient(self):
        z=torch.zeros(4,requires_grad=True);y=torch.tensor([1.,0,0,0]);v=torch.tensor([True,True,False,True])
        rank=group_loss(z,y,v,ARMS[1])[2];rank.backward()
        self.assertEqual(rank.item(),0);self.assertTrue(torch.all(z.grad==0))

if __name__=='__main__':unittest.main()
