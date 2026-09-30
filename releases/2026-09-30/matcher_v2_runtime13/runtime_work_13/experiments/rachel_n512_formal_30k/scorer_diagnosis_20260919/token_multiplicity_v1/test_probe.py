import unittest
import torch
from staging.pairwise_v0_2.models.rachel_decoupled_score import CrossAttentionPairHead
from .probe import copies, measure


class MultiplicityTests(unittest.TestCase):
    def test_counts_and_ids(self):
        a=torch.arange(12,dtype=torch.float32).reshape(4,3)
        member=torch.tensor([True,False,False,True])
        b,n=copies(a,member,'inliers4')
        self.assertEqual(n.tolist(),[4,1,1,4])
        self.assertEqual(len(b),10)
        self.assertTrue(torch.equal(b[0],a[0]))
        self.assertTrue(torch.equal(b[-1],a[-1]))

    def test_uniform_invariance_all_depths(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        a,b=torch.randn(5,8),torch.randn(7,8)
        ma=torch.tensor([True,True,False,False,False])
        mb=torch.tensor([False,True,True,False,False,False,False])
        for depth in (1,2,4):
            head=CrossAttentionPairHead(8,2,depth=depth).eval().requires_grad_(False)
            rows=measure(head,a,b,ma,mb)
            self.assertEqual(len(rows),6)
            for row in rows[1:4]:
                self.assertLess(abs(row['delta_logit']),2e-6)


if __name__=='__main__':
    unittest.main()
