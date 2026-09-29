import unittest
import numpy as np
from modes import weighted_modes, unique_modes


class ModeTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(weighted_modes([], [], 10), [])

    def test_weighted_center(self):
        m=weighted_modes([[0,0],[4,0]],[1,3],10)
        self.assertEqual(len(m),1)
        np.testing.assert_allclose(m[0]['center'],[3,0])

    def test_all_observations_initialized_not_topq(self):
        m=weighted_modes([[0,0],[60,0]],[1,.0001],10)
        self.assertEqual(len(m),2)

    def test_translation_equivariance(self):
        x=np.array([[0,0],[2,0],[60,0]],float)
        a=weighted_modes(x,[1,2,1],10); b=weighted_modes(x+[431,-183],[1,2,1],10)
        for m,n in zip(a,b): np.testing.assert_allclose(n['center']-m['center'],[431,-183])

    def test_rotation_equivariance(self):
        x=np.array([[0,0],[2,0],[60,0]],float)
        rot=np.array([[0,-1],[1,0]])
        a=weighted_modes(x,[1,2,1],10); b=weighted_modes(x@rot,[1,2,1],10)
        for m,n in zip(a,b): np.testing.assert_allclose(m['center']@rot,n['center'])

    def test_no_drift_chain_linkage(self):
        x=[[t,0] for t in [0,20,40,60,80]]
        self.assertEqual(len(unique_modes(weighted_modes(x,np.ones(5),16),16)),5)

    def test_members_radial(self):
        x=np.array([[t,0] for t in [0,4,10,15,200]])
        for m in weighted_modes(x,np.ones(5),10):
            expected=np.where(np.linalg.norm(x-m['center'],axis=1)<=10)[0]
            np.testing.assert_equal(expected,m['members'])

    def test_permutation(self):
        x=np.array([[0,0],[2,0],[60,0]],float); w=np.array([1,2,1])
        a=weighted_modes(x,w,10); b=weighted_modes(x[::-1],w[::-1],10)
        for m,n in zip(a,b): np.testing.assert_allclose(m['center'],n['center'])

    def test_invalid_not_silently_filtered(self):
        with self.assertRaises(ValueError): weighted_modes([[float('nan'),0]],[1],10)
        with self.assertRaises(ValueError): weighted_modes([[0,0]],[0],10)

if __name__=='__main__': unittest.main()
