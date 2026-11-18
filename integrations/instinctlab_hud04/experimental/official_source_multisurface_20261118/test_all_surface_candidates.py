import unittest
import numpy as np
from all_surface_candidates import enumerate_near,component_speeds,segment_triangles

def plane(z):return np.array([[[-2.,-2,z],[2,-2,z],[0,2,z]]])
class TestAllCandidates(unittest.TestCase):
    def test_second_surface_survives_deeper_nearest(self):
        t=np.concatenate([plane(0),plane(.095)])
        ids,m,_,_=enumerate_near([0,0,.1],[0,0,.1],t,.1,.02)
        self.assertEqual(ids.tolist(),[0,1]);self.assertEqual((abs(m)<=.02).tolist(),[True,False])
        self.assertFalse(abs(m.min())<=.02)
    def test_far_triangle_not_selected(self):
        self.assertEqual(len(enumerate_near([0,0,2],[0,0,2],plane(0),.1,.02)[0]),0)
    def test_corner_simultaneous_faces(self):
        t=np.concatenate([plane(0),plane(0)[:,:,[2,1,0]]]);ids,*_=enumerate_near([.1,0,.1],[.1,0,.1],t,.1,.02)
        self.assertEqual(len(ids),2)
    def test_complete_broadphase_against_exhaustive(self):
        rng=np.random.default_rng(11)
        for _ in range(12):
            t=rng.normal(size=(80,3,3));a,b=rng.normal(size=(2,3));d,*_=segment_triangles(a,b,t)
            ids,m,*_=enumerate_near(a,b,t,.2,.02)
            np.testing.assert_array_equal(ids,np.flatnonzero(d<=.22+1e-12));np.testing.assert_allclose(m,d[ids]-.2,atol=1e-12)
    def test_empty(self):self.assertEqual(len(enumerate_near([0,0,0],[0,0,0],np.empty((0,3,3)),.1,.02)[0]),0)
    def test_invalid_rejected(self):
        for radius,band in [(0,.02),(.1,-.01),(float('nan'),.02)]:
            with self.assertRaises(ValueError):enumerate_near([0,0,0],[0,0,0],plane(0),radius,band)
    def test_axis_coincidence_undefined(self):
        poses=np.tile(np.eye(4),(3,1,1));ps=np.zeros((1,3));d,*_=component_speeds(poses,1,ps,ps,.1,np.array([[0,0,1]]),90)
        self.assertFalse(d[0])
    def test_material_reversal_not_cancelled(self):
        poses=np.tile(np.eye(4),(3,1,1));poses[1,0,3]=.01
        d,local,vt,vn=component_speeds(poses,1,np.array([[.01,0,.1]]),np.array([[.01,0,0]]),.1,np.array([[0,0,1]]),90)
        self.assertTrue(d[0]);self.assertAlmostEqual(vt[0],.9);self.assertAlmostEqual(vn[0],0)
    def test_fixed_material_rotation(self):
        poses=np.tile(np.eye(4),(3,1,1));theta=.1;poses[2,:3,:3]=[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
        d,local,vt,vn=component_speeds(poses,1,np.array([[1,0,.1]]),np.array([[1,0,0]]),.1,np.array([[0,0,1]]),90)
        self.assertAlmostEqual(vt[0],180*np.sin(.05));np.testing.assert_allclose(local,[[1,0,0]],atol=1e-14,rtol=0)
    def test_endpoint_one_sided(self):
        poses=np.tile(np.eye(4),(2,1,1));poses[1,2,3]=.01
        _,_,vt,vn=component_speeds(poses,0,np.array([[0,0,.1]]),np.array([[0,0,0]]),.1,np.array([[0,0,1]]),90)
        self.assertAlmostEqual(vn[0],.9);self.assertAlmostEqual(vt[0],0)
if __name__=='__main__':unittest.main()