import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mass_properties import OriginalMassModel


class MassTests(unittest.TestCase):
    def setUp(self):
        self.model=OriginalMassModel(Path(__file__).with_name('mass_test_fixture.urdf'))
        self.q=np.array([[0.],[.7],[-.9]])
        self.roots=np.tile(np.eye(4),(3,1,1))

    def test_exact_planar_com_and_fixed_mass(self):
        self.assertEqual(self.model.total_mass,6.)
        self.assertEqual(self.model.movable,['hinge'])
        com=self.model.properties(self.q,['hinge'],self.roots)
        np.testing.assert_allclose(com,np.c_[np.cos(self.q[:,0])/3,np.sin(self.q[:,0])/3,np.full(3,.5)],atol=1e-15)

    def test_parallel_axis_independent_particle_formula(self):
        c,I=self.model.properties(self.q,['hinge'],self.roots,inertia=True)
        for i,theta in enumerate(self.q[:,0]):
            points=np.array([[0,0,0],[np.cos(theta),np.sin(theta),0],[0,0,1]])
            manual=np.eye(3)*.03
            for m,p in zip([1,2,3],points):
                d=p-c[i];manual+=m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
            np.testing.assert_allclose(I[i],manual,atol=1e-15)

    def test_world_rotation_translation_equivariance(self):
        c,I=self.model.properties(self.q,['hinge'],self.roots,inertia=True)
        S=Rotation.from_rotvec([.3,.8,-.4]).as_matrix();p=np.array([3,2,-5])
        roots=self.roots.copy();roots[:,:3,:3]=S;roots[:,:3,3]=p
        cc,II=self.model.properties(self.q,['hinge'],roots,inertia=True)
        np.testing.assert_allclose(cc,c@S.T+p,atol=1e-14)
        np.testing.assert_allclose(II,S@I@S.T,atol=1e-14)

    def test_com_jacobian_finite_difference(self):
        h=1e-6
        plus=self.model.properties(self.q+h,['hinge'],self.roots)
        minus=self.model.properties(self.q-h,['hinge'],self.roots)
        expected=np.c_[-np.sin(self.q[:,0])/3,np.cos(self.q[:,0])/3,np.zeros(3)]
        np.testing.assert_allclose((plus-minus)/(2*h),expected,atol=1e-10)

    def test_inertia_physical(self):
        self.assertTrue(all(x['positive_inertia_and_triangle_inequality'] for x in self.model.physical_audit))

    def test_reject_missing_joint(self):
        with self.assertRaises(ValueError):self.model.properties(np.zeros((3,0)),[],self.roots)

    def test_reject_unknown_joint(self):
        with self.assertRaises(ValueError):self.model.properties(self.q,['other'],self.roots)

    def test_reject_bad_root(self):
        roots=self.roots.copy();roots[0,0,0]=2
        with self.assertRaises(ValueError):self.model.properties(self.q,['hinge'],roots)

    def test_reject_nonfinite(self):
        q=self.q.copy();q[1,0]=np.nan
        with self.assertRaises(ValueError):self.model.properties(q,['hinge'],self.roots)


if __name__=='__main__':unittest.main(verbosity=2)