import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from so3_c2 import SharedSO3Quintic, left_jacobian_and_directional


def vee_skew(M):
    return np.array([M[2,1]-M[1,2],M[0,2]-M[2,0],M[1,0]-M[0,1]])/2


def matrix_derivative_check(curve,i,u,h=1e-4):
    """Independent omega/alpha from rotation matrices ONLY, in physical time."""
    Rm2,Rm1,R0,Rp1,Rp2 = [curve.state(i,u+k*h/curve.dt)[0] for k in [-2,-1,0,1,2]]
    Rd = (Rm2-8*Rm1+8*Rp1-Rp2)/(12*h)
    Rdd = (-Rp2+16*Rp1-30*R0+16*Rm1-Rm2)/(12*h*h)
    return vee_skew(Rd@R0.T),vee_skew(Rdd@R0.T)


class SO3Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(118)
        self.R = Rotation.from_rotvec(rng.normal(size=(6,3))*.2).as_matrix()
        self.w,self.a = (rng.normal(size=(6,3)) for _ in range(2))
        self.curve = SharedSO3Quintic(self.R,self.w,self.a,.2)

    def test_both_endpoint_orientations_and_world_rates(self):
        for i in range(5):
            for u,j in [(0,i),(1,i+1)]:
                r,w,a = self.curve.state(i,u)
                np.testing.assert_allclose(r,self.R[j],atol=2e-13)
                np.testing.assert_allclose(w,self.w[j],atol=2e-12)
                np.testing.assert_allclose(a,self.a[j],atol=3e-11)

    def test_every_join_is_world_angular_c2(self):
        for i in range(4):
            for left,right in zip(self.curve.state(i,1),self.curve.state(i+1,0)):
                np.testing.assert_allclose(left,right,atol=3e-11)

    def test_interior_angular_rates_from_independent_matrix_differences(self):
        for i in range(5):
            for u in [.2,.5,.8]:
                _,w,a = self.curve.state(i,u)
                wn,an = matrix_derivative_check(self.curve,i,u)
                np.testing.assert_allclose(w,wn,atol=2e-8)
                np.testing.assert_allclose(a,an,atol=2e-6)

    def test_jacobian_directional_finite_difference(self):
        rng = np.random.default_rng(1118)
        for mag in [0,1e-8,1e-3,.009,.011,.2,1.,2.9]:
            p = rng.normal(size=3); p*=mag/(np.linalg.norm(p)+1e-30)
            d = rng.normal(size=3)
            _,dj = left_jacobian_and_directional(p,d)
            h=1e-6
            jp = left_jacobian_and_directional(p+h*d,d)[0]
            jm = left_jacobian_and_directional(p-h*d,d)[0]
            np.testing.assert_allclose(dj,(jp-jm)/(2*h),atol=2e-8)

    def test_left_jacobian_from_matrix_exp_differential(self):
        p = np.array([.9,-.7,.3]); d=np.array([-.2,.8,.4]); h=1e-6
        R0 = Rotation.from_rotvec(p).as_matrix()
        rd = (Rotation.from_rotvec(p+h*d).as_matrix()-Rotation.from_rotvec(p-h*d).as_matrix())/(2*h)
        J,_ = left_jacobian_and_directional(p,d)
        np.testing.assert_allclose(J@d,vee_skew(rd@R0.T),atol=2e-10)

    def test_zero_jacobian(self):
        J,dJ = left_jacobian_and_directional(np.zeros(3),np.zeros(3))
        np.testing.assert_array_equal(J,np.eye(3))
        np.testing.assert_array_equal(dJ,np.zeros((3,3)))

    def test_world_frame_equivariance(self):
        S=Rotation.from_rotvec([.7,-.9,.4]).as_matrix()
        other=SharedSO3Quintic(S@self.R,self.w@S.T,self.a@S.T,.2)
        for u in [.1,.4,.9]:
            r,w,a=self.curve.state(2,u); rr,ww,aa=other.state(2,u)
            np.testing.assert_allclose(rr,S@r,atol=2e-13)
            np.testing.assert_allclose(ww,S@w,atol=2e-12)
            np.testing.assert_allclose(aa,S@a,atol=2e-11)

    def test_time_scaling(self):
        other=SharedSO3Quintic(self.R,self.w/2,self.a/4,.4)
        r,w,a=self.curve.state(1,.3);rr,ww,aa=other.state(1,.3)
        np.testing.assert_allclose(rr,r,atol=1e-13)
        np.testing.assert_allclose(ww,w/2,atol=1e-12)
        np.testing.assert_allclose(aa,a/4,atol=1e-11)

    def test_constant_axis_constant_world_rate(self):
        dt=.1;t=np.arange(5)*dt;axis=np.array([0.,0.,1.])
        R=Rotation.from_rotvec(t[:,None]*axis).as_matrix()
        curve=SharedSO3Quintic(R,np.tile(axis,(5,1)),np.zeros((5,3)),dt)
        for i in range(4):
            r,w,a=curve.state(i,.37)
            np.testing.assert_allclose(r,Rotation.from_rotvec(axis*(t[i]+.37*dt)).as_matrix(),atol=1e-13)
            np.testing.assert_allclose(w,axis,atol=1e-12)
            np.testing.assert_allclose(a,0,atol=1e-10)

    def test_rotations_remain_orthonormal(self):
        for u in np.linspace(0,1,21):
            r,_,_=self.curve.state(1,float(u))
            np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(r),1.,places=13)

    def test_stationary(self):
        R=np.tile(np.eye(3),(3,1,1));z=np.zeros((3,3))
        curve=SharedSO3Quintic(R,z,z,.1)
        r,w,a=curve.state(1,.3)
        np.testing.assert_array_equal(r,np.eye(3))
        np.testing.assert_array_equal(w,np.zeros(3));np.testing.assert_array_equal(a,np.zeros(3))

    def test_near_pi_rejected(self):
        R=np.array([np.eye(3),Rotation.from_rotvec([np.pi,0,0]).as_matrix()])
        with self.assertRaises(ValueError): SharedSO3Quintic(R,np.zeros((2,3)),np.zeros((2,3)),.1)

    def test_silent_rotation_projection_rejected(self):
        R=self.R.copy();R[0,0,0]+=.1
        with self.assertRaises(ValueError): SharedSO3Quintic(R,self.w,self.a,.2)

    def test_nonfinite_rejected(self):
        w=self.w.copy();w[1,0]=np.nan
        with self.assertRaises(ValueError): SharedSO3Quintic(self.R,w,self.a,.2)

    def test_bad_time_and_shape_rejected(self):
        for dt in [0,-1,float('inf')]:
            with self.assertRaises(ValueError): SharedSO3Quintic(self.R,self.w,self.a,dt)
        with self.assertRaises(ValueError): SharedSO3Quintic(self.R,self.w[:-1],self.a,.2)

    def test_chart_interior_excursion_rejected(self):
        R=np.tile(np.eye(3),(2,1,1));w=np.array([[100.,0,0],[-100.,0,0]])
        curve=SharedSO3Quintic(R,w,np.zeros((2,3)),1.)
        with self.assertRaises(ValueError):curve.state(0,.5)


if __name__=='__main__':unittest.main(verbosity=2)