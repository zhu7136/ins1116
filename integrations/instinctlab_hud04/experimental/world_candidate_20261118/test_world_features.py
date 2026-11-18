import sys,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from world_features import problem_data,perturb,check_roots,weights,TK

class WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m,cls.q,cls.roots,cls.desired,cls.correspondence=problem_data()

    def test_full_native_contract(self):
        self.assertEqual(self.q.shape,(997,29));self.assertEqual(self.desired.shape,(997,90))
        self.assertEqual(len(weights()),90);self.assertTrue(np.all(weights()>0))
        self.assertFalse(self.correspondence['reference_acceptance_authorized_by_this_mapping'])

    def test_initial_torso_not_copied_root(self):
        v=self.m.evaluate(self.q,self.roots)
        np.testing.assert_allclose(v[:,:3],self.desired[:,:3],atol=5e-16,rtol=0)
        np.testing.assert_allclose(v[:,21:30],self.desired[:,21:30],atol=1e-14,rtol=0)
        with np.load(TK.MOTION,allow_pickle=False) as z:self.assertGreater(np.max(abs(z['base_pos_w']-self.roots[:,:3,3])),.01)

    def test_35_columns_finite_difference(self):
        ids=np.array([0,257,495,996]);q=self.q[ids];r=self.roots[ids]
        _,j=self.m.evaluate(q,r,True);err=0.
        for col in range(35):
            d=np.zeros((4,6));qp=q.copy();qm=q.copy();h=1e-6
            if col<6:d[:,col]=h
            else:qp[:,col-6]+=h;qm[:,col-6]-=h
            numeric=(self.m.evaluate(qp,perturb(r,d))-self.m.evaluate(qm,perturb(r,-d)))/(2*h)
            err=max(err,float(np.max(abs(numeric-j[:,:,col]))))
        self.assertLess(err,2e-8)

    def test_independent_fk_world_features(self):
        ids=np.array([0,257,495,996]);q=self.q[ids];r=self.roots[ids]
        tree=TK.K.RobotTree.read(TK.TARGET);fk=tree.fk(q,self.m.names)
        world={l:r@t for l,t in fk.items()}
        p=np.stack([np.einsum('nij,j->ni',world[l][:,:3,:3],v)+world[l][:,:3,3] for _,l,v in self.m.positions],axis=1)
        b=np.stack([world[l][:,:3,:3] for l in self.m.orientation_links],axis=1)
        value=self.m.evaluate(q,r)
        np.testing.assert_allclose(value[:,:21],p.reshape(4,-1),rtol=0,atol=3e-15)
        np.testing.assert_allclose(value[:,21:66],b.reshape(4,-1),rtol=0,atol=3e-15)

    def test_world_equivariance(self):
        q=self.q[[0,257]];roots=self.roots[[0,257]];value=self.m.evaluate(q,roots)
        transform=np.eye(4);transform[:3,:3]=Rotation.from_rotvec([.3,-.5,.1]).as_matrix();transform[:3,3]=[1,-2,3]
        other=self.m.evaluate(q,transform@roots)
        p=value[:,:21].reshape(2,7,3)@transform[:3,:3].T+transform[:3,3]
        r=transform[:3,:3]@value[:,21:66].reshape(2,5,3,3)
        u=value[:,66:].reshape(2,8,3)@transform[:3,:3].T
        np.testing.assert_allclose(other,np.c_[p.reshape(2,-1),r.reshape(2,-1),u.reshape(2,-1)],atol=2e-15,rtol=0)

    def test_non_se3_and_missing_rejected(self):
        r=self.roots[:1].copy();r[0,0,0]+=1
        with self.assertRaises(ValueError):check_roots(r,1)
        with self.assertRaises(ValueError):self.m.evaluate(self.q[:1,:28],self.roots[:1])

    def test_endpoints_observable_not_rank16(self):
        _,j=self.m.evaluate(self.q[[0,257,495,996]],self.roots[[0,257,495,996]],True)
        self.assertEqual([int(np.linalg.matrix_rank(a,tol=1e-8)) for a in j],[35]*4)

if __name__=='__main__':unittest.main(verbosity=2)