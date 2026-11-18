import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from wholebody_terms import WholeBodyTerms,supports,pair_projection,triangle_projection
from robot_shape_inventory import Shape,load_shapes
from world_features import WorldFeatures,perturb,TK
PRIOR=Path('/workspace/hud04-world-candidate-20261118/WORLD_TRIAL_005_FOOT_PRIORITY/STEP1_WORLD_CANDIDATE_UNACCEPTED.npz')
def box():return Shape('box','link','box',np.eye(4),{'size':[1.,1.,1.]})
def pose(p):
    t=np.eye(4);t[:3,3]=p;return t
class GeometryTests(unittest.TestCase):
    def test_boxes_separate(self):
        g,n,a,b=pair_projection(box(),pose([1.2,0,0]),box(),np.eye(4));self.assertAlmostEqual(g,.2)
        self.assertAlmostEqual(n@(a-b),g)
    def test_boxes_overlap_not_depth_claim(self):
        g,*_=pair_projection(box(),pose([.5,0,0]),box(),np.eye(4));self.assertAlmostEqual(g,-.5)
    def test_spheres_exact(self):
        a=Shape('sphere','link','sphere',np.eye(4),{'radius':.2})
        self.assertAlmostEqual(pair_projection(a,pose([1,2,3]),a,np.eye(4))[0],np.sqrt(14)-.4)
    def test_finite_triangle_not_infinite_plane(self):
        tri=np.array([[[0,10,10],[0,11,10],[0,10,11]]],float)
        g,*_=triangle_projection(box(),np.eye(4),tri);self.assertGreater(g[0],0)
    def test_side_wall_contact(self):
        tri=np.array([[[.2,-2,-2],[.2,2,-2],[.2,0,2]]]);g,*_=triangle_projection(box(),np.eye(4),tri)
        self.assertLess(g[0],0)
    def test_world_transform_equivariance(self):
        h=pose([3,2,1]);h[:3,:3]=Rotation.from_rotvec([.6,.7,-.2]).as_matrix()
        a=pose([.7,.3,.1]);b=pose([0,0,0]);b[:3,:3]=Rotation.from_rotvec([.2,0,.3]).as_matrix()
        self.assertAlmostEqual(pair_projection(box(),a,box(),b)[0],pair_projection(box(),h@a,box(),h@b)[0])
    def test_triangle_world_transform_equivariance(self):
        h=pose([2,1,3]);h[:3,:3]=Rotation.from_rotvec([.4,.2,.1]).as_matrix()
        tri=np.array([[[.2,-2,-2],[.2,2,-2],[.2,0,2]]]);moved=tri@h[:3,:3].T+h[:3,3]
        self.assertAlmostEqual(triangle_projection(box(),np.eye(4),tri)[0][0],triangle_projection(box(),h,moved)[0][0])
    def test_original_supports_all43(self):
        _,shapes,_=load_shapes();t=pose([.3,-.2,.1]);t[:3,:3]=Rotation.from_rotvec([.4,.2,-.1]).as_matrix()
        d=np.array([[1.,2.,3.],[-1.,-2.,.2],[0,0,1],[1,0,0]])
        for s in shapes:np.testing.assert_allclose(supports(s,t@s.local,d),s.support(d,t),atol=1e-14)
class DerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body=WholeBodyTerms();cls.model=WorldFeatures('target')
        with np.load(PRIOR,allow_pickle=False) as z:
            cls.q=z['joint_pos'][[495]];cls.root=TK.K.world_pose(z['base_pos_w'][[495]],z['base_quat_w'][[495]],'wxyz')
        cls.g,cls.j,cls.records,cls.count=cls.body.evaluate(cls.model,cls.q,cls.root,True)
    def test_all35_frozen_jacobian(self):
        for k in range(35):
            d=np.zeros((1,35));d[:,k]=1e-6
            a=self.body.frozen_values(self.model,self.q+d[:,6:],perturb(self.root,d[:,:6]),self.records)
            b=self.body.frozen_values(self.model,self.q-d[:,6:],perturb(self.root,-d[:,:6]),self.records)
            np.testing.assert_allclose((a-b)/2e-6,self.j[:,k].toarray().ravel(),atol=2e-8,rtol=2e-7)
    def test_witness_values_reproduce(self):
        np.testing.assert_allclose(self.g,self.body.frozen_values(self.model,self.q,self.root,self.records),atol=1e-14)
    def test_no_adjacency_exemption(self):
        self.assertEqual(self.count['shape_pairs'],903);self.assertTrue(self.count['all_adjacency_pairs_retained'])
        self.assertTrue(any(r['kind']=='self' for r in self.records));self.assertTrue(any(r['kind']=='terrain' for r in self.records))
    def test_root_translation_self_cancels(self):
        ids=[i for i,r in enumerate(self.records) if r['kind']=='self']
        np.testing.assert_allclose(self.j[ids,:3].toarray(),0,atol=1e-15)
    def test_newly_entered_contacts_are_detected(self):
        above=self.root.copy();above[:,2,3]+=10
        _,before,count=self.body.evaluate(self.model,self.q,above)
        self.assertEqual(count['terrain_pairs_broadphase'],0)
        self.assertTrue(any(r['kind']=='terrain' for r in self.records))
if __name__=='__main__':unittest.main(verbosity=2)