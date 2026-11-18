import unittest
import numpy as np
from scipy import sparse
from world_features import WorldFeatures,TK,perturb
from terrain_temporal_terms import FootTerms,box_clip,temporal_values,temporal_system
from collision_temporal_solver import l1_inequalities,plan

class TermsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load('/workspace/hud04-world-candidate-20261118/WORLD_TRIAL_002_BLOCK31/STEP1_WORLD_CANDIDATE_UNACCEPTED.npz',allow_pickle=False) as z:
            cls.q=z['joint_pos'];cls.roots=TK.K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz')
        cls.model=WorldFeatures('target');cls.foot=FootTerms()

    def test_temporal_derivative_all35_with_fixed_neighbours(self):
        first,last=492,494;q=self.q.copy();r=self.roots.copy();v,j=temporal_system(q,r,first,last)
        np.testing.assert_array_equal(v,temporal_values(q,r,first,last))
        eps=1e-6
        for column in range(35):
            d=np.zeros((3,35));d[1,column]=eps
            a=q.copy();b=q.copy();a[first:last+1]+=d[:,6:];b[first:last+1]-=d[:,6:]
            ra=r.copy();rb=r.copy();ra[first:last+1]=perturb(ra[first:last+1],d[:,:6]);rb[first:last+1]=perturb(rb[first:last+1],-d[:,:6])
            fd=(temporal_values(a,ra,first,last)-temporal_values(b,rb,first,last))/(2*eps)
            np.testing.assert_allclose(fd,j[:,35+column].toarray().ravel(),atol=2e-7,rtol=2e-7)
        self.assertEqual(j.shape,(5*41,3*35))

    def test_triangle_strict_interior_and_outside(self):
        tri=np.array([[-2,-2,0],[2,-2,0],[0,2,0.]])
        inside=box_clip(tri,np.ones(3));self.assertGreater(len(inside),2);self.assertTrue(np.all(abs(inside)<=1+1e-12))
        self.assertEqual(len(box_clip(tri+[0,0,2],np.ones(3))),0)

    def test_foot_cuts_every_column_fd(self):
        # All997 are audited separately; select a nonempty actual terrain
        # neighbourhood without choosing a coordinate convention by gap.
        chosen=None
        for frame in [495,674,257,726,0,996]:
            q=self.q[frame:frame+1];r=self.roots[frame:frame+1];g,j,records=self.foot.cuts(self.model,q,r)
            if len(records):chosen=(q,r,g,j,records);break
        self.assertIsNotNone(chosen);q,r,g,j,records=chosen
        np.testing.assert_allclose(g,self.foot.fixed_cut_gaps(self.model,q,r,records),atol=1e-12)
        for k in range(35):
            d=np.zeros((1,35));d[0,k]=1e-6
            a=self.foot.fixed_cut_gaps(self.model,q+d[:,6:],perturb(r,d[:,:6]),records)
            b=self.foot.fixed_cut_gaps(self.model,q-d[:,6:],perturb(r,-d[:,:6]),records)
            np.testing.assert_allclose((a-b)/2e-6,j[:,k].toarray().ravel(),atol=1e-8,rtol=1e-7)

    def test_l1_rows(self):
        a=sparse.csr_matrix([[2.,-1.],[.5,3.]]);r=np.array([-1.,2.]);d=np.array([.1,-.2]);s=abs(a@d+r)
        g,h=l1_inequalities(a,r,2)
        self.assertLessEqual(np.max(g@np.r_[d,s]-h),1e-12)
        self.assertGreater(np.max(g@np.r_[d,s-1e-4]-h),1e-5)

    def test_block11_full_coverage(self):
        count=np.zeros(997,int)
        for first,last in plan():
            self.assertEqual(last-first+1,11);count[first:last+1]+=1
        self.assertEqual(len(plan()),200);self.assertTrue(np.all(count>=2))

if __name__=='__main__':unittest.main()