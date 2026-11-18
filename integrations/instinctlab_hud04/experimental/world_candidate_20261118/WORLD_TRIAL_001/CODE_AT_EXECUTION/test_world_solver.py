import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from world_features import problem_data
from world_solver import plan,hard_rows,solve,joint_feasibility

class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m,cls.q,cls.roots,cls.want,_=problem_data()

    def test_exact_native_coverage(self):
        count=np.zeros(997,int)
        for a,b in plan():count[a:b+1]+=1;self.assertEqual(b-a+1,93)
        self.assertEqual(len(plan()),34);self.assertTrue(np.all(count>=2))

    def test_hard_rows_match_independent_global_constraints(self):
        q=self.q.copy();first,last=244,336;n=93
        g,h=hard_rows(self.m,q,first,last);rng=np.random.default_rng(94)
        d=rng.normal(size=(n,35))*.001;trial=q.copy();trial[first:last+1]+=d[:,6:]
        actual=float(np.max(g@d.reshape(-1)-h));m=self.m.model
        # All and only position rows in window and native velocity edges that
        # touch it, with both CURRENT outer neighbours held fixed.
        edges=np.arange(first-1,last+1)
        independent=max(np.max(trial[first:last+1]-m.upper),np.max(m.lower-trial[first:last+1]),
                        np.max(abs(trial[edges+1]-trial[edges])-m.vmax/90))
        self.assertAlmostEqual(actual,independent,places=13)

    def test_root_columns_not_joint_bounds(self):
        g,h=hard_rows(self.m,self.q,0,92)
        for i in range(93):self.assertEqual(g[:,35*i:35*i+6].nnz,0)
        self.assertLessEqual(float(np.max(-h)),1e-6)

    def test_bad_native_size_rejected(self):
        with self.assertRaises(ValueError):solve(self.m,self.q[:93],self.roots[:93],self.want[:93],lambda r:None)

    def test_failed_lp_no_result_returned(self):
        fake=SimpleNamespace(success=False,status=1,nit=12000)
        with patch('world_solver.linprog',return_value=fake):
            with self.assertRaisesRegex(RuntimeError,'NO motion export'):solve(self.m,self.q,self.roots,self.want,lambda r:None)

    def test_seed_full_limits(self):
        self.assertLess(max(joint_feasibility(self.m,self.q).values()),1e-6)

if __name__=='__main__':unittest.main(verbosity=2)