"""Algebraic block solver software tests, not motion acceptance."""
import unittest
import numpy as np
from inputs import load
from overlap_solver import schedule,window_rows

class OverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(c):c.s,c.t,c.q,c.desired,c.binding=load()
    def test_plan_exact_finite_and_bidirectional(self):
        p=schedule();self.assertEqual(len(p),84);self.assertEqual(p[0][2:],(0,92));self.assertEqual(p[21][2:],(904,996))
        self.assertEqual(sum(r[1]=='coverage' for r in p),68);self.assertEqual(sum(r[1]=='endpoint_extra' for r in p),16)
    def test_all_frames_covered_each_sweep(self):
        for sweep in range(4):
            count=np.zeros(997,dtype=int)
            for s,role,a,b in schedule():
                if s==sweep:count[a:b+1]+=1
            self.assertTrue(np.all(count>=1));self.assertLessEqual(int(count.max()),4)
    def test_all_native_internal_edges_solved_not_just_anchored(self):
        covered=np.zeros(996,dtype=bool)
        for _,_,a,b in schedule():covered[a:b]=True
        self.assertTrue(covered.all())
    def test_hard_rows_contain_seed_for_every_unique_window(self):
        for a,b in sorted({(a,b) for _,_,a,b in schedule()}):
            g,h=window_rows(self.t,self.q,a,b)
            self.assertLessEqual(float(np.max(g@self.q[a:b+1].reshape(-1)-h)),1e-6)
    def test_outside_neighbours_are_current_not_original(self):
        current=self.q.copy();current[99,0]+=.001
        g,h=window_rows(self.t,current,100,192);g0,h0=window_rows(self.t,self.q,100,192)
        self.assertGreater(float(np.max(np.abs(h-h0))),.0009)
        self.assertEqual((g-g0).nnz,0)
    def test_direct_arbitrary_native_edge_values(self):
        current=self.q.copy();a,b=100,192;g,h=window_rows(self.t,current,a,b)
        n=93*29;row=(g@current[a:b+1].reshape(-1))[2*n:2*n+94*29].reshape(94,29)
        # Internal rows have no outside constant, and match original differences.
        np.testing.assert_allclose(row[1:-1],np.diff(current[a:b+1],axis=0),atol=1e-14)
    def test_endpoint_has_only_one_external_edge(self):
        g,h=window_rows(self.t,self.q,0,92);self.assertEqual(g.shape,(2*93*29+2*93*29,93*29))
        g,h=window_rows(self.t,self.q,904,996);self.assertEqual(g.shape,(2*93*29+2*93*29,93*29))
    def test_block_guard(self):
        for a,b in ((-1,90),(0,93),(904,997),(2,1),(0.,92)):
            with self.assertRaises(ValueError):window_rows(self.t,self.q,a,b)

if __name__=='__main__':unittest.main()