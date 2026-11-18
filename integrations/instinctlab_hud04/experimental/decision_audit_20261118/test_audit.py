import unittest
from types import SimpleNamespace
import numpy as np
from audit import feasible_box,angles

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.m=SimpleNamespace(lower=np.array([-1.]),upper=np.array([1.]),vmax=np.array([9.]))
    def test_both_native_neighbours(self):
        lo,hi=feasible_box(self.m,np.array([[.1],[.15],[.2]]),1)
        np.testing.assert_allclose(lo,[.1]);np.testing.assert_allclose(hi,[.2])
    def test_endpoint_has_one_neighbour(self):
        lo,hi=feasible_box(self.m,np.array([[.1],[.15],[.2]]),0)
        np.testing.assert_allclose(lo,[.05]);np.testing.assert_allclose(hi,[.25])
    def test_position_limit_not_relaxed(self):
        lo,hi=feasible_box(self.m,np.array([[1.],[1.]]),0)
        np.testing.assert_allclose(lo,[.9]);np.testing.assert_allclose(hi,[1.])
    def test_infeasible_baseline_rejected(self):
        with self.assertRaises(ValueError):feasible_box(self.m,np.array([[.1],[.9],[.1]]),1)
    def test_inputs_unchanged(self):
        q=np.array([[.1],[.15],[.2]]);saved=q.copy();feasible_box(self.m,q,1)
        np.testing.assert_array_equal(q,saved)
    def test_no_clipping_for_angles(self):
        np.testing.assert_allclose(angles(np.array([[1.,0,0],[-1.,0,0]]),np.array([[1.,0,0],[1.,0,0]])),[0,180])

if __name__=='__main__':unittest.main()