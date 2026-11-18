import unittest
import numpy as np
from scipy.interpolate import BPoly
from euclidean_c2 import coefficients, evaluate, component_extrema


class SharedC2Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261118)
        self.p, self.v, self.a = (rng.normal(size=(18, 3)) for _ in range(3))
        self.dt = 1/90
        self.c = coefficients(self.p, self.v, self.a, self.dt)

    def test_both_endpoints(self):
        for order, nodes in enumerate([self.p, self.v, self.a]):
            np.testing.assert_allclose(evaluate(self.c,self.dt,0,order), nodes[:-1], atol=1e-7)
            np.testing.assert_allclose(evaluate(self.c,self.dt,1,order), nodes[1:], atol=1e-7)

    def test_every_global_join_not_only_windows(self):
        for order in range(3):
            np.testing.assert_allclose(evaluate(self.c,self.dt,1,order)[:-1],
                                       evaluate(self.c,self.dt,0,order)[1:], atol=1e-7)

    def test_scipy_independent_representation(self):
        t = np.arange(len(self.p))*self.dt
        bp = BPoly.from_derivatives(t, [list(x) for x in zip(self.p,self.v,self.a)])
        for u in [.13,.37,.81]:
            for order in range(3):
                np.testing.assert_allclose(evaluate(self.c,self.dt,u,order),
                    bp(t[:-1]+u*self.dt, nu=order), rtol=1e-7,atol=2e-7)

    def test_quintic_zero_endpoint_rates_interior_speed(self):
        p = np.array([[0.],[1.]])
        c = coefficients(p, np.zeros_like(p), np.zeros_like(p), 2.)
        _, hi, _, arg = component_extrema(c,2.,1)
        self.assertAlmostEqual(hi.item(),1.875/2.,places=12)
        self.assertAlmostEqual(arg.item(),.5,places=12)

    def test_interior_position_overshoot_detected(self):
        p = np.zeros((2,1)); v = np.array([[4.],[-4.]])
        c = coefficients(p,v,p,1.)
        self.assertGreater(component_extrema(c,1.,0)[1].item(),1.)

    def test_linear_motion(self):
        p = np.arange(4.)[:,None]*.1
        c = coefficients(p,np.ones_like(p)*2,np.zeros_like(p),.05)
        np.testing.assert_allclose(component_extrema(c,.05,1)[0],2.,atol=1e-12)
        np.testing.assert_allclose(evaluate(c,.05,.42,2),0.,atol=1e-10)

    def test_dt_scaling(self):
        c = coefficients(self.p,self.v/2,self.a/4,2*self.dt)
        np.testing.assert_allclose(c,self.c)
        np.testing.assert_allclose(evaluate(c,2*self.dt,.3,1),evaluate(self.c,self.dt,.3,1)/2)

    def test_node_change_affects_both_neighbors_only(self):
        v = self.v.copy(); v[8] += 1
        new = coefficients(self.p,v,self.a,self.dt)
        changed = np.where(np.any(new != self.c,axis=(1,2)))[0].tolist()
        self.assertEqual(changed,[7,8])

    def test_dense_samples_within_stationary_extrema(self):
        for order in [0,1,2]:
            lo,hi,*_ = component_extrema(self.c,self.dt,order)
            for u in np.linspace(0,1,101):
                val = evaluate(self.c,self.dt,float(u),order)
                self.assertTrue(np.all(val >= lo-1e-7))
                self.assertTrue(np.all(val <= hi+1e-7))

    def test_reject_nonfinite(self):
        p = self.p.copy(); p[0,0] = np.nan
        with self.assertRaises(ValueError): coefficients(p,self.v,self.a,self.dt)

    def test_reject_mismatched_nodes(self):
        with self.assertRaises(ValueError): coefficients(self.p,self.v[:-1],self.a,self.dt)

    def test_reject_invalid_time(self):
        for dt in [0,-1,float('inf')]:
            with self.assertRaises(ValueError): coefficients(self.p,self.v,self.a,dt)
        with self.assertRaises(ValueError): evaluate(self.c,self.dt,1.1)


if __name__ == '__main__': unittest.main(verbosity=2)