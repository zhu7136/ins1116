import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from tail_solver import angles, objective, plan, solve, window_rows


class SimpleModel:
    lower = np.full(29, -1.)
    upper = np.ones(29)
    vmax = np.full(29, 10.)

    def evaluate(self, q, jacobian=False):
        t = q[:, 0]
        u = np.zeros((len(q), 8, 3)); u[:, :, 0] = np.cos(t[:, None]); u[:, :, 1] = np.sin(t[:, None])
        if not jacobian:
            return u
        j = np.zeros((len(q), 8, 3, 29))
        j[:, :, 0, 0] = -np.sin(t[:, None]); j[:, :, 1, 0] = np.cos(t[:, None])
        return u, j


class TailTests(unittest.TestCase):
    def setUp(self):
        self.m = SimpleModel(); self.q = np.zeros((997, 29))

    def test_fixed_bounded_plan(self):
        p = plan(np.zeros((997, 8)))
        self.assertLessEqual(len(p), 144)
        self.assertEqual(len(p), len(plan(np.zeros((997, 8)))))
        visited = np.zeros(997, bool)
        for _, _, a, b in p:
            self.assertLessEqual(b-a+1, 93); visited[a:b+1] = True
        self.assertTrue(visited.all())

    def test_true_angle_not_clipped_cosine(self):
        a = np.array([[[1., 0., 0.]]]); b = np.array([[[-1., 0., 0.]]])
        self.assertAlmostEqual(float(angles(a, b)[0, 0]), np.pi)

    def test_tail_penalty_increases(self):
        self.assertGreater(objective(np.array([[1.]]), np.zeros(1)), objective(np.array([[.2]]), np.zeros(1)))

    def test_regularizer(self):
        e = np.zeros((1, 1))
        self.assertAlmostEqual(objective(e, np.array([1.])), .001)

    def test_window_has_both_outside_edges(self):
        g, h = window_rows(self.m, self.q, 100, 192)
        trial = np.zeros((93, 29)); trial[0, 0] = .12
        self.assertGreater(float((g@trial.reshape(-1)-h).max()), 0.)
        trial = np.zeros((93, 29)); trial[-1, 0] = .12
        self.assertGreater(float((g@trial.reshape(-1)-h).max()), 0.)

    def test_fixed_outside_values(self):
        self.q[99, 0] = .2
        g, h = window_rows(self.m, self.q, 100, 192)
        self.assertGreater(float((-h).max()), 0.)

    def test_native_first_last(self):
        for a, b in [(0, 92), (904, 996)]:
            g, h = window_rows(self.m, self.q, a, b)
            self.assertLessEqual(float((-h).max()), 0.)

    def test_no_joint_clipping(self):
        self.q[0, 0] = 2.
        with self.assertRaisesRegex(ValueError, 'Hard feasible'):
            solve(self.m, self.q, self.m.evaluate(np.zeros_like(self.q)))

    def test_no_velocity_clipping(self):
        self.q[1, 0] = .2
        with self.assertRaisesRegex(ValueError, 'Hard feasible'):
            solve(self.m, self.q, self.m.evaluate(np.zeros_like(self.q)))

    def test_no_resampling(self):
        with self.assertRaisesRegex(ValueError, 'All997'):
            solve(self.m, self.q[:-1], self.m.evaluate(self.q)[:-1])

    def test_wall_bound(self):
        with self.assertRaisesRegex(ValueError, 'Finite algorithm'):
            solve(self.m, self.q, self.m.evaluate(self.q), seconds=91.)

    def test_unit_targets(self):
        with self.assertRaisesRegex(ValueError, 'unit'):
            solve(self.m, self.q, self.m.evaluate(self.q)*2)

    def test_one_real_synthetic_lp_improves(self):
        self.q[:] = 0.; self.q[:, 0] = .5
        target = self.m.evaluate(np.zeros_like(self.q))
        with patch('tail_solver.plan', return_value=[(0, 'synthetic', 0, 92)]):
            out, report = solve(self.m, self.q, target)
        self.assertEqual(report['actual_lp_calls'], 1)
        self.assertEqual(report['accepted_updates'], 1)
        self.assertLess(report['final_objective'], report['initial_objective'])
        np.testing.assert_array_equal(out[93:], self.q[93:])
        self.assertLessEqual(report['velocity_violation_rad_s'], 1e-6)
        self.assertFalse(report['accepted_reference'])

    def test_failed_lp_does_not_retry(self):
        target = self.m.evaluate(self.q)
        with patch('tail_solver.linprog', return_value=SimpleNamespace(success=False, status=1)) as p:
            out, report = solve(self.m, self.q, target)
        self.assertEqual(p.call_count, 1)
        np.testing.assert_array_equal(out, self.q)
        self.assertEqual(report['termination'], 'LP_stopped_status_1')


if __name__ == '__main__':
    unittest.main()