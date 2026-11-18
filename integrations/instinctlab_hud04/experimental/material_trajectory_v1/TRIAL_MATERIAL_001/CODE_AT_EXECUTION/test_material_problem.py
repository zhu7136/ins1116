import unittest
from unittest.mock import patch
from scipy import sparse
from types import SimpleNamespace
import numpy as np
from material_problem import problem, perturb, hard_rows, feasibility, K, CONTRACT, sha
from wholebody_step_guard import compare_states, EndpointState


class MaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.q, cls.roots, cls.want, cls.w, cls.support, cls.body, cls.guard, cls.kernel = problem()

    def test_native_contract(self):
        self.assertEqual(self.q.shape, (997, 29)); self.assertEqual(len(self.m.names), 29)
        self.assertFalse(any('head' in n for n in self.m.names))
        self.assertEqual(int((~self.support.any(axis=1)).sum()), 346)
        self.assertLess(max(feasibility(self.m, self.q).values()), 1e-6)

    def test_joint_root_material_jacobian(self):
        q, r = self.q[[0, 300, 996]], self.roots[[0, 300, 996]]
        value, jac = self.m.evaluate(q, r, True)
        for c in range(35):
            step = np.zeros((3, 35)); step[:, c] = 1e-6
            plus = self.m.evaluate(q+step[:, 6:], perturb(r, step[:, :6]))
            minus = self.m.evaluate(q-step[:, 6:], perturb(r, -step[:, :6]))
            np.testing.assert_allclose((plus-minus)/2e-6, jac[:, :, c], rtol=0, atol=5e-9)

    def test_material_world_independent_fk(self):
        q, r = self.q[[0, 300, 996]], self.roots[[0, 300, 996]]
        tree = self.guard.tree.fk(q, self.m.names)
        t = np.stack([r@tree[row['target_link']]@np.array(row['target_T_link_material']) for row in self.m.rows], axis=1)
        expected = np.c_[t[:, :, :3, 3].reshape(3, 18), t[:, :, :3, 2].reshape(3, 18)]
        np.testing.assert_allclose(self.m.evaluate(q, r), expected, rtol=0, atol=2e-14)

    def test_original_solid_support_adapter(self):
        rng = np.random.default_rng(941); d = rng.normal(size=(16, 3))
        for converted, (_, solid, local) in zip(self.body.shapes, self.guard.inventory):
            p = self.kernel['supports'](converted, local, d)
            lo, hi = solid.interval(d[None], local[None])
            np.testing.assert_allclose(np.sum(p*d, axis=1), hi[0], rtol=0, atol=1e-12)

    def test_original_limits_zero_step(self):
        for first, last in [(0, 60), (290, 350), (936, 996)]:
            a, b = hard_rows(self.m, self.q, first, last)
            self.assertLess(float(np.max(-b)), 1e-6)
            self.assertEqual(a.shape[1], (last-first+1)*35)

    def test_native_neighbour_constraints(self):
        a, b = hard_rows(self.m, self.q, 300, 300)
        step = np.zeros(35); step[6] = 10.
        self.assertGreater(float(np.max(a@step-b)), 0)
        self.assertEqual(a.shape[0], 29*6)

    def test_frozen_geometry_jacobian(self):
        q, r = self.q[[300]], self.roots[[300]]
        gaps, jac, records, counts = self.body.evaluate(self.m, q, r, True)
        np.testing.assert_allclose(gaps, self.body.frozen_values(self.m, q, r, records), rtol=0, atol=1e-12)
        for c in range(35):
            step = np.zeros((1, 35)); step[0, c] = 1e-6
            p = self.body.frozen_values(self.m, q+step[:, 6:], perturb(r, step[:, :6]), records)
            n = self.body.frozen_values(self.m, q-step[:, 6:], perturb(r, -step[:, :6]), records)
            np.testing.assert_allclose((p-n)/2e-6, jac[:, c].toarray().ravel(), rtol=0, atol=5e-9)

    def test_no_silent_nonfinite_tangent(self):
        with self.assertRaises(ValueError): perturb(self.roots[:1], np.full((1, 6), np.nan))

    def test_original_guard_requires_all_pairs(self):
        b = 'a'*64; old = EndpointState(b, (300,), frozenset(), frozenset(), 903, 43*8068)
        new = EndpointState(b, (300,), frozenset({('terrain', 300, 0, 0)}), frozenset(), 903, 43*8068)
        self.assertFalse(compare_states(old, new)[0])

    def test_finite_schedule_covers_all_native_frames(self):
        from solve_material_once import schedule
        counts = np.zeros(997, int)
        for a, b in schedule(): counts[a:b+1] += 1
        self.assertEqual(len(schedule()), 34)
        self.assertTrue((counts > 0).all())
        self.assertTrue(all(b-a == 60 for a, b in schedule()))

    def test_real_solver_control_flow_with_explicit_zero_step_stub(self):
        from solve_material_once import solve
        # Software control-flow fixture, NOT a real optimization or geometry audit.
        class EmptyBody:
            margin = .001
            shapes = []
            def evaluate(self, m, q, r, jacobian=False):
                if jacobian: return np.empty(0), sparse.csr_matrix((0, len(q)*35)), [], {}
                return np.empty(0), [], {}
        def zero_lp(c, **kwargs):
            return SimpleNamespace(success=True, status=0, x=np.zeros(len(c)))
        m = SimpleNamespace(model=self.m.model, names=self.m.names,
                            evaluate=lambda q, r, jacobian=False: (np.zeros((len(q),36)), np.zeros((len(q),36,35))) if jacobian else np.zeros((len(q),36)))
        progress = []
        with patch('solve_material_once.linprog', zero_lp):
            q, r, result = solve(m, self.q, self.roots, np.zeros((997,36)), np.zeros((997,36)),
                                 self.support, EmptyBody(), None, progress.append)
        self.assertEqual(result['actual_lp_calls'], 34)
        self.assertEqual(result['visited_frames'], 997)
        self.assertEqual(result['accepted_updates'], 0)
        np.testing.assert_array_equal(q, self.q)
        np.testing.assert_array_equal(r, self.roots)


if __name__ == '__main__': unittest.main()