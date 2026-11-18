import unittest
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from nominal_certificates import Solid, classify, project_triangle, classify_triangles


def poses(n=1):
    return np.broadcast_to(np.eye(4), (n, 4, 4)).copy()


def solids():
    v = np.array(np.meshgrid(*[[-1., 1.]] * 3)).reshape(3, -1).T
    h = ConvexHull(v)
    return [Solid('sphere', 'sphere', [1.]), Solid('box', 'box', [1., 1., 1.]),
            Solid('cylinder', 'cylinder', [1., 1.]), Solid('mesh', 'mesh', [], v, h.equations)]


class Certificates(unittest.TestCase):
    def test_all_sixteen_pair_types_strict(self):
        for a in solids():
            for b in solids():
                with self.subTest(a=a.kind, b=b.kind):
                    t = poses(); t[:, 0, 3] = 1.5
                    r = classify(a, poses(), b, t)
                    self.assertTrue(r['strict'][0]); self.assertFalse(r['separated'][0])
                    self.assertGreater(a.margin(r['common_point'], poses())[0], 1e-8)
                    self.assertGreater(b.margin(r['common_point'], t)[0], 1e-8)

    def test_all_sixteen_pair_types_separated(self):
        for a in solids():
            for b in solids():
                t = poses(); t[:, 1, 3] = 2.3
                r = classify(a, poses(), b, t)
                self.assertTrue(r['separated'][0]); self.assertFalse(r['strict'][0])
                self.assertAlmostEqual(r['separating_gap'][0], .3)

    def test_touching_is_not_strict_or_safe(self):
        for a in solids():
            t = poses(); t[:, 0, 3] = 2
            r = classify(a, poses(), a, t)
            self.assertTrue(r['unresolved'][0])

    def test_cylinder_is_not_capsule(self):
        cylinder = solids()[2]; ball = Solid('ball', 'sphere', [.1])
        t = poses(); t[:, 2, 3] = 1.5
        self.assertTrue(classify(cylinder, poses(), ball, t)['separated'][0])

    def test_support_matches_sampled_primitives(self):
        rng = np.random.default_rng(76)
        d = rng.normal(size=(1, 37, 3)); d /= np.linalg.norm(d, axis=2, keepdims=True)
        for s in solids():
            lo, hi = s.interval(d, poses())
            if s.kind == 'box':
                v = np.array(np.meshgrid(*[[-1., 1.]] * 3)).reshape(3, -1).T
            elif s.kind == 'mesh':
                v = s.vertices
            else:
                z = rng.normal(size=(1000, 3))
                if s.kind == 'sphere':
                    v = z / np.linalg.norm(z, axis=1, keepdims=True)
                else:
                    z[:, :2] /= np.linalg.norm(z[:, :2], axis=1, keepdims=True)
                    z[:, 2] = np.sign(z[:, 2]); v = z
            values = d[0] @ v.T
            self.assertTrue(np.all(values <= hi[0, :, None] + 1e-12))
            self.assertTrue(np.all(values >= lo[0, :, None] - 1e-12))
            if s.kind in ('box', 'mesh'):
                np.testing.assert_allclose(values.max(axis=1), hi[0], atol=1e-12, rtol=0)

    def test_random_rigid_invariance_and_rotated_inputs(self):
        rng = np.random.default_rng(567)
        for a in solids():
            b = solids()[2]; n = 30
            x = poses(n); x[:, :3, :3] = Rotation.random(n, random_state=rng).as_matrix()
            x[:, :3, 3] = rng.normal(size=(n, 3))
            y = x.copy(); y[:15, :3, 3] += [.01, .01, .01]; y[15:, :3, 3] += [4, 0, 0]
            r = classify(a, x, b, y)
            g = poses(n); g[:, :3, :3] = Rotation.random(n, random_state=rng).as_matrix()
            g[:, :3, 3] = rng.normal(size=(n, 3))
            q = classify(a, g @ x, b, g @ y)
            np.testing.assert_array_equal(r['strict'], q['strict'])
            np.testing.assert_array_equal(r['separated'], q['separated'])
            self.assertTrue(q['strict'][:15].all()); self.assertTrue(q['separated'][15:].all())

    def test_query_budget_never_claims_unresolved_safe(self):
        a = solids()[0]; t = poses(); t[:, 0, 3] = 2
        r = classify(a, poses(), a, t, iterations=1)
        self.assertTrue(r['unresolved'][0])

    def test_invalid_geometry_fail_closed(self):
        a = solids()[0]
        for dim in ([float('nan')], [-1], [1, 2]):
            with self.assertRaises(ValueError): Solid('bad', 'sphere', dim)
        for tolerance in (0, -1, float('nan'), 1.):
            with self.assertRaises(ValueError): classify(a, poses(), a, poses(), tolerance=tolerance)
        for budget in (0, 129, 1.2, True):
            with self.assertRaises(ValueError): classify(a, poses(), a, poses(), iterations=budget)
        for t in (poses(2), np.ones((1, 4, 4)), np.full((1, 4, 4), np.nan)):
            with self.assertRaises(ValueError): classify(a, poses(), a, t)

    def test_triangle_interior_edge_and_vertex_projection(self):
        tri=np.broadcast_to([[0.,0,0],[2,0,0],[0,2,0]],(3,3,3)).copy()
        p=project_triangle([[.3,.4,1],[1,1,2],[-2,-2,1]],tri)
        np.testing.assert_allclose(p,[[.3,.4,0],[1,1,0],[0,0,0]],atol=1e-12)

    def test_all_four_solids_surface_hit_separation_touch(self):
        for a in solids():
            tri=np.array([[[-3.,-3,0],[3,-3,0],[0,3,0]]])
            self.assertTrue(classify_triangles(a,poses(),tri)['strict'][0])
            tri[:,:,2]=3
            self.assertTrue(classify_triangles(a,poses(),tri)['separated'][0])
            tri[:,:,2]=1
            self.assertTrue(classify_triangles(a,poses(),tri)['unresolved'][0])

    def test_finite_triangle_not_infinite_plane(self):
        tri=np.array([[[3.,0,0],[4,0,0],[3,1,0]]])
        r=classify_triangles(solids()[1],poses(),tri)
        self.assertTrue(r['separated'][0]);self.assertFalse(r['strict'][0])

    def test_triangle_winding_rigid_invariance(self):
        rng=np.random.default_rng(42);n=30
        g=poses(n);g[:,:3,:3]=Rotation.random(n,random_state=rng).as_matrix()
        g[:,:3,3]=rng.normal(size=(n,3))
        tri=np.broadcast_to([[-3.,-3,0],[3,-3,0],[0,3,0]],(n,3,3)).copy()
        world=np.einsum('nij,nkj->nki',g[:,:3,:3],tri)+g[:,None,:3,3]
        for a in solids():
            r=classify_triangles(a,g,world[:,::-1])
            self.assertTrue(r['strict'].all());self.assertFalse(r['separated'].any())

    def test_degenerate_triangle_never_safe(self):
        with self.assertRaises(ValueError):
            classify_triangles(solids()[0],poses(),np.zeros((1,3,3)))
        with self.assertRaises(ValueError):
            project_triangle([[0,0,0]],np.zeros((1,3,3)))


class ExplicitMotionBinding(unittest.TestCase):
    def test_alternate_motion_requires_wellformed_explicit_digest(self):
        from audit_self_pairs import load_case
        for motion,digest in [('missing.npz',None),(None,'a'*64),('missing.npz','short'),('missing.npz','Z'*64)]:
            with self.assertRaises(ValueError):
                load_case(motion,digest)


if __name__ == '__main__':
    unittest.main()