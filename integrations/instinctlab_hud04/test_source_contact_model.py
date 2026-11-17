"""Analytic/sampled geometry checks; not physics or reference acceptance."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import audit_source_contact_model as A
import contact_geometry as G
import kinematic_diagnostics as K


def primitive(kind, **kwargs):
    return dict(kind=kind, transform=np.eye(4), **kwargs)


class SupportTests(unittest.TestCase):
    def test_box_bounds(self):
        np.testing.assert_allclose(G.bounds([primitive('box', size=np.array([2, 4, 6]))]), [[-1, -2, -3], [1, 2, 3]])

    def test_sphere_nonunit_direction(self):
        p = G.support(primitive('sphere', radius=2), [[0, 0, -10]])
        np.testing.assert_allclose(p, [[0, 0, -2]])

    def test_capsule_length_excludes_caps(self):
        np.testing.assert_allclose(G.bounds([primitive('capsule', radius=2, length=6)]), [[-2, -2, -5], [2, 2, 5]])

    def test_cylinder_does_not_add_radius_axially(self):
        np.testing.assert_allclose(G.bounds([primitive('cylinder', radius=2, length=6)]), [[-2, -2, -3], [2, 2, 3]])

    def test_union_not_intersection(self):
        a, b = primitive('sphere', radius=1), primitive('sphere', radius=2)
        b['transform'][:3, 3] = [4, 0, 0]
        np.testing.assert_allclose(G.bounds([a, b]), [[-1, -2, -2], [6, 2, 2]])

    def test_collider_origin_rotation_and_translation(self):
        p = primitive('capsule', radius=.1, length=2)
        p['transform'][:3, :3] = K.axis_rotation([0, 1, 0], np.pi/2)
        p['transform'][:3, 3] = [2, 3, 4]
        np.testing.assert_allclose(G.bounds([p]), [[.9, 2.9, 3.9], [3.1, 3.1, 4.1]], atol=1e-12)

    def test_mesh_extrema(self):
        vertices = np.array([[1, 2, 3], [-1, 0, 4], [0, -3, -5]])
        np.testing.assert_allclose(G.bounds([primitive('mesh', vertices=vertices)]), [[-1, -3, -5], [1, 2, 4]])

    def test_mesh_chunked_queries(self):
        rng = np.random.default_rng(17)
        vertices, d = rng.normal(size=(90, 3)), rng.normal(size=(200, 3))
        points = G.support(primitive('mesh', vertices=vertices), d)
        np.testing.assert_allclose(np.sum(points*d, axis=1), (d@vertices.T).max(axis=1))

    def test_capsule_against_independent_sampled_surface(self):
        rng = np.random.default_rng(7)
        d = rng.normal(size=(70, 3))
        unit = rng.normal(size=(15000, 3)); unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        # Minkowski sum: segment endpoints + unit sphere; independent brute-force envelope.
        samples = np.concatenate([unit + [0, 0, z] for z in (-2, 2)])
        actual = np.sum(G.support(primitive('capsule', radius=1, length=4), d)*d, axis=1)
        lower = (d@samples.T).max(axis=1)
        self.assertTrue(np.all(actual >= lower-1e-12))
        self.assertLess(np.max(actual-lower), .005)

    def test_bad_directions(self):
        for d in ([[0, 0, 0]], [[0, np.nan, 1]], [1, 2, 3], np.ones((4097, 3)), np.zeros((0, 3))):
            with self.subTest(d=np.shape(d)), self.assertRaises(ValueError):
                G.support(primitive('sphere', radius=1), d)

    def test_empty_union_rejected(self):
        with self.assertRaises(ValueError):
            G.bounds([])

    def test_positive_dimensions(self):
        for x in (0, -1, 'nan', 'inf'):
            with self.subTest(x=x), self.assertRaises(ValueError):
                G.positive(x)


class ParsingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'model.urdf'

    def model(self, geometry):
        self.path.write_text('<robot><link name="a"><collision><geometry>'+geometry+'</geometry></collision></link></robot>')

    def test_replacement_requires_explicit_bool(self):
        self.model('<sphere radius="1"/>')
        for flag in (None, 1, 'true'):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                G.collisions(self.path, 'a', replace_cylinders=flag)

    def test_cylinder_conversion_flag(self):
        self.model('<cylinder radius=".5" length="2"/>')
        self.assertEqual(G.collisions(self.path, 'a', replace_cylinders=True)[0]['kind'], 'capsule')
        self.assertEqual(G.collisions(self.path, 'a', replace_cylinders=False)[0]['kind'], 'cylinder')

    def test_nonpositive_box_rejected(self):
        self.model('<box size="1 -1 2"/>')
        with self.assertRaises(ValueError):
            G.collisions(self.path, 'a', replace_cylinders=False)

    def test_unsupported_shape_rejected(self):
        self.model('<cone radius="1"/>')
        with self.assertRaises(ValueError):
            G.collisions(self.path, 'a', replace_cylinders=False)

    def test_multiple_shapes_per_collider_rejected(self):
        self.model('<sphere radius="1"/><sphere radius="2"/>')
        with self.assertRaises(ValueError):
            G.collisions(self.path, 'a', replace_cylinders=False)

    def test_no_collider_rejected(self):
        self.path.write_text('<robot><link name="a"/></robot>')
        with self.assertRaises(ValueError):
            G.collisions(self.path, 'a', replace_cylinders=False)

    def test_mesh_not_allowlisted_rejected(self):
        self.model('<mesh filename="hand.STL"/>')
        with self.assertRaises(ValueError):
            G.collisions(self.path, 'a', replace_cylinders=False)

    def test_remote_or_absolute_mesh_rejected(self):
        for name in ('https://example.com/mesh.STL', '/workspace/mesh.STL'):
            self.model('<mesh filename="'+name+'"/>')
            with self.subTest(name=name), self.assertRaises(ValueError):
                G.collisions(self.path, 'a', replace_cylinders=False)

    def test_pin_failure_before_loading_motion(self):
        self.path.write_text('not the pinned model')
        with self.assertRaises(ValueError):
            A.audit(self.path, self.path, self.path, self.path)

    def test_cli_has_no_solve_option(self):
        p = subprocess.run([sys.executable, '-B', str(Path(A.__file__)), '--help'], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertNotIn('--solve', p.stdout)
        self.assertNotIn('--execute', p.stdout)

    def test_report_not_overwritten(self):
        output = Path(self.temp.name)/'output.json'
        output.write_text('original')
        p = subprocess.run([sys.executable, '-B', str(Path(A.__file__)), '--task-urdf', str(self.path),
            '--earlier-urdf', str(self.path), '--target-urdf', str(self.path), '--motion', str(self.path),
            '--report', str(output)], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual(output.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()