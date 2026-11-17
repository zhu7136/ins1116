"""Synthetic CPU checks only; not solver, robot simulation or contact acceptance."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('retarget_diagnostics_tested', ROOT/'kinematic_diagnostics.py')
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)


def tree(joint_type='revolute', axis='0 0 1', origin='1 0 0', extra=''):
    return K.RobotTree(('''<robot name="test"><link name="a"/><link name="b"/>
      <joint name="j" type="%s"><parent link="a"/><child link="b"/>
      <origin xyz="%s"/><axis xyz="%s"/>%s</joint></robot>''' % (joint_type, origin, axis, extra)).encode())


class FKTests(unittest.TestCase):
    def test_revolute_translation_then_rotation(self):
        fk = tree().fk([[np.pi/2]], ['j'])
        np.testing.assert_allclose(fk['b'][0, :3, 3], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(fk['b'][0, :3, :3]@[1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_fixed(self):
        fk = tree('fixed').fk(np.zeros((2, 0)), [])
        np.testing.assert_allclose(fk['b'][:, :3, 3], [[1, 0, 0]]*2)

    def test_axis_normalized(self):
        np.testing.assert_allclose(tree(axis='0 0 9').fk([[1.]], ['j'])['b'], tree().fk([[1.]], ['j'])['b'])

    def test_continuous_joint(self):
        np.testing.assert_allclose(tree('continuous').fk([[2*np.pi]], ['j'])['b'][0, :3, :3], np.eye(3), atol=1e-12)

    def test_rpy_order(self):
        import xml.etree.ElementTree as ET
        t = K.origin_transform(ET.fromstring('<origin rpy="1.5707963267948966 0 1.5707963267948966"/>'))
        np.testing.assert_allclose(t[:3, :3]@[0, 1, 0], [0, 0, 1], atol=1e-12)

    def test_root_reroot_roundtrip(self):
        fk = tree().fk([[.3], [.9]], ['j'])
        w = K.world_pose([[2, 3, 4]]*2, [[1, 0, 0, 0]]*2, 'wxyz')
        world = K.at_world_anchor(fk, w, 'b')
        np.testing.assert_allclose(world['b'], w, atol=1e-12)
        np.testing.assert_allclose(world['a']@fk['b'], w, atol=1e-12)

    def test_inverse_rigid(self):
        t = K.world_pose([[1, 2, 3]], [[.5, .5, .5, .5]], 'wxyz')
        np.testing.assert_allclose(t@K.inverse_rigid(t), np.eye(4)[None], atol=1e-12)

    def test_nonzero_points_rotation(self):
        t = K.world_pose([[1, 2, 3]], [[0, 0, 0, 1]], 'wxyz')
        np.testing.assert_allclose(K.transform_points(t, [[1, 0, 0]]), [[[0, 2, 3]]], atol=1e-12)

    def test_quaternion_orders(self):
        w = K.world_pose([[0, 0, 0]], [[1, 0, 0, 0]], 'wxyz')
        x = K.world_pose([[0, 0, 0]], [[0, 0, 0, 1]], 'xyzw')
        np.testing.assert_allclose(w, x)

    def test_bad_quaternion(self):
        with self.assertRaises(ValueError):
            K.world_pose([[0, 0, 0]], [[1, 1, 0, 0]], 'wxyz')

    def test_implicit_quaternion_order_rejected(self):
        with self.assertRaises(ValueError):
            K.world_pose([[0, 0, 0]], [[1, 0, 0, 0]], 'auto')

    def test_names_and_values_strict(self):
        for values, names in [([[1]], ['other']), ([[np.nan]], ['j']), ([[1, 2]], ['j', 'j']),
                              ([[1, 2]], ['j', 'extra']), (np.zeros((0, 1)), ['j']),
                              (np.zeros((4097, 1)), ['j'])]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                tree().fk(values, names)

    def test_unsupported_joint(self):
        for kind in ('floating', 'planar', 'prismatic'):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                tree(kind)

    def test_mimic_joint(self):
        with self.assertRaises(ValueError):
            tree(extra='<mimic joint="x"/>')

    def test_zero_axis(self):
        with self.assertRaises(ValueError):
            tree(axis='0 0 0')

    def test_duplicate_link(self):
        with self.assertRaises(ValueError):
            K.RobotTree(b'<robot><link name="a"/><link name="a"/></robot>')

    def test_disconnected(self):
        with self.assertRaises(ValueError):
            K.RobotTree(b'<robot><link name="a"/><link name="b"/></robot>')

    def test_dtd(self):
        with self.assertRaises(ValueError):
            K.RobotTree(b'<!DOCTYPE robot><robot><link name="a"/></robot>')

    def test_cycle_in_disconnected_component(self):
        with self.assertRaises(ValueError):
            K.RobotTree(b'''<robot><link name="a"/><link name="b"/><link name="c"/>
            <joint name="j" type="fixed"><parent link="b"/><child link="c"/></joint>
            <joint name="k" type="fixed"><parent link="c"/><child link="b"/></joint></robot>''')

    def test_box_origin_and_corners(self):
        model = K.RobotTree(b'''<robot><link name="a"><collision><origin xyz="0 0 -1"/>
        <geometry><box size="2 4 6"/></geometry></collision></link></robot>''')
        probes = model.box_probes('a')
        self.assertEqual(probes.shape, (8, 3))
        np.testing.assert_allclose(probes.min(axis=0), [-1, -2, -4])
        np.testing.assert_allclose(probes.max(axis=0), [1, 2, 2])

    def test_no_invented_box(self):
        with self.assertRaises(ValueError):
            tree().box_probes('a')


class TerrainTests(unittest.TestCase):
    def setUp(self):
        self.tri = np.array([[[0, 0, 0], [1, 0, 1], [0, 1, 2]]], dtype=float)

    def test_sloped_surface(self):
        np.testing.assert_allclose(K.upper_envelope_height(self.tri, [[.2, .3], [0, 0], [1, 0]]), [.8, 0, 1])

    def test_upper_not_lower(self):
        stacked = np.concatenate([self.tri, self.tri+[0, 0, 3]])
        np.testing.assert_allclose(K.upper_envelope_height(stacked, [[.2, .3]]), [3.8])

    def test_reversed_winding(self):
        np.testing.assert_allclose(K.upper_envelope_height(self.tri[:, ::-1], [[.2, .3]]), [.8])

    def test_outside_not_zero(self):
        self.assertTrue(np.isnan(K.upper_envelope_height(self.tri, [[1, 1]])[0]))

    def test_vertical_no_fabricated_height(self):
        tri = np.array([[[0, 0, 0], [0, 0, 1], [0, 1, 2]]], dtype=float)
        self.assertTrue(np.isnan(K.upper_envelope_height(tri, [[0, .5]])[0]))

    def test_chunk_boundaries(self):
        np.testing.assert_allclose(K.upper_envelope_height(self.tri, [[.2, .3]]*137), [.8]*137)

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            K.upper_envelope_height(self.tri, [[np.nan, .2]])

    def test_invalid_binary_stl(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'broken.stl'
            path.write_bytes(b'0'*84)
            with self.assertRaises(ValueError):
                K.read_triangles(path)


class EntryPointTests(unittest.TestCase):
    def test_no_solve_flag(self):
        args = [item for name in ('source-urdf', 'target-urdf', 'motion', 'terrain', 'report')
                for item in ('--'+name, 'unused')]
        result = subprocess.run([sys.executable, str(ROOT/'retarget_dry_run.py'), *args, '--solve'],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unrecognized arguments: --solve', result.stderr)
        self.assertFalse('Traceback' in result.stderr)

    def test_existing_report_refuses_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'existing.json'
            path.write_text('preserve me')
            args = [item for name in ('source-urdf', 'target-urdf', 'motion', 'terrain')
                    for item in ('--'+name, 'nonexistent')]
            result = subprocess.run([sys.executable, str(ROOT/'retarget_dry_run.py'), *args,
                                     '--report', str(path)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Report already exists', result.stderr)
            self.assertEqual(path.read_text(), 'preserve me')

    def test_pinned_hash_fail_closed(self):
        spec = importlib.util.spec_from_file_location('retarget_entry_tested', ROOT/'retarget_dry_run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.require_hash(ROOT/'retarget_dry_run.py', '0'*64)


if __name__ == '__main__':
    unittest.main(verbosity=2)