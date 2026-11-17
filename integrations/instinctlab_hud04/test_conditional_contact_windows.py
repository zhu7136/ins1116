"""Synthetic geometry tests, not simulation or real reference acceptance."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import audit_conditional_contact_windows as A


def capsule():
    return dict(kind='capsule',radius=.2,length=1.,transform=np.eye(4))


class ConditionalTests(unittest.TestCase):
    def test_identity_down(self):
        np.testing.assert_allclose(A.world_down_support(capsule(),np.eye(4)[None]),[[0,0,-.7]])

    def test_translation(self):
        t = np.eye(4)[None]
        t[0,:3,3] = [1,2,3]
        np.testing.assert_allclose(A.world_down_support(capsule(),t),[[1,2,2.3]])

    def test_rotated_capsule_analytic_minimum_z(self):
        t = np.eye(4)[None]
        angle = .7
        t[0,:3,:3] = [[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
        point = A.world_down_support(capsule(),t)
        self.assertAlmostEqual(point[0,2],-.2-.5*np.cos(angle))

    def test_bad_pose_shape_rejected(self):
        with self.assertRaises(ValueError):
            A.world_down_support(capsule(),np.eye(3)[None])

    def test_scale_and_reflection_rejected(self):
        for x in (2.,-1.):
            t = np.eye(4)[None]
            t[0,0,0] = x
            with self.subTest(x=x),self.assertRaises(ValueError):
                A.world_down_support(capsule(),t)

    def test_nonfinite_rejected(self):
        t = np.eye(4)[None]
        t[0,0,3] = np.nan
        with self.assertRaises(ValueError):
            A.world_down_support(capsule(),t)

    def test_no_height_outside_envelope(self):
        triangles = np.array([[[0,0,0],[1,0,0],[0,1,0]]],dtype=float)
        d = A.raw_gap_report([[2,2,1]],triangles)
        self.assertEqual(d['outside_raw_xy_envelope'],1)
        self.assertIsNone(d['vertical_gap_raw_units']['percentiles'])
        self.assertIsNone(d['contact_label'])

    def test_gap_counts_do_not_create_contact_labels(self):
        triangles = np.array([[[0,0,0],[1,0,0],[0,1,0]]],dtype=float)
        d = A.raw_gap_report([[.1,.1,-.01],[.1,.1,.01]],triangles)
        self.assertEqual(d['below_upper_envelope_by_more_than_0p005_raw_units'],1)
        self.assertEqual(d['abs_gap_within_0p020_raw_units'],2)
        self.assertFalse(d['signed_penetration_distance'])
        self.assertFalse(d['full_collision_test'])

    def test_existing_report_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'existing.json'
            p.write_text('keep')
            r = subprocess.run([sys.executable,'-B',A.__file__,'--source-urdf','missing',
                '--motion','missing','--terrain','missing','--report',str(p)],capture_output=True,timeout=10)
            self.assertNotEqual(r.returncode,0)
            self.assertEqual(p.read_text(),'keep')

    def test_model_pin_precedes_data_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'other.urdf'
            p.write_text('<robot/>')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                A.audit(p,'missing','missing')


if __name__ == '__main__':
    unittest.main()