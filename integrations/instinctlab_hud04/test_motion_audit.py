"""CPU software tests only; synthetic fixtures are not accepted reference motions."""
import importlib.util
import os
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent
DELIVERY = ROOT.parents[1]
os.environ['HUD04_DELIVERY_ROOT'] = str(DELIVERY)
SPEC = importlib.util.spec_from_file_location('motion_audit_tested', ROOT/'audit_motion.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
URDF = DELIVERY/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='audit-test-', dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names, self.model = A.C.validate_model(URDF)
        self.arrays = dict(joint_names=np.array(self.names),
            joint_pos=np.tile([self.model[n]['default'] for n in self.names], (4, 1)),
            base_pos_w=np.tile([0., 0., .9], (4, 1)),
            base_quat_w=np.tile([1., 0., 0., 0.], (4, 1)), framerate=np.array(50.))

    def save(self):
        file = self.root/'synthetic_retargeted.npz'
        np.savez(file, **self.arrays)
        return file

    def test_numeric_pass_never_means_acceptance(self):
        file = self.save()
        before = file.read_bytes()
        result = A.audit_dataset(self.root, URDF)
        self.assertTrue(result['motions'][0]['numerical_limits_pass'])
        self.assertFalse(result['accepted_reference'])
        self.assertFalse(result['training_ready'])
        self.assertFalse(result['motions'][0]['quaternion_order_verified'])
        self.assertEqual(file.read_bytes(), before)

    def test_position_and_speed_incompatibility(self):
        col = self.names.index('left_knee_joint')
        self.arrays['joint_pos'][2, col] = self.model['left_knee_joint']['upper'] + .2
        self.save()
        result = A.audit_dataset(self.root, URDF)['motions'][0]
        self.assertEqual(result['position_violating_frames'], 1)
        self.assertEqual(result['overspeed_intervals'], 2)
        self.assertAlmostEqual(result['peak_position_excess_rad'], .2)

    def test_object_arrays_refused_without_pickle(self):
        self.arrays['joint_names'] = self.arrays['joint_names'].astype(object)
        self.save()
        result = A.audit_dataset(self.root, URDF)
        self.assertEqual(len(result['errors']), 1)
        self.assertEqual(result['motions'], [])

    def test_head_or_wrong_name_is_not_silently_remapped(self):
        self.arrays['joint_names'][-1] = 'head_yaw_joint'
        self.save()
        self.assertEqual(len(A.audit_dataset(self.root, URDF)['errors']), 1)

    def test_missing_metadata_motion_detected(self):
        file = self.root/'metadata.yaml'
        file.write_text('motion_files:\n- motion_file: absent.npz\n  terrain_id: 0\nterrains:\n- terrain_file: absent.stl\n  terrain_id: 0\n')
        result = A.audit_metadata(file, self.root)
        self.assertFalse(result['file_and_id_links_resolve'])
        self.assertFalse(result['accepted_reference'])

    def test_metadata_escape_refused(self):
        with self.assertRaises(ValueError):
            A.contained(self.root, '../outside.npz')

    def test_binary_stl_bounds_are_not_assumed_metres(self):
        file = self.root/'test.stl'
        raw = b'synthetic fixture'.ljust(80, b'\0') + struct.pack('<I', 1)
        raw += struct.pack('<12fH', 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, .5, 0)
        file.write_bytes(raw)
        result = A.audit_binary_stl(file)
        self.assertEqual(result['triangle_count'], 1)
        self.assertEqual(result['bounds_raw_units'], [[0., 0., 0.], [1., 1., .5]])
        self.assertFalse(result['units_verified'])

    def test_reordered_motion_preserves_named_mapping(self):
        self.arrays['joint_names'] = self.arrays['joint_names'][::-1]
        self.arrays['joint_pos'] = self.arrays['joint_pos'][:, ::-1]
        self.save()
        self.assertTrue(A.audit_dataset(self.root, URDF)['motions'][0]['numerical_limits_pass'])


if __name__ == '__main__':
    unittest.main()