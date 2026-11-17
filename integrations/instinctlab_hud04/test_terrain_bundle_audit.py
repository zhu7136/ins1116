"""Synthetic CPU tests; no simulator, accepted references or external service calls."""
import importlib.util
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
DELIVERY = ROOT.parents[1]
os.environ['HUD04_DELIVERY_ROOT'] = str(DELIVERY)
SPEC = importlib.util.spec_from_file_location('terrain_audit_tested', ROOT / 'audit_terrain_bundle.py')
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)
URDF = DELIVERY / 'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'


class TerrainBundleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='terrain-audit-', dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        names, model = T.A.C.validate_model(URDF)
        self.arrays = dict(joint_names=np.array(names),
            joint_pos=np.tile([model[n]['default'] for n in names], (4, 1)),
            base_pos_w=np.tile([0., 0., .9], (4, 1)),
            base_quat_w=np.tile([1., 0., 0., 0.], (4, 1)), framerate=np.array(50.))
        self.metadata = dict(motion_files=[dict(motion_file='synthetic_retargeted.npz', terrain_id=0)],
                             terrains=[dict(terrain_file='test.stl', terrain_id=0)])
        self.mesh((0, 0, 0, 1, 0, 0, 0, 1, .5))

    def mesh(self, vertices):
        (self.root / 'test.stl').write_bytes(b'synthetic fixture'.ljust(80, b'\0') +
            struct.pack('<I', 1) + struct.pack('<12fH', 0, 0, 1, *vertices, 0))

    def run_audit(self, tile=(9, 12)):
        np.savez(self.root / 'synthetic_retargeted.npz', **self.arrays)
        (self.root / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        return T.audit_bundle(self.root, 'metadata.yaml', URDF, tile)

    def test_static_pass_is_never_acceptance(self):
        report = self.run_audit()
        self.assertTrue(report['static_loader_contract_pass'])
        self.assertTrue(report['all_referenced_motion_numerics_pass'])
        for key in ('accepted_reference', 'training_ready', 'physical_replay_verified',
                    'terrain_frame_or_units_verified', 'source_data_modified'):
            self.assertFalse(report[key])
        self.assertEqual(report['gpu_submissions'], 0)
        self.assertEqual(report['solver_calls'], 0)

    def test_noncontiguous_ids_refused_even_with_matching_references(self):
        self.metadata['motion_files'][0]['terrain_id'] = 7
        self.metadata['terrains'][0]['terrain_id'] = 7
        report = self.run_audit()
        self.assertTrue(report['metadata']['file_and_id_links_resolve'])
        self.assertFalse(report['static_loader_contract_pass'])
        self.assertIn('terrain_ids_must_equal_list_indices_for_pinned_match_scene', report['loader_contract_issues'])

    def test_reordered_ids_refused(self):
        self.metadata['terrains'] = [dict(terrain_file='test.stl', terrain_id=1),
                                     dict(terrain_file='test.stl', terrain_id=0)]
        self.assertFalse(self.run_audit()['static_loader_contract_pass'])

    def test_missing_reference_is_reported_without_loading(self):
        self.metadata['motion_files'][0]['motion_file'] = 'absent.npz'
        report = self.run_audit()
        self.assertFalse(report['static_loader_contract_pass'])
        self.assertFalse(report['all_referenced_motion_numerics_pass'])
        self.assertEqual(report['motions'], [])

    def test_sampling_weights_invalid_values(self):
        for weight in (-1, float('nan'), float('inf'), True, '1'):
            with self.subTest(weight=weight):
                self.metadata['motion_files'][0]['weight'] = weight
                report = self.run_audit()
                self.assertFalse(report['static_loader_contract_pass'])

    def test_zero_weight_total_refused(self):
        self.metadata['motion_files'][0]['weight'] = 0
        self.assertIn('motion_weights_have_no_positive_mass', self.run_audit()['loader_contract_issues'])

    def test_invalid_crop_dimensions_refused(self):
        for tile in ((0, 12), (-1, 12), (float('nan'), 12), (float('inf'), 12), (9,)):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                self.run_audit(tile)

    def test_raw_root_crop_counts_are_conditional_only(self):
        self.arrays['base_pos_w'][1, :2] = (0, -6.3)
        report = self.run_audit()
        self.assertEqual(report['motions'][0]['root_frames_outside_raw_crop'], 1)
        self.assertTrue(report['crop_diagnostics_assume_common_raw_frame_and_units'])
        self.assertFalse(report['root_frame_verified'])

    def test_entire_triangle_beyond_one_crop_plane(self):
        self.mesh((0, -7, 0, 1, -7, 0, 0, -8, .5))
        terrain = self.run_audit()['terrains'][0]
        self.assertEqual(terrain['vertex_occurrences_outside_raw_crop'], 3)
        self.assertEqual(terrain['triangles_entirely_beyond_one_raw_crop_plane'], 1)
        self.assertFalse(terrain['exact_clipped_mesh_computed'])

    def test_spanning_triangle_not_falsely_claimed_removed(self):
        self.mesh((-5, 0, 0, 5, 0, 0, 0, 7, .5))
        terrain = self.run_audit()['terrains'][0]
        self.assertEqual(terrain['triangles_touching_outside_raw_crop'], 1)
        self.assertEqual(terrain['triangles_entirely_beyond_one_raw_crop_plane'], 0)

    def test_motion_numeric_failure_does_not_hide_structural_pass(self):
        self.arrays['joint_pos'][2, 0] = 100
        report = self.run_audit()
        self.assertTrue(report['static_loader_contract_pass'])
        self.assertFalse(report['all_referenced_motion_numerics_pass'])
        self.assertFalse(report['accepted_reference'])

    def test_duplicate_yaml_keys_refused(self):
        path = self.root / 'metadata.yaml'
        path.write_text('terrains: []\nterrains: []\nmotion_files: []\n')
        with self.assertRaisesRegex(ValueError, 'unique string'):
            T.audit_bundle(self.root, 'metadata.yaml', URDF, (9, 12))

    def test_escape_and_duplicate_terrain_ids_refused(self):
        self.metadata['motion_files'][0]['motion_file'] = '../outside.npz'
        with self.assertRaises(ValueError):
            self.run_audit()
        self.metadata['motion_files'][0]['motion_file'] = 'synthetic_retargeted.npz'
        self.metadata['terrains'].append(dict(terrain_file='test.stl', terrain_id=0))
        with self.assertRaisesRegex(ValueError, 'Duplicate terrain id'):
            self.run_audit()

    def test_inputs_unchanged(self):
        self.run_audit()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        T.audit_bundle(self.root, 'metadata.yaml', URDF, (9, 12))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})


if __name__ == '__main__':
    unittest.main()