"""CPU geometry/contract tests. Synthetic supports never certify real dynamics."""
import copy
import json
import os
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
import reconstruction as R
import verify_reconstruction as V

SOURCE = Path(os.environ.get('HUD04_SOURCE_MODEL', '/workspace/InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf'))


class FrameTests(unittest.TestCase):
    def test_frame_rotation_and_position(self):
        t = R.frame([1, 2, 3], [0, 0, -2], [1, 0, 0])
        np.testing.assert_allclose(t[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(t[:3, :3].T@t[:3, :3], np.eye(3))
        self.assertAlmostEqual(np.linalg.det(t[:3, :3]), 1)

    def test_zero_normal_rejected(self):
        with self.assertRaises(ValueError): R.frame([0]*3, [0]*3, [1, 0, 0])

    def test_parallel_tangent_rejected(self):
        with self.assertRaises(ValueError): R.frame([0]*3, [0, 0, 1], [0, 0, -1])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError): R.frame([np.nan]*3, [0, 0, 1], [1, 0, 0])

    def test_bad_dimension_rejected(self):
        with self.assertRaises(ValueError): R.frame([0, 1], [0, 0, 1], [1, 0, 0])

    def test_no_input_mutation(self):
        n = np.array([0., 0., 2.]); x = np.array([1., 0., 1.])
        R.frame([0]*3, n, x)
        np.testing.assert_array_equal(n, [0, 0, 2])
        np.testing.assert_array_equal(x, [1, 0, 1])


class ScheduleTests(unittest.TestCase):
    def test_run_inclusive_and_minimum(self):
        self.assertEqual(R.runs(np.array([True, True, False, True, False]), 2), [(0, 1)])

    def test_empty_runs(self):
        self.assertEqual(R.runs(np.zeros(997, bool), 4), [])

    def test_all_run(self):
        self.assertEqual(R.runs(np.ones(997, bool), 4), [(0, 996)])

    def test_integer_mask_rejected(self):
        with self.assertRaises(ValueError): R.runs(np.array([1, 0]), 1)

    def test_minimum_rejected(self):
        with self.assertRaises(ValueError): R.runs(np.ones(9, bool), 0)

    def plane_case(self, height=.01, normal=(0, 0, -1)):
        self.triangle = np.array([[[-5., -5., 0.], [5., -5., 0.], [0., 5., 0.]]])
        p = np.zeros((997, 3)); p[:, 2] = height
        n = np.tile(normal, (997, 1))
        return p, n, R.SurfaceIndex(self.triangle), np.array([[0., 0., 1.]])

    def test_stationary_near_surface_is_only_proposal(self):
        result, detail = R.propose_windows(*self.plane_case(), R.RULES)
        self.assertEqual([(x['first'], x['last']) for x in result['proposed_windows']], [(0, 996)])
        self.assertFalse(result['observed_contact']); self.assertFalse(result['validated_support'])
        np.testing.assert_allclose(detail[0], .01)

    def test_far_surface_no_support(self):
        result, _ = R.propose_windows(*self.plane_case(height=.1), R.RULES)
        self.assertEqual(result['proposed_windows'], [])

    def test_wrong_material_side_no_support(self):
        result, _ = R.propose_windows(*self.plane_case(normal=(0, 0, 1)), R.RULES)
        self.assertEqual(result['proposed_windows'], [])

    def test_ambiguous_edge_normal_is_not_support(self):
        triangles = np.array([[[-1., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                              [[-1., 0., 0.], [0., -1., .17632698], [1., 0., 0.]]])
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        result, detail = R.propose_windows(np.zeros((997, 3)), np.tile([0., 0., -1.], (997, 1)),
                                          R.SurfaceIndex(triangles), normals, R.RULES)
        self.assertEqual(result['proposed_windows'], [])
        self.assertEqual(result['ambiguous_nearest_surface_normals_frames'], 997)
        self.assertTrue(np.all(detail[-1]))

    def test_high_speed_not_support(self):
        p, n, index, normals = self.plane_case(); p[:, 0] = np.arange(997)*.004
        result, _ = R.propose_windows(p, n, index, normals, R.RULES)
        self.assertEqual(result['proposed_windows'], [])

    def test_short_contact_not_promoted(self):
        p, n, index, normals = self.plane_case(); p[:, 2] = .1; p[10:13, 2] = .01
        result, _ = R.propose_windows(p, n, index, normals, R.RULES)
        self.assertEqual(result['proposed_windows'], [])

    def test_slow_drift_without_fixed_anchor_rejected(self):
        p, n, index, normals = self.plane_case(); p[:, 0] = np.linspace(0, .2, 997)
        result, _ = R.propose_windows(p, n, index, normals, R.RULES)
        self.assertEqual(result['proposed_windows'], []); self.assertEqual(len(result['rejected_windows']), 1)

    def test_unknown_threshold_changes_rejected(self):
        rules = dict(R.RULES, max_surface_distance_m=.2)
        with self.assertRaises(ValueError): R.propose_windows(*self.plane_case(), rules)

    def test_nonfinite_points_rejected(self):
        p, n, index, normals = self.plane_case(); p[3, 2] = np.nan
        with self.assertRaises(ValueError): R.propose_windows(p, n, index, normals, R.RULES)

    def test_partial_frames_rejected(self):
        p, n, index, normals = self.plane_case()
        with self.assertRaises(ValueError): R.propose_windows(p[:90], n[:90], index, normals, R.RULES)

    def test_nonunit_normals_rejected(self):
        p, n, index, normals = self.plane_case()
        with self.assertRaises(ValueError): R.propose_windows(p, n*2, index, normals, R.RULES)


class RealMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = (R.sha(SOURCE), R.sha(R.TARGET))
        cls.models, cls.frames, cls.rows = R.materials(SOURCE, R.TARGET)
        cls.target_xml = {l.get('name'): l for l in ET.parse(R.TARGET).getroot().findall('link')}

    def test_six_fixed_materials(self):
        self.assertEqual(set(self.frames), {s+'_'+k for s in ['left', 'right'] for k in ['sole', 'palm', 'knee']})

    def test_models_unchanged_and_hash_bound(self):
        self.assertEqual(self.before, (R.D.SOURCE_SHA, R.D.TARGET_SHA))
        self.assertEqual(self.before, (R.sha(SOURCE), R.sha(R.TARGET)))

    def test_29_body_no_head(self):
        self.assertEqual(len(self.models['target'].names), 29)
        self.assertFalse(any('head' in n for n in self.models['target'].names))

    def test_materials_not_link_pivots(self):
        for kinds in self.frames.values():
            for link, t in kinds.values(): self.assertGreater(np.linalg.norm(t[:3, 3]), .02)

    def test_material_membership_residuals(self):
        for row in self.rows:
            self.assertLess(row['source_membership_residual_m'], 1e-9)
            self.assertLess(row['target_membership_residual_m'], 1e-9)
            self.assertFalse(row['source_truth'])

    def test_zero_anatomical_axes_match(self):
        local = {kind: model.fk(np.zeros((1, 29)))[0] for kind, model in self.models.items()}
        for label, kinds in self.frames.items():
            bases = []
            for kind, (link, material) in kinds.items():
                anchor = local[kind][self.models[kind].anchor][0, :3, :3]
                bases.append(anchor.T@local[kind][link][0, :3, :3]@material[:3, :3])
            np.testing.assert_allclose(*bases, rtol=0, atol=1e-10)

    def test_sole_box_membership_independent(self):
        for side in ['left', 'right']:
            link, t = self.frames[side+'_sole']['target']
            _, origin, g = R.collision(self.target_xml[link], 0)
            p = (t[:3, 3]-origin[:3, 3])@origin[:3, :3]
            size = R.K.vector(g.get('size'))
            np.testing.assert_allclose(p, [0, 0, -size[2]/2], atol=1e-12)

    def test_knee_sphere_membership_independent(self):
        for row in self.rows:
            if not row['label'].endswith('knee'): continue
            _, origin, g = R.collision(self.target_xml[row['target_link']], row['target_collision_index'])
            p = np.array(row['target_T_link_material'])[:3, 3]
            self.assertAlmostEqual(np.linalg.norm(p-origin[:3, 3]), float(g.get('radius')), places=12)

    def test_hand_nominal_hull_membership_independent(self):
        for side in ['left', 'right']:
            link, t = self.frames[side+'_palm']['target']
            _, origin, g = R.collision(self.target_xml[link], 0)
            path = (R.TARGET.parent/g.get('filename')).resolve()
            v = np.unique(R.K.read_triangles(path).reshape(-1, 3), axis=0)*R.K.vector(g.get('scale'))
            equations = ConvexHull(v).equations
            p = (t[:3, 3]-origin[:3, 3])@origin[:3, :3]
            residual = equations[:, :3]@p+equations[:, 3]
            self.assertLessEqual(residual.max(), 1e-10)
            self.assertLess(abs(residual.max()), 1e-10)


class OutputDefinitionTests(unittest.TestCase):
    def doc(self):
        return json.loads((R.HERE/'RECONSTRUCTION_002/CONTRACT.json').read_text())

    def test_current_definition(self): V.check_definition(self.doc())

    def test_changed_root_rejected(self):
        c = self.doc(); c['scene']['source_root_link'] = 'pelvis'
        with self.assertRaises(ValueError): V.check_definition(c)

    def test_changed_quaternion_rejected(self):
        c = self.doc(); c['scene']['source_quaternion_order'] = 'xyzw'
        with self.assertRaises(ValueError): V.check_definition(c)

    def test_fitted_scene_transform_rejected(self):
        c = self.doc(); c['scene']['terrain_to_engineering_world'][0][3] = 1
        with self.assertRaises(ValueError): V.check_definition(c)

    def test_automatic_wrist_exemption_rejected(self):
        c = self.doc(); c['collision_policy']['automatic_pair_exclusions'] = ['left_wrist']
        with self.assertRaises(ValueError): V.check_definition(c)

    def test_threshold_relaxation_rejected(self):
        c = self.doc(); c['support_proposal_rules']['max_surface_distance_m'] = .15
        with self.assertRaises(ValueError): V.check_definition(c)

    def test_false_acceptance_rejected(self):
        c = self.doc(); c['acceptance']['accepted_reference'] = True
        with self.assertRaises(ValueError): V.check_definition(c)

    def test_false_source_provenance_rejected(self):
        c = self.doc(); c['source_provenance_recovered'] = True
        with self.assertRaises(ValueError): V.check_definition(c)


if __name__ == '__main__': unittest.main(verbosity=2)