"""Synthetic software checks only. No PhysX, contact or balance evidence."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from control_contract import (FEET, control_values, finite_vector, foot_box_bottoms,
                              permutation, position_targets, proprioception, read_model)
from worker import evidence, verify_evidence

ROOT = Path(__file__).resolve().parent
URDF = ROOT.parent/'hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
# Actual ordering from qnl7cmm4 report.json at verified commit 3d5b43b6.
RUNTIME = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint',
 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint',
 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint',
 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint']


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names, cls.model = read_model(URDF)

    def test_29_body_no_head(self):
        self.assertEqual(len(self.names), 29)
        self.assertFalse(any('head' in n for n in self.names))

    def test_real_order_is_not_urdf_order(self):
        self.assertNotEqual(self.names, RUNTIME)
        self.assertEqual(set(self.names), set(RUNTIME))

    def test_verified_order_roundtrip(self):
        values = np.arange(29)
        a = values[permutation(self.names, RUNTIME)]
        np.testing.assert_array_equal(a[permutation(RUNTIME, self.names)], values)

    def test_random_order_roundtrips(self):
        rng = np.random.default_rng(1116)
        for _ in range(20):
            destination = rng.permutation(self.names).tolist()
            values = rng.normal(size=29)
            np.testing.assert_array_equal(values[permutation(self.names, destination)]
                [permutation(destination, self.names)], values)

    def test_duplicate_names_rejected(self):
        bad = self.names.copy(); bad[-1] = bad[0]
        with self.assertRaises(ValueError):
            permutation(self.names, bad)

    def test_head_column_rejected(self):
        bad = self.names.copy(); bad[-1] = 'head_pitch_joint'
        with self.assertRaises(ValueError):
            permutation(self.names, bad)

    def test_extra_and_missing_rejected(self):
        for names in [self.names[:-1], self.names+['head_yaw_joint']]:
            with self.assertRaises(ValueError):
                permutation(names, self.names)

    def test_zero_action_is_provisional_default(self):
        np.testing.assert_array_equal(position_targets(np.zeros(29), RUNTIME, self.model),
            [self.model[n]['default'] for n in RUNTIME])

    def test_all_coordinate_signs_and_scales(self):
        zero = position_targets(np.zeros(29), RUNTIME, self.model)
        for i, name in enumerate(RUNTIME):
            for sign in [-1, 1]:
                action = np.zeros(29); action[i] = sign*.05
                delta = position_targets(action, RUNTIME, self.model)-zero
                expected = np.zeros(29); expected[i] = sign*.05*self.model[name]['scale']
                np.testing.assert_allclose(delta, expected, atol=1e-15)

    def test_default_offsets_are_runtime_configurable(self):
        defaults = np.linspace(-.01, .01, 29)
        np.testing.assert_array_equal(position_targets(np.zeros(29), RUNTIME, self.model, defaults), defaults)

    def test_nan_inf_wrong_shapes_rejected(self):
        for value in [np.full(29, np.nan), np.full(29, np.inf), np.zeros((1, 29)), np.zeros(28)]:
            with self.assertRaises(ValueError):
                finite_vector(value)

    def test_no_silent_action_clipping(self):
        target = position_targets(np.full(29, 1000.), RUNTIME, self.model)
        self.assertTrue(any(target[i] > self.model[n]['upper'] for i, n in enumerate(RUNTIME)))

    def test_observations_last_action_raw(self):
        a = np.full(29, .05)
        q = position_targets(a, RUNTIME, self.model)
        obs = proprioception(q, np.zeros(29), a, RUNTIME, self.model)
        np.testing.assert_array_equal(obs['last_action'], a)
        np.testing.assert_allclose(obs['joint_pos'], [self.model[n]['scale']*.05 for n in RUNTIME])

    def test_nominal_observation_zero(self):
        q = position_targets(np.zeros(29), RUNTIME, self.model)
        for value in proprioception(q, np.zeros(29), np.zeros(29), RUNTIME, self.model).values():
            np.testing.assert_array_equal(value, np.zeros(29))

    def test_forward_kinematics_spawn_clearance(self):
        bottoms = foot_box_bottoms(URDF)
        self.assertEqual(set(bottoms), set(FEET))
        height = .01-min(bottoms.values())
        self.assertTrue(.5 < height < 1.2)
        self.assertAlmostEqual(min(height+v for v in bottoms.values()), .01)
        self.assertAlmostEqual(bottoms[FEET[0]], bottoms[FEET[1]], places=5)

    def test_control_values_match_verified_asset_source(self):
        # AST extracts only the existing pure function, no Isaac import.
        tree = ast.parse((ROOT.parent/'hud04_asset_camera_probe/asset_config.py').read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_control_values')
        namespace = {}
        exec(compile(ast.Module(body=[func], type_ignores=[]), '<verified-function>', 'exec'), namespace)
        for name in self.names:
            self.assertEqual(control_values(name), namespace['_control_values'](name))

    def test_unknown_actuator_rejected(self):
        with self.assertRaises(ValueError):
            control_values('unknown')

    def test_report_evidence_roundtrip_and_tamper(self):
        report = {'training_ready': False, 'episodes': [{'fall': True}]}
        item = evidence(report)
        self.assertEqual(verify_evidence(json.loads(json.dumps(item))), report)
        changed = copy.deepcopy(item); changed['report']['training_ready'] = True
        with self.assertRaises(ValueError):
            verify_evidence(changed)

    def test_evidence_rejects_large_nonfinite(self):
        for report in [{'a': 'x'*33000}, {'a': float('nan')}]:
            with self.assertRaises(ValueError):
                evidence(report)

    def test_default_entrypoints_plan_only(self):
        for file in ['probe.py', 'worker.py']:
            result = subprocess.run([sys.executable, str(ROOT/file)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertFalse(data['gpu_started'])
            self.assertEqual(data['training_updates'], 0)

    def test_new_sources_parse_without_isaac_import(self):
        for file in ['probe.py', 'worker.py', 'control_contract.py']:
            ast.parse((ROOT/file).read_text())
        self.assertNotIn('isaaclab', sys.modules)
        self.assertNotIn('osqp', sys.modules)

    def test_source_manifest(self):
        manifest = json.loads((ROOT/'manifest.json').read_text())
        self.assertEqual(set(manifest['files']), {'probe.py', 'worker.py', 'control_contract.py'})
        for name, digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest)

    def test_execute_without_commit_fails_before_gpu(self):
        result = subprocess.run([sys.executable, str(ROOT/'worker.py'), '--execute-probe'],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Explicit expected commit required', result.stderr)
        self.assertNotIn('isaac_initialization_started', result.stdout)
        self.assertFalse((ROOT/'floating_probe.log').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)