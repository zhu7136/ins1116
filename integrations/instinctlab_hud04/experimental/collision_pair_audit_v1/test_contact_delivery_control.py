"""Software/AST regressions only; no simulator or physics evidence."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest
import contact_delivery_control as trial


class ControlTests(unittest.TestCase):
    def fixture(self):
        return dict(matched_contact_point_counts={'isolated_sphere_pair':1},binding_error=None,error=None),dict(error=None,completed_steps=8)

    def test_default_plan_only(self):
        p=subprocess.run([sys.executable,trial.__file__],capture_output=True,text=True,timeout=5)
        self.assertEqual(p.returncode,0)
        v=json.loads(p.stdout);self.assertEqual(v['phase'],'plan_only')
        self.assertFalse(v['platform_hard_cap_verified']);self.assertFalse(v['training_ready'])

    def test_bounded_one_trial(self):
        self.assertEqual(trial.CHILD_SECONDS,285);self.assertEqual(trial.MAX_STEPS,8)
        self.assertEqual(len(trial.REGISTRY),1)

    def test_missing_execution_args_rejected(self):
        p=subprocess.run([sys.executable,trial.__file__,'--execute'],capture_output=True,text=True,timeout=5)
        self.assertEqual(p.returncode,2)

    def test_delivery_does_not_promote_robot_or_friction(self):
        c,p=self.fixture();r=trial.delivery_decision(c,p)
        self.assertTrue(r['diagnostic_delivery_observed'])
        self.assertFalse(r['accepted_reference']);self.assertFalse(r['training_ready'])
        self.assertFalse(r['robot_support_validated']);self.assertIsNone(r['effective_pair_friction'])

    def test_zero_and_unrelated_points_fail(self):
        c,p=self.fixture();c['matched_contact_point_counts']={'unrelated':99}
        self.assertFalse(trial.delivery_decision(c,p)['diagnostic_delivery_observed'])

    def test_binding_and_ingestion_errors_fail(self):
        for key in ('binding_error','error'):
            c,p=self.fixture();c[key]={'message':'incompatible'}
            self.assertFalse(trial.delivery_decision(c,p)['diagnostic_delivery_observed'])

    def test_phase_error_or_incomplete_steps_fail(self):
        c,p=self.fixture();p['error']='outside_window'
        self.assertFalse(trial.delivery_decision(c,p)['diagnostic_delivery_observed'])
        c,p=self.fixture();p['completed_steps']=7
        self.assertFalse(trial.delivery_decision(c,p)['diagnostic_delivery_observed'])

    def test_subscription_precedes_reset(self):
        text=Path(trial.__file__).read_text()
        self.assertLess(text.index('holder=capture.subscribe_existing('),text.index('sim.reset()'))
        self.assertIn("phases['current']='initialization'",text)
        self.assertIn("processing_after_reset is not False",text)

    def test_import_has_no_sdk_or_execution_side_effects(self):
        tree=ast.parse(Path(trial.__file__).read_text())
        for n in tree.body:
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                self.assertFalse(any(s in ast.unparse(n) for s in ('isaaclab','omni.physx','pxr')))
        self.assertFalse(any(isinstance(n,ast.Call) and ast.unparse(n.func).endswith('.learn') for n in ast.walk(tree)))


if __name__=='__main__':unittest.main()