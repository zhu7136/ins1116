import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import runtime_trial as trial


class TrialTests(unittest.TestCase):
    def test_budget_and_root(self):
        self.assertEqual(trial.MAX_STEPS, 8)
        self.assertLessEqual(trial.CHILD_SECONDS + 2, 300)
        self.assertTrue((trial.REPO / 'flux/hud04_asset_camera_probe').is_dir())

    def test_default_is_plan_only(self):
        p = subprocess.run([sys.executable, str(Path(trial.__file__))], capture_output=True, text=True, timeout=5)
        self.assertEqual(p.returncode, 0)
        r = json.loads(p.stdout)
        self.assertEqual(r['phase'], 'plan_only')
        self.assertEqual(r['ppo_updates'], 0)
        self.assertFalse(r['platform_hard_cap_verified'])

    def test_execute_requires_explicit_args(self):
        p = subprocess.run([sys.executable, str(Path(trial.__file__)), '--execute'], capture_output=True, text=True, timeout=5)
        self.assertEqual(p.returncode, 2)

    def test_no_top_level_simulator_or_training_import(self):
        module = ast.parse(Path(trial.__file__).read_text())
        imports = [n for n in module.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('isaac' in ast.unparse(n) or 'omni' in ast.unparse(n) for n in imports))
        calls = [ast.unparse(n.func) for n in ast.walk(module) if isinstance(n, ast.Call)]
        self.assertNotIn('learn', calls)
        self.assertNotIn('request_convex_collision_representation', calls)

    def test_writes_exclusively(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as name:
            path = Path(name) / 'result.json'
            trial.write(path, {'ok': True})
            self.assertEqual(json.loads(path.read_text()), {'ok': True})
            with self.assertRaises(FileExistsError):
                trial.write(path, {})

    def test_unique_event(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            trial.emit('test', ppo_updates=0)
        self.assertEqual(json.loads(out.getvalue())['stage'], 'hud04_runtime_trial')


if __name__ == '__main__':
    unittest.main()