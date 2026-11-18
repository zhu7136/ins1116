"""CPU-only startup ENTRY/PRELAUNCH/RUNNER barrier tests, no Isaac/solver/GPU.

Original train.py fixture is byte-identical to the pinned public upstream file.
Main-body tests execute the actual patched function AST with explicit fake
environment/runner dependencies, not a hand-written imitation of its sequence.
"""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import test_cpu as cpu_tests
import test_runtime_gate as runtime_tests

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load("startup_under_test", ROOT / "startup_gate.py")
P = load("patch_under_test", ROOT / "startup_patch.py")
I = load("install_gate_under_test", ROOT / "install_overlay.py")
ORIGINAL = (ROOT / "fixtures/upstream_train.py.txt").read_bytes()
PLANE = "Instinct-Perceptive-Shadowing-HUD04-Plane-v0"
TERRAIN = "Instinct-Perceptive-Shadowing-HUD04-Terrain-v0"


class CliTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.build = patch.object(S, "StartupGate")
        self.construct = self.build.start()
        self.addCleanup(self.build.stop)

    def reject(self, args):
        with self.assertRaises(ValueError): S.prelaunch(args, "train.py")
        self.construct.assert_not_called()

    def test_default_is_check_only(self):
        S.prelaunch(["--task", PLANE], "train.py")
        self.construct.assert_called_once_with("train.py", "plane", "check-only")

    def test_terrain_equals_syntax(self):
        S.prelaunch(["--task=" + TERRAIN, "--num_envs=2"], "train.py")
        self.construct.assert_called_once_with("train.py", "terrain", "check-only")

    def test_explicit_train_still_constructs_gate(self):
        os.environ["HUD04_STARTUP_MODE"] = "train"
        S.prelaunch(["--task", PLANE], "train.py")
        self.construct.assert_called_once_with("train.py", "plane", "train")

    def test_g1_no_gate(self):
        self.assertIsNone(S.prelaunch(["--task", "Instinct-G1-v0"], "train.py"))
        self.construct.assert_not_called()

    def test_missing_task(self): self.reject(["--task", PLANE, "--task"])
    def test_unknown_hud04_task(self): self.reject(["--task", "HUD04-unknown"])
    def test_play_is_not_training_alias(self): self.reject(["--task", PLANE.replace("-v0", "-Play-v0")])
    def test_task_abbreviation_rejected(self): self.reject(["--tas", PLANE])
    def test_duplicate_tasks_rejected(self): self.reject(["--task", PLANE, "--task", "G1"])
    def test_wrong_env_count(self): self.reject(["--task", PLANE, "--num_envs", "4096"])
    def test_duplicate_env_count(self): self.reject(["--task", PLANE, "--num_envs=2", "--num_envs=2"])
    def test_hydra_override(self): self.reject(["--task", PLANE, "env.actions.joint_pos.scale=1"])
    def test_distributed(self): self.reject(["--task", PLANE, "--distributed"])
    def test_video(self): self.reject(["--task", PLANE, "--video"])
    def test_debug(self): self.reject(["--task", PLANE, "--debug"])
    def test_local_rank(self):
        os.environ["LOCAL_RANK"] = "0"
        self.reject(["--task", PLANE])
    def test_no_skip_mode(self):
        os.environ["HUD04_STARTUP_MODE"] = "skip"
        self.reject(["--task", PLANE])


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="startup-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.upstream = Path(self.temp.name)
        self.train = self.upstream / P.TRAIN
        self.train.parent.mkdir(parents=True)
        self.train.write_bytes(ORIGINAL)
        package = self.upstream / "source/instinctlab/instinctlab"
        self.files = [(p, package / p.relative_to(ROOT / "overlay/instinctlab"), False)
                      for p in sorted((ROOT / "overlay/instinctlab").rglob("*.py"))]

    def install(self):
        changes = P.plan(self.upstream, self.files)
        for source, target, _ in self.files:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        P.apply(changes)
        return changes

    def test_upstream_fixture_exact_hash(self): self.assertEqual(P.sha(ORIGINAL), P.TRAIN_SHA256)

    def test_plan_no_writes(self):
        self.assertEqual(len(P.plan(self.upstream, self.files)), 4)
        self.assertEqual(self.train.read_bytes(), ORIGINAL)
        self.assertFalse(self.train.with_name("hud04_startup_gate.py").exists())

    def test_applied_identity_receipt(self):
        self.install()
        root, _ = S.installed_receipt(self.train)
        self.assertEqual(root, self.upstream)

    def test_reapply_exact_is_noop(self):
        self.install()
        with patch.object(P.__dict__.get("subprocess", subprocess), "check_output", return_value=ORIGINAL):
            self.assertTrue(all(x[2] for x in P.plan(self.upstream, self.files)))

    def test_changed_train_refused(self):
        self.train.write_bytes(ORIGINAL + b"\n# user change\n")
        with patch.object(subprocess, "check_output", return_value=ORIGINAL):
            with self.assertRaises(ValueError): P.plan(self.upstream, self.files)

    def test_partial_patch_refused(self):
        self.train.write_bytes(P.patch_train(ORIGINAL).replace(b"_hud04_gate.validate_environment(env)", b"pass"))
        with patch.object(subprocess, "check_output", return_value=ORIGINAL):
            with self.assertRaises(ValueError): P.plan(self.upstream, self.files)

    def test_different_helper_refused_before_write(self):
        self.train.with_name("hud04_startup_gate.py").write_text("# user file\n")
        with self.assertRaises(FileExistsError): P.plan(self.upstream, self.files)
        self.assertEqual(self.train.read_bytes(), ORIGINAL)

    def test_symlink_target_refused(self):
        self.train.with_name("hud04_startup_gate.py").symlink_to(self.train)
        with self.assertRaises(ValueError): P.plan(self.upstream, self.files)

    def test_tampered_helper_detected(self):
        self.install()
        self.train.with_name("hud04_runtime_gate.py").write_text("# changed\n")
        with self.assertRaises(ValueError): S.installed_receipt(self.train)

    def test_tampered_overlay_detected(self):
        self.install()
        self.files[0][1].write_text("# changed\n")
        with self.assertRaises(ValueError): S.installed_receipt(self.train)

    def test_actual_patched_entry_rejects_before_isaac_import(self):
        self.install()
        env = os.environ.copy()
        env.pop("HUD04_REFERENCE_MANIFEST", None)
        env.pop("LOCAL_RANK", None)
        result = subprocess.run([sys.executable, "-B", str(self.train), "--task", PLANE],
                                env=env, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HUD04 startup gate", result.stderr)
        self.assertNotIn("No module named 'isaaclab'", result.stderr)

    def test_prelaunch_precedes_AppLauncher_import(self):
        text = P.patch_train(ORIGINAL).decode()
        self.assertLess(text.index("_hud04_gate = _hud04_prelaunch"), text.index("from isaaclab.app"))

    def run_installer_main(self, apply):
        (self.upstream / "source/instinctlab/instinctlab").mkdir(parents=True, exist_ok=True)
        def git_response(args, **kwargs):
            return I.PIN + "\n" if "rev-parse" in args else ORIGINAL
        out = io.StringIO()
        args = ["install_overlay.py", "--upstream", str(self.upstream)] + (["--apply"] if apply else [])
        with patch.object(sys, "argv", args), patch.dict(I.AUDITED, {}, clear=True), \
             patch.object(subprocess, "check_output", side_effect=git_response), contextlib.redirect_stdout(out):
            I.main()
        return json.loads(out.getvalue())

    def test_main_plan_includes_gate_without_writes(self):
        report = self.run_installer_main(False)
        self.assertTrue(report["startup_gate_default_on"])
        self.assertEqual(self.train.read_bytes(), ORIGINAL)
        self.assertFalse(self.train.with_name("hud04_startup_install.json").exists())

    def test_main_apply_installs_gate_by_default_and_is_idempotent(self):
        report = self.run_installer_main(True)
        self.assertEqual(report["source_overwrites"], 1)
        self.assertIn(b"_hud04_gate.validate_environment(env)", self.train.read_bytes())
        S.installed_receipt(self.train)
        second = self.run_installer_main(True)
        self.assertEqual(second["source_overwrites"], 0)

    def test_missing_patch_anchor_rejected(self):
        with self.assertRaises(ValueError): P.patch_train(ORIGINAL.replace(b"    # create runner from instinct-rl\n", b""))


class RuntimeBarrierTests(unittest.TestCase):
    def setUp(self):
        # Reuse explicit fake managers from read-only gate tests, NOT an Isaac environment.
        fixture = runtime_tests.LiveHookFixtureTests()
        fixture.setUp()
        self.fixture = fixture
        self.wrapper, self.base = fixture.wrapper, fixture.base
        self.base.unwrapped = self.base
        self.events = []
        self.wrapper.reset = Mock(side_effect=lambda: (self.events.append("reset") or None,
                                            {"observations": fixture.observations}))
        self.wrapper.close = Mock(side_effect=lambda: self.events.append("close"))
        loader = type("HUD04AmassMotion", (), {"__module__": "instinctlab.motion_reference.motion_files.hud04_motion"})
        self.cfg = NS(path=str(ROOT), class_type=loader, motion_interpolate_func=None, skip_frames=0,
                      clip_joint_ref_to_robot_limits=False, ensure_link_below_zero_ground=False, motion_start_height_offset=0)
        Config = type("HUD04PerceptivePlaneEnvCfg", (), {
            "__module__": "instinctlab.tasks.shadowing.perceptive.config.hud04.perceptive_shadowing_cfg"})
        self.base.cfg = Config()
        self.base.cfg.scene = NS(num_envs=2, motion_reference=NS(motion_buffers={"AMASSMotion": self.cfg}))
        self.base.num_envs = 2
        self.base.cfg.sim = NS(device="cpu")
        self.gate = S.StartupGate.__new__(S.StartupGate)
        self.gate.__dict__.update(train_path=ROOT / "unused_train.py", kind="plane", mode="train",
            upstream=ROOT, receipt_path=ROOT / "startup_gate.py", receipt_digest=S.digest(ROOT / "startup_gate.py"),
            runtime=runtime_tests.gate, inputs={"synthetic": "unchanged"}, names=runtime_tests.NAMES,
            control_values=runtime_tests.control.control_values, sources={}, validated=False, wrapper=None)
        for active in (
            patch.object(S, "installed_receipt", return_value=(ROOT, ROOT / "startup_gate.py")),
            patch.object(S, "input_receipt", return_value=(self.gate.inputs, self.gate.names, self.gate.control_values, ROOT)),
            patch.object(self.gate.runtime, "selected_source_receipt", return_value={}),
            patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            active.__enter__()
            self.addCleanup(active.__exit__, None, None, None)

    def call_main(self):
        tree = ast.parse(P.patch_train(ORIGINAL))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        main.decorator_list = []
        for arg in main.args.args:
            arg.annotation = None
        calls = self.events
        class FakeRunner:
            def __init__(self, *a, **k): calls.append("runner")
            def add_git_repo_to_log(self, *a): pass
            def learn(self, **k): calls.append("learn")
        args = NS(num_envs=None, max_iterations=None, device=None, logroot=str(ROOT), video=False, cprofile=False,
                  task=PLANE)
        agent = NS(max_iterations=0, seed=0, run_name="", resume=False, device="cpu", to_dict=lambda: {})
        from datetime import datetime
        namespace = {"_hud04_gate": self.gate, "args_cli": args, "os": os, "datetime": datetime,
            "cli_args": NS(update_instinct_rl_cfg=lambda cfg, _: cfg), "__file__": "patched_train.py",
            "gym": NS(make=lambda *a, **k: self.wrapper), "DirectMARLEnv": type("Unused", (), {}),
            "InstinctRlVecEnvWrapper": lambda env: env, "OnPolicyRunner": FakeRunner, "dump_yaml": lambda *a: None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), "patched_train.py", "exec"), namespace)
        namespace["main"](self.base.cfg, agent)

    def rejected(self):
        with self.assertRaises((ValueError, KeyError, RuntimeError)): self.call_main()
        self.assertNotIn("runner", self.events)
        self.assertNotIn("learn", self.events)
        self.assertIn("close", self.events)

    def test_actual_main_passes_before_runner(self):
        self.call_main()
        self.assertEqual(self.events, ["reset", "runner", "learn", "close"])

    def test_terrain_full_task_uses_same_runner_barrier(self):
        self.gate.kind = "terrain"
        self.base.cfg.__class__ = type("HUD04PerceptiveTerrainEnvCfg", (), {
            "__module__": "instinctlab.tasks.shadowing.perceptive.config.hud04.terrain_shadowing_cfg"})
        self.cfg.class_type = type("HUD04TerrainMotion", (), {
            "__module__": "instinctlab.motion_reference.motion_files.hud04_motion"})
        self.base.cfg.scene.motion_reference.motion_buffers = {"TerrainMotion": self.cfg}
        self.call_main()
        self.assertEqual(self.events, ["reset", "runner", "learn", "close"])

    def test_nonexplicit_runtime_pass_blocks_runner(self):
        with patch.object(self.gate.runtime, "check_live_environment", return_value={}):
            self.rejected()

    def test_runner_permission_is_independently_required(self):
        with patch.object(self.gate, "validate_environment", return_value=None):
            with self.assertRaisesRegex(ValueError, "Runner construction not permitted"):
                self.call_main()
        self.assertNotIn("runner", self.events)

    def test_check_only_cleanup_failure_is_not_success(self):
        self.gate.mode = "check-only"
        self.wrapper.close.side_effect = RuntimeError("close failed")
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.call_main()
        self.assertNotIn("runner", self.events)

    def test_close_environment_is_idempotent(self):
        self.gate.wrapper = self.wrapper
        self.gate.close_environment()
        self.gate.close_environment()
        self.wrapper.close.assert_called_once()

    def test_normal_completion_finalizer_does_not_double_close(self):
        self.call_main()
        self.gate.close_environment()
        self.wrapper.close.assert_called_once()

    def test_application_finalizer_clears_context_before_close(self):
        app = NS(close=Mock(side_effect=lambda **k: self.events.append(("app_close", k))))
        context = NS(clear_instance=lambda: self.events.append("clear_instance"))
        with patch.dict(sys.modules, {"isaaclab.sim": NS(SimulationContext=context)}):
            self.gate.close_application(app)
        self.assertEqual(self.events, ["clear_instance", ("app_close", {"wait_for_replicator": False})])

    def test_actual_entry_finally_runs_on_environment_construction_failure(self):
        tree = ast.parse(P.patch_train(ORIGINAL))
        tail = tree.body[-1]
        close = Mock()
        namespace = {"__name__": "__main__", "_hud04_gate": NS(close_application=close),
            "simulation_app": object(), "main": Mock(side_effect=RuntimeError("construction failed"))}
        with self.assertRaisesRegex(RuntimeError, "construction failed"):
            exec(compile(ast.Module(body=[tail], type_ignores=[]), "patched_train.py", "exec"), namespace)
        close.assert_called_once_with(namespace["simulation_app"])

    def test_default_check_only_exits_before_runner(self):
        self.gate.mode = "check-only"
        with self.assertRaises(SystemExit) as exc: self.call_main()
        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(self.events, ["reset", "close"])

    def test_head_or_order_failure_blocks_runner(self):
        self.fixture.term._joint_ids = list(reversed(range(29)))
        self.rejected()

    def test_depth_layout_failure_blocks_runner(self):
        self.wrapper.get_obs_format()["policy"]["depth_image"] = (18, 32, 1)
        self.rejected()

    def test_selected_loaded_source_mismatch_blocks_runner(self):
        self.gate.runtime.selected_source_receipt.side_effect = ValueError("wrong installed source")
        self.rejected()

    def test_prelaunch_loaded_module_path_drift_blocks_runner(self):
        self.gate.sources = {"module": ROOT / "wrong.py"}
        self.rejected()

    def test_reference_input_changes_block_runner(self):
        S.input_receipt.return_value = ({"synthetic": "changed"}, [], None, ROOT)
        self.rejected()
        self.wrapper.reset.assert_not_called()

    def test_reference_loader_replaced_blocks_runner(self):
        self.cfg.class_type = type("AmassMotion", (), {})
        self.rejected()

    def test_reference_path_changed_blocks_runner(self):
        self.cfg.path = str(ROOT / "elsewhere")
        self.rejected()

    def test_reference_clipping_blocks_runner(self):
        self.cfg.clip_joint_ref_to_robot_limits = True
        self.rejected()

    def test_reset_failure_blocks_runner(self):
        self.wrapper.reset.side_effect = RuntimeError("reset failure")
        self.rejected()

    def test_missing_reset_observations_blocks_runner(self):
        self.wrapper.reset.side_effect = None
        self.wrapper.reset.return_value = (None, {})
        self.rejected()

    def test_wrong_task_config_blocks_runner(self):
        self.base.cfg.__class__ = type("WrongConfig", (), {})
        self.rejected()

    def test_missing_accepted_input_blocks_before_AppLauncher(self):
        with patch.object(S, "installed_receipt", return_value=(ROOT, ROOT)), \
             patch.object(S, "load_file", return_value=NS(SOURCE_PINS={})), \
             patch.object(S, "check_source_paths", return_value={}), \
             patch.object(S, "input_receipt", side_effect=ValueError("no accepted reference")):
            with self.assertRaisesRegex(ValueError, "no accepted reference"):
                S.StartupGate("train.py", "plane", "check-only")


class SourceResolutionTests(unittest.TestCase):
    def test_find_source_does_not_execute_parent_package(self):
        with tempfile.TemporaryDirectory(prefix="resolve-test-", dir=ROOT) as temp:
            root = Path(temp)
            package = root / "synthetic_gate_source"
            package.mkdir()
            (package / "__init__.py").write_text("raise AssertionError('parent imported')\n")
            source = package / "leaf.py"
            source.write_text("raise AssertionError('leaf imported')\n")
            with patch.object(sys, "path", [str(root)] + sys.path):
                self.assertEqual(S.resolve_source("synthetic_gate_source.leaf"), source)
                self.assertEqual(S.check_source_paths({"synthetic_gate_source.leaf": S.digest(source)}),
                                 {"synthetic_gate_source.leaf": source})
            self.assertNotIn("synthetic_gate_source", sys.modules)

    def test_mismatching_prelaunch_source_fails(self):
        with patch.object(S, "resolve_source", return_value=ROOT / "startup_gate.py"):
            with self.assertRaisesRegex(ValueError, "source mismatch before AppLauncher"):
                S.check_source_paths({"synthetic": "wrong"})


class CpuInputTests(unittest.TestCase):
    """Temporary SYNTHETIC acceptance DECLARATIONS test software, NOT physical approval."""
    def setUp(self):
        fixture = cpu_tests.ReferenceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        fixture.arrays["joint_pos"] = np.tile(fixture.arrays["joint_pos"][0], (997, 1))
        fixture.arrays["base_pos_w"] = np.tile(fixture.arrays["base_pos_w"][0], (997, 1))
        fixture.arrays["base_quat_w"] = np.tile(fixture.arrays["base_quat_w"][0], (997, 1))
        fixture.arrays["framerate"] = np.array(90.)
        fixture.validate()
        temp = tempfile.TemporaryDirectory(prefix="cpu-input-test-", dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.upstream = Path(temp.name)
        assets = self.upstream / S.PREFIX / "assets"
        assets.mkdir(parents=True)
        for filename in ("hud04_contract.py", "hud04_motion.py"):
            (assets / filename).write_bytes((ROOT / "overlay/instinctlab/assets" / filename).read_bytes())
        env = patch.dict(os.environ, {
            "HUD04_REFERENCE_MANIFEST": str(fixture.path),
            "HUD04_DELIVERY_ROOT": str(cpu_tests.DELIVERY),
            "HUMANOID_DESCRIPTION_DIR": str(cpu_tests.URDF.parents[2]),
            "HUD04_CAMERA_PROFILE": "sim_chest_35deg_v1"})
        env.start()
        self.addCleanup(env.stop)

    def check(self):
        return S.input_receipt(self.upstream, "plane")

    def test_actual_cpu_input_preflight_does_not_import_isaac_asset(self):
        import builtins
        original = builtins.__import__
        def reject_sim_import(name, *a, **kw):
            if name.split(".")[0] in ("isaaclab", "omni", "pxr"):
                raise AssertionError("Pre-AppLauncher simulator import: " + name)
            return original(name, *a, **kw)
        with patch.object(builtins, "__import__", side_effect=reject_sim_import):
            files, names, values, root = self.check()
        self.assertEqual(len(names), 29)
        self.assertEqual(root, self.fixture.root)
        self.assertIn(str(cpu_tests.URDF.resolve()), files)
        self.assertEqual(len(values(names[0])), 4)

    def test_candidate_is_rejected_by_actual_manifest_gate(self):
        self.fixture.manifest["status"] = "candidate"
        self.fixture.path.write_text(json.dumps(self.fixture.manifest))
        with self.assertRaisesRegex(ValueError, "acceptance mismatch"): self.check()

    def test_short_reference_rejected_by_actual_strict_loader(self):
        for key in ("joint_pos", "base_pos_w", "base_quat_w"):
            self.fixture.arrays[key] = self.fixture.arrays[key][:4]
        self.fixture.validate()
        with self.assertRaisesRegex(ValueError, "997-frame"): self.check()

    def test_missing_manifest_rejected_without_fallback(self):
        with patch.dict(os.environ, {"HUD04_REFERENCE_MANIFEST": ""}):
            with self.assertRaisesRegex(ValueError, "no fallback"): self.check()

    def test_camera_missing_profile_or_calibration_rejected(self):
        with patch.dict(os.environ, {"HUD04_CAMERA_PROFILE": "manual", "HUD04_CAMERA_POS": "", "HUD04_CAMERA_QUAT_WXYZ": ""}):
            with self.assertRaises(ValueError): self.check()

    def test_terrain_does_not_accept_plane_manifest(self):
        with self.assertRaisesRegex(ValueError, "acceptance mismatch"):
            S.input_receipt(self.upstream, "terrain")


if __name__ == "__main__":
    unittest.main(verbosity=2)