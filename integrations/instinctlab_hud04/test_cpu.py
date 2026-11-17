"""Synthetic CPU tests only: no IsaacLab import, physics, GPU or real motions."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent
DELIVERY = Path(os.environ.get("HUD04_DELIVERY_ROOT", str(ROOT.parents[1])))
OVERLAY = ROOT / "overlay/instinctlab"
os.environ["HUD04_DELIVERY_ROOT"] = str(DELIVERY)
URDF = DELIVERY / "flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load("contract_under_test", OVERLAY / "assets/hud04_contract.py")
INSTALL = load("installer_under_test", ROOT / "install_overlay.py")
CFG = OVERLAY / "tasks/shadowing/perceptive/config/hud04/perceptive_shadowing_cfg.py"


class ModelTests(unittest.TestCase):
    def test_29_body_zero_head(self):
        names, _ = C.validate_model(URDF)
        self.assertEqual(len(names), 29)
        self.assertFalse(any("head" in n for n in names))

    def test_semantic_keys_exist_and_unique(self):
        self.assertEqual(len(C.KEY_LINKS), len(set(C.KEY_LINKS)))
        self.assertEqual(C.ROOT_LINK, "base_link")
        self.assertEqual(C.CHEST_LINK, "waist_pitch_link")

    def test_hand_anchors_are_downstream_roll(self):
        self.assertEqual(C.HANDS, ("left_wrist_roll_link", "right_wrist_roll_link"))

    def test_reuse_three_baseline_modules(self):
        for _, (relative, expected) in C.VERIFIED_MODULES.items():
            self.assertEqual(C.sha256(DELIVERY / relative), expected)

    def test_missing_delivery_refused(self):
        with patch.dict(os.environ, {"HUD04_DELIVERY_ROOT": ""}):
            with self.assertRaises(RuntimeError):
                C.load_verified_module("control")

    def test_sim_camera_is_explicit_and_not_calibration(self):
        camera = C.load_verified_module("camera")
        with self.assertRaises(ValueError):
            camera.chest_camera_mount(URDF, {})
        self.assertFalse(camera.chest_camera_mount(URDF, {"HUD04_CAMERA_PROFILE": "sim_chest_35deg_v1"})["calibration_validated"])


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="synthetic-hud04-", dir=ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        names, limits = C.validate_model(URDF)
        # Synthetic software fixture, not accepted robot motion or simulation evidence.
        self.arrays = dict(joint_names=np.array(names),
            joint_pos=np.tile([limits[n]["default"] for n in names], (4, 1)),
            base_pos_w=np.tile([0., 0., .9], (4, 1)),
            base_quat_w=np.tile([1., 0., 0., 0.], (4, 1)), framerate=np.array(50.))
        self.data = self.root / "synthetic_retargeted.npz"
        self.manifest = dict(schema_version=1, robot="HUD04", root_link="base_link", quaternion_order="wxyz",
            angle_unit="rad", position_unit="m", status="accepted_for_plane_replay", model_sha256=C.sha256(URDF),
            review_evidence={k: "SYNTHETIC UNIT TEST ONLY" for k in ("constraints", "root_frame", "fk_contact_replay")},
            files=[dict(path=self.data.name, sha256="")])
        self.path = self.root / "manifest.json"

    def validate(self):
        np.savez(self.data, **self.arrays)
        self.manifest["files"][0]["sha256"] = C.sha256(self.data)
        self.path.write_text(json.dumps(self.manifest))
        return C.validate_reference_manifest(self.path, URDF)

    def test_synthetic_numerical_contract(self):
        _, reports = self.validate()
        self.assertEqual(reports[0]["frames"], 4)
        self.assertAlmostEqual(reports[0]["duration_s"], .06)

    def test_reordered_joint_columns(self):
        self.arrays["joint_names"] = self.arrays["joint_names"][::-1]
        self.arrays["joint_pos"] = self.arrays["joint_pos"][:, ::-1]
        self.validate()

    def test_unaccepted_status_rejected(self):
        self.manifest["status"] = "candidate"
        with self.assertRaises(ValueError): self.validate()

    def test_wrong_robot_root_and_units(self):
        original = copy.deepcopy(self.manifest)
        for key, value in (("robot", "G1"), ("root_link", "torso_link"), ("angle_unit", "deg"),
                           ("position_unit", "cm"), ("quaternion_order", "xyzw"), ("model_sha256", "wrong")):
            with self.subTest(key=key):
                self.manifest = copy.deepcopy(original)
                self.manifest[key] = value
                with self.assertRaises(ValueError): self.validate()

    def test_missing_review_evidence(self):
        self.manifest["review_evidence"] = {}
        with self.assertRaises(ValueError): self.validate()

    def test_duplicate_joint_names(self):
        self.arrays["joint_names"][-1] = self.arrays["joint_names"][0]
        with self.assertRaises(ValueError): self.validate()

    def test_head_action_column(self):
        self.arrays["joint_names"][-1] = "head_pitch_joint"
        with self.assertRaises(ValueError): self.validate()

    def test_pickle_object_arrays_rejected(self):
        self.arrays["joint_names"] = self.arrays["joint_names"].astype(object)
        with self.assertRaises(ValueError): self.validate()

    def test_nan_rejected(self):
        self.arrays["joint_pos"][0, 0] = np.nan
        with self.assertRaises(ValueError): self.validate()

    def test_joint_limit_violation(self):
        self.arrays["joint_pos"][0, 0] = 100.
        with self.assertRaises(ValueError): self.validate()

    def test_joint_velocity_violation(self):
        self.arrays["joint_pos"][1, 0] += 1.
        with self.assertRaises(ValueError): self.validate()

    def test_nonunit_quaternion(self):
        self.arrays["base_quat_w"][0] *= 2
        with self.assertRaises(ValueError): self.validate()

    def test_shape_mismatch(self):
        self.arrays["base_pos_w"] = np.zeros((3, 3))
        with self.assertRaises(ValueError): self.validate()

    def test_bad_framerate(self):
        self.arrays["framerate"] = np.array(float("inf"))
        with self.assertRaises(ValueError): self.validate()

    def test_duplicate_files(self):
        self.manifest["files"] *= 2
        with self.assertRaises(ValueError): self.validate()

    def test_escape_path(self):
        self.manifest["files"][0]["path"] = "../outside_retargeted.npz"
        with self.assertRaises(ValueError): self.validate()

    def test_unreviewed_extra_motion(self):
        np.savez(self.root / "unreviewed_retargeted.npz", **self.arrays)
        with self.assertRaises(ValueError): self.validate()

    def test_checksum_tampering(self):
        self.validate()
        with self.data.open("ab") as stream: stream.write(b"tampered")
        with self.assertRaises(ValueError): C.validate_reference_manifest(self.path, URDF)


class IntegrationSourceTests(unittest.TestCase):
    def test_reference_loader_preserves_native_rate_and_samples(self):
        # Execute the actual _motion_cfg body with explicit CPU-only stubs.
        # This exercises the passed options; it is NOT an IsaacLab import.
        tree = ast.parse(CFG.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_motion_cfg")
        ns = dict(os=NS(environ={"HUD04_REFERENCE_MANIFEST": "SYNTHETIC-TEST-ONLY"}),
                  validate_reference_manifest=lambda *args, **kwargs: (Path("synthetic-reviewed-directory"), []),
                  HUD04_URDF=str(URDF), ROOT_LINK=C.ROOT_LINK, KEY_LINKS=C.KEY_LINKS,
                  MotionReferenceManagerCfg=lambda **kw: NS(**kw),
                  AmassMotionCfg=lambda **kw: NS(**kw))
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<synthetic-motion-cfg>", "exec"), ns)
        motion = ns["_motion_cfg"]().motion_buffers["AMASSMotion"]
        self.assertIsNone(motion.motion_interpolate_func)
        self.assertEqual(motion.skip_frames, 0)
        self.assertFalse(motion.clip_joint_ref_to_robot_limits)
        self.assertIsNone(motion.retargetting_func)
        self.assertFalse(motion.ensure_link_below_zero_ground)
        self.assertEqual(motion.motion_start_height_offset, 0.0)

    def test_all_overlay_sources_compile(self):
        files = list(OVERLAY.rglob("*.py"))
        self.assertEqual(len(files), 10)
        for path in files: compile(path.read_bytes(), str(path), "exec")

    def test_registration_is_lazy_and_additive(self):
        calls = []
        fake = NS(register=lambda **kwargs: calls.append(kwargs))
        path = CFG.parent / "__init__.py"
        with patch.dict(sys.modules, {"gymnasium": fake}):
            load("synthetic_registration", path)
        self.assertEqual([c["id"] for c in calls], ["Instinct-Perceptive-Shadowing-HUD04-Plane-v0",
            "Instinct-Perceptive-Shadowing-HUD04-Plane-Play-v0",
            "Instinct-Perceptive-Shadowing-HUD04-Terrain-v0",
            "Instinct-Perceptive-Shadowing-HUD04-Terrain-Play-v0"])
        self.assertTrue(all("instinct_rl_cfg_entry_point" in c["kwargs"] for c in calls))
        self.assertNotIn("isaaclab", sys.modules)

    def test_no_g1_asset_or_environment_inheritance(self):
        for path in OVERLAY.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(".g1", node.module or "")
                    self.assertNotIn("unitree_g1", node.module or "")

    def test_synthetic_postinit_assignments(self):
        # Execute the real method body against explicit fake config objects.
        # This is NOT an IsaacLab configclass/import or simulator validation.
        term = lambda: NS(params={"asset_cfg": NS(), "sensor_cfg": NS(), "reference_cfg": NS()})
        group = lambda: NS(**{k: term() for k in ("joint_pos", "joint_vel", "link_pos", "link_rot")})
        cfg = NS(reference_acceptance_status="accepted_for_plane_replay",
                 scene=NS(motion_reference=None, terrain=NS(), height_scanner=NS(),
                          camera=NS(offset=NS(), pattern_cfg=NS())),
                 actions=NS(joint_pos=NS()), observations=NS(policy=group(), critic=group()),
                 events=NS(**{k: term() for k in ("physics_material", "add_joint_default_pos", "base_com",
                    "randomize_ray_offsets", "randomize_actuator_gains", "randomize_rigid_body_mass",
                    "reset_robot", "bin_fail_counter_smoothing")}), curriculum=NS(),
                 rewards=NS(rewards=NS(undesired_contacts=term())),
                 terminations=NS(illegal_reset_contact=term(), link_pos_too_far=term()))
        tree = ast.parse(CFG.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HUD04PerceptivePlaneEnvCfg")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef))
        method.body = method.body[1:]  # skip only super().__post_init__()
        method.decorator_list = []
        names, limits = C.validate_model(URDF)
        control = C.load_verified_module("control")
        ns = {k: getattr(C, k) for k in ("ROOT_LINK", "CHEST_LINK", "KEY_LINKS", "FEET", "HANDS")}
        ns.update(_motion_cfg=lambda *args: NS(), HUD04_URDF=str(URDF), HUD04_JOINT_NAMES=names,
                  HUD04_ACTION_SCALE={n: control.control_values(n)[3] for n in names},
                  HUD04_VISUAL_LINKS=list(C.KEY_LINKS), SceneEntityCfg=lambda *a, **kw: NS(**kw),
                  load_verified_module=C.load_verified_module, get_link_prim_targets=lambda names: list(names), os=os)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "<synthetic-postinit>", "exec"), ns)
        with patch.dict(os.environ, {"HUD04_CAMERA_PROFILE": "sim_chest_35deg_v1"}):
            ns["__post_init__"](cfg)
        self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
        self.assertIsNone(cfg.scene.terrain.terrain_generator)
        self.assertFalse(cfg.actions.joint_pos.preserve_order)
        self.assertTrue(cfg.actions.joint_pos.use_default_offset)
        self.assertEqual(cfg.observations.critic.link_pos.params["asset_cfg"].body_names, list(C.KEY_LINKS))
        self.assertTrue(cfg.scene.camera.prim_path.endswith("/waist_pitch_link"))
        self.assertEqual(cfg.scene.camera.pattern_cfg.width, 48)
        self.assertIsNone(cfg.events.base_com)
        self.assertIsNone(cfg.events.bin_fail_counter_smoothing)
        self.assertIsNone(cfg.curriculum.beyond_adaptive_sampling)
        self.assertIsNone(cfg.terminations.out_of_border)
        self.assertFalse(set(C.FEET) & set(cfg.rewards.rewards.undesired_contacts.params["sensor_cfg"].body_names))
        self.assertEqual(cfg.events.reset_robot.params["randomize_pose_range"], {})

    def test_runner_is_one_iteration_no_resume(self):
        text = (CFG.parent / "agents/instinct_rl_ppo_cfg.py").read_text()
        self.assertIn("max_iterations = 1", text)
        self.assertIn("resume = False", text)
        self.assertIn("num_mini_batches = 1", text)

    def test_plan_entrypoint_has_no_simulator_import(self):
        result = subprocess.run([sys.executable, str(ROOT / "preflight.py")], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["gpu_started"])
        self.assertFalse(report["training_ready"])


class InstallerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="synthetic-upstream-", dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.upstream = Path(temp.name)
        self.package = self.upstream / "source/instinctlab/instinctlab"
        self.package.mkdir(parents=True)
        (self.package / "stub.py").write_text("# synthetic upstream fixture\n")
        self.audit = patch.dict(INSTALL.AUDITED, {"stub.py": C.sha256(self.package / "stub.py")}, clear=True)
        self.audit.start()
        self.addCleanup(self.audit.stop)

    def plan(self, head=INSTALL.PIN):
        with patch.object(INSTALL.subprocess, "check_output", return_value=head + "\n"):
            return INSTALL.plan(self.upstream)

    def test_plan_only_writes_nothing(self):
        self.assertEqual(len(self.plan()), 10)
        self.assertEqual(list(self.package.iterdir()), [self.package / "stub.py"])

    def test_wrong_upstream_commit_rejected(self):
        with self.assertRaises(ValueError): self.plan("wrong")

    def test_modified_audited_source_rejected(self):
        (self.package / "stub.py").write_text("# modified\n")
        with self.assertRaises(ValueError): self.plan()

    def test_existing_different_file_never_overwritten(self):
        target = self.package / "assets/hud04.py"
        target.parent.mkdir()
        target.write_text("# existing user code\n")
        with self.assertRaises(FileExistsError): self.plan()
        self.assertEqual(target.read_text(), "# existing user code\n")

    def test_identical_existing_file_is_noop(self):
        target = self.package / "assets/hud04.py"
        target.parent.mkdir()
        target.write_bytes((OVERLAY / "assets/hud04.py").read_bytes())
        self.assertEqual(sum(exists for _, _, exists in self.plan()), 1)

    def test_symlink_destination_rejected(self):
        (self.package / "assets").symlink_to(self.upstream, target_is_directory=True)
        with self.assertRaises(ValueError): self.plan()


if __name__ == "__main__":
    unittest.main(verbosity=2)