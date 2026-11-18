"""Explicit synthetic CPU fixtures: not a real env/reset/sensor acceptance."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import runtime_gate as gate

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("reviewed_control", ROOT.parents[1] / "flux/hud04_floating_control_probe/control_contract.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
NAMES = [side + "_" + joint + "_joint" for side in ("left", "right")
         for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
NAMES += ["waist_" + axis + "_joint" for axis in ("yaw", "roll", "pitch")]
NAMES += [side + "_" + joint + "_joint" for side in ("left", "right")
          for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")]


def fixture():
    # Deliberately use a different articulation order from this fixture's URDF order.
    names = NAMES[::2] + NAMES[1::2]
    policy = dict(gate.POLICY_SEGMENTS)
    critic = {"fixture_only_critic": (42,)}
    return {"robot_joint_names": names, "action_joint_names": names.copy(),
            "reference_joint_names": names.copy(), "position_observation_joint_names": names.copy(),
            "velocity_observation_joint_names": names.copy(), "action_term_names": ["joint_pos"],
            "action_dim": 29, "num_envs": 2, "preserve_order": False,
            "use_default_offset": True, "clip": None,
            "scale": np.array([[control.control_values(n)[3] for n in names]]),
            "offset": np.zeros((2, 29)), "default_joint_pos": np.zeros((2, 29)),
            "obs_format": {"policy": policy, "critic": critic},
            "observations": {"policy": np.zeros((2, gate.segments_width(policy))),
                             "critic": np.zeros((2, 42))}}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.state = fixture()

    def check(self):
        return gate.check_state(self.state, NAMES, control.control_values)

    def reject(self, key, value):
        self.state[key] = value
        with self.assertRaises(ValueError):
            self.check()

    def test_valid_fixture_is_not_training_acceptance(self):
        result = self.check()
        self.assertEqual(result["policy_flat_dim"], 1990)
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["reference_acceptance_proven_by_gate"])

    def test_no_input_mutation(self):
        previous = copy.deepcopy(self.state)
        self.check()
        np.testing.assert_array_equal(self.state["observations"]["policy"], previous["observations"]["policy"])
        self.assertEqual(self.state["robot_joint_names"], previous["robot_joint_names"])

    def test_action_permutation_is_not_silent(self):
        self.reject("action_joint_names", NAMES)

    def test_reference_permutation_is_not_silent(self):
        self.reject("reference_joint_names", NAMES)

    def test_position_order(self):
        self.reject("position_observation_joint_names", NAMES)

    def test_velocity_order(self):
        self.reject("velocity_observation_joint_names", NAMES)

    def test_head_joint_rejected(self):
        self.state["robot_joint_names"][0] = "head_yaw_joint"
        with self.assertRaises(ValueError): self.check()

    def test_duplicate_joint_rejected(self):
        self.state["robot_joint_names"][0] = self.state["robot_joint_names"][1]
        with self.assertRaises(ValueError): self.check()

    def test_action_dim(self): self.reject("action_dim", 31)
    def test_extra_action(self): self.reject("action_term_names", ["joint_pos", "head"])
    def test_environment_count(self): self.reject("num_envs", 1)
    def test_preserve_order(self): self.reject("preserve_order", True)
    def test_default_offset_flag(self): self.reject("use_default_offset", False)
    def test_hidden_clipping(self): self.reject("clip", {".*": [-1, 1]})
    def test_wrong_scale(self): self.reject("scale", np.full((1, 29), .25))
    def test_wrong_offset(self): self.reject("offset", np.ones((2, 29)))
    def test_bad_shape(self): self.reject("default_joint_pos", np.zeros((1, 29)))
    def test_nonfinite(self): self.reject("scale", np.full((1, 29), np.nan))

    def test_policy_order(self):
        p = self.state["obs_format"]["policy"]
        self.state["obs_format"]["policy"] = dict(reversed(list(p.items())))
        with self.assertRaises(ValueError): self.check()

    def test_depth_resolution(self):
        self.state["obs_format"]["policy"]["depth_image"] = (1, 27, 48)
        with self.assertRaises(ValueError): self.check()

    def test_history(self):
        self.state["obs_format"]["policy"]["last_action"] = (29,)
        with self.assertRaises(ValueError): self.check()

    def test_missing_critic(self):
        self.state["obs_format"].pop("critic")
        with self.assertRaises(ValueError): self.check()

    def test_flat_width(self):
        self.state["observations"]["policy"] = np.zeros((2, 1989))
        with self.assertRaises(ValueError): self.check()

    def test_critic_nonfinite(self):
        self.state["observations"]["critic"][0, 0] = np.inf
        with self.assertRaises(ValueError): self.check()

    def test_wrong_batch(self):
        self.state["observations"]["policy"] = np.zeros((1, 1990))
        with self.assertRaises(ValueError): self.check()


class LiveHookFixtureTests(unittest.TestCase):
    """Fake managers only: exercise extraction without importing a simulator."""
    def setUp(self):
        state = fixture()
        names = state["robot_joint_names"]
        term_type = type("JointPositionAction", (), {"__module__": "isaaclab.envs.mdp.actions.joint_actions"})
        self.term = term_type()
        self.term._joint_ids = slice(None)
        self.term.cfg = NS(preserve_order=False, use_default_offset=True, clip=None)
        self.term._scale = state["scale"]
        self.term._offset = state["offset"]
        self.manager = NS(active_terms=["joint_pos"], total_action_dim=29,
                          get_term=lambda key: self.term)
        obs_cfgs = [NS(params={"asset_cfg": NS(joint_ids=slice(None))}) for _ in gate.POLICY_SEGMENTS]
        self.obs_manager = NS(active_terms={"policy": list(gate.POLICY_SEGMENTS)},
                              _group_obs_term_cfgs={"policy": obs_cfgs})
        self.base = NS(scene={"robot": NS(joint_names=names, data=NS(default_joint_pos=state["default_joint_pos"])),
                             "motion_reference": NS(joint_names=names)},
                       action_manager=self.manager, observation_manager=self.obs_manager, num_envs=2)
        self.env = NS(unwrapped=self.base)
        wrapper_type = type("InstinctRlVecEnvWrapper", (), {"__module__": "instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper"})
        self.wrapper = wrapper_type()
        self.wrapper.unwrapped = self.base
        self.wrapper.get_obs_format = lambda: state["obs_format"]
        self.observations = state["observations"]

    def check(self):
        with patch.object(gate, "selected_source_receipt", return_value={"fixture_only": True}):
            return gate.check_live_environment(self.env, self.wrapper, self.observations, NAMES, control.control_values)

    def test_fixture_hook_no_reset_compute_or_step_methods_exist(self):
        self.assertTrue(self.check()["runtime_interface_checks_passed"])
        self.assertFalse(hasattr(self.base, "reset"))
        self.assertFalse(hasattr(self.obs_manager, "compute"))
        self.assertFalse(hasattr(self.base, "step"))

    def test_wrong_wrapper_identity(self):
        self.wrapper.unwrapped = NS()
        with self.assertRaises(ValueError): self.check()

    def test_resolved_observation_indices_not_unresolved_regex(self):
        index = list(gate.POLICY_SEGMENTS).index("joint_pos")
        self.obs_manager._group_obs_term_cfgs["policy"][index].params["asset_cfg"].joint_ids = list(reversed(range(29)))
        with self.assertRaises(ValueError): self.check()

    def test_wrong_action_runtime_indices(self):
        self.term._joint_ids = list(reversed(range(29)))
        with self.assertRaises(ValueError): self.check()

    def test_wrong_action_implementation(self):
        self.term = NS()
        with self.assertRaises(ValueError): self.check()


class SourceTests(unittest.TestCase):
    def test_modules_missing(self):
        with self.assertRaises(ValueError): gate.selected_source_receipt({})

    def test_wrong_source(self):
        with tempfile.TemporaryDirectory(dir="/workspace") as tmp:
            p = Path(tmp) / "fixture.py"
            p.write_text("# Explicit fixture, not IsaacLab\n")
            modules = {name: NS(__file__=str(p)) for name in gate.SOURCE_PINS}
            with self.assertRaises(ValueError): gate.selected_source_receipt(modules)

    def test_explicit_fixture_identity_and_changed_bytes(self):
        with tempfile.TemporaryDirectory(dir="/workspace") as tmp:
            p = Path(tmp) / "fixture.py"
            p.write_text("# Test-only fixture\n")
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            with patch.object(gate, "SOURCE_PINS", {"test_only_fixture": digest}):
                modules = {"test_only_fixture": NS(__file__=str(p))}
                self.assertEqual(gate.selected_source_receipt(modules)["test_only_fixture"]["sha256"], digest)
                p.write_text("# Changed fixture\n")
                with self.assertRaises(ValueError): gate.selected_source_receipt(modules)


if __name__ == "__main__":
    unittest.main()