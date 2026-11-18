"""Read-only live-manager assertions, NOT a simulator or training entry point.

Call after a *normal* accepted-reference env reset/wrapper construction and before
constructing an RL runner. Pass the observations already produced by that reset;
this module never calls reset/compute/step or changes history buffers.
No receipt here constitutes contact/dynamic/motion acceptance or GPU permission.
"""
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

# Source bytes reviewed in the CPU interface audit. These are a selected-file
# identity gate, not proof of the complete dependency tree or installed binaries.
# A mismatch blocks this candidate: review the actual image, never repin on the fly.
SOURCE_PINS = {
    "isaaclab.envs.mdp.actions.joint_actions": "6bbf0f7837636eb7d6b17db3e8d25a433e8a69ba671dde0e993b0a0b2025cf55",
    "isaaclab.envs.mdp.observations": "3cbe93ba4e37b9ad9063a65dab244547ad4582798960c0190375fd74d930fd54",
    "isaaclab.managers.observation_manager": "57539db34658e98f92961ea6a182d5b1326ec9c9ec5a1cc39e5c032bfdd2cf01",
    "isaaclab.utils.string": "2bee47035b9e060511ba7006000b84e44b28e266e47d4be3b0bb136522597152",
    "isaaclab.utils.buffers.circular_buffer": "7e3d298e9c43fb05edf96ff3bc9333ca1081faacba7d73b412e5edd5ca60198b",
    "instinctlab.utils.noise.noise_model": "c519e787a2aae99ddfec0432a154f3838ddb66cadcfbff32434537e93d25daee",
    "instinctlab.envs.mdp.observations.exteroception": "9b7d386707d2f9b2b40bb60a77e45c20431fcd1857c428c34a7817716fedd18c",
    "instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper": "4a2feebf922e03e5f236fdd43431533a94ea10e6ace5bdd7dd207e6082e3e3c1",
    "instinctlab.motion_reference.motion_files.amass_motion": "aa233bc12102403fa4f9ea90dcc0d7b2b1ecdf670b5c7c4bcc4f2ff068ed6909",
}

POLICY_SEGMENTS = {
    "joint_pos_ref": (10, 29), "joint_vel_ref": (10, 29),
    "position_ref": (10, 3), "rotation_ref": (10, 6),
    "depth_image": (1, 18, 32), "projected_gravity": (24,),
    "base_ang_vel": (24,), "joint_pos": (232,),
    "joint_vel": (232,), "last_action": (232,),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array(value):
    # Torch tensor -> bounded CPU array; no torch/simulator import on CPU.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    require(result.dtype.kind in "fiu" and np.isfinite(result).all(), "Nonfinite/nonnumeric runtime tensor")
    return result


def named_order(names, expected):
    names, expected = list(names), list(expected)
    require(len(names) == len(expected) == 29 and len(set(names)) == 29 and len(set(expected)) == 29
            and set(names) == set(expected) and not any("head" in n.lower() for n in names),
            "Expected exactly 29 unique HUD04 body joints, no head actions")
    return names


def selected_source_receipt(modules=None):
    """Hash files of already imported modules, never import ISAAC on CPU.

    modules injection is solely for explicit CPU fixtures. Production callers
    omit it and use sys.modules. Native binaries and monkeypatches are not attested.
    """
    modules = sys.modules if modules is None else modules
    receipt = {}
    for name, digest in SOURCE_PINS.items():
        module = modules.get(name)
        filename = getattr(module, "__file__", None)
        require(filename is not None, "Required module not loaded: " + name)
        path = Path(filename)
        require(path.suffix == ".py" and path.is_file(), "Expected inspectable Python source: " + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == digest, "Selected installed source mismatch: " + name)
        receipt[name] = {"path": str(path.resolve()), "sha256": actual}
    return receipt


def segments_width(segments):
    width = 0
    for shape in segments.values():
        require(isinstance(shape, (list, tuple)) and len(shape) > 0
                and all(type(d) is int and d > 0 for d in shape), "Invalid observation segment")
        width += math.prod(shape)
    return width


def check_state(state, names, control_values):
    """Pure validator for a captured live state; fixtures prove software only."""
    runtime = named_order(state["robot_joint_names"], names)
    require(state["action_joint_names"] == runtime, "Action order != articulation order")
    require(state["reference_joint_names"] == runtime, "Reference order != articulation order")
    require(state["position_observation_joint_names"] == runtime, "Joint position observation order mismatch")
    require(state["velocity_observation_joint_names"] == runtime, "Joint velocity observation order mismatch")
    require(state["action_term_names"] == ["joint_pos"] and state["action_dim"] == 29,
            "Unexpected action terms/dimension")
    require(state["num_envs"] == 2, "First integration gate requires exactly two environments")
    require(state["preserve_order"] is False and state["use_default_offset"] is True
            and state["clip"] is None, "Unexpected action ordering/offset/clip settings")
    expected_scale = np.array([control_values(n)[3] for n in runtime])
    scale, offset, default = (array(state[k]) for k in ("scale", "offset", "default_joint_pos"))
    require(scale.shape in ((1, 29), (2, 29)) and offset.shape == default.shape == (2, 29),
            "Unexpected scale/default tensor shape")
    require(np.allclose(scale, expected_scale, atol=1e-7, rtol=1e-6), "Name-to-action scale mismatch")
    require(np.allclose(offset, default, atol=1e-7, rtol=0), "Action offset != runtime default joint positions")
    policy = state["obs_format"].get("policy", {})
    require(list(policy) == list(POLICY_SEGMENTS)
            and all(tuple(policy[k]) == POLICY_SEGMENTS[k] for k in POLICY_SEGMENTS),
            "Policy term order/shape or history/depth contract mismatch")
    require("critic" in state["obs_format"], "Missing critic observation format")
    widths = {k: segments_width(state["obs_format"][k]) for k in ("policy", "critic")}
    for group, width in widths.items():
        require(array(state["observations"][group]).shape == (2, width), "Observation batch/width mismatch: " + group)
    return {"runtime_interface_checks_passed": True, "actions": 29, "head_actions": 0,
            "num_envs": 2, "policy_flat_dim": widths["policy"], "critic_flat_dim": widths["critic"],
            "runtime_joint_names": runtime, "selected_source_identity_only": True,
            "gpu_steps_executed_by_gate": 0, "training_updates": 0,
            "reference_acceptance_proven_by_gate": False, "training_ready": False}


def check_live_environment(env, wrapper, observations, names, control_values):
    """Assertion hook for an EXISTING normally loaded/reset HUD04 environment.

    Does not bypass reference acceptance, construct a wrapper/runner or launch
    anything. install_overlay.py now connects this hook through startup_gate.py
    to audited train.py. Other callers must also propagate exceptions as failures.
    observations is the flattened group dict already emitted by wrapper.reset().
    """
    receipt = selected_source_receipt()
    base = env.unwrapped
    require(wrapper.unwrapped is base, "Wrapper/environment identity mismatch")
    require(type(wrapper).__module__ == "instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper"
            and type(wrapper).__name__ == "InstinctRlVecEnvWrapper",
            "Unexpected live wrapper implementation")
    robot = base.scene["robot"]
    term = base.action_manager.get_term("joint_pos")
    require(type(term).__module__ == "isaaclab.envs.mdp.actions.joint_actions"
            and type(term).__name__ == "JointPositionAction",
            "Unexpected live action implementation")
    runtime_names = list(robot.joint_names)

    def observation_names(name):
        # Pinned IsaacLab keeps resolved term configs in this private table;
        # fail if it changes. Read resolved IDs, not unresolved regex config.
        manager = base.observation_manager
        index = manager.active_terms["policy"].index(name)
        cfg = manager._group_obs_term_cfgs["policy"][index]
        ids = cfg.params["asset_cfg"].joint_ids
        return np.asarray(runtime_names)[ids].tolist()

    state = {"robot_joint_names": runtime_names,
             "action_joint_names": np.asarray(runtime_names)[term._joint_ids].tolist(),
             "reference_joint_names": list(base.scene["motion_reference"].joint_names),
             "position_observation_joint_names": observation_names("joint_pos"),
             "velocity_observation_joint_names": observation_names("joint_vel"),
             "action_term_names": list(base.action_manager.active_terms),
             "action_dim": base.action_manager.total_action_dim,
             "num_envs": base.num_envs, "preserve_order": term.cfg.preserve_order,
             "use_default_offset": term.cfg.use_default_offset, "clip": term.cfg.clip,
             "scale": term._scale, "offset": term._offset,
             "default_joint_pos": robot.data.default_joint_pos,
             "obs_format": wrapper.get_obs_format(), "observations": observations}
    report = check_state(state, names, control_values)
    report["selected_loaded_sources"] = receipt
    return report