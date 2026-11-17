"""HUD04 Plane/PLAY integration candidate, not an accepted training environment.

No G1 config inheritance, motion conversion, solver, training or GPU submission.
Importing task registrations stays cheap; constructing this config requires an
explicitly reviewed HUD04 reference manifest and chest-camera profile.
"""
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from instinctlab.assets.hud04 import (HUD04_ACTION_SCALE, HUD04_CFG, HUD04_JOINT_NAMES,
                                     HUD04_URDF, HUD04_VISUAL_LINKS, reference_robot_cfg)
from instinctlab.assets.hud04_contract import (CHEST_LINK, FEET, HANDS, KEY_LINKS, ROOT_LINK,
                                              load_verified_module, validate_reference_manifest)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import get_link_prim_targets
from instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg import (
    PerceptiveShadowingEnvCfg, PerceptiveShadowingSceneCfg)


def _motion_cfg():
    manifest = os.environ.get("HUD04_REFERENCE_MANIFEST")
    if not manifest:
        raise RuntimeError("HUD04_REFERENCE_MANIFEST is required; no fallback to G1/raw AMASS motions")
    directory, _ = validate_reference_manifest(manifest, HUD04_URDF)
    return MotionReferenceManagerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + ROOT_LINK,
        robot_model_path=HUD04_URDF,
        reference_prim_path="/World/envs/env_.*/RobotReference/" + ROOT_LINK,
        link_of_interests=list(KEY_LINKS),
        symmetric_augmentation_link_mapping=None,
        symmetric_augmentation_joint_mapping=None,
        symmetric_augmentation_joint_reverse_buf=None,
        frame_interval_s=0.1, update_period=0.02, num_frames=10,
        data_start_from="current_time", mp_split_method="None",
        visualizing_robot_offset=(2.0, 0.0, 0.0),
        visualizing_robot_from="reference_frame", visualizing_marker_types=["links"],
        motion_buffers={"AMASSMotion": AmassMotionCfg(
            path=str(directory), supported_file_endings=["retargeted.npz", "retargetted.npz"],
            retargetting_func=None, clip_joint_ref_to_robot_limits=False,
            ensure_link_below_zero_ground=False, motion_start_from_middle_range=[0.0, 0.0],
            motion_start_height_offset=0.0, motion_bin_length_s=None,
            buffer_device="output_device", motion_interpolate_func=motion_interpolate_bilinear,
            velocity_estimation_method="frontbackward", env_starting_stub_sampling_strategy="independent",
        )},
    )


@configclass
class HUD04PerceptivePlaneEnvCfg(PerceptiveShadowingEnvCfg):
    scene: PerceptiveShadowingSceneCfg = PerceptiveShadowingSceneCfg(
        num_envs=2, robot=HUD04_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"))

    def __post_init__(self):
        super().__post_init__()
        self.scene.motion_reference = _motion_cfg()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Do not use finite motion-matched terrain bounds on an infinite plane.
        self.terminations.out_of_border = None
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + ROOT_LINK
        self.actions.joint_pos.joint_names = list(HUD04_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = False  # runtime articulation order, not URDF order
        self.actions.joint_pos.scale = dict(HUD04_ACTION_SCALE)
        self.actions.joint_pos.use_default_offset = True

        # Keep the upstream depth normalization/crop/resize (18x32), not raw probe depth.
        camera_module = load_verified_module("camera")
        mount = camera_module.chest_camera_mount(HUD04_URDF, os.environ)
        if mount["parent_link"] != CHEST_LINK:
            raise ValueError("HUD04 chest-camera anchor mismatch")
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/" + CHEST_LINK
        self.scene.camera.offset.pos = mount["pos"]
        self.scene.camera.offset.rot = mount["rot"]
        self.scene.camera.offset.convention = mount["convention"]
        camera_module.apply_sim_camera_settings(self.scene.camera, mount)
        self.scene.camera.mesh_prim_paths = ["/World/ground"] + get_link_prim_targets(HUD04_VISUAL_LINKS)

        for name in ("link_pos", "link_rot"):
            term = getattr(self.observations.critic, name)
            term.params["asset_cfg"].body_names = list(KEY_LINKS)
            term.params["asset_cfg"].preserve_order = True
        for group in (self.observations.policy, self.observations.critic):
            for name in ("joint_pos", "joint_vel"):
                getattr(group, name).params["asset_cfg"] = SceneEntityCfg(
                    "robot", joint_names=list(HUD04_JOINT_NAMES), preserve_order=False)

        # Initial integration uses deterministic settings, not unreviewed G1 randomization.
        for name in ("physics_material", "add_joint_default_pos", "base_com",
                     "randomize_ray_offsets", "randomize_actuator_gains", "randomize_rigid_body_mass"):
            if not hasattr(self.events, name):
                raise RuntimeError("Upstream event schema changed: " + name)
            setattr(self.events, name, None)
        self.events.reset_robot.params["randomize_pose_range"] = {}
        self.events.reset_robot.params["randomize_velocity_range"] = {}
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)
        self.events.bin_fail_counter_smoothing = None
        self.curriculum.beyond_adaptive_sampling = None

        # Plane replay initially allows only sole contact, not hand support.
        # Select collision-bearing visual links, excluding feet, to avoid empty fixed helper bodies.
        nonfeet = [name for name in HUD04_VISUAL_LINKS if name not in FEET]
        self.rewards.rewards.undesired_contacts.params["sensor_cfg"].body_names = nonfeet
        self.terminations.illegal_reset_contact.params["sensor_cfg"].body_names = nonfeet
        self.terminations.link_pos_too_far.params["reference_cfg"].body_names = list(FEET + HANDS)
        self.run_name = "hud04_perceptive_plane_integration_candidate"


@configclass
class HUD04PerceptivePlaneEnvCfg_PLAY(HUD04PerceptivePlaneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.robot_reference = reference_robot_cfg()
        self.scene.motion_reference.debug_vis = True
        self.observations.policy.enable_corruption = False
        self.scene.camera.debug_vis = False
        self.observations.policy.depth_image.params["debug_vis"] = False