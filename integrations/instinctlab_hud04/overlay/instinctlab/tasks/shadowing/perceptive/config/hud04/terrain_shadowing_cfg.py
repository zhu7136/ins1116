"""Dedicated HUD04 TerrainMotion candidate; not GPU-accepted or training-ready.

Reuse HUD04 camera/action/reset setup, NEVER G1 config/model/weights. No fallback
dataset, support labels, root convention, reference acceptance or terrain file.
"""
import os
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from instinctlab.assets.hud04 import reference_robot_cfg
from instinctlab.assets.hud04_motion import validate_terrain_bundle
from instinctlab.motion_reference.motion_files.hud04_motion import HUD04TerrainMotionCfg
from instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg import PerceptiveShadowingSceneCfg, TerminationsCfg
from .perceptive_shadowing_cfg import HUD04PerceptivePlaneEnvCfg
from .conditional_contacts import conditional_undesired_contacts,illegal_support_contact

@configclass
class HUD04PerceptiveTerrainEnvCfg(HUD04PerceptivePlaneEnvCfg):
    reference_acceptance_status: str='accepted_for_terrain_replay'
    hud04_support_schedule: dict={}

    def __post_init__(self):
        super().__post_init__()
        root,metadata,schedule=validate_terrain_bundle(os.environ['HUD04_REFERENCE_MANIFEST'])
        self.hud04_support_schedule=schedule
        reference=self.scene.motion_reference
        old=reference.motion_buffers.pop('AMASSMotion')
        reference.motion_buffers={'TerrainMotion':HUD04TerrainMotionCfg(
            path=str(root),metadata_yaml=str(metadata),max_origins_per_motion=4,
            supported_file_endings=['retargeted.npz','retargetted.npz'],
            retargetting_func=None,clip_joint_ref_to_robot_limits=False,
            ensure_link_below_zero_ground=False,motion_start_from_middle_range=[0.,0.],
            motion_start_height_offset=0.,motion_bin_length_s=None,buffer_device='output_device',
            motion_interpolate_func=None,skip_frames=0,velocity_estimation_method='frontbackward',
            env_starting_stub_sampling_strategy='independent')}
        # Restore audited upstream crop/border-height/translation implementation,
        # not a hand-written plane or a raw-STL bypass.
        self.scene.terrain=PerceptiveShadowingSceneCfg().terrain.copy()
        generator=self.scene.terrain.terrain_generator
        generator.num_rows=2;generator.num_cols=2
        generator.curriculum=False
        sub=generator.sub_terrains['motion_matched']
        sub.path=str(root);sub.metadata_yaml=str(metadata)
        self.terminations.out_of_border=TerminationsCfg().out_of_border.copy()
        self.rewards.rewards.undesired_contacts.func=conditional_undesired_contacts
        self.rewards.rewards.undesired_contacts.params={'sensor_cfg':SceneEntityCfg('contact_forces'),'threshold':1.}
        self.terminations.illegal_reset_contact.func=illegal_support_contact
        self.terminations.illegal_reset_contact.time_out=False
        self.terminations.illegal_reset_contact.params={'sensor_cfg':SceneEntityCfg('contact_forces'),'threshold':500.}
        # Thresholds are inherited diagnostic defaults, NOT accepted tolerances
        # for penetration/slip/tracking or tuned HUD04 force safety limits.
        self.run_name='hud04_perceptive_terrain_integration_candidate'

@configclass
class HUD04PerceptiveTerrainEnvCfg_PLAY(HUD04PerceptiveTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.scene.robot_reference=reference_robot_cfg()
        self.scene.motion_reference.debug_vis=True
        self.observations.policy.enable_corruption=False
        self.scene.camera.debug_vis=False
        self.observations.policy.depth_image.params['debug_vis']=False