"""Name-aware strict loaders wired into both Plane and TerrainMotion paths."""
from pathlib import Path
import torch
from isaaclab.utils import configclass
from instinctlab.assets.hud04 import HUD04_URDF
from instinctlab.assets.hud04_contract import validate_model
from instinctlab.assets.hud04_motion import read_named_motion
from .amass_motion import AmassMotion
from .amass_motion_cfg import AmassMotionCfg
from .terrain_motion import TerrainMotion
from .terrain_motion_cfg import TerrainMotionCfg

class _NamedStep1:
    def _read_motion_file(self,motion_file_idx):
        # Override dispatch too: no AMASS/SOMA fallback through the base class.
        return self._read_retargetted_motion_file(self._all_motion_files[motion_file_idx])

    def _read_retargetted_motion_file(self,filepath):
        if self.cfg.motion_interpolate_func is not None or self.cfg.skip_frames!=0 or self.cfg.clip_joint_ref_to_robot_limits or self.cfg.ensure_link_below_zero_ground or self.cfg.motion_start_height_offset!=0:
            raise ValueError('Reference transformations/clipping disabled')
        _,limits=validate_model(HUD04_URDF)
        q,p,r,fps=read_named_motion(filepath,self.isaac_joint_names,limits)
        arrays=[torch.as_tensor(a,dtype=torch.float,device=self.buffer_device) for a in (p,r,q)]
        return self._pack_retargetted_motion_sequence(*arrays,fps)

class HUD04AmassMotion(_NamedStep1,AmassMotion):pass

class HUD04TerrainMotion(_NamedStep1,TerrainMotion):
    def support_source_frames(self,env_ids,sample_timestamp):
        """Pinned upstream frame formula, including random reset stub offset.

        Do NOT use episode step or rounded 50Hz control step as a 90Hz frame.
        Out-of-range indices remain out of range so support fails closed.
        """
        ids=self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)
        motion_ids=self._assigned_env_motion_selection[ids]
        frames=torch.round((self._motion_buffer_start_time_s[ids]+sample_timestamp.to(self.buffer_device))*
                           self._all_motion_sequences.framerate[motion_ids]).long()
        return frames.to(self.output_device)

@configclass
class HUD04AmassMotionCfg(AmassMotionCfg):
    class_type:type=HUD04AmassMotion

@configclass
class HUD04TerrainMotionCfg(TerrainMotionCfg):
    class_type:type=HUD04TerrainMotion