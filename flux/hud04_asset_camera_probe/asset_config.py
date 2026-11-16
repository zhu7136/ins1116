"""HUD04 serial model, fixed-head variant. Simulation integration is unvalidated.

PD gains/action scales/armature are initial values from LimX GMR
train_params/HU_D04_01_series.yaml (81f8197), not tuned InstinctLab results.
Set HUMANOID_DESCRIPTION_DIR to the checkout containing HU_D04_description.
"""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg


def _model_path():
    directory = os.environ.get("HUMANOID_DESCRIPTION_DIR")
    if not directory:
        raise RuntimeError("Set HUMANOID_DESCRIPTION_DIR to the humanoid-description checkout")
    path = Path(directory).expanduser() / "HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf"
    if not path.is_file():
        raise FileNotFoundError(
            f"Run prepare_hud04_fixed_head.py, then prepare_hud04_body_collisions.py first: {path}"
        )
    return str(path.resolve())


HUD04_URDF = _model_path()
_root = ET.parse(HUD04_URDF).getroot()
_joints = [j for j in _root.findall("joint") if j.get("type") == "revolute"]
HUD04_JOINT_NAMES = [j.get("name") for j in _joints]
assert len(HUD04_JOINT_NAMES) == 29
assert not any(name.startswith("head_") for name in HUD04_JOINT_NAMES)
HUD04_VISUAL_LINKS = [link.get("name") for link in _root.findall("link") if link.find("visual") is not None]


def _control_values(name):
    # stiffness, damping, reflected inertia, action scale
    if "hip_" in name or "knee_" in name:
        return 139.41, 17.75, 0.14125, 0.2511
    if "ankle_" in name or name.startswith("waist_"):
        return 93.65, 11.92, 0.1845504, 0.1121
    if "shoulder_" in name or "elbow_" in name:
        return 87.51, 11.14, 0.0886706, 0.1200
    if "wrist_" in name:
        return 15.12, 1.93, 0.0153218, 0.3141
    raise ValueError(name)


HUD04_ACTION_SCALE = {n: _control_values(n)[3] for n in HUD04_JOINT_NAMES}
HUD04_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=HUD04_URDF,
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        self_collision=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Provisional spawn pose only. Motion resets use the HUD04 reference.
        pos=(0.0, 0.0, 0.9),
        joint_pos={n: (-0.2 if "hip_pitch" in n or "ankle_pitch" in n
                       else 0.4 if "knee" in n else 0.0) for n in HUD04_JOINT_NAMES},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "body": DelayedPDActuatorCfg(
            joint_names_expr=HUD04_JOINT_NAMES,
            effort_limit={j.get("name"): float(j.find("limit").get("effort")) for j in _joints},
            velocity_limit={j.get("name"): float(j.find("limit").get("velocity")) for j in _joints},
            stiffness={n: _control_values(n)[0] for n in HUD04_JOINT_NAMES},
            damping={n: _control_values(n)[1] for n in HUD04_JOINT_NAMES},
            armature={n: _control_values(n)[2] for n in HUD04_JOINT_NAMES},
            min_delay=0, max_delay=0,
        ),
    },
)
HUD04_CFG.spawn.joint_drive.gains.stiffness = 0.0
HUD04_CFG.spawn.joint_drive.gains.damping = 0.0