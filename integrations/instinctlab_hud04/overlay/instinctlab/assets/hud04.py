"""HUD04 asset adapter; reuse, do not rewrite, the GPU-tested probe asset.

The imported PD values remain provisional. Full InstinctLab integration is NOT
GPU-validated by the standalone probe results.
"""
from .hud04_contract import load_verified_module, validate_model

_asset = load_verified_module("asset")
HUD04_URDF = _asset.HUD04_URDF
HUD04_JOINT_NAMES, _ = validate_model(HUD04_URDF)
HUD04_CFG = _asset.HUD04_CFG.copy()
HUD04_ACTION_SCALE = dict(_asset.HUD04_ACTION_SCALE)
HUD04_VISUAL_LINKS = list(_asset.HUD04_VISUAL_LINKS)


def reference_robot_cfg():
    """Visualization must not collide with or add contacts to the real robot."""
    cfg = HUD04_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference")
    cfg.spawn.collision_props = _asset.sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    cfg.spawn.rigid_props.disable_gravity = True
    cfg.spawn.articulation_props.enabled_self_collisions = False
    cfg.spawn.activate_contact_sensors = False
    return cfg