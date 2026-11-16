"""Chest mount validation and an opt-in, explicitly approximate simulation profile."""
import math
import xml.etree.ElementTree as ET

CHEST_CAMERA_PARENT = "waist_pitch_link"
CAMERA_CONVENTIONS = ("world", "ros", "opengl")
SIM_PROFILE = "sim_chest_35deg_v1"


def apply_sim_camera_settings(camera, mount):
    """Apply only the named simulation profile; preserve manual-mode behavior.

    Pure config assignments, not evidence of Isaac asset or sensor validation.
    FOV follows InstinctLab's existing depth-ray pattern, not measured HUD04 K.
    """
    if mount.get("profile") != SIM_PROFILE:
        return
    imaging = mount["simulation_imaging"]
    camera.pattern_cfg.focal_length = 1.0
    camera.pattern_cfg.horizontal_aperture = 2 * math.tan(math.radians(imaging["hfov_deg"]) / 2)
    camera.pattern_cfg.vertical_aperture = 2 * math.tan(math.radians(imaging["vfov_deg"]) / 2)
    camera.pattern_cfg.width = imaging["width"]
    camera.pattern_cfg.height = imaging["height"]
    camera.update_period = 1 / imaging["fps"]
    camera.min_distance = imaging["min_distance_m"]
    camera.max_distance = imaging["max_distance_m"]


def _vector(settings, key, size):
    text = settings.get(key)
    if not text:
        raise ValueError(f"Set {key} to {size} calibrated values relative to {CHEST_CAMERA_PARENT}")
    values = tuple(float(x) for x in text.split(","))
    if len(values) != size or not all(math.isfinite(x) for x in values):
        raise ValueError(f"Invalid {key}")
    return values


def chest_camera_mount(urdf_path, settings):
    """Return local optical pose; WXYZ is quaternion order, not an axis convention.

    Existing head-relative translations/quaternions must be recalibrated or
    transformed; they must not be reused as chest-relative numbers.
    URDF's commented waist_camera_link pose is not accepted as optical calibration.
    """
    root = ET.parse(urdf_path).getroot()
    links = {link.get("name") for link in root.findall("link")}
    if CHEST_CAMERA_PARENT not in links:
        raise ValueError(f"Missing chest-camera parent: {CHEST_CAMERA_PARENT}")
    profile = settings.get("HUD04_CAMERA_PROFILE", "manual")
    if profile not in ("manual", SIM_PROFILE):
        raise ValueError(f"Unknown HUD04_CAMERA_PROFILE: {profile}")
    if profile == SIM_PROFILE:
        # Deliberate world-convention pose: +X forward, +Z image-up.
        # Do NOT interpret the commented URDF Ry(125deg) as a full ROS optical pose.
        if any(settings.get(key) for key in ("HUD04_CAMERA_POS", "HUD04_CAMERA_QUAT_WXYZ",
                                             "HUD04_CAMERA_CONVENTION")):
            raise ValueError("Simulation profile conflicts with manual camera overrides; unset them or use manual")
        half_pitch = math.radians(35.0) / 2
        return dict(parent_link=CHEST_CAMERA_PARENT, pos=(0.092, 0.0175, 0.2751),
                    rot=(math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0),
                    convention="world", profile=profile, calibration_validated=False,
                    simulation_imaging=dict(width=48, height=27, hfov_deg=87.0,
                                            vfov_deg=58.0, fps=30.0,
                                            min_distance_m=0.3, max_distance_m=2.0),
                    pose_source="URDF local translation + manual 35deg pitch + chosen upright roll",
                    imaging_source="Simulation choices; FOV and resolution retain upstream depth-ray pattern")
    convention = settings.get("HUD04_CAMERA_CONVENTION") or "world"
    if convention not in CAMERA_CONVENTIONS:
        raise ValueError(f"HUD04_CAMERA_CONVENTION must be one of {CAMERA_CONVENTIONS}")
    pos = _vector(settings, "HUD04_CAMERA_POS", 3)
    quat = _vector(settings, "HUD04_CAMERA_QUAT_WXYZ", 4)
    if abs(math.sqrt(sum(x*x for x in quat)) - 1.0) > 0.001:
        raise ValueError("HUD04_CAMERA_QUAT_WXYZ must be a unit quaternion")
    return dict(parent_link=CHEST_CAMERA_PARENT, pos=pos, rot=quat, convention=convention)