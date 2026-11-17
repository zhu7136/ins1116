"""CPU-only integration gates. These checks do not establish dynamic feasibility."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

INSTINCTLAB_COMMIT = "de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057"
ROOT_LINK = "base_link"
CHEST_LINK = "waist_pitch_link"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
HANDS = ("left_wrist_roll_link", "right_wrist_roll_link")
KEY_LINKS = (ROOT_LINK, CHEST_LINK, "left_shoulder_roll_link", "right_shoulder_roll_link",
             "left_elbow_link", "right_elbow_link", *HANDS,
             "left_hip_roll_link", "right_hip_roll_link", "left_knee_link", "right_knee_link", *FEET)
VERIFIED_MODULES = {
    "asset": ("flux/hud04_asset_camera_probe/asset_config.py", "4fdb08b4aa352893a6feef5afb6deb8ec273e1fbdcfc2d67ac0e691729321021"),
    "camera": ("flux/hud04_asset_camera_probe/camera_mount.py", "02695e048a19f4d62d9c2798c9810b48570f482357893f79e0bfb27474bc8e2f"),
    "control": ("flux/hud04_floating_control_probe/control_contract.py", "6e133dbe2ae214f6d71156fea49c86eb74ae24503ec1ebfa2bef54f7fb7de71f"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_verified_module(kind):
    directory = os.environ.get("HUD04_DELIVERY_ROOT")
    if not directory:
        raise RuntimeError("Set HUD04_DELIVERY_ROOT to your ins1116 checkout")
    relative, expected = VERIFIED_MODULES[kind]
    path = Path(directory).expanduser().resolve() / relative
    if sha256(path) != expected:
        raise ValueError("Verified baseline module changed: " + relative)
    spec = importlib.util.spec_from_file_location("_hud04_verified_" + kind, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_model(path):
    """Reuse the reviewed body/head control contract, then audit semantic links."""
    names, limits = load_verified_module("control").read_model(path)
    root = ET.parse(path).getroot()
    links = [link.get("name") for link in root.findall("link")]
    if len(set(links)) != len(links):
        raise ValueError("Duplicate links")
    roots = set(links) - {joint.find("child").get("link") for joint in root.findall("joint")}
    if roots != {ROOT_LINK} or not set(KEY_LINKS).issubset(links):
        raise ValueError("HUD04 root/key-link contract mismatch")
    # Roll links are downstream of yaw/pitch: do not silently retain G1 wrist-yaw semantics.
    for side in ("left", "right"):
        joint = next(j for j in root.findall("joint") if j.get("name") == side + "_wrist_roll_joint")
        if joint.find("child").get("link") != side + "_wrist_roll_link":
            raise ValueError("Unexpected hand anchor")
    return names, limits


def validate_reference_manifest(manifest_path, urdf_path):
    """Only explicitly accepted, checksummed HUD04 Plane references can be loaded.

    Never retarget, rescale, clip, infer quaternion order, or create reference data.
    Human/physics review evidence is required in addition to numerical gates.
    """
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text())
    expected = dict(schema_version=1, robot="HUD04", root_link=ROOT_LINK,
                    quaternion_order="wxyz", angle_unit="rad", position_unit="m",
                    status="accepted_for_plane_replay", model_sha256=sha256(urdf_path))
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Reference identity/convention/acceptance mismatch")
    evidence = manifest.get("review_evidence")
    if not isinstance(evidence, dict) or not all(isinstance(evidence.get(k), str) and evidence[k].strip()
            for k in ("constraints", "root_frame", "fk_contact_replay")):
        raise ValueError("Missing recovered-constraint/frame/contact review evidence")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries or len(entries) > 100:
        raise ValueError("Expected 1..100 explicitly accepted files")
    names, limits = validate_model(urdf_path)
    seen, reports = set(), []
    for item in entries:
        relative = item["path"]
        file = (path.parent / relative).resolve()
        if (Path(relative).is_absolute() or not file.is_relative_to(path.parent)
                or not file.name.endswith(("retargeted.npz", "retargetted.npz")) or file in seen):
            raise ValueError("Duplicate, unsupported or escaping reference path")
        seen.add(file)
        if file.stat().st_size > 128 * 1024 * 1024 or sha256(file) != item["sha256"]:
            raise ValueError("Reference size/checksum mismatch")
        # Bounded decompression; object arrays/pickles deliberately unsupported.
        import zipfile
        with zipfile.ZipFile(file) as archive:
            if sum(i.file_size for i in archive.infolist()) > 256 * 1024 * 1024:
                raise ValueError("Reference decompressed size exceeds limit")
        with np.load(file, allow_pickle=False) as data:
            stored_names = data["joint_names"]
            if stored_names.ndim != 1 or stored_names.dtype.kind != "U":
                raise ValueError("joint_names must be a Unicode vector, not pickle/object")
            order = load_verified_module("control").permutation(stored_names.tolist(), names)
            q = np.asarray(data["joint_pos"], dtype=np.float64)
            p = np.asarray(data["base_pos_w"], dtype=np.float64)
            quat = np.asarray(data["base_quat_w"], dtype=np.float64)
            fps_array = np.asarray(data["framerate"])
            if fps_array.size != 1:
                raise ValueError("Invalid framerate")
            fps = float(fps_array.item())
            n = len(q)
            if not 3 <= n <= 100000 or q.shape != (n, 29) or p.shape != (n, 3) or quat.shape != (n, 4):
                raise ValueError("Invalid reference tensor shapes")
            if not math.isfinite(fps) or not 1 <= fps <= 1000 or not all(np.isfinite(a).all() for a in (q, p, quat)):
                raise ValueError("Nonfinite reference or invalid framerate")
            q = q[:, order]
            lower, upper = (np.array([limits[k][bound] for k in names]) for bound in ("lower", "upper"))
            if np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6):
                raise ValueError("Reference exceeds body joint limits")
            if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-3, rtol=0):
                raise ValueError("Reference quaternion is not unit length")
            peak_speed = np.max(np.abs(np.diff(q, axis=0)) * fps, axis=0)
            if np.any(peak_speed > np.array([limits[k]["velocity"] for k in names]) + 1e-6):
                raise ValueError("Reference exceeds URDF joint velocity limits")
            # No guessed root-speed or contact thresholds: those require the cited review.
            reports.append(dict(path=relative, frames=n, duration_s=(n-1)/fps, framerate=fps))
    discovered = {f.resolve() for f in path.parent.rglob("*.npz")}
    if discovered != seen:
        raise ValueError("Dataset must contain exactly the reviewed NPZ files; use a dedicated directory")
    return path.parent, reports