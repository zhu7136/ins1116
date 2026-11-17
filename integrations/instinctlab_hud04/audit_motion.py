"""Read-only candidate audit, NOT reference acceptance, retargeting or simulation.

Inputs are local NPZ/YAML/binary STL data; never unpickle or execute them.
No network, GPU, solver, training, data rewriting or manifest approval.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import zipfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('hud04_audit_contract', ROOT / 'overlay/instinctlab/assets/hud04_contract.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
EPS = 1e-6


def contained(root, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('Expected relative dataset path')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Escaping dataset path')
    return path


def audit_npz(file, names, model):
    if file.stat().st_size > 128 * 1024 * 1024:
        raise ValueError('NPZ size limit')
    with zipfile.ZipFile(file) as archive:
        info = archive.infolist()
        if (len(info) > 32 or len({i.filename for i in info}) != len(info)
                or sum(i.file_size for i in info) > 256 * 1024 * 1024):
            raise ValueError('NPZ expansion/member limit')
    with np.load(file, allow_pickle=False) as data:
        source = data['joint_names']
        if source.ndim != 1 or source.dtype.kind != 'U':
            raise ValueError('Expected Unicode joint names; pickle forbidden')
        order = C.load_verified_module('control').permutation(source.tolist(), names)
        q, p, quat = (np.asarray(data[k], dtype=np.float64) for k in ('joint_pos', 'base_pos_w', 'base_quat_w'))
        fps_array = data['framerate']
        if fps_array.size != 1:
            raise ValueError('Invalid framerate')
        fps = float(fps_array.item())
        n = len(q)
        if not math.isfinite(fps) or not 1 <= fps <= 1000:
            raise ValueError('Invalid framerate')
        if not 3 <= n <= 100000 or q.shape != (n, 29) or p.shape != (n, 3) or quat.shape != (n, 4):
            raise ValueError('Invalid motion tensor shapes')
        if not all(np.isfinite(a).all() for a in (q, p, quat)):
            raise ValueError('Nonfinite motion values')
        q = q[:, order]
        lo, hi, vmax = (np.array([model[name][key] for name in names]) for key in ('lower', 'upper', 'velocity'))
        excess = np.maximum(np.maximum(lo-q, q-hi), 0)
        speed = np.abs(np.diff(q, axis=0)) * fps
        position_bad, speed_bad = excess > EPS, speed > vmax + EPS
        norms = np.linalg.norm(quat, axis=1)
        norm_error = float(np.abs(norms - 1).max())
        joints = []
        for j, name in enumerate(names):
            if not position_bad[:, j].any() and not speed_bad[:, j].any():
                continue
            frame = int(np.argmax(excess[:, j]))
            interval = int(np.argmax(speed[:, j]))
            joints.append(dict(name=name, lower_rad=lo[j], upper_rad=hi[j],
                observed_min_rad=float(q[:, j].min()), observed_max_rad=float(q[:, j].max()),
                violated_frames=int(position_bad[:, j].sum()), peak_limit_excess_rad=float(excess[:, j].max()),
                peak_excess_frame_zero_based=frame, peak_excess_time_s=frame/fps,
                velocity_limit_rad_s=vmax[j], peak_step_speed_rad_s=float(speed[:, j].max()),
                overspeed_intervals=int(speed_bad[:, j].sum()), peak_speed_interval_zero_based=interval))
        return dict(name=file.name, sha256=C.sha256(file), frames=n, fps=fps, duration_s=(n-1)/fps,
            body_channels=29, head_channels=0, position_violating_frames=int(position_bad.any(axis=1).sum()),
            peak_position_excess_rad=float(excess.max()), peak_position_excess_deg=float(np.rad2deg(excess.max())),
            overspeed_intervals=int(speed_bad.any(axis=1).sum()), max_quaternion_norm_error=norm_error,
            numerical_limits_pass=bool(not position_bad.any() and not speed_bad.any() and norm_error <= 1e-3),
            root_height_m=[float(p[:, 2].min()), float(p[:, 2].max())], joints=joints,
            accepted_reference=False, robot_identity_verified=False,
            quaternion_order_verified=False, root_frame_verified=False, contact_replay_verified=False)


def audit_metadata(path, root):
    if path.stat().st_size > 65536:
        raise ValueError('Metadata size limit')
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError('Expected metadata mapping')
    references = []
    for group, key, suffix in (('motion_files', 'motion_file', '.npz'), ('terrains', 'terrain_file', '.stl')):
        rows = data.get(group)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
            raise ValueError('Invalid metadata entries')
        ids = set()
        for row in rows:
            if not isinstance(row, dict) or type(row.get('terrain_id')) is not int:
                raise ValueError('Invalid terrain id')
            if group == 'terrains':
                if row['terrain_id'] in ids:
                    raise ValueError('Duplicate terrain id')
                ids.add(row['terrain_id'])
            file = contained(root, row[key])
            if file.suffix != suffix:
                raise ValueError('Unexpected metadata file type')
            references.append(dict(group=group, path=row[key], terrain_id=row['terrain_id'], exists=file.is_file()))
    terrain_ids = {r['terrain_id'] for r in references if r['group'] == 'terrains'}
    ids_match = all(r['terrain_id'] in terrain_ids for r in references if r['group'] == 'motion_files')
    return dict(name=path.name, sha256=C.sha256(path), references=references,
        file_and_id_links_resolve=bool(ids_match and all(r['exists'] for r in references)),
        terrain_frame_or_units_verified=False, accepted_reference=False)


def audit_binary_stl(path):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('STL size limit')
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError('Short binary STL')
    count = struct.unpack_from('<I', raw, 80)[0]
    if not count or len(raw) != 84 + 50 * count:
        raise ValueError('Only exact-length binary STL supported')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    mesh = np.frombuffer(raw, dtype=dtype, count=count, offset=84)
    vertices = mesh['vertices'].reshape(-1, 3)
    if not np.isfinite(vertices).all() or not np.isfinite(mesh['normal']).all():
        raise ValueError('Nonfinite mesh')
    return dict(name=path.name, sha256=C.sha256(path), triangle_count=count,
        bounds_raw_units=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        units_verified=False, collision_or_contact_validated=False)


def audit_dataset(root, urdf):
    root = Path(root).resolve()
    names, model = C.validate_model(urdf)
    files = sorted(root.iterdir())
    candidates = [f for f in files if f.suffix in ('.npz', '.yaml', '.stl')]
    if not candidates or len(candidates) > 100:
        raise ValueError('Expected 1..100 candidate files')
    report = dict(schema_version=1, model_sha256=C.sha256(urdf), numeric_comparison_tolerance=EPS,
        motions=[], metadata=[], terrains=[], errors=[], accepted_reference=False, training_ready=False,
        gpu_submissions=0, solver_calls=0, source_data_modified=False,
        limitations=['Named joint sets do not establish matching kinematics or source robot identity',
            'Step speeds are abs(diff(q))*fps, not measured actuator velocities',
            'No frame, quaternion-order, mesh-unit, FK/contact, dynamics or training acceptance',
            'Terrain-linked climbing data is not automatically a valid feet-only Plane reference'])
    for file in candidates:
        try:
            file = contained(root, file.name)
            if file.suffix == '.npz':
                report['motions'].append(audit_npz(file, names, model))
            elif file.suffix == '.yaml':
                report['metadata'].append(audit_metadata(file, root))
            else:
                report['terrains'].append(audit_binary_stl(file))
        except (ValueError, KeyError, OSError, zipfile.BadZipFile, yaml.YAMLError) as exc:
            report['errors'].append(dict(name=file.name, error=str(exc)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--report', type=Path, help='New output path only; never overwrite input data')
    args = parser.parse_args()
    report = audit_dataset(args.dataset, args.urdf)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.report:
        with args.report.open('x') as stream:
            stream.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()