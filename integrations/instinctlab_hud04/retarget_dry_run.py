"""G1 -> HUD04 deterministic preparation ONLY; no solve/execute mode exists.

Evaluates declared root/quaternion hypotheses without choosing one, never rewrites
motions, assigns contact labels or creates an accepted manifest. No network calls.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def local_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = local_module('retarget_motion_audit', 'audit_motion.py')
K = local_module('retarget_kinematic_diagnostics', 'kinematic_diagnostics.py')
SOURCE_MODEL_SHA256 = 'b25f61293db260f8b936b456578f5fff76152d28b368f5bcb343dca1d9d6a203'
TARGET_MODEL_SHA256 = '865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7'
MOTION_SHA256 = '56158e8679580910ccff375963d025b138636acfe70f5b1f88381837209b2056'
TERRAIN_SHA256 = '3aee33efea6793974d901dd0f7e46859e2d12d2dfd79f38f12c777686566c2de'
WINDOWS = ((611, 641), (659, 689), (711, 741))  # Zero-based, inclusive; independent windows.


def require_hash(path, expected):
    if A.C.sha256(path) != expected:
        raise ValueError('Pinned input SHA256 mismatch: ' + Path(path).name)


def distribution(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return dict(samples=int(values.size), finite_samples=int(finite.size),
                percentiles=None if not len(finite) else dict(zip(
                    ('min', 'p05', 'median', 'p95', 'max'), np.percentile(finite, [0, 5, 50, 95, 100]).tolist())))


def foot_diagnostic(world_frames, tree, link, triangles, selected, fps):
    probes = K.transform_points(world_frames[link], tree.box_probes(link))
    selected_points = probes[selected]
    height = K.upper_envelope_height(triangles, selected_points[:, :, :2].reshape(-1, 2))
    gaps = selected_points[:, :, 2].reshape(-1) - height
    # Speeds use each sample's actual preceding source frame, NEVER bridge disjoint windows.
    velocities = np.linalg.norm(probes[1:]-probes[:-1], axis=2) * fps
    selected_intervals = np.array([i for start, end in WINDOWS for i in range(start, end)])
    return dict(link=link, probe_type='eight_full_collision_box_corners',
        raw_vertex_vertical_gap_to_upper_envelope=distribution(gaps),
        vertex_samples_below_upper_envelope_by_more_than_0p005_raw_units=int(np.sum(gaps < -.005)),
        vertex_samples_with_abs_gap_within_0p020_raw_units=int(np.sum(np.abs(gaps) <= .02)),
        raw_corner_speed_inside_windows=distribution(velocities[selected_intervals]),
        contact_label=None, signed_penetration_distance=False, full_collision_test=False,
        assumes_common_raw_world_frame_and_units=True, raw_world_units_verified=False)


def prepare(source_urdf, target_urdf, motion_path, terrain_path):
    paths = [Path(p).resolve() for p in (source_urdf, target_urdf, motion_path, terrain_path)]
    expected = (SOURCE_MODEL_SHA256, TARGET_MODEL_SHA256, MOTION_SHA256, TERRAIN_SHA256)
    for path, digest in zip(paths, expected):
        require_hash(path, digest)
    source, target, motion_path, terrain_path = paths
    names, model = A.C.validate_model(target)
    motion_audit = A.audit_npz(motion_path, names, model)
    if motion_audit['frames'] != 997 or motion_audit['fps'] != 90:
        raise ValueError('Expected reviewed 997-frame 90-Hz motion')
    source_tree, target_tree = K.RobotTree.read(source), K.RobotTree.read(target)
    if source_tree.root != 'torso_link' or target_tree.root != 'base_link':
        raise ValueError('Pinned root contract mismatch')
    with np.load(motion_path, allow_pickle=False) as data:
        q = data['joint_pos'].astype(float)
        stored_names = data['joint_names'].tolist()
        position, quaternion = (data[key].astype(float) for key in ('base_pos_w', 'base_quat_w'))
    source_fk = source_tree.fk(q, stored_names)
    triangles = K.read_triangles(terrain_path)
    selected = np.array([i for start, end in WINDOWS for i in range(start, end+1)])
    hypotheses = []
    for order in ('wxyz', 'xyzw'):
        anchor_pose = K.world_pose(position, quaternion, order)
        for anchor in ('torso_link', 'pelvis'):
            world = K.at_world_anchor(source_fk, anchor_pose, anchor)
            feet = [foot_diagnostic(world, source_tree, side+'_ankle_roll_link', triangles,
                                    selected, motion_audit['fps']) for side in ('left', 'right')]
            hypotheses.append(dict(npz_root_link_hypothesis=anchor, quaternion_order_hypothesis=order,
                selected_or_accepted=False, feet=feet,
                wrist_origins_not_calibrated_palms={side: dict(
                    link=side+'_wrist_yaw_link',
                    raw_world_xyz_bounds=[world[side+'_wrist_yaw_link'][selected, :3, 3].min(axis=0).tolist(),
                                          world[side+'_wrist_yaw_link'][selected, :3, 3].max(axis=0).tolist()])
                    for side in ('left', 'right')}))
    root_to_pelvis = source_fk['pelvis']
    relative_rotation = root_to_pelvis[0, :3, :3].T @ root_to_pelvis[:, :3, :3]
    angle = np.arccos(np.clip((np.trace(relative_rotation, axis1=1, axis2=2)-1)/2, -1, 1))
    target_q = np.array([[model[name]['default'] for name in names]])
    target_fk = target_tree.fk(target_q, names)
    nominal = {}
    for side in ('left', 'right'):
        foot, hand = side+'_ankle_roll_link', side+'_wrist_roll_link'
        corners = K.transform_points(target_fk[foot], target_tree.box_probes(foot))[0]
        nominal[side] = dict(foot_box_xyz_bounds_in_target_root=[corners.min(axis=0).tolist(), corners.max(axis=0).tolist()],
                             hand_link=hand, hand_origin_in_target_root=target_fk[hand][0, :3, 3].tolist())
    # Detect concurrent source/model changes as well as preserving bytes ourselves.
    for path, digest in zip(paths, expected):
        require_hash(path, digest)
    return dict(schema_version=1, mode='deterministic_retarget_preparation_only',
        source_family='G1', source_model_is_candidate_not_verified_export_model=True,
        source_model_sha256=SOURCE_MODEL_SHA256, target_model_sha256=TARGET_MODEL_SHA256,
        motion_sha256=MOTION_SHA256, terrain_sha256=TERRAIN_SHA256,
        windows_zero_based_inclusive=[list(w) for w in WINDOWS], selected_frame_count=int(len(selected)),
        source_frames=997, source_fps=90, body_channels=29, head_channels=0,
        actual_source_model_root=source_tree.root, actual_target_model_root=target_tree.root,
        source_numeric_audit_against_target=motion_audit,
        source_torso_to_pelvis_motion=dict(
            translation_raw_bounds=[root_to_pelvis[:, :3, 3].min(axis=0).tolist(), root_to_pelvis[:, :3, 3].max(axis=0).tolist()],
            max_translation_change_from_first_raw=float(np.linalg.norm(root_to_pelvis[:, :3, 3]-root_to_pelvis[0, :3, 3], axis=1).max()),
            max_orientation_change_from_first_deg=float(np.rad2deg(angle.max())),
            fixed_offset_conversion_valid=False),
        hypotheses=hypotheses, target_unchanged_default_geometry=nominal,
        proposed_solver_budget_not_executed=dict(independent_windows=3, frames_per_window=31,
            cpu_threads_max=2, gpu_count=0, outer_iterations_per_window_max=100,
            wall_seconds_per_window_max=90, total_wall_seconds_max=600, retries=0, parameter_search=False),
        blockers=['Exact generating G1 model and NPZ root convention not proven',
            'Quaternion order and common world/mesh units/frame not proven',
            'Palm/sole contact frames and hand/foot/knee support schedule not reviewed',
            'No optimizer/supervisor/feasibility or dynamics implementation in this entry point'],
        source_data_modified=False, candidate_motion_generated=False, contact_labels_generated=False,
        solver_implemented=False, solver_calls=0, gpu_submissions=0, ppo_updates=0,
        accepted_reference=False, training_ready=False,
        limitations=['Four hypotheses are inspected, never ranked or selected as ground truth',
            'Upper-envelope vertical gaps are not signed distances, contact labels or full mesh collisions',
            'Default target geometry is not a copied-q retarget or a standing solution',
            'No values changed, no rate/limits relaxed, no dataset or accepted manifest exported'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-urdf', 'target-urdf', 'motion', 'terrain'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True, help='New JSON path only; no motion export')
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('Report already exists; never overwrite')
    report = prepare(args.source_urdf, args.target_urdf, args.motion, args.terrain)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('mode', 'selected_frame_count', 'solver_calls',
        'gpu_submissions', 'accepted_reference', 'blockers')}, indent=2))


if __name__ == '__main__':
    main()