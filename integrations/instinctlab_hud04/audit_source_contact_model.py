"""Pinned upstream task-model/contact preflight; no solve, replay or training.

This establishes the G1 task's configured interpretation, NOT provenance of the
supplied NPZ. Earlier simplified-model diagnostics remain conditional evidence.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import contact_geometry as G
import kinematic_diagnostics as K
import retarget_dry_run as R

TASK_MODEL_SHA256 = '5e99d930af64f3f0bfd34235239e288cc380ba62c8b01fcfcf8cc9f0996e7fa4'
UPSTREAM_COMMIT = 'de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057'
ISAAC_SIM_COMMIT = '47d886f2858d1ceed556b21c88927aa67bc81c12'
HAND_MESH_HASHES = {
    'left': '5aa0af4ccdaf46d498267b0fae247b8d547d8d72dd8db4f7691fe42b567eb5d6',
    'right': 'e4804755e9991f05ad5389340e1e83ee1ef687a403896c21c0f785c28ff871fc',
}


def marker_check(tree, side, foot_bounds):
    suffix = {'left': 'L', 'right': 'R'}[side]
    output = {}
    for label in ('center', 'heel', 'tip'):
        child = 'contact_foot_' + label + '_' + suffix
        joint, = [j for j in tree.joints if j['child'] == child]
        if joint['kind'] != 'fixed' or joint['parent'] != side + '_ankle_roll_link':
            raise ValueError('Unexpected foot marker graph')
        xyz = joint['origin'][:3, 3]
        lo, hi = np.array(foot_bounds)
        outside = np.maximum(np.maximum(lo - xyz, xyz - hi), 0)
        output[child] = dict(xyz_in_ankle_roll_link=xyz.tolist(),
            delta_from_collision_bounds_bottom_center=(xyz - np.array([(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]])).tolist(),
            outside_collision_bounds_by_axis=outside.tolist(),
            above_collision_bottom=float(xyz[2]-lo[2]),
            accepted_contact_target=False)
    return output


def audit(task_urdf, earlier_urdf, target_urdf, motion):
    paths = [Path(p).resolve() for p in (task_urdf, earlier_urdf, target_urdf, motion)]
    hashes = (TASK_MODEL_SHA256, R.SOURCE_MODEL_SHA256, R.TARGET_MODEL_SHA256, R.MOTION_SHA256)
    for p, digest in zip(paths, hashes):
        R.require_hash(p, digest)
    task_urdf, earlier_urdf, target_urdf, motion = paths
    task, earlier, target = [K.RobotTree.read(p) for p in paths[:3]]
    if task.root != 'torso_link' or target.root != 'base_link' or len(task.movable) != 29:
        raise ValueError('Pinned task/target root/DOF mismatch')
    with np.load(motion, allow_pickle=False) as data:
        q, names = data['joint_pos'].astype(float), data['joint_names'].tolist()
        if q.shape != (997, 29) or float(data['framerate'].item()) != 90:
            raise ValueError('Expected reviewed 997-frame 90-Hz source')
    fk_task, fk_earlier = task.fk(q, names), earlier.fk(q, names)
    per_link = {}
    for link in sorted(set(task.links) & set(earlier.links)):
        delta = fk_task[link][:, :3, 3] - fk_earlier[link][:, :3, 3]
        per_link[link] = dict(max_origin_displacement=float(np.linalg.norm(delta, axis=1).max()),
            max_rotation_element_difference=float(np.abs(fk_task[link][:, :3, :3] - fk_earlier[link][:, :3, :3]).max()))
    source_geometry, target_geometry, markers = {}, {}, {}
    mesh_allowlist = {}
    for side in ('left', 'right'):
        mesh = target_urdf.parent.parent / 'meshes' / 'HU_D04_01' / (side+'_wrist_roll_link.STL')
        if G.sha256(mesh) != HAND_MESH_HASHES[side]:
            raise ValueError('Pinned HUD04 hand mesh mismatch')
        mesh_allowlist[mesh] = HAND_MESH_HASHES[side]
        for suffix in ('ankle_roll_link', 'knee_link', 'wrist_yaw_link'):
            link = side + '_' + suffix
            source_geometry[link] = G.describe(G.collisions(task_urdf, link, replace_cylinders=True))
        # HUD04's distal roll follows yaw and pitch; a same-named wrist-yaw
        # origin is NOT an equivalent hand frame and has no collider here.
        for suffix in ('ankle_roll_link', 'knee_link', 'wrist_pitch_link', 'wrist_roll_link'):
            link = side + '_' + suffix
            target_geometry[link] = G.describe(G.collisions(target_urdf, link,
                replace_cylinders=False, allowed_meshes=mesh_allowlist))
        markers.update(marker_check(target, side, target_geometry[side+'_ankle_roll_link']['local_bounds']))
    attributes = []
    by_model = [{j.get('name'): j.find('limit') for j in ET.parse(p).getroot().findall('joint')}
                for p in (earlier_urdf, task_urdf)]
    for name in sorted(set(by_model[0]) & set(by_model[1])):
        a, b = [None if table[name] is None else table[name].attrib for table in by_model]
        if a != b:
            attributes.append(dict(joint=name, earlier=a, upstream_task=b))
    for p, digest in zip(paths, hashes):
        R.require_hash(p, digest)
    for mesh, digest in mesh_allowlist.items():
        R.require_hash(mesh, digest)
    return dict(stage='configured_upstream_task_model_and_collision_geometry_only',
        upstream_commit=UPSTREAM_COMMIT, isaac_sim_importer_commit=ISAAC_SIM_COMMIT,
        task_model_sha256=TASK_MODEL_SHA256, earlier_candidate_sha256=R.SOURCE_MODEL_SHA256,
        target_model_sha256=R.TARGET_MODEL_SHA256, motion_sha256=R.MOTION_SHA256,
        configured_upstream_asset='g1_29dof_torsobase_popsicle.urdf',
        configured_reference_fk_uses_same_asset=True,
        configured_upstream_root='torso_link', configured_loader_quaternion_order='wxyz',
        npz_export_model_and_conventions_proven=False,
        source_replace_cylinders_with_capsules=True, target_replace_cylinders_with_capsules=False,
        source_frames=997, body_channels=29, head_action_channels=0,
        earlier_candidate_comparison=dict(shared_links=len(per_link), per_link=per_link,
            max_origin_displacement=max(x['max_origin_displacement'] for x in per_link.values()),
            changed_joint_limit_attributes=attributes),
        source_collision_geometry=source_geometry, target_collision_geometry=target_geometry,
        target_existing_foot_markers=markers,
        limitations=['Configured loader interpretation is not proof of NPZ export provenance',
            'Earlier simplified-model FK/foot-box gaps cannot be reused for this task model',
            'Mesh support bounds are not cooked PhysX collision or a calibrated palm frame',
            'Existing HUD04 marker names do not establish their contact geometry accuracy',
            'No terrain alignment, physical contact labels or anatomical correspondence accepted'],
        source_data_modified=False, source_assets_modified=False, candidate_motion_generated=False,
        solver_calls=0, gpu_submissions=0, ppo_updates=0, accepted_reference=False, training_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('task-urdf', 'earlier-urdf', 'target-urdf', 'motion', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve()
    inputs = {p.resolve() for p in (args.task_urdf, args.earlier_urdf, args.target_urdf, args.motion)}
    if report.suffix != '.json' or report in inputs or report.exists():
        parser.error('A new non-input .json report path is required')
    result = audit(args.task_urdf, args.earlier_urdf, args.target_urdf, args.motion)
    with report.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(report=str(report), task_model_sha256=TASK_MODEL_SHA256,
        max_candidate_link_origin_displacement=result['earlier_candidate_comparison']['max_origin_displacement'],
        accepted_reference=False, training_ready=False, gpu_submissions=0)))


if __name__ == '__main__':
    main()