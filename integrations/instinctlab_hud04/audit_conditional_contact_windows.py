"""Conditional raw-terrain probes of configured G1 feet/hands/knees; no acceptance.

One world-down support extremum PER primitive is a reproducible geometry probe,
not full mesh collision or signed penetration. Vertical walls/overhangs are not
resolved by an upper XY envelope. No hypothesis is chosen or ranked.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import audit_source_contact_model as S
import contact_geometry as G
import kinematic_diagnostics as K
import retarget_dry_run as R


def world_down_support(primitive, transforms):
    """Extremal world-down point, independently for each input link pose."""
    transforms = np.asarray(transforms, dtype=float)
    if (transforms.ndim != 3 or transforms.shape[1:] != (4, 4)
            or not 1 <= len(transforms) <= 4096 or not np.isfinite(transforms).all()
            or not np.allclose(transforms[:, 3, :], [0, 0, 0, 1], atol=1e-10)):
        raise ValueError('Bounded finite rigid transforms required')
    rot = transforms[:, :3, :3]
    if not np.allclose(np.swapaxes(rot,1,2)@rot, np.eye(3), atol=1e-10) or not np.allclose(np.linalg.det(rot),1,atol=1e-10):
        raise ValueError('Expected proper rotations')
    directions = -rot[:, 2, :]
    local = G.support(primitive, directions)
    return np.einsum('nij,nj->ni',rot,local)+transforms[:, :3, 3]


def raw_gap_report(points, triangles):
    points = np.asarray(points, dtype=float).reshape(-1,3)
    height = K.upper_envelope_height(triangles, points[:, :2])
    gaps = points[:, 2]-height
    return dict(probe_points=len(points),outside_raw_xy_envelope=int(np.sum(~np.isfinite(height))),
        vertical_gap_raw_units=R.distribution(gaps),
        below_upper_envelope_by_more_than_0p005_raw_units=int(np.sum(gaps < -.005)),
        abs_gap_within_0p020_raw_units=int(np.sum(np.abs(gaps) <= .020)),
        contact_label=None,signed_penetration_distance=False,full_collision_test=False,
        ground_contact_vs_robot_contact_disambiguated=False)


def audit(source_urdf, motion, terrain):
    inputs = list(map(lambda p:Path(p).resolve(),(source_urdf,motion,terrain)))
    expected = (S.TASK_MODEL_SHA256,R.MOTION_SHA256,R.TERRAIN_SHA256)
    for path,digest in zip(inputs,expected):
        R.require_hash(path,digest)
    source_urdf,motion,terrain = inputs
    tree = K.RobotTree.read(source_urdf)
    with np.load(motion,allow_pickle=False) as data:
        q,names = data['joint_pos'].astype(float),data['joint_names'].tolist()
        position,quaternion = (data[k].astype(float) for k in ('base_pos_w','base_quat_w'))
        if q.shape != (997,29) or float(data['framerate'].item()) != 90:
            raise ValueError('Expected reviewed 997-frame 90-Hz motion')
    fk = tree.fk(q,names)
    selected = np.array([i for start,end in R.WINDOWS for i in range(start,end+1)])
    triangles = K.read_triangles(terrain)
    primitives = {side+'_'+suffix:G.collisions(source_urdf,side+'_'+suffix,replace_cylinders=True)
        for side in ('left','right') for suffix in ('ankle_roll_link','wrist_yaw_link','knee_link')}
    hypotheses = []
    for order in ('wxyz','xyzw'):
        anchor_pose = K.world_pose(position,quaternion,order)
        for anchor in ('torso_link','pelvis'):
            world = K.at_world_anchor(fk,anchor_pose,anchor)
            links = {}
            for link,shapes in primitives.items():
                points = np.stack([world_down_support(p,world[link][selected]) for p in shapes],axis=1)
                links[link] = dict(primitive_count=len(shapes),**raw_gap_report(points,triangles),
                    world_probe_xyz_bounds=[points.reshape(-1,3).min(axis=0).tolist(),
                                            points.reshape(-1,3).max(axis=0).tolist()])
            hypotheses.append(dict(npz_anchor_hypothesis=anchor,quaternion_order_hypothesis=order,
                selected_or_accepted=False,links=links))
    for path,digest in zip(inputs,expected):
        R.require_hash(path,digest)
    return dict(stage='conditional_raw_terrain_probe_only',source_model_sha256=expected[0],
        motion_sha256=expected[1],terrain_sha256=expected[2],source_frames=997,source_fps=90,
        windows_zero_based_inclusive=R.WINDOWS,selected_frame_count=len(selected),
        point_rule='one world-down support extremum per nominal converted capsule per frame',
        hypotheses=hypotheses,hypotheses_ranked=False,source_export_provenance_verified=False,
        source_root_and_quaternion_convention_verified=False,world_units_and_alignment_verified=False,
        raw_terrain_only=True,actual_terrain_crop_and_border_shift_applied=False,
        vertical_walls_and_overhang_contacts_resolved=False,calibrated_palms_or_soles=False,
        velocities_of_support_points_reported=False,contact_schedule_generated=False,
        source_data_modified=False,candidate_motion_generated=False,solver_calls=0,
        gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-urdf','motion','terrain','report'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    output = args.report.resolve()
    if output.suffix != '.json' or output.exists() or output in {p.resolve() for p in (args.source_urdf,args.motion,args.terrain)}:
        parser.error('New separate .json report required')
    result = audit(args.source_urdf,args.motion,args.terrain)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(report=str(output),selected_frame_count=result['selected_frame_count'],
                          hypotheses=4,solver_calls=0,gpu_submissions=0,training_ready=False)))


if __name__ == '__main__':
    main()