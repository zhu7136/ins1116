"""Bounded CPU terrain/surface diagnostics; not contact labels or acceptance.

Four coordinate hypotheses remain unselected. No retargeting, GPU or source edits.
Requires explicit local reviewed inputs and an unoccupied output JSON path.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import audit_conditional_contact_windows as C
import audit_source_contact_model as S
import contact_geometry as G
import kinematic_diagnostics as K
import retarget_dry_run as R
import terrain_surface_geometry as T


def audit(source_urdf,motion,terrain,tile_size):
    paths=list(map(lambda p:Path(p).resolve(),(source_urdf,motion,terrain)))
    hashes=[S.TASK_MODEL_SHA256,R.MOTION_SHA256,R.TERRAIN_SHA256]
    for p,h in zip(paths,hashes):R.require_hash(p,h)
    source_urdf,motion,terrain=paths
    mesh,shift,terrain_report=T.processed_terrain(terrain,tile_size)
    clipped=np.asarray(mesh.triangles)
    lateral=np.abs(mesh.face_normals[:,2])<=.1
    surface=T.SurfaceIndex(clipped)
    lateral_ids=np.flatnonzero(lateral)
    if not len(lateral_ids):raise ValueError('Expected reviewed terrain with lateral faces')
    lateral_surface=T.SurfaceIndex(clipped[lateral_ids])
    terrain_report.update(lateral_definition='abs(unit_face_normal_z)<=0.1',
        lateral_faces=int(lateral.sum()),lateral_area_raw_squared=float(mesh.area_faces[lateral].sum()),
        total_area_raw_squared=float(mesh.area),watertight=bool(mesh.is_watertight))
    with np.load(motion,allow_pickle=False) as data:
        q=data['joint_pos'].astype(float);names=data['joint_names'].tolist()
        position=data['base_pos_w'].astype(float);quaternion=data['base_quat_w'].astype(float)
        if q.shape!=(997,29) or float(data['framerate'].item())!=90:
            raise ValueError('Reviewed 997-frame 90-Hz input required')
    tree=K.RobotTree.read(source_urdf);fk=tree.fk(q,names)
    selected=np.array([i for a,b in R.WINDOWS for i in range(a,b+1)])
    primitives={side+'_'+suffix:G.collisions(source_urdf,side+'_'+suffix,replace_cylinders=True)
        for side in ('left','right') for suffix in ('ankle_roll_link','wrist_yaw_link','knee_link')}
    raw=K.read_triangles(terrain)
    hypotheses=[];witnesses=[];height_changes=0;probe_count=0;max_height_change=0.;max_shared_shift_error=0.
    for order in ('wxyz','xyzw'):
        anchor_pose=K.world_pose(position,quaternion,order)
        for anchor in ('torso_link','pelvis'):
            world=K.at_world_anchor(fk,anchor_pose,anchor)
            links={}
            for link,shapes in primitives.items():
                transforms=world[link][selected]
                points=np.stack([C.world_down_support(p,transforms) for p in shapes],axis=1)
                flat=points.reshape(-1,3)
                probe_count+=len(flat)
                raw_h=K.upper_envelope_height(raw,flat[:,:2])
                crop_h=K.upper_envelope_height(clipped,flat[:,:2])
                shifted_h=K.upper_envelope_height(clipped+shift,(flat+shift)[:,:2])
                finite=np.isfinite(raw_h)&np.isfinite(crop_h)
                delta=np.abs(raw_h[finite]-crop_h[finite])
                changed=int(np.sum(delta>1e-9))+int(np.sum(np.isfinite(raw_h)!=np.isfinite(crop_h)))
                height_changes+=changed
                if len(delta):max_height_change=max(max_height_change,float(delta.max()))
                shifted_finite=np.isfinite(shifted_h)&np.isfinite(crop_h)
                if shifted_finite.any():
                    max_shared_shift_error=max(max_shared_shift_error,float(np.max(np.abs(
                        (flat[:,2]-crop_h)[shifted_finite]-(flat[:,2]+shift[2]-shifted_h)[shifted_finite]))))
                margins=[];lateral_margins=[];min_witness=None
                for primitive_id,p in enumerate(shapes):
                    if p['kind']!='capsule':raise ValueError('Reviewed source capsules required')
                    axis=np.array([[0,0,-p['length']/2],[0,0,p['length']/2]])
                    local=axis@p['transform'][:3,:3].T+p['transform'][:3,3]
                    endpoints=K.transform_points(transforms,local)
                    pm=[];lm=[]
                    for i,(a,b) in enumerate(endpoints):
                        distance,face,ps,pt=surface.closest(a,b)
                        latdistance,latface,_,_=lateral_surface.closest(a,b)
                        margin=distance-p['radius'];pm.append(margin);lm.append(latdistance-p['radius'])
                        if min_witness is None or margin<min_witness['surface_margin_raw']:
                            min_witness=dict(anchor=anchor,order=order,link=link,primitive=primitive_id,
                                frame=int(selected[i]),start_raw=a.tolist(),end_raw=b.tolist(),radius_raw=p['radius'],
                                surface_margin_raw=float(margin),centerline_distance_raw=distance,face_index=face,
                                closest_axis_point_raw=ps.tolist(),closest_triangle_point_raw=pt.tolist(),
                                face_normal=mesh.face_normals[face].tolist())
                    margins.append(pm);lateral_margins.append(lm)
                margins=np.array(margins).T;lateral_margins=np.array(lateral_margins).T
                per_frame=margins.min(axis=1);lateral_per_frame=lateral_margins.min(axis=1)
                witnesses.append(min_witness)
                links[link]=dict(primitive_count=len(shapes),
                    world_down_probes=len(flat),raw_vs_cropped_height_changed_probes=changed,
                    raw_below_upper_envelope_by_more_than_0p005=int(np.sum(flat[:,2]-raw_h<-.005)),
                    cropped_below_upper_envelope_by_more_than_0p005=int(np.sum(flat[:,2]-crop_h<-.005)),
                    outside_raw_envelope=int(np.sum(~np.isfinite(raw_h))),
                    outside_cropped_envelope=int(np.sum(~np.isfinite(crop_h))),
                    min_per_frame_capsule_surface_margin_raw=R.distribution(per_frame),
                    capsule_surface_intersection_candidate_frames=int(np.sum(per_frame<-.005)),
                    any_capsule_abs_surface_margin_within_0p020_frames=int(np.sum(np.any(np.abs(margins)<=.020,axis=1))),
                    any_capsule_abs_lateral_margin_within_0p020_frames=int(np.sum(np.any(np.abs(lateral_margins)<=.020,axis=1))),
                    near_any_surface_candidate_intervals=T.candidate_intervals(selected,np.min(np.abs(margins),axis=1),.020),
                    near_lateral_surface_candidate_intervals=T.candidate_intervals(selected,np.min(np.abs(lateral_margins),axis=1),.020),
                    per_frame_min_surface_margin_raw=per_frame.tolist(),
                    per_frame_min_lateral_margin_raw=lateral_per_frame.tolist(),
                    accepted_support_intervals=None,contact_label=None)
                print('finished',anchor,order,link,flush=True)
            hypotheses.append(dict(anchor_hypothesis=anchor,quaternion_order_hypothesis=order,links=links,
                selected_or_accepted=False))
    for p,h in zip(paths,hashes):R.require_hash(p,h)
    if probe_count!=6696:raise ValueError('Reviewed probe cardinality mismatch')
    return dict(stage='conditional_processed_terrain_surface_diagnostics',
        source_model_sha256=hashes[0],motion_sha256=hashes[1],terrain_sha256=hashes[2],
        source_frames=997,source_fps=90,selected_frames=selected.tolist(),
        windows_zero_based_inclusive=R.WINDOWS,tile_size_raw=list(map(float,tile_size)),terrain=terrain_report,
        conditional_world_down_probe_points=probe_count,
        raw_vs_cropped_height_changed_probe_points=height_changes,
        raw_vs_cropped_max_finite_height_change=max_height_change,
        max_common_translation_gap_error=max_shared_shift_error,
        surface_margin_definition='unsigned_distance(capsule_centerline,triangle_surface)-radius',
        thresholds_raw_units={'surface_margin_candidate_band':.020,'intersection_candidate_below':-.005},
        hypotheses=hypotheses,independent_verification_witnesses=witnesses,
        hypotheses_ranked=False,exporter_provenance_verified=False,root_quaternion_conventions_verified=False,
        world_units_and_alignment_verified=False,contact_schedule_generated=False,
        signed_solid_distance=False,solid_containment_excluded=False,physx_cooked_contacts_verified=False,
        source_data_modified=False,candidate_motion_generated=False,solver_calls=0,gpu_submissions=0,
        ppo_updates=0,accepted_reference=False,training_ready=False,
        limitations=['Installed CPU trimesh version is explicit, not proof of original GPU environment',
            'Nonflat border mean is loader behavior, not a calibrated ground plane',
            'Surface intersections are not signed penetration depths; containment may remain undetected',
            'Near-surface intervals are geometric review candidates, not support/contact labels',
            'Positive surface margin or fewer near/intersection candidates cannot select coordinate provenance',
            'Only reviewed source G1 nominal capsules considered; target feasibility/self-collision not solved'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-urdf','motion','terrain','report'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--tile-size',nargs=2,type=float,required=True)
    args=parser.parse_args();out=args.report.resolve()
    if out.exists() or out.suffix!='.json':parser.error('New .json output path required')
    result=audit(args.source_urdf,args.motion,args.terrain,args.tile_size)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'report':str(out),'stage':result['stage'],'training_ready':False}))


if __name__=='__main__':main()