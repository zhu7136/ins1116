"""One bounded CPU algebra/conditional-correspondence review, never a solve.

Requires explicit source paths and a NEW output filename. Reuses pinned old
supervision; creates no trajectory, W&B run, GPU task or acceptance manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation
from context import (HERE,REPO,INTEGRATION,TARGET,BASE_COMMIT,DIAGNOSTIC,
                     DIAGNOSTIC_SHA,PINS,dependency,models,features,diagnostic,sha)
from pose_tasks import evaluate,neutral_orientation_proposal,build_joint_pose_window


def stats(value):
    return dict(mean_deg=float(np.mean(value)),p95_deg=float(np.percentile(value,95)),max_deg=float(np.max(value)))


def worker(gate,payload):
    gate.step()
    source_root=Path(payload['instinct_root']);dataset=Path(payload['dataset'])
    source_path=source_root/'source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf'
    raw_path=dataset/'kneelClimbStep1-x-0.1-ziwen-retargeted.npz'
    manifest_path=INTEGRATION/'MOTION_REVIEW_20261117.json'
    manifest=json.loads(manifest_path.read_text())
    original=manifest['motions']+manifest['metadata']+[manifest['terrain']]
    checks=[]
    archive=REPO/'datasets/hud04/source-only-20261117/hud04-source-only-20261117.zip'
    archive_receipt=json.loads((archive.parent/'LOCAL_ARCHIVE_RECEIPT.json').read_text())
    if sha(archive)!=archive_receipt['sha256']:raise ValueError('Source archive SHA mismatch')
    with zipfile.ZipFile(archive) as z:
        am=json.loads(z.read('ARCHIVE_MANIFEST.json'))
        for e in am['files']:
            b=z.read(e['path'])
            if len(b)!=e['bytes'] or hashlib.sha256(b).hexdigest()!=e['sha256']:
                raise ValueError('Archive entry mismatch')
        for row in original:
            raw=(dataset/row['name']).read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if digest!=row['sha256'] or raw!=z.read('original/'+row['name']):raise ValueError('Source bytes mismatch')
            checks.append(dict(name=row['name'],sha256=digest,bytes=len(raw),matches_existing_archive=True))
        chain=json.loads(z.read('evidence/source_interface_chain.json'))
        prior_geometry=json.loads(z.read('evidence/CONTACT_CORRESPONDENCE_GEOMETRY_REVIEW_20261117.json'))
    source_checks=[]
    for e in chain['source_checks']:
        if sha(source_root/e['path'])!=e['sha256']:raise ValueError('Pinned official consumer changed')
        source_checks.append(dict(path=e['path'],sha256=e['sha256'],verified_this_run=True))
    gate.step()
    direction=dependency('directions');source=direction.DirectionModel(source_path,'source',wrist_endpoint='first_wrist_pivot')
    target,point_model=models();fs=features();q=diagnostic(target)
    with np.load(raw_path,allow_pickle=False) as z:
        if sha(raw_path)!=direction.MOTION_SHA or float(z['framerate'].item())!=90:raise ValueError('Raw motion pin')
        sq=source.order(z['joint_pos'],z['joint_names'].tolist())
        source_base_pos=z['base_pos_w'].copy();source_base_quat=z['base_quat_w'].copy()
    source_links=['torso_link','left_wrist_yaw_link','right_wrist_yaw_link','left_ankle_roll_link','right_ankle_roll_link']
    sp,_=source.fk(sq);sz,_=source.fk(np.zeros((1,29)))
    source_bases=np.stack([np.swapaxes(sp[source.anchor][:,:3,:3],-1,-2)@sp[l][:,:3,:3] for l in source_links],axis=1)
    source_zero=np.stack([sz[source.anchor][0,:3,:3].T@sz[l][0,:3,:3] for l in source_links])
    _,target_zero=evaluate(target,np.zeros((1,29)),np.eye(4)[None],fs,jacobian=False)
    proposal,offsets=neutral_orientation_proposal(source_bases,source_zero,target_zero[0])
    _,saved_bases=evaluate(target,q,np.tile(np.eye(4),(997,1,1)),fs,jacobian=False)
    errors=np.rad2deg(Rotation.from_matrix((np.swapaxes(saved_bases,-1,-2)@proposal).reshape(-1,3,3)).magnitude()).reshape(997,5)
    orientation_summary=[]
    for k,f in enumerate(fs):
        row=dict(feature=f.name,source_link=source_links[k],target_link=f.link,
                 neutral_source_to_target_right_rotation=offsets[k].tolist(),
                 role=f.role,**stats(errors[:,k]))
        row['frame_of_max']=int(np.argmax(errors[:,k]));row['risk244_345']=stats(errors[244:346,k])
        orientation_summary.append(row)
    gate.step()
    # Source root read is interpretation-only. Never copy it into a target trajectory.
    old=dependency('independent_fk');source_world=old.world_pose(source_base_pos,source_base_quat,'wxyz')
    pelvis_world=source_world@sp['pelvis']
    source_independent=old.RobotTree.read(source_path).fk(sq,source.names)
    source_fk_error=max(float(np.max(np.abs(sp[l]-source_independent[l]))) for l in sp)
    pelvis_independent=source_world@source_independent['pelvis']
    root_error=float(np.max(np.abs(pelvis_world-pelvis_independent)))
    target_independent_all=old.RobotTree.read(TARGET).fk(q,target.names)
    target_bases_independent=np.stack([target_independent_all[f.link][:,:3,:3]@f.link_from_feature[:3,:3] for f in fs],axis=1)
    independent_source_bases=np.stack([np.swapaxes(source_independent[source.anchor][:,:3,:3],-1,-2)@source_independent[l][:,:3,:3] for l in source_links],axis=1)
    independent_relative=np.swapaxes(target_bases_independent,-1,-2)@(independent_source_bases@offsets)
    # Independent atan2 trace/skew angle, not scipy Rotation.magnitude used above.
    sines=np.linalg.norm(np.stack([independent_relative[...,2,1]-independent_relative[...,1,2],
                       independent_relative[...,0,2]-independent_relative[...,2,0],
                       independent_relative[...,1,0]-independent_relative[...,0,1]],axis=-1),axis=-1)/2
    cosines=(np.trace(independent_relative,axis1=-2,axis2=-1)-1)/2
    independent_errors=np.rad2deg(np.arctan2(sines,cosines))
    full_angle_recheck_error=float(np.max(np.abs(errors-independent_errors)))
    if full_angle_recheck_error>1e-9:raise ValueError('Independent full997 conditional rotation-angle mismatch')
    # A separate algebraic root, NOT the source pelvis, supplies finite-difference fixtures.
    ids=np.array([0,257,495,996]);small=q[ids]
    roots=np.tile(np.eye(4),(4,1,1));roots[:,:3,:3]=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();roots[:,:3,3]=[2,-1,.8]
    p,r,jp,jr=evaluate(target,small,roots,fs)
    tp=old.RobotTree.read(TARGET).fk(small,target.names)
    expected=np.stack([roots@tp[f.link]@f.link_from_feature for f in fs],axis=1)
    fk_error=max(float(np.max(np.abs(p-expected[:,:,:3,3]))),float(np.max(np.abs(r-expected[:,:,:3,:3]))))
    fd_point=fd_rotation=0.;eps=1e-6;perturb=dependency('named_target').perturb
    for c in range(35):
        d=np.zeros(35);d[c]=eps
        plus=evaluate(target,small+d[6:],np.array([perturb(a,d[:6]) for a in roots]),fs,jacobian=False)
        minus=evaluate(target,small-d[6:],np.array([perturb(a,-d[:6]) for a in roots]),fs,jacobian=False)
        fd_point=max(fd_point,float(np.max(np.abs((plus[0]-minus[0])/(2*eps)-jp[...,c]))))
        fd_rotation=max(fd_rotation,float(np.max(np.abs((plus[1]-minus[1])/(2*eps)-jr[...,c]))))
    pose_only=np.concatenate([jp.reshape(4,-1,35),jr.reshape(4,-1,35)],axis=1)
    _,directions_jac=target.evaluate(small,True)
    directions_jac=np.pad(directions_jac.reshape(4,24,29),((0,0),(0,0),(6,0)))
    combined=np.concatenate([pose_only,directions_jac],axis=1)
    pose_sv=np.linalg.svd(pose_only,compute_uv=False)
    sv=np.linalg.svd(combined,compute_uv=False);ranks=(sv>1e-8).sum(axis=1)
    if not np.all(ranks==35):raise ValueError('Expected locally full combined rank in these fixtures')
    if max(source_fk_error,root_error,fk_error)>1e-12 or max(fd_point,fd_rotation)>1e-8:raise ValueError('Independent FK/derivative mismatch')
    gate.step()
    # Same original native indices for q, root fixture, soft tasks and existing hard rows.
    frames=np.r_[np.arange(244,275),np.arange(480,511),np.arange(966,997)]
    algebraic_roots=np.repeat(roots[:1],len(frames),axis=0)
    desired_world_bases=algebraic_roots[:,None,:3,:3]@proposal[frames]
    new=build_joint_pose_window(dependency('window_problem').build,point_model,target,q[frames],target.names,
                               algebraic_roots,frames,fs,desired_bases=desired_world_bases,desired_directions=source.evaluate(sq)[frames])
    system=new['existing_window'];channels=new['pose_channels']
    prior=dependency('window_problem').build(point_model,q[frames],target.names,algebraic_roots,frames)
    if (system.hard_matrix!=prior.hard_matrix).nnz or not np.array_equal(system.hard_lower,prior.hard_lower) or not np.array_equal(system.hard_upper,prior.hard_upper):
        raise ValueError('Existing hard rows changed')
    # Every wrist has a rotationally symmetric capsule proxy. It does not define a palm normal.
    hand_evidence=[]
    for row in prior_geometry['rows']:
        primitive=row['hand']['source_nominal']['primitives'][0]
        hand_evidence.append(dict(side=row['side'],source_kind=primitive['kind'],radius=primitive['radius'],length=primitive['length'],
            source_collision_local_axis=np.asarray(primitive['transform'])[:3,2].tolist(),
            axial_rotation_symmetry=True,unique_palm_normal_inferable_from_collision_proxy=False,
            target_hand_frame_used='distal_wrist_URDF_frame_NOT_palm'))
    gate.step()
    bindings={str(p.relative_to(REPO)):sha(p) for p in [DIAGNOSTIC,TARGET,manifest_path,archive]}
    bindings.update({str(INTEGRATION/rel):digest for rel,digest in PINS.values()})
    if sha(DIAGNOSTIC)!=DIAGNOSTIC_SHA or any(sha(INTEGRATION/rel)!=digest for rel,digest in PINS.values()):raise ValueError('Prior inputs modified')
    return dict(stage='explicit_35d_pose_channels_and_conditional_neutral_mapping_NOT_reference',base_commit=BASE_COMMIT,
        raw_recovery=dict(files=checks,all_seven_verified=True,existing_archive_already_had_raw_inputs=True,
                          archive_entries_checked=len(am['files']),download_was_recovery_not_new_research=True),
        official_consumer=dict(root='torso_link',quaternion_order='wxyz',source_checks=source_checks,
             scope='intended release interface, not original exporter proof or source/STL physical units',
             source_root_evaluation_only=True,target_root_copied=False,source_all_links_independent_fk_error=source_fk_error,
             source_world_pelvis_independent_error=root_error),
        neutral_orientation_proposal=dict(formula='R_des=R_source_pelvis_local(q) @ R_source_pelvis_local(0).T @ R_target_base_local(0)',
             shared_anatomical_anchor_basis_is_explicit_unverified_assumption=True,
             neutral_pose_is_not_standing_or_joint_feasibility_evidence=True,
             full997_baseline_comparison_only=True,task_calibration_accepted=False,features=orientation_summary),
        independent_conditional_angle_recheck_max_deg=full_angle_recheck_error,
        hand_symmetry_evidence=hand_evidence,
        derivative_verification=dict(frames=ids.tolist(),root_fixture='fixed_algebraic_SE3_NOT_source_or_candidate_root',
             independent_all_feature_fk_max_error=fk_error,point_all35_finite_difference_error=fd_point,
             rotation_all35_finite_difference_error=fd_rotation,epsilon=eps,
             pose_only_local_rank=(pose_sv>1e-8).sum(axis=1).tolist(),
             pose_only_independent_conditions_upper_bound=30,
             combined_point_rotation_and_bone_direction_local_rank=ranks.tolist(),rank_tolerance=1e-8,
             minimum_singular_value=float(sv[:,-1].min()),global_feasibility_or_identifiability_proven=False),
        window=dict(native_frame_ranges=[[244,274],[480,510],[966,996]],native_fps=90,frames=len(frames),
             variables=35*len(frames),point_jacobian_shape=list(channels['point_jacobian'].shape),
             rotation_jacobian_shape=list(channels['basis_jacobian'].shape),hard_joint_shape=list(system.hard_matrix.shape),
             existing_bone_direction_jacobian_shape=list(channels['direction_jacobian'].shape),
             real_adjacent_edges=len(system.edges),cross_gap_edges=0,original_hard_rows_unchanged=True,
             zero_increment_max_joint_violation=float(max(0,system.hard_lower.max(),-system.hard_upper.min())),
             point_targets_and_residuals=None,orientation_targets='conditional_neutral_proposal_only_NOT_material_calibration',
             bone_direction_targets='existing_first_wrist_pivot_pairs_in_explicit_assumed_corresponding_pelvis_base_axes',
             units='point_coordinate_units_vs_dimensionless_rotation_basis_kept_in_separate_unweighted_channels',
             no_new_limits_weights_or_contact_thresholds_selected=True),
        preserved_input_sha256=bindings,new_code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        still_missing=['reviewed palm/foot/knee material-frame correspondence and support intervals',
             'source-world and actual with-wall Terrain physical-unit/alignment acceptance',
             'actual Terrain/self-collision constraint assembly and independent full motion checks',
             'fresh explicitly bounded optimizer authorization; old QP budget stays zero',
             'actual full-task reset, tracking, depth and lifecycle GPU acceptance'],
        solver_import_setup_solve=0,gpu_submissions=0,ppo_updates=0,new_motion_generated=False,
        source_data_modified=False,accepted_reference=False,training_ready=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instinct-root',required=True);parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--run',action='store_true')
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise ValueError('Refuse to overwrite an existing execution report')
    budget=dependency('bounded_runtime').Budget(window_seconds=30,total_seconds=40,iterations=8,numeric_threads=2,max_windows=1)
    if not args.run:
        print(json.dumps(dict(plan_only=True,solver_calls=0,gpu_submissions=0,budget=budget.__dict__)));return
    report=dependency('bounded_runtime').supervise('review_pose_tasks','worker',
             [dict(instinct_root=str(Path(args.instinct_root).resolve()),dataset=str(Path(args.dataset).resolve()))],budget)
    dependency('bounded_runtime').save_diagnostic(out,report)
    print(json.dumps(report,indent=2),flush=True)
    if report['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()