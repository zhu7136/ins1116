"""Read-only source/evidence audit. No GPU, motion optimizer, or policy change.

All outputs are new files; prior candidate and evidence bytes are never changed.
Run from any working directory. Uses repository-relative, not lost private paths.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
REPO=HERE.parents[3]
MODEL=REPO/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
PRIOR=EXP/'pair_feasibility_obstruction_20261118'
SNAPS=EXP/'import_observation_20261117/recovered_failure_evidence_20261117'
MEASURED=EXP/'physx_detached_wrist_v2_20261118/measured_task138'
TRIAL=EXP/'material_trajectory_v1/TRIAL_MATERIAL_003_FULL_SEQUENCE'
MODEL_SHA='865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7'
SNAP_SHA={
    'post_import.json':'41c5826335c5ff3ec0decb382bac0de399d97cf96a90dad570ff1547a2f96082',
}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def one(items):
    items=list(items)
    if len(items)!=1:raise ValueError('Exact unique binding required')
    return items[0]
def transform(node):
    t=np.eye(4)
    if node is not None:
        t[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return t
def stl_vertices(path):
    data=path.read_bytes()
    n=int.from_bytes(data[80:84],'little')
    if len(data)!=84+50*n:raise ValueError('Reviewed binary STL required')
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
    return np.unique(np.frombuffer(data,offset=84,count=n,dtype=dtype)['vertices'].reshape(-1,3).astype(float),axis=0)
def geometry(element):
    g=one(element.find('geometry'))
    out=dict(type=g.tag,attributes=dict(g.attrib),link_T_geometry=transform(element.find('origin')).tolist())
    if g.tag=='mesh':
        p=(MODEL.parent/g.get('filename')).resolve()
        out.update(mesh_repository_path=str(p.relative_to(REPO)),mesh_sha256=digest(p))
    return out
def joint_metadata(joint,root,parent,child):
    if joint['body0']!=[root+'/'+parent] or joint['body1']!=[root+'/'+child]:raise ValueError('Exact body binding mismatch')
    for key in ['collision_enabled','joint_enabled','exclude_from_articulation']:
        if not joint[key]['present']:raise ValueError('Missing joint attribute')
    if joint['joint_enabled']['value'] is not True or joint['exclude_from_articulation']['value'] is not False:
        raise ValueError('Inactive or excluded joint')
    return dict(joint_path=joint['path'],body0=joint['body0'],body1=joint['body1'],
                collision_enabled=joint['collision_enabled'],metadata_only=True,actual_filter_cause_proven=False)
def finite_differences(x,dt=1/90):
    x=np.asarray(x,float)
    return np.diff(x,axis=0)/dt,np.diff(x,n=2,axis=0)/dt**2
def main():
    if any((HERE/p).exists() for p in ['AUDIT.json','BOUNDARY_DIAGNOSTICS.json']):raise FileExistsError('Do not overwrite audit')
    inputs={}
    def bind(p,expected=None):
        h=digest(p)
        if expected is not None and h!=expected:raise ValueError('Input identity mismatch: '+str(p))
        inputs[str(p.relative_to(REPO))]=h
    bind(MODEL,MODEL_SHA)
    for name in ['INVARIANT_WRIST_OBSTRUCTION.json','BOUND_PAIR_POLICY_EVIDENCE.json']:bind(PRIOR/name)
    proof=read(PRIOR/'INVARIANT_WRIST_OBSTRUCTION.json');bound=read(PRIOR/'BOUND_PAIR_POLICY_EVIDENCE.json')
    snapshots={}
    for name,record in bound['snapshots'].items():
        bind(SNAPS/name,record['file_sha256']);snapshots[name]=read(SNAPS/name)
    bind(MEASURED/'INDEPENDENT_MEASURED_REVIEW.json')
    measured=read(MEASURED/'INDEPENDENT_MEASURED_REVIEW.json')
    for name,h in measured['report_sha256'].items():bind(MEASURED/name,h)
    bind(TRIAL/'INDEPENDENT_VERIFICATION.json');verified=read(TRIAL/'INDEPENDENT_VERIFICATION.json')
    for name,h in verified['files_sha256'].items():bind(TRIAL/name,h)
    xml=ET.parse(MODEL).getroot();links={l.get('name'):l for l in xml.findall('link')}
    with np.load(TRIAL/'FULL997_ENGINEERING_CANDIDATE_UNACCEPTED.npz',allow_pickle=False) as z:
        q=z['joint_pos'].copy();r=z['root_transforms_w'].copy();names=z['joint_names'].tolist()
        if q.shape!=(997,29) or float(z['framerate'])!=90 or bool(z['accepted_reference']) or bool(z['training_ready']):raise ValueError('Frozen candidate contract')
    pairs=[]
    for old in proof['pairs']:
        joint=one(j for j in xml.findall('joint') if j.get('name')==old['joint'])
        parent=joint.find('parent').get('link');child=joint.find('child').get('link');origin=transform(joint.find('origin'))
        if old['first']!=parent+'#1' or old['second']!=child+'#0':raise ValueError('Index identity mismatch')
        cylinder=links[parent].findall('collision')[1];mesh=links[child].findall('collision')[0]
        cg,mg=geometry(cylinder),geometry(mesh)
        if cg['type']!='cylinder' or mg['type']!='mesh' or not np.allclose(mg['link_T_geometry'],np.eye(4),atol=0,rtol=0):raise ValueError('Nominal shapes changed')
        bind(MODEL.parent/mg['attributes']['filename'],old['mesh_sha256'])
        point=np.array(old['point_child']);axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
        if not np.array_equal(axis,[1,0,0]) or not np.array_equal(origin[:3,:3],np.eye(3)) or not np.array_equal(point[1:],[0,0]):raise ValueError('Analytic invariant assumptions')
        cylinder_T=np.array(cg['link_T_geometry']);parent_point=(origin@np.r_[point,1])[:3]
        if not np.array_equal(cylinder_T[:3,3],origin[:3,3]):raise ValueError('Axis centre changed')
        local=(np.linalg.inv(cylinder_T)@np.r_[parent_point,1])[:3]
        vertices=stl_vertices((MODEL.parent/mg['attributes']['filename']).resolve())
        planes=ConvexHull(vertices).equations
        margins=dict(cylinder_radial=float(cg['attributes']['radius'])-np.linalg.norm(local[:2]),
                     cylinder_cap=float(cg['attributes']['length'])/2-abs(local[2]),hull_halfspace=float(-(planes[:,:3]@point+planes[:,3]).max()))
        if min(margins.values())<=0:raise ValueError('No strict common interior point')
        observations=[]
        for phase,snap in snapshots.items():
            root=snap['robot_root'];joint_record=one(j for j in snap['joints'] if j['path']==root+'/joints/'+old['joint'])
            record=joint_metadata(joint_record,root,parent,child);record['phase']=phase
            record['colliders']=[]
            for link,collision,g in [(parent,cylinder,cg),(child,mesh,mg)]:
                prefix=root+'/'+link+'/collisions/'+collision.get('name')+'/'
                collider=one(c for c in snap['colliders'] if c['path'].startswith(prefix) and c['rigid_body_path']==root+'/'+link)
                record['colliders'].append(collider)
            a,b=record['colliders']
            world_a=np.array(a['world_transform_gf_row_vector']).T
            world_b=np.array(b['world_transform_gf_row_vector']).T
            record['cylinder_T_mesh_metadata']= (np.linalg.inv(world_a)@world_b).tolist()
            record['shape_ids_or_cooked_vertices_bound']=False
            observations.append(record)
        side=parent.split('_')[0]
        pairs.append(dict(first=old['first'],second=old['second'],joint=old['joint'],
            collision=[dict(name=cylinder.get('name'),**cg),dict(name=mesh.get('name'),**mg)],
            visuals={link:[geometry(v) for v in links[link].findall('visual')] for link in [parent,child]},
            parent_T_child_at_zero=origin.tolist(),joint_axis_child=axis.tolist(),
            relative_transform_function='parent_T_child(q) = parent_T_child_at_zero @ homogeneous(Rx(q), 0); NOT a fixed full transform',
            candidate_joint_angle_range_rad=[float(q[:,names.index(old['joint'])].min()),float(q[:,names.index(old['joint'])].max())],
            analytic_common_point_child=point.tolist(),independently_recomputed_interior_margins_raw=margins,
            all_angle_nominal_intersection=True,interior_margins_are_not_penetration_depth=True,
            runtime_observations=observations,historical_measured_body_pair_channels={k:measured['channels'][k] for k in ['articulated_'+side,'detached_'+side]},
            historical_gpu_task='TASK_20261118_138',measured_shape_pair_cause_proven=False,
            actual_runtime_pair_enabled='UNRESOLVED; USD intent disabled; historical articulated body-pair contact absent',
            nominal_penetration_depth_raw=None,physx_signed_separation_or_penetration_depth=None,
            depth_status='NOT_MEASURED; neither contact force, count, nor interior slack is penetration depth',
            classification='METADATA_DISABLED_AND_CONTEXT_DIFFERENCE_OBSERVED; three-way causal classification still unresolved',
            effective_pair_policy='UNCHANGED_NO_EXCLUSIONS'))
    for pair in pairs:
        for rows in pair['visuals'].values():
            for row in rows:
                if row['type']=='mesh':bind(REPO/row['mesh_repository_path'])
    # Joint-space/root-translation derivatives across all16 overlapping-window joins.
    # A single stored global state means no duplicate knot positions; it does NOT
    # imply physical C2 interpolation or acceleration acceptance.
    qv,qa=finite_differences(q);pv,pa=finite_differences(r[:,:3,3])
    starts=np.rint(np.linspace(0,936,17)).astype(int)
    seams=[]
    for f in starts[1:]:
        seams.append(dict(frame=int(f),joint_velocity_backward_forward_difference_max_rad_s=float(abs(qv[f]-qv[f-1]).max()),
            root_translation_velocity_backward_forward_difference_norm_m_s=float(np.linalg.norm(pv[f]-pv[f-1])),
            joint_acceleration_left_right_difference_max_rad_s2=float(abs(qa[f]-qa[f-2]).max()),
            root_translation_acceleration_left_right_difference_norm_m_s2=float(np.linalg.norm(pa[f]-pa[f-2])),
            derivative_differences_are_not_C2_discontinuity_proofs=True))
    audit=dict(stage='CPU_EXISTING_EVIDENCE_AUDIT_NOT_NEW_PHYSICS',baseline_commit='fcf38047fce7eba1ea893fc756334f3bdc0a097a',
        model_sha256=MODEL_SHA,pairs=pairs,frozen_candidate_sha256=inputs[str((TRIAL/'FULL997_ENGINEERING_CANDIDATE_UNACCEPTED.npz').relative_to(REPO))],
        prior_task366_failed_exit_minus11_retained=True,task138_cooked_equivalence_and_filter_causality_unresolved=True,
        input_files_sha256=inputs,source_motion_modified=False,automatic_pair_exclusions=0,
        optimizer_calls=0,gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False)
    boundary=dict(frames=997,fps=90,body_dof=29,head_action_dof=0,window_joins=seams,
        global_joint_acceleration_max_rad_s2=float(abs(qa).max()),global_root_translation_acceleration_max_m_s2=float(np.linalg.norm(pa,axis=1).max()),
        interpolation_C2_verified=False,root_angular_dynamics_verified=False,physical_acceleration_limits_bound=False,
        support_unassigned_frames=346,accepted_reference=False,training_ready=False)
    for name,value in [('AUDIT.json',audit),('BOUNDARY_DIAGNOSTICS.json',boundary)]:
        with (HERE/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
        if read(HERE/name)!=value:raise ValueError('Output readback mismatch')
    print(json.dumps(dict(pairs=len(pairs),input_files=len(inputs),frozen_candidate_sha256=audit['frozen_candidate_sha256'],
                         joins_audited=len(seams),automatic_pair_exclusions=0,optimizer_calls=0,gpu_submissions=0,training_ready=False)),flush=True)

if __name__=='__main__':main()