"""Independent saved997candidate verification via legacy RobotTree + XML.

No solver import; NO accepted manifest. Nominal foot BOX / actual cropped
triangle surface intersections are a rejection diagnostic, NOT signed depth,
complete robot collision check, support labels or runtime cooked geometry.
"""
import hashlib,json,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
sys.path[:0]=['/workspace/hud04-cpu-deps-20261117','/workspace/hud04-runtime-preflight-20261117',
              '/workspace/hud04-solid-geometry-20261117','/workspace/hud04-terrain-binding-20261117']
import task_kinematics as TK
from convex_solid import Hull
from terrain_binding import ActualTerrain

ROOT=Path(__file__).parent
TRIAL=ROOT/'WORLD_TRIAL_002_BLOCK31'
OUTPUT=TRIAL/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
SHA='7c2c98d7811e3c847c7a20239612006647d56f42777a1ddea914c52f2711f253'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):return dict(mean=float(np.mean(x)),p95=float(np.percentile(x,95)),maximum=float(np.max(x)))

def main():
    start=time.monotonic()
    if sha(OUTPUT)!=SHA:raise ValueError('Saved candidate changed')
    report=json.loads((TRIAL/'WORLD_CANDIDATE_RESULT.json').read_text())
    for p,d in report['inputs_sha256'].items():
        if sha(p)!=d:raise ValueError('Pinned input changed')
    xml=ET.parse(TK.TARGET).getroot();joints=[j for j in xml.findall('joint') if j.get('type')!='fixed']
    expected=[j.get('name') for j in joints]
    with np.load(OUTPUT,allow_pickle=False) as z:
        names=z['joint_names'].tolist();q=z['joint_pos'].astype(float);p=z['base_pos_w'].astype(float);quat=z['base_quat_w'].astype(float)
        if names!=expected or q.shape!=(997,29) or p.shape!=(997,3) or quat.shape!=(997,4):raise ValueError('Full native shape/order')
        if z['framerate'].item()!=90 or not np.array_equal(z['frame_indices'],np.arange(997)) or z['accepted_reference'].item()!=False:raise ValueError('Contract/acceptance')
        if not all(np.isfinite(v).all() for v in [q,p,quat]):raise ValueError('Finite poses')
    lower,upper,vmax=[np.array([float(j.find('limit').get(k)) for j in joints]) for k in ['lower','upper','velocity']]
    badp=np.any((q<lower-1e-6)|(q>upper+1e-6),axis=1);badv=np.any(abs(np.diff(q,axis=0))*90>vmax+1e-6,axis=1)
    for n in ['head_yaw_joint','head_pitch_joint']:
        if next(j for j in xml.findall('joint') if j.get('name')==n).get('type')!='fixed':raise ValueError('Fixed head')
    qwerr=float(abs(np.linalg.norm(quat,axis=1)-1).max())
    if qwerr>1e-10:raise ValueError('Unit quaternions')
    roots=TK.K.world_pose(p,quat,'wxyz');tree=TK.K.RobotTree.read(TK.TARGET)
    world={l:roots@v for l,v in tree.fk(q,names).items()}
    proposal=json.loads((TRIAL/'CORRESPONDENCE_PROPOSAL_NOT_ACCEPTED.json').read_text())
    with np.load(TK.MOTION,allow_pickle=False) as z:
        source_tree=TK.K.RobotTree.read(TK.SOURCE)
        sw=TK.K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz')
        source={l:sw@v for l,v in source_tree.fk(z['joint_pos'],z['joint_names'].tolist()).items()}
    target_points=np.stack([np.einsum('nij,j->ni',world[r['link']][:,:3,:3],r['local_point'])+world[r['link']][:,:3,3] for r in proposal['target_positions']],axis=1)
    source_points=np.stack([np.einsum('nij,j->ni',source[r['link']][:,:3,:3],r['local_point'])+source[r['link']][:,:3,3] for r in proposal['source_positions']],axis=1)
    per_pos=np.linalg.norm(target_points-source_points,axis=2)
    a=np.stack([world[l][:,:3,:3] for l in proposal['target_orientation_links']],axis=1)
    b=np.stack([source[l][:,:3,:3] for l in proposal['source_orientation_links']],axis=1)@np.array(proposal['source_target_orientation_offsets'])
    angles=np.rad2deg(Rotation.from_matrix((np.swapaxes(b,2,3)@a).reshape(-1,3,3)).magnitude()).reshape(997,5)
    metric_disagreement=max(float(np.max(abs(per_pos.mean(axis=0)-report['final']['position_mean_raw']))),
                            float(np.max(abs(angles.max(axis=0)-report['final']['orientation_max_degrees']))))
    if metric_disagreement>1e-9:raise ValueError('Independent report disagreement')
    terrain=ActualTerrain();tri=terrain.triangles;lo=tri.min(axis=1);hi=tri.max(axis=1)
    rows=[];hits={s:[] for s in ['left','right']};triangle_queries=0
    corners=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    for side in ['left','right']:
        link=side+'_ankle_roll_link';node=next(l for l in xml.findall('link') if l.get('name')==link)
        box=next(c for c in node.findall('collision') if c.find('geometry/box') is not None)
        size=np.fromstring(box.find('geometry/box').get('size'),sep=' ');local=TK.K.origin_transform(box.find('origin'))
        body=Hull.from_points(corners*size/2)
        for frame in range(997):
            transform=world[link][frame]@local
            vertices=(corners*size/2)@transform[:3,:3].T+transform[:3,3]
            ids=np.flatnonzero(np.all(hi>=vertices.min(axis=0)-1e-12,axis=1)&np.all(lo<=vertices.max(axis=0)+1e-12,axis=1))
            hit=None
            for tid in ids:
                triangle_queries+=1;triangle=(tri[tid]-transform[:3,3])@transform[:3,:3]
                polygon=body.clip_triangle(triangle)
                if len(polygon):
                    material=polygon.mean(axis=0);margin=float(np.min(size/2-abs(material)))
                    if margin>1e-7:
                        point=transform[:3,:3]@material+transform[:3,3]
                        # Independent barycentric least squares confirms the
                        # certificate is ON the original actual terrain face.
                        t=tri[tid];uv,res,rank,_=np.linalg.lstsq(np.column_stack([t[1]-t[0],t[2]-t[0]]),point-t[0],rcond=None)
                        err=float(np.linalg.norm(t[0]+uv[0]*(t[1]-t[0])+uv[1]*(t[2]-t[0])-point))
                        if rank!=2 or err>1e-9 or min(uv[0],uv[1],1-uv.sum()) < -1e-9:raise ValueError('Intersection witness not on triangle')
                        hit=dict(side=side,frame=frame,triangle_id=int(tid),material_point_in_box=material.tolist(),point_world=point.tolist(),
                                 inside_all_six_box_planes_margin_raw=margin,triangle_barycentric=[float(1-uv.sum()),float(uv[0]),float(uv[1])],
                                 triangle_projection_error=err,triangle_normal=terrain.mesh.face_normals[tid].tolist())
                        break
            if hit:hits[side].append(frame);rows.append(hit)
    # Save EVERY rejection witness, not just cherry-picked examples. Retain
    # possible surface containment/other shapes as UNKNOWN, never as passed.
    with (TRIAL/'FOOT_TERRAIN_INTERIOR_WITNESSES.json').open('x') as f:json.dump(rows,f,indent=2,allow_nan=False)
    review=dict(stage='independent_full997_saved_world_candidate_check_NOT_acceptance',artifact_sha256=SHA,
        method='legacy RobotTree FK/direct named targetXML/actual crop triangle clipped against nominal footbox + independent barycentric witness',
        frames=997,intervals=996,fps=90,body_joints=29,head_fixed_zero=True,position_bad_frames=int(badp.sum()),overspeed_intervals=int(badv.sum()),
        position_max_violation_rad=float(max(0,np.max(lower-q),np.max(q-upper))),velocity_max_violation_rad_s=float(max(0,np.max(abs(np.diff(q,axis=0))*90-vmax))),
        quaternion_norm_max_error=qwerr,adjacent_quaternion_dot_min=float(np.sum(quat[1:]*quat[:-1],axis=1).min()),
        independent_metric_max_disagreement=metric_disagreement,
        position_metrics=[dict(label=r['label'],**stats(per_pos[:,i])) for i,r in enumerate(proposal['target_positions'])],
        orientation_metrics=[dict(link=l,**stats(angles[:,i])) for i,l in enumerate(proposal['target_orientation_links'])],
        root_linear_speed_raw_per_s=stats(np.linalg.norm(np.diff(p,axis=0)*90,axis=1)),
        root_linear_acceleration_raw_per_s2=stats(np.linalg.norm(np.diff(p,n=2,axis=0)*8100,axis=1)),
        root_angular_speed_rad_per_s=stats(Rotation.from_matrix(np.swapaxes(roots[:-1,:3,:3],1,2)@roots[1:,:3,:3]).magnitude()*90),
        foot_box_surface_strict_interior_hit_frames={s:len(v) for s,v in hits.items()},all_rejection_witnesses= len(rows),foot_triangle_narrow_queries=triangle_queries,
        nominal_footbox_geometry_rejection=bool(rows),full_collision_free_proven=False,signed_penetration_depth_computed=False,
        support_labels_proven=False,cooked_filters_proven=False,source_unit_exporter_proven=False,dynamics_tested=False,
        terrain_evidence=terrain.evidence(),accepted_reference=False,training_ready=False,elapsed_seconds=time.monotonic()-start)
    with (TRIAL/'INDEPENDENT_WORLD_CANDIDATE_REVIEW.json').open('x') as f:json.dump(review,f,indent=2,allow_nan=False)
    print(json.dumps({k:review[k] for k in ['frames','position_bad_frames','overspeed_intervals','quaternion_norm_max_error','independent_metric_max_disagreement','foot_box_surface_strict_interior_hit_frames','foot_triangle_narrow_queries','nominal_footbox_geometry_rejection','root_linear_speed_raw_per_s','root_linear_acceleration_raw_per_s2','accepted_reference','elapsed_seconds']},indent=2),flush=True)
if __name__=='__main__':main()