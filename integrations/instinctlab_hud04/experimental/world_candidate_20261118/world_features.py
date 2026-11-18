"""World-frame 35D features under an EXPLICIT, UNACCEPTED correspondence.

Positions at wrists/knees are LINK pivots, NOT palm/knee contacts. Sole centres
are nominal collision-bounds centres, NOT accepted support material frames.
Zero-coordinate orientation offsets are a kinematic hypothesis, not calibrated
source-target palms. Never selects a root/quaternion interpretation by distance.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path[:0]=['/workspace/hud04-direction-solve-20261118','/workspace/hud04-runtime-preflight-20261117']
from anatomical_directions import DirectionModel,MOTION_SHA,SOURCE_SHA,TARGET_SHA
import task_kinematics as TK

ROOT=Path(__file__).parent
BASE_COMMIT='70393c676c85888db0bdd92f8ab7e64fe83fa88d'
SEED=Path('/workspace/hud04-full-direction-20261118/OVERLAP_V4/STEP1_FULL_LENGTH_JOINT_DIAGNOSTIC.npz')
SEED_SHA='2aa620d3086ed5f1c260c210e38fcb3d3a94cbaea549911844b8e4c31127ddc0'
GEOMETRY=Path('/workspace/hud04-multipatch-review-20261117/CONTACT_CORRESPONDENCE_GEOMETRY_REVIEW_20261117.json')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check_roots(roots,n):
    roots=np.asarray(roots,float)
    if roots.shape!=(n,4,4) or not np.isfinite(roots).all():raise ValueError('Finite full root transforms')
    if not np.allclose(roots[:,3],np.array([0,0,0,1]),rtol=0,atol=1e-12):raise ValueError('Homogeneous roots')
    if not np.allclose(np.swapaxes(roots[:,:3,:3],1,2)@roots[:,:3,:3],np.eye(3),rtol=0,atol=1e-10) or not np.allclose(np.linalg.det(roots[:,:3,:3]),1,rtol=0,atol=1e-10):raise ValueError('SO3 roots')
    return roots

def perturb(roots,delta):
    roots=check_roots(roots,len(roots));delta=np.asarray(delta,float)
    if delta.shape!=(len(roots),6) or not np.isfinite(delta).all():raise ValueError('Six finite tangent increments per frame')
    out=roots.copy();out[:,:3,3]+=delta[:,:3]
    out[:,:3,:3]=Rotation.from_rotvec(delta[:,3:]).as_matrix()@roots[:,:3,:3]
    return check_roots(out,len(roots))

class WorldFeatures:
    def __init__(self,kind):
        self.model=DirectionModel(TK.SOURCE if kind=='source' else TK.TARGET,kind)
        self.kind=kind;self.names=self.model.names
        torso='torso_link' if kind=='source' else 'waist_pitch_link'
        wrist='wrist_yaw_link' if kind=='source' else 'wrist_roll_link'
        self.positions=[('torso_link_pivot',torso,np.zeros(3))]
        rows=json.loads(GEOMETRY.read_text())['rows']
        for side in ['left','right']:
            foot=next(r for r in rows if r['side']==side)['foot']
            box=foot['source_nominal_collision_union'] if kind=='source' else foot['target_nominal_box']
            lo,hi=np.array(box['local_bounds']);bottom=(lo+hi)/2;bottom[2]=lo[2]
            self.positions.extend([(side+'_distal_wrist_pivot',side+'_'+wrist,np.zeros(3)),
                                   (side+'_knee_pivot',side+'_knee_link',np.zeros(3)),
                                   (side+'_nominal_sole_bounds_center',side+'_ankle_roll_link',bottom)])
        self.orientation_links=[torso]+[s+'_'+l for s in ['left','right'] for l in [wrist,'ankle_roll_link']]

    def evaluate(self,q,roots,jacobian=False):
        q=self.model.order(q,self.names);n=len(q);roots=check_roots(roots,n)
        local,axes=self.model.fk(q);world={l:roots@t for l,t in local.items()}
        p=np.stack([np.einsum('nij,j->ni',world[l][:,:3,:3],v)+world[l][:,:3,3] for _,l,v in self.positions],axis=1)
        r=np.stack([world[l][:,:3,:3] for l in self.orientation_links],axis=1)
        raw=np.stack([world[b][:,:3,3]-world[a][:,:3,3] for _,a,b in self.model.pairs],axis=1)
        length=np.linalg.norm(raw,axis=2);u=raw/length[:,:,None]
        value=np.concatenate([p.reshape(n,-1),r.reshape(n,-1),u.reshape(n,-1)],axis=1)
        if not jacobian:return value
        if self.kind!='target':raise ValueError('Target derivatives only')
        jp=np.zeros((n,7,3,35));jr=np.zeros((n,5,3,3,35));ju=np.zeros((n,8,3,35));jp[:,:,:,:3]=np.eye(3)
        for k in range(3):
            jp[:,:,:,3+k]=np.cross(np.eye(3)[k],p-roots[:,None,:3,3])
            ju[:,:,:,3+k]=np.cross(np.eye(3)[k],u)
            for col in range(3):jr[:,:,:,col,3+k]=np.cross(np.eye(3)[k],r[:,:,:,col])
        for c,name in enumerate(self.names):
            pivot,axis=axes[name]
            pivot=np.einsum('nij,nj->ni',roots[:,:3,:3],pivot)+roots[:,:3,3]
            axis=np.einsum('nij,nj->ni',roots[:,:3,:3],axis)
            for k,(_,link,_) in enumerate(self.positions):
                if name in self.model.ancestors[link]:jp[:,k,:,6+c]=np.cross(axis,p[:,k]-pivot)
            for k,link in enumerate(self.orientation_links):
                if name in self.model.ancestors[link]:
                    for col in range(3):jr[:,k,:,col,6+c]=np.cross(axis,r[:,k,:,col])
            for k,(_,a,b) in enumerate(self.model.pairs):
                dr=np.zeros((n,3))
                if name in self.model.ancestors[b]:dr+=np.cross(axis,world[b][:,:3,3]-pivot)
                if name in self.model.ancestors[a]:dr-=np.cross(axis,world[a][:,:3,3]-pivot)
                ju[:,k,:,6+c]=(dr-u[:,k]*np.einsum('ni,ni->n',u[:,k],dr)[:,None])/length[:,k,None]
        return value,np.concatenate([jp.reshape(n,-1,35),jr.reshape(n,-1,35),ju.reshape(n,-1,35)],axis=1)

def problem_data():
    for p,sha in [(TK.SOURCE,SOURCE_SHA),(TK.TARGET,TARGET_SHA),(TK.MOTION,MOTION_SHA),(SEED,SEED_SHA)]:
        if digest(p)!=sha:raise ValueError('Pinned input changed: '+str(p))
    source=WorldFeatures('source');target=WorldFeatures('target')
    with np.load(TK.MOTION,allow_pickle=False) as z:
        if int(z['framerate'].item())!=90:raise ValueError('90Hz required')
        sq=source.model.order(z['joint_pos'],z['joint_names'].tolist())
        sw=TK.K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz')
    with np.load(SEED,allow_pickle=False) as z:
        if not np.array_equal(z['frame_indices'],np.arange(997)) or int(z['framerate'].item())!=90:raise ValueError('Full997 seed required')
        q=target.model.order(z['joint_pos'],z['joint_names'].tolist())
    if sq.shape!=q.shape or len(q)!=997:raise ValueError('Native997 frames')
    desired=source.evaluate(sq,sw)
    # Fixed per-end-link offsets derived in the two models' ZERO-coordinate
    # pelvis/base anatomical basis. NOT learned from the candidate or terrain.
    sz,_=source.model.fk(np.zeros((1,29)));tz,_=target.model.fk(np.zeros((1,29)))
    sa=sz[source.model.anchor][0,:3,:3];ta=tz[target.model.anchor][0,:3,:3]
    offsets=[]
    for sl,tl in zip(source.orientation_links,target.orientation_links):
        rs=sa.T@sz[sl][0,:3,:3];rt=ta.T@tz[tl][0,:3,:3]
        offsets.append(rs.T@rt)
    offsets=np.array(offsets)
    desired[:,21:66]=(desired[:,21:66].reshape(997,5,3,3)@offsets).reshape(997,45)
    # Initialize HUD04 root from the configured SOURCE TORSO, compensating
    # the CURRENT TARGET waist transform, not by copying source root/pelvis.
    local,_=target.model.fk(q);torso=local['waist_pitch_link']
    roots=np.tile(np.eye(4),(997,1,1))
    roots[:,:3,:3]=desired[:,21:30].reshape(997,3,3)@np.swapaxes(torso[:,:3,:3],1,2)
    roots[:,:3,3]=desired[:,:3]-np.einsum('nij,nj->ni',roots[:,:3,:3],torso[:,:3,3])
    check_roots(roots,997)
    correspondence=dict(role='EXPLICIT_UNACCEPTED_model_zero_coordinate_kinematic_correspondence',
        source_contract='CONDITIONED_on_official_consumer_torso_link_wxyz_unchanged_raw_units',
        source_target_orientation_offsets=offsets.tolist(),source_orientation_links=source.orientation_links,target_orientation_links=target.orientation_links,
        source_positions=[dict(label=a,link=b,local_point=c.tolist()) for a,b,c in source.positions],
        target_positions=[dict(label=a,link=b,local_point=c.tolist()) for a,b,c in target.positions],
        position_correspondences_are_soft=True,source_root_copied=False,zero_offset_is_not_anatomical_palm_calibration=True,
        source_units_exporter_proven=False,contact_schedule_accepted=False,source_target_material_calibration_accepted=False,
        reference_acceptance_authorized_by_this_mapping=False)
    return target,q,roots,desired,correspondence

def weights():
    # Fixed BEFORE execution. Raw position units assumed under the conditional
    # source contract. These are optimization weights, NOT acceptance limits.
    return np.r_[np.repeat([3.,2.,2.,20.,2.,2.,20.],3),np.repeat([1.,.35,1.,.35,1.],9),np.full(24,.5)]

def metrics(model,q,roots,desired):
    v=model.evaluate(q,roots);p=np.linalg.norm(v[:,:21].reshape(-1,7,3)-desired[:,:21].reshape(-1,7,3),axis=2)
    a=v[:,21:66].reshape(-1,5,3,3);b=desired[:,21:66].reshape(-1,5,3,3)
    rel=np.swapaxes(b,2,3)@a
    angles=np.rad2deg(Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude()).reshape(-1,5)
    out=dict(position_mean_raw=p.mean(axis=0).tolist(),position_max_raw=p.max(axis=0).tolist(),
             orientation_mean_degrees=angles.mean(axis=0).tolist(),orientation_max_degrees=angles.max(axis=0).tolist(),
             weighted_l1=float(np.abs((v-desired)*weights()).sum()),
             root_max_linear_speed_raw_per_s=float(np.linalg.norm(np.diff(roots[:,:3,3],axis=0)*90,axis=1).max()),
             root_max_linear_acceleration_raw_per_s2=float(np.linalg.norm(np.diff(roots[:,:3,3],n=2,axis=0)*8100,axis=1).max()))
    return out