"""Explicit 35-DoF world-pose task linearization; no solver or motion exporter.

World root increments match window_problem: translation plus world-axis
rotation about CURRENT root origin, then 29 named joint increments. A feature
frame is a caller-supplied link-local SE(3); it is never inferred from contact.
Unknown targets stay None. A declared neutral orientation mapping is a modeling
proposal, NOT material-contact calibration or reference acceptance.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse


def rotation(value):
    a=np.asarray(value,float)
    if a.shape[-2:]!=(3,3) or not np.isfinite(a).all():
        raise ValueError('Finite rotation matrices required')
    if not np.allclose(np.swapaxes(a,-1,-2)@a,np.eye(3),atol=1e-10,rtol=0) or not np.allclose(np.linalg.det(a),1,atol=1e-10,rtol=0):
        raise ValueError('Proper rotations required; no silent projection')
    return a


def rigid(value):
    a=np.asarray(value,float)
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1],atol=1e-12,rtol=0):
        raise ValueError('Explicit finite SE3 required')
    rotation(a[:3,:3]);return a


@dataclass(frozen=True)
class Feature:
    name: str
    link: str
    link_from_feature: object
    role: str

    def __post_init__(self):
        if not self.name or not self.link or self.role not in ('URDF_link_diagnostic','nominal_box_bottom_NOT_accepted_support'):
            raise ValueError('Explicit feature name/link/diagnostic role required')
        a=rigid(self.link_from_feature).copy();a.flags.writeable=False
        object.__setattr__(self,'link_from_feature',a)


def evaluate(model,q,world_roots,features,*,jacobian=True):
    """World positions, rotation bases, optional exact [N,K,3,(3),35] Jacobians."""
    if model.kind!='target' or model.root!='base_link' or model.anchor!=model.root:
        raise ValueError('Pinned fixed-head target model required')
    q=model.order(q,model.names);n=len(q)
    roots=np.asarray(world_roots,float)
    if roots.shape!=(n,4,4):raise ValueError('An explicit root for every frame required')
    for root in roots:rigid(root)
    features=tuple(features)
    if not features or not all(isinstance(f,Feature) for f in features) or len({f.name for f in features})!=len(features):
        raise ValueError('Unique nonempty explicit features required')
    poses,axes=model.fk(q)
    if not all(f.link in poses for f in features):raise ValueError('Feature link missing')
    local=np.stack([poses[f.link]@f.link_from_feature for f in features],axis=1)
    world=roots[:,None]@local
    points=world[:,:,:3,3].copy();bases=world[:,:,:3,:3].copy()
    if not jacobian:return points,bases
    jp=np.zeros((n,len(features),3,35));jr=np.zeros((n,len(features),3,3,35))
    jp[:,:,:,:3]=np.eye(3)
    for c in range(3):
        axis=np.eye(3)[c]
        jp[:,:,:,3+c]=np.cross(axis,points-roots[:,None,:3,3])
        for col in range(3):jr[:,:,:,col,3+c]=np.cross(axis,bases[:,:,:,col])
    for c,name in enumerate(model.names):
        pivot,axis=axes[name]
        pivot=np.einsum('nij,nj->ni',roots[:,:3,:3],pivot)+roots[:,:3,3]
        axis=np.einsum('nij,nj->ni',roots[:,:3,:3],axis)
        for k,f in enumerate(features):
            if name not in model.ancestors[f.link]:continue
            jp[:,k,:,6+c]=np.cross(axis,points[:,k]-pivot)
            for col in range(3):jr[:,k,:,col,6+c]=np.cross(axis,bases[:,k,:,col])
    return points,bases,jp,jr


def neutral_orientation_proposal(source_bases,source_neutral,target_neutral):
    """R_desired = R_source * inv(R_source_zero) * R_target_zero.

    Inputs must ALREADY use explicitly corresponding anchor bases. Does not
    invent that correspondence, establish world units, or calibrate a palm.
    """
    source_bases=rotation(source_bases);source_neutral=rotation(source_neutral);target_neutral=rotation(target_neutral)
    if source_bases.ndim!=4 or source_neutral.shape!=source_bases.shape[1:] or target_neutral.shape!=source_neutral.shape:
        raise ValueError('Explicit [N,K,3,3] and matching neutral bases required')
    offsets=np.swapaxes(source_neutral,-1,-2)@target_neutral
    return source_bases@offsets,offsets


def stack_channels(model,q,roots,features,*,desired_points=None,desired_bases=None,desired_directions=None):
    """Separate unweighted channels: no mixing coordinate units and radians.

    Basis residual is nine chordal entries per frame/feature, not a 3-vector
    rotation log. It cannot distinguish a physically appropriate palm without
    reviewed frame definitions. No weights or feasibility verdicts are chosen.
    """
    points,bases,jp,jr=evaluate(model,q,roots,features)
    if len(q)>93:raise ValueError('At most 93 frames per window construction')
    pr=None;br=None;dr=None
    directions,jd=model.evaluate(q,True)
    # Bone directions are target-base-local: root motion has zero derivative.
    jd=np.pad(jd.reshape(len(q),-1,29),((0,0),(0,0),(6,0)))
    if desired_points is not None:
        dp=np.asarray(desired_points,float)
        if dp.shape!=points.shape or not np.isfinite(dp).all():raise ValueError('Explicit finite point targets required')
        pr=(points-dp).reshape(-1)
    if desired_bases is not None:
        db=rotation(desired_bases)
        if db.shape!=bases.shape:raise ValueError('Explicit rotation target shape required')
        br=(bases-db).reshape(-1)
    if desired_directions is not None:
        dd=np.asarray(desired_directions,float)
        if dd.shape!=directions.shape or not np.isfinite(dd).all() or not np.allclose(np.linalg.norm(dd,axis=-1),1,atol=1e-10,rtol=0):
            raise ValueError('Explicit named-pair unit directions required; no silent normalization')
        dr=(directions-dd).reshape(-1)
    return {'point_jacobian':sparse.block_diag([sparse.csr_matrix(j.reshape(-1,35)) for j in jp],format='csr'),
            'point_residual':pr,
            'basis_jacobian':sparse.block_diag([sparse.csr_matrix(j.reshape(-1,35)) for j in jr],format='csr'),
            'basis_residual':br,
            'direction_jacobian':sparse.block_diag([sparse.csr_matrix(j) for j in jd],format='csr'),
            'direction_residual':dr,
            'direction_pair_names':[p[0] for p in model.pairs],
            'feature_names':[f.name for f in features],
            'feature_roles':[f.role for f in features],
            'variables_per_frame':35,'head_variables':0,
            'material_calibration_accepted':False,'support_intervals':None,
            'terrain_constraints_implemented':False,'self_collision_constraints_implemented':False,
            'solver_calls':0,'accepted_reference':False,'training_ready':False}


def build_joint_pose_window(window_builder,point_model,pose_model,q,stored_names,roots,frame_ids,features,
                            *,desired_points=None,desired_bases=None,desired_directions=None):
    """Build both channels and old hard rows from the SAME named inputs.

    No API for attaching an unrelated same-shaped trajectory is provided.
    window_builder is the pinned window_problem.build dependency.
    """
    if point_model.names!=pose_model.names or not all(np.array_equal(getattr(point_model,k),getattr(pose_model,k)) for k in ('lower','upper','vmax')):
        raise ValueError('Pinned models must share exact named order and bounds')
    q=pose_model.order(q,stored_names)
    if len(frame_ids)!=len(q):raise ValueError('One native frame ID per configuration')
    system=window_builder(point_model,q,pose_model.names,roots,frame_ids,fps=90)
    channels=stack_channels(pose_model,q,roots,features,desired_points=desired_points,desired_bases=desired_bases,
                            desired_directions=desired_directions)
    return {'existing_window':system,'pose_channels':channels,
            'hard_constraints_reused_without_modification':True,
            'optimizer_implemented':False,'accepted_reference':False,'training_ready':False}