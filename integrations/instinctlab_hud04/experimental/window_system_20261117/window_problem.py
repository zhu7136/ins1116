"""35-DoF/window linearization for a future constrained retargeting solve.

No optimizer or motion writer. Unknown anatomical/source targets remain None,
not fake zero residuals. Constraint rows operate on joint increments relative
to an explicitly named seed; root updates use world-axis rotation at root.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse

@dataclass
class WindowSystem:
    tracking_jacobian: object
    tracking_residual: object
    hard_matrix: object
    hard_lower: object
    hard_upper: object
    step_lower: object
    step_upper: object
    frames: object
    edges: object
    joint_names: object
    target_objective_supplied: bool

def frame_edges(frame_ids):
    f=np.asarray(frame_ids)
    if f.ndim!=1 or not len(f) or f.dtype.kind not in 'iu' or np.any(f<0) or np.any(f>=997) or np.any(f[1:]<=f[:-1]):raise ValueError('Unique increasing native source frame IDs')
    return f,[(i,i+1) for i in range(len(f)-1) if f[i+1]==f[i]+1]

def named_order(stored,expected):
    if len(stored)!=29 or len(set(stored))!=29 or set(stored)!=set(expected):raise ValueError('Exact named 29 body joints')
    return np.array([stored.index(n) for n in expected])

def direction_residual(points,jacobian,start,end,target_direction):
    """Optional dimensionless soft bone-direction term, NOT a contact target.

    Callers must supply an explicit reviewed semantic pair/target. This kernel
    does not infer source anatomy, root alignment or world units.
    """
    p=np.asarray(points,float);j=np.asarray(jacobian,float);d=np.asarray(target_direction,float)
    if type(start) is not int or type(end) is not int or p.ndim!=2 or p.shape[1]!=3 or j.shape!=(len(p),3,35) or not 0<=start<len(p) or not 0<=end<len(p) or start==end or d.shape!=(3,) or not all(np.isfinite(a).all() for a in (p,j,d)):raise ValueError('Finite explicit direction feature')
    if not np.isclose(np.linalg.norm(d),1,rtol=0,atol=1e-10):raise ValueError('Unit target direction required, no silent normalization')
    v=p[end]-p[start];length=np.linalg.norm(v)
    if length<1e-9:raise ValueError('Degenerate direction')
    u=v/length
    return u-d,((np.eye(3)-np.outer(u,u))/length)@(j[end]-j[start])

def build(target,seed,stored_names,roots,frame_ids,*,fps=90,desired_points=None,
          root_translation_step=.02,root_rotation_step=.03,joint_step=.03):
    frames,edges=frame_edges(frame_ids);n=len(frames)
    if type(fps) is not int or fps!=90:raise ValueError('Native 90Hz only')
    q=np.asarray(seed,float);roots=np.asarray(roots,float)
    if q.shape!=(n,29) or roots.shape!=(n,4,4) or not np.isfinite(q).all():raise ValueError('Explicit window seed/root shapes')
    q=q[:,named_order(list(stored_names),target.names)]
    if n>93:raise ValueError('At most three previous 31-frame diagnostic windows')
    for v,cap in ((root_translation_step,.02),(root_rotation_step,.03),(joint_step,.03)):
        if isinstance(v,bool) or not np.isfinite(v) or not 0<v<=cap:raise ValueError('Finite bounded differential trust region')
    jac=[];points=[]
    for row in range(n):
        p,j=target.evaluate(q[row],roots[row]);points.append(p);jac.append(sparse.csr_matrix(j.reshape(-1,35)))
    a=sparse.block_diag(jac,format='csr');points=np.array(points)
    if desired_points is None:r=None
    else:
        desired_points=np.asarray(desired_points,float)
        if desired_points.shape!=points.shape or not np.isfinite(desired_points).all():raise ValueError('Explicit finite target points')
        r=(points-desired_points).reshape(-1)
    # Joint extraction from frame-major [root translation,root rotation,29dq].
    extraction=sparse.coo_matrix((np.ones(29*n),(np.arange(29*n),np.array([i*35+6+j for i in range(n) for j in range(29)]))),shape=(29*n,35*n)).tocsr()
    erows=[];cols=[];data=[]
    for e,(i,k) in enumerate(edges):
        for j in range(29):
            erows.extend([29*e+j]*2);cols.extend([29*i+j,29*k+j]);data.extend([-1.,1.])
    d=sparse.coo_matrix((data,(erows,cols)),shape=(29*len(edges),29*n)).tocsr()
    matrix=sparse.vstack([extraction,d@extraction],format='csr')
    flat=q.reshape(-1);delta=d@flat
    lower=np.r_[np.tile(target.lower,n)-flat,-np.tile(target.vmax/90,len(edges))-delta]
    upper=np.r_[np.tile(target.upper,n)-flat,np.tile(target.vmax/90,len(edges))-delta]
    step=np.tile(np.r_[np.full(3,root_translation_step),np.full(3,root_rotation_step),np.full(29,joint_step)],n)
    if not all(np.isfinite(x).all() for x in (matrix.data,lower,upper,a.data,step)) or np.any(lower>upper):raise ValueError('Finite ordered constraints')
    return WindowSystem(a,r,matrix,lower,upper,-step,step,frames,edges,list(target.names),desired_points is not None)

def summarize(system):
    return dict(stage='root_and_joint_window_linear_system_not_motion',frames=len(system.frames),
        variables=system.tracking_jacobian.shape[1],variables_per_frame=35,root_variables_per_frame=6,
        body_variables_per_frame=29,head_variables=0,native_fps=90,
        tracking_jacobian_shape=list(system.tracking_jacobian.shape),hard_joint_matrix_shape=list(system.hard_matrix.shape),
        adjacent_intervals=len(system.edges),cross_gap_velocity_edges=0,
        target_objective_supplied=system.target_objective_supplied,
        tracking_residual_is_null=system.tracking_residual is None,
        zero_joint_increment_max_violation=float(max(0.,system.hard_lower.max(),-system.hard_upper.min())),
        trust_region_units='target_model_coordinate_units_translation_and_radians_rotation_joints_NOT_source_unit_proof',
        model_limits_modified=False,posthoc_angle_clipping=False,
        solver_implemented=False,solver_calls=0,candidate_motion_generated=False,
        terrain_constraints_implemented=False,self_collision_constraints_implemented=False,
        accepted_reference=False,training_ready=False)