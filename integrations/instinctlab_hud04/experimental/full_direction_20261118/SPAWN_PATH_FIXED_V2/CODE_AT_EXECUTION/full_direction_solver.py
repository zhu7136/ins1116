"""All native frames share ONE sparse joint-direction optimization.

NOT a world/root/terrain/dynamics retargeter. No angle clipping or resampling.
Dependencies are the previously published, separately hash-bound direction FK.
"""
import time
import warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog,OptimizeWarning

def validate(model,q,desired):
    q=np.asarray(q,float);desired=np.asarray(desired,float)
    if q.ndim!=2 or q.shape[1]!=29 or not 2<=len(q)<=997 or not np.isfinite(q).all():raise ValueError('2..997 finite native body frames')
    if desired.shape!=(len(q),8,3) or not np.isfinite(desired).all() or not np.allclose(np.linalg.norm(desired,axis=2),1,atol=1e-10,rtol=0):raise ValueError('Explicit unit directions')
    if model.kind!='target' or model.wrist_endpoint!='first_wrist_pivot' or len(model.names)!=29 or any('head' in n for n in model.names):raise ValueError('Explicit first-wrist HUD04 target')
    lower,upper,vmax=(np.asarray(x,float) for x in (model.lower,model.upper,model.vmax))
    if any(x.shape!=(29,) or not np.isfinite(x).all() for x in (lower,upper,vmax)) or np.any(lower>=upper) or np.any(vmax<=0):raise ValueError('Original model limits')
    if max(0.,np.max(lower-q),np.max(q-upper),np.max(np.abs(np.diff(q,axis=0))*90-vmax))>1e-6:raise ValueError('Hard-feasible original seed required')
    return q.copy(),desired.copy()

def inequalities(model,n):
    if type(n) is not int or not 2<=n<=997:raise ValueError('Bounded native sequence length')
    eye=sparse.eye(n*29,format='csr')
    time_difference=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csr')
    difference=sparse.kron(time_difference,sparse.eye(29,format='csr'),format='csr')
    g=sparse.vstack([eye,-eye,difference,-difference],format='csr')
    h=np.r_[np.tile(model.upper,n),-np.tile(model.lower,n),np.tile(model.vmax/90,n-1),np.tile(model.vmax/90,n-1)]
    return g,h

def stats(a,b):
    degrees=np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=2),np.sum(a*b,axis=2)))
    return dict(mean_degrees=float(np.mean(degrees)),p95_degrees=float(np.percentile(degrees,95)),max_degrees=float(np.max(degrees)),
        sum_l1_chordal=float(np.abs(a-b).sum()),per_direction_mean_degrees=np.mean(degrees,axis=0).tolist(),
        over25deg_observations=int((degrees>25).sum()),over25deg_frames=int(np.any(degrees>25,axis=1).sum()))

def solve(model,seed,desired,gate,*,outer_limit=12,trust_radius=.15,lp_seconds=4.,progress=None):
    if type(outer_limit) is not int or not 1<=outer_limit<=12:raise ValueError('12 outer iterations maximum')
    if isinstance(trust_radius,bool) or not isinstance(trust_radius,(int,float)) or not np.isfinite(trust_radius) or not 0<trust_radius<=.15:raise ValueError('Finite joint trust bound')
    if isinstance(lp_seconds,bool) or not isinstance(lp_seconds,(int,float)) or not np.isfinite(lp_seconds) or not 0<lp_seconds<=4:raise ValueError('Finite LP time budget')
    baseline,desired=validate(model,seed,desired);q=baseline.copy();n=len(q);width=n*29;count=n*24
    g,h=inequalities(model,n);i=sparse.eye(width,format='csr');it=sparse.eye(count,format='csr')
    hard=sparse.hstack([g,sparse.csr_matrix((g.shape[0],count+width))],format='csr')
    reg=sparse.vstack([sparse.hstack([i,sparse.csr_matrix((width,count)),-i]),sparse.hstack([-i,sparse.csr_matrix((width,count)),-i])],format='csr')
    cost=np.r_[np.zeros(width),np.ones(count),np.full(width,.001)]
    bounds=[(-trust_radius,trust_radius)]*width+[(0,None)]*(count+width)
    history=[];calls=0;accepted=0;line_evaluations=0;start=time.monotonic();reason='outer_budget_reached'
    initial=stats(model.evaluate(q),desired)
    def objective(values):return float(np.abs(model.evaluate(values)-desired).sum()+.001*np.abs(values-baseline).sum())
    value=objective(q)
    for iteration in range(outer_limit):
        gate.step();u,j=model.evaluate(q,True)
        a=sparse.block_diag([sparse.csr_matrix(x.reshape(24,29)) for x in j],format='csr');res=(u-desired).reshape(-1)
        track=sparse.vstack([sparse.hstack([a,-it,sparse.csr_matrix((count,width))]),sparse.hstack([-a,-it,sparse.csr_matrix((count,width))])],format='csr')
        depart=(q-baseline).reshape(-1);matrix=sparse.vstack([hard,track,reg],format='csr')
        rhs=np.r_[h-g@q.reshape(-1),-res,res,-depart,depart]
        with warnings.catch_warnings():
            # SciPy forwards these documented native HiGHS options verbatim.
            warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
            result=linprog(cost,A_ub=matrix,b_ub=rhs,bounds=bounds,method='highs',
                options={'time_limit':float(lp_seconds),'maxiter':10000,'threads':2,'parallel':False})
        calls+=1
        row=dict(iteration=iteration,lp_status=int(result.status),lp_iterations=int(result.nit),accepted=False,wall_seconds=time.monotonic()-start)
        if not result.success:
            reason='LP_stopped_status_'+str(result.status);history.append(row)
            if progress:progress(row)
            break
        step=result.x[:width].reshape(n,29)
        for alpha in (1.,.5,.25,.125,.0625):
            candidate=q+alpha*step
            if np.max(g@candidate.reshape(-1)-h)>1e-6:raise ValueError('Hard feasibility lost; clipping forbidden')
            test=objective(candidate);line_evaluations+=1
            if test<value-1e-9:
                row.update(accepted=True,alpha=alpha,objective_before=value,objective_after=test,max_joint_step=float(np.abs(alpha*step).max()))
                q=candidate;value=test;accepted+=1;break
        history.append(row)
        if progress:progress(row)
        if not row['accepted']:reason='no_nonlinear_descent_in_fixed_line_search';break
    position=float(max(0.,np.max(model.lower-q),np.max(q-model.upper)))
    velocity=float(max(0.,np.max(np.abs(np.diff(q,axis=0))*90-model.vmax)))
    if position>1e-6 or velocity>1e-6:raise ValueError('Direct full-sequence hard constraint check failed')
    report=dict(schema_version=1,stage='full_length_root_local_joint_direction_diagnostic_NOT_reference',frames=n,native_fps=90,native_velocity_edges=n-1,
        single_global_problem=True,lp_method='highs',native_highs_threads=2,window_stitching=False,resampling=False,posthoc_joint_clipping=False,all_frames_are_optimization_variables=True,
        joint_unknowns=width,lp_unknowns=width*2+count,lp_inequality_rows=int(matrix.shape[0]) if calls else None,
        joint_trust_radius_rad=trust_radius,lp_time_limit_seconds=lp_seconds,outer_iteration_limit=outer_limit,
        history=history,actual_lp_calls=calls,accepted_steps=accepted,line_search_evaluations=line_evaluations,termination=reason,
        initial=initial,final=stats(model.evaluate(q),desired),elapsed_seconds=time.monotonic()-start,
        position_max_violation_rad=position,velocity_max_violation_rad_s=velocity,
        changed_frames=int(np.any(np.abs(q-baseline)>1e-9,axis=1).sum()),
        changed_joints=[name for c,name in enumerate(model.names) if np.any(np.abs(q[:,c]-baseline[:,c])>1e-9)],
        max_joint_departure_rad=float(np.abs(q-baseline).max()),
        world_root_optimized=False,npz_root_fields_read=False,world_units_inferred=False,terrain_constraints_implemented=False,
        self_collision_constraints_implemented=False,support_constraints_implemented=False,root_motion_generated=False,
        full_length_joint_diagnostic_generated=bool(accepted),candidate_motion_generated=False,accepted_reference=False,training_ready=False)
    return q,report