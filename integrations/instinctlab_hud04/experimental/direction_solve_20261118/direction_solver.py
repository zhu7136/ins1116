"""Bounded sequential LP for ROOT-LOCAL direction diagnostics, not full motion.

No terrain/support constraints, no source root, no full-body candidate exporter.
Native adjacent velocities and fixed outside-window neighbours are HARD rows.
"""
import time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from anatomical_directions import direction_statistics

def joint_inequalities(model,seed,first,last):
    seed=np.asarray(seed,float)
    if seed.shape!=(997,29) or not np.isfinite(seed).all() or type(first) is not int or type(last) is not int or not 0<=first<=last<997 or last-first+1>93:raise ValueError('Native bounded window')
    n=last-first+1;eye=sparse.eye(n*29,format='csr')
    rows=[];cols=[];vals=[];offset=[]
    # Include edges to UNCHANGED neighbours outside this diagnostic window.
    for t in range(max(0,first-1),min(996,last+1)):
        for j in range(29):
            row=len(offset);constant=0.
            for frame,sign in ((t,-1.),(t+1,1.)):
                if first<=frame<=last:rows.append(row);cols.append((frame-first)*29+j);vals.append(sign)
                else:constant+=sign*seed[frame,j]
            offset.append(constant)
    d=sparse.coo_matrix((vals,(rows,cols)),shape=(len(offset),n*29)).tocsr();offset=np.array(offset)
    vmax=np.tile(model.vmax/90,len(offset)//29)
    g=sparse.vstack([eye,-eye,d,-d],format='csr')
    h=np.r_[np.tile(model.upper,n),-np.tile(model.lower,n),vmax-offset,vmax+offset]
    return g,h

def solve(model,seed,desired,first,last,gate,*,outer_limit=12,trust_radius=.03):
    if type(outer_limit) is not int or not 1<=outer_limit<=12:raise ValueError('Finite outer limit')
    if isinstance(trust_radius,bool) or not isinstance(trust_radius,(int,float)) or not np.isfinite(trust_radius) or not 0<trust_radius<=.15:raise ValueError('Bounded joint trust radius')
    g,h=joint_inequalities(model,seed,first,last)
    baseline=seed[first:last+1].copy();q=baseline.copy();n=len(q);desired=np.asarray(desired,float)
    if desired.shape!=(n,8,3) or not np.isfinite(desired).all() or not np.allclose(np.linalg.norm(desired,axis=2),1,atol=1e-10,rtol=0):raise ValueError('Explicit unit source directions')
    if max(0.,float(np.max(g@q.reshape(-1)-h)))>1e-7:raise ValueError('Hard-feasible initial seed required')
    history=[];calls=0;evaluations=0;start=time.monotonic();reason='outer_budget_reached';regularization=.001
    def objective(values):
        nonlocal evaluations
        evaluations+=1
        return float(np.abs(model.evaluate(values)-desired).sum()+regularization*np.abs(values-baseline).sum())
    initial=direction_statistics(model.evaluate(q),desired);value=objective(q)
    for iteration in range(outer_limit):
        gate.step();u,j=model.evaluate(q,True);evaluations+=1
        a=sparse.block_diag([sparse.csr_matrix(x.reshape(24,29)) for x in j],format='csr');res=(u-desired).reshape(-1)
        width=29*n;m=len(res);z=sparse.csr_matrix((g.shape[0],m));i=sparse.eye(width,format='csr');it=sparse.eye(m,format='csr')
        # Variables: dq, absolute linearized direction residual, absolute total
        # departure from the original feasible seed (not just this iteration).
        hard=sparse.hstack([g,z,sparse.csr_matrix((g.shape[0],width))],format='csr')
        track=sparse.vstack([sparse.hstack([a,-it,sparse.csr_matrix((m,width))]),sparse.hstack([-a,-it,sparse.csr_matrix((m,width))])],format='csr')
        reg=sparse.vstack([sparse.hstack([i,sparse.csr_matrix((width,m)),-i]),sparse.hstack([-i,sparse.csr_matrix((width,m)),-i])],format='csr')
        depart=(q-baseline).reshape(-1)
        matrix=sparse.vstack([hard,track,reg],format='csr');rhs=np.r_[h-g@q.reshape(-1),-res,res,-depart,depart]
        result=linprog(np.r_[np.zeros(width),np.ones(m),np.full(width,regularization)],A_ub=matrix,b_ub=rhs,
            bounds=[(-trust_radius,trust_radius)]*width+[(0,None)]*(m+width),method='highs',options={'time_limit':2.,'maxiter':10000})
        calls+=1
        if not result.success:
            reason='LP_stopped_status_'+str(result.status);history.append(dict(iteration=iteration,lp_status=int(result.status),accepted=False));break
        step=result.x[:width].reshape(n,29);accepted=False
        for alpha in (1.,.5,.25,.125,.0625):
            candidate=q+alpha*step
            if np.max(g@candidate.reshape(-1)-h)>1e-7:raise ValueError('Hard feasibility lost; no clipping allowed')
            candidate_value=objective(candidate)
            if candidate_value<value-1e-9:
                old=value;q=candidate;value=candidate_value;accepted=True
                history.append(dict(iteration=iteration,lp_status=0,accepted=True,alpha=alpha,objective_before=old,objective_after=value,max_accepted_joint_step=float(np.abs(alpha*step).max())))
                break
        if not accepted:
            reason='no_nonlinear_descent_within_fixed_line_search';history.append(dict(iteration=iteration,lp_status=0,accepted=False));break
    stitched=seed.copy();stitched[first:last+1]=q
    position=float(max(0.,np.max(model.lower-stitched),np.max(stitched-model.upper)))
    velocity=float(max(0.,np.max(np.abs(np.diff(stitched,axis=0))*90-model.vmax)))
    if position>1e-6 or velocity>1e-6:raise ValueError('Independent full-997-frame hard check failed')
    return q,dict(stage='root_local_direction_joint_diagnostic_NOT_full_reference',first=first,last=last,frames=n,fps=90,
        objective='sum L1 unit-direction chordal residual + .001 total joint departure',
        initial=initial,final=direction_statistics(model.evaluate(q),desired),termination=reason,history=history,
        solver_calls=calls,nonlinear_evaluations=evaluations,elapsed_seconds=time.monotonic()-start,joint_trust_radius_rad=trust_radius,
        full_sequence_position_max_violation_rad=position,full_sequence_velocity_max_violation_rad_s=velocity,
        outside_window_neighbours_fixed=True,cross_gap_velocity_edges=0,posthoc_joint_clipping=False,
        full_retarget_solver_calls=0,world_root_optimized=False,root_quaternions_read=False,world_units_inferred=False,
        terrain_constraints_implemented=False,self_collision_constraints_implemented=False,support_labels_generated=False,
        partial_joint_direction_fit_generated=True,candidate_motion_generated=False,accepted_reference=False,training_ready=False)