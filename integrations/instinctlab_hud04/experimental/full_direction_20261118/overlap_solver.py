"""Bounded overlapping block-coordinate descent of the SAME global objective.

All 997 frames remain present. Every local update enforces native velocity rows
to current neighbours; no interpolation, blending or missing-frame stitching.
The .001 regularizer always references the ORIGINAL pinned feasible seed.
"""
import time,warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog,OptimizeWarning
from full_direction_solver import validate,stats,inequalities

def schedule():
    starts=np.rint(np.linspace(0,904,17)).astype(int).tolist()
    if starts[0]!=0 or starts[-1]!=904 or len(set(starts))!=17 or any(b-a>93 for a,b in zip(starts,starts[1:])):raise ValueError('Complete native coverage')
    windows=[(s,s+92) for s in starts];plan=[]
    for sweep in range(4):
        for first,last in (windows if sweep%2==0 else windows[::-1]):plan.append((sweep,'coverage',first,last))
        for repeat in range(2):
            for first,last in (windows[0],windows[-1]):plan.append((sweep,'endpoint_extra',first,last))
    if len(plan)!=84:raise ValueError('Fixed finite update plan')
    return plan

def window_rows(model,current,first,last):
    if current.shape!=(997,29) or type(first) is not int or type(last) is not int or not 0<=first<=last<997 or last-first+1>93:raise ValueError('Native bounded block')
    n=last-first+1;eye=sparse.eye(n*29,format='csr');rows=[];cols=[];values=[];offset=[]
    for edge in range(max(0,first-1),min(996,last+1)):
        for j in range(29):
            row=len(offset);constant=0.
            for frame,sign in ((edge,-1.),(edge+1,1.)):
                if first<=frame<=last:rows.append(row);cols.append((frame-first)*29+j);values.append(sign)
                else:constant+=sign*current[frame,j]
            offset.append(constant)
    d=sparse.coo_matrix((values,(rows,cols)),shape=(len(offset),n*29)).tocsr();offset=np.array(offset);speed=np.tile(model.vmax/90,len(offset)//29)
    return sparse.vstack([eye,-eye,d,-d],format='csr'),np.r_[np.tile(model.upper,n),-np.tile(model.lower,n),speed-offset,speed+offset]

def solve(model,seed,desired,gate,progress=None):
    baseline,desired=validate(model,seed,desired)
    if len(baseline)!=997:raise ValueError('Complete native Step1 only')
    q=baseline.copy();history=[];visited=np.zeros(997,dtype=int);start=time.monotonic();accepted=0;reason='planned_84_updates_completed';line_evaluations=0
    initial=stats(model.evaluate(q),desired);global_value=initial['sum_l1_chordal']
    for index,(sweep,role,first,last) in enumerate(schedule()):
        gate.step();piece=q[first:last+1].copy();original=baseline[first:last+1];want=desired[first:last+1]
        n=len(piece);width=n*29;m=n*24;g,h=window_rows(model,q,first,last)
        u,j=model.evaluate(piece,True);res=(u-want).reshape(-1)
        a=sparse.block_diag([sparse.csr_matrix(x.reshape(24,29)) for x in j],format='csr');i=sparse.eye(width,format='csr');it=sparse.eye(m,format='csr')
        hard=sparse.hstack([g,sparse.csr_matrix((g.shape[0],m+width))],format='csr')
        track=sparse.vstack([sparse.hstack([a,-it,sparse.csr_matrix((m,width))]),sparse.hstack([-a,-it,sparse.csr_matrix((m,width))])],format='csr')
        reg=sparse.vstack([sparse.hstack([i,sparse.csr_matrix((width,m)),-i]),sparse.hstack([-i,sparse.csr_matrix((width,m)),-i])],format='csr')
        departure=(piece-original).reshape(-1);matrix=sparse.vstack([hard,track,reg],format='csr')
        rhs=np.r_[h-g@piece.reshape(-1),-res,res,-departure,departure]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
            result=linprog(np.r_[np.zeros(width),np.ones(m),np.full(width,.001)],A_ub=matrix,b_ub=rhs,
                bounds=[(-.15,.15)]*width+[(0,None)]*(m+width),method='highs',
                options={'time_limit':1.,'maxiter':10000,'threads':2,'parallel':False})
        row=dict(update=index,sweep=sweep,role=role,first=first,last=last,lp_status=int(result.status),lp_iterations=int(result.nit),accepted=False)
        visited[first:last+1]+=1
        if not result.success:
            history.append(row);reason='LP_stopped_status_'+str(result.status)
            if progress:progress(row)
            break
        step=result.x[:width].reshape(n,29);old=float(np.abs(u-want).sum()+.001*np.abs(piece-original).sum())
        for alpha in (1.,.5,.25,.125,.0625):
            trial=piece+alpha*step
            if np.max(g@trial.reshape(-1)-h)>1e-6:raise ValueError('Hard row loss; NO joint clipping permitted')
            value=float(np.abs(model.evaluate(trial)-want).sum()+.001*np.abs(trial-original).sum());line_evaluations+=1
            if value<old-1e-9:
                q[first:last+1]=trial;accepted+=1;global_value+=value-old
                row.update(accepted=True,alpha=alpha,local_objective_before=old,local_objective_after=value,
                    global_objective_after=global_value,max_joint_step=float(np.max(np.abs(alpha*step))))
                break
        history.append(row)
        if progress:progress(row)
        # A zero-descent block is a stationary LOCAL update, not convergence;
        # proceed to other blocks in the PREDECLARED plan (not a retry).
    g,h=inequalities(model,997)
    hard=float(max(0.,np.max(g@q.reshape(-1)-h)))
    position=float(max(0.,np.max(model.lower-q),np.max(q-model.upper)))
    velocity=float(max(0.,np.max(np.abs(np.diff(q,axis=0))*90-model.vmax)))
    if hard>1e-6 or position>1e-6 or velocity>1e-6:raise ValueError('Full-sequence hard feasibility lost')
    final=stats(model.evaluate(q),desired);final_objective=final['sum_l1_chordal']+.001*float(np.abs(q-baseline).sum())
    if abs(global_value-final_objective)>1e-7:raise ValueError('Global merit accounting mismatch')
    return q,dict(schema_version=1,stage='overlapping_full_length_root_local_joint_direction_diagnostic_NOT_reference',frames=997,native_fps=90,
        native_velocity_edges=996,single_global_objective=True,one_monolithic_LP=False,overlapping_block_coordinate_descent=True,
        fixed_original_seed_regularizer=True,interpolated_or_blended_frames=0,resampling=False,posthoc_joint_clipping=False,
        planned_lp_updates=84,actual_lp_calls=len(history),accepted_steps=accepted,line_search_evaluations=line_evaluations,
        history=history,termination=reason,lp_method='highs',native_highs_threads=2,lp_time_limit_seconds=1.,lp_iteration_limit=10000,
        joint_trust_radius_rad=.15,original_seed_variable_count=28913,maximum_block_frames=93,coverage_sweeps=4,endpoint_extra_updates=16,
        initial=initial,final=final,initial_global_objective=initial['sum_l1_chordal'],final_global_objective=final_objective,
        optimized_or_visited_frames=int((visited>0).sum()),per_frame_visit_min=int(visited.min()),per_frame_visit_max=int(visited.max()),
        changed_frames=int(np.any(np.abs(q-baseline)>1e-9,axis=1).sum()),max_joint_departure_rad=float(np.abs(q-baseline).max()),
        elapsed_seconds=time.monotonic()-start,position_max_violation_rad=position,velocity_max_violation_rad_s=velocity,
        world_root_optimized=False,npz_root_fields_read=False,world_units_inferred=False,terrain_constraints_implemented=False,
        self_collision_constraints_implemented=False,support_constraints_implemented=False,root_motion_generated=False,
        full_length_joint_diagnostic_generated=bool(accepted),candidate_motion_generated=False,accepted_reference=False,training_ready=False)