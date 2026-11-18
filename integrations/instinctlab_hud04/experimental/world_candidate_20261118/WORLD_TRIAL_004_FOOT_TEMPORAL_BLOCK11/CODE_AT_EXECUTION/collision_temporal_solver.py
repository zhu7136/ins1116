"""Third bounded trial: nominal local foot cuts + native temporal L1 terms.

Does NOT assert collision free or training ready. Failed LP => no export.
"""
import time,warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog,OptimizeWarning
from world_features import perturb,weights,check_roots
from world_solver import hard_rows,joint_feasibility
from terrain_temporal_terms import FootTerms,temporal_system,temporal_values

def plan():
    starts=np.rint(np.linspace(0,986,100)).astype(int).tolist()
    return [(s,s+10) for s in starts]+[(s,s+10) for s in starts[::-1]]

def l1_inequalities(a,res,nv):
    identity=sparse.eye(len(res),format='csr')
    return sparse.vstack([sparse.hstack([a,-identity]),sparse.hstack([-a,-identity])],format='csr'),np.r_[-res,res]

def solve(model,q,roots,desired,progress):
    q=q.copy();roots=check_roots(roots,len(q)).copy();seed=q.copy();initial_roots=roots.copy()
    if q.shape!=(997,29) or desired.shape!=(997,90):raise ValueError('Full native997 only')
    if max(joint_feasibility(model,q).values())>1e-6:raise ValueError('Feasible named seed required')
    start=time.monotonic();history=[];coverage=np.zeros(997,int);w=weights();rho=.001;foot_weight=100.;foot=FootTerms()
    for number,(first,last) in enumerate(plan()):
        if time.monotonic()-start>80:break
        frames=slice(first,last+1);piece=q[frames];root=roots[frames];n=len(piece);nv=n*35;nq=n*29
        value,j=model.evaluate(piece,root,True);res=((value-desired[frames])*w).reshape(-1)
        a=sparse.block_diag([sparse.csr_matrix(x*w[:,None]) for x in j],format='csr')
        hard,rhs=hard_rows(model,q,first,last)
        select=sparse.coo_matrix((np.ones(nq),(np.arange(nq),[i*35+6+k for i in range(n) for k in range(29)])),shape=(nq,nv)).tocsr()
        dep=(piece-seed[frames]).reshape(-1)
        temp,tj=temporal_system(q,roots,first,last)
        gaps,gj,records=foot.cuts(model,piece,root)
        # L1 blocks share one variable vector [delta, abs_task, abs_q,
        # abs_temporal, negative_foot_slack]. Foot slacks guarantee geometric
        # linearization cannot make joint feasibility impossible.
        blocks=[a,rho*select,tj];residuals=[res,rho*dep,temp]
        all_a=sparse.vstack(blocks,format='csr');all_res=np.concatenate(residuals);na=len(all_res);ng=len(gaps)
        eq,er=l1_inequalities(all_a,all_res,nv)
        matrix=sparse.vstack([
            sparse.hstack([hard,sparse.csr_matrix((len(rhs),na+ng))]),
            sparse.hstack([eq,sparse.csr_matrix((2*na,ng))]),
            sparse.hstack([-foot_weight*gj,sparse.csr_matrix((ng,na)),-sparse.eye(ng,format='csr')])],format='csr')
        b=np.r_[rhs,er,foot_weight*gaps]
        trust=np.tile(np.r_[np.full(3,.025),np.full(3,.10),np.full(29,.15)],n)
        bounds=np.c_[-trust,trust].tolist()+[(0,None)]*(na+ng)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
            lp=linprog(np.r_[np.zeros(nv),np.ones(na+ng)],A_ub=matrix,b_ub=b,bounds=bounds,method='highs',
                       options={'time_limit':1.5,'maxiter':12000,'threads':2,'parallel':False})
        row=dict(update=number,first=first,last=last,lp_status=int(lp.status),lp_iterations=int(lp.nit),accepted=False,
                 active_triangle_plane_cuts=ng,old_negative_cut_l1_raw=float(np.maximum(-gaps,0).sum()),old_temporal_l1=float(abs(temp).sum()))
        coverage[frames]+=1
        if lp.success:
            d=lp.x[:nv].reshape(n,35);old=float(abs(all_res).sum()+foot_weight*np.maximum(-gaps,0).sum())
            for alpha in [1.,.5,.25,.125,.0625]:
                tq=piece+alpha*d[:,6:];tr=perturb(root,alpha*d[:,:6]);newq=q.copy();newr=roots.copy();newq[frames]=tq;newr[frames]=tr
                if np.max(hard@(alpha*d.reshape(-1))-rhs)>1e-6:raise ValueError('Hard velocity/position violation; no clipping')
                newgap=foot.fixed_cut_gaps(model,tq,tr,records);newtemp=temporal_values(newq,newr,first,last)
                merit=float(abs((model.evaluate(tq,tr)-desired[frames])*w).sum()+rho*abs(tq-seed[frames]).sum()+abs(newtemp).sum()+foot_weight*np.maximum(-newgap,0).sum())
                if merit<old-1e-9:
                    q[frames]=tq;roots[frames]=tr
                    row.update(accepted=True,alpha=alpha,merit_before=old,merit_after=merit,new_negative_cut_l1_raw=float(np.maximum(-newgap,0).sum()),
                               new_temporal_l1=float(abs(newtemp).sum()),max_joint_step_rad=float(abs(alpha*d[:,6:]).max()))
                    break
        history.append(row);progress(row)
        if not lp.success:raise RuntimeError('LP unsuccessful; NO motion export; inspect progress status '+str(lp.status))
    feasible=joint_feasibility(model,q)
    if max(feasible.values())>1e-6 or not np.all(coverage>0):raise ValueError('Full997 feasibility/coverage lost; no export')
    return q,roots,dict(actual_lp_calls=len(history),accepted_updates=sum(x['accepted'] for x in history),history=history,
        planned_lp_calls=len(plan()),coverage_min=int(coverage.min()),coverage_max=int(coverage.max()),
        final_joint_regularizer_l1=float(rho*abs(q-seed).sum()),elapsed_seconds=time.monotonic()-start,
        termination='finite_plan_finished_not_convergence' if len(history)==len(plan()) else 'finite_budget_stop_not_convergence',
        root_translation_change_from_compensated_seed_max_raw=float(np.linalg.norm(roots[:,:3,3]-initial_roots[:,:3,3],axis=1).max()),
        frames_changed=int(np.any(abs(q-seed)>1e-9,axis=1).sum()),root_6d_tangent_optimized=True,body_29d_optimized=True,
        footbox_triangle_escape_soft=True,footbox_weight=foot_weight,temporal_weights_before_dt_squared=[.02,.005,.0001],
        actual_nominal_footbox_cuts_are_not_complete_collision_constraints=True,**feasible)