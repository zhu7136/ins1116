"""Bounded CPU trial006: all43 new-contact refresh on EVERY backtracking step.

Finite pose-task correspondence is still unaccepted. No full motion acceptance.
"""
import time,warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog,OptimizeWarning
from world_features import perturb,weights,check_roots
from world_solver import hard_rows,joint_feasibility
from collision_temporal_solver import l1_inequalities
from terrain_temporal_terms import temporal_system,temporal_values
from wholebody_terms import WholeBodyTerms

def plan():
    return [(int(s),int(s)+10) for s in np.rint(np.linspace(0,986,100)).astype(int)]

def geometry_cost(body,gaps):return float(np.maximum(body.margin-gaps,0).sum())

def solve(model,q,roots,desired,progress,*,enforce_foot_guard=False):
    q=q.copy();roots=check_roots(roots,len(q)).copy();seed=q.copy()
    if q.shape!=(997,29) or desired.shape!=(997,90) or max(joint_feasibility(model,q).values())>1e-6:raise ValueError('Native feasible997 only')
    body=WholeBodyTerms();w=weights();rho=.001;weight=1000.;start=time.monotonic();history=[];coverage=np.zeros(997,int)
    for number,(first,last) in enumerate(plan()):
        if time.monotonic()-start>140:break
        frames=slice(first,last+1);piece=q[frames];root=roots[frames];n=len(piece);nv=n*35;nq=n*29
        value,j=model.evaluate(piece,root,True);res=((value-desired[frames])*w).reshape(-1)
        a=sparse.block_diag([sparse.csr_matrix(x*w[:,None]) for x in j],format='csr')
        hard,rhs=hard_rows(model,q,first,last)
        select=sparse.coo_matrix((np.ones(nq),(np.arange(nq),[i*35+6+k for i in range(n) for k in range(29)])),shape=(nq,nv)).tocsr()
        dep=(piece-seed[frames]).reshape(-1);temp,tj=temporal_system(q,roots,first,last)
        gaps,gj,records,counts=body.evaluate(model,piece,root,True)
        if enforce_foot_guard:
            from foot_guard import foot_state,guard
            old_foot=foot_state(body,gaps,records)
        # Same-link shapes remain RECORDED in nonlinear merit. Their
        # ADAPTIVE-axis relative score is rigid-motion invariant. Frozen-axis
        # Jacobians need not vanish; do not let that surrogate spend a step
        # attempting to change the invariant same-body overlap.
        same_body=np.array([r['kind']=='self' and body.shapes[r['first']].link==body.shapes[r['second']].link for r in records],bool)
        movable=np.asarray(gj.getnnz(axis=1)>0).ravel()&~same_body;gaps_lp=gaps[movable];gj=gj[movable]
        all_a=sparse.vstack([a,rho*select,tj],format='csr');all_res=np.r_[res,rho*dep,temp];na=len(all_res);ng=len(gaps_lp)
        eq,er=l1_inequalities(all_a,all_res,nv)
        matrix=sparse.vstack([sparse.hstack([hard,sparse.csr_matrix((len(rhs),na+ng))]),
            sparse.hstack([eq,sparse.csr_matrix((2*na,ng))]),
            sparse.hstack([-weight*gj,sparse.csr_matrix((ng,na)),-sparse.eye(ng,format='csr')])],format='csr')
        b=np.r_[rhs,er,weight*(gaps_lp-body.margin)];trust=np.tile(np.r_[np.full(3,.015),np.full(3,.04),np.full(29,.08)],n)
        bounds=np.c_[-trust,trust].tolist()+[(0,None)]*(na+ng)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
            lp=linprog(np.r_[np.zeros(nv),np.ones(na+ng)],A_ub=matrix,b_ub=b,bounds=bounds,method='highs',
                options={'time_limit':1.5,'maxiter':12000,'threads':2,'parallel':False})
        row=dict(update=number,first=first,last=last,lp_status=int(lp.status),lp_iterations=int(lp.nit),accepted=False,
            geometry_rows_recorded=len(gaps),geometry_rows_with_movable_jacobian=ng,old_geometry_score_raw=geometry_cost(body,gaps),old_counts=counts,backtracking=[])
        coverage[frames]+=1
        if lp.success:
            d=lp.x[:nv].reshape(n,35);old=float(abs(all_res).sum()+weight*geometry_cost(body,gaps))
            for alpha in [1.,.5,.25,.125,.0625]:
                tq=piece+alpha*d[:,6:];tr=perturb(root,alpha*d[:,:6]);newq=q.copy();newr=roots.copy();newq[frames]=tq;newr[frames]=tr
                if np.max(hard@(alpha*d.reshape(-1))-rhs)>1e-6:raise ValueError('Hard violation; NO clipping')
                # Crucial change: fresh ALL shapes/903pairs/full finite terrain
                # candidate set; never accept on only the old fixed cuts.
                newgap,newrecords,newcounts=body.evaluate(model,tq,tr,False)
                newtemp=temporal_values(newq,newr,first,last)
                merit=float(abs((model.evaluate(tq,tr)-desired[frames])*w).sum()+rho*abs(tq-seed[frames]).sum()+abs(newtemp).sum()+weight*geometry_cost(body,newgap))
                oldkeys={(r['kind'],r['frame'],r['first'],r['second']) for r in records}
                newkeys={(r['kind'],r['frame'],r['first'],r['second']) for r in newrecords}
                row['backtracking'].append(dict(alpha=alpha,merit=merit,geometry_score_raw=geometry_cost(body,newgap),
                    new_active_shape_pairs=len(newkeys-oldkeys),all43_and903_rechecked=True))
                if enforce_foot_guard:
                    passed,detail=guard(old_foot,foot_state(body,newgap,newrecords))
                    row['backtracking'][-1].update(foot_guard_passed=passed,foot_guard_detail=detail)
                    if not passed:continue
                if merit<old-1e-9:
                    q[frames]=tq;roots[frames]=tr
                    row.update(accepted=True,alpha=alpha,merit_before=old,merit_after=merit,new_geometry_score_raw=geometry_cost(body,newgap),
                        new_counts=newcounts,max_joint_step_rad=float(abs(alpha*d[:,6:]).max()))
                    break
        history.append(row);progress(row)
        if not lp.success:raise RuntimeError('LP failure; NO motion export')
    feasible=joint_feasibility(model,q)
    if max(feasible.values())>1e-6:raise ValueError('Full997 named limits lost')
    return q,roots,dict(actual_lp_calls=len(history),accepted_updates=sum(r['accepted'] for r in history),
        planned_lp_calls=100,coverage_min=int(coverage.min()),coverage_max=int(coverage.max()),
        visited_frames=int(np.count_nonzero(coverage)),changed_frames=int(np.any(abs(q-seed)>1e-9,axis=1).sum()),
        elapsed_seconds=time.monotonic()-start,termination='finite_plan_finished_NOT_convergence' if len(history)==100 else 'finite140sec_budget_NOT_convergence',
        nonlinear_all43_newcontact_refresh=True,sampled_separation_axes_are_not_penetration_depth=True,
        no_new_foot_frame_and_no_foot_overlap_score_increase_guard=enforce_foot_guard,
        same_link_adaptive_scores_constant_not_geometry_exemptions=True,accepted_reference=False,training_ready=False,**feasible)