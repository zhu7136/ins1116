"""Finite overlapping 35D root+joint trial, NOT contact/collision acceptance.

Hard original model position/90Hz velocity constraints, including fixed outside
neighbours. Full original 997frames retained. No interpolation or q clipping.
"""
import sys,time,warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog,OptimizeWarning
from world_features import perturb,weights,check_roots
sys.path.insert(0,'/workspace/hud04-full-direction-20261118')
from overlap_solver import window_rows

def plan():
    starts=np.rint(np.linspace(0,966,49)).astype(int).tolist()
    return [(s,s+30) for s in starts]+[(s,s+30) for s in starts[::-1]]

def hard_rows(model,q,first,last):
    g,h=window_rows(model.model,q,first,last);n=last-first+1
    extract=sparse.coo_matrix((np.ones(n*29),(np.arange(n*29),np.array([i*35+6+j for i in range(n) for j in range(29)]))),shape=(n*29,n*35)).tocsr()
    return g@extract,h-g@q[first:last+1].reshape(-1)

def joint_feasibility(model,q):
    m=model.model
    return dict(position_max_violation_rad=float(max(0,np.max(m.lower-q),np.max(q-m.upper))),
                velocity_max_violation_rad_s=float(max(0,np.max(abs(np.diff(q,axis=0))*90-m.vmax))))

def solve(model,q,roots,desired,progress):
    q=q.copy();roots=check_roots(roots,len(q)).copy();seed=q.copy();initial_roots=roots.copy()
    if q.shape!=(997,29) or desired.shape!=(997,90):raise ValueError('Full native997 only')
    if max(joint_feasibility(model,q).values())>1e-6:raise ValueError('Feasible named seed required')
    start=time.monotonic();history=[];coverage=np.zeros(997,int);w=weights();rho=.001
    for number,(first,last) in enumerate(plan()):
        if time.monotonic()-start>80:break
        frames=slice(first,last+1);piece=q[frames];root=roots[frames];n=len(piece);nv=n*35;nm=n*90;nq=n*29
        value,j=model.evaluate(piece,root,True);res=((value-desired[frames])*w).reshape(-1)
        a=sparse.block_diag([sparse.csr_matrix(x*w[:,None]) for x in j],format='csr')
        hard,rhs=hard_rows(model,q,first,last)
        select=sparse.coo_matrix((np.ones(nq),(np.arange(nq),np.array([i*35+6+k for i in range(n) for k in range(29)]))),shape=(nq,nv)).tocsr()
        dep=(piece-seed[frames]).reshape(-1)
        ident=sparse.eye(nm,format='csr');regident=sparse.eye(nq,format='csr')
        zero=sparse.csr_matrix((nm,nq))
        matrix=sparse.vstack([
            sparse.hstack([hard,sparse.csr_matrix((len(rhs),nm+nq))]),
            sparse.hstack([a,-ident,zero]),sparse.hstack([-a,-ident,zero]),
            sparse.hstack([select,sparse.csr_matrix((nq,nm)),-regident]),
            sparse.hstack([-select,sparse.csr_matrix((nq,nm)),-regident])],format='csr')
        b=np.r_[rhs,-res,res,-dep,dep]
        trust=np.tile(np.r_[np.full(3,.025),np.full(3,.10),np.full(29,.15)],n)
        bounds=np.c_[-trust,trust].tolist()+[(0,None)]*(nm+nq)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
            lp=linprog(np.r_[np.zeros(nv),np.ones(nm),np.full(nq,rho)],A_ub=matrix,b_ub=b,
                       bounds=bounds,method='highs',options={'time_limit':1.5,'maxiter':12000,'threads':2,'parallel':False})
        row=dict(update=number,first=first,last=last,lp_status=int(lp.status),lp_iterations=int(lp.nit),accepted=False)
        coverage[frames]+=1
        if lp.success:
            d=lp.x[:nv].reshape(n,35);old=float(abs(res).sum()+rho*abs(dep).sum())
            for alpha in [1.,.5,.25,.125,.0625]:
                tq=piece+alpha*d[:,6:];tr=perturb(root,alpha*d[:,:6])
                if np.max(hard@(alpha*d.reshape(-1))-rhs)>1e-6:raise ValueError('Hard velocity/position violation; no clipping')
                merit=float(abs((model.evaluate(tq,tr)-desired[frames])*w).sum()+rho*abs(tq-seed[frames]).sum())
                if merit<old-1e-9:
                    q[frames]=tq;roots[frames]=tr
                    row.update(accepted=True,alpha=alpha,merit_before=old,merit_after=merit,
                               max_root_translation_step_raw=float(abs(alpha*d[:,:3]).max()),
                               max_root_rotation_step_rad=float(abs(alpha*d[:,3:6]).max()),max_joint_step_rad=float(abs(alpha*d[:,6:]).max()))
                    break
        history.append(row);progress(row)
        if not lp.success:raise RuntimeError('LP unsuccessful; NO motion export; inspect progress status '+str(lp.status))
    feasible=joint_feasibility(model,q)
    if max(feasible.values())>1e-6:raise ValueError('Full997 feasibility lost; no export')
    if not np.all(coverage>0):raise ValueError('Incomplete full coverage; no export')
    return q,roots,dict(actual_lp_calls=len(history),accepted_updates=sum(x['accepted'] for x in history),history=history,
        planned_lp_calls=len(plan()),coverage_min=int(coverage.min()),coverage_max=int(coverage.max()),
        final_joint_regularizer_l1=float(rho*abs(q-seed).sum()),
        root_translation_change_from_compensated_seed_max_raw=float(np.linalg.norm(roots[:,:3,3]-initial_roots[:,:3,3],axis=1).max()),
        frames_changed=int(np.any(abs(q-seed)>1e-9,axis=1).sum()),elapsed_seconds=time.monotonic()-start,
        termination='finite_plan_finished_not_convergence' if len(history)==len(plan()) else 'finite_budget_or_LP_stop_not_convergence',
        source_root_copied=False,root_6d_tangent_optimized=True,body_29d_optimized=True,**feasible)