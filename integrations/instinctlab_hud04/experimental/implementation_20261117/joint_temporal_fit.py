"""Bounded L1 temporal joint fit, NOT terrain/root/contact-aware retargeting.

Solves joint trajectories jointly over time; never clips joint-angle output.
No source root/quaternion interpretation, reference manifest, or training export.
"""
import time
import warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, OptimizeWarning


def residuals(q, lower, upper, vmax, fps=90):
    return dict(position_max_violation_rad=float(max(0.,np.max(lower-q),np.max(q-upper))),
                velocity_max_violation_rad_s=float(max(0.,np.max(np.abs(np.diff(q,axis=0))*fps-vmax))))


def fit(source, lower, upper, vmax, *, fps=90, seconds_per_joint=2., total_seconds=75.):
    source=np.asarray(source,dtype=float)
    lower,upper,vmax=(np.asarray(a,dtype=float) for a in (lower,upper,vmax))
    if source.ndim!=2 or source.shape[0]<3 or source.shape[1]!=len(lower): raise ValueError('Shape mismatch')
    if upper.shape!=lower.shape or vmax.shape!=lower.shape or not all(np.isfinite(a).all() for a in (source,lower,upper,vmax)): raise ValueError('Finite compatible arrays required')
    if np.any(lower>=upper) or np.any(vmax<=0) or fps!=90: raise ValueError('Native 90Hz and model limits required')
    if not 0<seconds_per_joint<=2 or not 0<total_seconds<=75: raise ValueError('Bounded CPU budget required')
    n,joints=source.shape
    identity=sparse.eye(n,format='csr')
    diff=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csr')
    zero=sparse.csr_matrix((n-1,n))
    matrix=sparse.vstack([sparse.hstack([identity,-identity]),sparse.hstack([-identity,-identity]),
                           sparse.hstack([diff,zero]),sparse.hstack([-diff,zero])],format='csr')
    cost=np.r_[np.zeros(n),np.ones(n)]
    start=time.monotonic();out=np.empty_like(source);rows=[]
    for j in range(joints):
        remaining=total_seconds-(time.monotonic()-start)
        if remaining<=0: raise TimeoutError('CPU fit deadline')
        b=np.r_[source[:,j],-source[:,j],np.full(2*(n-1),vmax[j]/fps)]
        bounds=[(float(lower[j]),float(upper[j]))]*n+[(0,None)]*n
        # SciPy forwards this HiGHS option; it is explicit and verified by the
        # solver, not a claim that all OS threads in the process are capped.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*',category=OptimizeWarning)
            result=linprog(cost,A_ub=matrix,b_ub=b,bounds=bounds,method='highs',options={
                'time_limit':min(seconds_per_joint,remaining),'maxiter':10000,'threads':1,
                'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        rows.append(dict(joint_index=j,status=int(result.status),iterations=int(result.nit),
                         success=bool(result.success),l1_objective=None if result.fun is None else float(result.fun)))
        if not result.success: raise RuntimeError('LP did not reach optimum for joint '+str(j))
        out[:,j]=result.x[:n]  # Solver primal variables, NOT clipped/projected afterward.
    report=dict(stage='joint_temporal_fit_only',solver='scipy.linprog/HiGHS/L1',
        elapsed_seconds=time.monotonic()-start,frames=n,framerate=fps,joints=joints,solver_calls=joints,
        per_joint=rows,residuals=residuals(out,lower,upper,vmax,fps),
        rms_change_rad=float(np.sqrt(np.mean((out-source)**2))),max_change_rad=float(np.max(np.abs(out-source))),
        posthoc_angle_clipping=False,full_retarget_solver_calls=0,root_generated=False,
        terrain_constraints_implemented=False,self_collision_constraints_implemented=False,
        contact_constraints_implemented=False,accepted_reference=False,training_ready=False)
    if report['elapsed_seconds']>total_seconds: raise TimeoutError('CPU budget exceeded')
    if max(report['residuals'].values())>1e-6: raise RuntimeError('Independent hard-row residual failed')
    return out,report