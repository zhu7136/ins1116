"""Bounded GJK distance brackets for explicit convex support mappings.

Not penetration depth. Nonconvergence is UNRESOLVED, never safe or accepted.
Support points retain witnesses on each ORIGINAL convex shape.
"""
from itertools import combinations
import numpy as np


def closest_simplex(vertices):
    if not 1<=len(vertices)<=5:raise ValueError('Simplex budget exceeded')
    best=None
    for count in range(1,min(4,len(vertices))+1):
        for subset in combinations(range(len(vertices)),count):
            p=np.array([vertices[i][0] for i in subset])
            if count==1:weights=np.ones(1)
            else:
                gram=p@p.T
                matrix=np.block([[gram,np.ones((count,1))],[np.ones((1,count)),np.zeros((1,1))]])
                try:weights=np.linalg.solve(matrix,np.r_[np.zeros(count),1.])[:count]
                except np.linalg.LinAlgError:continue
                if not np.isfinite(weights).all() or np.any(weights<0):continue
                weights/=weights.sum()
            v=weights@p;norm=float(v@v)
            if best is None or norm<best[0]:best=norm,subset,weights,v
    _,subset,weights,v=best
    first=sum(w*vertices[i][1] for i,w in zip(subset,weights));second=sum(w*vertices[i][2] for i,w in zip(subset,weights))
    kept=[vertices[i] for i,w in zip(subset,weights) if w>0]
    return kept,v,first,second


def convex_distance(first,first_world,second,second_world,*,iterations=96,tolerance=1e-9):
    if type(iterations) is not int or not 1<=iterations<=128 or not np.isfinite(tolerance) or not 1e-12<=tolerance<=1e-8:
        raise ValueError('Bounded numerical query required')
    def support(direction):
        a=first.support([direction],first_world)[0];b=second.support([-direction],second_world)[0]
        return a-b,a,b
    simplex=[support(np.array([1.,0,0]))];lower=0.;converged=False
    for count in range(1,iterations+1):
        simplex,v,a,b=closest_simplex(simplex);upper=float(np.linalg.norm(v))
        if upper<=tolerance:converged=True;break
        new=support(-v);plane_lower=float(v@new[0])/upper
        lower=max(lower,0.,plane_lower)
        if lower>upper+tolerance:raise RuntimeError('Inconsistent support bracket')
        if upper-lower<=tolerance:converged=True;break
        if any(np.array_equal(new[0],old[0]) for old in simplex):break
        if count<iterations:simplex.append(new)
    return dict(status=('within_numerical_tolerance' if upper<=tolerance else 'separated') if converged else 'unresolved_iteration_or_stagnation',
        converged=converged,distance_lower_bound=lower,distance_upper_bound=upper,
        first_witness=a.tolist(),second_witness=b.tolist(),iterations=count,tolerance=tolerance,
        penetration_depth_computed=False,nominal_geometry_only=True,cooked_physics_validated=False,
        accepted_reference=False,training_ready=False)