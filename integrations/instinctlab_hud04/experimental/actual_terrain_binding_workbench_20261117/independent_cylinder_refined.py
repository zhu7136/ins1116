"""Bounded refinement of independent projection/tangent DISTANCE brackets.

SLSQP is used only for two barycentric triangle coordinates, NOT retargeting.
Optimizer success flags are not certificates: convex tangent lower bounds are.
"""
import numpy as np
from scipy.optimize import minimize
from independent_cylinder import cylinder_surface_bracket,project_cylinder


def refined_bracket(shape,world,triangles):
    base,state=cylinder_surface_bracket(shape,world,triangles,return_state=True)
    lower=state['lower'].copy();upper=base['upper'];triangles=state['triangles'];points=state['points']
    radius=shape.parameters['radius'];half=shape.parameters['length']/2;calls=0
    for _ in range(32):
        if upper-float(lower.min())<=1e-8:break
        index=int(np.argmin(lower));tri=triangles[index];origin=tri[0];edges=(tri[1:]-origin).T
        initial=np.linalg.lstsq(edges,points[index]-origin,rcond=None)[0]
        initial=np.maximum(0,initial);initial/=max(1.,float(initial.sum()))
        def objective(beta):
            point=origin+edges@beta;delta=point-project_cylinder(point[None],radius,half)[0]
            return .5*float(delta@delta),edges.T@delta
        result=minimize(objective,initial,jac=True,method='SLSQP',bounds=[(0.,1.),(0.,1.)],
            constraints=[dict(type='ineq',fun=lambda x:1.-x.sum(),jac=lambda x:-np.ones(2))],
            options=dict(maxiter=100,ftol=1e-18))
        calls+=1;beta=np.maximum(0,result.x);beta/=max(1.,float(beta.sum()))
        point=origin+edges@beta;delta=point-project_cylinder(point[None],radius,half)[0];distance=float(np.linalg.norm(delta))
        upper=min(upper,distance)
        tangent=0. if distance==0 else distance+float(np.min((tri-point)@(delta/distance)))
        old=lower[index];lower[index]=max(lower[index],0.,tangent)
        # A non-nearest triangle's lower bound CAN exceed the GLOBAL upper.
        # Only its OWN feasible-point upper is a valid local consistency check.
        if lower[index]-distance>1e-8:raise RuntimeError('Inconsistent independent local distance bounds')
        if lower[index]<=old+1e-14 and upper-float(lower.min())>1e-8:break
    bound=float(lower.min())
    return dict(status='converged' if upper-bound<=1e-8 else 'unresolved_bounded_refinement',
        lower=bound,upper=upper,initial_status=base['status'],initial_width=base['upper']-base['lower'],
        local_refinement_calls=calls,max_local_calls=32,max_iterations_per_local_call=100,
        all_subset_triangles_evaluated=len(triangles),gjk_called=False,primary_aabb_candidates_used=False,
        optimizer_success_is_not_certificate=True,certificate='convex_distance_tangent_lower_and_feasible_point_upper',
        retarget_solver_called=False)