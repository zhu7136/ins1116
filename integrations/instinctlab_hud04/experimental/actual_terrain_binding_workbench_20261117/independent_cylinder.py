"""Independent finite-cylinder vs ALL triangles via convex point projection.

No GJK, no support mapping, no primary AABB candidate list. Per-triangle convex
Frank-Wolfe bounds plus bounded line minimization; retains unresolved brackets.
"""
import numpy as np
from terrain_binding import check_rigid


def project_cylinder(points,radius,half_length):
    p=np.asarray(points,float);q=p.copy();rho=np.linalg.norm(p[:,:2],axis=1)
    q[:,:2]*=np.minimum(1,np.divide(radius,rho,out=np.ones_like(rho),where=rho>0))[:,None]
    q[:,2]=np.clip(q[:,2],-half_length,half_length)
    return q


def cylinder_surface_bracket(shape,world,triangles,*,iterations=80,tolerance=1e-8,return_state=False):
    if shape.kind!='cylinder' or type(iterations) is not int or not 1<=iterations<=100:raise ValueError('Bounded explicit finite cylinder required')
    tf=check_rigid(world)@check_rigid(shape.local)
    t=(np.asarray(triangles,float)-tf[:3,3])@tf[:3,:3]
    if t.ndim!=3 or t.shape[1:]!=(3,3) or not 1<=len(t)<=100000 or not np.isfinite(t).all():raise ValueError('Bounded triangles')
    radius=shape.parameters['radius'];half=shape.parameters['length']/2
    p=t.mean(axis=1);lower=np.zeros(len(t));active=np.arange(len(t));upper=np.inf;best=None
    for count in range(1,iterations+1):
        tri=t[active];points=p[active];projected=project_cylinder(points,radius,half)
        delta=points-projected;distances=np.linalg.norm(delta,axis=1)
        i=int(np.argmin(distances))
        if distances[i]<upper:upper=float(distances[i]);best=int(active[i])
        gradient=np.divide(delta,distances[:,None],out=np.zeros_like(delta),where=distances[:,None]>0)
        scores=np.einsum('nij,nj->ni',tri,gradient);vertex=tri[np.arange(len(tri)),np.argmin(scores,axis=1)]
        tangent=distances+np.einsum('ni,ni->n',gradient,vertex-points)
        lower[active]=np.maximum(lower[active],np.maximum(0.,tangent))
        global_lower=float(lower.min())
        if upper-global_lower<=tolerance:break
        keep=lower[active]<upper-tolerance;active=active[keep]
        if not len(active):break
        start=points[keep];direction=vertex[keep]-start
        left=np.zeros(len(active));right=np.ones(len(active))
        def squared(fraction):
            point=start+fraction[:,None]*direction;res=point-project_cylinder(point,radius,half)
            return np.einsum('ni,ni->n',res,res)
        for _ in range(36):
            a=(2*left+right)/3;b=(left+2*right)/3;choose=squared(a)<squared(b)
            left=np.where(choose,left,a);right=np.where(choose,b,right)
        fraction=(left+right)/2
        # Include endpoints so approximate line search cannot worsen a bound.
        choices=np.stack([squared(np.zeros(len(active))),squared(fraction),squared(np.ones(len(active)))],axis=1)
        picked=np.argmin(choices,axis=1);fraction=np.where(picked==0,0.,np.where(picked==2,1.,fraction))
        p[active]=start+fraction[:,None]*direction
    lower_bound=float(lower.min())
    result=dict(status='converged' if upper-lower_bound<=tolerance else 'unresolved_iteration_budget',
        lower=lower_bound,upper=upper,iterations=count,triangle_count=len(t),best_triangle=best,
        method='closed_form_cylinder_projection_and_triangle_convex_tangent_bounds',
        gjk_called=False,primary_support_mapping_called=False,primary_aabb_candidates_used=False)
    if return_state:return result,dict(triangles=t,points=p,lower=lower)
    return result