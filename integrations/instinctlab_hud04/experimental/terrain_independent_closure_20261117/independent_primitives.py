"""Independent ALL-triangle primitive distances; no GJK or primary broadphase.

Cylinder: projection onto finite cylinder + convex triangle tangent lower bounds.
Bounded 2-D barycentric minimization only tightens these checkable bounds.
Nothing here solves robot motion or changes model/joint limits.
"""
import itertools
import numpy as np
import trimesh
from scipy.optimize import minimize


def local_triangles(shape,world,triangles):
    transform=world@shape.local
    return (triangles-transform[:3,3])@transform[:3,:3]


def project_cylinder(points,radius,half):
    p=np.asarray(points,float);result=p.copy();radial=np.linalg.norm(p[...,:2],axis=-1)
    ratio=np.minimum(1.,np.divide(radius,radial,out=np.ones_like(radial),where=radial>0))
    result[...,:2]*=ratio[...,None];result[...,2]=np.clip(p[...,2],-half,half)
    return result


def sphere_distance(shape,world,triangles):
    t=local_triangles(shape,world,triangles)
    nearest=trimesh.triangles.closest_point(t,np.zeros((len(t),3)))
    return max(0.,float(np.linalg.norm(nearest,axis=1).min())-shape.parameters['radius'])


def segment_distances(a,b,c,d):
    # One box edge vs a vector of triangle edges, including parallel/endpoints.
    u=b-a;v=d-c;w=a-c;uu=float(u@u);vv=np.sum(v*v,axis=1);uv=v@u;uw=w@u;vw=np.sum(v*w,axis=1)
    out=np.full(len(c),np.inf)
    for p in [a,b]:
        t=np.divide(np.sum((p-c)*v,axis=1),vv,out=np.zeros_like(vv),where=vv>0)
        out=np.minimum(out,np.sum((p-c-np.clip(t,0,1)[:,None]*v)**2,axis=1))
    for p in [c,d]:
        s=np.clip((p-a)@u/uu,0,1);out=np.minimum(out,np.sum((p-a-s[:,None]*u)**2,axis=1))
    det=uu*vv-uv*uv;valid=det>np.finfo(float).eps*uu*vv*8
    s=np.divide(uv*vw-vv*uw,det,out=np.zeros_like(det),where=valid)
    t=np.divide(uu*vw-uv*uw,det,out=np.zeros_like(det),where=valid)
    valid&=(s>=0)&(s<=1)&(t>=0)&(t<=1)
    return np.minimum(out,np.where(valid,np.sum((w+s[:,None]*u-t[:,None]*v)**2,axis=1),np.inf))


def box_distance(shape,world,triangles):
    t=local_triangles(shape,world,triangles);half=np.array(shape.parameters['size'])/2
    # Full 13-axis triangle/AABB SAT detects crossings and complete containment.
    axes=[np.broadcast_to(a,(len(t),3)) for a in np.eye(3)]
    edges=np.roll(t,-1,axis=1)-t
    axes+=[np.cross(edges[:,0],edges[:,1])]
    axes+=[np.cross(edges[:,i],a) for i in range(3) for a in np.eye(3)]
    separated=np.zeros(len(t),bool)
    for axis in axes:
        proj=np.einsum('nki,ni->nk',t,axis);radius=abs(axis)@half
        separated|=(proj.min(axis=1)>radius)|(proj.max(axis=1)<-radius)
    if np.any(~separated):return 0.
    best=np.sum((t-np.clip(t,-half,half))**2,axis=2).min(axis=1)
    vertices=np.array(list(itertools.product([-1,1],repeat=3)))*half
    for point in vertices:
        near=trimesh.triangles.closest_point(t,np.broadcast_to(point,(len(t),3)))
        best=np.minimum(best,np.sum((near-point)**2,axis=1))
    for i,a in enumerate(vertices):
        for j,b in enumerate(vertices):
            if j<=i or np.count_nonzero(a!=b)!=1:continue
            for k in range(3):best=np.minimum(best,segment_distances(a,b,t[:,k],t[:,(k+1)%3]))
    return float(np.sqrt(max(0.,best.min())))


def cylinder_bracket(shape,world,triangles,*,max_fw=80,max_local=32,include_face_plane_lower=False):
    t=local_triangles(shape,world,triangles);radius=shape.parameters['radius'];half=shape.parameters['length']/2
    distances=np.linalg.norm(t-project_cylinder(t,radius,half),axis=2)
    p=t[np.arange(len(t)),np.argmin(distances,axis=1)].copy();lower=np.zeros(len(t));upper=float(distances.min())
    if include_face_plane_lower:
        normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);norm=np.linalg.norm(normal,axis=1)
        normal=np.divide(normal,norm[:,None],out=np.zeros_like(normal),where=norm[:,None]>0)
        projection=np.einsum('nki,ni->nk',t,normal)
        extent=radius*np.linalg.norm(normal[:,:2],axis=1)+half*abs(normal[:,2])
        lower=np.maximum(0.,np.maximum(projection.min(axis=1)-extent,-extent-projection.max(axis=1)))
    count=0
    for count in range(1,max_fw+1):
        ids=np.flatnonzero(lower<upper-1e-10)
        if len(ids)==0 or upper-float(lower.min())<=1e-8:break
        point=p[ids];delta=point-project_cylinder(point,radius,half);distance=np.linalg.norm(delta,axis=1)
        upper=min(upper,float(distance.min()))
        gradient=np.divide(delta,distance[:,None],out=np.zeros_like(delta),where=distance[:,None]>0)
        tangent=distance+np.min(np.einsum('nki,ni->nk',t[ids]-point[:,None],gradient),axis=1)
        lower[ids]=np.maximum(lower[ids],np.maximum(0.,tangent))
        if upper-float(lower.min())<=1e-8:break
        vertex=t[ids,np.argmin(np.einsum('nki,ni->nk',t[ids],gradient),axis=1)]
        direction=vertex-point;left=np.zeros(len(ids));right=np.ones(len(ids))
        def objective(x):
            z=point+x[:,None]*direction;d=z-project_cylinder(z,radius,half)
            return np.sum(d*d,axis=1)
        for _ in range(24):
            a=(2*left+right)/3;b=(left+2*right)/3;choose=objective(a)<objective(b)
            left=np.where(choose,left,a);right=np.where(choose,b,right)
        p[ids]=point+((left+right)/2)[:,None]*direction
    initial_width=upper-float(lower.min());calls=0
    for _ in range(max_local):
        if upper-float(lower.min())<=1e-8:break
        i=int(np.argmin(lower));tri=t[i];origin=tri[0];edges=(tri[1:]-origin).T
        beta=np.maximum(0.,np.linalg.lstsq(edges,p[i]-origin,rcond=None)[0]);beta/=max(1.,float(beta.sum()))
        def objective(beta):
            point=origin+edges@beta;delta=point-project_cylinder(point,radius,half)
            return .5*float(delta@delta),edges.T@delta
        result=minimize(objective,beta,jac=True,method='SLSQP',bounds=[(0.,1.),(0.,1.)],
            constraints=[dict(type='ineq',fun=lambda x:1.-x.sum(),jac=lambda x:-np.ones(2))],
            options=dict(maxiter=100,ftol=1e-18))
        calls+=1;beta=np.maximum(0.,result.x);beta/=max(1.,float(beta.sum()))
        point=origin+edges@beta;delta=point-project_cylinder(point,radius,half);distance=float(np.linalg.norm(delta))
        upper=min(upper,distance);tangent=0. if distance==0 else distance+float(np.min((tri-point)@(delta/distance)))
        old=lower[i];lower[i]=max(lower[i],0.,tangent)
        # A local lower bound may exceed GLOBAL upper. Compare only LOCAL bounds.
        if lower[i]>distance+1e-8:raise RuntimeError('Inconsistent local bracket')
        if lower[i]<=old+1e-14 and upper-float(lower.min())>1e-8:break
    lb=float(lower.min())
    if lb>upper+1e-8:raise RuntimeError('Inconsistent global bracket')
    return dict(status='converged' if upper-lb<=1e-8 else 'unresolved_bounded_geometry_query',
        lower=lb,upper=upper,initial_fw_width=initial_width,fw_iterations=count,local_refinement_calls=calls,
        triangle_count=len(t),max_fw_iterations=max_fw,max_local_calls=max_local,max_local_iterations=100,
        analytic_face_plane_lower_enabled=include_face_plane_lower,
        optimizer_success_used_as_certificate=False,certificate='convex_tangent_lower_and_feasible_point_upper',
        gjk_called=False,primary_broadphase_used=False,retarget_solver_called=False)