"""Independent all-triangle sphere/OBB surface distances, no GJK/broadphase.

Transforms triangles into shape frame. OBB uses segment/triangle distances,
triangle-vertex to box clamp, and independent segment/slab intersections.
Unsigned solid-vs-SURFACE distance only; open terrain interior not excluded.
"""
from itertools import product
import numpy as np
import trimesh
from terrain_binding import check_rigid
from terrain_surface_geometry import segment_triangles


def slab_crossings(start,end,half):
    start,end=np.asarray(start,float),np.asarray(end,float)
    delta=end-start;left=np.zeros(len(start));right=np.ones(len(start));possible=np.ones(len(start),bool)
    for axis in range(3):
        parallel=np.abs(delta[:,axis])<1e-15
        possible &= ~parallel | (np.abs(start[:,axis])<=half[axis])
        low=np.divide(-half[axis]-start[:,axis],delta[:,axis],out=np.full(len(start),-np.inf),where=~parallel)
        high=np.divide(half[axis]-start[:,axis],delta[:,axis],out=np.full(len(start),np.inf),where=~parallel)
        left=np.maximum(left,np.minimum(low,high));right=np.minimum(right,np.maximum(low,high))
    return possible & (left<=right)


def distance(shape,link_world,triangles):
    world=check_rigid(link_world)@check_rigid(shape.local)
    t=(np.asarray(triangles,float)-world[:3,3])@world[:3,:3]
    if shape.kind=='sphere':
        p=np.zeros((len(t),3));closest=trimesh.triangles.closest_point(t,p)
        return max(0.,float(np.linalg.norm(closest,axis=1).min())-shape.parameters['radius'])
    if shape.kind!='box':raise ValueError('Only sphere/box independently implemented')
    half=np.array(shape.parameters['size'])/2
    vertices=t.reshape(-1,3);clamped=np.minimum(np.maximum(vertices,-half),half)
    best=float(np.linalg.norm(vertices-clamped,axis=1).min())
    for i,j in [(0,1),(1,2),(2,0)]:
        if slab_crossings(t[:,i],t[:,j],half).any():return 0.
    corners=np.array(list(product([-1,1],repeat=3)))*half
    for i,a in enumerate(corners):
        for b in corners[i+1:]:
            if np.count_nonzero(a!=b)!=1:continue
            distances,_,_=segment_triangles(a,b,t)
            best=min(best,float(distances.min()))
    return best