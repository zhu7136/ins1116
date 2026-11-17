"""CPU triangle-surface diagnostics, not signed solid distances/contact labels.

Capsule margin = min(centerline, triangle surface) - radius. Negative means the
nominal capsule intersects a triangle surface, NOT a signed penetration depth.
A positive value does not exclude containment inside a closed obstacle.
All lengths inherit input units. No solver, simulator, source edits or labels.
"""
import numpy as np
from scipy.spatial import cKDTree


def processed_terrain(path, size):
    """Mirror reviewed crop/border operations with the installed trimesh version.

    Returns the cropped but untranslated mesh and shared translation. Does not
    mutate the source file, add walls, or declare units/provenance verified.
    """
    import trimesh
    size=np.asarray(size,dtype=float)
    if size.shape!=(2,) or not np.isfinite(size).all() or np.any(size<=0):
        raise ValueError('Explicit positive finite tile size required')
    mesh=trimesh.load(path,force='mesh')
    input_faces=len(mesh.faces);input_vertices=len(mesh.vertices)
    for normal,origin in [([-1,0,0],[size[0]/2,0,0]),([1,0,0],[-size[0]/2,0,0]),
                          ([0,-1,0],[0,size[1]/2,0]),([0,1,0],[0,-size[1]/2,0])]:
        mesh=trimesh.intersections.slice_mesh_plane(mesh,plane_normal=normal,plane_origin=origin)
    if not len(mesh.faces):raise ValueError('Crop removed all faces')
    mask=(np.abs(mesh.vertices[:,0])>size[0]/2-.05)|(np.abs(mesh.vertices[:,1])>size[1]/2-.05)
    heights=mesh.vertices[mask,2]
    border=float(np.mean(heights)) if len(heights) else 0.
    shift=np.r_[size/2,-border]
    return mesh,shift,dict(trimesh_version=trimesh.__version__,input_faces=input_faces,
        input_processed_vertices=input_vertices,cropped_faces=len(mesh.faces),cropped_vertices=len(mesh.vertices),
        crop_bounds_raw=mesh.bounds.tolist(),border_vertex_count=int(mask.sum()),
        border_height_raw=border,border_fallback_used=not bool(len(heights)),
        border_z_range_raw=[float(heights.min()),float(heights.max())] if len(heights) else None,
        border_z_std_raw=float(heights.std()) if len(heights) else None,
        terrain_function_shared_shift_raw=shift.tolist())


def point_segment(p, a, b):
    ab = b-a
    den = np.sum(ab*ab, axis=-1)
    u = np.divide(np.sum((p-a)*ab, axis=-1), den,
                  out=np.zeros_like(den), where=den > 0)
    return a+np.clip(u, 0, 1)[..., None]*ab


def point_triangles(p, triangles):
    """Closest point on every triangle, including edge/point degeneracies."""
    p = np.broadcast_to(np.asarray(p,dtype=float),(len(triangles),3))
    a, b, c = triangles.transpose(1, 0, 2)
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    n2 = np.sum(normal*normal, axis=1)
    t = np.divide(np.sum((p-a)*normal, axis=1), n2,
                  out=np.zeros_like(n2), where=n2 > 0)
    proj = p-t[:, None]*normal
    ap = proj-a
    d00, d01, d11 = (np.sum(v*w, axis=1) for v,w in [(ab,ab),(ab,ac),(ac,ac)])
    d20, d21 = np.sum(ap*ab, axis=1), np.sum(ap*ac, axis=1)
    den = d00*d11-d01*d01
    good = den > np.finfo(float).eps*np.maximum(d00*d11, np.finfo(float).tiny)*8
    u = np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=good)
    v = np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=good)
    inside = good & (u >= 0) & (v >= 0) & (u+v <= 1)
    choices = np.stack([proj,point_segment(p,a,b),point_segment(p,b,c),point_segment(p,c,a)], axis=1)
    dist2 = np.sum((choices-p[:,None,:])**2, axis=2)
    dist2[~inside,0] = np.inf
    return choices[np.arange(len(triangles)),np.argmin(dist2,axis=1)]


def segment_triangles(start, end, triangles):
    """Unsigned exact centerline-to-surface distances up to float tolerance."""
    start, end, triangles = map(lambda x:np.asarray(x,dtype=float), (start,end,triangles))
    if start.shape!=(3,) or end.shape!=(3,) or triangles.ndim!=3 or triangles.shape[1:]!=(3,3):
        raise ValueError('Expected segment endpoints and Nx3x3 triangles')
    if not all(np.isfinite(x).all() for x in (start,end,triangles)) or not len(triangles):
        raise ValueError('Nonempty finite geometry required')
    direction=end-start
    length2=float(direction@direction)
    n=len(triangles)
    sa=np.broadcast_to(start,(n,3)); sb=np.broadcast_to(end,(n,3))
    ta, tb = point_triangles(start,triangles), point_triangles(end,triangles)
    best=np.sum((sa-ta)**2,axis=1); best_s=sa.copy(); best_t=ta.copy()
    def update(ps,pt,valid=None):
        nonlocal best,best_s,best_t
        distance=np.sum((ps-pt)**2,axis=1)
        choose=distance<best
        if valid is not None:choose &= valid
        best[choose]=distance[choose];best_s[choose]=ps[choose];best_t[choose]=pt[choose]
    update(sb,tb)
    for i,j in [(0,1),(1,2),(2,0)]:
        a,b=triangles[:,i],triangles[:,j]
        update(sa,point_segment(start,a,b));update(sb,point_segment(end,a,b))
        update(point_segment(a,sa,sb),a);update(point_segment(b,sa,sb),b)
        edge=b-a; w=start-a
        ee=np.sum(edge*edge,axis=1);de=edge@direction
        dw=w@direction;ew=np.sum(edge*w,axis=1)
        den=length2*ee-de*de
        good=den>np.finfo(float).eps*np.maximum(length2*ee,np.finfo(float).tiny)*8
        s=np.divide(de*ew-ee*dw,den,out=np.zeros_like(den),where=good)
        t=np.divide(length2*ew-de*dw,den,out=np.zeros_like(den),where=good)
        update(start+s[:,None]*direction,a+t[:,None]*edge,
               good & (s>=0)&(s<=1)&(t>=0)&(t<=1))
    # Interior segment/triangle-plane intersections need an explicit candidate.
    a,b,c=triangles.transpose(1,0,2)
    normal=np.cross(b-a,c-a);den=normal@direction
    good=np.abs(den)>np.finfo(float).eps*np.maximum(np.linalg.norm(normal,axis=1)*np.sqrt(length2),np.finfo(float).tiny)*8
    t=np.divide(np.sum((a-start)*normal,axis=1),den,out=np.zeros_like(den),where=good)
    crossing=start+t[:,None]*direction
    nearest=point_triangles(crossing,triangles)
    tolerance=1e-12*np.maximum(1,np.linalg.norm(crossing,axis=1))
    inside=np.linalg.norm(crossing-nearest,axis=1)<=tolerance
    update(crossing,nearest,good & (t>=0)&(t<=1)&inside)
    return np.sqrt(best),best_s,best_t


class SurfaceIndex:
    """Exact AABB broadphase: a nearest endpoint vertex supplies an upper bound."""
    def __init__(self, triangles):
        self.triangles=np.asarray(triangles,dtype=float)
        if self.triangles.ndim!=3 or self.triangles.shape[1:]!=(3,3) or not len(self.triangles) or not np.isfinite(self.triangles).all():
            raise ValueError('Nonempty finite triangle soup required')
        self.lo=self.triangles.min(axis=1);self.hi=self.triangles.max(axis=1)
        self.vertices=cKDTree(self.triangles.reshape(-1,3))

    def closest(self,start,end):
        start,end=map(lambda p:np.asarray(p,dtype=float),(start,end))
        if start.shape!=(3,) or end.shape!=(3,) or not np.isfinite([start,end]).all():
            raise ValueError('Finite endpoints required')
        upper=float(np.min(self.vertices.query([start,end])[0]))
        lo,hi=np.minimum(start,end),np.maximum(start,end)
        gap=np.maximum(0,np.maximum(self.lo-hi,lo-self.hi))
        ids=np.flatnonzero(np.sum(gap*gap,axis=1) <= upper*upper+1e-12*max(1,upper*upper))
        distance,ps,pt=segment_triangles(start,end,self.triangles[ids])
        i=int(np.argmin(distance))
        return float(distance[i]),int(ids[i]),ps[i],pt[i]


def candidate_intervals(frames, values, threshold):
    """Runs of near-surface *geometry candidates*, never accepted support labels."""
    frames=np.asarray(frames);values=np.asarray(values)
    if frames.ndim!=1 or values.shape!=frames.shape or not len(frames) or not np.issubdtype(frames.dtype,np.integer) or np.any(np.diff(frames)<=0):
        raise ValueError('Strictly increasing integer frames required')
    if not np.isfinite(values).all() or not np.isfinite(threshold) or threshold<0:
        raise ValueError('Finite values and nonnegative threshold required')
    result=[];start=previous=None
    for frame,active in zip(frames,np.abs(values)<=threshold):
        frame=int(frame)
        if active:
            if previous is None or frame!=previous+1:
                if previous is not None:result.append([start,previous])
                start=frame
            previous=frame
        elif previous is not None:
            result.append([start,previous]);start=previous=None
    if previous is not None:result.append([start,previous])
    return result