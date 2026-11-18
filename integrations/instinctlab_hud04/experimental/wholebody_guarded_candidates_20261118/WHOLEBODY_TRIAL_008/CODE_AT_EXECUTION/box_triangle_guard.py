"""Fresh finite-triangle OBB rejection checks. No optimizer or acceptance writer.

Positive SAT separation is NOT Euclidean distance. Negative SAT gap means
strict intersection for the nominal box/triangle pair, not terrain-solid depth.
Clear surfaces do NOT exclude containment, swept collision, or other bodies.
"""
import hashlib
import numpy as np


def rigid(value):
    t=np.asarray(value,float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1],atol=1e-12,rtol=0):
        raise ValueError('Finite SE3 required')
    if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-10,rtol=0) or abs(np.linalg.det(t[:3,:3])-1)>1e-10:
        raise ValueError('Proper rigid rotation required')
    return t


def sat_gaps(triangles,halfsize):
    """13 separating axes in BOX coordinates; exact finite triangles, no winding.

    Degenerate triangles are rejected as unresolved, never treated as clear.
    Returns largest normalized directional separation for each triangle.
    """
    tri=np.asarray(triangles,float);h=np.asarray(halfsize,float)
    if tri.ndim!=3 or tri.shape[1:]!=(3,3) or h.shape!=(3,) or not np.isfinite(tri).all() or not np.isfinite(h).all() or np.any(h<=0):
        raise ValueError('Finite triangles and positive halfsize required')
    if not len(tri):return np.empty(0)
    edges=np.roll(tri,-1,axis=1)-tri
    normals=np.cross(edges[:,0],edges[:,1])
    if np.any(np.linalg.norm(normals,axis=1)<1e-14):raise ValueError('Degenerate triangle unresolved')
    axes=np.concatenate([np.broadcast_to(np.eye(3),(len(tri),3,3)),normals[:,None,:],
                         np.cross(edges[:,:,None,:],np.eye(3)[None,None,:,:]).reshape(-1,9,3)],axis=1)
    lengths=np.linalg.norm(axes,axis=2);good=lengths>1e-14
    axes=np.divide(axes,lengths[:,:,None],out=np.zeros_like(axes),where=good[:,:,None])
    projections=np.einsum('nvi,nai->nav',tri,axes);radius=abs(axes)@h
    gaps=np.maximum(projections.min(axis=2)-radius,-radius-projections.max(axis=2))
    gaps[~good]=-np.inf
    return gaps.max(axis=1)


def clip_polygon(vertices,axis,bound,sign):
    """Independent polygon halfspace clipping; sign*(x_axis-bound)<=0."""
    p=np.asarray(vertices,float).reshape(-1,3);out=[]
    if not len(p):return p
    f=sign*(p[:,axis]-bound)
    for k,a in enumerate(p):
        b=p[(k+1)%len(p)];fa,fb=f[k],f[(k+1)%len(p)]
        if fa<=0:out.append(a)
        if (fa<=0)!=(fb<=0):out.append(a+fa/(fa-fb)*(b-a))
    return np.asarray(out,float).reshape(-1,3)


def cropped_triangles(original,size):
    """Independent no-cap XY crop, retaining originating raw triangle IDs.

    Does not reproduce trimesh vertex merge/order; do not call IDs runtime IDs.
    """
    t=np.asarray(original,float);size=np.asarray(size,float)
    if t.ndim!=3 or t.shape[1:]!=(3,3) or not len(t) or not np.isfinite(t).all() or size.shape!=(2,) or not np.isfinite(size).all() or np.any(size<=0):
        raise ValueError('Finite nonempty triangles and explicit positive crop size required')
    result=[];source=[]
    for tid,triangle in enumerate(t):
        p=triangle.copy()
        for axis in range(2):
            for sign in [1,-1]:p=clip_polygon(p,axis,sign*size[axis]/2,sign)
        for i in range(1,len(p)-1):
            piece=np.array([p[0],p[i],p[i+1]])
            if np.linalg.norm(np.cross(piece[1]-piece[0],piece[2]-piece[0]))<=1e-14:continue
            result.append(piece);source.append(tid)
    if not result:raise ValueError('Empty crop')
    return np.array(result),np.array(source)


class BoxTriangleGuard:
    def __init__(self,triangles,*,tolerance):
        self.triangles=np.array(triangles,float,copy=True)
        if self.triangles.ndim!=3 or self.triangles.shape[1:]!=(3,3) or not len(self.triangles) or not np.isfinite(self.triangles).all():
            raise ValueError('Nonempty finite triangle soup required')
        if not np.isfinite(tolerance) or not 0<tolerance<=1e-6:raise ValueError('Explicit numerical tolerance required')
        if np.any(np.linalg.norm(np.cross(self.triangles[:,1]-self.triangles[:,0],self.triangles[:,2]-self.triangles[:,0]),axis=1)<1e-14):
            raise ValueError('Degenerate terrain unresolved')
        self.tolerance=float(tolerance)
        self.geometry_sha256=hashlib.sha256(self.triangles.astype('<f8').tobytes()).hexdigest()
        self.triangles.setflags(write=False)
        self.lo=self.triangles.min(axis=1);self.hi=self.triangles.max(axis=1)

    def scan(self,poses,halfsize,*,pose_ids):
        """Rebuild broadphase EVERY call; all supplied poses and finite faces.

        Diagnostic score is per-pose maximum negative SAT gap, not a physical
        penetration integral and not used to choose a coordinate hypothesis.
        """
        poses=np.asarray(poses,float);h=np.asarray(halfsize,float)
        if poses.ndim!=3 or poses.shape[1:]!=(4,4) or not len(poses):raise ValueError('Explicit nonempty box poses required')
        if not isinstance(pose_ids,list) or len(pose_ids)!=len(poses) or any(not isinstance(p,str) or not p for p in pose_ids) or len(set(pose_ids))!=len(pose_ids):
            raise ValueError('Explicit unique ordered string pose IDs required')
        if h.shape!=(3,) or not np.isfinite(h).all() or np.any(h<=0):raise ValueError('Positive box halfsize required')
        records=[];query_count=0;score=np.zeros(len(poses));hit_frames=[]
        for f,t in enumerate(poses):
            rigid(t);r=t[:3,:3];c=t[:3,3];extent=abs(r)@h
            ids=np.flatnonzero(np.all(self.hi>=c-extent-self.tolerance,axis=1)&np.all(self.lo<=c+extent+self.tolerance,axis=1))
            local=(self.triangles[ids]-c)@r
            gaps=sat_gaps(local,h);query_count+=len(ids)
            hit=gaps < -self.tolerance
            if hit.any():
                score[f]=float(np.max(-gaps[hit]));hit_frames.append(f)
                for tid,gap in zip(ids[hit],gaps[hit]):records.append(dict(frame=f,triangle_id=int(tid),sat_gap_raw=float(gap)))
        return dict(strict_hit_frames=hit_frames,hit_frame_count=len(hit_frames),strict_pairs=len(records),
                    per_pose_max_negative_sat_raw=score.tolist(),maximum_negative_sat_raw=float(score.max()),
                    fresh_triangle_queries=query_count,records=records,full_surface_scan=True,
                    geometry_sha256=self.geometry_sha256,halfsize=h.tolist(),pose_ids=list(pose_ids),numerical_tolerance=self.tolerance,
                    solid_containment_excluded=False,other_robot_shapes_checked=False,swept_path_checked=False,
                    accepted_reference=False)


def reject_new_or_worse(before,after,*,numerical_slack):
    """Conservative line-search gate, NOT reference-motion acceptance.

    No new hit pose and no increase in each pose's max SAT overlap score.
    Requiring objective decrease remains the caller's separate responsibility.
    The same guard, triangle soup, pose IDs/order and halfsize MUST be used.
    """
    if not np.isfinite(numerical_slack) or not 0<=numerical_slack<=1e-8:raise ValueError('Explicit numerical slack')
    for key in ['geometry_sha256','halfsize','pose_ids','numerical_tolerance']:
        if before[key]!=after[key]:raise ValueError('Changed comparison binding: '+key)
    a=np.asarray(before['per_pose_max_negative_sat_raw'],float);b=np.asarray(after['per_pose_max_negative_sat_raw'],float)
    if a.shape!=b.shape or a.ndim!=1 or not len(a) or not np.isfinite([a,b]).all() or np.any(a<0) or np.any(b<0):raise ValueError('Matching finite nonnegative pose scores required')
    for report,values in [(before,a),(after,b)]:
        if report['full_surface_scan'] is not True or report['strict_hit_frames']!=np.flatnonzero(values>report['numerical_tolerance']).tolist():
            raise ValueError('Incomplete or inconsistent geometry scan')
    new=sorted(set(after['strict_hit_frames'])-set(before['strict_hit_frames']))
    worse=np.flatnonzero(b>a+numerical_slack).tolist()
    return dict(reject=bool(new or worse),new_hit_pose_indices=new,worsened_pose_indices=worse,
                scope='nominal_box_triangle_endpoint_guard_only_NOT_full_collision_or_motion_acceptance')