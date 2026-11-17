"""Bounded nominal convex-solid queries; never motion/PhysX acceptance.

Distances of overlapping solids are ZERO, not penetration depth. Convexification
must be explicitly requested by constructing Hull from vertices. No mesh repair.
"""
from dataclasses import dataclass
import sys
import numpy as np
from scipy.spatial import ConvexHull, QhullError

# Existing reviewed workbench primitives, no training/runtime registration.
sys.path.insert(0, '/workspace/hud04-motion-checkpoint-20261117')
from hand_mesh_surface import triangle_surface_distances
sys.path.insert(0, '/workspace/hud04-runtime-preflight-20261117')
from nominal_collision import segment_triangles

TOL=1e-12


def points(value,minimum=1,maximum=20000):
    p=np.asarray(value,float)
    if p.ndim!=2 or p.shape[1]!=3 or not minimum<=len(p)<=maximum or not np.isfinite(p).all():
        raise ValueError('Bounded finite Nx3 points required')
    return p


@dataclass
class Hull:
    vertices: np.ndarray
    triangles: np.ndarray
    planes: np.ndarray

    @classmethod
    def from_points(cls,value):
        p=points(value,4)
        try:h=ConvexHull(p)
        except QhullError as exc:raise ValueError('Nondegenerate convex solid required') from exc
        if len(h.simplices)>2000:raise ValueError('Hull face budget exceeded')
        eq=h.equations.copy();eq/=np.linalg.norm(eq[:,:3],axis=1)[:,None]
        return cls(p[h.vertices].copy(),p[h.simplices].copy(),eq)

    def contains(self,value):
        p=points(value)
        if len(p)*len(self.planes)>2000000:raise RuntimeError('Point/plane budget exceeded')
        return (p@self.planes[:,:3].T+self.planes[:,3]).max(axis=1)<=TOL

    def clip_segment(self,start,end):
        a,b=points([start,end]);lower,upper=0.,1.
        for plane in self.planes:
            fa=float(plane[:3]@a+plane[3]);slope=float(plane[:3]@(b-a))
            if abs(slope)<=1e-15:
                if fa>TOL:return None
                continue
            crossing=(TOL-fa)/slope
            if slope>0:upper=min(upper,crossing)
            else:lower=max(lower,crossing)
            if lower>upper:return None
        return a+lower*(b-a),a+upper*(b-a)

    def clip_triangle(self,triangle):
        polygon=points(triangle,3,3).copy()
        for plane in self.planes:
            if len(polygon)==0:break
            signed=polygon@plane[:3]+plane[3]-TOL
            output=[]
            for i,start in enumerate(polygon):
                end=polygon[(i+1)%len(polygon)];fa=signed[i];fb=signed[(i+1)%len(polygon)]
                if fa<=0:output.append(start)
                if (fa<=0)!=(fb<=0):output.append(start+(fa/(fa-fb))*(end-start))
            polygon=np.asarray(output,float).reshape(-1,3)
            if len(polygon)>len(self.planes)+3:raise RuntimeError('Polygon budget exceeded')
        return polygon

    def segment_clearance(self,start,end):
        start,end=points([start,end]);hit=self.clip_segment(start,end)
        if hit is not None:
            # An interior clipped witness avoids roundoff at an expanded plane.
            witness=(hit[0]+hit[1])/2
            return report(0.,witness,witness,True,'segment_convex_solid')
        d,a,b=segment_triangles(start,end,self.triangles);i=int(np.argmin(d))
        return report(float(d[i]),a[i],b[i],False,'segment_convex_solid')

    def triangle_clearance(self,triangle):
        triangle=points(triangle,3,3);clipped=self.clip_triangle(triangle)
        if len(clipped):
            witness=clipped.mean(axis=0)
            return report(0.,witness,witness,True,'triangle_convex_solid')
        d,a,b=triangle_surface_distances(triangle,self.triangles);i=int(np.argmin(d))
        return report(float(d[i]),a[i],b[i],False,'triangle_convex_solid')

    def capsule_overlap(self,start,end,radius):
        # Only for explicitly requested CAPSULES; never replace URDF cylinders.
        if isinstance(radius,bool) or not np.isfinite(radius) or radius<=0:raise ValueError('Positive radius required')
        result=self.segment_clearance(start,end)
        result.update(query='explicit_capsule_convex_solid',axis_distance=result['unsigned_solid_distance'],
            overlap_or_within_tolerance=result['unsigned_solid_distance']<=radius+TOL,
            axis_distance_minus_radius=result['unsigned_solid_distance']-radius)
        result['unsigned_solid_distance']=max(0.,result['axis_distance']-radius)
        # Witnesses belong to the AXIS and hull, not capsule surface.
        result['witness_scope']='axis_and_hull_not_capsule_surface'
        return result


def report(distance,first,second,hit,query):
    return dict(query=query,unsigned_solid_distance=distance,
        first_witness=np.asarray(first).tolist(),second_witness=np.asarray(second).tolist(),
        overlap_or_within_tolerance=bool(hit or distance<=TOL),intersection_tolerance=TOL,
        penetration_depth_computed=False,nominal_geometry_only=True,cooked_physics_validated=False,
        accepted_reference=False,training_ready=False)