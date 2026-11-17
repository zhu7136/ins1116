"""Exact-support shape to triangle-SURFACE distance with conservative bounds.

Caller MUST supply explicit terrain-to-world transform; no root/unit guessing.
Open-terrain solid containment is NOT excluded even if surface distance > 0.
"""
import numpy as np
from scipy.spatial import cKDTree
from robot_shape_inventory import Shape,check_rigid
from support_distance import convex_distance


class TerrainShapeIndex:
    def __init__(self,triangles,*,terrain_to_world):
        t=np.asarray(triangles,float);transform=check_rigid(terrain_to_world)
        if t.ndim!=3 or t.shape[1:]!=(3,3) or not 1<=len(t)<=100000 or not np.isfinite(t).all():
            raise ValueError('Explicit bounded finite terrain triangles required')
        self.triangles=t@transform[:3,:3].T+transform[:3,3]
        self.lo=self.triangles.min(axis=1);self.hi=self.triangles.max(axis=1)
        self.vertex_index=cKDTree(self.triangles.reshape(-1,3))

    def closest(self,shape,link_world,*,max_queries=256):
        if type(max_queries) is not int or not 1<=max_queries<=512:raise ValueError('Bounded triangle-query budget required')
        lower_box,upper_box=shape.aabb(link_world)
        gap=np.maximum(0,np.maximum(self.lo-upper_box,lower_box-self.hi));bounds=np.linalg.norm(gap,axis=1)
        support=shape.support(np.vstack([np.eye(3),-np.eye(3)]),link_world)
        distances,ids=self.vertex_index.query(support);seed=int(np.argmin(distances))
        upper=float(distances[seed]);first=support[seed];second=self.triangles.reshape(-1,3)[ids[seed]]
        candidates=np.argsort(bounds);queries=0;minimum_queried_lower=np.inf;next_lower=np.inf;best_triangle=int(ids[seed])//3
        budget_exhausted=False
        for triangle_id in candidates:
            lb=float(bounds[triangle_id])
            if lb>upper+1e-9:next_lower=lb;break
            if queries>=max_queries:budget_exhausted=True;next_lower=lb;break
            triangle=Shape('terrain#'+str(triangle_id),'terrain','mesh',np.eye(4),{},self.triangles[triangle_id])
            result=convex_distance(shape,link_world,triangle,np.eye(4));queries+=1
            minimum_queried_lower=min(minimum_queried_lower,result['distance_lower_bound'])
            if result['distance_upper_bound']<upper:
                upper=result['distance_upper_bound'];first=np.array(result['first_witness']);second=np.array(result['second_witness']);best_triangle=int(triangle_id)
            if upper<=1e-9:next_lower=0.;break
        lower=min(minimum_queried_lower,next_lower,upper)
        converged=upper-lower<=1e-9
        return dict(status='surface_distance_bracket_converged' if converged else 'unresolved_surface_query',
            distance_lower_bound=lower,distance_upper_bound=upper,triangle_queries=queries,
            triangle_budget_exhausted=budget_exhausted,best_triangle=best_triangle,
            first_witness=first.tolist(),second_witness=second.tolist(),
            terrain_solid_containment_excluded=False,contact_or_support_label=False,
            source_root_or_units_inferred=False,cooked_physics_validated=False,
            accepted_reference=False,training_ready=False)