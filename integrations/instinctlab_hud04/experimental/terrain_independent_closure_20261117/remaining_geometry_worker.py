"""Four hand-hull queries + distinct analytic-plane cylinder refinement.
Independent from GJK; no retargeting or accepted-reference output.
"""
from pathlib import Path
import json
import numpy as np
import trimesh
from terrain_binding import ActualTerrain
from robot_shape_inventory import load_shapes,TargetTasks
from convex_solid import Hull
from independent_primitives import local_triangles,cylinder_bracket

ROOT=Path(__file__).resolve().parent


def hand_bracket(shape,world,triangles,gate):
    t=local_triangles(shape,world,triangles);hull=Hull.from_points(shape.vertices)
    # Own LOCAL-frame box lower bounds, not the primary WORLD-frame candidates.
    gap=np.maximum(0.,np.maximum(t.min(axis=1)-shape.vertices.max(axis=0),shape.vertices.min(axis=0)-t.max(axis=1)))
    lower=np.linalg.norm(gap,axis=1)
    vertex=shape.vertices[0];nearest=trimesh.triangles.closest_point(t,np.broadcast_to(vertex,(len(t),3)))
    upper=float(np.linalg.norm(nearest-vertex,axis=1).min());queries=0
    for i in np.argsort(lower):
        if lower[i]>upper+1e-10:break
        if queries>=128:break
        if queries%8==0:gate.step()
        result=hull.triangle_clearance(t[i]);distance=result['unsigned_solid_distance']
        lower[i]=distance;upper=min(upper,distance);queries+=1
        if upper<=1e-12:break
    lb=min(float(lower.min()),upper)
    return dict(lower=lb,upper=upper,status='converged' if upper-lb<=1e-8 else 'unresolved_triangle_budget',
        triangle_count=len(t),local_hull_triangle_queries=queries,max_queries=128,
        method='local_frame_AABB_proof_then_halfspace_clipping_and_explicit_triangle_surface_features',
        gjk_called=False,primary_broadphase_reused=False,solid_containment_checked_for_hand_hull=True,
        terrain_solid_containment_excluded=False,nominal_hull_not_cooked=True)


def run(gate,payload):
    out=ROOT/'REMAINING_GEOMETRY_CHECK_20261117.json'
    if out.exists():raise FileExistsError('No overwrite')
    gate.step();terrain=ActualTerrain();tree,shapes,_=load_shapes();target=TargetTasks()
    q=np.array([(target.limits[n]['lower']+target.limits[n]['upper'])/2 for n in target.names]);world={n:p[0] for n,p in tree.fk(q[None],target.names).items()}
    primary=json.loads((ROOT/'ACTUAL_TERRAIN_DETAIL_20261117.json').read_text());checks=[]
    for shape in [s for s in shapes if s.kind=='mesh' or s.identity=='base_link#0']:
        for subset in (['all','lateral'] if shape.kind=='mesh' else ['all']):
            gate.step();t=terrain.triangles if subset=='all' else terrain.triangles[terrain.lateral_ids]
            independent=hand_bracket(shape,world[shape.link],t,gate) if shape.kind=='mesh' else cylinder_bracket(shape,world[shape.link],t,include_face_plane_lower=True)
            ref=next(r['result'] for r in primary['rows'] if r['shape']==shape.identity and r['subset']==subset)
            discrepancy=max(0.,independent['lower']-ref['distance_upper_bound'],ref['distance_lower_bound']-independent['upper'])
            checks.append(dict(shape=shape.identity,subset=subset,independent=independent,discrepancy=discrepancy,
                fully_verified=independent['status']=='converged' and discrepancy<=1e-8))
    report=dict(stage='remaining_independent_nominal_geometry_NOT_acceptance',checks=checks,
        analytic_plane_refinement_changes_no_iteration_or_acceptance_tolerance=True,
        prior_unresolved_record_retained=True,independent_verified_count=sum(r['fully_verified'] for r in checks),
        max_discrepancy=max(r['discrepancy'] for r in checks),gpu_submissions=0,retarget_solver_calls=0,ppo_updates=0,
        cooked_physics_validated=False,accepted_reference=False,training_ready=False)
    text=json.dumps(report,indent=2,allow_nan=False)
    with out.open('x') as stream:stream.write(text+'\n')
    if json.loads(out.read_text())!=report:raise RuntimeError('Readback failed')
    return report