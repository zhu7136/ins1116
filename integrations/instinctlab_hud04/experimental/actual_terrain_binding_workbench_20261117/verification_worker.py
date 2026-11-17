"""Separate numerical formulation for target primitive surface queries."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from terrain_binding import ActualTerrain
from independent_primitives import distance
from robot_shape_inventory import load_shapes,TargetTasks
from binding_worker import persist

ROOT=Path(__file__).resolve().parent


def run(gate,payload):
    output=ROOT/'INDEPENDENT_TARGET_TERRAIN_CHECK_20261117.json'
    if output.exists():raise FileExistsError('Keep existing verification')
    detail=json.loads((ROOT/'TARGET_ACTUAL_TERRAIN_DETAIL_20261117.json').read_text())
    terrain=ActualTerrain();tree,shapes,policy=load_shapes();target=TargetTasks()
    q=np.array([(target.limits[n]['lower']+target.limits[n]['upper'])/2 for n in target.names])
    world={name:p[0] for name,p in tree.fk(q[None],target.names).items()};by_id={s.identity:s for s in shapes}
    records=[];all_membership=[]
    for row in detail['rows']:
        gate.step();shape=by_id[row['shape']];r=row['result'];tf=world[shape.link]@shape.local
        p=(np.array(r['first_witness'])-tf[:3,3])@tf[:3,:3]
        if shape.kind=='sphere':residual=float(np.linalg.norm(p)-shape.parameters['radius'])
        elif shape.kind=='box':residual=float(np.max(np.abs(p)-np.array(shape.parameters['size'])/2))
        elif shape.kind=='cylinder':residual=max(float(np.linalg.norm(p[:2])-shape.parameters['radius']),float(abs(p[2])-shape.parameters['length']/2))
        else:
            eq=ConvexHull(shape.vertices).equations;residual=float(np.max(eq[:,:3]@p+eq[:,3]))
        if residual>1e-8:raise RuntimeError('Target witness not in shape')
        all_membership.append(dict(shape=shape.identity,subset=row['subset'],local_membership_residual=residual))
        if shape.kind not in ['sphere','box']:continue
        triangles=terrain.triangles if row['subset']=='all' else terrain.triangles[terrain.lateral_ids]
        value=distance(shape,world[shape.link],triangles)
        error=abs(value-r['distance_upper_bound'])
        if not r['distance_lower_bound']-1e-8<=value<=r['distance_upper_bound']+1e-8:raise RuntimeError('Independent sphere/OBB full-surface mismatch')
        records.append(dict(shape=shape.identity,kind=shape.kind,subset=row['subset'],independent_distance=value,
            primary_lower=r['distance_lower_bound'],primary_upper=r['distance_upper_bound'],upper_error=error,
            all_subset_triangles_evaluated=len(triangles),gjk_called=False,aabb_candidates_used=False))
    terrain.verify_unchanged()
    result=dict(stage='independent_target_full_triangle_sphere_box_checks_NOT_acceptance',
        independent_global_queries=len(records),sphere_queries=sum(r['kind']=='sphere' for r in records),
        box_queries=sum(r['kind']=='box' for r in records),checks=records,
        all_target_witness_membership_checks=len(all_membership),witness_checks=all_membership,
        maximum_distance_difference=max(r['upper_error'] for r in records),
        max_target_membership_outside_residual=max(max(0.,r['local_membership_residual']) for r in all_membership),
        independent_cylinder_hand_global_distances_verified=False,
        limitations=['Cylinder and hand witnesses checked for membership only, not independently recomputed global distances',
            'Pose is algebraic in-limit midpoint; not retargeted or accepted motion','Surface separation does not exclude terrain solid containment'],
        accepted_reference=False,training_ready=False,gpu_submissions=0,solver_calls=0,ppo_updates=0)
    receipt=persist(output,result)
    return {k:v for k,v in result.items() if k not in ['checks','witness_checks']}|dict(detail=receipt)