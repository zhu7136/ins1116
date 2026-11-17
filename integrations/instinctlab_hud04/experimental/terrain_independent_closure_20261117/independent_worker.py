"""Independent full-mesh verification; distinct algorithms, no motion fitting."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np
from terrain_binding import ActualTerrain
from robot_shape_inventory import load_shapes,TargetTasks
from independent_primitives import sphere_distance,box_distance,cylinder_bracket

ROOT=Path(__file__).resolve().parent


def run(gate,payload):
    output=ROOT/'INDEPENDENT_PRIMITIVE_CHECK_20261117.json';journal=ROOT/'independent_progress.jsonl'
    if output.exists() or journal.exists():raise FileExistsError('No overwrite or replay')
    gate.step();terrain=ActualTerrain();tree,shapes,_=load_shapes();target=TargetTasks()
    q=np.array([(target.limits[n]['lower']+target.limits[n]['upper'])/2 for n in target.names]);world={n:p[0] for n,p in tree.fk(q[None],target.names).items()}
    primary_path=ROOT/'ACTUAL_TERRAIN_DETAIL_20261117.json';primary=json.loads(primary_path.read_text());checks=[]
    for shape in shapes:
        if shape.kind=='mesh':continue
        for subset in ['all','lateral']:
            gate.step();triangles=terrain.triangles if subset=='all' else terrain.triangles[terrain.lateral_ids]
            if shape.kind=='cylinder':independent=cylinder_bracket(shape,world[shape.link],triangles)
            else:
                distance=(sphere_distance if shape.kind=='sphere' else box_distance)(shape,world[shape.link],triangles)
                independent=dict(status='converged',lower=distance,upper=distance,triangle_count=len(triangles),gjk_called=False,
                    primary_broadphase_used=False,method='center_triangle_minus_radius' if shape.kind=='sphere' else '13axis_SAT_and_exhaustive_vertex_face_edge_edge')
            ref=next(r['result'] for r in primary['rows'] if r['shape']==shape.identity and r['subset']==subset)
            discrepancy=max(0.,independent['lower']-ref['distance_upper_bound'],ref['distance_lower_bound']-independent['upper'])
            row=dict(shape=shape.identity,kind=shape.kind,subset=subset,independent=independent,
                primary_bracket=[ref['distance_lower_bound'],ref['distance_upper_bound']],brackets_compatible=discrepancy<=1e-8,
                discrepancy=discrepancy,fully_verified=independent['status']=='converged' and discrepancy<=1e-8)
            checks.append(row)
            with journal.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    result=dict(stage='independent_primitive_to_actual_terrain_NOT_reference_acceptance',checks=checks,
        query_count=len(checks),type_counts=dict(Counter(r['kind'] for r in checks)),
        independent_status_counts=dict(Counter(r['independent']['status'] for r in checks)),
        compatible_count=sum(r['brackets_compatible'] for r in checks),fully_verified_count=sum(r['fully_verified'] for r in checks),
        max_bracket_discrepancy=max(r['discrepancy'] for r in checks),primary_sha256=hashlib.sha256(primary_path.read_bytes()).hexdigest(),
        all_triangles_in_each_subset_evaluated=True,primary_broadphase_reused=False,gjk_used=False,
        nominal_hand_hulls_independently_full_mesh_verified=False,cooked_physics_validated=False,
        source_coordinate_hypothesis_selected=False,gpu_submissions=0,retarget_solver_calls=0,ppo_updates=0,
        accepted_reference=False,training_ready=False)
    raw=json.dumps(result,indent=2,allow_nan=False).encode()
    with output.open('xb') as stream:stream.write(raw)
    if output.read_bytes()!=raw:raise RuntimeError('Readback failed')
    return {k:v for k,v in result.items() if k!='checks'}|dict(detail_sha256=hashlib.sha256(raw).hexdigest())