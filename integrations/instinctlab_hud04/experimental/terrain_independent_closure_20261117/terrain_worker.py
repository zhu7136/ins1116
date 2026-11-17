"""Actual crop/side-face queries at one algebraic pose; no motion writer."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from terrain_binding import ActualTerrain
from robot_shape_inventory import load_shapes,TargetTasks

ROOT=Path(__file__).resolve().parent


def membership_residual(shape,link_world,point):
    t=link_world@shape.local;p=(np.asarray(point)-t[:3,3])@t[:3,:3]
    if shape.kind=='sphere':return max(0.,float(np.linalg.norm(p))-shape.parameters['radius'])
    if shape.kind=='box':return float(np.maximum(0.,abs(p)-np.array(shape.parameters['size'])/2).max())
    if shape.kind=='cylinder':return max(0.,float(np.linalg.norm(p[:2]))-shape.parameters['radius'],abs(float(p[2]))-shape.parameters['length']/2)
    eq=ConvexHull(shape.vertices).equations
    return max(0.,float(np.max(eq[:,:3]@p+eq[:,3])))


def run(gate,payload):
    output=ROOT/'ACTUAL_TERRAIN_DETAIL_20261117.json';journal=ROOT/'terrain_progress.jsonl'
    if output.exists() or journal.exists():raise FileExistsError('No overwrite or silent replay')
    gate.step();terrain=ActualTerrain();tree,shapes,policy=load_shapes();target=TargetTasks()
    q=np.array([(target.limits[n]['lower']+target.limits[n]['upper'])/2 for n in target.names])
    world={n:p[0] for n,p in tree.fk(q[None],target.names).items()}
    rows=[];transformed=[]
    for shape in shapes:
        for subset in ['all','lateral']:
            gate.step();result=terrain.indices['raw',subset].closest(shape,world[shape.link],max_queries=256)
            original_id=terrain.original_triangle_id(subset,result['best_triangle']);tri=terrain.triangles[original_id]
            witness=np.array(result['second_witness']);near=trimesh.triangles.closest_point(tri[None],witness[None])[0]
            triangle_residual=float(np.linalg.norm(near-witness));shape_residual=membership_residual(shape,world[shape.link],result['first_witness'])
            if triangle_residual>1e-8 or shape_residual>1e-8:raise RuntimeError('Witness geometry membership failed')
            row=dict(shape=shape.identity,link=shape.link,kind=shape.kind,layout='raw',subset=subset,
                result=result,original_face_index=original_id,original_face_normal=terrain.mesh.face_normals[original_id].tolist(),
                independent_witness_triangle_residual=triangle_residual,independent_witness_shape_residual=shape_residual)
            rows.append(row)
            with journal.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    selected=[s for s in shapes if s.kind=='mesh' or (s.kind=='box' and 'ankle' in s.link)]
    for shape in selected:
        for subset in ['all','lateral']:
            gate.step();shifted_world=terrain.transform('function_shifted')@world[shape.link]
            result=terrain.indices['function_shifted',subset].closest(shape,shifted_world,max_queries=256)
            raw=next(r['result'] for r in rows if r['shape']==shape.identity and r['subset']==subset)
            overlap=max(raw['distance_lower_bound'],result['distance_lower_bound'])<=min(raw['distance_upper_bound'],result['distance_upper_bound'])+1e-8
            if not overlap:raise RuntimeError('Common crop-shift equivariance failed')
            transformed.append(dict(shape=shape.identity,subset=subset,raw_bracket=[raw['distance_lower_bound'],raw['distance_upper_bound']],
                shifted_bracket=[result['distance_lower_bound'],result['distance_upper_bound']],compatible=True,
                is_coordinate_hypothesis_selection=False))
    report=dict(stage='actual_matched_terrain_one_algebraic_target_pose_NOT_reference',terrain=terrain.evidence(),
        importer_policy=policy,pose='29 joint-limit midpoints / identity root; NOT a retargeted source frame',
        joint_names=target.names,joint_positions=q.tolist(),root_transform=np.eye(4).tolist(),rows=rows,common_shift_checks=transformed,
        shape_count=len(shapes),surface_queries=len(rows),query_status_counts=dict(Counter(r['result']['status'] for r in rows)),
        max_shape_membership_residual=max(r['independent_witness_shape_residual'] for r in rows),
        max_triangle_membership_residual=max(r['independent_witness_triangle_residual'] for r in rows),
        max_converged_bracket=max(r['result']['distance_upper_bound']-r['result']['distance_lower_bound'] for r in rows if r['result']['status']=='surface_distance_bracket_converged'),
        complete_mesh_solid_containment_excluded=False,full_signed_collision_distance=False,
        contact_support_schedule_accepted=False,cooked_physics_validated=False,
        source_coordinate_hypothesis_selected=False,gpu_submissions=0,retarget_solver_calls=0,ppo_updates=0,
        accepted_reference=False,training_ready=False)
    raw=json.dumps(report,indent=2,allow_nan=False).encode()
    if len(raw)>1000000:raise ValueError('Evidence size budget')
    with output.open('xb') as stream:stream.write(raw)
    if output.read_bytes()!=raw:raise RuntimeError('Readback failed')
    return {k:v for k,v in report.items() if k not in ['rows','joint_names','joint_positions','root_transform','common_shift_checks']}|dict(
        common_shift_checks=len(transformed),detail_file=output.name,detail_sha256=hashlib.sha256(raw).hexdigest())