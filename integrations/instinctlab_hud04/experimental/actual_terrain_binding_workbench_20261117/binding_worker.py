"""Bounded real-mesh software diagnostics, never a source/target reference fit."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from terrain_binding import ActualTerrain,SourceCapsule,SurfaceIndex
from robot_shape_inventory import load_shapes,TargetTasks

ROOT=Path(__file__).resolve().parent


def persist(path,rows):
    raw=json.dumps(rows,indent=2,allow_nan=False).encode()
    if len(raw)>2000000:raise ValueError('Evidence size cap')
    with path.open('xb') as f:f.write(raw)
    if path.read_bytes()!=raw:raise ValueError('Readback mismatch')
    return dict(file=path.name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def run(gate,payload):
    mode=payload['mode']
    if mode not in ['source','target']:raise ValueError('Explicit diagnostic mode')
    output=ROOT/(mode.upper()+'_ACTUAL_TERRAIN_DETAIL_20261117.json')
    journal=ROOT/(mode+'_progress.jsonl')
    if output.exists() or journal.exists():raise FileExistsError('No overwrite/retry')
    gate.step();terrain=ActualTerrain()
    transforms={'raw':np.eye(4),'terrain_function':terrain.function_transform,'single_tile_final':terrain.final_transform}
    layouts=list(transforms) if mode=='source' else ['raw']
    indices={(name,subset):terrain.index(subset=subset,transform=transforms[name]) for name in layouts for subset in ['all','lateral']}
    gate.step();rows=[];bracket_misses=0;membership_error=0.;source_reference_error=0.;translation_comparisons=[]
    if mode=='source':
        inputs=terrain.historical['independent_verification_witnesses']
        analytic={subset:SurfaceIndex(terrain.triangles if subset=='all' else terrain.triangles[terrain.lateral_ids]) for subset in ['all','lateral']}
        fixtures=[(dict(anchor=w['anchor'],order=w['order'],link=w['link'],frame=w['frame'],primitive=w['primitive']),
            SourceCapsule(w['start_raw'],w['end_raw'],w['radius_raw']),np.eye(4),w) for w in inputs]
    else:
        tree,shapes,policy=load_shapes();target=TargetTasks()
        q=np.array([(target.limits[n]['lower']+target.limits[n]['upper'])/2 for n in target.names])
        world={name:pose[0] for name,pose in tree.fk(q[None],target.names).items()}
        fixtures=[(dict(shape=s.identity,link=s.link,kind=s.kind),s,world[s.link],None) for s in shapes]
    with journal.open('x') as partial:
        for identity,shape,pose,historical in fixtures:
            gate.step();group=[]
            for subset in ['all','lateral']:
                expected=None
                if mode=='source':
                    distance,_,_,_=analytic[subset].closest(shape.endpoints[0],shape.endpoints[1])
                    if subset=='all':source_reference_error=max(source_reference_error,abs(distance-historical['centerline_distance_raw']))
                    expected=max(0.,distance-shape.radius)
                for layout in layouts:
                    index,face_ids=indices[layout,subset];placed=transforms[layout]@pose
                    result=index.closest(shape,placed,max_queries=128)
                    triangle=int(face_ids[result['best_triangle']]);surface_point=np.array(result['second_witness'])
                    vertices=terrain.triangles[triangle]@transforms[layout][:3,:3].T+transforms[layout][:3,3]
                    closest=trimesh.triangles.closest_point(vertices[None],surface_point[None])[0]
                    residual=float(np.linalg.norm(closest-surface_point));membership_error=max(membership_error,residual)
                    if residual>1e-8:raise RuntimeError('Returned terrain witness not on its recorded triangle')
                    bracket_ok=expected is None or result['distance_lower_bound']-1e-8<=expected<=result['distance_upper_bound']+1e-8
                    bracket_misses+=int(not bracket_ok)
                    row=dict(**identity,layout=layout,subset=subset,result=result,
                        original_cropped_face_index=triangle,original_face_normal=terrain.normals[triangle].tolist(),
                        triangle_membership_residual=residual,independent_segment_minus_radius_expected=expected,
                        independent_value_inside_query_bracket=bracket_ok,
                        coordinate_hypothesis_selected=False,actual_reference_or_contact_label=False)
                    rows.append(row);group.append(row);partial.write(json.dumps(row,allow_nan=False)+'\n');partial.flush()
                if mode=='source':
                    values=[r for r in group if r['subset']==subset]
                    if all(r['result']['status']=='surface_distance_bracket_converged' for r in values):
                        ds=[r['result']['distance_upper_bound'] for r in values]
                        translation_comparisons.append(max(ds)-min(ds))
            print('completed real-terrain fixture',mode,len(rows),flush=True)
    terrain.verify_unchanged()
    if bracket_misses or source_reference_error>1e-8:raise RuntimeError('Independent real-terrain bracket mismatch')
    # Descriptive result, not an acceptance gate: unresolved distances are retained.
    evidence=dict(stage='actual_processed_terrain_SOFTWARE_binding_NOT_motion_validation',mode=mode,
        terrain=terrain.evidence(),rows=rows,source_reference_max_centerline_error=source_reference_error,
        max_terrain_witness_membership_residual=membership_error,independent_bracket_misses=bracket_misses,
        max_common_translation_distance_spread=max(translation_comparisons,default=None),
        common_translation_converged_groups=len(translation_comparisons),
        all_four_hypotheses_retained=(mode=='source'),source_or_target_pose_accepted=False,
        target_pose_kind='in_limit_joint_midpoints_identity_root_not_source_mapping' if mode=='target' else None,
        queries=dict(Counter(r['result']['status'] for r in rows)),
        source_world_units_and_alignment_proven=False,actual_physx_or_filters_observed=False,
        solid_containment_excluded=False,contact_schedule_generated=False,
        solver_calls=0,gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False)
    receipt=persist(output,evidence)
    return dict(mode=mode,terrain=terrain.evidence(),fixture_count=len(fixtures),query_count=len(rows),
        query_status_counts=evidence['queries'],total_triangle_queries=sum(r['result']['triangle_queries'] for r in rows),
        exhausted_triangle_budgets=sum(r['result']['triangle_budget_exhausted'] for r in rows),
        independent_bracket_misses=bracket_misses,source_reference_max_centerline_error=source_reference_error,
        max_terrain_witness_membership_residual=membership_error,
        common_translation_converged_groups=len(translation_comparisons),
        max_common_translation_distance_spread=evidence['max_common_translation_distance_spread'],detail=receipt,
        target_motion_or_pose_accepted=False,cooked_physics_validated=False,
        candidate_motion_generated=False,accepted_reference=False,training_ready=False)