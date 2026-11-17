import json
from pathlib import Path
import numpy as np
from independent_cylinder import cylinder_surface_bracket
from terrain_binding import ActualTerrain
from robot_shape_inventory import load_shapes,TargetTasks
from binding_worker import persist

ROOT=Path(__file__).resolve().parent


def run(gate,payload):
    refined=payload.get('refined',False)
    if type(refined) is not bool:raise ValueError('Explicit diagnostic version required')
    if refined:
        from independent_cylinder_refined import refined_bracket
    suffix='_REFINED' if refined else ''
    if payload.get('corrected_local_assertion') is True:
        if not refined:raise ValueError('Refined diagnostic required')
        suffix+='_V2'
    output=ROOT/('INDEPENDENT_CYLINDER_TERRAIN_CHECK'+suffix+'_20261117.json')
    journal=ROOT/('cylinder_verification'+suffix+'_progress.jsonl')
    if output.exists() or journal.exists():raise FileExistsError('No overwrite/retry')
    terrain=ActualTerrain();tree,shapes,_=load_shapes();target=TargetTasks();by_id={s.identity:s for s in shapes}
    q=np.array([(target.limits[n]['lower']+target.limits[n]['upper'])/2 for n in target.names])
    world={name:p[0] for name,p in tree.fk(q[None],target.names).items()}
    detail=json.loads((ROOT/'TARGET_ACTUAL_TERRAIN_DETAIL_20261117.json').read_text());checks=[]
    with journal.open('x') as partial:
        for row in detail['rows']:
            shape=by_id[row['shape']]
            if shape.kind!='cylinder':continue
            gate.step();triangles=terrain.triangles if row['subset']=='all' else terrain.triangles[terrain.lateral_ids]
            independent=refined_bracket(shape,world[shape.link],triangles) if refined else cylinder_surface_bracket(shape,world[shape.link],triangles)
            primary=row['result']
            agrees=max(independent['lower'],primary['distance_lower_bound'])<=min(independent['upper'],primary['distance_upper_bound'])+1e-8
            if not agrees:raise RuntimeError('Independent cylinder/primary brackets incompatible')
            record=dict(shape=shape.identity,subset=row['subset'],independent=independent,
                primary_lower=primary['distance_lower_bound'],primary_upper=primary['distance_upper_bound'],brackets_compatible=True)
            checks.append(record);partial.write(json.dumps(record)+'\n');partial.flush()
    terrain.verify_unchanged()
    result=dict(stage='independent_finite_cylinder_actual_terrain_brackets_NOT_acceptance',checks=checks,
        bounded_refinement_enabled=refined,
        queries=len(checks),converged=sum(x['independent']['status']=='converged' for x in checks),
        unresolved=sum(x['independent']['status']!='converged' for x in checks),
        maximum_bracket_width=max(x['independent']['upper']-x['independent']['lower'] for x in checks),
        maximum_upper_difference=max(abs(x['independent']['upper']-x['primary_upper']) for x in checks),
        gjk_called=False,primary_aabb_candidates_used=False,all_subset_triangles_evaluated=True,
        target_pose_is_only_algebraic_fixture=True,physics_or_motion_validated=False,
        accepted_reference=False,training_ready=False,gpu_submissions=0,solver_calls=0,ppo_updates=0)
    receipt=persist(output,result)
    return {k:v for k,v in result.items() if k!='checks'}|dict(detail=receipt)