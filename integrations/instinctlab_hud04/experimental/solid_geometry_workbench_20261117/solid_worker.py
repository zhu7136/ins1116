"""One bounded algebraic pose, not motion replay or physical collision proof."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from robot_shape_inventory import load_shapes,all_pairs,TargetTasks,TARGET,R,CONFIG
from convex_solid import Hull
from support_distance import convex_distance

ROOT=Path(__file__).resolve().parent


def run(gate,payload):
    output=ROOT/'ALL_SHAPE_PAIR_DETAIL_20261117.json'
    if output.exists():raise FileExistsError('Diagnostic already exists')
    gate.step();tree,shapes,policy=load_shapes();target=TargetTasks()
    lower,upper=[np.array([target.limits[n][k] for n in target.names]) for k in ['lower','upper']]
    q=(lower+upper)/2;world={name:pose[0] for name,pose in tree.fk(q[None],target.names).items()}
    pairs,boxes=all_pairs(tree,shapes,world);by_id={s.identity:s for s in shapes};narrow=Counter();queries=0
    max_bracket=0.;max_witness_error=0.;hand_queries=0
    for index,pair in enumerate(pairs):
        if index%20==0:gate.step()
        if pair['status']!='unresolved_narrowphase_required':continue
        a,b=by_id[pair['first']],by_id[pair['second']]
        result=convex_distance(a,world[a.link],b,world[b.link]);pair['narrowphase']=result
        queries+=1;narrow[result['status']]+=1;hand_queries+=int(a.kind=='mesh' or b.kind=='mesh')
        if result['converged']:max_bracket=max(max_bracket,result['distance_upper_bound']-result['distance_lower_bound'])
        max_witness_error=max(max_witness_error,abs(np.linalg.norm(np.array(result['first_witness'])-result['second_witness'])-result['distance_upper_bound']))
    gate.step();hand_checks=[]
    for shape in shapes:
        if shape.kind!='mesh':continue
        transform=world[shape.link]@shape.local;vertices=shape.vertices@transform[:3,:3].T+transform[:3,3]
        hull=Hull.from_points(vertices);center=vertices.mean(axis=0)
        contained=hull.segment_clearance(center,center)
        if not contained['overlap_or_within_tolerance']:raise RuntimeError('Contained hand point missed')
        lo,hi=vertices.min(axis=0),vertices.max(axis=0)
        checks=[]
        for axis in [0,2]:
            u,v=[i for i in range(3) if i!=axis];corners=[]
            for first,second in [(-1,-1),(1,-1),(1,1),(-1,1)]:
                p=np.zeros(3);p[axis]=hi[axis]+.05;p[u]=lo[u]-1 if first<0 else hi[u]+1;p[v]=lo[v]-1 if second<0 else hi[v]+1;corners.append(p)
            triangles=np.array(corners)[[[0,1,2],[0,2,3]]]
            d=min(hull.triangle_clearance(t)['unsigned_solid_distance'] for t in triangles)
            if abs(d-.05)>1e-10:raise RuntimeError('Actual hand nominal hull sidewall/floor mismatch')
            checks.append(dict(normal_axis=axis,expected_distance=.05,distance=d,error=abs(d-.05)))
        hand_checks.append(dict(link=shape.link,contained_point_detected=True,analytic_planes=checks,
            actual_task_terrain=False,cooked_physics=False))
    R.require_hash(TARGET,R.TARGET_MODEL_SHA256)
    detail=dict(stage='all_pairs_on_one_in_limit_algebraic_pose_NOT_trajectory',importer_policy=policy,
        target_sha256=R.TARGET_MODEL_SHA256,config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        joint_names=target.names,joint_positions=q.tolist(),root_transform=np.eye(4).tolist(),
        shapes=[dict(identity=s.identity,link=s.link,kind=s.kind,parameters=s.parameters,
            collision_origin=s.local.tolist(),world_aabb=[x.tolist() for x in boxes[s.identity]]) for s in shapes],
        pairs=pairs,hand_checks=hand_checks,
        all_pairs_listed=True,no_graph_adjacency_auto_exclusion=True,
        actual_task_terrain_queried=False,actual_collision_filters_observed=False,
        is_motion_or_standing_validation=False,retarget_solver_calls=0,gpu_submissions=0,ppo_updates=0,
        accepted_reference=False,training_ready=False)
    raw=json.dumps(detail,indent=2,allow_nan=False).encode()
    if len(raw)>1500000:raise RuntimeError('Evidence size budget exceeded')
    with output.open('xb') as stream:stream.write(raw)
    if output.read_bytes()!=raw:raise RuntimeError('Evidence readback failed')
    return dict(stage=detail['stage'],shape_count=len(shapes),collision_link_count=len({s.link for s in shapes}),
        shape_type_counts=dict(Counter(s.kind for s in shapes)),all_shape_pairs=len(pairs),
        cross_link_pairs=sum(not p['same_link'] for p in pairs),same_link_records=sum(p['same_link'] for p in pairs),
        link_pair_count=len({tuple(sorted([by_id[p['first']].link,by_id[p['second']].link])) for p in pairs if not p['same_link']}),
        broadphase_status_counts=dict(Counter(p['status'] for p in pairs)),gjk_narrowphase_queries=queries,
        gjk_status_counts=dict(narrow),gjk_queries_involving_hands=hand_queries,
        max_converged_distance_bracket=max_bracket,max_witness_distance_error=max_witness_error,
        hand_checks=hand_checks,detail_sha256=hashlib.sha256(raw).hexdigest(),detail_bytes=len(raw),
        detail_file=output.name,importer_policy=policy,
        overlap_results_are_not_penetration_depth=True,self_collision_policy_accepted=False,
        actual_task_terrain_queried=False,cooked_physics_validated=False,
        retarget_solver_calls=0,gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False)