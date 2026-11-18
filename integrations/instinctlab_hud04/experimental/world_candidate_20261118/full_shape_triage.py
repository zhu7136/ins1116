"""All997 / all43 exact-support nominal AABB coverage + bounded3D queries.

Broadphase overlaps are UNRESOLVED, not collisions. GJKzero is not penetration
depth. No adjacency exclusion, no unit inference or physical acceptance.
"""
import json,sys,time
from pathlib import Path
from itertools import combinations
from collections import Counter
import numpy as np
sys.path[:0]=['/workspace/hud04-cpu-deps-20261117','/workspace/hud04-solid-geometry-20261117',
              '/workspace/hud04-runtime-preflight-20261117','/workspace/hud04-terrain-binding-20261117']
from robot_shape_inventory import load_shapes,Shape
from terrain_binding import ActualTerrain
import task_kinematics as TK
from support_distance import convex_distance
from run_world_candidate import sha,write

ROOT=Path(__file__).parent;TRIAL=ROOT/'WORLD_TRIAL_005_FOOT_PRIORITY'

def main():
    output=TRIAL/'FULL_SHAPE_TRIAGE.json'
    if output.exists():raise FileExistsError('Do not replay')
    start=time.monotonic();path=TRIAL/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
    if sha(path)!='9353b185d92c21ceeb56a289b926f89d3dc72c5343745fc14e26889bc6d728d4':raise ValueError('Pinned005')
    with np.load(path,allow_pickle=False) as z:
        q=z['joint_pos'];names=z['joint_names'].tolist();roots=TK.K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz')
    tree,shapes,policy=load_shapes();world={l:roots@v for l,v in tree.fk(q,names).items()}
    low=[];high=[]
    for s in shapes:
        transform=world[s.link]@s.local;r=transform[:,:3,:3];c=transform[:,:3,3]
        if s.kind=='sphere':ext=np.ones((997,3))*s.parameters['radius']
        elif s.kind=='box':ext=np.einsum('nij,j->ni',abs(r),np.array(s.parameters['size'])/2)
        elif s.kind=='cylinder':
            axis=r[:,:,2];ext=np.sqrt(np.maximum(0,1-axis**2))*s.parameters['radius']+abs(axis)*s.parameters['length']/2
        elif s.kind=='mesh':
            vertices=np.einsum('nij,kj->nki',r,s.vertices)+c[:,None,:]
            low.append(vertices.min(axis=1));high.append(vertices.max(axis=1));continue
        low.append(c-ext);high.append(c+ext)
    low=np.stack(low,axis=1);high=np.stack(high,axis=1)
    # Independently cross-check vectorized formulas against six supports from
    # preexisting Shape implementation, every43shape at three actual poses.
    error=0.
    for i,s in enumerate(shapes):
        for f in [0,495,996]:
            a,b=s.aabb(world[s.link][f]);error=max(error,float(abs(a-low[f,i]).max()),float(abs(b-high[f,i]).max()))
    if error>1e-10:raise ValueError('AABB exact-support disagreement')
    adjacency={frozenset([j['parent'],j['child']]) for j in tree.joints};pairs=[];self_jobs=[]
    for a,b in combinations(range(43),2):
        same=shapes[a].link==shapes[b].link;adj=frozenset([shapes[a].link,shapes[b].link]) in adjacency
        overlap=np.all(high[:,a]>=low[:,b]-1e-12,axis=1)&np.all(high[:,b]>=low[:,a]-1e-12,axis=1)
        ids=np.flatnonzero(overlap);row=dict(first=shapes[a].identity,second=shapes[b].identity,same_link=same,directly_joint_connected=adj,
            overlap_aabb_frames=len(ids),overlap_frame_indices=ids.tolist(),automatically_excluded=False)
        pairs.append(row)
        if len(ids) and not same:
            extent=np.maximum(0,np.minimum(high[ids,a],high[ids,b])-np.maximum(low[ids,a],low[ids,b]));f=int(ids[np.argmax(np.prod(extent,axis=1))])
            self_jobs.append((adj,a,b,f))
    terrain=ActualTerrain();tl=terrain.triangles.min(axis=1);th=terrain.triangles.max(axis=1)
    env_rows=[];env_jobs=[]
    for s in range(43):
        counts=np.zeros(997,int);representatives=[]
        for first in range(0,997,64):
            last=min(997,first+64)
            overlap=np.all(th[None]>=low[first:last,s,None,:]-1e-12,axis=2)&np.all(tl[None]<=high[first:last,s,None,:]+1e-12,axis=2)
            counts[first:last]=overlap.sum(axis=1)
            for local,frame in enumerate(range(first,last)):
                ids=np.flatnonzero(overlap[local])
                if len(ids):representatives.append((frame,ids[:4].tolist()))
        env_rows.append(dict(shape=shapes[s].identity,kind=shapes[s].kind,frames_with_triangle_aabb_overlap=int(np.count_nonzero(counts)),candidate_frame_triangle_pairs=int(counts.sum())))
        # One real representative triangle/pose per shape, even for bodies
        # not included in the optimizer's foot term. No cherry-picked min gap.
        if representatives:
            f,ids=representatives[len(representatives)//2]
            env_jobs.extend((s,f,tid) for tid in ids)
    print(json.dumps(dict(stage='all997x43nominal_AABBs_complete',pairs=len(pairs),self_query_jobs=len(self_jobs),terrain_query_jobs=len(env_jobs))),flush=True)
    queries=[]
    for adj,a,b,f in sorted(self_jobs)[:64]:
        if time.monotonic()-start>70:break
        result=convex_distance(shapes[a],world[shapes[a].link][f],shapes[b],world[shapes[b].link][f])
        queries.append(dict(kind='self',first=shapes[a].identity,second=shapes[b].identity,frame=f,directly_joint_connected=adj,**result))
    for s,f,tid in env_jobs[:128]:
        if time.monotonic()-start>70:break
        triangle=Shape('terrain#'+str(tid),'terrain','mesh',np.eye(4),{},terrain.triangles[tid])
        result=convex_distance(shapes[s],world[shapes[s].link][f],triangle,np.eye(4))
        queries.append(dict(kind='terrain',first=shapes[s].identity,triangle_id=tid,frame=f,**result))
    review=dict(stage='FULL997_ALL43_NOMINAL_BROADPHASE_plus_BOUNDED_REPRESENTATIVE_3D_QUERIES_NOT_acceptance',
        artifact_sha256=sha(path),frames=997,shapes=43,links=19,shape_pairs=903,aabb_exact_support_crosscheck129_max_error=error,
        pair_records=pairs,terrain_broadphase=env_rows,query_results=queries,
        query_status_counts=dict(Counter((r['kind']+':'+r['status']) for r in queries)),
        importer_policy=policy,all_frames_and_shapes_broadphase_complete=True,narrowphase_all_frames_complete=False,
        adjacency_auto_exclusions=0,penetration_depth_computed=False,open_terrain_solid_containment_excluded=False,
        nominal_zero_distance_is_not_a_support_label=True,cooked_collision_validated=False,accepted_reference=False,training_ready=False,
        elapsed_seconds=time.monotonic()-start)
    write(output,review)
    print(json.dumps({k:review[k] for k in ['frames','shapes','shape_pairs','aabb_exact_support_crosscheck129_max_error','query_status_counts','narrowphase_all_frames_complete','elapsed_seconds','accepted_reference']},indent=2),flush=True)
if __name__=='__main__':main()