"""All-native-frame nominal collision diagnostic, not acceptance.
Legacy RobotTree FK independently supplies poses. Projection queries give
SEPARATION certificates only. A finite GJK sample must not imply full narrowphase.
"""
import hashlib,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).parent;sys.path[:0]=[str(ROOT),'/workspace/hud04-runtime-preflight-20261117']
from wholebody_terms import WholeBodyTerms
from support_distance import convex_distance
import task_kinematics as TK
TRIAL=ROOT/'WHOLEBODY_TRIAL_007';SHA='76fe75f0fe63b58fd4f4f3e71e535fced0dd0e6f4dc221fab84a46a64da3face'
def write(name,x):
    with (TRIAL/name).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    if (TRIAL/'ALL997_WHOLEBODY_AUDIT.json').exists():raise FileExistsError('Do not replay')
    start=time.monotonic();path=TRIAL/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA:raise ValueError('Candidate changed')
    body=WholeBodyTerms();tree=TK.K.RobotTree.read(TK.TARGET)
    with np.load(path,allow_pickle=False) as z:q=z['joint_pos'];names=z['joint_names'].tolist();roots=TK.K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz')
    local=tree.fk(q,names);world={l:roots@p for l,p in local.items()}
    class LegacyAdapter:
        def fk(self,poses):
            if not np.array_equal(poses,q):raise ValueError('Full997 legacy adapter only')
            return local,None
    gaps,records,counts=body.evaluate(SimpleNamespace(model=LegacyAdapter()),q,roots,False)
    groups=Counter();frames=defaultdict(set);foot=defaultdict(set);sample_groups=defaultdict(list)
    for k,(g,r) in enumerate(zip(gaps,records)):
        first=body.shapes[r['first']]
        category='terrain' if r['kind']=='terrain' else ('self_same_link' if first.link==body.shapes[r['second']].link else 'self_distinct_links')
        if g < -1e-7:
            groups[category]+=1;frames[category].add(r['frame']);sample_groups[category].append(k)
            if category=='terrain' and first.kind=='box' and first.link in ['left_ankle_roll_link','right_ankle_roll_link']:foot[first.link].add(r['frame'])
    independent=json.loads((TRIAL/'FOOT_TERRAIN_INTERIOR_WITNESSES.json').read_text())
    for r in independent:
        if r['frame'] not in foot[r['side']+'_ankle_roll_link']:raise ValueError('Independently certified foot witness absent from allshape SAT negative frame set')
    chosen=[]
    for category,ids in sorted(sample_groups.items()):
        chosen.extend(ids[i] for i in np.unique(np.r_[np.linspace(0,len(ids)-1,min(20,len(ids))).astype(int),np.argsort(gaps[ids])[:4]]))
    selected_foot=[k for k,r in enumerate(records) if r['kind']=='terrain' and body.shapes[r['first']].link in ['left_ankle_roll_link','right_ankle_roll_link'] and gaps[k]<-1e-7]
    chosen.extend(selected_foot[i] for i in np.linspace(0,len(selected_foot)-1,min(12,len(selected_foot))).astype(int))
    chosen=sorted(set(chosen))
    if len(chosen)>84:raise ValueError('Finite narrow query count cap')
    rows=[];timed_out=False
    class Triangle:
        def __init__(self,vertices):self.vertices=vertices
        def support(self,directions,pose):
            d=np.asarray(directions);return self.vertices[np.argmax(self.vertices@d.T,axis=0)]
    for k in chosen:
        if time.monotonic()-start>70:timed_out=True;break
        r=records[k];f=r['frame'];a=body.shapes[r['first']];wa=world[a.link][f]
        if r['kind']=='terrain':b=Triangle(body.tri[r['second']]);wb=np.eye(4)
        else:b=body.shapes[r['second']];wb=world[b.link][f]
        result=convex_distance(a,wa,b,wb,iterations=64,tolerance=1e-9)
        rows.append(dict(record=r,sampled_projection_gap_raw=float(gaps[k]),**result))
    write('REPRESENTATIVE_GJK_QUERY_RECORDS.json',rows)
    review=dict(stage='all997_nominal43shape_newcontact_audit_NOT_acceptance',artifact_sha256=SHA,
        poses_from_independent_legacy_RobotTree=True,projection_code_shared_with_solver=True,
        frames=997,shapes=43,pairs_per_frame=903,all_native_frame_broadphase_complete=True,
        broadphase_counts=counts,negative_projection_records_gt_1e_minus7=dict(groups),
        frames_with_negative_projection_by_group={k:len(v) for k,v in frames.items()},
        negative_projection_NOT_signed_depth_or_full_collision_proof=True,
        nominal_box_terrain_SAT_strict_negative_frames={k:len(v) for k,v in foot.items()},
        independent_saved_foot_witnesses_covered=len(independent),
        bounded_GJK_selection='up to20evenly-spaced+4worst per 3 category +12foot; <=84 total, <=64iterations/query, 70second cooperative deadline',
        GJK_queries=len(rows),GJK_budget_stop=timed_out,
        GJK_counts=dict(Counter(r['record']['kind']+':'+r['status'] for r in rows)),
        GJK_not_signed_penetration_depth=True,all_frame_narrowphase_complete=False,
        adjacency_auto_exclusions=0,cooked_actual_filters_verified=False,
        no_support_labels_inferred=True,source_physical_units_proven=False,
        foot_collision_rejection=True,accepted_reference=False,training_ready=False,
        gpu_submissions=0,ppo_updates=0,elapsed_seconds=time.monotonic()-start)
    write('ALL997_WHOLEBODY_AUDIT.json',review);print(json.dumps(review,indent=2),flush=True)
if __name__=='__main__':main()