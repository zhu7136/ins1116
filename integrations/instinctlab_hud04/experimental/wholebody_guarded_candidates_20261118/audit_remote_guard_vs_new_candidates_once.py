"""Read-only cross-check using concurrent remote guard and independent crop.
Does not run any solver or modify original/derived candidates.
"""
import hashlib,json,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).parent;REMOTE=ROOT/'REMOTE_ADVANCE_REVIEW_86f05/nonlinear_guard_20261118'
sys.path[:0]=[str(REMOTE),'/workspace/hud04-runtime-preflight-20261117']
from box_triangle_guard import BoxTriangleGuard,cropped_triangles,reject_new_or_worse
import task_kinematics as TK
TERRAIN=Path('/workspace/hud04-original-drive-20261117/50cmboxes_19_36_49.stl')
CASES=[('seed005',Path('/workspace/hud04-world-candidate-20261118/WORLD_TRIAL_005_FOOT_PRIORITY'),'9353b185d92c21ceeb56a289b926f89d3dc72c5343745fc14e26889bc6d728d4'),
       ('trial006',ROOT/'WHOLEBODY_TRIAL_006','ac20ec8ff3406c00d3762f4ffeeee5014ca28023850d6bf2f6f33df8ec8563fd'),
       ('trial007',ROOT/'WHOLEBODY_TRIAL_007','76fe75f0fe63b58fd4f4f3e71e535fced0dd0e6f4dc221fab84a46a64da3face')]
OUTPUT_NAME='REMOTE_GUARD_NEW_CANDIDATE_RECONCILIATION.json'
AUDITED_TRIAL='WHOLEBODY_TRIAL_007'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    output=ROOT/OUTPUT_NAME
    if output.exists():raise FileExistsError('No overwrite/replay')
    start=time.monotonic()
    if sha(TERRAIN)!='3aee33efea6793974d901dd0f7e46859e2d12d2dfd79f38f12c777686566c2de':raise ValueError('Original terrain changed')
    raw=TK.K.read_triangles(TERRAIN);tri,source=cropped_triangles(raw,[9,12]);guard=BoxTriangleGuard(tri,tolerance=1e-9)
    xml=ET.parse(TK.TARGET).getroot();tree=TK.K.RobotTree.read(TK.TARGET);scans={};reviews=[];binding={}
    for name,folder,pin in CASES:
        path=folder/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
        if sha(path)!=pin:raise ValueError('Pinned candidate changed')
        with np.load(path,allow_pickle=False) as z:
            q=z['joint_pos'];names=z['joint_names'].tolist();root=TK.K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz')
            if q.shape!=(997,29) or z['framerate'].item()!=90 or z['accepted_reference'].item()!=False:raise ValueError('Native unaccepted contract')
        world={k:root@p for k,p in tree.fk(q,names).items()};scans[name]={};sides={}
        prior=json.loads((folder/'INDEPENDENT_WORLD_CANDIDATE_REVIEW.json').read_text())
        for side in ['left','right']:
            link=side+'_ankle_roll_link';node=next(n for n in xml.findall('link') if n.get('name')==link)
            boxes=[c for c in node.findall('collision') if c.find('geometry/box') is not None]
            if len(boxes)!=1:raise ValueError('Unique nominal footbox')
            box=boxes[0];half=np.fromstring(box.find('geometry/box').get('size'),sep=' ')/2;pose=world[link]@TK.K.origin_transform(box.find('origin'))
            result=guard.scan(pose,half,pose_ids=[side+':'+str(i) for i in range(997)])
            if result['hit_frame_count']!=prior['foot_box_surface_strict_interior_hit_frames'][side]:raise ValueError('Independent crop/SAT vs saved polygon witness disagreement')
            scans[name][side]=result
            sides[side]={k:v for k,v in result.items() if k!='records'}
        reviews.append(dict(trial=name,artifact_sha256=pin,sides=sides));binding[str(path)]=pin
    comparison={}
    for name in [r[0] for r in CASES if r[0]!='seed005']:
        comparison[name]={side:reject_new_or_worse(scans['seed005'][side],scans[name][side],numerical_slack=1e-12) for side in ['left','right']}
    steps=[json.loads(s) for s in (ROOT/AUDITED_TRIAL/'PROGRESS.jsonl').read_text().splitlines()];accepted=[s for s in steps if s['accepted']]
    for step in accepted:
        r=step['backtracking'][-1]
        if not r['foot_guard_passed'] or r['foot_guard_detail']['new_foot_collision_frames']!=0:raise ValueError('Trial007 logged accepted step violates its declared aggregate guard')
        if AUDITED_TRIAL=='WHOLEBODY_TRIAL_008' and not r['per_pose_guard_passed']:raise ValueError('Trial008 accepted step violated per-pose guard')
    review=dict(stage='Concurrent86f05_guard_independent_numpy_crop_vs_new006007_saved_candidates',remote_head='86f05dffa8696465147d7e3ebf99aff568a84c26',
        remote_guard_sha256=sha(REMOTE/'box_triangle_guard.py'),original_terrain_sha256=sha(TERRAIN),inputs_sha256=binding,
        crop=dict(original_triangles=len(raw),cropped_triangles=len(tri),raw_coordinates_no_alignment_inferred=True),trials=reviews,
        stricter_remote_per_pose_guard_vs_seed005=comparison,
        audited_trial=AUDITED_TRIAL,aggregate_guard_accepted_steps_audited=len(accepted),
        difference='Trial007 guarded no new hit frame and aggregate negative projection sum. Remote86f05 additionally rejects ANY per-pose maximum SAT overlap increase; not equivalent. Saved007 is NOT asserted to pass that stricter guard.',
        trial008_if_present_adds_pinned_per_pose_guard=True,
        optimizer_calls=0,new_motion_files=0,gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False,elapsed_seconds=time.monotonic()-start)
    with output.open('x') as f:json.dump(review,f,indent=2,allow_nan=False)
    print(json.dumps(dict(hits={r['trial']:{s:x['hit_frame_count'] for s,x in r['sides'].items()} for r in reviews},
        stricter_guard={n:{s:dict(reject=x['reject'],new_hits=len(x['new_hit_pose_indices']),worse_poses=len(x['worsened_pose_indices'])) for s,x in sides.items()} for n,sides in comparison.items()},
        accepted_steps_audited=len(accepted),accepted_reference=False,elapsed_seconds=review['elapsed_seconds']),indent=2))
if __name__=='__main__':main()