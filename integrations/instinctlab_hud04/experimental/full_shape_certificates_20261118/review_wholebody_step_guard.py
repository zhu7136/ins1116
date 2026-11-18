"""Verify NEW in-solver guard against saved certificates, no motion solve."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from audit_self_pairs import HERE, REPO, sha
from wholebody_step_guard import WholeBodyStepGuard, EndpointState, compare_states

MOTIONS={
    '005':('world_candidate_20261118/WORLD_TRIAL_005_FOOT_PRIORITY/STEP1_WORLD_CANDIDATE_UNACCEPTED.npz','9353b185d92c21ceeb56a289b926f89d3dc72c5343745fc14e26889bc6d728d4'),
    '008':('wholebody_guarded_candidates_20261118/WHOLEBODY_TRIAL_008/STEP1_WORLD_CANDIDATE_UNACCEPTED.npz','d5cdb5382b5eb910b3273adab662de286e0452376ee9d134124d4ceddd2f30cf')}

def cached_state(guard, self_path, terrain_path):
    own=json.loads(self_path.read_text());terrain=json.loads(terrain_path.read_text())
    for report in [own,terrain]:
        for text,digest in report['input_sha256'].items():
            p=Path(text);p=p if p.is_absolute() else REPO/p
            if sha(p)!=digest:raise ValueError('Historical certificate input changed: '+str(p))
        for name in ['nominal_certificates.py','audit_self_pairs.py']:
            if sha(HERE/name)!=report['code_sha256'][name]:raise ValueError('Certificate classification code changed')
    if own['frames']!=997 or len(own['pairs'])!=903 or len(terrain['per_shape'])!=43:
        raise ValueError('Incomplete historical certificates')
    identities={s.identity:i for i,(_,s,_) in enumerate(guard.inventory)}
    strict=set();unknown=set()
    for row in own['pairs']:
        if row['same_rigid_link']:continue
        a,b=identities[row['first']],identities[row['second']]
        strict.update(('self',int(r['frame']),a,b) for r in row['strict_common_point_certificates'])
        unknown.update(('self',int(f),a,b) for f in row['unresolved_frames'])
    for row in terrain['per_shape']:
        a=identities[row['shape']]
        strict.update(('terrain',int(r['frame']),a,int(r['cropped_triangle_id'])) for r in row['strict_common_point_certificates'])
        unknown.update(('terrain',int(f),a,int(t)) for f,t in row['unresolved_frame_triangle_pairs'])
    return EndpointState(guard.binding,tuple(range(997)),frozenset(strict),frozenset(unknown),997*903,997*43*8068)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
    p.add_argument('--terrain',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not args.run:
        print(json.dumps(dict(plan_only=True,motion_solver_calls=0,gpu_submissions=0)));return
    if args.output.exists():raise FileExistsError('Refuse overwrite or repeated review receipt')
    start=time.monotonic();path=HERE.parent/MOTIONS['008'][0]
    guard=WholeBodyStepGuard(motion=path,motion_sha256=MOTIONS['008'][1],terrain=args.terrain)
    old=cached_state(guard,HERE/'FINAL_SELF_PAIR_CERTIFICATES.json',HERE/'FINAL_TERRAIN_CERTIFICATES.json')
    expected=cached_state(guard,HERE/'TRIAL008_SELF_PAIR_CERTIFICATES.json',HERE/'TRIAL008_TERRAIN_CERTIFICATES.json')
    with np.load(path,allow_pickle=False) as z:
        q=z['joint_pos'].copy();roots=guard.K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz')
        model=SimpleNamespace(names=z['joint_names'].tolist())
    current=guard.scan(model,q,roots,np.arange(997))
    if current!=expected:raise AssertionError('New guard differs from saved008 certificate classifications')
    allowed,detail=compare_states(old,current)
    if allowed:raise AssertionError('Known nonregressing-foot but wholebody-regressing005->008 transition must reject')
    shapes=[s.identity for _,s,_ in guard.inventory]
    examples=[]
    for k in detail['rejected_examples']:
        kind,frame,a,b=k
        examples.append(dict(kind=kind,frame=frame,first=shapes[a],second=shapes[b] if kind=='self' else b,
            current_class='strict' if tuple(k) in current.strict else 'unresolved'))
    result=dict(stage='new_endpoint_guard_real997_classification_parity_and_historical_step_rejection',
        endpoint_guard_binding=guard.binding,all997_all43_all903_parity=True,
        current_strict_self=sum(k[0]=='self' for k in current.strict),
        current_strict_terrain=sum(k[0]=='terrain' for k in current.strict),
        current_unresolved_self=sum(k[0]=='self' for k in current.unresolved),
        current_unresolved_terrain=sum(k[0]=='terrain' for k in current.unresolved),
        historical005_to008_would_be_allowed=allowed,comparison=detail,shape_examples=examples,
        inputs_sha256={str(path.relative_to(REPO)):MOTIONS['008'][1],
            str((HERE.parent/MOTIONS['005'][0]).relative_to(REPO)):MOTIONS['005'][1]},
        code_sha256={n:sha(HERE/n) for n in ['wholebody_step_guard.py','nominal_certificates.py','review_wholebody_step_guard.py']},
        motion_optimizer_calls=0,new_motion_files=0,gpu_submissions=0,ppo_updates=0,
        runtime_pair_filters_verified=False,material_support_calibrated=False,
        accepted_reference=False,training_ready=False,elapsed_seconds=time.monotonic()-start)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()