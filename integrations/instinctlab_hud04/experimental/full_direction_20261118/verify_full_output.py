"""Independent saved-file FK + explicit mapping + direct named XML limits.

No source root/quaternion/world-unit inference. No optimization performed.
"""
import json,sys,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from inputs import HERE,SOURCE,TARGET,MOTION,SEED,sha,load

sys.path.insert(0,'/workspace/hud04-recovery-bf4a057/integrations/instinctlab_hud04')
from kinematic_diagnostics import RobotTree

def directions(tree,q,names,source):
    frames=tree.fk(q,names);basis=frames['pelvis' if source else 'base_link'][:,:3,:3];output=[]
    wrist='wrist_roll_link' if source else 'wrist_yaw_link'
    for side in ('left','right'):
        for a,b in (('hip_pitch_link','knee_link'),('knee_link','ankle_pitch_link'),('shoulder_pitch_link','elbow_link'),('elbow_link',wrist)):
            vector=frames[side+'_'+b][:,:3,3]-frames[side+'_'+a][:,:3,3]
            local=np.einsum('nji,nj->ni',basis,vector)
            output.append(local/np.linalg.norm(local,axis=1)[:,None])
    return np.stack(output,axis=1)

def angles(a,b):return np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=2),np.sum(a*b,axis=2)))
def summaries(x):return dict(mean_degrees=float(x.mean()),p95_degrees=float(np.percentile(x,95)),max_degrees=float(x.max()),
    over25deg_observations=int((x>25).sum()),over25deg_frames=int(np.any(x>25,axis=1).sum()))

def runs(mask):
    intervals=[];start=None
    for i,v in enumerate(np.r_[mask,False]):
        if v and start is None:start=i
        elif not v and start is not None:intervals.append([start,i-1]);start=None
    return intervals

def main():
    folder=HERE/'OVERLAP_V4';report=json.loads((folder/'FULL_DIRECTION_REPORT.json').read_text());path=folder/report['artifact']
    _,_,_,_,bindings=load()
    for key,value in bindings.items():
        if report[key]!=value:raise ValueError('Pinned input binding no longer matches')
    if sha(path)!=report['artifact_sha256']:raise ValueError('Output SHA256 mismatch')
    source=RobotTree.read(SOURCE);target=RobotTree.read(TARGET)
    with np.load(MOTION,allow_pickle=False) as z:sq=z['joint_pos'].copy();sn=z['joint_names'].tolist()
    with np.load(SEED,allow_pickle=False) as z:baseline=z['joint_pos'].copy();names=z['joint_names'].tolist()
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'joint_pos','joint_names','frame_indices','framerate','role'}:raise ValueError('Joint diagnostic only, no world root')
        np.testing.assert_array_equal(z['frame_indices'],np.arange(997))
        if z['joint_pos'].shape!=(997,29) or float(z['framerate'].item())!=90 or str(z['role'].item())!='full_length_joint_diagnostic_NOT_reference_NO_WORLD_ROOT':raise ValueError('997 native diagnostic contract')
        n=z['joint_names'].tolist()
        if len(n)!=29 or len(set(n))!=29 or set(n)!=set(names):raise ValueError('Explicit exact body DOF')
        q=z['joint_pos'][:,[n.index(name) for name in names]]
    desired=directions(source,sq,sn,True);initial=directions(target,baseline,names,False);actual=directions(target,q,names,False)
    before=angles(initial,desired);after=angles(actual,desired);checks=[]
    for label,x in (('initial',before),('final',after)):
        for key,value in summaries(x).items():
            error=abs(value-report[label][key]);checks.append(error)
            if error>1e-10:raise ValueError('Independent FK disagrees with numerical report')
    for name,digest in report['code_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Numerical execution source changed')
    xml=ET.fromstring(TARGET.read_bytes());nodes=xml.findall('joint');limits={j.get('name'):j.find('limit') for j in nodes if j.get('type')!='fixed'}
    if set(limits)!=set(names) or any('head' in name for name in names):raise ValueError('29 body actions only')
    for name in ('head_yaw_joint','head_pitch_joint'):
        node=next(j for j in nodes if j.get('name')==name)
        if node.get('type')!='fixed':raise ValueError('Fixed head gate')
        origin=node.find('origin')
        if origin is not None and np.linalg.norm(np.fromstring(origin.get('rpy','0 0 0'),sep=' '))>1e-12:raise ValueError('Fixed head not zero-angle origin')
    lower,upper,vmax=[np.array([float(limits[n].get(k)) for n in names]) for k in ('lower','upper','velocity')]
    position=np.any((q<lower-1e-6)|(q>upper+1e-6),axis=1);speed=np.abs(np.diff(q,axis=0))*90;overspeed=np.any(speed>vmax+1e-6,axis=1)
    if position.any() or overspeed.any():raise ValueError('Native full-sequence target limit failure')
    if sha(Path(report['archive']['backup_path']))!=sha(path):raise ValueError('Local archive changed')
    if json.loads(Path(report['archive']['manifest_path']).read_text())['sha256']!=sha(path):raise ValueError('Backup manifest mismatch')
    max_difference=0.;same93=[];prior_all=[];now_all=[]
    for first,last in ((0,30),(480,510),(966,996)):
        old=HERE.parent/'hud04-direction-solve-20261118'/'FIRST_WRIST_COMPARISON'/('DIRECTION_WINDOW_'+str(first)+'_'+str(last)+'.npz')
        with np.load(old,allow_pickle=False) as z:old_q=z['joint_pos'][:,[z['joint_names'].tolist().index(n) for n in names]]
        old_angles=angles(directions(target,old_q,names,False),desired[first:last+1]);current=after[first:last+1]
        same93.append(dict(first=first,last=last,prior=summaries(old_angles),now=summaries(current),
            mean_regressed=bool(current.mean()>old_angles.mean()),max_regressed=bool(current.max()>old_angles.max())))
        prior_all.append(old_angles);now_all.append(current)
    # Full trajectory tails, not merely the formerly sampled windows.
    labels=['left_thigh','left_shin','left_upper_arm','left_forearm','right_thigh','right_shin','right_upper_arm','right_forearm']
    counts=np.zeros(997,dtype=int)
    for row in report['history']:counts[row['first']:row['last']+1]+=1
    tails=[]
    for flat in np.argsort(after.reshape(-1))[-20:][::-1]:
        f,k=divmod(int(flat),8);tails.append(dict(frame=f,direction=labels[k],degrees=float(after[f,k]),block_visits=int(counts[f])))
    high=[dict(direction=label,over25deg_observations=int((after[:,i]>25).sum()),runs=runs(after[:,i]>25)) for i,label in enumerate(labels)]
    merit=report['initial_global_objective']
    for row in report['history']:
        if row['lp_status']!=0:raise ValueError('V4 contains a stopped LP')
        if row['accepted']:
            if row['local_objective_after']>=row['local_objective_before']:raise ValueError('Non-descent update accepted')
            merit+=row['local_objective_after']-row['local_objective_before']
    explicit_final=float(np.abs(actual-desired).sum()+.001*np.abs(q-baseline).sum())
    if abs(merit-explicit_final)>1e-7 or abs(merit-report['final_global_objective'])>1e-7:raise ValueError('Global objective bookkeeping does not agree with saved file')
    result=dict(stage='independent_saved_full997_joint_direction_verification',method='legacy RobotTree + separately explicit anatomical pivots + direct named target XML limits',
        report_sha256=sha(folder/'FULL_DIRECTION_REPORT.json'),artifact_sha256=sha(path),frames=997,intervals=996,direction_observations=997*8,
        position_bad_frames=int(position.sum()),overspeed_intervals=int(overspeed.sum()),head_fixed_zero_confirmed=True,
        max_position_violation_rad=float(max(0.,np.max(lower-q),np.max(q-upper))),
        max_velocity_violation_rad_s=float(max(0.,np.max(speed-vmax))),
        initial=summaries(before),final=summaries(after),max_report_statistic_disagreement=float(max(checks)),
        mean_relative_improvement=float(1-after.mean()/before.mean()),all_997_frames_changed=bool(np.all(np.any(np.abs(q-baseline)>1e-9,axis=1))),
        same_93_frame_comparison=same93,same_93_aggregate_prior=summaries(np.concatenate(prior_all)),same_93_aggregate_now=summaries(np.concatenate(now_all)),
        worst_twenty=tails,high_error_intervals_by_direction=high,observed_high_error_threshold_degrees=25,
        threshold_is_diagnostic_not_reference_acceptance=True,monotone_global_objective_recomputed_from_saved_q=True,
        local_backup_readback_verified=True,backup_scope='same sandbox only until separately verified publication',
        all_numeric_code_hashes_match=True,lp_calls_real_data_this_continuation=86,lp_calls_in_successful_v4=84,
        source_inputs_unchanged=True,world_root_optimized=False,terrain_or_self_collision_validated=False,
        accepted_reference=False,training_ready=False,gpu_submissions=0,ppo_updates=0)
    with (HERE/'FULL_INDEPENDENT_VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()