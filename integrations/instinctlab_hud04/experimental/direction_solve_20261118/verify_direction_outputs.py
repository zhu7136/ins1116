"""Independent legacy FK, direct named limits and saved-byte rechecks."""
import hashlib,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).parent
sys.path.insert(0,'/workspace/hud04-recovery-bf4a057/integrations/instinctlab_hud04')
from kinematic_diagnostics import RobotTree
import xml.etree.ElementTree as ET

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def local(tree,q,names,kind,endpoint):
    pose=tree.fk(q,names);anchor='pelvis' if kind=='source' else 'base_link'
    wrist=('wrist_roll_link' if kind=='source' else 'wrist_yaw_link') if endpoint=='first_wrist_pivot' else ('wrist_yaw_link' if kind=='source' else 'wrist_roll_link')
    output=[]
    for side in ('left','right'):
        for a,b in (('hip_pitch_link','knee_link'),('knee_link','ankle_pitch_link'),('shoulder_pitch_link','elbow_link'),('elbow_link',wrist)):
            delta=pose[side+'_'+b][:,:3,3]-pose[side+'_'+a][:,:3,3]
            vector=np.einsum('nji,nj->ni',pose[anchor][:,:3,:3],delta)
            output.append(vector/np.linalg.norm(vector,axis=1)[:,None])
    return np.stack(output,axis=1)

def statistics(a,b):
    angles=np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1)))
    return dict(mean_degrees=float(np.mean(angles)),p95_degrees=float(np.percentile(angles,95)),max_degrees=float(np.max(angles)),sum_l1_chordal=float(np.sum(np.abs(a-b)))),angles

def main():
    from run_direction_diagnostic import SOURCE,TARGET,MOTION,SEED
    source=RobotTree.read(SOURCE);target=RobotTree.read(TARGET)
    with np.load(MOTION,allow_pickle=False) as z:sq=z['joint_pos'].copy();sn=z['joint_names'].tolist()
    with np.load(SEED,allow_pickle=False) as z:seed=z['joint_pos'].copy();tn=z['joint_names'].tolist()
    limits={j.get('name'):j.find('limit') for j in ET.fromstring(TARGET.read_bytes()).findall('joint') if j.get('type')!='fixed'}
    lower,upper,vmax=[np.array([float(limits[n].get(key)) for n in tn]) for key in ('lower','upper','velocity')]
    records=[]
    for folder,endpoint in ((HERE,'distal_chain'),(HERE/'FIRST_WRIST_COMPARISON','first_wrist_pivot')):
        combined=seed.copy();all_initial=[];all_final=[];worst=[];max_error=0.
        for first,last in ((0,30),(480,510),(966,996)):
            stem='DIRECTION_WINDOW_'+str(first)+'_'+str(last);report=json.loads((folder/(stem+'.json')).read_text());artifact=folder/(stem+'.npz')
            if sha(artifact)!=report['artifact_sha256']:raise ValueError('Artifact bytes changed')
            with np.load(artifact,allow_pickle=False) as z:
                q=z['joint_pos'][:,[z['joint_names'].tolist().index(n) for n in tn]]
                np.testing.assert_array_equal(z['frame_indices'],np.arange(first,last+1))
                if float(z['framerate'].item())!=90 or set(z.files)!={'joint_pos','joint_names','frame_indices','framerate','role'}:raise ValueError('No world motion allowed')
            desired=local(source,sq[first:last+1],sn,'source',endpoint)
            before,ba=statistics(local(target,seed[first:last+1],tn,'target',endpoint),desired)
            after,aa=statistics(local(target,q,tn,'target',endpoint),desired)
            for kind,computed in (('initial',before),('final',after)):
                for key,value in computed.items():max_error=max(max_error,abs(value-report[kind][key]))
            if max_error>1e-10:raise ValueError('Independent saved-output direction check failed')
            for t,b in zip(*np.where(aa>25)):worst.append(dict(frame=int(first+t),direction=int(b),degrees=float(aa[t,b])))
            combined[first:last+1]=q;all_initial.append(ba);all_final.append(aa)
            backup=Path(report['archive']['backup_path'])
            if sha(backup)!=report['artifact_sha256']:raise ValueError('Backup changed')
        a=np.concatenate(all_initial);b=np.concatenate(all_final)
        pos=np.any((combined<lower-1e-6)|(combined>upper+1e-6),axis=1)
        vel=np.any(np.abs(np.diff(combined,axis=0))*90>vmax+1e-6,axis=1)
        if pos.any() or vel.any():raise ValueError('Independent combined 997-frame hard check')
        records.append(dict(variant=endpoint,independent_method='legacy RobotTree FK + explicit pivot mapping + direct named XML limits',
            unique_frames=93,direction_observations=744,frames_in_combined_hard_check=997,intervals_in_combined_hard_check=996,
            output_position_bad_frames=int(pos.sum()),output_overspeed_intervals=int(vel.sum()),
            mean_degrees_before=float(a.mean()),mean_degrees_after=float(b.mean()),p95_degrees_before=float(np.percentile(a,95)),p95_degrees_after=float(np.percentile(b,95)),
            mean_relative_improvement=float(1-b.mean()/a.mean()),max_report_abs_difference=max_error,
            high_error_over25deg_count=len(worst),high_error_over25deg_frame_ids=sorted({x['frame'] for x in worst}),
            worst_ten=sorted(worst,key=lambda x:-x['degrees'])[:10],unchanged_middle_frames=904,
            combined_sequence_saved=False,all_window_npz_and_local_backups_sha256_verified=True,
            full_body_candidate_generated=False,accepted_reference=False,training_ready=False))
    result=dict(stage='independent_saved_direction_joint_diagnostic_verification',variants=records,
        code_hashes_first_wrist_match=all(sha(HERE/n)==v for n,v in json.loads((HERE/'FIRST_WRIST_COMPARISON/DIRECTION_WINDOW_0_30.json').read_text())['code_sha256'].items()),
        gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False)
    if result['code_hashes_first_wrist_match'] is not True:raise ValueError('Code changed since first-wrist run')
    with (HERE/'DIRECTION_INDEPENDENT_VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()