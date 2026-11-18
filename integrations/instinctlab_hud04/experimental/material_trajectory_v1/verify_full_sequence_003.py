"""Independent FK/XML/reader-byte audit of saved003, no solver invocation."""
import json
import xml.etree.ElementTree as ET
import numpy as np
from material_problem import HERE, CONTRACT, K, sha, problem


def main():
    out=HERE/'TRIAL_MATERIAL_003_FULL_SEQUENCE'
    intent=json.loads((out/'INTENT.json').read_text());report=json.loads((out/'RESULT.json').read_text())
    full=out/'FULL997_ENGINEERING_CANDIDATE_UNACCEPTED.npz';reader=out/'STEP1_READER_LAYOUT_UNACCEPTED.npz'
    assert sha(full)==report['candidate_sha256']
    with np.load(full,allow_pickle=False) as z:
        q,r,names=z['joint_pos'],z['root_transforms_w'],z['joint_names'].tolist()
        assert q.shape==(997,29) and r.shape==(997,4,4)
        assert z['frame_indices'].tolist()==list(range(997)) and float(z['framerate'])==90
        assert not bool(z['accepted_reference']) and not bool(z['training_ready'])
    with np.load(reader,allow_pickle=False) as z:
        assert z['joint_names'].tolist()==names and z['frame_indices'].tolist()==list(range(997))
        assert float(z['framerate'])==90 and not bool(z['accepted_reference']) and not bool(z['training_ready'])
        np.testing.assert_array_equal(z['joint_pos'],q)
        rr=K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz')
        root_difference=float(abs(rr-r).max());assert root_difference<2e-14
        continuity=float(np.sum(z['base_quat_w'][1:]*z['base_quat_w'][:-1],axis=1).min());assert continuity>=0
    c=json.loads(CONTRACT.read_text());target=next(p for p in c['input_sha256'] if 'fixed_head' in p)
    tree=K.RobotTree.read(target);local=tree.fk(q,names)
    materials=np.stack([rr@local[row['target_link']]@np.array(row['target_T_link_material']) for row in c['materials']],axis=1)
    numeric=np.c_[materials[:,:,:3,3].reshape(997,18),materials[:,:,:3,2].reshape(997,18)]
    m,_,_,want,_,support,_,_,_=problem();difference=float(abs(numeric-m.evaluate(q,r)).max());assert difference<1e-12
    xml=ET.parse(target).getroot();joints={j.get('name'):j for j in xml.findall('joint') if j.get('type')!='fixed'}
    assert len(joints)==29 and set(names)==set(joints)
    for name in ('head_yaw_joint','head_pitch_joint'):
        j=next(j for j in xml.findall('joint') if j.get('name')==name);assert j.get('type')=='fixed'
    limits={k:np.array([float(joints[n].find('limit').get(k)) for n in names]) for k in ('lower','upper','velocity','effort')}
    pos=float(max(0.,(limits['lower']-q).max(),(q-limits['upper']).max()))
    vel=float(max(0.,(abs(np.diff(q,axis=0))*90-limits['velocity']).max()))
    assert max(pos,vel)<1e-6
    error=np.linalg.norm((numeric[:,:18]-want[:,:18]).reshape(997,6,3),axis=2)
    assert abs(error.mean()-report['final']['material_position_mean_m'])<1e-12
    assert abs(error[support].mean()-report['final']['proposed_support_position_mean_m'])<1e-12
    windows=[json.loads((out/f'WINDOW_{i:02d}/RESULT.json').read_text()) for i in range(17)]
    sups=[json.loads((out/f'WINDOW_{i:02d}/SUPERVISION.json').read_text()) for i in range(17)]
    assert all(s['child_reaped'] and s['child_exit']==0 and not s['timed_out'] for s in sups)
    covered=set(f for a,b in intent['windows'] for f in range(a,b+1));assert covered==set(range(997))
    assert sum(x['lp_calls'] for x in windows)==17 and sum(x['accepted_update'] for x in windows)==17
    for index,row in enumerate(windows):
        assert sha(out/f'WINDOW_{index:02d}/CANDIDATE_UNACCEPTED.npz')==row['candidate_sha256']
        passed=[x for x in row['backtracking'] if x.get('endpoint_guard',{}).get('passed')]
        assert len(passed)==1 and passed[0]['improved']
    audit=json.loads((out/'AUDIT_SUPERVISION.json').read_text());assert audit['child_reaped'] and audit['child_exit']==0
    self_rows=report['self_strict_by_pair']['final']
    invariant=[x for x in self_rows if 'wrist_pitch_link#1' in x['first'] and 'wrist_roll_link#0' in x['second']]
    invariant_count=sum(x['frame_count'] for x in invariant)
    total_self=sum(x['frame_count'] for x in self_rows)
    cert=report['full_endpoint_guard'];strict=cert['new_strict_pairs']
    final=dict(independent_reader_root_max_error=root_difference,independent_material_fk_max_error=difference,
               quaternion_sign_continuity_min_dot=continuity,exact_997_90hz_29body_0head_verified=True,
               original_XML_joint_position_violation_rad=pos,original_XML_joint_velocity_violation_rad_s=vel,
               original_XML_effort_limits_present=bool(np.all(limits['effort']>0)),effort_feasibility_checked=False,
               windows_completed=17,lp_calls=17,all18_direct_children_reaped=True,
               self_strict_pair_frame_records=total_self,invariant_wrist_strict_pair_frame_records=invariant_count,
               other_self_strict_pair_frame_records=total_self-invariant_count,
               terrain_strict_pair_triangle_frame_records=strict-total_self,
               unresolved_pair_frame_records=cert['new_unresolved_pairs'],newly_strict_pairs=cert['newly_strict_pairs'],
               lost_separation_pairs=cert['lost_certified_separation_pairs'],
               per_window_endpoint_scans_and_full997_two_endpoint_scan_completed=True,
               joint_acceleration_increased=report['final']['joint_acceleration_max_rad_s2']>report['initial']['joint_acceleration_max_rad_s2'],
               support_unassigned_frames=int((~support.any(axis=1)).sum()),
               initial_and_final_contact_or_flight_not_inferred=True,
               runtime_filtered_pair_acceptance_approved=False,automatic_pair_exclusions=0,
               accepted_reference=False,training_ready=False,gpu_submissions=0,ppo_updates=0,
               files_sha256={p.name:sha(p) for p in [full,reader,out/'RESULT.json',out/'INTENT.json']})
    with (out/'INDEPENDENT_VERIFICATION.json').open('x') as f:json.dump(final,f,indent=2,allow_nan=False)
    import wandb
    run=wandb.Api(timeout=45).run('zhul136-university-of-science-and-technology-of-china/ins1116/'+intent['wandb_run_id'])
    summary=dict(run.summary)
    verified={k:summary.get(k) for k in ('accepted_reference','training_ready','computation_complete','covered_frames','lp_calls','accepted_updates','strict_pair_frame_records','unresolved_pair_frame_records','gpu_submissions','ppo_updates')}
    assert run.state=='finished' and verified==dict(accepted_reference=False,training_ready=False,computation_complete=True,covered_frames=997,lp_calls=17,accepted_updates=17,strict_pair_frame_records=strict,unresolved_pair_frame_records=cert['new_unresolved_pairs'],gpu_submissions=0,ppo_updates=0)
    with (out/'WANDB_SERVER_READBACK.json').open('x') as f:json.dump(dict(id=run.id,state=run.state,summary=verified),f,indent=2)
    print(json.dumps(final,indent=2),flush=True)


if __name__=='__main__':main()