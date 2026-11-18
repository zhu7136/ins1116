"""NEW independent measured arithmetic + one explicit OFFLINE W&B audit run."""
import datetime,hashlib,json,math,os
from pathlib import Path
ROOT=Path(__file__).parent;OUT=ROOT/'MEASURED_TASK138'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):
    with (OUT/n).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def main():
    OUT.mkdir(exist_ok=False)
    fixed=ROOT/'FIXED_OVERLOAD';d=json.loads((fixed/'GPU_RECOVERED_EVIDENCE/contact_report.json').read_text());s=json.loads((fixed/'GPU_RECOVERED_EVIDENCE/supervisor_result.json').read_text());terminal=json.loads((fixed/'GPU_TERMINAL_REVIEW.json').read_text())
    if terminal['status']!='5' or terminal['actual_commit']!=d['source_commit'] or s['source_commit']!=d['source_commit'] or s['controlled_probe_completed'] is not True or s['return_code']!=0 or s['stop_reason'] is not None:raise ValueError('Actual completion and source required')
    checksum={n:sha(fixed/'GPU_RECOVERED_EVIDENCE'/n) for n in ['contact_report.json','supervisor_result.json']}
    if checksum!=terminal['artifacts']:raise ValueError('Transport hash mismatch')
    summary={}
    for k,series in d['samples'].items():
        n=d['counts'][k]
        if len(series)!=8 or len(n)!=8 or any(not math.isfinite(x) or x<0 for x in series) or any(type(x) is not int or not 0<=x<256 for x in n):raise ValueError('Finite actual data')
        summary[k]=dict(force_min=min(series),force_max=max(series),positive_force_steps=sum(x>1e-6 for x in series),minimum_contact_points=min(n),maximum_contact_points=max(n),positive_count_steps=sum(x>0 for x in n))
    error=0.;quat_error=0.
    for row in d['trace']:
        for side,dx in [('left',5.),('right',10.)]:
            for role in ['Parent','Child']:
                r=row['before'][side+'_'+role];a=r['source_xyzw'];b=r['detached_xyzw'];expected=[a[0]+dx,a[1],a[2]+2.,*a[3:]]
                error=max(error,max(abs(x-y) for x,y in zip(b,expected)))
                quat_error=max(quat_error,abs(sum(x*x for x in a[3:])-1),abs(sum(x*x for x in b[3:])-1))
    if error>2e-6 or quat_error>2e-5:raise ValueError('Pose/quaternion check')
    positive=summary['box']['positive_force_steps']==8 and summary['box']['minimum_contact_points']>0
    comparison={side:positive and summary['detached_'+side]['positive_force_steps']==8 and summary['detached_'+side]['minimum_contact_points']>0 and summary['articulated_'+side]['force_max']==0 and summary['articulated_'+side]['maximum_contact_points']==0 for side in ['left','right']}
    if not all(comparison.values()):raise ValueError('Expected context finding not supported by actual data')
    report=dict(stage='independently_recomputed_measured_context_difference_NOT_filter_or_reference_acceptance',task_id=terminal['task_id'],source_commit=d['source_commit'],actual_child_seconds=s['child_wall_seconds'],report_sha256=checksum,channels=summary,maximum_translated_pose_readback_error=error,maximum_quaternion_squared_norm_error=quat_error,
        observed_context_difference=comparison,source_vs_copy_selected_usd_attributes_equal=all(x['original']==x['clone'] for x in d['copied_shapes']),
        child_mesh_descendant_present=all(any(y['is_mesh'] for y in x['descendants']) for x in d['copied_shapes'] if x['role']=='Child'),
        measured_contact_is_body_pair_not_per_shape_id=True,disabled_other_copy_shape_not_independently_read_back_after_reset=True,
        isolated_parent_kinematic_child_dynamic_vs_articulation_is_a_confound=True,exact_cooked_geometry_equivalence_proven=False,articulation_filter_causal_mechanism_proven=False,
        automatic_pair_exclusions=0,new_reference_files=0,motion_solver_calls_this_stage=0,ppo_updates_this_stage=0,accepted_reference=False,training_ready=False,
        redundant_same_commit_task133_disclosed=True,failed_task137_retained=True,online_wandb_verified=False)
    save('INDEPENDENT_MEASURED_REVIEW.json',report)
    import wandb
    config=dict(objective='HUD04 wrist contact measurement audit, NOT motion acceptance',runner='Flux GPU results audited on sandbox CPU',task_ids=['TASK_20261118_133','TASK_20261118_137',terminal['task_id']],source_commit=d['source_commit'],automatic_pair_exclusions=0,frames_original=997,fps_original=90,body_actions=29,head_action_dof=0,online_sync_verified=False)
    provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if os.environ.get(v)}
    if provenance:config['_wb_agent']=provenance
    work=OUT/'wandb-offline';work.mkdir()
    with wandb.init(entity='zhul136-university-of-science-and-technology-of-china',project='ins1116',name='hud04-measured133-137-138-context-audit',job_type='diagnostic-audit',mode='offline',dir=str(work),config=config,settings=wandb.Settings(save_code=False,disable_git=True)) as run:
        table=wandb.Table(columns=['channel','native_force_min','native_force_max','positive_force_steps','contact_points_min','contact_points_max'],data=[[k,v['force_min'],v['force_max'],v['positive_force_steps'],v['minimum_contact_points'],v['maximum_contact_points']] for k,v in summary.items()])
        run.log({'actual_measured_channels':table,'detached_left_peak':summary['detached_left']['force_max'],'detached_right_peak':summary['detached_right']['force_max'],'box_peak':summary['box']['force_max'],'child_seconds':s['child_wall_seconds'],'accepted_reference':False,'training_ready':False})
        rid=run.id
    receipt=dict(run_id=rid,mode='offline',directory=str(work),online_sync_verified=False,project='zhul136-university-of-science-and-technology-of-china/ins1116',new_gpu_submissions=0,new_solver_calls=0)
    save('OFFLINE_WANDB_RECEIPT.json',receipt)
    if json.loads((OUT/'INDEPENDENT_MEASURED_REVIEW.json').read_text())!=report:raise ValueError('Readback')
    print(json.dumps(dict(review=report,offline_record=receipt),indent=2),flush=True)
if __name__=='__main__':main()