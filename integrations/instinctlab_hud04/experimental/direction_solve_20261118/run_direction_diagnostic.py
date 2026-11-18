"""One-shot CPU diagnostic; never submits a GPU task or exports full motion."""
import datetime,hashlib,json,os,sys
from pathlib import Path

CODE=Path(__file__).parent
VARIANT=os.environ.get('HUD04_DIRECTION_DIAGNOSTIC_VARIANT','distal_chain')
if VARIANT not in ('distal_chain','first_wrist_pivot'):raise ValueError('Unknown diagnostic variant')
ROOT=CODE if VARIANT=='distal_chain' else CODE/'FIRST_WRIST_COMPARISON'
PRIOR=Path('/workspace/hud04-window-system-20261117')
SOURCE=Path('/workspace/hud04-recovery-fd12fe1/upstream/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf')
TARGET=Path('/workspace/hud04-recovery-bf4a057/flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf')
MOTION=Path('/workspace/hud04-original-drive-20261117/kneelClimbStep1-x-0.1-ziwen-retargeted.npz')
SEED=Path('/workspace/hud04-implementation-20261117/step1.joint_fit.npz')
ENTITY='zhul136-university-of-science-and-technology-of-china'
PROJECT='ins1116'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    raw=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with path.open('x') as f:f.write(raw)
    if json.loads(path.read_text())!=value:raise ValueError('Report readback mismatch')

def worker(gate,payload):
    import numpy as np
    from anatomical_directions import DirectionModel,MOTION_SHA,SEED_SHA
    from direction_solver import solve
    sys.path.insert(0,str(PRIOR))
    from motion_archive import freeze_file
    first,last=payload['first'],payload['last']
    if (first,last) not in ((0,30),(480,510),(966,996)):raise ValueError('Only predetermined first/middle/last windows')
    if sha(MOTION)!=MOTION_SHA or sha(SEED)!=SEED_SHA:raise ValueError('Pinned unmodified inputs')
    source=DirectionModel(SOURCE,'source',wrist_endpoint=VARIANT);target=DirectionModel(TARGET,'target',wrist_endpoint=VARIANT)
    with np.load(MOTION,allow_pickle=False) as z:
        if float(z['framerate'].item())!=90 or z['joint_pos'].shape!=(997,29):raise ValueError('997 native 90-Hz frames')
        # Intentionally do not index root_pos/root_quat/body_pos/body_quat.
        source_q=source.order(z['joint_pos'],z['joint_names'].tolist())
    with np.load(SEED,allow_pickle=False) as z:
        if float(z['framerate'].item())!=90:raise ValueError('Seed native rate')
        seed=target.order(z['joint_pos'],z['joint_names'].tolist())
    desired=source.evaluate(source_q[first:last+1]);joint,report=solve(target,seed,desired,first,last,gate,outer_limit=12,trust_radius=.03 if VARIANT=='distal_chain' else .15)
    stem='DIRECTION_WINDOW_'+str(first)+'_'+str(last)
    result=ROOT/(stem+'.npz')
    with result.open('xb') as f:
        np.savez_compressed(f,joint_pos=joint,joint_names=np.array(target.names),frame_indices=np.arange(first,last+1),
                            framerate=np.array(90),role=np.array('diagnostic_not_motion_NO_WORLD_ROOT'))
    with np.load(result,allow_pickle=False) as z:np.testing.assert_array_equal(z['joint_pos'],joint)
    archive=freeze_file(result,Path('/workspace/hud04-backups/sha256'),role='diagnostic_not_motion',source_sha256=MOTION_SHA,model_sha256=sha(TARGET))
    if sha(MOTION)!=MOTION_SHA or sha(SEED)!=SEED_SHA:raise ValueError('Input changed during diagnostic')
    report.update(source_sha256=MOTION_SHA,seed_sha256=SEED_SHA,source_task_model_sha256=sha(SOURCE),target_model_sha256=sha(TARGET),
                  artifact=result.name,artifact_sha256=sha(result),archive=archive,source_data_modified=False,source_assets_modified=False,
                  direction_names=[x[0] for x in source.pairs],source_pairs=[list(x) for x in source.pairs],target_pairs=[list(x) for x in target.pairs],
                  correspondence_status='explicit diagnostic joint-pivot mapping NOT reviewed palm/sole/material contacts',
                  anatomical_basis_assumption='URDF pelvis/base anatomical axes correspond; no NPZ export/world calibration claim',
                  wrist_endpoint=VARIANT,code_sha256={name:sha(CODE/name) for name in ('anatomical_directions.py','direction_solver.py','run_direction_diagnostic.py')})
    save(ROOT/(stem+'.json'),report)
    return report

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline-after-audience-failure',action='store_true')
    args=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'DIRECTION_SOLVE_INTENT.json').exists():raise FileExistsError('Inspect existing intent/results; NEVER automatically rerun')
    sys.path.insert(0,str(PRIOR))
    from bounded_runtime import Budget,supervise,save_diagnostic
    import wandb
    access='not_checked';online=False
    if args.offline_after_audience_failure:
        failure=json.loads((CODE/'WANDB_INIT_FAILURE.json').read_text())
        if failure.get('write_access')!='invalid_audience' or failure.get('solver_started') is not False:raise ValueError('Explicit existing pre-solve audience failure required')
        access='write_invalid_audience_despite_successful_project_read'
    else:
        try:
            project=wandb.Api(timeout=15).project(PROJECT,entity=ENTITY)
            if project is None:access='project_not_returned'
            else:access='project_read_succeeded';online=True
        except Exception as e:
            access='invalid_audience' if 'invalid audience' in str(e).lower() else type(e).__name__
        save(ROOT/'WANDB_ACCESS_CHECK.json',dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target=ENTITY+'/'+PROJECT,result=access,online=online))
    provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if os.environ.get(v)}
    # This creates one explicit diagnostic experiment record. Offline is clearly
    # pending sync, not claimed to exist on the server.
    try:
        run=wandb.init(entity=ENTITY,project=PROJECT,name='hud04-step1-pelvis-direction-window-diagnostic-NOT-reference',job_type='root-local-direction-diagnostic',
            mode='online' if online else 'offline',dir=str(ROOT),settings=wandb.Settings(init_timeout=20),
            config={'_wb_agent':provenance,'base_commit':'87646a750abed29fdbf65335e993166507317e96','cpu_only':True,'windows':[[0,30],[480,510],[966,996]],
                    'max_outer_iterations_per_window':12,'lp_time_limit_seconds':2,'fps':90,'accepted_reference':False,'training_ready':False,
                    'wrist_endpoint':VARIANT,'joint_trust_radius_rad':.03 if VARIANT=='distal_chain' else .15})
    except Exception as e:
        save(ROOT/'WANDB_INIT_FAILURE.json',dict(target=ENTITY+'/'+PROJECT,read_access=access,write_access='invalid_audience' if 'invalid audience' in str(e).lower() else type(e).__name__,solver_started=False,gpu_submissions=0))
        raise
    budget=Budget(window_seconds=90,total_seconds=180,iterations=12,numeric_threads=2,max_windows=3)
    save(ROOT/'DIRECTION_SOLVE_INTENT.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_id=run.id,wandb_online=online,wandb_access=access,
        cpu_only=True,gpu_submissions=0,mode='pelvis-local-direction-joints-only diagnostic; NO world motion',max_windows=3,max_frames=93,accepted_reference=False,
        wrist_endpoint=VARIANT,joint_trust_radius_rad=.03 if VARIANT=='distal_chain' else .15))
    report=supervise('run_direction_diagnostic','worker',[dict(first=a,last=b) for a,b in ((0,30),(480,510),(966,996))],budget)
    rows=[w['diagnostic'] for w in report['windows'] if w['status']=='completed']
    report['solver_calls']=sum(x['solver_calls'] for x in rows)
    report['partial_joint_direction_fit_generated']=bool(rows)
    report['full_retarget_solver_calls']=0
    report['logging_mode']='online' if online else 'offline_pending_sync'
    report['wandb_access_result']=access
    save_diagnostic(ROOT/'DIRECTION_SOLVE_RESULT.json',report)
    for row in rows:
        run.log({'first_frame':row['first'],'initial_mean_direction_degrees':row['initial']['mean_degrees'],
                 'final_mean_direction_degrees':row['final']['mean_degrees'],'initial_p95_direction_degrees':row['initial']['p95_degrees'],
                 'final_p95_direction_degrees':row['final']['p95_degrees'],'lp_calls':row['solver_calls'],
                 'position_max_violation_rad':row['full_sequence_position_max_violation_rad'],
                 'velocity_max_violation_rad_s':row['full_sequence_velocity_max_violation_rad_s']})
    run.summary.update({'completed_windows':len(rows),'actual_lp_calls':report['solver_calls'],'candidate_motion_generated':False,'accepted_reference':False,'training_ready':False})
    run.finish(exit_code=0 if report['status']=='completed' else 1)
    receipt=dict(run_id=run.id,logging_mode=report['logging_mode'],server_readback_verified=False,offline_directory=None if online else str(Path(run.dir).parent),
        status=report['status'],completed_windows=len(rows),actual_lp_calls=report['solver_calls'],gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False)
    if online:
        check=wandb.Api(timeout=15).run(ENTITY+'/'+PROJECT+'/'+run.id)
        if check.summary.get('completed_windows')!=len(rows):raise ValueError('W&B readback mismatch')
        receipt['server_readback_verified']=True;receipt['url']=check.url
    save(ROOT/'DIRECTION_SOLVE_RECEIPT.json',receipt)
    print(json.dumps(dict(receipt=receipt,windows=[dict(first=x['first'],last=x['last'],initial=x['initial'],final=x['final'],solver_calls=x['solver_calls'],termination=x['termination'],
        max_position_violation=x['full_sequence_position_max_violation_rad'],max_velocity_violation=x['full_sequence_velocity_max_violation_rad_s']) for x in rows]),indent=2),flush=True)

if __name__=='__main__':main()