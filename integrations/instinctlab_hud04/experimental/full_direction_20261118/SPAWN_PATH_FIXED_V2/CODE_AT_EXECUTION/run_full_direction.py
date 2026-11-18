"""ONE-SHOT bounded CPU diagnostic; no GPU, world-root or motion exporter."""
import datetime,json,os,sys,traceback
from pathlib import Path
from inputs import HERE as CODE,PRIOR,BASE_COMMIT,load,sha,dependency,SOURCE,TARGET,MOTION,SEED

EXECUTION=os.environ.get('HUD04_FULL_DIRECTION_EXECUTION','initial')
if EXECUTION not in ('initial','spawn_path_fixed_v2'):raise ValueError('Explicit execution version')
HERE=CODE if EXECUTION=='initial' else CODE/'SPAWN_PATH_FIXED_V2'

ENTITY='zhul136-university-of-science-and-technology-of-china'
PROJECT='ins1116'
def save(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
    if json.loads(Path(path).read_text())!=obj:raise ValueError('JSON readback failed')

def worker(gate,payload):
    try:return work(gate,payload)
    except Exception:
        with (HERE/'WORKER_FAILURE.log').open('x') as f:f.write(traceback.format_exc())
        raise

def work(gate,payload):
    import numpy as np
    from full_direction_solver import solve,stats
    if payload!={'native_frames':997}:raise ValueError('One explicit native full sequence')
    source,target,seed,desired,binding=load()
    def progress(row):
        with (HERE/'LP_PROGRESS.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        print('full_direction_iteration',json.dumps(row),flush=True)
    q,report=solve(target,seed,desired,gate,outer_limit=12,trust_radius=.15,lp_seconds=4.,progress=progress)
    report.update(binding)
    report.update(code_sha256={name:sha(CODE/name) for name in ('inputs.py','full_direction_solver.py','run_full_direction.py','run_full_direction_v2.py')},
        source_pairs=[list(x) for x in source.pairs],target_pairs=[list(x) for x in target.pairs],
        correspondence_status='conditional URDF-anatomical-basis joint-pivot mapping; NOT accepted material/contact correspondence',
        anatomical_basis_assumption='URDF pelvis/base anatomical axes correspond; no exporter/world calibration assertion')
    # Compare identical native frames, not 997-frame averages against 93 frames.
    rows=[]
    for first,last in ((0,30),(480,510),(966,996)):
        old_path=PRIOR/'FIRST_WRIST_COMPARISON'/('DIRECTION_WINDOW_'+str(first)+'_'+str(last)+'.npz')
        old_report=old_path.with_suffix('.json')
        old_meta=json.loads(old_report.read_text())
        if sha(old_path)!=old_meta['artifact_sha256']:raise ValueError('Prior window artifact changed')
        with np.load(old_path,allow_pickle=False) as z:old=target.order(z['joint_pos'],z['joint_names'].tolist())
        rows.append(dict(first=first,last=last,prior_artifact_sha256=sha(old_path),prior=stats(target.evaluate(old),desired[first:last+1]),
            global_fit=stats(target.evaluate(q[first:last+1]),desired[first:last+1])))
    report['same_93_frame_comparison']=rows
    if report['accepted_steps']:
        out=HERE/'STEP1_FULL_LENGTH_JOINT_DIAGNOSTIC.npz'
        with out.open('xb') as f:
            np.savez_compressed(f,joint_pos=q,joint_names=np.array(target.names),frame_indices=np.arange(997,dtype=np.int64),
                framerate=np.array(90),role=np.array('full_length_joint_diagnostic_NOT_reference_NO_WORLD_ROOT'))
        with np.load(out,allow_pickle=False) as z:np.testing.assert_array_equal(z['joint_pos'],q)
        archive=dependency('motion_archive.py').freeze_file(out,Path('/workspace/hud04-backups/sha256'),role='diagnostic_not_motion',
            source_sha256=binding['source_motion_sha256'],model_sha256=binding['target_model_sha256'])
        report.update(artifact=out.name,artifact_sha256=sha(out),archive=archive)
    # Re-read all pinned source/seed/models after computation and writing.
    _,_,_,_,after=load()
    if after!=binding:raise ValueError('Inputs changed during solve')
    save(HERE/'FULL_DIRECTION_REPORT.json',report)
    return report

def main():
    HERE.mkdir(exist_ok=True)
    if (HERE/'FULL_DIRECTION_INTENT.json').exists():raise FileExistsError('Existing intent: inspect, NEVER replay')
    failure=json.loads((PRIOR/'WANDB_INIT_FAILURE.json').read_text())
    if failure.get('write_access')!='invalid_audience':raise ValueError('Explicit prior audience failure required for offline path')
    runtime=dependency('bounded_runtime.py')
    import wandb
    provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if os.environ.get(v)}
    run=wandb.init(entity=ENTITY,project=PROJECT,name='hud04-full997-joint-direction-diagnostic-NOT-reference',mode='offline',dir=str(HERE),
        job_type='full-length-joint-direction-cpu',config={'_wb_agent':provenance,'base_commit':BASE_COMMIT,'frames':997,'fps':90,
        'body_dof':29,'fixed_head':True,'outer_limit':12,'lp_seconds':4,'wall_seconds_hard_parent':90,'native_highs_threads':2,
        'root_optimized':False,'terrain_constraints':False,'accepted_reference':False,'training_ready':False,'gpu_submissions':0})
    save(HERE/'FULL_DIRECTION_INTENT.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_commit=BASE_COMMIT,
        cpu_only=True,frames=997,body_dof=29,run_id=run.id,wandb_mode='offline_pending_sync',
        offline_reason='previous online init invalid audience; NOT an assertion of present online write access',
        prior_write_failure_sha256=sha(PRIOR/'WANDB_INIT_FAILURE.json'),max_outer_iterations=12,lp_seconds=4,hard_parent_seconds=90,
        source_motion_sha256=sha(MOTION),seed_sha256=sha(SEED),source_model_sha256=sha(SOURCE),target_model_sha256=sha(TARGET),
        gpu_submissions=0,ppo_updates=0,candidate_motion_generated=False,accepted_reference=False,training_ready=False))
    result=runtime.supervise('run_full_direction','worker',[{'native_frames':997}],runtime.Budget(window_seconds=90,total_seconds=90,iterations=12,numeric_threads=2,max_windows=1))
    completed=[w['diagnostic'] for w in result['windows'] if w['status']=='completed']
    result['solver_calls']=sum(r['actual_lp_calls'] for r in completed)
    result['actual_solver_call_count_complete']=bool(completed)
    result['full_length_joint_diagnostic_generated']=bool(completed and completed[0]['accepted_steps'])
    result['logging_mode']='offline_pending_sync'
    runtime.save_diagnostic(HERE/'FULL_DIRECTION_SUPERVISION.json',result)
    if completed:
        r=completed[0]
        run.log({'initial_mean_direction_degrees':r['initial']['mean_degrees'],'final_mean_direction_degrees':r['final']['mean_degrees'],
            'final_p95_direction_degrees':r['final']['p95_degrees'],'final_max_direction_degrees':r['final']['max_degrees'],
            'lp_calls':r['actual_lp_calls'],'accepted_steps':r['accepted_steps'],'position_max_violation_rad':r['position_max_violation_rad'],
            'velocity_max_violation_rad_s':r['velocity_max_violation_rad_s']})
    success=bool(completed and completed[0]['accepted_steps'])
    run.summary.update({'cpu_only':True,'accepted_reference':False,'training_ready':False,'full_length_joint_diagnostic_generated':success})
    run_id=run.id;offline_dir=str(Path(run.dir).parent);run.finish(exit_code=0 if success else 1)
    save(HERE/'FULL_DIRECTION_RECEIPT.json',dict(status=result['status'],numerical_descent_generated=success,run_id=run_id,
        mode='offline_pending_sync',offline_directory=offline_dir,server_readback_verified=False,
        supervision_sha256=sha(HERE/'FULL_DIRECTION_SUPERVISION.json'),gpu_submissions=0,ppo_updates=0))
    print(json.dumps({'status':result['status'],'numerical_descent_generated':success,'solver_calls':result['solver_calls'],'wall_seconds':result['wall_seconds']},indent=2),flush=True)
    if not success:raise SystemExit(1)

if __name__=='__main__':main()