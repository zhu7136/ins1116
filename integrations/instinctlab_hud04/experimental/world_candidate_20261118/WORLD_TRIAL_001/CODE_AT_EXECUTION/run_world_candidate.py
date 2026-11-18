"""NEW one-shot CPU candidate trial. Offline W&B; NO GPU/Flux mutations.

No accepted manifest. A complete 35D candidate is still forbidden for training
until independent correspondence, terrain, support and dynamic gates pass.
"""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).parent
def write(path,body):
    with Path(path).open('x') as f:json.dump(body,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(directory):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from world_features import problem_data,metrics,BASE_COMMIT,digest,TK,SEED,GEOMETRY
    from world_solver import solve
    import wandb
    directory=Path(directory)
    wb={key:os.environ[value] for key,value in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if os.environ.get(value)}
    run=wandb.init(entity='zhul136-university-of-science-and-technology-of-china',project='ins1116',
        name='hud04-full997-world-root-end-orientation-UNACCEPTED',job_type='bounded-world-motion-candidate',mode='offline',dir=str(directory),
        config={'_wb_agent':wb,'base_commit':BASE_COMMIT,'frames':997,'fps':90,'body_dof':29,'root_tangent_dof':6,'head_dof':0,
                'accepted_reference':False,'cpu_only':True,'max_lp_calls':34,'logging_reason':'previous_online_invalid_audience_not_assumed_resolved'},settings=wandb.Settings(init_timeout=15))
    try:
        model,q,roots,desired,correspondence=problem_data()
        inputs={str(p):digest(p) for p in [TK.SOURCE,TK.TARGET,TK.MOTION,SEED,GEOMETRY]}
        write(directory/'CORRESPONDENCE_PROPOSAL_NOT_ACCEPTED.json',correspondence)
        initial=metrics(model,q,roots,desired)
        def progress(row):
            with (directory/'PROGRESS.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
        q,roots,report=solve(model,q,roots,desired,progress)
        final=metrics(model,q,roots,desired)
        for p,d in inputs.items():
            if digest(p)!=d:raise ValueError('Input changed while solving')
        # Numeric export to a NEW quarantine directory only. Unit quaternions
        # from actual solved SO3, continuous sign for serialization, no q clip.
        xyzw=Rotation.from_matrix(roots[:,:3,:3]).as_quat();quat=xyzw[:,[3,0,1,2]]
        for i in range(1,len(quat)):
            if quat[i]@quat[i-1]<0:quat[i]*=-1
        output=directory/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
        with output.open('xb') as f:np.savez_compressed(f,framerate=np.array(90),joint_names=np.array(model.names),joint_pos=q,
            base_pos_w=roots[:,:3,3],base_quat_w=quat,frame_indices=np.arange(997),
            role=np.array('FULL997_WORLD_CANDIDATE_UNACCEPTED_FORBIDDEN_FOR_TRAINING'),accepted_reference=np.array(False),training_ready=np.array(False))
        with np.load(output,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['joint_pos'],q);np.testing.assert_array_equal(z['base_pos_w'],roots[:,:3,3])
            np.testing.assert_array_equal(z['base_quat_w'],quat)
        report.update(initial=initial,final=final,inputs_sha256=inputs,artifact=output.name,artifact_sha256=digest(output),
            source_commit=BASE_COMMIT,frames=997,intervals=996,fps=90,body_dof=29,head_dof=0,
            world_root_optimized=True,source_contract_conditional=True,source_target_material_calibration_accepted=False,
            terrain_constraints_implemented=False,self_collision_constraints_implemented=False,support_constraints_implemented=False,
            geometry_acceptance=False,dynamics_acceptance=False,accepted_reference=False,training_ready=False,
            new_full_world_candidate_generated=True,posthoc_joint_clipping=False,source_or_limits_modified=False,
            run_id=run.id,wandb_mode='offline_pending_sync',server_readback_verified=False)
        write(directory/'WORLD_CANDIDATE_RESULT.json',report)
        run.log({'objective_initial':initial['weighted_l1'],'objective_final':final['weighted_l1'],
                 'lp_calls':report['actual_lp_calls'],'position_max_violation_rad':report['position_max_violation_rad'],
                 'velocity_max_violation_rad_s':report['velocity_max_violation_rad_s']})
        run.summary['accepted_reference']=False;run.summary['training_ready']=False;run.finish()
        print(json.dumps({k:report[k] for k in ['artifact','artifact_sha256','actual_lp_calls','accepted_updates','frames_changed','position_max_violation_rad','velocity_max_violation_rad_s','accepted_reference']}),flush=True)
    except BaseException:
        run.finish(exit_code=1);raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
    if args.worker:return worker(args.worker)
    # Explicit one-shot intent BEFORE work. Never repeat this launcher; a later
    # scientific trial must be new, versioned, with reviewed scope and code.
    directory=ROOT/'WORLD_TRIAL_001';directory.mkdir(exist_ok=False)
    write(directory/'INTENT.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stage='bounded_full997_world35D_UNACCEPTED',parent_wall_seconds=100,child_algorithm_seconds=80,
        planned_lp_calls=34,lp_seconds=1.5,lp_iterations=12000,numeric_threads=2,automatic_retries=0,
        root_step_raw=.025,root_angle_step_rad=.1,joint_step_rad=.15,accepted_reference=False,gpu_submissions=0,ppo_updates=0))
    snapshot=directory/'CODE_AT_EXECUTION';snapshot.mkdir()
    for p in sorted(ROOT.glob('*.py')):(snapshot/p.name).write_bytes(p.read_bytes())
    write(directory/'CODE_HASHES.json',{p.name:sha(p) for p in sorted(snapshot.glob('*.py'))})
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
    env['CUDA_VISIBLE_DEVICES']=''
    start=time.monotonic();timed_out=False
    with (directory/'EXECUTION.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(directory)],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:child.wait(timeout=100)
        except subprocess.TimeoutExpired:
            timed_out=True;child.terminate()
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
    receipt=dict(child_exit=child.returncode,child_reaped=child.poll() is not None,parent_timeout=timed_out,
        elapsed_seconds=time.monotonic()-start,result_exists=(directory/'WORLD_CANDIDATE_RESULT.json').exists(),
        artifact_exists=(directory/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz').exists(),gpu_submissions=0,ppo_updates=0,accepted_reference=False)
    write(directory/'EXECUTION_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
    if child.returncode or timed_out:raise SystemExit(1)

if __name__=='__main__':main()