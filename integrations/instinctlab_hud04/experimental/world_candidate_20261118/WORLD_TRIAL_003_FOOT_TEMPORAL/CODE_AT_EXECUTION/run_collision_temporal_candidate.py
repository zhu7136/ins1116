"""One-shot trial003. Existing trial dirs block replay. Parent100s; GPU0."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
from run_world_candidate import write,sha
ROOT=Path(__file__).parent
PRIOR=ROOT/'WORLD_TRIAL_002_BLOCK31'

def worker(directory):
    import numpy as np
    import wandb
    from world_features import problem_data,metrics,TK,SEED,GEOMETRY
    from scipy.spatial.transform import Rotation
    from collision_temporal_solver import solve
    from world_solver import plan
    seed=PRIOR/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
    if sha(seed)!='7c2c98d7811e3c847c7a20239612006647d56f42777a1ddea914c52f2711f253':raise ValueError('Previous seed changed')
    target,_,_,desired,desc=problem_data();write(directory/'CORRESPONDENCE_PROPOSAL_NOT_ACCEPTED.json',desc)
    with np.load(seed,allow_pickle=False) as z:q=z['joint_pos'].copy();roots=TK.K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz')
    before=metrics(target,q,roots,desired);initialq=q.copy()
    inputs={str(p):sha(p) for p in [TK.SOURCE,TK.TARGET,TK.MOTION,SEED,GEOMETRY,seed]}
    provenance={key:os.environ[value] for key,value in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if value in os.environ}
    run=wandb.init(entity='zhul136-university-of-science-and-technology-of-china',project='ins1116',mode='offline',dir=str(directory),
        name='hud04-world997-foot-temporal-UNACCEPTED-20261118',job_type='cpu-retarget-candidate-not-accepted',
        config={'_wb_agent':provenance,'full_native_frames':997,'fps':90,'body_dof':29,'accepted_reference':False,'cpu_only':True,
                'max_lp_calls':len(plan()),'logging_reason':'previous_invalid_audience_offline_not_online_verified'},settings=wandb.Settings(init_timeout=15))
    try:
        with (directory/'PROGRESS.jsonl').open('x') as stream:
            def progress(row):
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();print(json.dumps(row),flush=True)
                run.log({k:v for k,v in row.items() if isinstance(v,(int,float,bool))})
            q,roots,report=solve(target,q,roots,desired,progress)
        after=metrics(target,q,roots,desired)
        for p,d in inputs.items():
            if sha(p)!=d:raise ValueError('Input mutated')
        artifact=directory/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
        quat=Rotation.from_matrix(roots[:,:3,:3]).as_quat()[:,[3,0,1,2]]
        for i in range(1,len(quat)):
            if quat[i]@quat[i-1]<0:quat[i]*=-1
        np.savez_compressed(artifact,base_pos_w=roots[:,:3,3],base_quat_w=quat,joint_pos=q,
            joint_names=np.array(target.names),framerate=np.array(90),frame_indices=np.arange(997),accepted_reference=np.array(False),
            artifact_role=np.array('UNACCEPTED_world_candidate_NEVER_training_input'),training_ready=np.array(False))
        with np.load(artifact,allow_pickle=False) as z:
            if not np.array_equal(z['joint_pos'],q) or not np.array_equal(z['base_pos_w'],roots[:,:3,3]) or z['accepted_reference'].item():raise ValueError('Readback')
        result=dict(stage='WORLD997_WITH_NOMINAL_FOOT_CUTS_AND_TEMPORAL_REGULARIZATION_NOT_ACCEPTED',artifact=artifact.name,artifact_sha256=sha(artifact),inputs_sha256=inputs,
            initial=before,final=after,**report,body_joint_names=target.names,head_fixed_zero=True,action_dimensions=29,
            frames_changed_from_previous_candidate=int(np.any(abs(q-initialq)>1e-9,axis=1).sum()),
            offline_wandb_run_id=run.id,online_wandb_verified=False,gpu_submissions=0,ppo_updates=0,
            full_collision_free_proven=False,actual_dynamics_replay_passed=False,accepted_reference=False,training_ready=False)
        write(directory/'WORLD_CANDIDATE_RESULT.json',result)
        run.summary.update({'accepted_reference':False,'training_ready':False,'candidate_frames':997,'updates':report['accepted_updates']})
        a=wandb.Artifact('hud04-world997-foot-temporal-UNACCEPTED-20261118',type='diagnostic-motion-not-reference',metadata={'accepted_reference':False,'training_ready':False})
        a.add_file(str(artifact));a.add_file(str(directory/'WORLD_CANDIDATE_RESULT.json'));run.log_artifact(a);run.finish()
        print(json.dumps({k:result[k] for k in ['artifact_sha256','actual_lp_calls','accepted_updates','accepted_reference']}),flush=True)
    except BaseException:
        run.finish(exit_code=1);raise

def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':return worker(Path(sys.argv[2]))
    directory=ROOT/'WORLD_TRIAL_003_FOOT_TEMPORAL';directory.mkdir(exist_ok=False)
    write(directory/'INTENT.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        change='Add nominal finite footbox/actual terrain triangle local soft escape cuts + root-translation/rotation-matrix and joint temporal second differences',
        previous_candidate_status='REJECTED_nominal_foot_terrain_interior_hits_L240_R162_and_root_jitter',
        unchanged='997frames90Hz/body29/fixedHead/originalnamedPositionVelocityLimits/no_clipping/no_resample/no_longPPO',
        new_weights={'foot_hinge':100,'root_translation_second_derivative':.02,'rotation_matrix_second_derivative':.005,'joint_second_derivative':.0001},
        foot_activation_padding_raw=.025,triangle_normals_used='raw actual crop mesh winding; conditional world/unit binding unchanged',
        parent_wall_seconds=100,child_algorithm_seconds=80,planned_lp_calls=98,maximum_block_frames=31,lp_seconds=1.5,lp_iterations=12000,
        numeric_threads=2,automatic_retries=0,accepted_reference=False,gpu_submissions=0,ppo_updates=0))
    snapshot=directory/'CODE_AT_EXECUTION';snapshot.mkdir()
    for p in sorted(ROOT.glob('*.py')):(snapshot/p.name).write_bytes(p.read_bytes())
    write(directory/'CODE_HASHES.json',{p.name:sha(p) for p in sorted(snapshot.glob('*.py'))})
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
    env['CUDA_VISIBLE_DEVICES']=''
    start=time.monotonic();timeout=False
    with (directory/'EXECUTION.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__)), '--worker',str(directory)],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:child.wait(timeout=100)
        except subprocess.TimeoutExpired:
            timeout=True;child.terminate()
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
    receipt=dict(child_exit=child.returncode,child_reaped=child.poll() is not None,parent_timeout=timeout,elapsed_seconds=time.monotonic()-start,
                 result_exists=(directory/'WORLD_CANDIDATE_RESULT.json').exists(),artifact_exists=(directory/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz').exists(),accepted_reference=False,gpu_submissions=0,ppo_updates=0)
    write(directory/'EXECUTION_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
    if child.returncode or timeout:raise SystemExit(1)
if __name__=='__main__':main()