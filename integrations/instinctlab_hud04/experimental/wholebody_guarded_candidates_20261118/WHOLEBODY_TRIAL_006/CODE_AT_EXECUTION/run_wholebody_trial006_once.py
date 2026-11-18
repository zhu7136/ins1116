"""NEW one-shot trial006; existing directory prevents duplicate CPU execution."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent
sys.path[:0]=[str(ROOT),'/workspace/hud04-world-candidate-20261118']
from run_world_candidate import sha,write
SEED=Path('/workspace/hud04-world-candidate-20261118/WORLD_TRIAL_005_FOOT_PRIORITY/STEP1_WORLD_CANDIDATE_UNACCEPTED.npz')
SEED_SHA='9353b185d92c21ceeb56a289b926f89d3dc72c5343745fc14e26889bc6d728d4'

def worker(directory):
    import numpy as np,wandb
    from scipy.spatial.transform import Rotation
    from world_features import problem_data,metrics,TK,GEOMETRY
    from wholebody_solver import solve
    if sha(SEED)!=SEED_SHA:raise ValueError('Pinned005 changed')
    model,_,_,desired,proposal=problem_data();write(directory/'CORRESPONDENCE_NOT_ACCEPTED.json',proposal)
    with np.load(SEED,allow_pickle=False) as z:q=z['joint_pos'].copy();roots=TK.K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz')
    inputs={str(p):sha(p) for p in [SEED,TK.SOURCE,TK.TARGET,TK.MOTION,GEOMETRY]}
    before=metrics(model,q,roots,desired)
    provenance={k:os.environ[e] for k,e in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if e in os.environ}
    run=wandb.init(entity='zhul136-university-of-science-and-technology-of-china',project='ins1116',mode='offline',dir=str(directory),
        name='hud04-wholebody-newcontact-006-NOT-ACCEPTED',job_type='cpu-retarget-candidate-not-accepted',
        config={'_wb_agent':provenance,'base_commit':'e50115ec3fdabc3583224ea060e3054d61631939','max_lp_calls':100,
        'algorithm_wall_seconds':140,'all_shapes':43,'all_pairs':903,'geometry_score_not_depth':True,'accepted_reference':False,
        'offline_reason':'previous invalid audience; no assertion of online project recovery'},settings=wandb.Settings(init_timeout=15))
    try:
        with (directory/'PROGRESS.jsonl').open('x') as f:
            def progress(row):
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();print(json.dumps(row,allow_nan=False),flush=True)
                run.log({k:v for k,v in row.items() if isinstance(v,(float,int,bool))})
            q,roots,report=solve(model,q,roots,desired,progress)
        for p,h in inputs.items():
            if sha(p)!=h:raise ValueError('Pinned original input modified')
        quat=Rotation.from_matrix(roots[:,:3,:3]).as_quat()[:,[3,0,1,2]]
        for k in range(1,997):
            if quat[k]@quat[k-1]<0:quat[k]*=-1
        artifact=directory/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
        np.savez_compressed(artifact,base_pos_w=roots[:,:3,3],base_quat_w=quat,joint_pos=q,joint_names=np.array(model.names),
            framerate=np.array(90),frame_indices=np.arange(997),accepted_reference=np.array(False),training_ready=np.array(False),
            artifact_role=np.array('UNACCEPTED_ALL43_NEWCONTACT_CANDIDATE_NEVER_TRAINING_INPUT'))
        with np.load(artifact,allow_pickle=False) as z:
            if not np.array_equal(z['joint_pos'],q) or z['accepted_reference'].item():raise ValueError('Artifact local readback')
        result=dict(**report,initial=before,final=metrics(model,q,roots,desired),artifact_sha256=sha(artifact),inputs_sha256=inputs,
            offline_wandb_run_id=run.id,online_verified=False,head_fixed_zero=True,frames=997,fps=90,body_actions=29,
            no_joint_clipping=True,no_resampling=True,gpu_submissions=0,ppo_updates=0)
        write(directory/'RESULT.json',result);run.summary.update({'accepted_reference':False,'training_ready':False,'updates':report['accepted_updates']})
        artifact_log=wandb.Artifact('hud04-wholebody006-UNACCEPTED',type='diagnostic-motion-not-reference',metadata={'accepted_reference':False})
        artifact_log.add_file(str(artifact));run.log_artifact(artifact_log);run.finish()
        print(json.dumps({'artifact_sha256':result['artifact_sha256'],'accepted_updates':report['accepted_updates'],'accepted_reference':False}),flush=True)
    except BaseException:run.finish(exit_code=1);raise

def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':return worker(Path(sys.argv[2]))
    log=(ROOT/'WHOLEBODY_FINAL_PREEXECUTION_TESTS.log').read_text()
    if 'Ran 16 tests' not in log or not log.rstrip().endswith('OK'):raise ValueError('Pre-execution tests gate')
    directory=ROOT/'WHOLEBODY_TRIAL_006';directory.mkdir(exist_ok=False)
    write(directory/'INTENT.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base='e50115ec3fdabc3583224ea060e3054d61631939',seed_sha256=SEED_SHA,
        change='all43 true primitive supports/finite triangles/all903pairs; fresh full-shape query on EVERY candidate alpha',
        trial_is_not_single_factor_ablation=True,correspondence_still_unaccepted=True,
        root_xyz_trust_raw=.015,root_rot_trust_rad=.04,joint_trust_rad=.08,geometry_weight=1000.,
        raw_margin=.001,margin_is_not_acceptance_threshold=True,padding_raw=.015,
        frames=997,fps=90,body_actions=29,head_fixed_zero=True,parent_seconds=170,algorithm_seconds=140,
        maximum_lp_calls=100,lp_seconds=1.5,lp_iterations=12000,numeric_threads=2,automatic_retries=0,
        gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False))
    snapshot=directory/'CODE_AT_EXECUTION';snapshot.mkdir()
    dependencies=[Path('/workspace/hud04-world-candidate-20261118')/n for n in ['world_features.py','world_solver.py','collision_temporal_solver.py','terrain_temporal_terms.py']]
    for p in sorted(ROOT.glob('*.py'))+dependencies:
        with (snapshot/p.name).open('xb') as f:f.write(p.read_bytes())
    write(directory/'CODE_HASHES.json',{p.name:sha(p) for p in sorted(snapshot.glob('*.py'))})
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='2'
    env['CUDA_VISIBLE_DEVICES']='';start=time.monotonic();timeout=False
    with (directory/'EXECUTION.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__)),'--worker',str(directory)],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:child.wait(timeout=170)
        except subprocess.TimeoutExpired:
            timeout=True;child.terminate()
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
    receipt=dict(child_exit=child.returncode,child_reaped=child.poll() is not None,parent_timeout=timeout,elapsed_seconds=time.monotonic()-start,
        result_exists=(directory/'RESULT.json').exists(),artifact_exists=(directory/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz').exists(),
        accepted_reference=False,training_ready=False,gpu_submissions=0,ppo_updates=0)
    write(directory/'EXECUTION_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
    if child.returncode or timeout:raise SystemExit(1)
if __name__=='__main__':main()