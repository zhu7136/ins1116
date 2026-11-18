"""New single finite CPU trial with concurrent86f05 per-pose guard.
No old launcher main is invoked; exact seed005 retained for comparison.
"""
import datetime,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from run_wholebody_trial006_once import worker,SEED_SHA,sha,write
def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':return worker(Path(sys.argv[2]))
    log=(ROOT/'PER_POSE_FINAL_PREEXECUTION_TESTS.log').read_text()
    if 'Ran 24 tests' not in log or not log.rstrip().endswith('OK'):raise ValueError('24 preexecution tests gate')
    directory=ROOT/'WHOLEBODY_TRIAL_008';directory.mkdir(exist_ok=False)
    write(directory/'INTENT.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base='86f05dffa8696465147d7e3ebf99aff568a84c26',seed_sha256=SEED_SHA,enforce_foot_guard=True,enforce_per_pose_guard=True,
        change='Same43shapes/all903pairs/fresh-nonlinear geometry; add pinned concurrent86f05 exact13axis footbox-SAT per-native-frame max overlap rejection before assignment; retains aggregate no-new-frame guard',
        comparison_seed='005; NOT chaining006 or007',foot_SAT_not_penetration_depth=True,geometry_not_acceptance=True,correspondence_still_unaccepted=True,
        frames=997,fps=90,body_actions=29,head_fixed_zero=True,parent_seconds=190,algorithm_seconds=160,
        maximum_lp_calls=100,lp_seconds=1.5,lp_iterations=12000,numeric_threads=2,automatic_retries=0,
        gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False))
    snapshot=directory/'CODE_AT_EXECUTION';snapshot.mkdir()
    dependencies=[Path('/workspace/hud04-world-candidate-20261118')/n for n in ['world_features.py','world_solver.py','collision_temporal_solver.py','terrain_temporal_terms.py']]
    dependencies.append(ROOT/'REMOTE_ADVANCE_REVIEW_86f05/nonlinear_guard_20261118/box_triangle_guard.py')
    for p in sorted(ROOT.glob('*.py'))+dependencies:
        with (snapshot/p.name).open('xb') as f:f.write(p.read_bytes())
    write(directory/'CODE_HASHES.json',{p.name:sha(p) for p in sorted(snapshot.glob('*.py'))})
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='2'
    env['CUDA_VISIBLE_DEVICES']='';start=time.monotonic();timeout=False
    with (directory/'EXECUTION.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__)), '--worker',str(directory)],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:child.wait(timeout=190)
        except subprocess.TimeoutExpired:
            timeout=True;child.terminate()
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
    receipt=dict(child_exit=child.returncode,child_reaped=child.poll() is not None,parent_timeout=timeout,elapsed_seconds=time.monotonic()-start,
        result_exists=(directory/'RESULT.json').exists(),artifact_exists=(directory/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz').exists(),
        accepted_reference=False,training_ready=False,gpu_submissions=0,ppo_updates=0)
    write(directory/'EXECUTION_RECEIPT.json',receipt);print(__import__('json').dumps(receipt,indent=2),flush=True)
    if child.returncode or timeout:raise SystemExit(1)
if __name__=='__main__':main()