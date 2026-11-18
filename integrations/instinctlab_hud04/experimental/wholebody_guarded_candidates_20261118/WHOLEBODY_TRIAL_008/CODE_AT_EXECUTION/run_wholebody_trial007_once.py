"""New single bounded CPU trial, adding strict no-foot-regression guard.
Shared worker is called directly, NOT the old trial006 launch main.
"""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from run_wholebody_trial006_once import worker,SEED_SHA,sha,write
def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':return worker(Path(sys.argv[2]))
    log=(ROOT/'GUARD_FINAL_PREEXECUTION_TESTS.log').read_text()
    if 'Ran 20 tests' not in log or not log.rstrip().endswith('OK'):raise ValueError('20 preexecution tests gate')
    directory=ROOT/'WHOLEBODY_TRIAL_007';directory.mkdir(exist_ok=False)
    write(directory/'INTENT.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base='e50115ec3fdabc3583224ea060e3054d61631939',seed_sha256=SEED_SHA,enforce_foot_guard=True,
        change='Trial006 full43/all903/fresh-new-contacts + reject EVERY alpha introducing new footbox/finite-triangle intersecting frame or increasing sum negative foot SAT projection overlap',
        foot_guard_not_penetration_depth=True,geometry_not_acceptance=True,correspondence_still_unaccepted=True,
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
        child=subprocess.Popen([sys.executable,str(Path(__file__)), '--worker',str(directory)],env=env,stdout=log,stderr=subprocess.STDOUT)
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