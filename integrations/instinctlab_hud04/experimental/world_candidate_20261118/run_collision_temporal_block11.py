"""NEW one-shot trial004, only shrink local block to11, no old replay."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
from run_world_candidate import write,sha
ROOT=Path(__file__).parent

def main():
    directory=ROOT/'WORLD_TRIAL_004_FOOT_TEMPORAL_BLOCK11';directory.mkdir(exist_ok=False)
    write(directory/'INTENT.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        change_from_trial003='ONLY block31->11 / full coverage schedule; samefoot/temporal/fitweights and named hardlimits, seedtrial002',
        previous_failure='trial003 first LP hit1.5s time cap,4379iterations, no artifact export; not infeasibility proof',
        parent_wall_seconds=100,child_algorithm_seconds=80,planned_lp_calls=200,maximum_block_frames=11,lp_seconds=1.5,lp_iterations=12000,
        numeric_threads=2,automatic_retries=0,accepted_reference=False,gpu_submissions=0,ppo_updates=0))
    snapshot=directory/'CODE_AT_EXECUTION';snapshot.mkdir()
    for p in sorted(ROOT.glob('*.py')):(snapshot/p.name).write_bytes(p.read_bytes())
    write(directory/'CODE_HASHES.json',{p.name:sha(p) for p in sorted(snapshot.glob('*.py'))})
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
    env['CUDA_VISIBLE_DEVICES']=''
    start=time.monotonic();timeout=False
    with (directory/'EXECUTION.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(ROOT/'run_collision_temporal_candidate.py'),'--worker',str(directory)],env=env,stdout=log,stderr=subprocess.STDOUT)
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