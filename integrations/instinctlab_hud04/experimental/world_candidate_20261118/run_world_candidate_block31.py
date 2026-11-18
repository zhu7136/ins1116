"""NEW second scientific trial: SAME objective, only31frame blocks, no retries."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
from run_world_candidate import write,sha

ROOT=Path(__file__).parent
def main():
    directory=ROOT/'WORLD_TRIAL_002_BLOCK31';directory.mkdir(exist_ok=False)
    write(directory/'INTENT.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        change_from_trial001='ONLY local block size93->31 and full coverage schedule; same objective/weights/seed/trust/limits/LPbound',
        previous_failure='update19 LPstatus1 iteration12000 reached; no artifact exported',
        stage='bounded_full997_world35D_UNACCEPTED',parent_wall_seconds=100,child_algorithm_seconds=80,
        planned_lp_calls=98,maximum_block_frames=31,lp_seconds=1.5,lp_iterations=12000,numeric_threads=2,automatic_retries=0,
        root_step_raw=.025,root_angle_step_rad=.1,joint_step_rad=.15,accepted_reference=False,gpu_submissions=0,ppo_updates=0))
    snapshot=directory/'CODE_AT_EXECUTION';snapshot.mkdir()
    for p in sorted(ROOT.glob('*.py')):(snapshot/p.name).write_bytes(p.read_bytes())
    write(directory/'CODE_HASHES.json',{p.name:sha(p) for p in sorted(snapshot.glob('*.py'))})
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
    env['CUDA_VISIBLE_DEVICES']=''
    start=time.monotonic();timed_out=False
    with (directory/'EXECUTION.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(ROOT/'run_world_candidate.py'),'--worker',str(directory)],env=env,stdout=log,stderr=subprocess.STDOUT)
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