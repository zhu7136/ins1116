"""NEW standalone import observer entry. Default plan only; no Flux mutations."""
import argparse,importlib.util,json,os,subprocess,sys
from pathlib import Path
from bounded_import_runner import emit_evidence,run_child
from import_observation_probe import plan,verify_bundle

def main(args):
    if not args.execute:
        print(json.dumps(plan()),flush=True);return
    if not args.repo_root or not args.expected_commit or not args.output:raise ValueError('Explicit repo-root/expected-commit/output required')
    repo=Path(args.repo_root).resolve()
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=5).strip()!=args.expected_commit:raise ValueError('Commit mismatch')
    verify_bundle(repo)
    if args.isaac_launcher:
        launcher=Path(args.isaac_launcher).resolve()
    elif importlib.util.find_spec('isaaclab') is not None:launcher=Path(sys.executable).resolve()
    else:
        # Historical worker supported these; only choose a real local file.
        # A pathname alone does not prove it works; runtime report must do that.
        launcher=next((p for p in (Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')) if p.is_file()),None)
    if launcher is None or not launcher.is_file():raise RuntimeError('No local Isaac launcher; do not install or retry')
    env=os.environ.copy()
    for key in ['flux1117','github1117','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY']:env.pop(key,None)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PXR_WORK_THREAD_LIMIT']:env[key]='2'
    # The supervisor creates this directory before starting the child. Keep
    # conversion scratch inside our new output, without changing model data.
    env['TMPDIR']=str(Path(args.output).resolve())
    result=run_child([str(launcher),str(Path(__file__).with_name('import_observation_probe.py')),'--execute',
        '--repo-root',str(repo),'--expected-commit',args.expected_commit,'--output',str(Path(args.output).resolve())],
        args.output,seconds=240,env=env)
    print(json.dumps(result),flush=True)
    if not result['observation_completed']:raise SystemExit(1)
    emit_evidence(args.output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--repo-root')
    p.add_argument('--expected-commit');p.add_argument('--output');p.add_argument('--isaac-launcher');main(p.parse_args())