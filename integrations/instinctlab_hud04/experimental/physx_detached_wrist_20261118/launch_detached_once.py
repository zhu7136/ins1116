"""NEW bounded detached-original-shape control; plan-only unless --execute."""
import argparse,hashlib,json,os,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'integrations/instinctlab_hud04/experimental/physx_wrist_contact_20261118'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def main(a):
    if not a.execute:
        print(json.dumps(dict(stage='plan_only',gpu_started=False,gpu_count=1,child_seconds=240,steps=8,automatic_retries=0,accepted_reference=False,training_ready=False)));return
    if not re.fullmatch('[0-9a-f]{40}',a.expected_commit or ''):raise ValueError('Commit required')
    deps={'contact_probe.py':'a1aff6a7ab4485962852010ad655377569e67be79003a2c99f46b23daf440861','launch_once.py':'d4fb1290bfaef03a6e96ca527c87c435259b4f10731e24bb2cd38bcb85bc5f2d'}
    for n,h in deps.items():
        if sha(V2/n)!=h:raise ValueError('Pinned helper changed')
    sys.path.insert(0,str(V2))
    from contact_probe import verify
    from launch_once import run_owned
    from detached_contract import validate_report
    verify(ROOT,a.expected_commit)
    out=ROOT/('hud04_detached_'+a.expected_commit[:12]);out.mkdir(exist_ok=False)
    env=os.environ.copy()
    for k in ['github1117','flux1117','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY']:env.pop(k,None)
    env['TMPDIR']=str(out)
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PXR_WORK_THREAD_LIMIT']:env[k]='2'
    probe=Path(__file__).with_name('detached_probe.py')
    result=run_owned([sys.executable,str(probe),'--repo-root',str(ROOT),'--expected-commit',a.expected_commit,'--output',str(out)],out,env)
    report=None;error=None;p=out/'contact_report.json'
    if p.is_file() and p.stat().st_size<=65536:
        try:report=json.loads(p.read_text());validate_report(report,a.expected_commit,sha(probe))
        except (ValueError,KeyError,TypeError) as e:error=type(e).__name__+': '+str(e);report=None
    complete=result['return_code']==0 and result['stop_reason'] is None and report is not None
    result.update(stage='hud04_detached_supervisor',source_commit=a.expected_commit,controlled_probe_completed=complete,report_validation_error=error,normal_native_exit_observed=result['return_code']==0 and result['stop_reason'] is None,fast_shutdown_not_graceful_full_cleanup=True,accepted_reference=False,training_ready=False,motion_solver_calls=0,ppo_updates=0)
    save(out/'supervisor_result.json',result)
    # Logs use only currently authorized secret names, no credential output.
    def clean(t):
        for k in ['github1117','flux1117','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY']:
            v=os.environ.get(k)
            if v:t=t.replace(v,'[REDACTED]')
        return re.sub(r'https?://\S+','[URL omitted]',t)
    for n in ['contact_report.json','contact_error.json','supervisor_result.json']:
        p=out/n
        if p.is_file() and p.stat().st_size<=65536:
            d=json.loads(p.read_text());t=json.dumps(dict(stage='hud04_pair_artifact',name=n,sha256=sha(p),payload=d))
            if clean(t)!=t:raise ValueError('Sensitive artifact withheld')
            print(t,flush=True)
    if not complete:
        print(json.dumps(dict(stage='hud04_pair_failure_tail',text=clean('\n'.join((out/'child.log').read_text(errors='replace').splitlines()[-65:]))[-14000:])),flush=True);raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--expected-commit');main(p.parse_args())