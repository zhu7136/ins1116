"""One bounded GPU diagnostic. Default plan-only. No task API or old launcher execution."""
import argparse, hashlib, json, os, re, selectors, signal, subprocess, sys, time
from pathlib import Path
from contact_probe import evaluate,verify

LIMIT=240
MAX_LOG=8*1024*1024
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_new(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def validate_report(d,commit,probe_sha):
    if d.get('stage')!='hud04_controlled_wrist_contact_observation' or d.get('source_commit')!=commit or d.get('probe_sha256')!=probe_sha:raise ValueError('Report identity mismatch')
    if d.get('accepted_reference') is not False or d.get('training_ready') is not False or d.get('full_task_tested') is not False or d.get('graceful_full_cleanup_proven') is not False:raise ValueError('No acceptance or graceful full cleanup claim permitted')
    if d.get('body_dof')!=29 or d.get('head_action_dof')!=0 or d.get('explicit_physics_steps')!=8 or d.get('num_envs')!=3:raise ValueError('Diagnostic scope mismatch')
    if d.get('verdict')!=evaluate(d.get('measured_force_norms',{})):raise ValueError('Measured verdict mismatch')
    bindings=d.get('exact_joint_body_bindings',[])
    roles=('original_fallback','explicit_true','explicit_false')
    expected={(r,s) for r in roles for s in ('left','right')}
    if len(bindings)!=6 or {(b['role'],b['side']) for b in bindings}!=expected:raise ValueError('Exact bindings required')
    for b in bindings:
        root=f'/World/envs/env_{roles.index(b["role"])}/Robot';side=b['side']
        if b['joint']!=root+'/joints/'+side+'_wrist_roll_joint' or b['body0']!=[root+'/'+side+'_wrist_pitch_link'] or b['body1']!=[root+'/'+side+'_wrist_roll_link']:raise ValueError('Binding path mismatch')
        if b['before']!={'value':False,'authored':False} or b['observed_value'] is not (b['role']=='explicit_true') or b['observed_authored'] is not (b['role']!='original_fallback'):raise ValueError('Policy treatment mismatch')
    return True
def sanitize(text):
    for k in ('github1117','flux1117','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY'):
        v=os.environ.get(k)
        if v:text=text.replace(v,'[REDACTED]')
    text=re.sub(r'https?://\S+','[URL omitted]',text)
    text=re.sub(r'\b(?:gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|gm_sk_[A-Za-z0-9_]+)\b','[REDACTED]',text)
    return text
def run_owned(argv,out,env):
    start=time.monotonic();reason=None
    with (out/'child.log').open('xb') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,start_new_session=True,bufsize=0)
        sel=selectors.DefaultSelector();sel.register(child.stdout,selectors.EVENT_READ);eof=False;written=0
        try:
            while not eof or child.poll() is None:
                if time.monotonic()-start>=LIMIT:reason='owned_child_wall_deadline';break
                if not sel.select(.03):continue
                chunk=os.read(child.stdout.fileno(),65536)
                if not chunk:eof=True;sel.unregister(child.stdout);continue
                remaining=MAX_LOG-written;log.write(chunk[:remaining]);written+=min(remaining,len(chunk))
                if len(chunk)>remaining:reason='owned_child_log_budget';break
        finally:
            sel.close();child.stdout.close()
            for sig in (signal.SIGTERM,signal.SIGKILL):
                try:os.killpg(child.pid,sig)
                except ProcessLookupError:pass
                if sig==signal.SIGTERM:
                    try:child.wait(timeout=.3)
                    except subprocess.TimeoutExpired:pass
            code=child.wait(timeout=.5)
    return dict(return_code=code,stop_reason=reason,direct_child_reaped=True,child_wall_seconds=time.monotonic()-start,
        child_seconds_max=LIMIT,child_log_sha256=sha(out/'child.log'),platform_hard_deadline_verified=False)
def main(args):
    if not args.execute:
        print(json.dumps(dict(stage='plan_only',gpu_started=False,max_new_tasks=1,gpu_count=1,child_seconds_max=LIMIT,
            num_envs=3,explicit_physics_steps=8,automatic_retries=0,motion_solver_calls=0,ppo_updates=0,
            accepted_reference=False,training_ready=False)));return
    if not re.fullmatch('[0-9a-f]{40}',args.expected_commit or ''):raise ValueError('Full commit required')
    repo=Path(__file__).resolve().parents[4];verify(repo,args.expected_commit)
    out=repo/('hud04_controlled_pair_'+args.expected_commit[:12]);out.mkdir(exist_ok=False)
    env=os.environ.copy()
    for k in ('github1117','flux1117','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY'):env.pop(k,None)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PXR_WORK_THREAD_LIMIT'):env[k]='2'
    env['TMPDIR']=str(out)
    if __import__('importlib.util',fromlist=['find_spec']).find_spec('isaaclab') is not None:launcher=Path(sys.executable)
    else:launcher=next((p for p in (Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')) if p.is_file()),None)
    if launcher is None:raise RuntimeError('No established Isaac runtime; no installation/retry')
    print(json.dumps(dict(stage='hud04_pair_entry_started',expected_commit=args.expected_commit,child_seconds=LIMIT,automatic_retries=0)),flush=True)
    probe=Path(__file__).with_name('contact_probe.py')
    result=run_owned([str(launcher),str(probe),'--repo-root',str(repo),'--expected-commit',args.expected_commit,'--output',str(out)],out,env)
    report=None;error=None
    p=out/'contact_report.json'
    if p.is_file() and p.stat().st_size<=65536:
        try:report=json.loads(p.read_text());validate_report(report,args.expected_commit,sha(probe))
        except (ValueError,KeyError,TypeError) as exc:error=type(exc).__name__+': '+str(exc);report=None
    complete=result['return_code']==0 and result['stop_reason'] is None and report is not None
    result.update(stage='hud04_pair_supervisor_result',source_commit=args.expected_commit,controlled_probe_completed=complete,
        normal_native_exit_observed=result['return_code']==0 and result['stop_reason'] is None,
        fast_shutdown_not_graceful_full_cleanup=True,report_validation_error=error,
        accepted_reference=False,training_ready=False,motion_solver_calls=0,ppo_updates=0)
    write_new(out/'supervisor_result.json',result)
    for name in ('contact_report.json','contact_error.json','supervisor_result.json'):
        p=out/name
        if p.is_file() and p.stat().st_size<=65536:
            d=json.loads(p.read_text());text=json.dumps(dict(stage='hud04_pair_artifact',name=name,sha256=sha(p),payload=d))
            if sanitize(text)!=text:raise ValueError('Potential sensitive artifact; withheld')
            print(text,flush=True)
    if not complete:
        text=(out/'child.log').read_text(errors='replace')
        print(json.dumps(dict(stage='hud04_pair_failure_tail',text=sanitize('\n'.join(text.splitlines()[-65:]))[-14000:])),flush=True)
        raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--expected-commit');main(p.parse_args())