"""Two isolated normal-shutdown controls; no robot, sensor, policy, or training.

Default is CPU-only plan. One explicitly started worker runs app-only and
empty-physics children sequentially, never in parallel, with no retries.
Worker deadlines do not establish a platform billing limit.
"""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
from worker import diagnostic_tail,stop_child

ROOT=Path(__file__).resolve().parent
MODES=('app-only','empty-physics')

def event(stage,**data):
    print(json.dumps(dict(stage=stage,**data)),flush=True)

def phases(text):
    result=[]
    decoder=json.JSONDecoder()
    for line in text.splitlines():
        start=line.find('{"stage": "lifecycle_')
        if start<0:continue
        try:item,_=decoder.raw_decode(line[start:])
        except ValueError:continue
        if isinstance(item,dict):result.append(item)
    return result

def deadline(age,initialized,close_age):
    if age>=120:return 'child_wall_limit'
    if close_age is not None and close_age>=30:return 'normal_shutdown_deadline'
    if age>=90 and not initialized:return 'app_initialization_deadline'
    return None

def normal_process_exit(reason,code,stages,mode,physics):
    # SimulationApp.close may terminate the interpreter successfully without
    # returning to Python; an after-close marker is not required in that case.
    return (reason is None and code==0 and 'lifecycle_app_initialized' in stages
            and 'lifecycle_app_closing' in stages
            and (mode=='app-only' or physics is not None))

def child(mode):
    os.environ['ENABLE_CAMERAS']='0'
    event('lifecycle_app_initializing',mode=mode)
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,enable_cameras=False,device='cuda:0')
    app=launcher.app
    event('lifecycle_app_initialized',mode=mode)
    sim=None
    try:
        if mode=='empty-physics':
            import torch
            import isaaclab.sim as sim_utils
            event('lifecycle_physics_initializing',mode=mode)
            sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device='cuda:0'))
            sim.reset()
            for _ in range(3):sim.step(render=False)
            event('lifecycle_physics_completed',mode=mode,steps=3,
                  simulation_device=str(sim.device),cuda_available=torch.cuda.is_available())
    finally:
        if sim is not None:
            # IsaacLab's standalone STOP subscription can wait for playback
            # during app.close. Public lifecycle API unsubscribes this callback.
            event('lifecycle_context_cleanup_started',mode=mode)
            sim.clear_instance()
            event('lifecycle_context_cleanup_completed',mode=mode)
        event('lifecycle_app_closing',mode=mode,wait_for_replicator=False)
        app.close(wait_for_replicator=False)
        event('lifecycle_app_close_returned',mode=mode)
    event('lifecycle_child_main_completed',mode=mode)

def case(mode,launcher,out):
    path=out/(mode+'.log')
    started=time.monotonic();closing=None;initialized=False;last=-15.;proc=None;reason=None;seen=[]
    command=[str(launcher),str(Path(__file__).resolve()),'--execute-probe','--child',mode]
    try:
        with path.open('x') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                age=time.monotonic()-started
                with path.open('rb') as reader:
                    reader.seek(max(0,path.stat().st_size-65536));tail=reader.read(65536).decode('utf-8',errors='replace')
                seen=phases(tail)
                initialized=initialized or any(e['stage']=='lifecycle_app_initialized' for e in seen)
                if closing is None and any(e['stage']=='lifecycle_app_closing' for e in seen):closing=age
                if age-last>=15:
                    event('lifecycle_heartbeat',mode=mode,age_s=round(age,1),
                          last_phase=seen[-1]['stage'] if seen else None)
                    last=age
                reason=deadline(age,initialized,None if closing is None else age-closing)
                if reason:break
                time.sleep(.2)
            if reason:stop_child(proc)
            code=proc.wait(timeout=5)
    finally:
        if proc is not None:stop_child(proc)
    with path.open('rb') as reader:
        reader.seek(max(0,path.stat().st_size-65536));tail=reader.read(65536).decode('utf-8',errors='replace')
    seen=phases(tail);stages=[x['stage'] for x in seen]
    closed='lifecycle_app_close_returned' in stages
    completed='lifecycle_child_main_completed' in stages
    physics=next((x for x in seen if x['stage']=='lifecycle_physics_completed'),None)
    passed=normal_process_exit(reason,code,stages,mode,physics)
    result=dict(mode=mode,passed=passed,return_code=code,stop_reason=reason,
                wall_s=round(time.monotonic()-started,3),app_initialized=initialized,
                normal_close_returned=closed,child_main_completed=completed,
                normal_process_exit_verified=passed,
                context_cleanup_completed='lifecycle_context_cleanup_completed' in stages,
                last_phase=stages[-1] if stages else None,physics=physics,
                training_updates=0,robot_loaded=False,sensor_loaded=False)
    event('lifecycle_case_result',**result)
    if not passed:event('lifecycle_failure_tail',mode=mode,text=diagnostic_tail(tail))
    return result

def main(args):
    if not args.execute_probe:
        event('lifecycle_plan',gpu_started=False,modes=list(MODES),training_updates=0,
              child_wall_cap_s=120,initialization_cap_s=90,shutdown_cap_s=30,
              automatic_retries=0,worker_limit_is_not_billing_cap=True)
        return
    if args.child:
        child(args.child);return
    from preflight import audit
    audit(ROOT)
    output=ROOT/'lifecycle-output';output.mkdir(exist_ok=False)
    candidates=[]
    if importlib.util.find_spec('isaaclab') is not None:candidates.append(Path(sys.executable))
    candidates.extend([Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')])
    launcher=next((p for p in candidates if p.is_file()),None)
    if launcher is None:raise RuntimeError('Verified Isaac Python launcher not found; no install attempted')
    results=[case(mode,launcher,output) for mode in MODES]
    data=dict(kind='empty-scene shutdown controls',cases=results,passed=all(x['passed'] for x in results),
              training_updates=0,training_ready=False,automatic_retries=0)
    raw=json.dumps(data,sort_keys=True,separators=(',',':')).encode()
    (output/'lifecycle-result.json').write_bytes(raw)
    event('lifecycle_result_evidence',sha256=hashlib.sha256(raw).hexdigest(),data=data)
    if not data['passed']:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-probe',action='store_true')
    parser.add_argument('--child',choices=MODES)
    main(parser.parse_args())
