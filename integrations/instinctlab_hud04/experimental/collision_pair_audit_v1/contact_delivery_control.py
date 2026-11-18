"""One isolated contact-delivery diagnostic, NOT HUD04/reference validation.

Default is plan-only. Child has no platform credentials. Supervisor has one
285s watchdog and no retries; this is NOT a service/billing deadline.
"""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import traceback
import zipfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
CHILD_SECONDS=285
MAX_STEPS=8
REGISTRY=[dict(key='isolated_sphere_pair',actor0='/World/ContactDeliveryControl/Dynamic',
    collider0='/World/ContactDeliveryControl/Dynamic/Collision',
    actor1='/World/ContactDeliveryControl/Kinematic',
    collider1='/World/ContactDeliveryControl/Kinematic/Collision')]


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def emit(phase,**kw):
    print(json.dumps(dict(stage='hud04_contact_delivery_control',phase=phase,**kw)),flush=True)


def verify(commit):
    if not commit or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Full immutable commit required')
    actual=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True,timeout=5).strip()
    if actual!=commit:raise ValueError('Source commit mismatch')
    dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True,timeout=5)
    if dirty.strip():raise ValueError('Tracked source modified; do not reset')


def delivery_decision(contacts,phases):
    points=contacts['matched_contact_point_counts'].get('isolated_sphere_pair',0)
    return dict(diagnostic_delivery_observed=bool(points>0 and not contacts.get('binding_error')
        and not contacts.get('error') and not phases['error'] and phases['completed_steps']==MAX_STEPS),
        matched_diagnostic_contact_points=points,
        robot_support_validated=False,effective_pair_friction=None,accepted_reference=False,training_ready=False)


def child(out,commit):
    verify(commit)
    os.environ['ENABLE_CAMERAS']='0'
    from isaaclab.app import AppLauncher
    emit('initializing',child_watchdog_s=CHILD_SECONDS)
    app=AppLauncher(headless=True,enable_cameras=False,device='cuda:0',fast_shutdown=True).app
    sim=None;holder=None;capture=None
    try:
        import carb
        import omni.physx
        import omni.usd
        import torch
        import isaaclab.sim as sim_utils
        from omni.physx.bindings import _physx
        from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdUtils,PhysicsSchemaTools,Gf
        from contact_report_1073 import ContactReport1073
        from prepare_runtime_observation import configured_scene_path

        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device='cuda:0'))
        stage=omni.usd.get_context().get_stage()
        scene_path=configured_scene_path(sim.cfg)
        meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage)
        if meters_per_unit!=1.:raise ValueError('Diagnostic geometry requires verified meter stage units')
        scene_prim=stage.GetPrimAtPath(scene_path)
        if not scene_prim.IsValid() or not scene_prim.IsA(UsdPhysics.Scene):
            raise ValueError('Configured physics Scene unavailable or wrong schema')
        UsdGeom.Xform.Define(stage,'/World/ContactDeliveryControl')
        readback=[]
        for actor,collider,z,kinematic in [
            (REGISTRY[0]['actor0'],REGISTRY[0]['collider0'],.2,False),
            (REGISTRY[0]['actor1'],REGISTRY[0]['collider1'],.105,True)]:
            body=UsdGeom.Xform.Define(stage,actor)
            body.AddTranslateOp().Set(Gf.Vec3d(0,0,z))
            rb=UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            rb.CreateRigidBodyEnabledAttr(True)
            rb.CreateKinematicEnabledAttr(kinematic)
            mass=UsdPhysics.MassAPI.Apply(body.GetPrim());mass.CreateMassAttr(1.)
            reporter=PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim())
            reporter.CreateThresholdAttr(0.)
            sphere=UsdGeom.Sphere.Define(stage,collider);sphere.CreateRadiusAttr(.05)
            collision=UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            collision.CreateCollisionEnabledAttr(True)
            readback.append(dict(actor=actor,collider=collider,kinematic=rb.GetKinematicEnabledAttr().Get(),
                radius_m=sphere.GetRadiusAttr().Get(),translation_m=[0,0,z],
                collision_enabled=collision.GetCollisionEnabledAttr().Get(),
                contact_report_threshold=reporter.GetThresholdAttr().Get(),
                mass_kg=mass.GetMassAttr().Get()))
        stage_id=int(UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
        if stage_id<=0:raise ValueError('Host stage not in cache before initialization; no guessed ID')
        settings=carb.settings.get_settings()
        processing_key='/physics/disableContactProcessing'
        processing_before=settings.get_as_bool(processing_key)
        settings.set_bool(processing_key,False)
        processing_after=settings.get_as_bool(processing_key)
        if processing_after is not False:raise ValueError('Diagnostic contact processing not enabled')
        write(out/'metadata.json',dict(source_commit=commit,stage_id=stage_id,
            configured_scene_path=scene_path,configured_scene_valid_and_typed=True,
            solver_scene_ownership_independently_verified=False,
            scene_source='HOST_sim.cfg.physics_prim_path',hardcoded_fallback_used=False,
            diagnostic_geometry_readback=readback,initial_analytical_overlap_m=.005,
            robot_present=False,hud04_acceptance_model_modified=False,
            stage_meters_per_unit=meters_per_unit,
            diagnostic_only_contact_processing=dict(key=processing_key,before=processing_before,after=processing_after)))
        phases=dict(current=None,records=[],error=None,completed_steps=0)

        class Capture(ContactReport1073):
            def callback(self,*args):
                label=phases['current']
                if label is None:phases['error']='Callback outside explicit initialization/step window'
                if time.monotonic()-self.started>30:phases['error']='Observer wall budget exceeded'
                if len(phases['records'])>=64:
                    phases['error']='Phase record budget exceeded';return
                before=len(self.observer.events)
                super().callback(*args)
                phases['records'].append(dict(phase=label,event_begin=before,event_end=len(self.observer.events)))

        capture=Capture(REGISTRY,PhysicsSchemaTools,expected_stage_id=stage_id,max_events=64,max_points=512)
        capture.started=time.monotonic()
        phases['current']='initialization'
        # Retain subscription BEFORE reset/play can produce contact reports.
        holder=capture.subscribe_existing(omni.physx.get_physx_simulation_interface())
        if holder is None:raise ValueError('Native subscription holder missing')
        sim.reset()
        processing_after_reset=settings.get_as_bool(processing_key)
        if processing_after_reset is not False:raise ValueError('Diagnostic contact processing disabled after reset')
        phases['current']=None
        for step in range(MAX_STEPS):
            if time.monotonic()-capture.started>30:raise TimeoutError('Observer budget; not a platform deadline')
            if not app.is_running():raise RuntimeError('Application stopped before explicit step budget')
            phases['current']='step_'+str(step)
            sim.step(render=False)
            phases['current']=None
            phases['completed_steps']+=1
            if capture.error or capture.observer.error or phases['error']:break
        contacts=capture.report()
        write(out/'contacts.json',contacts)
        write(out/'phases.json',phases)
        decision=delivery_decision(contacts,phases)
        native_file=getattr(_physx,'__file__',None)
        report=dict(source_commit=commit,image_reference='BJX00000178/V000446',
            stage='isolated_contact_delivery_control_completed',python=sys.version.split()[0],
            torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu_name=torch.cuda.get_device_name(0),
            usd_version=list(Usd.GetVersion()),actual_physx_build_version=getattr(_physx,'__version__',None),
            native_library_sha256=hashlib.sha256(Path(native_file).read_bytes()).hexdigest() if native_file and Path(native_file).is_file() else None,
            stage_id=stage_id,explicit_physics_steps=phases['completed_steps'],initialization_may_step_physics=True,
            callback_invocations=contacts['callback_invocations'],contact_points=contacts['contact_points_seen'],
            disable_contact_processing_after_reset=processing_after_reset,
            binding_error=contacts['binding_error'],ingestion_error=contacts['error'],phase_error=phases['error'],
            robot_present=False,camera_enabled=False,motion_files_used=0,solver_calls=0,ppo_updates=0,
            platform_hard_cap_verified=False,**decision)
        write(out/'runtime_report.json',report)
        emit('observation_saved',callbacks=report['callback_invocations'],**decision)
    except BaseException as exc:
        write(out/'runtime_error.json',dict(type=type(exc).__name__,message=str(exc)[:1500],
            frames=traceback.format_tb(exc.__traceback__)[-8:]))
        raise
    finally:
        holder=None
        emit('cleanup_started')
        if sim is not None:
            sim.clear_all_callbacks();sim.clear_instance()
        app.close(wait_for_replicator=False)
        emit('cleanup_returned')


def supervisor(out,commit):
    verify(commit);out.mkdir(exist_ok=False)
    if importlib.util.find_spec('isaaclab') is not None:launcher=Path(sys.executable)
    else:launcher=next((p for p in [Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')] if p.is_file()),None)
    if launcher is None:raise RuntimeError('Isaac interpreter absent; no install/retry')
    env=os.environ.copy()
    for key in ('ins1116','flux_key1103','limxanzhuangbao','FLUX_API_KEY','WANDB_API_KEY','OPENAI_API_KEY'):
        env.pop(key,None)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PXR_WORK_THREAD_LIMIT'):env[key]='2'
    env['TMPDIR']=str(out)
    start=time.monotonic()
    proc=subprocess.Popen([str(launcher),str(Path(__file__).resolve()),'--child','--expected-commit',commit,
        '--output',str(out)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    reason=None;written=0;heartbeat=-30
    with (out/'child.log').open('xb') as stream:
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        try:
            eof=False
            while not eof or proc.poll() is None:
                age=time.monotonic()-start
                if age-heartbeat>=20:emit('heartbeat',elapsed_s=round(age,1));heartbeat=age
                if age>=CHILD_SECONDS:reason='child_wall_deadline';break
                if not selector.select(.05):continue
                data=os.read(proc.stdout.fileno(),65536)
                if not data:eof=True;selector.unregister(proc.stdout);continue
                if written+len(data)>8*1024*1024:reason='log_budget';break
                stream.write(data);written+=len(data)
        finally:
            selector.close();proc.stdout.close()
            for sig in (signal.SIGTERM,signal.SIGKILL):
                try:os.killpg(proc.pid,sig)
                except ProcessLookupError:pass
                if sig==signal.SIGTERM:
                    try:proc.wait(timeout=.5)
                    except subprocess.TimeoutExpired:pass
            code=proc.wait(timeout=1)
    result=dict(source_commit=commit,return_code=code,stop_reason=reason,child_wall_s=time.monotonic()-start,
        child_reaped=True,child_watchdog_s=CHILD_SECONDS,automatic_retries=0,platform_hard_cap_verified=False,
        service_300s_prerequisite_waived_for_this_trial_only=True,ppo_updates=0,training_ready=False)
    write(out/'worker_result.json',result)
    names=['runtime_report.json','metadata.json','contacts.json','phases.json','runtime_error.json','worker_result.json']
    files={n:(out/n).read_bytes() for n in names if (out/n).is_file()}
    if sum(map(len,files.values()))>1_000_000:raise ValueError('Expanded output budget exceeded')
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name,raw in files.items():json.loads(raw);bundle.writestr(name,raw)
    raw=memory.getvalue()
    if len(raw)>512_000:raise ValueError('Compressed output budget exceeded')
    encoded=base64.b64encode(raw).decode();chunks=[encoded[i:i+8192] for i in range(0,len(encoded),8192)]
    sha=hashlib.sha256(raw).hexdigest()
    emit('evidence_manifest',sha256=sha,zip_bytes=len(raw),chunks=len(chunks),
        file_sha256={n:hashlib.sha256(v).hexdigest() for n,v in files.items()})
    for index,data in enumerate(chunks):emit('evidence_chunk',sha256=sha,index=index,data=data)
    emit('worker_finished',**result)
    # A completed negative/incompatible diagnostic must remain a failed test.
    if code or reason or not json.loads(files.get('runtime_report.json',b'{}')).get('diagnostic_delivery_observed'):
        raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--child',action='store_true')
    parser.add_argument('--expected-commit');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.execute or args.child:
        if not args.expected_commit or not args.output or not args.output.is_absolute():
            parser.error('Execution requires immutable commit and absolute unique output')
        (child if args.child else supervisor)(args.output,args.expected_commit)
    else:emit('plan_only',gpu_count=1,child_watchdog_s=CHILD_SECONDS,automatic_retries=0,robot_present=False,
        platform_hard_cap_verified=False,ppo_updates=0,accepted_reference=False,training_ready=False)