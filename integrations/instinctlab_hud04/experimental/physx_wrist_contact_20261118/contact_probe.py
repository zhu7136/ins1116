"""Controlled PhysX wrist A/B/C diagnostic; never reference acceptance or training."""
import argparse, faulthandler, hashlib, json, os, runpy, subprocess, sys, time, traceback
from pathlib import Path

MODEL_SHA='865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7'
ASSET_SHA='4fdb08b4aa352893a6feef5afb6deb8ec273e1fbdcfc2d67ac0e691729321021'
ROLES=('original_fallback','explicit_true','explicit_false')
SIDES=('left','right')
STEPS=8
DT=.005

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_new(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def emit(phase,**kw):print(json.dumps(dict(stage='hud04_wrist_contact_phase',phase=phase,**kw)),flush=True)

def evaluate(samples,instrumentation_review=None):
    """Strict measured evidence, not nominal-shape or all-articulation acceptance."""
    import math
    expected={'box_positive_control'}|{role+'_'+side for role in ROLES for side in SIDES}
    if set(samples)!=expected:raise ValueError('Exact seven measured contact channels required')
    for values in samples.values():
        if len(values)!=STEPS or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in values):raise ValueError('Invalid measured force series')
    peaks={k:max(v) for k,v in samples.items()}
    positive=peaks['box_positive_control']>1e-6
    if instrumentation_review is not None:positive=positive and instrumentation_review.get('positive_control_valid') is True
    verdict={side:(positive and peaks['explicit_true_'+side]>1e-6 and peaks['original_fallback_'+side]<=1e-9 and peaks['explicit_false_'+side]<=1e-9) for side in SIDES}
    return dict(peak_force_norm_native=peaks,independent_box_contact_positive=positive,
        controlled_wrist_body_pair_filter_supported=verdict,
        all_joint_pair_filters_verified=False,cooked_geometry_verified=False,
        force_norm_not_penetration_depth=True,source_motion_alignment_verified=False,
        reference_acceptance=False,accepted_reference=False,training_ready=False)

def verify(repo,commit):
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=5).strip()!=commit:raise ValueError('Checkout identity mismatch')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True,timeout=5).strip():raise ValueError('Tracked changes; never reset')
    asset=repo/'flux/hud04_asset_camera_probe'
    manifest=json.loads((asset/'manifest.json').read_text())
    for rel,sha in manifest['files'].items():
        p=(asset/rel).resolve()
        if not p.is_relative_to(asset.resolve()) or digest(p)!=sha:raise ValueError('Asset identity mismatch')
    model=asset/'humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
    if digest(model)!=MODEL_SHA or digest(asset/'asset_config.py')!=ASSET_SHA:raise ValueError('Pinned model/config mismatch')
    return asset,model

def main(args):
    repo=Path(args.repo_root).resolve();out=Path(args.output).resolve()
    asset,model=verify(repo,args.expected_commit)
    os.environ['HUMANOID_DESCRIPTION_DIR']=str(asset/'humanoid-description')
    os.environ['ENABLE_CAMERAS']='0'
    from isaaclab.app import AppLauncher
    # Deliberately test documented native fast shutdown instead of the failed
    # full destructor shutdown. Parent measures exit; no os._exit/exit masking.
    emit('before_app',fast_shutdown=True)
    app=AppLauncher(headless=True,enable_cameras=False,device='cuda:0',fast_shutdown=True).app
    faulthandler.enable();faulthandler.dump_traceback_later(90,repeat=True)
    sim=None
    try:
        import inspect
        import carb
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import ContactSensor,ContactSensorCfg
        from measured_task131.contact_processing_guard import enable_contact_processing
        from instrumentation_v2 import validate_instrumentation
        from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
        from isaaclab.utils import configclass
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd,UsdPhysics,PhysxSchema
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT,device='cuda:0'))
        contact_processing=enable_contact_processing(carb.settings.get_settings())
        cfg=runpy.run_path(str(asset/'asset_config.py'))['HUD04_CFG'].replace(prim_path='{ENV_REGEX_NS}/Robot')
        # All three copies use identical no-gravity conditions. This is a
        # repeated-state collision diagnostic, NOT a standing/motion trial.
        cfg.spawn.rigid_props.disable_gravity=True
        cfg.spawn.usd_dir=str(out/'conversion_cache')
        @configclass
        class SceneCfg(InteractiveSceneCfg):robot=cfg
        scene=InteractiveScene(SceneCfg(num_envs=3,env_spacing=5.,replicate_physics=False,filter_collisions=True))
        stage=scene.stage
        bindings=[]
        for env,role in enumerate(ROLES):
            for side in SIDES:
                root=f'/World/envs/env_{env}/Robot'
                joint=UsdPhysics.Joint(stage.GetPrimAtPath(root+'/joints/'+side+'_wrist_roll_joint'))
                if not joint or not joint.GetPrim().IsValid():raise ValueError('Exact wrist joint missing')
                b0=[str(x) for x in joint.GetBody0Rel().GetTargets()];b1=[str(x) for x in joint.GetBody1Rel().GetTargets()]
                if b0!=[root+'/'+side+'_wrist_pitch_link'] or b1!=[root+'/'+side+'_wrist_roll_link']:raise ValueError('Exact joint-body binding mismatch')
                attr=joint.GetCollisionEnabledAttr();before=dict(value=attr.Get(),authored=attr.HasAuthoredValueOpinion())
                if before!={'value':False,'authored':False}:raise ValueError('Original fallback changed')
                if role!='original_fallback':attr.Set(role=='explicit_true')
                for body in b0+b1:
                    api=PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(body))
                    api.CreateThresholdAttr().Set(0.)
                bindings.append(dict(role=role,side=side,joint=str(joint.GetPath()),body0=b0,body1=b1,before=before,
                    observed_value=attr.Get(),observed_authored=attr.HasAuthoredValueOpinion()))
        boxcfg=sim_utils.CuboidCfg(size=(.1,.1,.1),rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.),collision_props=sim_utils.CollisionPropertiesCfg(),activate_contact_sensors=True)
        boxcfg.func('/World/contact_box_A',boxcfg,translation=(25.,0.,1.))
        boxcfg.func('/World/contact_box_B',boxcfg,translation=(25.08,0.,1.))
        for path in ('/World/contact_box_A','/World/contact_box_B'):
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr().Set(0.)
        box_sensor=ContactSensor(ContactSensorCfg(prim_path='/World/contact_box_A',
            filter_prim_paths_expr=['/World/contact_box_B'],update_period=0.,history_length=0,
            track_pose=True,track_contact_points=True,max_contact_data_count_per_prim=256,debug_vis=False))
        emit('before_physics_reset')
        sim.reset();scene.reset()
        robot=scene['robot'];names=list(robot.joint_names)
        if robot.is_fixed_base or len(names)!=29 or len(set(names))!=29 or any('head' in n for n in names):raise ValueError('Runtime body/head contract failed')
        view=SimulationManager.get_physics_sim_view()
        channels={}
        for b in bindings:
            key=b['role']+'_'+b['side']
            rv=view.create_rigid_body_view(b['body0'][0])
            cv=view.create_rigid_contact_view(b['body0'][0],filter_patterns=b['body1'],max_contact_data_count=256)
            if rv.count!=1 or cv.filter_count!=1:raise ValueError('Exact singleton contact channel unresolved')
            channels[key]=cv
        control=view.create_rigid_contact_view('/World/contact_box_A',filter_patterns=['/World/contact_box_B'],max_contact_data_count=256)
        if control.filter_count!=1:raise ValueError('Box control unresolved')
        channels['box_positive_control']=control
        box_views={s:view.create_rigid_body_view('/World/contact_box_'+s) for s in ('A','B')}
        if any(v.count!=1 for v in box_views.values()):raise ValueError('Box state view not singleton')
        settings_after_reset=carb.settings.get_settings().get_as_bool('/physics/disableContactProcessing')
        if settings_after_reset is not False:raise ValueError('Contact processing disabled after reset')
        box_trace=[];contact_counts={k:[] for k in channels};sensor_vectors=[]
        samples={k:[] for k in channels}
        root=robot.data.default_root_state.clone();root[:,:3]+=scene.env_origins
        q=robot.data.default_joint_pos.clone();qd=torch.zeros_like(q)
        emit('contact_channels_ready',channels=len(channels),steps=STEPS)
        for i in range(STEPS):
            # Reset original diagnostic pose identically for each role each step.
            # These states are never exported or passed off as reference motion.
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(q,qd)
            robot.set_joint_position_target(q)
            # Native tensor transforms are xyzw, unlike the articulation's wxyz.
            # Recreate measurable overlap and approaching velocity every step;
            # never rely on the pre-reset spawn overlap remaining unresolved.
            box_before={}
            for s,x,vx in (('A',25.,.1),('B',25.08,-.1)):
                rv=box_views[s];indices=torch.tensor([0],device=sim.device,dtype=torch.int32)
                pose=torch.tensor([[x,0.,1.,0.,0.,0.,1.]],device=sim.device,dtype=torch.float32)
                velocity=torch.tensor([[vx,0.,0.,0.,0.,0.]],device=sim.device,dtype=torch.float32)
                rv.set_transforms(pose,indices=indices);rv.set_velocities(velocity,indices=indices)
                box_before[s]={'pose_xyzw':rv.get_transforms().clone().cpu().tolist()[0],
                    'velocity':rv.get_velocities().clone().cpu().tolist()[0]}
            scene.write_data_to_sim();sim.step(render=False);scene.update(DT)
            box_sensor.update(DT,force_recompute=True)
            sensor_matrix=box_sensor.data.force_matrix_w
            if tuple(sensor_matrix.shape)!=(1,1,1,3):raise ValueError('Official box sensor layout mismatch')
            sensor_vectors.append(sensor_matrix[0,0,0].clone().cpu().tolist())
            for name,cv in channels.items():
                matrix=cv.get_contact_force_matrix(dt=DT)
                if tuple(matrix.shape)!=(1,1,3):raise ValueError('Unexpected exact body-pair matrix shape '+str(tuple(matrix.shape)))
                samples[name].append(float(torch.linalg.vector_norm(matrix).item()))
                data=cv.get_contact_data(dt=DT)
                if len(data)!=6 or tuple(data[4].shape)!=(1,1):raise ValueError('Detailed contact count layout mismatch')
                contact_counts[name].append(int(data[4][0,0].item()))
            box_trace.append({'step':i,'before':box_before,
                'after':{s:{'pose_xyzw':rv.get_transforms().clone().cpu().tolist()[0],
                    'velocity':rv.get_velocities().clone().cpu().tolist()[0]} for s,rv in box_views.items()},
                'native_force_vector':control.get_contact_force_matrix(dt=DT)[0,0].clone().cpu().tolist()})
        instrumentation={'revision':2,'contact_processing':contact_processing,
            'disable_contact_processing_after_reset':settings_after_reset,
            'box_size':[.1,.1,.1],'box_state_trace':box_trace,'native_contact_counts':contact_counts,
            'official_sensor_force_vectors':sensor_vectors,'contact_capacity_per_channel':256,
            'box_threshold_zero_requested':True}
        instrumentation_review=validate_instrumentation(instrumentation,samples)
        verdict=evaluate(samples,instrumentation_review)
        report=dict(stage='hud04_controlled_wrist_contact_observation',source_commit=args.expected_commit,
            model_sha256=digest(model),asset_config_sha256=digest(asset/'asset_config.py'),
            probe_sha256=digest(Path(__file__)),python_version=sys.version.split()[0],usd_version=list(Usd.GetVersion()),
            contact_sensor_official_implementation_sha256=digest(Path(inspect.getfile(__import__('isaaclab.sensors.contact_sensor.contact_sensor',fromlist=['ContactSensor']).ContactSensor))),
            exact_joint_body_bindings=bindings,roles=list(ROLES),num_envs=3,body_dof=29,head_action_dof=0,
            dt=DT,explicit_physics_steps=STEPS,gravity_disabled_in_isolated_copies=True,repeated_state_diagnostic_not_trajectory=True,
            self_collision_requested=True,replicate_physics=False,measured_force_norms=samples,verdict=verdict,
            instrumentation=instrumentation,instrumentation_review=instrumentation_review,
            full_task_tested=False,terrain_tested=False,fast_shutdown_requested=True,
            graceful_full_cleanup_proven=False,accepted_reference=False,training_ready=False)
        if digest(model)!=MODEL_SHA or digest(asset/'asset_config.py')!=ASSET_SHA:raise ValueError('Original asset changed')
        write_new(out/'contact_report.json',report)
        emit('measurement_written',verdict=verdict)
    except BaseException as exc:
        write_new(out/'contact_error.json',dict(stage='contact_probe_error',error_type=type(exc).__name__,message=str(exc)[:2000],
            frames=traceback.format_tb(exc.__traceback__)[-8:],accepted_reference=False,training_ready=False))
        raise
    finally:
        emit('before_native_fast_shutdown')
        faulthandler.cancel_dump_traceback_later()
        if sim is not None:
            sim.clear_all_callbacks();sim.clear_instance()
        app.close()
        emit('native_close_returned')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo-root',required=True)
    p.add_argument('--expected-commit',required=True);p.add_argument('--output',required=True);main(p.parse_args())