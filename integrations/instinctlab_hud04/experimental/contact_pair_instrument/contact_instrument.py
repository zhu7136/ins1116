"""Independent contact-instrument calibration candidate; NOT a HUD04 motion.

Default is plan only. Explicit execution creates four known-mass boxes and a
local triangle Mesh, never imports a reference bundle or registers a task.
Real API compatibility and all physical checks remain UNVALIDATED until run.
"""
import argparse
import hashlib
import json
from pathlib import Path

DT = .005
GRAVITY = 9.81
CAPACITY = 128
PHASES = [('settle', 160), ('lift_env1', 20), ('restore', 80)]


def plan():
    return dict(mode='plan_only', gpu_started=False, physics_dt_s=DT,
                physics_steps_max=sum(n for _, n in PHASES), environments=2,
                bodies_per_environment=2, masses_kg=[1., 2.], capacity_per_view=CAPACITY,
                reference_loaded=False, reference_exported=False, solver_calls=0,
                ppo_updates=0, accepted_reference=False, training_ready=False,
                real_physics_validated=False, platform_submission_performed=False,
                NOT_TESTED=['HUD04 collision-body mapping', 'fixed head',
                            'articulation self collision', 'actual task mesh cooking',
                            'actual environment collision-group isolation',
                            'substep collector integration', 'HUD04 contact thresholds'])


def check_balance(rows, relative_tolerance=.15):
    """rows: [env][body][terrain scalar sum, nonterrain scalar sum, net Z].

    The lower box supports both masses. Equal opposing normal forces on it
    partly cancel in the net force. Compare each channel, not only the net.
    Tolerance is predeclared ONLY for this synthetic known-mass instrument.
    """
    import math
    if len(rows)!=2 or any(len(env)!=2 for env in rows):
        raise ValueError('Expected two environments with lower/upper boxes')
    expected = [[3*GRAVITY, 2*GRAVITY, GRAVITY], [0., 2*GRAVITY, 2*GRAVITY]]
    errors=[]
    for env in rows:
        for actual, wanted in zip(env, expected):
            if len(actual)!=3 or not all(math.isfinite(v) for v in actual):
                raise ValueError('Nonfinite or malformed balance sample')
            for value, target in zip(actual,wanted):
                error=abs(value-target)
                if error>max(.25,abs(target)*relative_tolerance):
                    raise RuntimeError('Known-mass normal-force balance failed')
                errors.append(error)
    return max(errors)


def event(stage, **values):
    print(json.dumps(dict(stage=stage, **values), allow_nan=False), flush=True)


def execute(output):
    output=Path(output)
    output.mkdir(exist_ok=False)
    report={**plan(), 'mode':'independent_physics_instrument', 'gpu_started':False,
            'instrument_passed':False, 'real_physics_validated':False,
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    app=None;sim=None
    try:
        event('instrument_initialization_started')
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True, enable_cameras=False, device='cuda:0').app
        import torch
        import carb
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
        from terrain_contact_bridge import _inventory, make_pair_plan, reduce_contact_points
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT,device='cuda:0',gravity=(0.,0.,-GRAVITY)))
        stage=sim_utils.get_current_stage()
        report['gpu_started']=True
        report['stage_meters_per_unit']=UsdGeom.GetStageMetersPerUnit(stage)
        report['stage_up_axis']=str(UsdGeom.GetStageUpAxis(stage))
        if report['stage_meters_per_unit']!=1. or report['stage_up_axis']!='Z':
            raise RuntimeError('Metre/Z-up stage required')
        carb.settings.get_settings().set_bool('/physics/disableContactProcessing',False)
        UsdGeom.Xform.Define(stage,'/World/ground')
        mesh=UsdGeom.Mesh.Define(stage,'/World/ground/mesh')
        mesh.CreatePointsAttr([(-4.,-4.,0.),(4.,-4.,0.),(4.,4.,0.),(-4.,4.,0.)])
        mesh.CreateFaceVertexCountsAttr([3,3]);mesh.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
        mesh.CreateSubdivisionSchemeAttr('none')
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
        paths=['/World/envs/env_0','/World/envs/env_1'];names=['lower','upper'];objects=[]
        for i,path in enumerate(paths):
            UsdGeom.Xform.Define(stage,path+'/Robot')
            for name,mass,z in [('lower',1.,.251),('upper',2.,.752)]:
                cfg=RigidObjectCfg(prim_path=path+'/Robot/'+name,
                    spawn=sim_utils.CuboidCfg(size=(.5,.5,.5),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                        collision_props=sim_utils.CollisionPropertiesCfg()),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.5+3*i,0.,z)))
                objects.append(RigidObject(cfg))
                prim=stage.GetPrimAtPath(path+'/Robot/'+name)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        sim.reset()
        for obj in objects:
            obj.reset()
            if not obj.data.root_state_w.is_cuda:raise RuntimeError('CUDA state required')
        inventory=_inventory(stage,Usd,UsdPhysics,PhysxSchema)
        pair_plan=make_pair_plan(inventory,paths,names)
        physics=SimulationManager.get_physics_sim_view();views=[]
        for row in pair_plan:
            body=physics.create_rigid_body_view(row['sensor'])
            if list(body.prim_paths)!=[row['sensor']]:raise RuntimeError('Exact owner path unresolved')
            views.append(physics.create_rigid_contact_view(row['sensor'],
                filter_patterns=row['filters'],max_contact_data_count=CAPACITY))
        report['collision_inventory']=inventory
        report['pair_plan']=pair_plan
        event('instrument_initialized', views=len(views))
        summaries={};ticks=0
        for phase,steps in PHASES:
            if phase in ('lift_env1','restore'):
                for i,obj in enumerate(objects):
                    state=obj.data.default_root_state.clone()
                    if phase=='lift_env1' and i>=2:state[:,2]+=2.
                    state[:,7:]=0.
                    obj.write_root_state_to_sim(state);obj.reset()
            samples=[]
            for step in range(steps):
                sim.step()
                for obj in objects:obj.update(DT)
                ticks+=1
                values=torch.zeros((2,2,3),device='cuda:0')
                for row,view in zip(pair_plan,views):
                    raw=view.get_contact_data(dt=DT)
                    matrix=view.get_contact_force_matrix(dt=DT)
                    net=view.get_net_contact_forces(dt=DT)
                    reduced=reduce_contact_points(raw,matrix,net,CAPACITY)
                    values[row['env_id'],row['body_id']]=torch.stack([
                        reduced['terrain_normal_force_N'][0],
                        reduced['nonterrain_normal_force_N'][0],net[0,2]])
                if step>=steps-10:samples.append(values)
            mean=torch.stack(samples).mean(0).cpu().tolist()
            if phase in ('settle','restore'):
                error=check_balance(mean)
                summaries[phase]=dict(mean_channels=mean,max_balance_error_N=error)
            else:
                # Separation test ONLY; does not prove actual cloned-env group filtering.
                peak=float(torch.stack(samples)[:,1,:,:].abs().max().item())
                if peak>.25:raise RuntimeError('Lifted environment retained contact or stale data')
                summaries[phase]=dict(mean_channels=mean,lifted_contact_peak_N=peak)
            if _inventory(stage,Usd,UsdPhysics,PhysxSchema)!=inventory:
                raise RuntimeError('Collision census changed')
            event('instrument_phase_completed',phase=phase,physics_steps=ticks)
        report.update(instrument_passed=True,real_physics_validated=True,
                      physics_steps_completed=ticks,phase_summaries=summaries)
    except Exception as error:
        report['error_type']=type(error).__name__
        report['error']=str(error)[:1000]
        raise
    finally:
        # A killed process cannot claim cleanup or successful lifecycle.
        report['normal_cleanup_completed']=False
        try:
            event('instrument_shutdown_started')
            if sim is not None:
                sim.stop();sim.clear_all_callbacks();sim.clear_instance()
            if app is not None:app.close()
            report['normal_cleanup_completed']=app is not None
        finally:
            with (output/'report.json').open('x') as stream:
                json.dump(report,stream,indent=2,allow_nan=False)
        event('instrument_cleanup_completed',completed=report['normal_cleanup_completed'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-instrument',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.execute_instrument:event('plan_only',**plan())
    elif args.output is None:parser.error('--output is required for explicit execution')
    else:execute(args.output)