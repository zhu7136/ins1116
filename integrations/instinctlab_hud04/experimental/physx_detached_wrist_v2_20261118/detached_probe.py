"""NEW detached original wrist-shape contact control. NOT motion acceptance."""
import argparse,hashlib,json,runpy,sys,traceback
from pathlib import Path
STEPS=8;DT=.005
V2='integrations/instinctlab_hud04/experimental/physx_wrist_contact_20261118'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def main(a):
    repo=Path(a.repo_root);out=Path(a.output);sys.path.insert(0,str(repo/V2))
    from contact_probe import verify,MODEL_SHA,ASSET_SHA
    asset,model=verify(repo,a.expected_commit)
    import os
    os.environ['HUMANOID_DESCRIPTION_DIR']=str(asset/'humanoid-description');os.environ['ENABLE_CAMERAS']='0'
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=False,device='cuda:0',fast_shutdown=True).app
    sim=None
    try:
        import carb,torch
        import isaaclab.sim as su
        from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
        from isaaclab.utils import configclass
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema
        from measured_task131.contact_processing_guard import enable_contact_processing
        sim=su.SimulationContext(su.SimulationCfg(dt=DT,device='cuda:0'))
        processing=enable_contact_processing(carb.settings.get_settings())
        cfg=runpy.run_path(str(asset/'asset_config.py'))['HUD04_CFG'].replace(prim_path='{ENV_REGEX_NS}/Robot')
        cfg.spawn.rigid_props.disable_gravity=True;cfg.spawn.usd_dir=str(out/'conversion_cache')
        @configclass
        class Cfg(InteractiveSceneCfg):robot=cfg
        scene=InteractiveScene(Cfg(num_envs=1,env_spacing=5.,replicate_physics=False));stage=scene.stage
        def deinstance(path):
            for p in path.GetPrefixes():
                prim=stage.GetPrimAtPath(p)
                if prim and prim.IsInstance():prim.SetInstanceable(False)
        copied=[];paths={}
        for side,shift in [('left',5.),('right',10.)]:
            paths[side]={}
            for role,link in [('Parent',side+'_wrist_pitch_link'),('Child',side+'_wrist_roll_link')]:
                source='/World/envs/env_0/Robot/'+link;target='/World/Detached_'+side+'/'+role
                UsdGeom.Xform.Define(stage,'/World/Detached_'+side)
                p=UsdGeom.Xform.Define(stage,target).GetPrim();p.GetReferences().AddInternalReference(source);p.SetInstanceable(False)
                rb=UsdPhysics.RigidBodyAPI(p)
                if not rb:raise ValueError('Referenced original body API missing')
                rb.CreateKinematicEnabledAttr().Set(role=='Parent')
                PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr().Set(True)
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr().Set(0.)
                # Isolate the analytically proved cylinder#1; preserve original
                # source and every selected collider's original shape/scale.
                if role=='Parent':
                    first=stage.GetPrimAtPath(target+'/collisions/official_'+link+'_1/cylinder')
                    if not first:raise ValueError('Exact first cylinder absent')
                    deinstance(first.GetPath());first=stage.GetPrimAtPath(first.GetPath())
                    UsdPhysics.CollisionAPI(first).GetCollisionEnabledAttr().Set(False)
                if p.HasAPI(UsdPhysics.ArticulationRootAPI):raise ValueError('Detached copy still articulation root')
                if p.HasAPI(UsdPhysics.FilteredPairsAPI) and UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets():raise ValueError('Unexpected copied explicit filter')
                selected='/collisions/official_'+link+('_2/cylinder' if role=='Parent' else '_1/node_STL_BINARY_')
                sp=stage.GetPrimAtPath(source+selected);cp=stage.GetPrimAtPath(target+selected)
                if not sp or not cp or not cp.HasAPI(UsdPhysics.CollisionAPI):raise ValueError('Original selected collider absent')
                def shape(p):
                    names=('radius','height','axis','xformOp:scale','physics:approximation','physics:collisionEnabled')
                    return dict(type=p.GetTypeName(),attrs={n:str(p.GetAttribute(n).Get()) for n in names if p.GetAttribute(n)})
                original=shape(sp);clone=shape(cp)
                if original!=clone:raise ValueError('Selected shape attribute mismatch')
                descendants=[dict(path=str(x.GetPath()).replace(target,'COPY'),type=x.GetTypeName(),is_mesh=x.IsA(UsdGeom.Mesh)) for x in Usd.PrimRange(cp,Usd.TraverseInstanceProxies())]
                copied.append(dict(side=side,role=role,source=source,target=target,selected_source=source+selected,selected_target=target+selected,original=original,clone=clone,descendants=descendants,shift=[shift,0.,2.]))
                paths[side][role]=dict(source=source,target=target,shift=shift)
        for name,x in [('A',20.),('B',20.08)]:
            su.CuboidCfg(size=(.1,.1,.1),rigid_props=su.RigidBodyPropertiesCfg(disable_gravity=True),mass_props=su.MassPropertiesCfg(mass=1),collision_props=su.CollisionPropertiesCfg(),activate_contact_sensors=True).func('/World/Box'+name,su.CuboidCfg(size=(.1,.1,.1),rigid_props=su.RigidBodyPropertiesCfg(disable_gravity=True),mass_props=su.MassPropertiesCfg(mass=1),collision_props=su.CollisionPropertiesCfg(),activate_contact_sensors=True),translation=(x,0.,1.))
        sim.reset();scene.reset();robot=scene['robot']
        if len(robot.joint_names)!=29 or any('neck' in n or 'head' in n for n in robot.joint_names):raise ValueError('Body action contract')
        view=SimulationManager.get_physics_sim_view()
        from detached_contract import make_pair_view
        bodies={};channels={}
        for side,roles in paths.items():
            for role,p in roles.items():
                for typ,key in [('source','source'),('target','target')]:
                    v=view.create_rigid_body_view(p[key])
                    if v.count!=1:raise ValueError('Body view cardinality')
                    bodies[side+'_'+role+'_'+typ]=v
            channels['detached_'+side]=make_pair_view(view,roles['Parent']['target'],roles['Child']['target'])
            channels['articulated_'+side]=make_pair_view(view,roles['Parent']['source'],roles['Child']['source'])
        channels['box']=make_pair_view(view,'/World/BoxA','/World/BoxB')
        box={n:view.create_rigid_body_view('/World/Box'+n) for n in ['A','B']}
        samples={n:[] for n in channels};counts={n:[] for n in channels};trace=[];idx=torch.tensor([0],device=sim.device,dtype=torch.int32)
        root=robot.data.default_root_state.clone();q=robot.data.default_joint_pos.clone();zero=torch.zeros_like(q)
        # Every step reads the actual source body transform before placing its
        # detached copy, preserving the exact pair relative pose.
        for i in range(STEPS):
            robot.write_root_pose_to_sim(root[:,:7]);robot.write_root_velocity_to_sim(torch.zeros_like(root[:,7:]));robot.write_joint_state_to_sim(q,zero);robot.set_joint_position_target(q)
            before={}
            for side,roles in paths.items():
                for role,p in roles.items():
                    source=bodies[side+'_'+role+'_source'].get_transforms().clone();target=source.clone();target[:,:3]+=torch.tensor([p['shift'],0.,2.],device=sim.device)
                    v=bodies[side+'_'+role+'_target'];v.set_transforms(target,indices=idx);v.set_velocities(torch.zeros((1,6),device=sim.device),indices=idx)
                    got=v.get_transforms().clone()
                    if not torch.allclose(got,target,atol=2e-6,rtol=0):raise ValueError('Detached transform write/readback')
                    before[side+'_'+role]=dict(source_xyzw=source.cpu().tolist()[0],detached_xyzw=got.cpu().tolist()[0])
            for name,x,vx in [('A',20.,.1),('B',20.08,-.1)]:
                box[name].set_transforms(torch.tensor([[x,0.,1.,0.,0.,0.,1.]],device=sim.device),indices=idx)
                box[name].set_velocities(torch.tensor([[vx,0.,0.,0.,0.,0.]],device=sim.device),indices=idx)
                before['box_'+name]=dict(pose_xyzw=box[name].get_transforms().clone().cpu().tolist()[0])
            scene.write_data_to_sim();sim.step(render=False);scene.update(DT)
            for name,v in channels.items():
                m=v.get_contact_force_matrix(dt=DT);d=v.get_contact_data(dt=DT)
                if tuple(m.shape)!=(1,1,3) or len(d)!=6 or tuple(d[4].shape)!=(1,1):raise ValueError('Exact contact layout')
                samples[name].append(float(torch.linalg.vector_norm(m)));counts[name].append(int(d[4][0,0]))
            trace.append(dict(step=i,before=before))
        from detached_contract import evaluate
        verdict=evaluate(samples,counts,trace,copied)
        if sha(model)!=MODEL_SHA or sha(asset/'asset_config.py')!=ASSET_SHA:raise ValueError('Asset changed')
        save(out/'contact_report.json',dict(stage='hud04_detached_wrist_control',source_commit=a.expected_commit,probe_sha256=sha(Path(__file__)),model_sha256=sha(model),asset_config_sha256=sha(asset/'asset_config.py'),dt=DT,explicit_steps=STEPS,body_dof=29,head_action_dof=0,contact_processing=processing,copied_shapes=copied,trace=trace,samples=samples,counts=counts,verdict=verdict,cooked_hull_exact_vertices_verified=False,all_joint_pairs_verified=False,source_motion_alignment_verified=False,accepted_reference=False,training_ready=False))
    except BaseException as e:
        save(out/'contact_error.json',dict(error_type=type(e).__name__,message=str(e)[:2000],frames=traceback.format_tb(e.__traceback__)[-8:],accepted_reference=False,training_ready=False));raise
    finally:
        if sim is not None:sim.clear_all_callbacks();sim.clear_instance()
        app.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo-root',required=True);p.add_argument('--expected-commit',required=True);p.add_argument('--output',required=True);main(p.parse_args())