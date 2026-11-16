"""Standalone Isaac asset/depth probe. No task, motion, policy or optimizer.

Default invocation only describes the probe; --execute-probe is required to
initialize CUDA/Isaac. Publishing this file is not GPU-start authorization.
Fixed-base static-target mode is diagnostic only, NOT balance/training evidence.
"""
import argparse,json,os,runpy,time,traceback
from pathlib import Path
from probe_checks import check_head_zero,require_finite_values

ROOT=Path(__file__).resolve().parent

def event(stage,**values):
    print(json.dumps(dict(stage=stage,**values)),flush=True)

def config_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-probe',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'output')
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--device',default='cuda:0')
    return p

def main(args):
    if not args.execute_probe:
        event('plan_only',gpu_started=False,training_updates=0,steps=args.steps)
        return
    if not 30<=args.steps<=240:raise ValueError('Probe limited to30..240physicssteps')
    # CPU integrity checks precede all simulator imports.
    from preflight import audit
    audit(ROOT)
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['HUMANOID_DESCRIPTION_DIR']=str(ROOT/'humanoid-description')
    os.environ['HUD04_CAMERA_PROFILE']='sim_chest_35deg_v1'
    # RayCasterCamera uses Warp/CUDA, not RTX or Replicator image rendering.
    # False alone does not override ENABLE_CAMERAS=1 in AppLauncher2.3.2.
    os.environ['ENABLE_CAMERAS']='0'
    # This isolated probe does not import instinctlab.__init__/task registry.
    # No motion folder or instinct_rl installation is needed.
    event('isaac_initialization_started')
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,enable_cameras=False,device=args.device)
    app=launcher.app
    try:
        import numpy as np
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sensors import RayCasterCamera,RayCasterCameraCfg,patterns
        from isaaclab.utils.math import subtract_frame_transforms
        from pxr import UsdGeom,Gf
        import omni.usd
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=args.device))
        stage=omni.usd.get_context().get_stage();stage.DefinePrim('/World','Xform')
        asset=runpy.run_path(str(ROOT/'asset_config.py'))
        cfg=asset['HUD04_CFG'].replace(prim_path='/World/HUD04')
        # Explicitly diagnostic fixed base; original copied config unchanged.
        cfg.spawn.fix_base=True
        robot=Articulation(cfg)
        # Offline-authored static target; no Nucleus/network asset dependency.
        mesh=UsdGeom.Mesh.Define(stage,'/World/DepthTarget')
        mesh.CreatePointsAttr([Gf.Vec3f(1.2,-3,0),Gf.Vec3f(1.2,-3,3),Gf.Vec3f(1.2,3,3),Gf.Vec3f(1.2,3,0)])
        mesh.CreateFaceVertexCountsAttr([3,3]);mesh.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
        camera_defs=runpy.run_path(str(ROOT/'camera_mount.py'))
        mount=camera_defs['chest_camera_mount'](asset['HUD04_URDF'],{'HUD04_CAMERA_PROFILE':'sim_chest_35deg_v1'})
        imaging=mount['simulation_imaging']
        import math
        cam_cfg=RayCasterCameraCfg(
            prim_path='/World/HUD04/'+mount['parent_link'],
            mesh_prim_paths=['/World/DepthTarget'],
            offset=RayCasterCameraCfg.OffsetCfg(pos=mount['pos'],rot=mount['rot'],convention=mount['convention']),
            pattern_cfg=patterns.PinholeCameraPatternCfg(focal_length=1.,
                horizontal_aperture=2*math.tan(math.radians(imaging['hfov_deg'])/2),
                vertical_aperture=2*math.tan(math.radians(imaging['vfov_deg'])/2),
                width=imaging['width'],height=imaging['height']),
            data_types=['distance_to_image_plane'],update_period=1/30,
            max_distance=imaging['max_distance_m'],depth_clipping_behavior='max',debug_vis=False)
        camera=RayCasterCamera(cam_cfg)
        sim.reset()
        # Config.default_joint_pos is a target/default buffer, not proof that
        # imported PhysX joints have been initialized to that posture.
        target=robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(target,robot.data.default_joint_vel.clone())
        robot.reset();sim.forward();robot.update(.005)
        camera.update(.005,force_recompute=True)
        names=list(robot.joint_names)
        if len(names)!=29 or set(names)!=set(asset['HUD04_JOINT_NAMES']) or any('head' in n for n in names):
            raise AssertionError('Imported movable joint list is not exactly29body joints')
        parent_ids=[robot.body_names.index(n) for n in ['waist_pitch_link','head_yaw_link']]
        child_ids=[robot.body_names.index(n) for n in ['head_yaw_link','head_pitch_link']]
        def head_relative():
            data=robot.data
            return subtract_frame_transforms(data.body_pos_w[:,parent_ids],data.body_quat_w[:,parent_ids],
                                            data.body_pos_w[:,child_ids],data.body_quat_w[:,child_ids])
        refpos,refquat=head_relative();refpos=refpos.clone();refquat=refquat.clone()
        check_head_zero(refpos[0].cpu().tolist(),refquat[0].cpu().tolist())
        max_head_translation=0.;max_head_rotation=0.;depths=[];valid_fractions=[]
        camera_frames=[]
        event('asset_initialized',body_dof=len(names),fixed_base_diagnostic=True)
        for step in range(args.steps):
            if not app.is_running():raise RuntimeError('Simulator stopped before requested steps')
            robot.set_joint_position_target(target);robot.write_data_to_sim()
            sim.step(render=False);robot.update(.005);camera.update(.005)
            pos,quat=head_relative()
            check_head_zero(pos[0].cpu().tolist(),quat[0].cpu().tolist())
            max_head_translation=max(max_head_translation,float((pos-refpos).abs().max()))
            qdiff=torch.minimum((quat-refquat).abs().amax(dim=-1),(quat+refquat).abs().amax(dim=-1))
            max_head_rotation=max(max_head_rotation,float(qdiff.max()))
            if not torch.isfinite(robot.data.joint_pos).all():raise AssertionError('Nonfinite joint state')
            if step%7==6:
                depth=camera.data.output['distance_to_image_plane'].detach().cpu().numpy().copy()
                camera_frames.append(int(camera.frame[0]))
                if depth.shape!=(1,27,48,1):raise AssertionError(f'Unexpected raw depth shape:{depth.shape}')
                if not np.isfinite(depth).all():raise AssertionError('Nonfinite clipped depth')
                valid=(depth>imaging['min_distance_m'])&(depth<imaging['max_distance_m']-.001)
                valid_fractions.append(float(valid.mean()));depths.append(depth)
        if max_head_translation>1e-4 or max_head_rotation>1e-4:raise AssertionError('Fixed head transform drift exceeds probe tolerance')
        require_finite_values([max_head_translation,max_head_rotation,*valid_fractions],'probe metrics')
        if len(set(camera_frames))<2:raise AssertionError('Camera did not produce multiple sensor frames')
        if not valid_fractions or min(valid_fractions)<.05:raise AssertionError('Static target not observed by enough depth rays')
        np.savez_compressed(args.output/'depth_samples.npz',depth=np.stack(depths[-8:]))
        report=dict(stage='asset_camera_probe_completed',probe_passed=True,training_ready=False,
            training_updates=0,motion_files_used=0,body_dof=len(names),joint_names=names,
            head_policy_dof=0,head_relative_translation_drift_m=max_head_translation,
            head_relative_quaternion_component_drift=max_head_rotation,
            raw_depth_shape=[1,27,48,1],min_valid_target_fraction=min(valid_fractions),
            unique_sensor_frames=len(set(camera_frames)),head_zero_pose_verified=True,
            physics_steps=args.steps,physics_dt_s=.005,fixed_base_diagnostic=True,
            simulation_device=str(sim.device),
            gpu_cuda_execution_verified=torch.device(sim.device).type=='cuda' and torch.cuda.is_available(),
            rtx_camera_rendering_enabled=False,
            camera_backend='IsaacLab_standard_RayCasterCamera',camera_calibration_validated=False,
            limitations=['Not InstinctLab custom grouped/noisy depth pipeline or18x32policyinput',
              'No body occlusion/terrain/contact/balance/actuator-load assessment',
              'No policy action-manager mapping, reward, reset, motion playback or learning',
              'Real camera calibration and depth metric accuracy not established'])
        (args.output/'report.json').write_text(json.dumps(report,indent=2));event(**report)
    finally:
        # This ray-casting probe has no Replicator writer jobs to drain.
        # IsaacSim v5.1.0 SimulationApp.close supports this documented option.
        event('simulator_shutdown_started',wait_for_replicator=False)
        app.close(wait_for_replicator=False)
        event('simulator_shutdown_completed')

if __name__=='__main__':
    try:main(config_parser().parse_args())
    except Exception as exc:
        event('probe_failed',error_type=type(exc).__name__,error=str(exc),training_ready=False)
        traceback.print_exc();raise