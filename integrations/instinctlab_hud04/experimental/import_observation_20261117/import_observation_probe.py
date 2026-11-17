"""NEW opt-in HUD04 import observation. No motion, PPO, or contact acceptance."""
import argparse,faulthandler,hashlib,json,os,runpy,subprocess,sys,time
from pathlib import Path

MODEL_SHA='865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7'
ASSET_SHA='4fdb08b4aa352893a6feef5afb6deb8ec273e1fbdcfc2d67ac0e691729321021'

PHASES={'before_app','after_app','before_scene_import','after_entity_import','before_post_import_snapshot',
    'after_post_import_snapshot','after_scene_clone','after_post_clone_snapshot','before_physics_reset',
    'after_physics_reset','after_post_physics_snapshot','before_cleanup','after_cleanup'}
def record_phase(phase):
    if phase not in PHASES:raise ValueError('Unknown diagnostic phase')
    print(json.dumps(dict(stage='hud04_import_progress',phase=phase,monotonic_seconds=time.monotonic(),
        proves_import_or_collision_acceptance=False,accepted_reference=False,training_ready=False)),flush=True)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_bundle(repo):
    import xml.etree.ElementTree as ET
    asset=repo/'flux/hud04_asset_camera_probe';manifest=json.loads((asset/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        p=(asset/name).resolve()
        if not p.is_relative_to(asset.resolve()) or sha(p)!=digest:raise ValueError('Asset manifest mismatch')
    model=asset/'humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
    if sha(model)!=MODEL_SHA or sha(asset/'asset_config.py')!=ASSET_SHA:raise ValueError('Pinned asset changed')
    root=ET.parse(model).getroot();moving=[j for j in root.findall('joint') if j.get('type') in ('revolute','continuous','prismatic')]
    if len(moving)!=29 or any('head' in j.get('name') for j in moving):raise ValueError('Body action contract')
    for name in ('head_yaw_joint','head_pitch_joint'):
        j=next(j for j in root.findall('joint') if j.get('name')==name)
        if j.get('type')!='fixed' or any(float(x)!=0 for x in j.find('origin').get('rpy','0 0 0').split()):raise ValueError('Head contract')
    return asset,model

def plan():
    return dict(stage='plan_only',gpu_started=False,num_envs=2,explicit_physics_steps=0,camera_enabled=False,
        observation_scope='standalone HUD04 import/clone/init ONLY; NOT actual perceptive task/Terrain/contacts',
        snapshot_phases=['post_import','post_clone','post_physics_init'],child_seconds_max=240,
        platform_deadline_verified=False,solver_calls=0,ppo_updates=0,accepted_reference=False,training_ready=False)

def main(args):
    if not args.execute:
        print(json.dumps(plan()),flush=True);return
    if not args.repo_root or not args.expected_commit or not args.output:raise ValueError('Explicit checkout/commit/output required')
    repo=Path(args.repo_root).resolve();out=Path(args.output).resolve()
    actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=5).strip()
    if actual!=args.expected_commit:raise ValueError('Actual checkout commit mismatch')
    dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True,timeout=5)
    if dirty.strip():raise ValueError('Tracked checkout modifications present; do not reset them')
    if not out.is_dir() or {p.name for p in out.iterdir()}!={'child.log'}:raise ValueError('Supervisor-created output with only own child.log required')
    asset,model=verify_bundle(repo)
    observer=repo/'integrations/instinctlab_hud04/experimental/usd_filter_observer_20261117'
    sys.path.insert(0,str(observer))
    from usd_filter_snapshot import collect,write_new
    os.environ['HUMANOID_DESCRIPTION_DIR']=str(asset/'humanoid-description');os.environ['ENABLE_CAMERAS']='0'
    from isaaclab.app import AppLauncher
    # Own Python stack only, no /proc, host/daemon or unrelated process access.
    # It records call sites (not values) within the already bounded child log.
    faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
    record_phase('before_app')
    app=AppLauncher(headless=True,enable_cameras=False,device='cuda:0').app
    record_phase('after_app')
    sim=None;report=None;phases=[]
    try:
        import inspect
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
        from isaaclab.utils import configclass
        from pxr import Usd
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device='cuda:0'))
        cfg=runpy.run_path(str(asset/'asset_config.py'))['HUD04_CFG'].replace(prim_path='{ENV_REGEX_NS}/Robot')
        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot=cfg
        class ObservedScene(InteractiveScene):
            def _add_entities_from_cfg(self):
                super()._add_entities_from_cfg()
                record_phase('after_entity_import');record_phase('before_post_import_snapshot')
                observation=collect(self.stage,'/World/envs/env_0/Robot')
                observation['runtime_phase']='post_import';observation['runtime_execution']=True
                write_new(out/'post_import.json',observation);phases.append('post_import')
                record_phase('after_post_import_snapshot')
        scene_cfg=SceneCfg(num_envs=2,env_spacing=4.,replicate_physics=True,filter_collisions=True)
        record_phase('before_scene_import')
        scene=ObservedScene(scene_cfg)
        record_phase('after_scene_clone')
        if phases!=['post_import']:raise RuntimeError('Expected import hook not observed; no fake fallback')
        observation=collect(scene.stage,'/World/envs/env_0/Robot')
        observation['runtime_phase']='post_clone';observation['runtime_execution']=True
        write_new(out/'post_clone.json',observation);phases.append('post_clone')
        record_phase('after_post_clone_snapshot');record_phase('before_physics_reset')
        sim.reset();scene.reset()
        record_phase('after_physics_reset')
        robot=scene['robot'];names=list(robot.joint_names)
        if len(names)!=29 or len(set(names))!=29 or any('head' in n for n in names):raise ValueError('Runtime joint identity contract')
        if robot.is_fixed_base:raise ValueError('Actual articulation must be floating')
        observation=collect(scene.stage,'/World/envs/env_0/Robot')
        observation['runtime_phase']='post_physics_init';observation['runtime_execution']=True
        write_new(out/'post_physics_init.json',observation);phases.append('post_physics_init')
        record_phase('after_post_physics_snapshot')
        scene_file=Path(inspect.getfile(InteractiveScene)).resolve()
        report=dict(stage='hud04_import_observed',source_commit=actual,model_sha256=sha(model),asset_config_sha256=sha(asset/'asset_config.py'),
            python_executable=sys.executable,python_version=sys.version.split()[0],usd_library_version=list(Usd.GetVersion()),
            interactive_scene_source_sha256=sha(scene_file),observer_sha256=sha(observer/'usd_filter_snapshot.py'),
            standalone_probe_not_full_task=True,num_envs=2,replicate_physics=True,filter_collisions_requested=True,
            explicit_physics_steps=0,initialization_may_step_physics=True,body_dof=len(names),head_policy_dof=0,joint_names=names,
            snapshot_phases=phases,cooked_geometry_verified=False,actual_physx_pair_filters_verified=False,
            dynamics_acceptance=False,solver_calls=0,ppo_updates=0,accepted_reference=False,training_ready=False,cleanup_completed=False)
    finally:
        record_phase('before_cleanup')
        try:
            if sim is not None:
                sim.stop();sim.clear_all_callbacks();sim.clear_instance()
        finally:
            app.close();record_phase('after_cleanup');faulthandler.cancel_dump_traceback_later()
    if report is None:raise RuntimeError('No observation completed')
    report['cleanup_completed']=True;write_new(out/'runtime_report.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--repo-root');p.add_argument('--expected-commit');p.add_argument('--output');main(p.parse_args())