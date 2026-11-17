"""UNVALIDATED per-rigid-body PhysX contact-pair bridge candidate.

No import-time simulator use, no task submission and no net-force fallback.
CPU fixtures check software only. Runtime filter resolution, cooked geometry,
contact offsets and buffer behavior still require real IsaacLab validation.
The sums are NORMAL contact-force magnitudes, not friction/support labels.
"""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from types import SimpleNamespace
import torch


def collision_body_names(model):
    """Actual collision-bearing URDF links, including fixed head, not visuals."""
    root=ET.parse(model).getroot()
    names=[link.attrib['name'] for link in root.findall('link') if link.findall('collision')]
    if not names or len(names)!=len(set(names)):raise ValueError('Invalid collision-link inventory')
    return names


def _tensor(value,shape,label):
    if not isinstance(value,torch.Tensor) or value.shape!=shape or not torch.isfinite(value).all():raise RuntimeError('Invalid '+label)
    return value


def reduce_contact_points(raw,matrix,net,capacity,*,atol=1e-4,rtol=1e-4):
    """One sensor body / M disjoint filter endpoints; never norm the net force.

    Forces from get_contact_data(dt=physics_dt) are in N. Cross-check
    point normals times scalar force against the pair matrix and full net
    normal force. The tolerances test numerical API consistency ONLY.
    Saturation is conservatively blocked, never silently truncated/accepted.
    """
    if not isinstance(raw,(tuple,list)) or len(raw)!=6:raise RuntimeError('Expected six PhysX contact arrays')
    forces,points,normals,separations,counts,starts=raw
    if not isinstance(counts,torch.Tensor) or counts.ndim!=2 or counts.shape[0]!=1 or counts.shape[1]<1 or counts.dtype not in (torch.int32,torch.int64):raise RuntimeError('One body / many filter counts required')
    if not isinstance(starts,torch.Tensor) or starts.shape!=counts.shape or starts.dtype not in (torch.int32,torch.int64):raise RuntimeError('Invalid contact starts')
    m=counts.shape[1];device=counts.device
    matrix=_tensor(matrix,(1,m,3),'pair force matrix');net=_tensor(net,(1,3),'net normal forces')
    if not isinstance(forces,torch.Tensor) or forces.shape not in [(capacity,),(capacity,1)]:raise RuntimeError('Unexpected contact capacity/force shape')
    for value,label,shape in [(points,'points',(capacity,3)),(normals,'normals',(capacity,3)),(separations,'separations',(capacity,1))]:
        # Unused PhysX slots need not contain valid data; check active slots below.
        if not isinstance(value,torch.Tensor) or value.shape!=shape:raise RuntimeError('Invalid '+label+' shape')
    if any(x.device!=device for x in [forces,points,normals,separations,starts,matrix,net]):raise RuntimeError('Mixed contact devices')
    if torch.any(counts<0):raise RuntimeError('Negative contact count')
    active=counts>0
    if torch.any(starts[active]<0) or torch.any(starts[active]+counts[active]>capacity):raise RuntimeError('Contact buffer bounds')
    total=int(counts.sum().item())
    if total>=capacity:raise RuntimeError('Possible contact buffer saturation')
    scalars=torch.zeros((1,m),device=device,dtype=matrix.dtype);reconstructed=torch.zeros_like(matrix)
    if total:
        flat_counts=counts.reshape(-1).long();flat_starts=starts.reshape(-1).long()
        rows=torch.repeat_interleave(torch.arange(m,device=device),flat_counts,output_size=total)
        cumulative=torch.cumsum(flat_counts,dim=0)-flat_counts
        idx=(torch.repeat_interleave(flat_starts,flat_counts,output_size=total)+torch.arange(total,device=device)
             -torch.repeat_interleave(cumulative,flat_counts,output_size=total))
        if torch.unique(idx).numel()!=total:raise RuntimeError('Overlapping filter contact records')
        f=forces.reshape(-1)[idx];n=normals[idx]
        if not all(torch.isfinite(x).all() for x in [f,n,points[idx],separations[idx]]) or torch.any(f<0):raise RuntimeError('Invalid active contact values')
        if not torch.allclose(torch.linalg.vector_norm(n,dim=-1),torch.ones_like(f),atol=atol,rtol=rtol):raise RuntimeError('Nonunit contact normals')
        scalars[0].scatter_add_(0,rows,f.to(matrix.dtype))
        reconstructed[0].scatter_add_(0,rows[:,None].expand(-1,3),(f[:,None]*n).to(matrix.dtype))
    if not torch.allclose(reconstructed,matrix,atol=atol,rtol=rtol):raise RuntimeError('Point-to-pair normal-force reconstruction mismatch')
    if not torch.allclose(matrix.sum(dim=1),net,atol=atol,rtol=rtol):raise RuntimeError('Filtered pairs do not reconstruct total normal force')
    return dict(terrain_normal_force_N=scalars[:,0],nonterrain_normal_force_N=scalars[:,1:].sum(dim=1),
                terrain_point_count=counts[:,0],nonterrain_point_count=counts[:,1:].sum(dim=1))


def _inventory(stage,Usd,UsdPhysics,PhysxSchema):
    """Expand instance proxies and record enabled collision leaves and owners."""
    result=[]
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False:continue
        if enabled is not True:raise RuntimeError('Unknown collision-enabled state')
        owner=prim
        while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):owner=owner.GetParent()
        owner_path=None;reporter=False;reporter_threshold=None
        if owner and not owner.IsPseudoRoot():
            if UsdPhysics.RigidBodyAPI(owner).GetRigidBodyEnabledAttr().Get() is not True:raise RuntimeError('Disabled/unknown rigid owner unsupported')
            owner_path=str(owner.GetPath());reporter=owner.HasAPI(PhysxSchema.PhysxContactReportAPI)
            if reporter:reporter_threshold=PhysxSchema.PhysxContactReportAPI(owner).GetThresholdAttr().Get()
        result.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),owner=owner_path,contact_reporter=reporter,reporter_threshold=reporter_threshold))
    return sorted(result,key=lambda x:x['path'])


def make_pair_plan(inventory,env_paths,body_names,ground_root='/World/ground'):
    """Every enabled collider must belong to exactly one explicit endpoint.

    Single combined generated terrain mesh only. Dynamic endpoints use exact
    rigid-owner paths; unrelated static colliders use exact collider paths.
    This plan does not itself prove PhysX static-filter path interpretation.
    """
    if not inventory or len({x['path'] for x in inventory})!=len(inventory):raise RuntimeError('Empty/duplicate collision census')
    terrain=[x for x in inventory if x['path'].startswith(ground_root+'/')]
    if len(terrain)!=1 or terrain[0]['type']!='Mesh' or terrain[0]['owner'] is not None:raise RuntimeError('Expected exactly one static generated terrain mesh, no Plane fallback')
    terrain_path=terrain[0]['path'];others=[x for x in inventory if x['path']!=terrain_path]
    expected={env_path+'/Robot/'+name for env_path in env_paths for name in body_names}
    if len(expected)!=len(env_paths)*len(body_names):raise RuntimeError('Duplicate environment/body selection')
    actual={x['owner'] for x in others if x['owner'] and any(x['owner'].startswith(p+'/Robot/') for p in env_paths)}
    if actual!=expected:raise RuntimeError('USD/URDF collision owner coverage mismatch; fixed-link merging or model changed')
    if any(not x['contact_reporter'] for x in others if x['owner'] in expected):raise RuntimeError('Missing PhysX contact reporter on robot collision owner')
    if any(x.get('reporter_threshold')!=0 for x in others if x['owner'] in expected):raise RuntimeError('Robot contact reporting threshold must be zero')
    # A mesh under a robot with no matching rigid owner must never silently
    # become an unrelated static endpoint and escape the collision-body audit.
    if any(x['owner'] not in expected for x in others if any(x['path'].startswith(p+'/Robot/') for p in env_paths)):raise RuntimeError('Unmapped robot collision geometry')
    endpoints=sorted({x['owner'] or x['path'] for x in others})
    plan=[]
    for env_id,env_path in enumerate(env_paths):
        for body_id,name in enumerate(body_names):
            sensor=env_path+'/Robot/'+name
            plan.append(dict(env_id=env_id,body_id=body_id,sensor=sensor,filters=[terrain_path]+[p for p in endpoints if p!=sensor]))
    return plan


class PhysXContactPairBridge:
    """Read every request anew; no cache surviving reset or physical steps."""
    def __init__(self,env,bundle,body_names,inventory,plan,views,inventory_reader,capacity):
        self.env=env;self.bundle=bundle;self.body_names=list(body_names);self.inventory=inventory
        self.plan=plan;self.views=views;self.inventory_reader=inventory_reader;self.capacity=capacity
        self.census_sha256=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
        if len(plan)!=env.num_envs*len(body_names) or len(views)!=len(plan):raise RuntimeError('Incomplete body view plan')
    def sample(self,force_threshold):
        if not math.isfinite(force_threshold) or force_threshold<0:raise ValueError('Invalid contact threshold')
        if self.inventory_reader()!=self.inventory:raise RuntimeError('Collision census changed; rebuild bridge explicitly')
        env=self.env;dt=float(env.physics_dt)
        if not math.isfinite(dt) or dt<=0:raise RuntimeError('Invalid actual physics dt')
        step=env.common_step_counter;physics_step=env._sim_step_counter
        shape=(env.num_envs,len(self.body_names));terrain=torch.zeros(shape,dtype=torch.bool,device=env.device);other=torch.zeros_like(terrain)
        terrain_forces=torch.zeros(shape,device=env.device);other_forces=torch.zeros_like(terrain_forces)
        for row,view in zip(self.plan,self.views):
            raw=view.get_contact_data(dt=dt);matrix=view.get_contact_force_matrix(dt=dt);net=view.get_net_contact_forces(dt=dt)
            if matrix.shape!=(1,len(row['filters']),3):raise RuntimeError('PhysX filter/sensor count mismatch')
            reduced=reduce_contact_points(raw,matrix,net,self.capacity)
            index=(row['env_id'],row['body_id']);tf=reduced['terrain_normal_force_N'][0];of=reduced['nonterrain_normal_force_N'][0]
            terrain_forces[index]=tf;other_forces[index]=of
            terrain[index]=tf>force_threshold;other[index]=of>force_threshold
        if step!=env.common_step_counter or physics_step!=env._sim_step_counter:raise RuntimeError('Physics changed while reading contact pairs')
        return dict(schema_version=1,step_id=step,physics_step_id=physics_step,review_record_sha256=self.bundle['review_record_sha256'],
                    body_names=self.body_names.copy(),terrain_contact=terrain,nonterrain_contact=other,
                    terrain_normal_force_N=terrain_forces,nonterrain_normal_force_N=other_forces,
                    collision_census_sha256=self.census_sha256,implementation_status='UNVALIDATED_PHYSX_CANDIDATE')


class PhysicsSubstepContactCollector:
    """Explicit InteractiveScene update/reset protocol adapter, NOT SensorBase.

    Pinned scene invokes these two methods every physics tick / environment
    reset. Dynamic insertion into scene.sensors is a candidate integration,
    not a claim of tested support by every SensorBase/UI/extension consumer.
    Read all substeps, preserve the peak per-body normal-force sum, then let
    BOTH terms read the same complete window. Missing/duplicate ticks fail.
    """
    has_debug_vis_implementation=False
    is_initialized=True
    def __init__(self,bridge,permission_reader):
        self.bridge=bridge;self.env=bridge.env;self.device=self.env.device
        self.permission_reader=permission_reader
        self.cfg=SimpleNamespace(debug_vis=False,update_period=0.,prim_path='/World')
        self.decimation=self.env.cfg.decimation
        if isinstance(self.decimation,bool) or not isinstance(self.decimation,int) or not 1<=self.decimation<=16:raise RuntimeError('Unsupported bounded decimation')
        self.last_physics_step=self.env._sim_step_counter;self.window=None
        shape=(self.env.num_envs,len(bridge.body_names));self.terrain_peak=torch.zeros(shape,device=self.device)
        self.other_peak=torch.zeros_like(self.terrain_peak);self.forbidden_peak=torch.zeros_like(self.terrain_peak)
        self.ticks=torch.zeros(self.env.num_envs,dtype=torch.long,device=self.device)
    def set_debug_vis(self,debug_vis):
        if debug_vis:raise RuntimeError('Contact collector has no debug visualizer')
        return False
    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.terrain_peak[ids]=0;self.other_peak[ids]=0;self.forbidden_peak[ids]=0;self.ticks[ids]=0
    def update(self,dt,force_recompute=False):
        if not math.isclose(float(dt),float(self.env.physics_dt),rel_tol=0.,abs_tol=1e-12):raise RuntimeError('Collector update must use actual physics dt')
        step=self.env._sim_step_counter
        if step!=self.last_physics_step+1:raise RuntimeError('Missing/duplicate/rewound physics update')
        window=(step-1)//self.decimation
        if window!=self.window:
            self.reset();self.window=window
        sample=self.bridge.sample(force_threshold=0.)
        if sample['physics_step_id']!=step:raise RuntimeError('Stale raw physics sample')
        allowed=self.permission_reader()
        if not isinstance(allowed,torch.Tensor) or allowed.dtype!=torch.bool or allowed.shape!=self.terrain_peak.shape or allowed.device!=self.terrain_peak.device:raise RuntimeError('Invalid per-substep terrain permission')
        self.terrain_peak=torch.maximum(self.terrain_peak,sample['terrain_normal_force_N'])
        self.other_peak=torch.maximum(self.other_peak,sample['nonterrain_normal_force_N'])
        forbidden=torch.where(allowed,0.,sample['terrain_normal_force_N'])
        self.forbidden_peak=torch.maximum(self.forbidden_peak,forbidden)
        self.ticks+=1;self.last_physics_step=step
    def sample(self,force_threshold):
        if not math.isfinite(force_threshold) or force_threshold<0:raise ValueError('Invalid contact threshold')
        env=self.env
        if (self.last_physics_step!=env._sim_step_counter or self.window is None
            or self.last_physics_step%self.decimation!=0 or env.common_step_counter!=self.window+1
            or not torch.all(self.ticks==self.decimation)):raise RuntimeError('Incomplete/stale/reset physics contact window')
        if self.bridge.inventory_reader()!=self.bridge.inventory:raise RuntimeError('Collision census changed after last substep')
        return dict(schema_version=2,step_id=env.common_step_counter,physics_step_id=env._sim_step_counter,
                    review_record_sha256=self.bridge.bundle['review_record_sha256'],body_names=self.bridge.body_names.copy(),
                    terrain_contact=self.terrain_peak>force_threshold,nonterrain_contact=self.other_peak>force_threshold,
                    forbidden_terrain_contact=self.forbidden_peak>force_threshold,
                    terrain_normal_force_N=self.terrain_peak.clone(),nonterrain_normal_force_N=self.other_peak.clone(),
                    collision_census_sha256=self.bridge.census_sha256,physics_substeps=self.decimation,
                    temporal_semantics='peak_forbidden_normal_sum_with_per_substep_90hz_permission',
                    implementation_status='UNVALIDATED_PHYSX_CANDIDATE')


def install_contact_pair_bridge(env,env_ids,bundle,body_names,model_path,capacity=2048):
    """Explicit startup event, AFTER physics initialization; no GPU submission.

    One exact rigid body per view follows pinned IsaacLab's one-to-many rule.
    Fixed-link merging/model changes fail closed against the USD census.
    """
    if getattr(env,'hud04_contact_pair_bridge',None) is not None:raise RuntimeError('Refusing to replace an existing contact bridge')
    collector_key='hud04_contact_pair_collector'
    if collector_key in env.scene.sensors:raise RuntimeError('Contact collector scene key already exists')
    if env.num_envs!=2 or isinstance(capacity,bool) or not isinstance(capacity,int) or capacity<2:raise RuntimeError('Only bounded two-env diagnostic candidate supported')
    if list(body_names)!=collision_body_names(model_path):raise RuntimeError('Model collision body order changed')
    model_hash=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    if model_hash!=bundle['model_sha256']:raise RuntimeError('Bridge model differs from reviewed bundle')
    spawn=env.cfg.scene.robot.spawn
    if spawn.merge_fixed_joints is not False or spawn.activate_contact_sensors is not True or spawn.articulation_props.enabled_self_collisions is not True:raise RuntimeError('Fixed-body/contact/self-collision configuration changed')
    from isaacsim.core.simulation_manager import SimulationManager
    import isaaclab.sim as sim_utils
    import carb
    from pxr import Usd,UsdPhysics,PhysxSchema
    if carb.settings.get_settings().get('/physics/disableContactProcessing') is not False:raise RuntimeError('Physics contact processing must be enabled')
    if not env.sim.is_playing():raise RuntimeError('Initialize physics before installing contact bridge')
    stage=sim_utils.get_current_stage();reader=lambda:_inventory(stage,Usd,UsdPhysics,PhysxSchema)
    inventory=reader();env_paths=list(env.scene.env_prim_paths)
    if len(env_paths)!=env.num_envs or len(set(env_paths))!=env.num_envs:raise RuntimeError('Environment path inventory mismatch')
    plan=make_pair_plan(inventory,env_paths,body_names);sim_view=SimulationManager.get_physics_sim_view();views=[]
    for row in plan:
        body_view=sim_view.create_rigid_body_view(row['sensor'])
        if list(body_view.prim_paths)!=[row['sensor']]:raise RuntimeError('Exact PhysX rigid-body path unresolved')
        views.append(sim_view.create_rigid_contact_view(row['sensor'],filter_patterns=row['filters'],max_contact_data_count=capacity))
    bridge=PhysXContactPairBridge(env,bundle,body_names,inventory,plan,views,reader,capacity)
    from .terrain_contact_terms import allowed_current_terrain_contacts
    if 'motion_reference' not in env.scene.sensors:raise RuntimeError('Reference sensor must update before contact collector')
    collector=PhysicsSubstepContactCollector(bridge,lambda:allowed_current_terrain_contacts(env,bundle,body_names))
    env.scene.sensors[collector_key]=collector
    env.hud04_contact_pair_bridge=collector