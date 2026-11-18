"""Necessary net external-force diagnosis for an explicitly assumed interpolation.

No contact optimization, full inverse dynamics, torque test, or motion export.
"""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from euclidean_c2 import coefficients,evaluate
from so3_c2 import SharedSO3Quintic
from mass_properties import OriginalMassModel

here=Path(__file__).resolve().parent
repo=here.parents[3]
source=here.parent/'material_trajectory_v1/TRIAL_MATERIAL_003_FULL_SEQUENCE/FULL997_ENGINEERING_CANDIDATE_UNACCEPTED.npz'
model_path=repo/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected='2b4070e32e3a5f120caea7942b42e91a6f31bbca9e0b6d7e8e135b50803ad2f4'
assert sha(source)==expected
assert sha(model_path)=='865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7'
output=here/'CENTROIDAL_FORCE_DIAGNOSTICS.json'
if output.exists():raise FileExistsError('Do not overwrite a prior diagnostic')
with np.load(source,allow_pickle=False) as z:
    q,roots,names=z['joint_pos'],z['root_transforms_w'],z['joint_names'].tolist()
    assert q.shape==(997,29) and int(z['framerate'])==90
    assert not bool(z['accepted_reference']) and not bool(z['training_ready'])
model=OriginalMassModel(model_path)
assert model.root=='base_link' and len(model.movable)==29
xml=ET.parse(model_path).getroot()
for name in ['head_yaw_joint','head_pitch_joint']:
    assert next(j for j in xml.findall('joint') if j.get('name')==name).get('type')=='fixed'
com,inertia=model.properties(q,names,roots,inertia=True)
# Independent Rodrigues FK implementation, no solver imports or invocation.
sys.path.insert(0,str(repo/'integrations/instinctlab_hud04'))
from kinematic_diagnostics import RobotTree,origin_transform
local=RobotTree.read(model_path).fk(q,names)
independent=np.zeros_like(com);mass_independent=0.
for link in xml.findall('link'):
    e=link.find('inertial');m=float(e.find('mass').get('value'))
    world=roots@local[link.get('name')]@origin_transform(e.find('origin'))
    independent+=m*world[:,:3,3];mass_independent+=m
independent/=mass_independent
fk_error=float(np.max(np.abs(com-independent)));assert fk_error<1e-12
dt=1/90
p=np.c_[q,roots[:,:3,3]]
v=np.gradient(p,dt,axis=0,edge_order=2);a=np.gradient(v,dt,axis=0,edge_order=2)
c=coefficients(p,v,a,dt)
R=roots[:,:3,:3]
interval_w=Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).as_rotvec()/dt
w=np.empty((997,3));w[1:-1]=(interval_w[:-1]+interval_w[1:])/2
w[0]=1.5*interval_w[0]-.5*interval_w[1];w[-1]=1.5*interval_w[-1]-.5*interval_w[-2]
so3=SharedSO3Quintic(R,w,np.gradient(w,dt,axis=0,edge_order=2),dt)

def acceleration(h):
    offsets=[-2,-1,0,1,2];coms=[]
    for k in offsets:
        u=.5+k*h/dt
        allp=evaluate(c,dt,u)
        root=np.tile(np.eye(4),(996,1,1));root[:,:3,3]=allp[:,29:]
        for i in range(996):root[i,:3,:3]=so3.state(i,u)[0]
        coms.append(model.properties(allp[:,:29],names,root))
    return (-coms[4]+16*coms[3]-30*coms[2]+16*coms[1]-coms[0])/(12*h*h)

print('Original inertia and independent COM binding complete',flush=True)
h=dt*.01
coarse=acceleration(h);fine=acceleration(h/2)
delta=np.linalg.norm(coarse-fine,axis=1)
gravity=np.array([0.,0.,-9.81]) # Explicit engineering convention, runtime equivalence NOT asserted.
force=model.total_mass*(fine-gravity)
ratio=np.linalg.norm(force,axis=1)/(model.total_mass*9.81)
peak=int(np.argmax(ratio))
negative=np.where(force[:,2]<0)[0]
result={
 'schema':'hud04.centroidal_force_initial_guess_diagnostic.v1',
 'candidate_sha256':expected,'original_urdf_sha256':sha(model_path),
 'frames':997,'body_dof':29,'head_policy_dof':0,'head_fixed_link_mass_retained':True,
 'urdf_root':'base_link','links_with_inertial':len(model.inertials),'original_total_mass_kg':model.total_mass,
 'inertial_audit':model.physical_audit,
 'independent_rodrigues_com_max_component_error_m':fk_error,
 'minimum_com_inertia_eigenvalue_over_original_nodes_kg_m2':float(np.linalg.eigvalsh(inertia).min()),
 'gravity_m_s2':gravity.tolist(),'gravity_runtime_binding_verified':False,
 'derivatives':'Same EXPLICIT assumed C2 initial guess as prior Euclidean/SO3 checks, not recovered physical rates or validated playback',
 'sampled_at':'996 segment midpoints, not a continuous force bound',
 'acceleration_method':'5-point centered COM finite difference at h and h/2; fine reported',
 'h_seconds':h,'fine_h_seconds':h/2,
 'max_acceleration_step_halving_difference_m_s2':float(delta.max()),
 'max_force_step_halving_difference_N':float(model.total_mass*delta.max()),
 'required_external_force_formula':'F_non_gravity = M*(COM_ddot - g); necessary resultant only, no root virtual actuator',
 'required_force_over_bodyweight_percentiles':dict(zip(['p50','p90','p99','max'],np.percentile(ratio,[50,90,99,100]).tolist())),
 'worst_midpoint':{'segment_start_frame':peak,'time_s':(peak+.5)*dt,
    'com_acceleration_m_s2':fine[peak].tolist(),'required_net_force_N':force[peak].tolist(),
    'required_force_over_bodyweight':float(ratio[peak]),'step_halving_force_difference_N':float(model.total_mass*delta[peak])},
 'negative_required_vertical_resultant_midpoint_count':len(negative),
 'negative_vertical_interpretation':'NOT a contact-feasibility verdict: actual terrain normals, active contact modes and friction remain unbound. No assumption that all contacts have upward normals; unassigned support frames are not flight.',
 'force_realizability_verified':False,'torque_or_full_floating_base_dynamics_verified':False,
 'contact_or_flight_labels_created':False,'new_motion_exported':False,'optimizer_calls':0,'gpu_tasks_created':0,
 'accepted_reference':False,'training_ready':False,'candidate_unchanged':sha(source)==expected,
}
with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k!='inertial_audit'},ensure_ascii=False,indent=2),flush=True)