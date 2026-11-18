"""Frozen997 SO3 component smoke check, NO solve/export/runtime validation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from so3_c2 import SharedSO3Quintic

here=Path(__file__).resolve().parent
source=here.parent/'material_trajectory_v1/TRIAL_MATERIAL_003_FULL_SEQUENCE/FULL997_ENGINEERING_CANDIDATE_UNACCEPTED.npz'
expected='2b4070e32e3a5f120caea7942b42e91a6f31bbca9e0b6d7e8e135b50803ad2f4'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
with np.load(source,allow_pickle=False) as z:
    R=z['root_transforms_w'][:,:3,:3]
    assert R.shape==(997,3,3) and int(z['framerate'])==90
    assert not bool(z['accepted_reference']) and not bool(z['training_ready'])
dt=1/90
interval_w=Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).as_rotvec()/dt
w=np.empty((997,3))
w[1:-1]=(interval_w[:-1]+interval_w[1:])/2
w[0]=1.5*interval_w[0]-.5*interval_w[1]
w[-1]=1.5*interval_w[-1]-.5*interval_w[-2]
a=np.gradient(w,dt,axis=0,edge_order=2)
curve=SharedSO3Quintic(R,w,a,dt)
endpoint=np.zeros(3);joins=np.zeros(3)
maximum_sampled_chart_angle=0.
for i in range(996):
    left=curve.state(i,0);right=curve.state(i,1)
    for k,target in enumerate([R,w,a]):
        endpoint[k]=max(endpoint[k],np.max(np.abs(left[k]-target[i])),np.max(np.abs(right[k]-target[i+1])))
    if i<995:
        nxt=curve.state(i+1,0)
        for k in range(3):joins[k]=max(joins[k],np.max(np.abs(right[k]-nxt[k])))
    for u in [0.,.25,.5,.75,1.]:
        current=curve.state(i,u)[0]
        angle=float(Rotation.from_matrix(R[i].T@current).magnitude())
        maximum_sampled_chart_angle=max(maximum_sampled_chart_angle,angle)

def vee(M):return np.array([M[2,1]-M[1,2],M[0,2]-M[2,0],M[1,0]-M[0,1]])/2
independent=[]
for i in [0,58,292,349,351,585,900,995]:
    u=.37;h=dt*1e-3
    rm2,rm1,r,rp1,rp2=[curve.state(i,u+k*h/dt)[0] for k in [-2,-1,0,1,2]]
    rd=(rm2-8*rm1+8*rp1-rp2)/(12*h)
    rdd=(-rp2+16*rp1-30*r+16*rm1-rm2)/(12*h*h)
    _,aw,aa=curve.state(i,u)
    independent.append({'segment':i,'omega_world_error_rad_s':float(np.max(np.abs(vee(rd@r.T)-aw))),
                        'alpha_world_error_rad_s2':float(np.max(np.abs(vee(rdd@r.T)-aa)))})
result={
    'schema':'hud04.so3_c2_component_frozen997_check.v1',
    'source_sha256':expected,'frames':997,'segments':996,'fps':90,
    'omega_alpha_convention':'WORLD; local Lie chart mapped by analytic left Jacobian and its directional derivative',
    'derivatives':'ASSUMED initial guess from adjacent world rotation logs + central averages/extrapolated endpoints, alpha gradient; NOT recovered/true/exported motion rates',
    'endpoint_max_component_error_rotation_matrix_omega_alpha':endpoint.tolist(),
    'all995_join_max_component_error_rotation_matrix_omega_alpha':joins.tolist(),
    'maximum_sampled_local_chart_angle_rad':maximum_sampled_chart_angle,
    'chart_interior_certified':False,'samples_per_segment':5,
    'independent_matrix_difference_checks':independent,
    'near_pi_recharting':'Required if encountered; guards reject, no automatic projection or recharting implemented',
    'global_optimizer_implemented':False,'collision_support_dynamics_torque_verified':False,
    'angular_acceleration_jerk_limit_binding':'UNRESOLVED',
    'accepted_reference':False,'training_ready':False,'new_motion_exported':False,
    'candidate_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==expected,
    'gpu_tasks_created':0,'optimizer_calls':0,
}
with (here/'FROZEN_SO3_COMPONENT_CHECK.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))