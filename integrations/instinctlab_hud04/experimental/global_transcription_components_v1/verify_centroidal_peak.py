"""Independent original XML/Rodrigues COM peak check; no mass_properties import."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
from so3_c2 import SharedSO3Quintic

here=Path(__file__).resolve().parent;repo=here.parents[3]
report=json.loads((here/'CENTROIDAL_FORCE_DIAGNOSTICS.json').read_text())
source=here.parent/'material_trajectory_v1/TRIAL_MATERIAL_003_FULL_SEQUENCE/FULL997_ENGINEERING_CANDIDATE_UNACCEPTED.npz'
model_path=repo/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
assert hashlib.sha256(source.read_bytes()).hexdigest()==report['candidate_sha256']
assert hashlib.sha256(model_path.read_bytes()).hexdigest()==report['original_urdf_sha256']
with np.load(source,allow_pickle=False) as z:q,roots,names=z['joint_pos'],z['root_transforms_w'],z['joint_names'].tolist()
sys.path.insert(0,str(repo/'integrations/instinctlab_hud04'))
from kinematic_diagnostics import RobotTree,origin_transform
tree=RobotTree.read(model_path);xml=ET.parse(model_path).getroot()
inertials=[(l.get('name'),float(l.find('inertial/mass').get('value')),origin_transform(l.find('inertial/origin'))) for l in xml.findall('link')]
mass=sum(m for _,m,_ in inertials);dt=1/90;i=report['worst_midpoint']['segment_start_frame']
p=np.c_[q,roots[:,:3,3]];v=np.gradient(p,dt,axis=0,edge_order=2);a=np.gradient(v,dt,axis=0,edge_order=2)
bp=BPoly.from_derivatives([0,dt],[[p[i],v[i],a[i]],[p[i+1],v[i+1],a[i+1]]])
R=roots[:,:3,:3];iw=Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).as_rotvec()/dt
w=np.empty((997,3));w[1:-1]=(iw[:-1]+iw[1:])/2;w[0]=1.5*iw[0]-.5*iw[1];w[-1]=1.5*iw[-1]-.5*iw[-2]
aa=np.gradient(w,dt,axis=0,edge_order=2)
so3=SharedSO3Quintic(R[i:i+2],w[i:i+2],aa[i:i+2],dt)

def check(h):
    times=dt*.5+np.arange(-2,3)*h
    state=bp(times);base=np.tile(np.eye(4),(5,1,1));base[:,:3,3]=state[:,29:]
    for k,t in enumerate(times):
        # Evaluate chart polynomial directly, independent of analytic angular derivatives/state().
        phi=np.array([np.polynomial.polynomial.polyval(t/dt,so3.coefficients[0,:,j]) for j in range(3)])
        base[k,:3,:3]=R[i]@Rotation.from_rotvec(phi).as_matrix()
    local=tree.fk(state[:,:29],names);com=np.zeros((5,3))
    for name,m,T in inertials:com+=m*(base@local[name]@T)[:,:3,3]
    com/=mass
    acc=(-com[4]+16*com[3]-30*com[2]+16*com[1]-com[0])/(12*h*h)
    f=mass*(acc-np.array(report['gravity_m_s2']))
    return acc,f

acc,f=check(dt*.01/2)
acc2,f2=check(dt*.01)
prior=np.array(report['worst_midpoint']['required_net_force_N'])
error=float(np.max(np.abs(f-prior)))
assert error<.01
result={'schema':'hud04.independent_centroidal_peak_check.v1','segment':i,
        'source_sha256':report['candidate_sha256'],'mass_from_independent_XML_sum_kg':mass,
        'independent_required_net_force_N':f.tolist(),
        'independent_required_force_over_bodyweight':float(np.linalg.norm(f)/(mass*9.81)),
        'max_force_component_difference_from_primary_N':error,
        'independent_step_halving_force_difference_N':float(np.linalg.norm(f-f2)),
        'method':'Independent XML/Rodrigues FK + SciPy Bernstein translation/joints + direct chart polynomial rotation; inherited SO3 endpoint construction, not a fully independent SO3 interpolant',
        'assumed_interpolation_only':True,'contact_realizability_proved':False,
        'accepted_reference':False,'training_ready':False,'candidate_modified':False}
with (here/'INDEPENDENT_CENTROIDAL_PEAK_VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))