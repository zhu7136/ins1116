"""Independent BPoly evaluation of extrema witnesses, no euclidean_c2 import."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import BPoly

here = Path(__file__).resolve().parent
report = json.loads((here/'FROZEN_INTERPOLATION_DIAGNOSTICS.json').read_text())
source = here.parent/'material_trajectory_v1/TRIAL_MATERIAL_003_FULL_SEQUENCE/FULL997_ENGINEERING_CANDIDATE_UNACCEPTED.npz'
model = here.parents[3]/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
assert hashlib.sha256(source.read_bytes()).hexdigest() == report['source_sha256']
assert hashlib.sha256(model.read_bytes()).hexdigest() == report['model_sha256']
with np.load(source,allow_pickle=False) as z:
    q, names = z['joint_pos'],z['joint_names'].tolist()
joints = {j.get('name'):j for j in ET.parse(model).getroot().findall('joint')}
limits = np.array([float(joints[n].find('limit').get('velocity')) for n in names])
dt = 1/90
v = np.gradient(q,dt,axis=0,edge_order=2)
a = np.gradient(v,dt,axis=0,edge_order=2)
t = np.arange(len(q))*dt
bp = BPoly.from_derivatives(t,[list(x) for x in zip(q,v,a)])
checks = []
for key,order in [('worst_position',0),('worst_velocity',1)]:
    w = report[key]
    i,j,u = w['segment_start_frame'],names.index(w['joint']),w['u']
    value = float(bp((i+u)*dt,nu=order)[j])
    error = abs(value-w['value'])
    assert error < 1e-8
    checks.append({'witness':key,'joint':w['joint'],'independent_value':value,
                   'difference_from_primary':error,'verified_excess':abs(value-w['limit'])})
avg = np.diff(q,axis=0)/dt
tol = 1e-9
mask = ((np.abs(np.abs(avg[:-1])-limits)<tol)
        & (np.abs(np.abs(avg[1:])-limits)<tol) & (avg[:-1]*avg[1:]<0))
opposites = [{'join_frame':int(i+1),'joint':names[j],
              'left_segment_average_rad_s':float(avg[i,j]),
              'right_segment_average_rad_s':float(avg[i+1,j]),
              'original_speed_limit_rad_s':float(limits[j])} for i,j in np.argwhere(mask)]
result = {'independent_scipy_witness_checks':checks,
          'opposite_near_saturated_adjacent_segment_count':len(opposites),
          'near_saturation_diagnostic_tolerance_rad_s':tol,
          'opposite_near_saturated_adjacent_segments':opposites,
          'interpretation':'If opposite saturation were exact, speed-bounded C1 interpolation preserving nodes would be impossible: integral saturation forces constant opposite rates on adjoining intervals. These floating-point near-equalities alone are NOT an exact infeasibility proof, nor a lower bound without an acceleration constraint.',
          'candidate_modified':False,'accepted_reference':False,'training_ready':False,
          'optimizer_calls':0,'gpu_tasks_created':0}
with (here/'INDEPENDENT_DIAGNOSTIC_VERIFICATION.json').open('x') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k != 'opposite_near_saturated_adjacent_segments'},ensure_ascii=False,indent=2))