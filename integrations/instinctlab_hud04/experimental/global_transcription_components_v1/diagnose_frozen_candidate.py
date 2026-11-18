"""Read-only diagnosis of one DECLARED interpolation, not a motion solve/export."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import BPoly
from euclidean_c2 import coefficients, evaluate, component_extrema

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = HERE.parent/'material_trajectory_v1/TRIAL_MATERIAL_003_FULL_SEQUENCE/FULL997_ENGINEERING_CANDIDATE_UNACCEPTED.npz'
MODEL = REPO/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
EXPECTED = '2b4070e32e3a5f120caea7942b42e91a6f31bbca9e0b6d7e8e135b50803ad2f4'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'FROZEN_INTERPOLATION_DIAGNOSTICS.json'
    if output.exists(): raise FileExistsError('Do not overwrite a previous audit')
    assert sha(SOURCE) == EXPECTED
    with np.load(SOURCE,allow_pickle=False) as z:
        assert not bool(z['accepted_reference']) and not bool(z['training_ready'])
        q, names = z['joint_pos'], z['joint_names'].tolist()
        root = z['root_transforms_w'][:,:3,3]
        assert q.shape == (997,29) and int(z['framerate']) == 90
        assert np.array_equal(z['frame_indices'],np.arange(997))
    xml = ET.parse(MODEL).getroot()
    joints = {j.get('name'): j for j in xml.findall('joint')}
    assert all(joints[n].get('type') == 'revolute' for n in names)
    heads = [j for n,j in joints.items() if 'head' in n and j.get('type') == 'fixed']
    assert len(heads) == 2
    lower = np.array([float(joints[n].find('limit').get('lower')) for n in names])
    upper = np.array([float(joints[n].find('limit').get('upper')) for n in names])
    vmax = np.array([float(joints[n].find('limit').get('velocity')) for n in names])
    dt = 1/90
    # Deliberate initial-guess construction. Not recovered derivatives or an approved playback contract.
    p = np.concatenate([q,root],axis=1)
    v = np.gradient(p,dt,axis=0,edge_order=2)
    a = np.gradient(v,dt,axis=0,edge_order=2)
    c = coefficients(p,v,a,dt)
    c2_errors = [float(np.max(np.abs(evaluate(c,dt,1,k)[:-1]-evaluate(c,dt,0,k)[1:])))
                 for k in range(3)]
    t = np.arange(len(p))*dt
    bp = BPoly.from_derivatives(t,[list(x) for x in zip(p,v,a)])
    independent_errors = [float(max(np.max(np.abs(evaluate(c,dt,u,k)-bp(t[:-1]+u*dt,nu=k)))
                                   for u in [.13,.37,.81])) for k in range(3)]
    print('Global coefficients and independent SciPy check complete',flush=True)
    lo,hi,arglo,arghi = component_extrema(c[:,:,:29],dt,0)
    vlo,vhi,argvlo,argvhi = component_extrema(c[:,:,:29],dt,1)
    pv = np.maximum(np.maximum(lower-lo,hi-upper),0)
    vv = np.maximum(np.maximum(-vmax-vlo,vhi-vmax),0)
    def witness(errors,lows,highs,argslow,argshigh,lowlimit,highlimit):
        i,j = np.unravel_index(np.argmax(errors),errors.shape)
        low = lowlimit[j]-lows[i,j] >= highs[i,j]-highlimit[j]
        u = argslow[i,j] if low else argshigh[i,j]
        return {'segment_start_frame':int(i),'joint':names[j], 'u':float(u),
                'value':float(lows[i,j] if low else highs[i,j]),
                'limit':float(lowlimit[j] if low else highlimit[j]),
                'excess':float(errors[i,j])}
    # Numeric tolerance below is only for counting numerical diagnostic exceedances.
    # Not a physical acceptance tolerance or a new model limit.
    tol = 1e-8
    result = {
        'schema':'hud04.euclidean_c2_initial_guess_diagnostic.v1',
        'source_sha256':sha(SOURCE),'model_sha256':sha(MODEL),
        'frames':997,'segments':996,'fps':90,'body_dof':29,'head_policy_dof':0,
        'purpose':'Global Euclidean interpolation component/preflight only; NO optimization',
        'derivatives':'Explicit assumed np.gradient(edge_order=2) initial guess, NOT measured/recovered',
        'root_rotation':'NOT IMPLEMENTED; no Euler/rotation-vector interpolation substitution',
        'c2_join_max_error_position_velocity_acceleration':c2_errors,
        'independent_scipy_max_error_position_velocity_acceleration':independent_errors,
        'node_position_violation_rad':float(max(np.max(lower-q),np.max(q-upper),0)),
        'neighbor_average_velocity_violation_rad_s':float(max(np.max(np.abs(np.diff(q,axis=0)/dt)-vmax),0)),
        'assumed_node_velocity_violation_rad_s':float(max(np.max(np.abs(v[:,:29])-vmax),0)),
        'interior_position_violation_rad':float(pv.max()),
        'interior_velocity_violation_rad_s':float(vv.max()),
        'position_exceedance_segment_joint_count':int(np.sum(pv>tol)),
        'velocity_exceedance_segment_joint_count':int(np.sum(vv>tol)),
        'diagnostic_count_roundoff_threshold':tol,
        'worst_position':witness(pv,lo,hi,arglo,arghi,lower,upper),
        'worst_velocity':witness(vv,vlo,vhi,argvlo,argvhi,-vmax,vmax),
        'extrema_method':'Endpoints + numerical polynomial stationary roots; NOT interval-certified',
        'acceleration_jerk_limits':'UNRESOLVED, no reviewed limits invented',
        'collision_support_dynamics_torque_swept_validation':False,
        'accepted_reference':False,'training_ready':False,'new_motion_exported':False,
        'source_unchanged_after_diagnosis':sha(SOURCE)==EXPECTED,
        'gpu_tasks_created':0,'motion_optimizer_calls':0,
    }
    with output.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__': main()