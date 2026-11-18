"""Read only pinned-model zero-coordinate correspondence, not palm calibration."""
import sys,json
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/hud04-direction-solve-20261118','/workspace/hud04-runtime-preflight-20261117']
from anatomical_directions import DirectionModel
import task_kinematics as TK
from scipy.spatial.transform import Rotation

def main():
    rows=[]
    for kind,path in [('source',TK.SOURCE),('target',TK.TARGET)]:
        m=DirectionModel(path,kind);poses,_=m.fk(np.zeros((1,29)))
        anchor=poses[m.anchor][0];inv=np.linalg.inv(anchor)
        links=['torso_link' if kind=='source' else 'waist_pitch_link']
        links += [s+'_'+l for s in ['left','right'] for l in ['shoulder_pitch_link','elbow_link', 'wrist_yaw_link' if kind=='source' else 'wrist_roll_link','hip_pitch_link','knee_link','ankle_roll_link']]
        rows.append(dict(kind=kind,zero_coordinate_in_position_bounds=bool(np.all(m.lower<=0) and np.all(m.upper>=0)),rows=[dict(link=l,position=(inv@poses[l][0])[:3,3].tolist(),basis=(inv@poses[l][0])[:3,:3].tolist()) for l in links]))
    p=Path(__file__).with_name('ZERO_COORDINATE_GEOMETRY.json')
    with p.open('x') as f:json.dump(dict(role='zero_coordinate_URDF_geometry_NOT_accepted_palm_or_sole_calibration',models=rows),f,indent=2,allow_nan=False)
    print(json.dumps(rows,indent=2),flush=True)
if __name__=='__main__':main()