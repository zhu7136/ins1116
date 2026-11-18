"""Pinned existing modules and explicit geometric feature proposals, no launch."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from pose_tasks import Feature

HERE=Path(__file__).resolve().parent
INTEGRATION=HERE.parents[1]
REPO=HERE.parents[3]
TARGET=REPO/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
BASE_COMMIT='ee8416303d1dd471b05cc6ae6702e5f52008d554'
DIAGNOSTIC=HERE.parent/'full_direction_20261118/OVERLAP_V4/STEP1_FULL_LENGTH_JOINT_DIAGNOSTIC.npz'
DIAGNOSTIC_SHA='2aa620d3086ed5f1c260c210e38fcb3d3a94cbaea549911844b8e4c31127ddc0'
PINS={
    'directions':('experimental/direction_solve_20261118/anatomical_directions.py','23660e03d2fbe7b1e0e334c47b506e9c07e4c8931430bd6f6c3afc762159587d'),
    'named_target':('experimental/window_system_20261117/named_target.py','ce946040de2a74dd4a72744a1153d78098d9b247e91bc3c8fdee1300e071a464'),
    'window_problem':('experimental/window_system_20261117/window_problem.py','2c5911dd8c5b80a3ebb285834b376dcb9c4eae81a119399bd7ef11ad16cc999d'),
    'bounded_runtime':('experimental/window_system_20261117/bounded_runtime.py','01974a3ec53c7d287d85d09d48c71756e70d2f7024666c603b8f5442c5ea3c35'),
    'independent_fk':('kinematic_diagnostics.py','943b63605c4e837451a47eee87fa754ac04d72afcef02fe8904c77252b438696')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dependency(key):
    relative,digest=PINS[key];path=INTEGRATION/relative
    if sha(path)!=digest:raise ValueError('Pinned dependency hash mismatch: '+key)
    name=path.stem
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve()!=path.resolve():raise ValueError('Pinned module name collision')
        return sys.modules[name]
    # Spawn must be able to import the supervisor by its real file/module name.
    if str(path.parent) not in sys.path:sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def models():
    d=dependency('directions')
    return d.DirectionModel(TARGET,'target',wrist_endpoint='first_wrist_pivot'),dependency('named_target').NamedTarget(TARGET)

def diagnostic(model):
    if sha(DIAGNOSTIC)!=DIAGNOSTIC_SHA:raise ValueError('Saved diagnostic SHA mismatch')
    with np.load(DIAGNOSTIC,allow_pickle=False) as z:
        if set(z.files)!={'joint_pos','joint_names','frame_indices','framerate','role'} or float(z['framerate'].item())!=90:
            raise ValueError('Joint diagnostic contract changed')
        np.testing.assert_array_equal(z['frame_indices'],np.arange(997))
        return model.order(z['joint_pos'],z['joint_names'].tolist())

def features():
    """Link frames for torso/hands; exact nominal target foot-box bottom frames.

    Wrist transforms are identity and EXPLICITLY NOT palm material frames.
    Foot +z points into the nominal box, -z is outward bottom normal. No
    source-foot planar patch is inferred from its seven capsule proxies.
    """
    result=[Feature('torso_URDF_frame','waist_pitch_link',np.eye(4),'URDF_link_diagnostic')]
    result.extend(Feature(side+'_distal_wrist_URDF_frame',side+'_wrist_roll_link',np.eye(4),'URDF_link_diagnostic') for side in ('left','right'))
    xml=ET.parse(TARGET).getroot()
    for side in ('left','right'):
        link=side+'_ankle_roll_link';node=xml.find("link[@name='"+link+"']")
        boxes=[c for c in node.findall('collision') if c.find('geometry/box') is not None]
        if len(boxes)!=1:raise ValueError('One pinned nominal foot box required')
        box=boxes[0];t=dependency('named_target').origin(box.find('origin'))
        size=np.fromstring(box.find('geometry/box').get('size'),sep=' ')
        t[:3,3]+=t[:3,:3]@np.array([0,0,-size[2]/2])
        result.append(Feature(side+'_nominal_box_bottom_frame',link,t,'nominal_box_bottom_NOT_accepted_support'))
    return tuple(result)