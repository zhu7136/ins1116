"""Read only pinned original joint channels; no root/quaternion inference."""
import hashlib,importlib.util,sys
from pathlib import Path

HERE=Path(__file__).parent
BASE_COMMIT='f4e25561c4c8655faa73b7f68086a61a6034f1de'
PRIOR=Path('/workspace/hud04-direction-solve-20261118')
SUPERVISOR=Path('/workspace/hud04-window-system-20261117')
SOURCE=Path('/workspace/hud04-recovery-fd12fe1/upstream/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf')
TARGET=Path('/workspace/hud04-recovery-bf4a057/flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf')
MOTION=Path('/workspace/hud04-original-drive-20261117/kneelClimbStep1-x-0.1-ziwen-retargeted.npz')
SEED=Path('/workspace/hud04-implementation-20261117/step1.joint_fit.npz')
PINS={'anatomical_directions.py':'23660e03d2fbe7b1e0e334c47b506e9c07e4c8931430bd6f6c3afc762159587d',
      'bounded_runtime.py':'01974a3ec53c7d287d85d09d48c71756e70d2f7024666c603b8f5442c5ea3c35',
      'motion_archive.py':'5a46c55da0721f518a910295a4f11ddc748319d0108905748b0c5fd44bca56d3'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dependency(name):
    path=(PRIOR if name=='anatomical_directions.py' else SUPERVISOR)/name
    if name not in PINS or sha(path)!=PINS[name]:raise ValueError('Pinned dependency changed')
    # multiprocessing spawn must also be able to import this module by name.
    if str(path.parent) not in sys.path:sys.path.insert(0,str(path.parent))
    module=name[:-3];spec=importlib.util.spec_from_file_location(module,path);loaded=importlib.util.module_from_spec(spec)
    sys.modules[module]=loaded;spec.loader.exec_module(loaded);return loaded

def load():
    import numpy as np
    directions=dependency('anatomical_directions.py')
    if sha(MOTION)!=directions.MOTION_SHA or sha(SEED)!=directions.SEED_SHA:raise ValueError('Pinned raw motion or joint seed changed')
    source=directions.DirectionModel(SOURCE,'source',wrist_endpoint='first_wrist_pivot')
    target=directions.DirectionModel(TARGET,'target',wrist_endpoint='first_wrist_pivot')
    with np.load(MOTION,allow_pickle=False) as z:
        if z['joint_pos'].shape!=(997,29) or float(z['framerate'].item())!=90:raise ValueError('997 native 90-Hz frames')
        sq=source.order(z['joint_pos'],z['joint_names'].tolist())
    with np.load(SEED,allow_pickle=False) as z:
        if z['joint_pos'].shape!=(997,29) or float(z['framerate'].item())!=90:raise ValueError('997 native seed frames')
        seed=target.order(z['joint_pos'],z['joint_names'].tolist())
    desired=source.evaluate(sq)
    return source,target,seed,desired,dict(source_motion_sha256=sha(MOTION),seed_sha256=sha(SEED),
        source_task_model_sha256=sha(SOURCE),target_model_sha256=sha(TARGET),base_commit=BASE_COMMIT,
        dependency_sha256=PINS.copy(),wrist_endpoint='first_wrist_pivot',source_root_fields_read=False)