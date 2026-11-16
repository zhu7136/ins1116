"""CPU naming/affine contract. No simulator, solver or training imports."""
import math
import xml.etree.ElementTree as ET

import numpy as np

FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')
HEAD = ('head_yaw_joint', 'head_pitch_joint')


def control_values(name):
    if 'hip_' in name or 'knee_' in name:
        return 139.41, 17.75, 0.14125, 0.2511
    if 'ankle_' in name or name.startswith('waist_'):
        return 93.65, 11.92, 0.1845504, 0.1121
    if 'shoulder_' in name or 'elbow_' in name:
        return 87.51, 11.14, 0.0886706, 0.1200
    if 'wrist_' in name:
        return 15.12, 1.93, 0.0153218, 0.3141
    raise ValueError('Unknown actuator family: ' + name)


def read_model(path):
    root = ET.parse(path).getroot()
    joints = root.findall('joint')
    for name in HEAD:
        joint = next(j for j in joints if j.get('name') == name)
        if joint.get('type') != 'fixed':
            raise ValueError('Head must be structurally fixed')
        if any(float(x) != 0 for x in joint.find('origin').get('rpy', '0 0 0').split()):
            raise ValueError('Unexpected head zero rotation')
    moving = [j for j in joints if j.get('type') in ('revolute', 'continuous', 'prismatic')]
    names = [j.get('name') for j in moving]
    if len(names) != 29 or len(set(names)) != 29 or any('head' in n for n in names):
        raise ValueError('Expected exactly 29 unique body joints')
    data = {}
    for j in moving:
        name, limit = j.get('name'), j.find('limit')
        default = -.2 if 'hip_pitch' in name or 'ankle_pitch' in name else .4 if 'knee' in name else 0.
        values = dict(lower=float(limit.get('lower')), upper=float(limit.get('upper')),
                      effort=float(limit.get('effort')), velocity=float(limit.get('velocity')),
                      default=default, scale=control_values(name)[3])
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Nonfinite model parameter')
        if not values['lower'] <= default <= values['upper'] or min(values['effort'], values['velocity']) <= 0:
            raise ValueError('Invalid limit/default')
        data[name] = values
    return names, data


def permutation(source, destination):
    """Indices to gather source columns into destination order; no ambiguity."""
    if len(source) != len(set(source)) or len(destination) != len(set(destination)):
        raise ValueError('Duplicate names')
    if len(source) != 29 or set(source) != set(destination) or any('head' in n for n in source):
        raise ValueError('Mismatched body joint sets')
    return [source.index(n) for n in destination]


def finite_vector(value, width=29):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (width,) or not np.isfinite(value).all():
        raise ValueError('Expected finite vector of width ' + str(width))
    return value


def position_targets(action, names, model, defaults=None):
    """JointPositionAction(use_default_offset=True, clip=None); no hidden clamp."""
    permutation(list(model), names)
    action = finite_vector(action)
    default = finite_vector(defaults if defaults is not None else [model[n]['default'] for n in names])
    return default + action * np.array([model[n]['scale'] for n in names])


def proprioception(q, qd, last_action, names, model, defaults=None):
    """One unnoised joint frame ONLY, not full reference/depth/history inputs.

    Default joint velocities are zero. Last action stays RAW and unscaled.
    """
    permutation(list(model), names)
    default = finite_vector(defaults if defaults is not None else [model[n]['default'] for n in names])
    return {'joint_pos': finite_vector(q) - default,
            'joint_vel': finite_vector(qd), 'last_action': finite_vector(last_action).copy()}


def _transform(xyz, rpy):
    from scipy.spatial.transform import Rotation
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    t[:3, 3] = xyz
    return t


def foot_box_bottoms(path):
    """URDF FK at provisional default pose, identity root. No IK/optimization.

    Uses existing URDF foot boxes, not differently sized MJCF boxes.
    """
    root = ET.parse(path).getroot()
    _, model = read_model(path)
    links = {l.get('name'): l for l in root.findall('link')}
    joints = root.findall('joint')
    roots = set(links) - {j.find('child').get('link') for j in joints}
    if len(roots) != 1:
        raise ValueError('Expected one root')
    transforms, pending = {roots.pop(): np.eye(4)}, list(joints)
    while pending:
        progress = False
        for j in list(pending):
            parent = j.find('parent').get('link')
            if parent not in transforms:
                continue
            origin = j.find('origin')
            xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0]*3
            rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0]*3
            t = _transform(xyz, rpy)
            if j.get('type') != 'fixed':
                from scipy.spatial.transform import Rotation
                if j.get('type') != 'revolute':
                    raise ValueError('Only this serial revolute model is supported')
                axis = np.array([float(v) for v in j.find('axis').get('xyz').split()])
                if not np.isclose(np.linalg.norm(axis), 1):
                    raise ValueError('Nonunit joint axis')
                rot = np.eye(4)
                rot[:3, :3] = Rotation.from_rotvec(axis * model[j.get('name')]['default']).as_matrix()
                t = t @ rot
            transforms[j.find('child').get('link')] = transforms[parent] @ t
            pending.remove(j)
            progress = True
        if not progress:
            raise ValueError('Disconnected/cyclic model')
    bottoms = {}
    for name in FEET:
        collisions = links[name].findall('collision')
        if len(collisions) != 1 or collisions[0].find('geometry/box') is None:
            raise ValueError('Expected original single foot box')
        origin = collisions[0].find('origin')
        xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0]*3
        rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0]*3
        size = np.array([float(v) for v in collisions[0].find('geometry/box').get('size').split()])
        corners = np.array([[x, y, z, 1] for x in [-size[0]/2, size[0]/2]
                            for y in [-size[1]/2, size[1]/2] for z in [-size[2]/2, size[2]/2]])
        points = (transforms[name] @ _transform(xyz, rpy) @ corners.T).T
        bottoms[name] = float(points[:, 2].min())
    return bottoms