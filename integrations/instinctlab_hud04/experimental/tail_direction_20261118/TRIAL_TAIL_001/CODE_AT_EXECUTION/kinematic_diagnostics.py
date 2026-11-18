"""Deterministic URDF FK and raw terrain probes; no optimizers or simulator.

Vertical gaps are distances to the mesh's upper XY envelope, NOT signed collision
distances or contact labels. Box vertices are geometry probes, NOT a contact patch.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def vector(text, count=3):
    result = np.array([float(v) for v in text.split()], dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError('Invalid finite vector')
    return result


def axis_rotation(axis, angles):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    angles = np.asarray(angles, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or norm < 1e-12 or not np.isfinite(angles).all():
        raise ValueError('Invalid rotation axis/angle')
    x, y, z = axis / norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angles)[..., None, None] * skew + (1-np.cos(angles))[..., None, None] * (skew @ skew)


def origin_transform(element):
    result = np.eye(4)
    if element is not None:
        xyz, rpy = (vector(element.get(k, '0 0 0')) for k in ('xyz', 'rpy'))
        # URDF fixed-axis roll/pitch/yaw: Rz(yaw) Ry(pitch) Rx(roll).
        result[:3, :3] = (axis_rotation([0, 0, 1], rpy[2]) @ axis_rotation([0, 1, 0], rpy[1])
                          @ axis_rotation([1, 0, 0], rpy[0]))
        result[:3, 3] = xyz
    return result


def inverse_rigid(transform):
    result = np.zeros_like(transform)
    result[..., :3, :3] = np.swapaxes(transform[..., :3, :3], -1, -2)
    result[..., :3, 3] = -(result[..., :3, :3] @ transform[..., :3, 3, None])[..., 0]
    result[..., 3, 3] = 1
    return result


def world_pose(position, quaternion, order):
    p, q = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or q.shape != (len(p), 4) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Invalid pose arrays')
    if order not in ('wxyz', 'xyzw'):
        raise ValueError('Explicit quaternion order required')
    norm = np.linalg.norm(q, axis=1)
    if np.any(np.abs(norm-1) > 1e-3):
        raise ValueError('Non-unit quaternion')
    q = q / norm[:, None]  # Numerical rotation evaluation only; inputs never modified.
    if order == 'xyzw':
        q = q[:, [3, 0, 1, 2]]
    w, x, y, z = q.T
    result = np.tile(np.eye(4), (len(p), 1, 1))
    result[:, :3, :3] = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]).transpose(2, 0, 1)
    result[:, :3, 3] = p
    return result


class RobotTree:
    def __init__(self, raw):
        if len(raw) > 1024*1024 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
            raise ValueError('URDF size/DTD limit')
        root = ET.fromstring(raw)
        nodes, joints = root.findall('link'), root.findall('joint')
        self.links = {x.get('name'): x for x in nodes}
        if not self.links or None in self.links or len(self.links) != len(nodes):
            raise ValueError('Invalid or duplicate links')
        self.joints = []
        children, names = set(), set()
        for node in joints:
            name, kind = node.get('name'), node.get('type')
            if not name or name in names or kind not in ('fixed', 'revolute', 'continuous') or node.find('mimic') is not None:
                raise ValueError('Duplicate/unsupported joint')
            names.add(name)
            parent, child = (node.find(k).get('link') for k in ('parent', 'child'))
            if parent not in self.links or child not in self.links or child in children or parent == child:
                raise ValueError('Invalid joint graph')
            children.add(child)
            axis_node = node.find('axis')
            axis = vector(axis_node.get('xyz', '1 0 0')) if axis_node is not None else np.array([1., 0., 0.])
            if kind != 'fixed' and np.linalg.norm(axis) < 1e-12:
                raise ValueError('Zero movable joint axis')
            self.joints.append(dict(name=name, kind=kind, parent=parent, child=child,
                                    origin=origin_transform(node.find('origin')), axis=axis))
        roots = set(self.links)-children
        if len(roots) != 1 or len(joints) != len(nodes)-1:
            raise ValueError('Expected one rooted tree')
        self.root, = roots
        pending, reached, ordered = self.joints.copy(), {self.root}, []
        while pending:
            ready = [j for j in pending if j['parent'] in reached]
            if not ready:
                raise ValueError('Disconnected/cyclic joint graph')
            for joint in ready:
                reached.add(joint['child'])
                ordered.append(joint)
                pending.remove(joint)
        self.joints = ordered
        self.movable = [j['name'] for j in ordered if j['kind'] != 'fixed']

    @classmethod
    def read(cls, path):
        path = Path(path)
        if path.stat().st_size > 1024*1024:
            raise ValueError('URDF size limit')
        return cls(path.read_bytes())

    def fk(self, q, names):
        q = np.asarray(q, dtype=float)
        if (q.ndim != 2 or q.shape[1] != len(names) or not 1 <= len(q) <= 4096 or
                len(set(names)) != len(names) or set(names) != set(self.movable) or not np.isfinite(q).all()):
            raise ValueError('FK requires exact named finite joint channels and bounded batch')
        columns = {name: i for i, name in enumerate(names)}
        poses = {self.root: np.tile(np.eye(4), (len(q), 1, 1))}
        for joint in self.joints:
            relative = np.tile(joint['origin'], (len(q), 1, 1))
            if joint['kind'] != 'fixed':
                rotation = np.tile(np.eye(4), (len(q), 1, 1))
                rotation[:, :3, :3] = axis_rotation(joint['axis'], q[:, columns[joint['name']]])
                relative = relative @ rotation
            poses[joint['child']] = poses[joint['parent']] @ relative
        return poses

    def box_probes(self, link):
        collisions = self.links[link].findall('collision')
        if len(collisions) != 1 or collisions[0].find('geometry/box') is None:
            raise ValueError('Probe requires exactly one explicit collision box')
        collision = collisions[0]
        size = vector(collision.find('geometry/box').get('size'))
        if np.any(size <= 0):
            raise ValueError('Invalid collision box size')
        points = np.array([[x, y, z, 1.] for x in (-size[0]/2, size[0]/2)
                           for y in (-size[1]/2, size[1]/2) for z in (-size[2]/2, size[2]/2)])
        return (origin_transform(collision.find('origin')) @ points.T).T[:, :3]


def at_world_anchor(local_fk, world_anchor, anchor):
    to_root = world_anchor @ inverse_rigid(local_fk[anchor])
    return {name: to_root @ transform for name, transform in local_fk.items()}


def transform_points(transforms, points):
    return np.einsum('nij,kj->nki', transforms[:, :3, :3], points) + transforms[:, None, :3, 3]


def read_triangles(path):
    import struct
    path = Path(path)
    if not 84 <= path.stat().st_size <= 64*1024*1024:
        raise ValueError('Binary STL size limit')
    raw = path.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    if not count or len(raw) != 84+50*count:
        raise ValueError('Exact-length binary STL required')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    triangles = np.frombuffer(raw, dtype=dtype, offset=84)['vertices'].astype(float)
    if not np.isfinite(triangles).all():
        raise ValueError('Nonfinite mesh vertices')
    return triangles


def upper_envelope_height(triangles, xy):
    """Highest vertical triangle intersection. NaN means outside all XY projections."""
    triangles, xy = np.asarray(triangles, dtype=float), np.asarray(xy, dtype=float)
    if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or xy.ndim != 2 or xy.shape[1] != 2
            or not np.isfinite(triangles).all() or not np.isfinite(xy).all()):
        raise ValueError('Invalid finite triangle/XY arrays')
    a, b, c = (triangles[:, i] for i in range(3))
    ab, ac = b-a, c-a
    denominator = ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0]
    valid = np.abs(denominator) > 1e-12  # Vertical/degenerate XY faces do not define a height.
    a, ab, ac, denominator = (x[valid] for x in (a, ab, ac, denominator))
    result = np.full(len(xy), np.nan)
    if not len(a):
        return result
    for start in range(0, len(xy), 64):
        delta = xy[start:start+64, None, :] - a[None, :, :2]
        u = (delta[:, :, 0]*ac[:, 1]-delta[:, :, 1]*ac[:, 0])/denominator
        v = (ab[:, 0]*delta[:, :, 1]-ab[:, 1]*delta[:, :, 0])/denominator
        inside = (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)
        z = a[:, 2] + u*ab[:, 2] + v*ac[:, 2]
        height = np.max(np.where(inside, z, -np.inf), axis=1)
        height[~np.isfinite(height)] = np.nan
        result[start:start+64] = height
    return result