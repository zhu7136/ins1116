"""Read-only collision support geometry, not contact labels or collision distances.

Capsule height is the line-segment length, matching Isaac Sim 5.1.0's URDF
importer (47d886f2858d1ceed556b21c88927aa67bc81c12, UrdfImporter.cpp:229-238,
277-283). Cylinder replacement MUST be supplied explicitly for each asset.
Mesh support is that of the input vertices / their convex hull, NOT a cooked
PhysX mesh or a calibrated anatomical palm. No simulator or solver is invoked.
"""
import hashlib
from pathlib import Path

import numpy as np

import kinematic_diagnostics as K


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Expected finite positive geometry dimension')
    return value


def collisions(urdf_path, link, *, replace_cylinders, allowed_meshes=None):
    """All colliders on one named link; refuse missing/unsupported geometry."""
    if type(replace_cylinders) is not bool:
        raise ValueError('Explicit boolean cylinder replacement required')
    path = Path(urdf_path).resolve()
    tree = K.RobotTree.read(path)
    nodes = tree.links[link].findall('collision')
    if not nodes or len(nodes) > 64:
        raise ValueError('Expected 1..64 colliders on requested link')
    result = []
    for i, node in enumerate(nodes):
        geometry = node.find('geometry')
        if geometry is None or len(geometry) != 1:
            raise ValueError('Expected exactly one geometry per collider')
        shape = geometry[0]
        kind = shape.tag
        item = dict(name=node.get('name', str(i)), urdf_kind=kind, kind=kind,
                    transform=K.origin_transform(node.find('origin')))
        if kind == 'box':
            size = K.vector(shape.get('size', ''))
            if np.any(size <= 0):
                raise ValueError('Expected positive box dimensions')
            item['size'] = size
        elif kind in ('cylinder', 'capsule', 'sphere'):
            item['radius'] = positive(shape.get('radius'))
            if kind != 'sphere':
                item['length'] = positive(shape.get('length'))
            if kind == 'cylinder' and replace_cylinders:
                item['kind'] = 'capsule'
        elif kind == 'mesh':
            filename = shape.get('filename', '')
            if not filename or '://' in filename or Path(filename).is_absolute():
                raise ValueError('Only explicit pinned local meshes supported')
            mesh = (path.parent / filename).resolve()
            allowed = {Path(p).resolve(): digest for p, digest in (allowed_meshes or {}).items()}
            if mesh not in allowed or sha256(mesh) != allowed[mesh]:
                raise ValueError('Missing/mismatching mesh SHA256 allowlist')
            if mesh.stat().st_size > 4*1024*1024:
                raise ValueError('Contact mesh exceeds diagnostic size budget')
            scale = K.vector(shape.get('scale', '1 1 1'))
            if np.any(scale <= 0):
                raise ValueError('Expected positive mesh scale')
            vertices = K.read_triangles(mesh).reshape(-1, 3) * scale
            item.update(vertices=vertices, mesh_sha256=allowed[mesh],
                        vertex_count=len(vertices), scale=scale)
        else:
            raise ValueError('Unsupported collision geometry: ' + kind)
        result.append(item)
    return result


def support(primitive, directions):
    """Link-local extremal points for 1..4096 nonzero directions (rows)."""
    directions = np.asarray(directions, dtype=float)
    if (directions.ndim != 2 or directions.shape[1] != 3 or not 1 <= len(directions) <= 4096
            or not np.isfinite(directions).all() or np.any(np.linalg.norm(directions, axis=1) < 1e-12)):
        raise ValueError('Expected bounded finite nonzero direction rows')
    t = primitive['transform']
    d = directions @ t[:3, :3]
    kind = primitive['kind']
    if kind == 'box':
        point = np.sign(d) * primitive['size']/2
    elif kind == 'sphere':
        point = primitive['radius'] * d / np.linalg.norm(d, axis=1, keepdims=True)
    elif kind == 'capsule':
        point = primitive['radius'] * d / np.linalg.norm(d, axis=1, keepdims=True)
        point[:, 2] += np.sign(d[:, 2]) * primitive['length']/2
    elif kind == 'cylinder':
        point = np.zeros_like(d)
        radial = np.linalg.norm(d[:, :2], axis=1, keepdims=True)
        point[:, :2] = primitive['radius'] * np.divide(d[:, :2], radial,
            out=np.zeros_like(d[:, :2]), where=radial > 0)
        point[:, 2] = np.sign(d[:, 2]) * primitive['length']/2
    elif kind == 'mesh':
        vertices = primitive['vertices']
        point = np.concatenate([vertices[np.argmax(d[i:i+64] @ vertices.T, axis=1)]
                                for i in range(0, len(d), 64)])
    else:
        raise ValueError('Unsupported primitive')
    return point @ t[:3, :3].T + t[:3, 3]


def bounds(primitives):
    """Exact axis-aligned bounds of the represented union, not a contact patch."""
    if not primitives:
        raise ValueError('Empty collision union')
    directions = np.concatenate([np.eye(3), -np.eye(3)])
    extrema = np.array([np.sum(support(p, directions)*directions, axis=1) for p in primitives]).max(axis=0)
    return np.array([-extrema[3:], extrema[:3]])


def describe(primitives):
    aabb = bounds(primitives)
    descriptions = []
    for item in primitives:
        descriptions.append({key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in item.items() if key != 'vertices'})
    return dict(collision_count=len(primitives), primitives=descriptions,
                local_bounds=aabb.tolist(), local_bounds_center=aabb.mean(axis=0).tolist(),
                anatomical_frame_accepted=False, cooked_runtime_geometry_verified=False,
                contact_label=None, signed_collision_distance=False)