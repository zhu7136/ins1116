"""Engineering material goals for a bounded, explicitly unaccepted 35D solve.

No contact/flight label is created for the 346 unassigned frames. Existing
24 windows are soft design goals, NOT physical support acceptance.
"""
import ast
import hashlib
import importlib.util
import json
from itertools import combinations
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
EXPERIMENTAL = HERE.parent
INTEGRATION = HERE.parents[1]
REPO = HERE.parents[3]
sys.path[:0] = [str(EXPERIMENTAL/'full_shape_certificates_20261118'), str(INTEGRATION)]
from wholebody_step_guard import WholeBodyStepGuard
from nominal_certificates import rigid
import kinematic_diagnostics as K

CONTRACT = EXPERIMENTAL/'engineering_reference_v1/RECONSTRUCTION_002/CONTRACT.json'
DIAGNOSTICS = CONTRACT.with_name('SOURCE_MATERIAL_DIAGNOSTICS.npz')
SEED = EXPERIMENTAL/'wholebody_guarded_candidates_20261118/WHOLEBODY_TRIAL_008/STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
SEED_SHA = 'd5cdb5382b5eb910b3273adab662de286e0452376ee9d134124d4ceddd2f30cf'
TERRAIN = Path('/workspace/hud04-original-drive-20261118/50cmboxes_19_36_49.stl')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def perturb(roots, delta):
    roots = rigid(roots); delta = np.asarray(delta, float)
    if delta.shape != (len(roots), 6) or not np.isfinite(delta).all():
        raise ValueError('Finite six-dimensional tangent increments required')
    out = roots.copy(); out[:, :3, 3] += delta[:, :3]
    out[:, :3, :3] = Rotation.from_rotvec(delta[:, 3:]).as_matrix() @ roots[:, :3, :3]
    return rigid(out)


class Materials:
    def __init__(self, model, rows):
        self.model = model; self.names = model.names
        self.rows = rows
        self.labels = [r['label'] for r in rows]
        if len(rows) != 6 or len(set(self.labels)) != 6:
            raise ValueError('Six unique engineering materials required')

    def evaluate(self, q, roots, jacobian=False):
        q = self.model.order(q, self.names); roots = rigid(roots)
        if len(q) != len(roots): raise ValueError('One root per frame')
        local, axes = self.model.fk(q)
        world = {name: roots @ pose for name, pose in local.items()}
        t = np.stack([world[r['target_link']] @ np.array(r['target_T_link_material']) for r in self.rows], axis=1)
        # Six material points + outward material normals; never link pivots.
        p, normal = t[:, :, :3, 3], t[:, :, :3, 2]
        value = np.concatenate([p.reshape(len(q), 18), normal.reshape(len(q), 18)], axis=1)
        if not jacobian: return value
        jp = np.zeros((len(q), 6, 3, 35)); jn = jp.copy()
        jp[:, :, :, :3] = np.eye(3)
        for c in range(3):
            jp[:, :, :, c+3] = np.cross(np.eye(3)[c], p-roots[:, None, :3, 3])
            jn[:, :, :, c+3] = np.cross(np.eye(3)[c], normal)
        for c, name in enumerate(self.names):
            pivot, axis = axes[name]
            pivot = np.einsum('nij,nj->ni', roots[:, :3, :3], pivot)+roots[:, :3, 3]
            axis = np.einsum('nij,nj->ni', roots[:, :3, :3], axis)
            for k, row in enumerate(self.rows):
                if name in self.model.ancestors[row['target_link']]:
                    jp[:, k, :, c+6] = np.cross(axis, p[:, k]-pivot)
                    jn[:, k, :, c+6] = np.cross(axis, normal[:, k])
        return value, np.concatenate([jp.reshape(len(q), 18, 35), jn.reshape(len(q), 18, 35)], axis=1)


def geometry_kernel(guard):
    """Execute only pinned pure numerical definitions with explicit inputs.

    Does not import obsolete private-workbench loaders, trimesh, or old runners.
    Solid interval conventions are converted explicitly and tested independently.
    """
    path = EXPERIMENTAL/'wholebody_guarded_candidates_20261118/wholebody_terms.py'
    if sha(path) != '9f3c10fc8f7ccc4ec628711bfc5cb4de5c2bc33f4bb13c3aae523ab5555368f8':
        raise ValueError('Pinned geometry kernel changed')
    shapes = []
    for link, solid, local in guard.inventory:
        dims = solid.dimensions
        params = ({'size': 2*dims} if solid.kind == 'box' else
                  {'radius': dims[0], 'length': 2*dims[1]} if solid.kind == 'cylinder' else
                  {'radius': dims[0]} if solid.kind == 'sphere' else {})
        shapes.append(SimpleNamespace(identity=solid.identity, link=link, kind=solid.kind,
                                      local=local, parameters=params, vertices=solid.vertices))
    scope = dict(np=np, sparse=sparse, combinations=combinations,
                 load_shapes=lambda: (guard.tree, shapes, {'runtime_verified': False}),
                 ActualTerrain=lambda: SimpleNamespace(triangles=guard.triangles))
    definitions = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), scope)
    return scope['WholeBodyTerms'](), scope


def problem():
    c = json.loads(CONTRACT.read_text())
    for path, digest in c['input_sha256'].items():
        if sha(path) != digest: raise ValueError('Original input changed: '+path)
    for path, digest in c['code_sha256'].items():
        if sha(REPO/path) != digest: raise ValueError('Definition dependency changed: '+path)
    directions = module('material_trajectory_directions', EXPERIMENTAL/'direction_solve_20261118/anatomical_directions.py')
    target = next(Path(p) for p in c['input_sha256'] if 'fixed_head' in p)
    model = Materials(directions.DirectionModel(target, 'target', wrist_endpoint='first_wrist_pivot'), c['materials'])
    if sha(SEED) != SEED_SHA: raise ValueError('Seed changed')
    guard = WholeBodyStepGuard(motion=SEED, motion_sha256=SEED_SHA, terrain=TERRAIN)
    with np.load(SEED, allow_pickle=False) as z:
        if z['joint_names'].tolist() != model.names or float(z['framerate']) != 90:
            raise ValueError('Native seed identity mismatch')
        q = model.model.order(z['joint_pos'], z['joint_names'].tolist())
        roots = K.world_pose(z['base_pos_w'], z['base_quat_w'], 'wxyz')
    with np.load(DIAGNOSTICS, allow_pickle=False) as z:
        source = np.stack([z[label+'__source_world_material'] for label in model.labels], axis=1)
    desired = np.concatenate([source[:, :, :3, 3].reshape(997, 18), source[:, :, :3, 2].reshape(997, 18)], axis=1)
    weights = np.tile(np.r_[np.full(18, 2.), np.full(18, .1)], (997, 1))
    support = np.zeros((997, 6), bool)
    normals = np.cross(guard.triangles[:, 1]-guard.triangles[:, 0], guard.triangles[:, 2]-guard.triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    for k, label in enumerate(model.labels):
        for row in c['supports'][label]['proposed_windows']:
            fs = slice(row['first'], row['last']+1)
            desired[fs, 3*k:3*k+3] = row['target_world_anchor_m']
            desired[fs, 18+3*k:21+3*k] = -normals[row['cropped_triangle_id']]
            weights[fs, 3*k:3*k+3] = 100.
            weights[fs, 18+3*k:21+3*k] = 2.
            support[fs, k] = True
    if int((~support.any(axis=1)).sum()) != 346:
        raise ValueError('The unassigned frames must not be silently filled or marked flight')
    body, kernel = geometry_kernel(guard)
    return model, q, roots, desired, weights, support, body, guard, kernel


def feasibility(model, q):
    m = model.model
    return dict(position_violation_rad=float(max(0, np.max(m.lower-q), np.max(q-m.upper))),
                velocity_violation_rad_s=float(max(0, np.max(abs(np.diff(q, axis=0))*90-m.vmax))))


def hard_rows(model, q, first, last):
    """Original position/velocity inequalities, including both fixed neighbours."""
    if q.shape != (997, 29) or not 0 <= first <= last < 997 or last-first > 92:
        raise ValueError('Native bounded window required')
    n = last-first+1; nv = n*35
    select = sparse.coo_matrix((np.ones(n*29), (np.arange(n*29),
        [i*35+6+j for i in range(n) for j in range(29)])), shape=(n*29, nv)).tocsr()
    rows, cols, data, offsets = [], [], [], []
    for edge in range(max(0, first-1), min(996, last+1)):
        for j in range(29):
            constant = 0.
            for frame, sign in [(edge, -1.), (edge+1, 1.)]:
                if first <= frame <= last:
                    rows.append(len(offsets)); cols.append((frame-first)*35+6+j); data.append(sign)
                else: constant += sign*q[frame, j]
            offsets.append(constant+sum(sign*q[frame, j] for frame, sign in [(edge, -1.), (edge+1, 1.)] if first <= frame <= last))
    diff = sparse.coo_matrix((data, (rows, cols)), shape=(len(offsets), nv)).tocsr()
    offsets = np.array(offsets); speed = np.tile(model.model.vmax/90, len(offsets)//29)
    return sparse.vstack([select, -select, diff, -diff], format='csr'), np.r_[
        (model.model.upper-q[first:last+1]).reshape(-1), (q[first:last+1]-model.model.lower).reshape(-1),
        speed-offsets, speed+offsets]