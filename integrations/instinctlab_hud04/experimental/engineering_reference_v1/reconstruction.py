"""Engineering reconstruction v1: never recovered mocap annotations or acceptance.

Fixed material frames are derived ON nominal collision geometry, not link pivots.
Support windows are proposals for a NEW trajectory problem, not observed forces.
No motion optimization, runtime configuration edits, pair filtering or GPU calls.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
INTEGRATION = HERE.parents[1]
REPO = HERE.parents[3]
sys.path.insert(0, str(INTEGRATION))
import kinematic_diagnostics as K
from terrain_surface_geometry import SurfaceIndex, point_triangles


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = load_module('engineering_directions', HERE.parent/'direction_solve_20261118/anatomical_directions.py')
CROP = load_module('engineering_crop', HERE.parent/'nonlinear_guard_20261118/box_triangle_guard.py')
TERRAIN_SHA = '3aee33efea6793974d901dd0f7e46859e2d12d2dfd79f38f12c777686566c2de'
TARGET = REPO/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
AUTHORIZATION = {
    'time_utc': '2026-11-18T03:15:07Z',
    'decision': '允许工程重建（推荐）；保留原始动作规格、模型限值和严格物理验收，不把估计当原始标注',
    'scope': 'New engineering definitions allowed, NOT physical acceptance or pair exemptions',
}
RULES = dict(max_surface_distance_m=0.035, max_material_speed_m_s=0.25,
             max_normal_error_deg=45., min_consecutive_frames=4,
             max_window_point_spread_m=0.06, min_upward_surface_normal_z=0.7071067811865476,
             max_tied_normal_disagreement_deg=1., nearest_distance_tie_tolerance_m=1e-9)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame(point, normal, preferred_x):
    """Right-handed material frame; Z outward, X projected preferred tangent."""
    p, z, x = [np.asarray(v, float).copy() for v in (point, normal, preferred_x)]
    if any(v.shape != (3,) for v in (p, z, x)) or not np.isfinite([p, z, x]).all():
        raise ValueError('Finite 3D material frame data required')
    if np.linalg.norm(z) < 1e-12:
        raise ValueError('Nonzero material normal')
    z /= np.linalg.norm(z)
    x -= z*(x@z)
    if np.linalg.norm(x) < 1e-8:
        raise ValueError('Tangent and normal cannot be parallel')
    x /= np.linalg.norm(x)
    t = np.eye(4)
    t[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    t[:3, 3] = p
    if not np.allclose(t[:3, :3].T@t[:3, :3], np.eye(3), atol=1e-12, rtol=0) or np.linalg.det(t[:3, :3]) < 1-1e-12:
        raise ValueError('Material frame SO3')
    return t


def collision(link, index):
    c = link.findall('collision')[index]
    return c, K.origin_transform(c.find('origin')), list(c.find('geometry'))[0]


def capsule_point(link, index, direction):
    """Source ONLY: official popsicle consumer replaces cylinders with capsules."""
    _, t, g = collision(link, index)
    if g.tag != 'cylinder':
        raise ValueError('Reviewed source cylinder-to-capsule only')
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    axis = t[:3, 2]
    # A tie chooses the segment centre, also a valid support point on its side.
    dot = float(axis@d)
    sign = 0 if abs(dot) < 1e-10 else np.sign(dot)
    p = t[:3, 3]+axis*float(g.get('length'))/2*sign+d*float(g.get('radius'))
    local = (p-t[:3, 3])@t[:3, :3]
    nearest = np.array([0., 0., np.clip(local[2], -float(g.get('length'))/2, float(g.get('length'))/2)])
    residual = abs(np.linalg.norm(local-nearest)-float(g.get('radius')))
    return p, residual


def target_hand_face(link, model_path):
    """Largest coplanar NOMINAL convex-hull face; deterministic, not palm sensing."""
    _, t, g = collision(link, 0)
    if g.tag != 'mesh':
        raise ValueError('Reviewed hand mesh required')
    path = (model_path.parent/g.get('filename')).resolve()
    scale = K.vector(g.get('scale', '1 1 1'))
    if np.any(scale <= 0):
        raise ValueError('Positive mesh scale')
    vertices = np.unique(K.read_triangles(path).reshape(-1, 3), axis=0)*scale
    hull = ConvexHull(vertices)
    # Group coplanar Qhull triangles without relying on face enumeration.
    groups = {}
    for triangle, equation in zip(hull.simplices, hull.equations):
        key = tuple(np.round(equation, 9))
        area = float(np.linalg.norm(np.cross(vertices[triangle[1]]-vertices[triangle[0]],
                                           vertices[triangle[2]]-vertices[triangle[0]]))/2)
        group = groups.setdefault(key, dict(area=0., ids=set(), equation=equation))
        group['area'] += area
        group['ids'].update(triangle.tolist())
    key, group = sorted(groups.items(), key=lambda kv: (-kv[1]['area'], kv[0]))[0]
    centre = vertices[sorted(group['ids'])].mean(axis=0)
    normal = group['equation'][:3]
    residual = float(abs(group['equation'][:3]@centre+group['equation'][3]))
    if np.max(hull.equations[:, :3]@centre+hull.equations[:, 3]) > 1e-9:
        raise ValueError('Nominal hull face membership')
    return t[:3, :3]@centre+t[:3, 3], t[:3, :3]@normal, residual, dict(
        mesh_sha256=sha(path), selected_face_area_m2=group['area'],
        selection='largest coplanar nominal convex-hull face; lexicographic plane breaks ties',
        is_measured_anatomical_palm=False, physx_cooked_face_verified=False)


def materials(source_path, target_path):
    models = {kind: D.DirectionModel(path, kind) for kind, path in [('source', source_path), ('target', target_path)]}
    trees = {kind: {x.get('name'): x for x in ET.parse(path).getroot().findall('link')}
             for kind, path in [('source', source_path), ('target', target_path)]}
    zero = {kind: model.fk(np.zeros((1, 29)))[0] for kind, model in models.items()}
    frames, rows = {}, []
    for side in ('left', 'right'):
        for kind in ('sole', 'palm', 'knee'):
            target_link = side+'_'+dict(sole='ankle_roll_link', palm='wrist_roll_link', knee='knee_link')[kind]
            source_link = side+'_'+dict(sole='ankle_roll_link', palm='wrist_yaw_link', knee='knee_link')[kind]
            node = trees['target'][target_link]
            info = {}
            if kind == 'sole':
                _, t, g = collision(node, 0)
                if g.tag != 'box':
                    raise ValueError('Reviewed sole box required')
                normal = t[:3, :3]@np.array([0., 0., -1.])
                point = t[:3, 3]+normal*K.vector(g.get('size'))[2]/2
                residual = 0.
                definition = 'unchanged nominal box bottom-face centre'
            elif kind == 'palm':
                point, normal, residual, info = target_hand_face(node, target_path)
                definition = 'engineered hand contact frame on unchanged nominal convex-hull face, not anatomical palm calibration'
            else:
                choices = [(i, *collision(node, i)[1:]) for i in range(len(node.findall('collision')))]
                i, t, g = max((item for item in choices if item[2].tag == 'sphere'), key=lambda x: x[1][2, 3])
                normal = np.array([1., 0., 0.])
                point = t[:3, 3]+float(g.get('radius'))*normal
                residual = 0.
                definition = 'forward +X point of proximal unchanged knee cap sphere'
                info['target_collision_index'] = i
            tz = zero['target'][target_link][0, :3, :3]
            sz = zero['source'][source_link][0, :3, :3]
            # Align both materials in each model's ZERO pelvis/base anatomical basis.
            ta = zero['target'][models['target'].anchor][0, :3, :3]
            sa = zero['source'][models['source'].anchor][0, :3, :3]
            common_normal = ta.T@tz@normal
            pref = np.array([1., 0., 0.])
            if abs(pref@common_normal) > .95:
                pref = np.array([0., 1., 0.])
            target_frame = frame(point, normal, tz.T@ta@pref)
            source_normal = sz.T@sa@common_normal
            source_index = 3 if kind == 'sole' else 0
            source_point, source_residual = capsule_point(trees['source'][source_link], source_index, source_normal)
            source_frame = frame(source_point, source_normal, sz.T@sa@pref)
            label = side+'_'+kind
            frames[label] = dict(source=(source_link, source_frame), target=(target_link, target_frame))
            rows.append(dict(label=label, source_link=source_link, target_link=target_link,
                source_T_link_material=source_frame.tolist(), target_T_link_material=target_frame.tolist(),
                target_definition=definition, source_definition='fixed engineering support point on consumer capsule; matched ZERO-anatomical normal and tangent',
                source_collision_index=source_index, source_membership_residual_m=float(source_residual),
                target_membership_residual_m=float(residual), source_truth=False, **info))
    if max(max(r['source_membership_residual_m'], r['target_membership_residual_m']) for r in rows) > 1e-8:
        raise ValueError('Material boundary membership failed')
    return models, frames, rows


def runs(mask, minimum):
    mask = np.asarray(mask)
    if mask.ndim != 1 or mask.dtype.kind != 'b' or type(minimum) is not int or minimum < 1:
        raise ValueError('Boolean vector and positive integer minimum')
    edge = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b-1)) for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)) if b-a >= minimum]


def propose_windows(points, normals, index, surface_normals, rules):
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    if rules != RULES:
        raise ValueError('v1 rules are versioned; modifying thresholds requires a new reconstruction version')
    if points.shape != (997, 3) or normals.shape != points.shape or not np.isfinite([points, normals]).all():
        raise ValueError('Full997 finite points and normals required')
    if not np.allclose(np.linalg.norm(normals, axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError('Unit material normals')
    near, triangle, distance, ambiguous = [], [], [], []
    for p in points:
        d, tid, _, target = index.closest(p, p)
        near.append(target); triangle.append(tid); distance.append(d)
        gaps = np.maximum(0, np.maximum(index.lo-p, p-index.hi))
        ids = np.flatnonzero(np.sum(gaps*gaps, axis=1) <= (d+rules['nearest_distance_tie_tolerance_m'])**2)
        witnesses = point_triangles(p, index.triangles[ids])
        tied = ids[np.abs(np.linalg.norm(witnesses-p, axis=1)-d) <= rules['nearest_distance_tie_tolerance_m']]
        ambiguous.append(bool(np.any(surface_normals[tied]@surface_normals[tid] < np.cos(np.deg2rad(rules['max_tied_normal_disagreement_deg'])))))
    near, triangle, distance, ambiguous = np.array(near), np.array(triangle), np.array(distance), np.array(ambiguous)
    # Conservative frame speeds: max of both incident edge speeds, no interpolation/downsampling.
    edge_speed = np.linalg.norm(np.diff(points, axis=0), axis=1)*90
    speed = np.maximum(np.r_[edge_speed[0], edge_speed], np.r_[edge_speed, edge_speed[-1]])
    opposition = -np.einsum('ni,ni->n', normals, surface_normals[triangle])
    eligible = (~ambiguous & (distance <= rules['max_surface_distance_m']) & (speed <= rules['max_material_speed_m_s']) &
                (opposition >= np.cos(np.deg2rad(rules['max_normal_error_deg']))))
    windows, rejected = [], []
    for start, end in runs(eligible, rules['min_consecutive_frames']):
        # Freeze a finite-terrain witness, not an unconstrained infinite plane.
        ids = np.arange(start, end+1)
        anchor_index = int(ids[np.argmin(np.linalg.norm(near[ids]-np.median(near[ids], axis=0), axis=1))])
        anchor = near[anchor_index]
        spread = float(np.linalg.norm(near[ids]-anchor, axis=1).max())
        row = dict(first=start, last=end, target_world_anchor_m=anchor.tolist(),
                   cropped_triangle_id=int(triangle[anchor_index]), max_point_spread_m=spread,
                   max_source_distance_m=float(distance[ids].max()), max_source_speed_m_s=float(speed[ids].max()))
        if spread > rules['max_window_point_spread_m']:
            rejected.append(row | dict(reason='source material drifts beyond fixed engineering anchor budget'))
        else:
            windows.append(row)
    return dict(proposed_windows=windows, rejected_windows=rejected, geometric_eligible_frames=int(eligible.sum()),
                ambiguous_nearest_surface_normals_frames=int(ambiguous.sum()),
                median_distance_m=float(np.median(distance)), min_distance_m=float(distance.min()),
                max_distance_m=float(distance.max()), median_speed_m_s=float(np.median(speed)),
                observed_contact=False, validated_support=False), (distance, speed, opposition, triangle, near, ambiguous)


def build(source_path, motion_path, terrain_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refuse to overwrite reconstruction evidence')
    pins = {Path(source_path): D.SOURCE_SHA, TARGET: D.TARGET_SHA, Path(motion_path): D.MOTION_SHA, Path(terrain_path): TERRAIN_SHA}
    for p, digest in pins.items():
        if sha(p) != digest:
            raise ValueError('Pinned input changed: '+str(p))
    models, local, rows = materials(Path(source_path), TARGET)
    source = models['source']
    with np.load(motion_path, allow_pickle=False) as z:
        q = source.order(z['joint_pos'], z['joint_names'].tolist())
        if len(q) != 997 or int(z['framerate'].item()) != 90:
            raise ValueError('997 native frames at 90Hz required')
        roots = K.world_pose(z['base_pos_w'], z['base_quat_w'], 'wxyz')
    fk, _ = source.fk(q)
    triangles, raw_ids = CROP.cropped_triangles(K.read_triangles(terrain_path), [9., 12.])
    if len(triangles) != 8068:
        raise ValueError('Actual finite crop inventory changed')
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    # Only upward faces propose weight-bearing supports in this version.
    # Side walls/undersides remain mandatory collision constraints, never disappear.
    upward = np.flatnonzero(normals[:, 2] >= RULES['min_upward_surface_normal_z'])
    if not len(upward):
        raise ValueError('No upward support surfaces')
    index = SurfaceIndex(triangles[upward])
    arrays, proposals = {}, {}
    support_count = np.zeros(997, int)
    for label, kinds in local.items():
        link, t = kinds['source']
        world = roots@fk[link]@t
        proposal, detail = propose_windows(world[:, :3, 3], world[:, :3, 2], index, normals[upward], RULES)
        for window in proposal['proposed_windows']+proposal['rejected_windows']:
            window['cropped_triangle_id'] = int(upward[window['cropped_triangle_id']])
            window['raw_stl_triangle_id'] = int(raw_ids[window['cropped_triangle_id']])
        for window in proposal['proposed_windows']:
            support_count[window['first']:window['last']+1] += 1
        proposals[label] = proposal
        for key, value in zip(['distance_m', 'speed_m_s', 'normal_opposition', 'upward_subset_triangle', 'closest_world_m', 'ambiguous_nearest_normal'], detail):
            arrays[label+'__'+key] = value
        arrays[label+'__source_world_material'] = world
    contract = dict(schema='hud04.engineering_reference.v1', status='engineering_definitions_and_PROPOSED_supports_NOT_accepted_reference',
        authorization=AUTHORIZATION, source_provenance_recovered=False,
        input_sha256={str(p): digest for p, digest in pins.items()},
        code_sha256={str(p.relative_to(REPO)): sha(p) for p in [Path(__file__), Path(D.__file__), Path(CROP.__file__),
                                                             Path(K.__file__), INTEGRATION/'terrain_surface_geometry.py']},
        retained=dict(frames=997, framerate_hz=90, body_actions=29, head_policy_dof=0,
                      head_fixed_at_model_zero=True, original_joint_velocity_effort_limits=True, original_geometry_unchanged=True),
        scene=dict(role='NEW engineering world definition; NOT recovered export calibration', units='metres by explicit new engineering convention',
                   source_root_link='torso_link', source_quaternion_order='wxyz', world_up='+Z',
                   source_to_engineering_world=np.eye(4).tolist(), terrain_to_engineering_world=np.eye(4).tolist(),
                   crop_xy_m=[9., 12.], crop_caps_added=False, terrain_triangles=len(triangles),
                   runtime_import_or_scene_equivalence_verified=False,
                   source_units_original_export_verified=False, source_consumer_common_terrain_motion_shift_cancels=True),
        materials=rows, support_proposal_rules=RULES, supports=proposals,
        support_plan_is_observed_contact=False, support_plan_dynamics_verified=False,
        no_proposed_support_does_not_imply_flight=True,
        unassigned_intervals=runs(support_count == 0, 1),
        frames_with_no_proposed_support=int((support_count == 0).sum()),
        frames_with_multiple_proposed_supports=int((support_count > 1).sum()),
        collision_policy=dict(automatic_pair_exclusions=[], collision_exemptions_authorized=[],
            wrist_nominal_all_angle_overlap_requires_runtime_cooking_and_pair_policy_evidence=True,
            original_geometry_edited=False, geometry_proximity_does_not_establish_contact=True,
            all43_shapes_and8068_finite_triangles_still_required=True,
            nominal_invariant_overlap_is_not_a_clearance_or_impossible_robot_motion_claim=True),
        acceptance=dict(accepted_reference=False, training_ready=False, ppo_updates=0,
            required=['35D root+body constrained trajectory solve', 'finite-material target anchoring and no-slip feasibility',
                      'all original position/velocity/effort limits', 'native997 continuity and between-frame collision',
                      'explicit runtime collision pair and cooked geometry evidence',
                      'full-body terrain and self-collision validation without automatic exclusions',
                      '997-frame replay, contact forces, friction, torque and floating-base dynamics',
                      'independent verifier against pinned inputs and recorded plan']),
        counters=dict(gpu_submissions=0, ppo_updates=0, motion_optimizer_calls=0, new_motion_files=0))
    for p, digest in pins.items():
        if sha(p) != digest:
            raise ValueError('Input changed during reconstruction')
    output.mkdir(parents=True)
    (output/'CONTRACT.json').write_text(json.dumps(contract, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    np.savez_compressed(output/'SOURCE_MATERIAL_DIAGNOSTICS.npz', **arrays)
    return contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['source', 'motion', 'terrain', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.motion, args.terrain, args.output)
    print(json.dumps(dict(supports={k: v['proposed_windows'] for k, v in result['supports'].items()},
                         frames_with_no_proposed_support=result['frames_with_no_proposed_support'],
                         accepted_reference=False, training_ready=False), indent=2))