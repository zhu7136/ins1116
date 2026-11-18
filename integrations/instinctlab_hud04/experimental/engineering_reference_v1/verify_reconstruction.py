"""Independent output reader: does not import/run the reconstruction generator.

Recomputes source world materials using RobotTree (not DirectionModel), checks
nominal geometry and each finite terrain anchor. Not a physics verifier.
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
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1]))
import kinematic_diagnostics as K
from terrain_surface_geometry import point_triangles, SurfaceIndex

PINNED = dict(source='5e99d930af64f3f0bfd34235239e288cc380ba62c8b01fcfcf8cc9f0996e7fa4',
              target='865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7',
              motion='56158e8679580910ccff375963d025b138636acfe70f5b1f88381837209b2056',
              terrain='3aee33efea6793974d901dd0f7e46859e2d12d2dfd79f38f12c777686566c2de')


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, reason):
    if not ok: raise ValueError(reason)


def rigid(t):
    require(t.shape == (4, 4) and np.isfinite(t).all(), 'finite material transform')
    require(np.allclose(t[3], [0, 0, 0, 1], atol=1e-12, rtol=0), 'material homogeneous row')
    require(np.allclose(t[:3, :3].T@t[:3, :3], np.eye(3), atol=1e-10, rtol=0), 'orthogonal material')
    require(abs(np.linalg.det(t[:3, :3])-1) < 1e-10, 'proper material rotation')


def check_definition(c):
    require(c['schema'] == 'hud04.engineering_reference.v1', 'schema')
    require(c['status'] == 'engineering_definitions_and_PROPOSED_supports_NOT_accepted_reference', 'no status promotion')
    require(c['acceptance']['accepted_reference'] is False and c['acceptance']['training_ready'] is False, 'not acceptance')
    require(c['support_plan_is_observed_contact'] is False and c['support_plan_dynamics_verified'] is False, 'no fabricated contact')
    require(c['collision_policy']['automatic_pair_exclusions'] == [] and c['collision_policy']['collision_exemptions_authorized'] == [], 'no pair exemptions')
    require(c['no_proposed_support_does_not_imply_flight'] is True, 'unassigned is not flight')
    require(c['retained'] == dict(frames=997, framerate_hz=90, body_actions=29, head_policy_dof=0,
        head_fixed_at_model_zero=True, original_joint_velocity_effort_limits=True, original_geometry_unchanged=True), 'retained specification')
    expected = dict(max_surface_distance_m=.035, max_material_speed_m_s=.25,
                    max_normal_error_deg=45., min_consecutive_frames=4,
                    max_window_point_spread_m=.06, min_upward_surface_normal_z=0.7071067811865476,
                    max_tied_normal_disagreement_deg=1., nearest_distance_tie_tolerance_m=1e-9)
    require(c['support_proposal_rules'] == expected, 'unmodified versioned proposal rules')
    scene = c['scene']
    require(scene['source_root_link'] == 'torso_link' and scene['source_quaternion_order'] == 'wxyz', 'source consumer convention')
    require(scene['world_up'] == '+Z' and scene['crop_xy_m'] == [9., 12.], 'explicit scene crop and up')
    require(scene['source_to_engineering_world'] == np.eye(4).tolist() and scene['terrain_to_engineering_world'] == np.eye(4).tolist(), 'v1 identity correspondence, not fitted alignment')
    require(scene['crop_caps_added'] is False and scene['terrain_triangles'] == 8068, 'unchanged uncapped crop')
    require(c['source_provenance_recovered'] is False and scene['source_units_original_export_verified'] is False, 'new engineering units, not recovered provenance')


def verify(directory, source, motion, terrain):
    directory, source, motion, terrain = map(Path, (directory, source, motion, terrain))
    c = json.loads((directory/'CONTRACT.json').read_text())
    check_definition(c)
    target = REPO/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
    for role, p in dict(source=source, target=target, motion=motion, terrain=terrain).items():
        require(digest(p) == PINNED[role], 'independent input pin '+role)
        require(c['input_sha256'].get(str(p)) == PINNED[role], 'manifest input pin '+role)
    for relative, sha in c['code_sha256'].items():
        p = (REPO/relative).resolve()
        require(p.is_relative_to(REPO) and digest(p) == sha, 'generator code pin')
    guard = HERE.parent/'nonlinear_guard_20261118/box_triangle_guard.py'
    spec = importlib.util.spec_from_file_location('verifier_crop', guard)
    cropper = importlib.util.module_from_spec(spec); spec.loader.exec_module(cropper)
    triangles, raw_ids = cropper.cropped_triangles(K.read_triangles(terrain), [9., 12.])
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    upward = np.flatnonzero(normal[:, 2] >= c['support_proposal_rules']['min_upward_surface_normal_z'])
    index = SurfaceIndex(triangles[upward])
    source_tree = K.RobotTree.read(source)
    target_xml = ET.parse(target).getroot()
    source_xml = ET.parse(source).getroot()
    require(all(any(j.get('name') == name and j.get('type') == 'fixed' for j in target_xml.findall('joint'))
                for name in ['head_yaw_joint', 'head_pitch_joint']), 'fixed model-zero head')
    links = {kind: {l.get('name'): l for l in tree.findall('link')}
             for kind, tree in [('source', source_xml), ('target', target_xml)]}
    with np.load(motion, allow_pickle=False) as z:
        require(z['joint_pos'].shape == (997, 29) and int(z['framerate'].item()) == 90, 'native997')
        fk = source_tree.fk(z['joint_pos'], z['joint_names'].tolist())
        roots = np.tile(np.eye(4), (997, 1, 1))
        roots[:, :3, :3] = Rotation.from_quat(z['base_quat_w'][:, [1, 2, 3, 0]]).as_matrix()
        roots[:, :3, 3] = z['base_pos_w']
    r = c['support_proposal_rules']; windows = 0; count = np.zeros(997, int)
    max_material_residual = 0.; max_world_residual = 0.; max_anchor_residual = 0.; tied_face_ids = 0
    with np.load(directory/'SOURCE_MATERIAL_DIAGNOSTICS.npz', allow_pickle=False) as z:
        for row in c['materials']:
            require(row['source_truth'] is False, 'engineering material provenance')
            label = row['label']
            for kind in ['source', 'target']:
                t = np.array(row[kind+'_T_link_material']); rigid(t)
                link = row[kind+'_link']
                idx = row['source_collision_index'] if kind == 'source' else row.get('target_collision_index', 0)
                node = links[kind][link].findall('collision')[idx]
                origin = K.origin_transform(node.find('origin')); g = list(node.find('geometry'))[0]
                p = (t[:3, 3]-origin[:3, 3])@origin[:3, :3]
                if kind == 'source':
                    require(g.tag == 'cylinder', 'source consumer capsule')
                    half = float(g.get('length'))/2
                    closest = np.array([0, 0, np.clip(p[2], -half, half)])
                    residual = abs(np.linalg.norm(p-closest)-float(g.get('radius')))
                elif g.tag == 'sphere':
                    residual = abs(np.linalg.norm(p)-float(g.get('radius')))
                elif g.tag == 'box':
                    h = K.vector(g.get('size'))/2
                    residual = abs(np.max(abs(p)-h))
                elif g.tag == 'mesh':
                    v = K.read_triangles(target.parent/g.get('filename')).reshape(-1, 3)*K.vector(g.get('scale'))
                    eq = ConvexHull(np.unique(v, axis=0)).equations
                    residual = abs((eq[:, :3]@p+eq[:, 3]).max())
                else: raise ValueError('unreviewed target material geometry')
                max_material_residual = max(max_material_residual, float(residual))
                require(residual < 1e-8, 'material on unchanged nominal geometry')
            source_t = np.array(row['source_T_link_material'])
            world = roots@fk[row['source_link']]@source_t
            err = float(np.max(abs(world-z[label+'__source_world_material'])))
            max_world_residual = max(max_world_residual, err)
            require(err < 1e-8, 'independent source FK')
            p = world[:, :3, 3]
            edge_speed = np.linalg.norm(np.diff(p, axis=0), axis=1)*90
            speed = np.maximum(np.r_[edge_speed[0], edge_speed], np.r_[edge_speed, edge_speed[-1]])
            require(np.max(abs(speed-z[label+'__speed_m_s'])) < 1e-8, 'independent material velocity')
            # Do not trust cached eligibility or nearest-surface measurements.
            for f, point in enumerate(p):
                distance, triangle_id, _, nearest = index.closest(point, point)
                require(abs(distance-z[label+'__distance_m'][f]) < 1e-8, 'recomputed nearest finite surface distance')
                cached_id = int(z[label+'__upward_subset_triangle'][f])
                require(0 <= cached_id < len(upward), 'cached triangle bounds')
                cached_nearest = point_triangles(point, triangles[upward[cached_id]:upward[cached_id]+1])[0]
                require(abs(np.linalg.norm(point-cached_nearest)-distance) < 1e-9, 'cached triangle actually attains global nearest distance')
                tied_face_ids += int(triangle_id != cached_id)
                require(np.linalg.norm(cached_nearest-z[label+'__closest_world_m'][f]) < 1e-8, 'recomputed cached finite witness')
                gaps = np.maximum(0, np.maximum(index.lo-point, point-index.hi))
                ids = np.flatnonzero(np.sum(gaps*gaps, axis=1) <= (distance+r['nearest_distance_tie_tolerance_m'])**2)
                closest = point_triangles(point, index.triangles[ids])
                tied = ids[abs(np.linalg.norm(closest-point, axis=1)-distance) <= r['nearest_distance_tie_tolerance_m']]
                ambiguous = bool(np.any(normal[upward[tied]]@normal[upward[cached_id]] < np.cos(np.deg2rad(r['max_tied_normal_disagreement_deg']))))
                require(ambiguous == bool(z[label+'__ambiguous_nearest_normal'][f]), 'independent nearest-face normal ambiguity')
                opposition = -float(world[f, :3, 2]@normal[upward[cached_id]])
                require(abs(opposition-z[label+'__normal_opposition'][f]) < 1e-8, 'recomputed normal opposition')
            proposal = c['supports'][label]
            require(proposal['observed_contact'] is False and proposal['validated_support'] is False, 'proposal only')
            previous = -1
            for window in proposal['proposed_windows']:
                a, b = window['first'], window['last']
                require(type(a) is int and type(b) is int and previous < a <= b < 997, 'ordered inclusive window')
                require(b-a+1 >= r['min_consecutive_frames'], 'minimum dwell')
                previous = b; windows += 1; count[a:b+1] += 1
                require(np.max(z[label+'__distance_m'][a:b+1]) <= r['max_surface_distance_m'], 'support proximity proposal bound')
                require(np.max(z[label+'__speed_m_s'][a:b+1]) <= r['max_material_speed_m_s'], 'support speed proposal bound')
                require(not np.any(z[label+'__ambiguous_nearest_normal'][a:b+1]), 'ambiguous normals cannot become assigned support')
                require(np.min(z[label+'__normal_opposition'][a:b+1]) >= np.cos(np.deg2rad(r['max_normal_error_deg'])), 'support normal proposal bound')
                tid = window['cropped_triangle_id']; anchor = np.array(window['target_world_anchor_m'])
                require(int(raw_ids[tid]) == window['raw_stl_triangle_id'], 'raw finite triangle ID')
                nearest = point_triangles(anchor, triangles[tid:tid+1])[0]
                err = float(np.linalg.norm(nearest-anchor)); max_anchor_residual = max(max_anchor_residual, err)
                require(err < 1e-8, 'anchor on finite triangle, not its infinite plane')
                require(normal[tid, 2] >= r['min_upward_surface_normal_z'], 'upward witness')
                spread = np.linalg.norm(z[label+'__closest_world_m'][a:b+1]-anchor, axis=1).max()
                require(spread <= r['max_window_point_spread_m'], 'fixed anchor spread')
    require(int((count == 0).sum()) == c['frames_with_no_proposed_support'], 'unassigned frame accounting')
    require(int((count > 1).sum()) == c['frames_with_multiple_proposed_supports'], 'multiple support accounting')
    return dict(stage='independent_reconstruction_GEOMETRY_only_not_physics', passed=True, materials=len(c['materials']),
                proposed_windows=windows, unassigned_frames=int((count == 0).sum()),
                max_material_membership_residual_m=max_material_residual,
                max_source_world_transform_component_difference=max_world_residual,
                max_finite_triangle_anchor_residual_m=max_anchor_residual,
                independent_FK_reader='RobotTree, not generator DirectionModel',
                all5982_material_frame_surface_queries_recomputed=True,
                equivalent_nearest_face_id_ties=tied_face_ids,
                verifier_sha256=digest(__file__), accepted_reference=False, training_ready=False,
                dynamics_verified=False, cooked_geometry_verified=False, gpu_submissions=0, ppo_updates=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['directory', 'source', 'motion', 'terrain', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'do not overwrite verification evidence')
    result = verify(args.directory, args.source, args.motion, args.terrain)
    with args.output.open('x') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))