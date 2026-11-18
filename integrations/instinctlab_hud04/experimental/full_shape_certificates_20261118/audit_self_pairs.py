"""Read-only all997/all903 nominal self-pair certificate audit of saved trial005.

No filtering based on adjacency. Same-link compound parts remain records, not
independent rigid-body constraints. Never claims runtime pair filters observed.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from nominal_certificates import Solid, classify

HERE = Path(__file__).resolve().parent
INTEGRATION = HERE.parents[1]
REPO = HERE.parents[3]
TARGET = REPO / 'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
TRIAL = HERE.parent / 'world_candidate_20261118/WORLD_TRIAL_005_FOOT_PRIORITY'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_case(motion=None, motion_sha256=None):
    if (motion is None) != (motion_sha256 is None):
        raise ValueError('Explicit alternate motion requires explicit SHA256')
    if motion_sha256 is not None and (len(motion_sha256)!=64 or any(c not in '0123456789abcdef' for c in motion_sha256)):
        raise ValueError('Require lowercase SHA256')
    motion = Path(motion).resolve() if motion is not None else TRIAL / 'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
    kd = INTEGRATION / 'kinematic_diagnostics.py'
    pins = {TARGET:'865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7',
            kd:'943b63605c4e837451a47eee87fa754ac04d72afcef02fe8904c77252b438696',
            motion:motion_sha256 or '9353b185d92c21ceeb56a289b926f89d3dc72c5343745fc14e26889bc6d728d4'}
    spec = importlib.util.spec_from_file_location('read_only_kd', kd)
    for p, digest in pins.items():
        if sha(p) != digest: raise ValueError('Pinned input changed: ' + str(p))
    K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
    xml = ET.parse(TARGET).getroot(); tree = K.RobotTree.read(TARGET)
    named = [j.get('name') for j in xml.findall('joint') if j.get('type') != 'fixed']
    with np.load(motion, allow_pickle=False) as z:
        names = z['joint_names'].tolist(); q = z['joint_pos'].copy()
        roots = K.world_pose(z['base_pos_w'], z['base_quat_w'], 'wxyz')
        if names != named or q.shape != (997,29) or float(z['framerate'].item()) != 90 or z['accepted_reference'].item() != False:
            raise ValueError('Require original unaccepted native997 motion contract')
        np.testing.assert_array_equal(z['frame_indices'], np.arange(997))
    world = {l:roots @ v for l,v in tree.fk(q,names).items()}
    mesh_pins = {'left':'5aa0af4ccdaf46d498267b0fae247b8d547d8d72dd8db4f7691fe42b567eb5d6',
                 'right':'e4804755e9991f05ad5389340e1e83ee1ef687a403896c21c0f785c28ff871fc'}
    shapes = []
    for link in xml.findall('link'):
        name = link.get('name')
        for i,c in enumerate(link.findall('collision')):
            geom = list(c.find('geometry'))
            if len(geom) != 1: raise ValueError('One geometry per collision required')
            g = geom[0]; v = planes = None
            if g.tag == 'box': dims = K.vector(g.get('size')) / 2
            elif g.tag == 'sphere': dims = [float(g.get('radius'))]
            elif g.tag == 'cylinder': dims = [float(g.get('radius')),float(g.get('length'))/2]
            elif g.tag == 'mesh':
                side = next((s for s in mesh_pins if name == s+'_wrist_roll_link'),None)
                if side is None: raise ValueError('Unreviewed mesh link')
                path = (TARGET.parent/g.get('filename')).resolve(); pins[path] = mesh_pins[side]
                if sha(path) != pins[path]: raise ValueError('Pinned hand mesh changed')
                scale = K.vector(g.get('scale','1 1 1'))
                if np.any(scale <= 0): raise ValueError('Positive mesh scale required')
                raw = np.unique(K.read_triangles(path).reshape(-1,3),axis=0)*scale
                hull = ConvexHull(raw); v = raw[hull.vertices]; planes = hull.equations.copy(); dims = []
            else: raise ValueError('Unsupported geometry')
            s = Solid(name+'#'+str(i),g.tag,dims,v,planes)
            poses = world[name] @ K.origin_transform(c.find('origin'))
            d = np.broadcast_to(np.eye(3),(997,3,3)); lo,hi = s.interval(d,poses)
            shapes.append((name,s,poses,lo,hi))
    if len(shapes) != 43 or Counter(s.kind for _,s,*_ in shapes) != dict(cylinder=24,sphere=14,box=3,mesh=2):
        raise ValueError('Nominal shape inventory mismatch')
    return K,pins,shapes,tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--motion',type=Path)
    parser.add_argument('--motion-sha256')
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(dict(plan_only=True,frames=997,shapes=43,pairs=903,optimizer_calls=0,gpu_submissions=0)))
        return
    if args.output.exists():
        raise FileExistsError('Refuse overwrite; no audit replay into existing evidence')
    start = time.monotonic()
    K,pins,shapes,tree=load_case(args.motion,args.motion_sha256)
    adjacency = {frozenset([j['parent'],j['child']]) for j in tree.joints}
    rows=[]; totals=Counter(); strict_frames=set(); nonadj_strict_frames=set()
    for i,j in combinations(range(43),2):
        la,a,ta,alo,ahi=shapes[i]; lb,b,tb,blo,bhi=shapes[j]
        adjacent=frozenset([la,lb]) in adjacency; same=la==lb
        overlap=np.all(ahi >= blo-1e-8,axis=1)&np.all(bhi >= alo-1e-8,axis=1)
        ids=np.flatnonzero(overlap)
        row=dict(first=a.identity,second=b.identity,same_rigid_link=same,directly_joint_connected=adjacent,
                 automatically_excluded_by_adjacency=False,actual_pair_filter_observed=False,
                 aabb_candidate_frames=len(ids),same_link_compound_record_only=same)
        if same:
            totals['same_link_compound_pose_records']+=997
            row['status']='same_rigid_link_compound_parts_not_independent_bodies'
        else:
            totals['different_link_pose_pairs']+=997
            totals['aabb_separated_pose_pairs']+=997-len(ids)
            row.update(aabb_separated_pose_pairs=997-len(ids),support_separated_frames=[],strict_common_point_certificates=[],unresolved_frames=[])
            if len(ids):
                result=classify(a,ta[ids],b,tb[ids])
                sep=ids[result['separated']]; hit=ids[result['strict']]; unresolved=ids[result['unresolved']]
                row['support_separated_frames']=sep.tolist();row['unresolved_frames']=unresolved.tolist()
                totals['support_separated_pose_pairs']+=len(sep)
                totals['strict_common_point_pose_pairs']+=len(hit)
                totals['unresolved_pose_pairs']+=len(unresolved)
                if not adjacent:totals['nonadjacent_strict_common_point_pose_pairs']+=len(hit)
                strict_frames.update(hit.tolist())
                if not adjacent: nonadj_strict_frames.update(hit.tolist())
                for k in np.flatnonzero(result['strict']):
                    row['strict_common_point_certificates'].append(dict(frame=int(ids[k]),point=result['common_point'][k].tolist(),
                        first_inside_margin_raw=float(result['first_inside_margin'][k]),second_inside_margin_raw=float(result['second_inside_margin'][k])))
            row['status']='all_poses_classified_not_all_clear'
        rows.append(row)
    accounted=sum(totals[k] for k in ['aabb_separated_pose_pairs','support_separated_pose_pairs','strict_common_point_pose_pairs','unresolved_pose_pairs'])
    if accounted != totals['different_link_pose_pairs'] or len(rows)!=903: raise ArithmeticError('Incomplete pair accounting')
    for p,digest in pins.items():
        if sha(p)!=digest:raise ValueError('Input changed during audit')
    result=dict(stage='all997_all903_nominal_self_pair_finite_certificates_not_acceptance',
        base_commit='86f05dffa8696465147d7e3ebf99aff568a84c26',input_sha256={str(p.relative_to(REPO)):d for p,d in pins.items()},
        code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},frames=997,shapes=43,shape_pairs=903,
        tolerance_raw=1e-8,projection_witness_search_iterations_cap=64,counts=dict(totals),
        frames_with_strict_common_point=sorted(strict_frames),nonadjacent_frames_with_strict_common_point=sorted(nonadj_strict_frames),
        pairs=rows,all997_all903_accounted=True,all_narrowphase_resolved=totals['unresolved_pose_pairs']==0,
        all_different_body_pairs_clear=False,adjacency_auto_exclusions=0,cooked_geometry_verified=False,
        actual_pair_filters_verified=False,terrain_checked=False,swept_geometry_checked=False,dynamics_checked=False,
        penetration_depth_computed=False,root_world_units_inferred=False,contact_labels_inferred=False,
        motion_optimizer_calls=0,new_motion_files=0,gpu_submissions=0,ppo_updates=0,
        accepted_reference=False,training_ready=False,elapsed_s=time.monotonic()-start)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['frames','shapes','shape_pairs','counts','elapsed_s','accepted_reference']},indent=2),flush=True)


if __name__=='__main__':main()