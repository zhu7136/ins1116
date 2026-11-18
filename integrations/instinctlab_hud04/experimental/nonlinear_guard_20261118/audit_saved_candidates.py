"""Deterministic CPU checks of existing REJECTED candidates, no optimization."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from box_triangle_guard import BoxTriangleGuard,cropped_triangles,clip_polygon

HERE=Path(__file__).resolve().parent
INTEGRATION=HERE.parents[1];REPO=HERE.parents[3]
TARGET=REPO/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
PINS={
    'WORLD_TRIAL_002_BLOCK31':'7c2c98d7811e3c847c7a20239612006647d56f42777a1ddea914c52f2711f253',
    'WORLD_TRIAL_004_FOOT_TEMPORAL_BLOCK11':'8788331db4442758ca2b0c03747ce25bb902e22b743618b2d6cd35376816701b',
    'WORLD_TRIAL_005_FOOT_PRIORITY':'9353b185d92c21ceeb56a289b926f89d3dc72c5343745fc14e26889bc6d728d4'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terrain',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:
        print(json.dumps(dict(plan_only=True,existing_candidates=len(PINS),frames_each=997,optimizer_calls=0,gpu_submissions=0)));return
    if args.output.exists():raise FileExistsError('Refuse output overwrite')
    start=time.monotonic();inputs={str(TARGET):'865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7',
        str(args.terrain):'3aee33efea6793974d901dd0f7e46859e2d12d2dfd79f38f12c777686566c2de',
        str(INTEGRATION/'kinematic_diagnostics.py'):'943b63605c4e837451a47eee87fa754ac04d72afcef02fe8904c77252b438696'}
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Input digest mismatch: '+path)
    spec=importlib.util.spec_from_file_location('independent_kd',INTEGRATION/'kinematic_diagnostics.py');K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
    raw=K.read_triangles(args.terrain);tri,source_ids=cropped_triangles(raw,[9,12])
    historic=json.loads((INTEGRATION/'PROCESSED_TERRAIN_SURFACES_20261117.json').read_text())['terrain']
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2
    area_error=float(abs(area-historic['total_area_raw_squared']))
    bounds=np.stack([tri.min(axis=(0,1)),tri.max(axis=(0,1))]);bounds_error=float(abs(bounds-np.array(historic['crop_bounds_raw'])).max())
    if area_error>1e-10 or bounds_error>1e-10:raise ValueError('Independent crop area/bounds disagreement')
    guard=BoxTriangleGuard(tri,tolerance=1e-9);xml=ET.parse(TARGET).getroot();tree=K.RobotTree.read(TARGET)
    named=[j.get('name') for j in xml.findall('joint') if j.get('type')!='fixed'];reviews=[]
    for trial,digest in PINS.items():
        folder=HERE.parent/'world_candidate_20261118'/trial;path=folder/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
        inputs[str(path)]=digest
        if sha(path)!=digest:raise ValueError('Candidate digest mismatch')
        with np.load(path,allow_pickle=False) as z:
            names=z['joint_names'].tolist();q=z['joint_pos'];p=z['base_pos_w'];quat=z['base_quat_w']
            if names!=named or q.shape!=(997,29) or float(z['framerate'].item())!=90 or z['accepted_reference'].item()!=False:
                raise ValueError('Unaccepted native997 contract required')
            np.testing.assert_array_equal(z['frame_indices'],np.arange(997))
        world={k:K.world_pose(p,quat,'wxyz')@v for k,v in tree.fk(q,names).items()};sides={}
        prior=json.loads((folder/'INDEPENDENT_WORLD_CANDIDATE_REVIEW.json').read_text())
        for side in ['left','right']:
            link=side+'_ankle_roll_link';node=next(l for l in xml.findall('link') if l.get('name')==link)
            boxes=[c for c in node.findall('collision') if c.find('geometry/box') is not None]
            if len(boxes)!=1:raise ValueError('One nominal footbox required')
            box=boxes[0];h=np.fromstring(box.find('geometry/box').get('size'),sep=' ')/2
            poses=world[link]@K.origin_transform(box.find('origin'))
            result=guard.scan(poses,h,pose_ids=[side+':'+str(f) for f in range(997)])
            # Independent geometric witness: clipped-triangle centroid strictly
            # inside the six box planes, not the SAT projection implementation.
            witness_checks=[]
            for row in result['records'][::max(1,len(result['records'])//100)][:100]:
                f=row['frame'];t=poses[f];poly=(tri[row['triangle_id']]-t[:3,3])@t[:3,:3]
                for axis in range(3):
                    for sign in [-1,1]:poly=clip_polygon(poly,axis,sign*h[axis],sign)
                if not len(poly) or np.min(h-abs(poly.mean(axis=0)))<=0:raise ValueError('SAT/clipping witness disagreement')
                witness_checks.append(dict(frame=f,cropped_triangle_id=row['triangle_id'],raw_triangle_id=int(source_ids[row['triangle_id']]),
                    clipping_centroid_inside_margin_raw=float(np.min(h-abs(poly.mean(axis=0))))))
            result.pop('records')
            result.update(independent_clipped_polygon_witnesses=witness_checks,
                prior_clipping_hit_frame_count=prior['foot_box_surface_strict_interior_hit_frames'][side],
                equal_prior_hit_frame_count=result['hit_frame_count']==prior['foot_box_surface_strict_interior_hit_frames'][side])
            sides[side]=result
        reviews.append(dict(trial=trial,artifact_sha256=digest,sides=sides,accepted_reference=False))
        print(json.dumps(dict(trial=trial,hits={s:r['hit_frame_count'] for s,r in sides.items()},elapsed_s=time.monotonic()-start)),flush=True)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Input mutated during audit')
    report=dict(stage='fresh_finite_triangle_OBB_guard_on_existing_rejected_full997_candidates',
        base_commit='e50115ec3fdabc3583224ea060e3054d61631939',inputs_sha256=inputs,
        code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        crop=dict(method='independent_numpy_halfspace_crop_no_caps_no_trimesh_vertex_order_claim',size=[9,12],
            input_triangles=len(raw),output_triangles=len(tri),area_raw_squared=float(area),historical_area_abs_error=area_error,
            historical_bounds_max_error=bounds_error,source_triangle_ids_retained=True,raw_coordinates_no_world_alignment_inferred=True),
        trials=reviews,whole_body_guard_implemented=False,self_collision_guard_implemented=False,
        production_solver_wired=False,source_target_material_calibration_accepted=False,
        optimizer_imports_setup_solve=0,new_motion_files=0,gpu_submissions=0,ppo_updates=0,
        accepted_reference=False,training_ready=False,elapsed_seconds=time.monotonic()-start)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ['elapsed_seconds','optimizer_imports_setup_solve','accepted_reference']},indent=2),flush=True)

if __name__=='__main__':main()