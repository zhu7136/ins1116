"""Bounded all-frame/all-near-triangle review; no solver, GPU or labels."""
from pathlib import Path
import hashlib,json,struct,sys,time
import numpy as np
from all_surface_candidates import enumerate_near,component_speeds,BASE,segment_triangles
sys.path.insert(0,str(BASE))
import kinematic_diagnostics as K,contact_geometry as G
GUARD=Path('/workspace/hud04-wholebody-20261118/REMOTE_ADVANCE_REVIEW_86f05/nonlinear_guard_20261118')
sys.path.insert(0,str(GUARD));from box_triangle_guard import cropped_triangles
ROOT=Path(__file__).parent/'ALL_SURFACE_REVIEW'
DATA=Path('/workspace/hud04-original-drive-20261117')
SOURCE=Path('/workspace/hud04-recovery-fd12fe1/upstream/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf')
LINKS=[s+'_'+p for s in ('left','right') for p in ('ankle_roll_link','wrist_yaw_link','knee_link')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    start=time.monotonic();ROOT.mkdir(exist_ok=False)
    paths=[SOURCE,DATA/'kneelClimbStep1-x-0.1-ziwen-retargeted.npz',DATA/'50cmboxes_19_36_49.stl',GUARD/'box_triangle_guard.py']
    expected=['5e99d930af64f3f0bfd34235239e288cc380ba62c8b01fcfcf8cc9f0996e7fa4','56158e8679580910ccff375963d025b138636acfe70f5b1f88381837209b2056','3aee33efea6793974d901dd0f7e46859e2d12d2dfd79f38f12c777686566c2de','f69a24efb4ccf34e82c53d9c45fb71b1dca6555aba97663bca920d38b6d0a797']
    assert [sha(p) for p in paths]==expected
    with np.load(paths[1],allow_pickle=False) as z:
        assert z['joint_pos'].shape==(997,29) and z['framerate'].item()==90
        tree=K.RobotTree.read(SOURCE);world=K.at_world_anchor(tree.fk(z['joint_pos'],z['joint_names'].tolist()),K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz'),'torso_link')
    raw=paths[2].read_bytes();count=struct.unpack('<I',raw[80:84])[0]
    assert len(raw)==84+count*50 and count==8087
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')]);original=np.frombuffer(raw,offset=84,dtype=dtype)['vertices'].astype(float)
    tri,original_ids=cropped_triangles(original,(9,12));assert len(tri)==8068
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    summary=[];qcount=0;all_min_errors=[];bruteforce=[]
    for link in LINKS:
        chunks=[];all_near=np.zeros(997,bool);all_lateral=np.zeros(997,bool);slow=np.zeros(997,bool);old_near=np.zeros(997,bool);old_lateral=np.zeros(997,bool);old_slow=np.zeros(997,bool);undefined=0;near_count=0;deep_count=0
        shapes=G.collisions(SOURCE,link,replace_cylinders=True)
        for prim,p in enumerate(shapes):
            assert p['kind']=='capsule'
            prior=json.loads((Path('/workspace/hud04-support-review-20261117/results')/(link+'.primitive'+str(prim)+'.json')).read_text())
            oldm=np.asarray(prior['values']['surface_margin_raw']);old_near|=abs(oldm)<=.020;old_lateral|=abs(np.asarray(prior['values']['lateral_margin_raw']))<=.020
            tangent=np.array([np.inf if v is None else v for v in prior['values']['material_tangential_speed_raw_per_s']]);normal=np.array([np.inf if v is None else v for v in prior['values']['material_normal_speed_abs_raw_per_s']]);old_slow|=(abs(oldm)<=.020)&(tangent<=.1)&(normal<=.05)&(np.asarray(prior['values']['normal_z_abs'])>=.7)
            local=np.array([[0,0,-p['length']/2],[0,0,p['length']/2]])@p['transform'][:3,:3].T+p['transform'][:3,3]
            axes=K.transform_points(world[link],local)
            for frame,(a,b) in enumerate(axes):
                if time.monotonic()-start>95:raise TimeoutError('Explicit CPU review budget, no retry')
                ids,m,ps,pt=enumerate_near(a,b,tri,p['radius'],.020);qcount+=1
                if len(ids):
                    err=abs(float(m.min())-oldm[frame]);all_min_errors.append(err)
                    if err>1e-8:raise ValueError('Historical nearest mismatch')
                elif oldm[frame]<=.020-1e-10:raise ValueError('Omitted historical close surface')
                if prim==0 and frame in (197,257,405):
                    d,*_=segment_triangles(a,b,tri);expected_ids=np.flatnonzero(d<=p['radius']+.020+1e-12)
                    assert np.array_equal(ids,expected_ids)
                    bruteforce.append(dict(link=link,primitive=prim,frame=frame,all_triangles=len(tri),selected=len(ids)))
                if not len(ids):continue
                defined,localmat,vt,vn=component_speeds(world[link],frame,ps,pt,p['radius'],normals[ids],90)
                near=abs(m)<=.020;lat=near&(abs(normals[ids,2])<=.1);sl=near&defined&(abs(normals[ids,2])>=.7)&(vt<=.1)&(vn<=.05)
                all_near[frame]|=near.any();all_lateral[frame]|=lat.any();slow[frame]|=sl.any();undefined+=int((~defined).sum());near_count+=int(near.sum());deep_count+=int((m<-.020).sum())
                # Every row preserves triangle and same-frame material identity. Placeholder material values are INVALID unless defined==1.
                chunks.append(np.column_stack([np.full(len(ids),frame),np.full(len(ids),prim),ids,original_ids[ids],m,ps,pt,normals[ids],defined,localmat,vt,vn]))
            print('PRIMITIVE_DONE',link,prim,'seconds',round(time.monotonic()-start,2),flush=True)
        rows=np.concatenate(chunks) if chunks else np.empty((0,20))
        columns=['frame','primitive','cropped_piece_id','raw_stl_face_id','axis_surface_distance_minus_radius_raw','axis_x','axis_y','axis_z','triangle_x','triangle_y','triangle_z','normal_x','normal_y','normal_z','material_defined','material_local_x','material_local_y','material_local_z','tangent_speed_raw_per_s','normal_speed_abs_raw_per_s']
        assert rows.shape[1]==len(columns)
        datafile=ROOT/(link+'.npz');np.savez_compressed(datafile,rows=rows,columns=np.array(columns),accepted_reference=False,accepted_support_labels=False)
        with np.load(datafile,allow_pickle=False) as z:assert np.array_equal(z['rows'],rows)
        r=dict(link=link,primitives=len(shapes),near_triangle_rows=near_count,deeper_triangle_rows=deep_count,all_candidate_rows=len(rows),undefined_material_rows=undefined,old_nearest_only_near_frames=int(old_near.sum()),all_triangle_near_frames=int(all_near.sum()),new_near_frames=np.flatnonzero(all_near&~old_near).tolist(),old_lateral_nearest_near_frames=int(old_lateral.sum()),all_triangle_lateral_near_frames=int(all_lateral.sum()),new_lateral_frames=np.flatnonzero(all_lateral&~old_lateral).tolist(),old_nearest_slow_horizontal_frames=int(old_slow.sum()),all_triangle_slow_horizontal_frames=int(slow.sum()),new_slow_horizontal_frames=np.flatnonzero(slow&~old_slow).tolist(),npz_sha256=sha(datafile))
        (ROOT/(link+'.summary.json')).write_text(json.dumps(r,indent=2,allow_nan=False));summary.append(r)
        print('LINK_DONE',json.dumps(r),flush=True)
    assert qcount==17946 and [sha(p) for p in paths]==expected
    report=dict(stage='official_torso_wxyz_conditioned_all_near_finite_triangles_NOT_support_labels',frames=997,fps=90,source_colliders=18,segment_queries=qcount,terrain_triangles=len(tri),thresholds_raw=dict(candidate_band=.020,diagnostic_tangent=.1,diagnostic_normal=.05),cpu_wall_seconds=time.monotonic()-start,links=summary,maximum_historical_minimum_error=max(all_min_errors),exhaustive_broadphase_checks=bruteforce,inputs_sha256=dict(zip(map(str,paths),expected)),numpy_clip_triangle_ids_are_not_runtime_ids=True,common_motion_terrain_translation_cancels=True,raw_physical_units_calibrated=False,all_material_undefined_rows_have_invalid_zero_placeholders=True,contact_manifold_or_connected_patch_certified=False,accepted_support_intervals=None,target_material_correspondence_calibrated=False,solver_calls=0,gpu_submissions=0,accepted_reference=False,training_ready=False)
    out=ROOT/'REPORT.json';out.write_text(json.dumps(report,indent=2,allow_nan=False));assert json.loads(out.read_text())==report
    print('COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()