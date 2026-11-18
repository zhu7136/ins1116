"""Audit ALL nearby surfaces of parallel FIXED materials; no support plan edits."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from all_surface_candidates import BASE
sys.path.insert(0,str(BASE));import kinematic_diagnostics as K
from terrain_surface_geometry import point_triangles
sys.path.insert(0,'/workspace/hud04-wholebody-20261118/REMOTE_ADVANCE_REVIEW_86f05/nonlinear_guard_20261118');from box_triangle_guard import cropped_triangles
ROOT=Path(__file__).parent
def main():
    start=time.monotonic();base=ROOT/'PARALLEL_ENGINEERING/RECONSTRUCTION_002';cp=base/'CONTRACT.json';contract=json.loads(cp.read_text());zpath=base/'SOURCE_MATERIAL_DIAGNOSTICS.npz'
    recovery=json.loads((ROOT/'PARALLEL_ENGINEERING/RECOVERY.json').read_text())
    for p in (cp,zpath):
        r=next(r for r in recovery['recovered'] if r['path'].endswith('/RECONSTRUCTION_002/'+p.name));assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    terrain=Path('/workspace/hud04-original-drive-20261117/50cmboxes_19_36_49.stl');assert hashlib.sha256(terrain.read_bytes()).hexdigest()=='3aee33efea6793974d901dd0f7e46859e2d12d2dfd79f38f12c777686566c2de'
    tri,rawids=cropped_triangles(K.read_triangles(terrain),[9,12]);n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.linalg.norm(n,axis=1)[:,None];lo,hi=tri.min(axis=1),tri.max(axis=1)
    rules=contract['support_proposal_rules'];band=rules['max_surface_distance_m'];opposition=np.cos(np.deg2rad(rules['max_normal_error_deg']));unassigned=np.zeros(997,bool)
    for a,b in contract['unassigned_intervals']:unassigned[a:b+1]=True
    assert unassigned.sum()==346
    summaries=[];global_eligible=np.zeros(997,bool);allrows=[];labels=[]
    with np.load(zpath,allow_pickle=False) as z:
        labels=[k.split('__')[0] for k in z.files if k.endswith('__source_world_material')]
        assert len(labels)==6
        for mi,label in enumerate(labels):
            poses=z[label+'__source_world_material'];speed=z[label+'__speed_m_s'];old_distance=z[label+'__distance_m'];old_op=z[label+'__normal_opposition'];old_amb=z[label+'__ambiguous_nearest_normal']
            old=(old_distance<=band)&(speed<=rules['max_material_speed_m_s'])&(old_op>=opposition)&~old_amb
            any_near=np.zeros(997,bool);any_lateral=np.zeros(997,bool);eligible=np.zeros(997,bool);nearest_checks=0
            for f,p in enumerate(poses[:,:3,3]):
                if time.monotonic()-start>80:raise TimeoutError('Bounded read-only diagnostic')
                gap=np.maximum(0,np.maximum(lo-p,p-hi));ids=np.flatnonzero(np.sum(gap*gap,axis=1)<=band*band+1e-12)
                if not len(ids):continue
                closest=point_triangles(p,tri[ids]);dist=np.linalg.norm(closest-p,axis=1);keep=dist<=band+1e-12;ids,closest,dist=ids[keep],closest[keep],dist[keep]
                if not len(ids):continue
                op=-(n[ids]@poses[f,:3,2]);up=n[ids,2]>=rules['min_upward_surface_normal_z'];lateral=abs(n[ids,2])<=.1
                ok=up&(op>=opposition)&(speed[f]<=rules['max_material_speed_m_s']);eligible[f]=ok.any();any_near[f]=True;any_lateral[f]=lateral.any()
                # No nearest-tie rejection or assignment is performed: keep all possibilities, do not select a normal/anchor.
                if up.any():
                    assert abs(float(dist[up].min())-old_distance[f])<1e-8;nearest_checks+=1
                allrows.append(np.column_stack([np.full(len(ids),mi),np.full(len(ids),f),ids,rawids[ids],dist,closest,n[ids],op,np.full(len(ids),speed[f]),ok]))
            global_eligible|=eligible
            summaries.append(dict(material=label,all_surface_near_frames=int(any_near.sum()),all_lateral_near_frames=int(any_lateral.sum()),old_nearest_pre_run_geometric_candidates=int(old.sum()),all_surface_pre_run_geometric_candidates=int(eligible.sum()),new_frame_candidates=np.flatnonzero(eligible&~old).tolist(),candidate_frames_within_original_346_unassigned=np.flatnonzero(eligible&unassigned).tolist(),historical_nearest_upward_distance_checks=nearest_checks))
    rows=np.concatenate(allrows) if allrows else np.empty((0,14));assert rows.shape[1]==14
    outdata=ROOT/'ENGINEERING_FIXED_MATERIAL_NEIGHBOURS.npz';columns=['material_index','frame','cropped_piece','raw_face','distance_engineering_m','point_x','point_y','point_z','normal_x','normal_y','normal_z','normal_opposition','speed_engineering_m_s','eligible_before_interval_anchor_and_ambiguity_review']
    with outdata.open('xb') as f:np.savez_compressed(f,rows=rows,columns=np.array(columns),materials=np.array(labels),accepted_reference=False,accepted_support_intervals=False)
    with np.load(outdata,allow_pickle=False) as z:assert np.array_equal(z['rows'],rows)
    report=dict(stage='parallel_engineering_FIXED_material_ALL_neighbour_geometry_NOT_revised_support_plan',parent_engineering_commit='2c7f6a3d3f31a24b993bd7d1a47078c01c9a4cb7',contract_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),rows=len(rows),queries=5982,frames=997,material_count=6,thresholds=rules,summaries=summaries,original_unassigned_frames=346,original_unassigned_with_at_least_one_pre_run_candidate=int((global_eligible&unassigned).sum()),original_unassigned_without_any_pre_run_candidate=int((~global_eligible&unassigned).sum()),cpu_seconds=time.monotonic()-start,npz_sha256=hashlib.sha256(outdata.read_bytes()).hexdigest(),new_support_plan_generated=False,tie_normal_or_anchor_selected=False,minimum_run_and_anchor_spread_applied=False,observed_source_units_recovered=False,units='new engineering convention only, as versioned parallel contract',solver_calls=0,gpu_submissions=0,accepted_support_intervals=None,accepted_reference=False,training_ready=False)
    out=ROOT/'ENGINEERING_FIXED_MATERIAL_REVIEW.json'
    with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    assert json.loads(out.read_text())==report;print(json.dumps(report,indent=2))
if __name__=='__main__':main()