"""Independent convex triangle/segment checks plus archive mapping audit."""
from pathlib import Path,PurePosixPath
import hashlib,json,sys,zipfile
import numpy as np
import yaml
from scipy.optimize import minimize
from all_surface_candidates import BASE
sys.path.insert(0,str(BASE));import kinematic_diagnostics as K,contact_geometry as G
sys.path.insert(0,'/workspace/hud04-support-review-20261117');from audit_material_support import material_velocity,speed_components
ROOT=Path(__file__).parent;OUT=ROOT/'ALL_SURFACE_REVIEW'
def digest(b):return hashlib.sha256(b).hexdigest()
def independent_segment_triangle(a,b,tri):
    # Convex squared-distance QP in three bounded coordinates: segment fraction and two triangle barycentrics.
    A=np.column_stack([b-a,-(tri[1]-tri[0]),-(tri[2]-tri[0])]);c=a-tri[0]
    fun=lambda x:.5*np.sum((A@x+c)**2)
    jac=lambda x:A.T@(A@x+c)
    r=minimize(fun,np.array([.5,.3,.3]),jac=jac,bounds=[(0,1)]*3,constraints=[dict(type='ineq',fun=lambda x:1-x[1]-x[2],jac=lambda x:np.array([0.,-1.,-1.]))],method='SLSQP',options=dict(ftol=1e-14,maxiter=200))
    if not r.success or r.x[1]+r.x[2]>1+1e-10:raise ValueError('Independent distance unresolved')
    return float(np.linalg.norm(A@r.x+c))
def main():
    audit=json.loads((ROOT/'OFFICIAL_RELEASE/CONTENT_AUDIT.json').read_text());archive=ROOT/'OFFICIAL_RELEASE/20251116_50cm_kneeClimbStep1.zip'
    assert digest(archive.read_bytes())==audit['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        entries=set(z.namelist());mapping=[]
        for path,text in audit['text_metadata'].items():
            if not path.endswith('.yaml'):continue
            meta=yaml.safe_load(text);parent=PurePosixPath(path).parent
            refs=[str(parent/r['motion_file']) for r in meta.get('motion_files',[])]+[str(parent/r['terrain_file']) for r in meta.get('terrains',[])]
            mapping.append(dict(metadata=path,sha256=digest(text.encode()),missing=[p for p in refs if p not in entries],reference_count=len(refs)))
        originals=json.loads(Path('/workspace/project-instinct-data-search-20261117/motion_source_comparison.json').read_text())['originals']
        exact={name:digest(z.read('20251116_50cm_kneeClimbStep1/50cm_kneeClimbStep/'+name))==v['sha256'] for name,v in originals.items()};assert len(exact)==7 and all(exact.values())
    # Use raw original STL triangle identities, not independently retriangulated IDs.
    data=Path('/workspace/hud04-original-drive-20261117');b=(data/'50cmboxes_19_36_49.stl').read_bytes();dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);tri=np.frombuffer(b,offset=84,dtype=dtype)['v'].astype(float)
    source=Path('/workspace/hud04-recovery-fd12fe1/upstream/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf')
    with np.load(data/'kneelClimbStep1-x-0.1-ziwen-retargeted.npz',allow_pickle=False) as z:
        tree=K.RobotTree.read(source);world=K.at_world_anchor(tree.fk(z['joint_pos'],z['joint_names'].tolist()),K.world_pose(z['base_pos_w'],z['base_quat_w'],'wxyz'),'torso_link')
    witnesses=[];speed_error=0.;point_error=0.;allrows=0
    for link in ['left_ankle_roll_link','left_wrist_yaw_link','left_knee_link','right_ankle_roll_link','right_wrist_yaw_link','right_knee_link']:
        p=OUT/(link+'.npz');r=json.loads((OUT/(link+'.summary.json')).read_text());assert digest(p.read_bytes())==r['npz_sha256']
        with np.load(p,allow_pickle=False) as z:rows=z['rows']
        allrows+=len(rows)
        shapes=G.collisions(source,link,replace_cylinders=True)
        for row in rows:
            if not row[14]:continue
            f=int(row[0]);ps,pt=row[5:8],row[8:11];radius=shapes[int(row[1])]['radius'];point=ps+radius*(pt-ps)/np.linalg.norm(pt-ps)
            local,v=material_velocity(world[link],f,point,90);sp,vt,vn=speed_components(v,row[11:14])
            speed_error=max(speed_error,abs(vt-row[18]),abs(vn-row[19]));point_error=max(point_error,float(abs(local-row[15:18]).max()))
        chosen=[]
        if link=='right_knee_link':
            for f in [300,370]:
                subset=rows[rows[:,0]==f];chosen.extend([subset[np.argmin(subset[:,4])],subset[np.argmin(abs(subset[:,4]))]])
        if link=='right_wrist_yaw_link':
            subset=rows[rows[:,0]==308];good=(abs(subset[:,4])<=.020)&(abs(subset[:,13])>=.7)&(subset[:,14]==1)&(subset[:,18]<=.1)&(subset[:,19]<=.05);chosen.extend(subset[good])
        for row in chosen:
            frame,prim,rawid=int(row[0]),int(row[1]),int(row[3]);t=tri[rawid]
            # These witness triangles are wholly inside the crop, avoiding polygon piece dependence.
            assert np.all(abs(t[:,:2])<=np.array([4.5,6.])+1e-12)
            shape=shapes[prim];axis=np.array([[0,0,-shape['length']/2],[0,0,shape['length']/2]])@shape['transform'][:3,:3].T+shape['transform'][:3,3];tf=world[link][frame];a,b=axis@tf[:3,:3].T+tf[:3,3]
            margin=independent_segment_triangle(a,b,t)-shape['radius'];err=abs(margin-row[4]);assert err<1e-8
            witnesses.append(dict(link=link,frame=frame,primitive=prim,raw_stl_face=rawid,margin_raw=float(row[4]),independent_margin_raw=margin,absolute_error=err,material_defined=bool(row[14]),tangent_speed_raw_per_s=float(row[18]) if row[14] else None,normal_speed_raw_per_s=float(row[19]) if row[14] else None))
    assert speed_error<1e-9 and point_error<1e-10
    result=dict(archive_sha256=audit['archive_sha256'],original_seven_exact_matches=exact,metadata_reference_audit=mapping,all_triangle_rows_reopened=allrows,independent_material_speed_max_error=speed_error,independent_material_point_max_error=point_error,independent_convex_qp_witnesses=witnesses,accepted_support_intervals=None,accepted_reference=False,training_ready=False)
    out=ROOT/'INDEPENDENT_RELEASE_AND_SURFACE_VERIFICATION.json'
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    assert json.loads(out.read_text())==result
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()