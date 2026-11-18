"""Independent constrained triangle projection and saved evidence consistency."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).parent
sys.path.insert(0,'/workspace/hud04-recovery-bf4a057/integrations/instinctlab_hud04');import kinematic_diagnostics as K
sys.path.insert(0,'/workspace/hud04-wholebody-20261118/REMOTE_ADVANCE_REVIEW_86f05/nonlinear_guard_20261118');from box_triangle_guard import cropped_triangles
def independent(p,tri):
    a,b,c=tri;basis=np.stack([b-a,c-a],axis=1)
    fun=lambda v:np.sum((a+basis@v-p)**2)
    jac=lambda v:2*basis.T@(a+basis@v-p)
    cons={'type':'ineq','fun':lambda v:1-v.sum(),'jac':lambda v:-np.ones(2)}
    r=minimize(fun,[1/3,1/3],jac=jac,method='SLSQP',bounds=[(0,1)]*2,constraints=[cons],options=dict(ftol=1e-15,maxiter=200))
    assert r.success and r.x.sum()<=1+1e-9
    return float(np.sqrt(max(r.fun,0)))
def main():
    with np.load(ROOT/'ENGINEERING_FIXED_MATERIAL_NEIGHBOURS.npz',allow_pickle=False) as z:r=z['rows'];labels=z['materials'].tolist()
    with np.load(ROOT/'PARALLEL_ENGINEERING/RECONSTRUCTION_002/SOURCE_MATERIAL_DIAGNOSTICS.npz',allow_pickle=False) as z:poses={label:z[label+'__source_world_material'] for label in labels};speeds={label:z[label+'__speed_m_s'] for label in labels}
    tri,raw=cropped_triangles(K.read_triangles('/workspace/hud04-original-drive-20261117/50cmboxes_19_36_49.stl'),[9,12]);errs=[];speed_errors=[]
    for mi,label in enumerate(labels):
        p=poses[label][:,:3,3];incident=np.linalg.norm(np.diff(p,axis=0)*90,axis=1);expected=np.r_[incident[0],np.maximum(incident[:-1],incident[1:]),incident[-1]];speed_errors.append(float(abs(expected-speeds[label]).max()))
    assert max(speed_errors)<1e-10
    witness_indices=[int(np.flatnonzero((r[:,0]==5)&(r[:,1]==f)&(r[:,-1]==1))[0]) for f in [271,273,280,288]]
    selected=np.unique(np.r_[np.linspace(0,len(r)-1,24,dtype=int),witness_indices]);witness=[]
    for idx in selected:
        row=r[idx];mi,f,piece,face=map(int,row[:4]);p=poses[labels[mi]][f,:3,3];t=tri[piece];assert raw[piece]==face
        actual=independent(p,t);err=abs(actual-row[4]);assert err<2e-8;errs.append(err)
        if idx in witness_indices:witness.append(dict(material=labels[mi],frame=f,raw_face=face,reported_distance_m=float(row[4]),independent_distance_m=actual,error_m=err))
    for mi,label in enumerate(labels):
        rows=r[r[:,0]==mi];frame=rows[:,1].astype(int);piece=rows[:,2].astype(int);p=poses[label][frame,:3,3];assert np.max(abs(np.linalg.norm(p-rows[:,5:8],axis=1)-rows[:,4]),initial=0)<1e-12
        a,b,c=np.moveaxis(tri[piece],1,0);v0,v1,v2=b-a,c-a,rows[:,5:8]-a;aa=np.einsum('ij,ij->i',v0,v0);ab=np.einsum('ij,ij->i',v0,v1);bb=np.einsum('ij,ij->i',v1,v1);av=np.einsum('ij,ij->i',v0,v2);bv=np.einsum('ij,ij->i',v1,v2);den=aa*bb-ab*ab;u=(bb*av-ab*bv)/den;v=(aa*bv-ab*av)/den
        assert np.all(u>=-1e-8) and np.all(v>=-1e-8) and np.all(u+v<=1+1e-8)
        normals=np.cross(v0,v1);normals/=np.linalg.norm(normals,axis=1)[:,None];assert np.max(abs(np.einsum('ij,ij->i',v2,normals)),initial=0)<1e-10
        assert np.max(abs(normals-rows[:,8:11]),initial=0)<1e-12
        for f in [0,90,174,271,273,280,288,473,996]:
            point=poses[label][f,:3,3]
            # Independent projection of ALL 8068 faces using constrained edge/plane cases.
            a,b,c=np.moveaxis(tri,1,0);v0,v1,vp=b-a,c-a,point-a;aa=np.einsum('ij,ij->i',v0,v0);ab=np.einsum('ij,ij->i',v0,v1);bb=np.einsum('ij,ij->i',v1,v1);av=np.einsum('ij,ij->i',v0,vp);bv=np.einsum('ij,ij->i',v1,vp);den=aa*bb-ab*ab;u=(bb*av-ab*bv)/den;v=(aa*bv-ab*av)/den;projected=a+u[:,None]*v0+v[:,None]*v1;inside=(u>=0)&(v>=0)&(u+v<=1);d=np.where(inside,np.linalg.norm(projected-point,axis=1),np.inf)
            for start,end in [(a,b),(b,c),(c,a)]:
                edge=end-start;lam=np.clip(np.einsum('ij,ij->i',point-start,edge)/np.einsum('ij,ij->i',edge,edge),0,1);d=np.minimum(d,np.linalg.norm(start+lam[:,None]*edge-point,axis=1))
            expected=set(np.flatnonzero(d<=.035+1e-12));got=set(rows[rows[:,1]==f,2].astype(int));assert got==expected
    report=dict(saved_rows_checked=len(r),independent_triangle_minimizations=len(selected),max_independent_distance_error=float(max(errs)),all_shape_frames_material_speed_checks=5982,max_speed_error=max(speed_errors),full_8068_surface_checks=54,witnesses=witness,new_support_plan_generated=False,accepted_reference=False,training_ready=False)
    out=ROOT/'ENGINEERING_FIXED_MATERIAL_INDEPENDENT_VERIFICATION.json'
    with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    assert json.loads(out.read_text())==report;print(json.dumps(report,indent=2))
if __name__=='__main__':main()