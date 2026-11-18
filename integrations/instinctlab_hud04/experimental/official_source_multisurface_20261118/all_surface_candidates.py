"""All nearby finite triangle candidates, NOT contact/support assignment."""
from pathlib import Path
import sys
import numpy as np
BASE=Path('/workspace/hud04-recovery-bf4a057/integrations/instinctlab_hud04')
sys.path.insert(0,str(BASE))
from terrain_surface_geometry import segment_triangles

def enumerate_near(a,b,triangles,radius,band):
    a,b,t=np.asarray(a,float),np.asarray(b,float),np.asarray(triangles,float)
    if a.shape!=(3,) or b.shape!=(3,) or t.ndim!=3 or t.shape[1:]!=(3,3) or not np.isfinite(t).all() or not np.isfinite([a,b]).all() or not np.isfinite([radius,band]).all() or radius<=0 or band<0:raise ValueError('Finite geometry and explicit positive radius/nonnegative band')
    lo,hi=np.minimum(a,b),np.maximum(a,b)
    if not len(t):return np.empty(0,int),np.empty(0),np.empty((0,3)),np.empty((0,3))
    gap=np.maximum(0,np.maximum(t.min(axis=1)-hi,lo-t.max(axis=1)))
    ids=np.flatnonzero(np.sum(gap*gap,axis=1)<=(radius+band)**2+1e-12)
    if not len(ids):return ids,np.empty(0),np.empty((0,3)),np.empty((0,3))
    d,ps,pt=segment_triangles(a,b,t[ids]);keep=d<=radius+band+1e-12
    return ids[keep],d[keep]-radius,ps[keep],pt[keep]

def component_speeds(poses,frame,ps,pt,radius,normals,fps):
    delta=pt-ps;length=np.linalg.norm(delta,axis=1);defined=length>1e-12
    # Zero placeholders never mean stationary: every velocity/statistic is masked.
    local=np.zeros_like(ps);tangent=np.zeros(len(ps));normal=np.zeros(len(ps))
    if defined.any():
        p=ps[defined]+radius*delta[defined]/length[defined,None]
        tf=poses[frame];local[defined]=(p-tf[:3,3])@tf[:3,:3]
        velocities=[]
        if frame>0:velocities.append((p-(local[defined]@poses[frame-1,:3,:3].T+poses[frame-1,:3,3]))*fps)
        if frame+1<len(poses):velocities.append(((local[defined]@poses[frame+1,:3,:3].T+poses[frame+1,:3,3])-p)*fps)
        v=np.stack(velocities);ns=normals[defined];vn=np.einsum('kni,ni->kn',v,ns)
        tangent[defined]=np.linalg.norm(v-vn[:,:,None]*ns[None],axis=2).max(axis=0);normal[defined]=abs(vn).max(axis=0)
    return defined,local,tangent,normal