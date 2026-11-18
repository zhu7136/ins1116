"""Explicit local nominal footbox/triangle cuts + SO3-preserving time terms.

NOT a complete collision solver: no body/hands/knees/selfcollision, no closed
terrain containment or runtime-cooked shape evidence. Triangle winding sets
the local escape side; not selected by a root-hypothesis/min-gap contest.
"""
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
sys.path[:0]=['/workspace/hud04-cpu-deps-20261117','/workspace/hud04-terrain-binding-20261117']
from terrain_binding import ActualTerrain

def box_clip(triangle,halfsize):
    poly=np.asarray(triangle,float).copy()
    for axis in range(3):
        for sign in [-1,1]:
            if len(poly)==0:return poly
            signed=sign*poly[:,axis]-halfsize[axis];out=[]
            for k,a in enumerate(poly):
                b=poly[(k+1)%len(poly)];fa,fb=signed[k],signed[(k+1)%len(poly)]
                if fa<=0:out.append(a)
                if (fa<=0)!=(fb<=0):out.append(a+fa/(fa-fb)*(b-a))
            poly=np.array(out,float).reshape(-1,3)
    return poly

class FootTerms:
    def __init__(self):
        import world_features as F
        self.terrain=ActualTerrain();self.tri=self.terrain.triangles
        self.normals=self.terrain.mesh.face_normals.copy();self.lo=self.tri.min(axis=1);self.hi=self.tri.max(axis=1)
        xml=ET.parse(F.TK.TARGET).getroot();boxes=[]
        for side in ['left','right']:
            link=next(l for l in xml.findall('link') if l.get('name')==side+'_ankle_roll_link')
            node=next(c for c in link.findall('collision') if c.find('geometry/box') is not None)
            size=np.fromstring(node.find('geometry/box').get('size'),sep=' ');transform=F.TK.K.origin_transform(node.find('origin'))
            if not np.allclose(transform[:3,:3],np.eye(3),atol=1e-15):raise ValueError('Footbox nonzero local rotation unsupported')
            boxes.append((size,transform[:3,3]))
        if any(not np.array_equal(a,b) for a,b in zip(boxes[0],boxes[1])):raise ValueError('Nonmatching foot boxes unsupported')
        self.halfsize=boxes[0][0]/2;self.center=boxes[0][1]
        self.corners=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*self.halfsize+self.center
        self.padding=.025  # Local cut activation distance in unchanged RAW units, NOT a physical acceptance threshold.

    def cuts(self,model,q,roots):
        n=len(q);local,axes=model.model.fk(q);world={l:roots@t for l,t in local.items()}
        corners=[];jac=[];records=[];gap=[];rows=[];cols=[];data=[]
        for side in ['left','right']:
            link=side+'_ankle_roll_link';pose=world[link]
            p=np.einsum('nij,kj->nki',pose[:,:3,:3],self.corners)+pose[:,None,:3,3]
            jp=np.zeros((n,8,3,35));jp[:,:,:,:3]=np.eye(3)
            for axis in range(3):jp[:,:,:,3+axis]=np.cross(np.eye(3)[axis],p-roots[:,None,:3,3])
            for k,name in enumerate(model.names):
                if name not in model.model.ancestors[link]:continue
                pivot,axis=axes[name];pivot=np.einsum('nij,nj->ni',roots[:,:3,:3],pivot)+roots[:,:3,3]
                axis=np.einsum('nij,nj->ni',roots[:,:3,:3],axis)
                jp[:,:,:,6+k]=np.cross(axis[:,None,:],p-pivot[:,None,:])
            for frame in range(n):
                lower=p[frame].min(axis=0)-self.padding;upper=p[frame].max(axis=0)+self.padding
                candidates=np.flatnonzero(np.all(self.hi>=lower,axis=1)&np.all(self.lo<=upper,axis=1));seen=set()
                transform=pose[frame];rr=transform[:3,:3];origin=transform[:3,3]+rr@self.center
                for tid in candidates:
                    normal=self.normals[tid];offset=float(normal@self.tri[tid,0])
                    if np.linalg.norm(normal)<.99:continue
                    # Exact finite triangle vs padded oriented box, not an
                    # infinite plane/vertical height substituted for distance.
                    triangle=(self.tri[tid]-origin)@rr
                    if not len(box_clip(triangle,self.halfsize+self.padding)):continue
                    key=tuple(np.rint(np.r_[normal,offset]*1e9).astype(np.int64))
                    if key in seen:continue
                    seen.add(key)
                    signed=p[frame]@normal-offset;vertex=int(np.argmin(signed));g=float(signed[vertex])
                    j=normal@jp[frame,vertex];row=len(gap);gap.append(g)
                    nonzero=np.flatnonzero(abs(j)>1e-15);rows.extend([row]*len(nonzero));cols.extend((35*frame+nonzero).tolist());data.extend(j[nonzero].tolist())
                    records.append(dict(frame=frame,side=side,triangle_id=int(tid),normal=normal.tolist(),offset=offset,vertex=vertex))
        matrix=sparse.coo_matrix((data,(rows,cols)),shape=(len(gap),35*n)).tocsr()
        return np.array(gap),matrix,records

    def fixed_cut_gaps(self,model,q,roots,records):
        pose,_=model.model.fk(q);world={l:roots@t for l,t in pose.items()};values=[]
        for r in records:
            t=world[r['side']+'_ankle_roll_link'][r['frame']]
            p=self.corners@t[:3,:3].T+t[:3,3]
            # True min over ALL8 corners, not only a linearization's argmin.
            values.append(float(np.min(p@r['normal']-r['offset'])))
        return np.array(values)

def states(q,roots):return np.c_[roots[:,:3,3],roots[:,:3,:3].reshape(len(q),9),q]

def temporal_values(q,roots,first,last):
    ids=np.arange(max(1,first-1),min(len(q)-2,last+1)+1)
    x=states(q,roots);w=np.r_[np.full(3,.02*8100),np.full(9,.005*8100),np.full(29,.0001*8100)]
    return ((x[ids-1]-2*x[ids]+x[ids+1])*w).reshape(-1)

def temporal_system(q,roots,first,last):
    ids=np.arange(max(1,first-1),min(len(q)-2,last+1)+1);n=last-first+1
    w=np.r_[np.full(3,.02*8100),np.full(9,.005*8100),np.full(29,.0001*8100)]
    state_j=np.zeros((n,41,35));state_j[:,:3,:3]=np.eye(3);state_j[:,12:,6:]=np.eye(29)
    for axis in range(3):
        for col in range(3):
            for i in range(3):state_j[:,3+3*i+col,3+axis]=np.cross(np.eye(3)[axis],roots[first:last+1,:3,col])[:,i]
    blocks=[]
    for t in range(first,last+1):
        jj=state_j[t-first]*w[:,None];jj=sparse.coo_matrix(jj);row=[];column=[];data=[]
        for shift,coef in [(-1,1),(0,-2),(1,1)]:
            positions=np.flatnonzero(ids==t+shift)
            if not len(positions):continue
            row.extend((positions[0]*41+jj.row).tolist());column.extend(jj.col.tolist());data.extend((coef*jj.data).tolist())
        blocks.append(sparse.coo_matrix((data,(row,column)),shape=(len(ids)*41,35)).tocsr())
    return temporal_values(q,roots,first,last),sparse.hstack(blocks,format='csr')