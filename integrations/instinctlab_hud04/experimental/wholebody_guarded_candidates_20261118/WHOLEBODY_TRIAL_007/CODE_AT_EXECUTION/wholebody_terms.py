"""All43 nominal shapes/all903 pairs: conservative separating-plane terms.

Positive projection gaps certify separation of ORIGINAL support shapes.
Negative values are sampled-axis overlap, NOT signed penetration depth.
No graph-based exclusions; same-link constraints are recorded constants.
Finite triangles, including walls, are retained; no vertical-height shortcut.
"""
import sys
from itertools import combinations
import numpy as np
from scipy import sparse
sys.path[:0]=['/workspace/hud04-cpu-deps-20261117','/workspace/hud04-solid-geometry-20261117',
              '/workspace/hud04-world-candidate-20261118','/workspace/hud04-terrain-binding-20261117']
from robot_shape_inventory import load_shapes
from terrain_binding import ActualTerrain

def supports(shape,pose,d):
    d=np.asarray(d);local=d@pose[:3,:3];p=np.zeros_like(local)
    if shape.kind=='sphere':p=local/np.maximum(np.linalg.norm(local,axis=1)[:,None],1e-300)*shape.parameters['radius']
    elif shape.kind=='box':p=np.sign(local)*np.array(shape.parameters['size'])/2
    elif shape.kind=='cylinder':
        p[:,:2]=local[:,:2]/np.maximum(np.linalg.norm(local[:,:2],axis=1)[:,None],1e-300)*shape.parameters['radius']
        p[:,2]=np.sign(local[:,2])*shape.parameters['length']/2
    elif shape.kind=='mesh':p=shape.vertices[np.argmax(local@shape.vertices.T,axis=1)]
    else:raise ValueError('Unknown original shape')
    return p@pose[:3,:3].T+pose[:3,3]

def normalized(d):
    norm=np.linalg.norm(d,axis=-1);return np.divide(d,norm[...,None],out=np.zeros_like(d),where=norm[...,None]>1e-12),norm>1e-12

def candidate_directions(a,b,center):
    # No accepted geometry/units/normal correspondence inferred from these axes.
    axes=np.r_[a.T,b.T,np.cross(a.T[:,None],b.T[None]).reshape(-1,3),np.asarray(center).reshape(1,3)]
    n,valid=normalized(axes);return n[valid]

def pair_projection(a,pa,b,pb):
    directions=candidate_directions(pa[:3,:3],pb[:3,:3],pa[:3,3]-pb[:3,3]);d=np.r_[directions,-directions]
    aa=supports(a,pa,-d);bb=supports(b,pb,d)
    values=np.einsum('ij,ij->i',aa-bb,d);i=int(np.argmax(values))
    return float(values[i]),d[i],aa[i],bb[i]

def triangle_projection(shape,pose,triangles):
    count=len(triangles)
    if not count:return np.empty(0),np.empty((0,3)),np.empty((0,3)),np.empty((0,3))
    edges=np.roll(triangles,-1,axis=1)-triangles
    normal=np.cross(edges[:,0],edges[:,1]);cross=np.cross(pose[:3,:3].T[None,:,None,:],edges[:,None,:,:]).reshape(count,9,3)
    raw=np.concatenate([np.tile(pose[:3,:3].T,(count,1,1)),normal[:,None,:],cross,
                        (pose[:3,3]-triangles.mean(axis=1))[:,None,:]],axis=1)
    directions,valid=normalized(raw);d=np.concatenate([directions,-directions],axis=1);valid=np.c_[valid,valid]
    first=supports(shape,pose,-d.reshape(-1,3)).reshape(d.shape)
    dots=np.einsum('nkj,nvj->nkv',d,triangles);ids=dots.argmax(axis=2)
    second=triangles[np.arange(count)[:,None],ids]
    values=np.einsum('nkj,nkj->nk',first-second,d);values[~valid]=-np.inf
    best=values.argmax(axis=1);r=np.arange(count)
    return values[r,best],d[r,best],first[r,best],second[r,best]

class WholeBodyTerms:
    def __init__(self,margin=.001,padding=.015):
        if margin!=.001 or padding!=.015:raise ValueError('Trial scope fixed; raw-unit margins NOT acceptance thresholds')
        self.tree,self.shapes,self.policy=load_shapes();self.terrain=ActualTerrain()
        self.tri=self.terrain.triangles;self.lo=self.tri.min(axis=1);self.hi=self.tri.max(axis=1)
        self.pairs=list(combinations(range(43),2));self.margin=margin;self.padding=padding

    def evaluate(self,model,q,roots,jacobian=False):
        q=np.asarray(q);n=len(q);local,axes=model.model.fk(q)
        world={link:roots@p for link,p in local.items()};records=[];gaps=[]
        count=dict(frames=n,shapes=43,shape_pairs=903,terrain_pairs_broadphase=0,self_pairs_broadphase=0,
                   negative_terrain_projection_pairs=0,negative_self_projection_pairs=0,
                   negative_self_same_link_pairs=0,all_adjacency_pairs_retained=True)
        for frame in range(n):
            poses=[world[s.link][frame]@s.local for s in self.shapes];lo=[];hi=[]
            for shape,pose in zip(self.shapes,poses):
                p=supports(shape,pose,np.r_[np.eye(3),-np.eye(3)]);hi.append(np.diag(p[:3]));lo.append(np.diag(p[3:]))
            lo=np.array(lo);hi=np.array(hi)
            def append(kind,a,b,g,d,point_a,point_b):
                if g>=self.margin:return
                gaps.append(float(g));records.append(dict(frame=frame,kind=kind,first=a,second=b,normal=d.tolist(),
                    first_local=((point_a-world[self.shapes[a].link][frame,:3,3])@world[self.shapes[a].link][frame,:3,:3]).tolist(),
                    second_local=(point_b if kind=='terrain' else (point_b-world[self.shapes[b].link][frame,:3,3])@world[self.shapes[b].link][frame,:3,:3]).tolist()))
            for a,shape in enumerate(self.shapes):
                ids=np.flatnonzero(np.all(self.hi>=lo[a]-self.padding,axis=1)&np.all(self.lo<=hi[a]+self.padding,axis=1))
                count['terrain_pairs_broadphase']+=len(ids)
                val,d,pa,pb=triangle_projection(shape,poses[a],self.tri[ids])
                count['negative_terrain_projection_pairs']+=int(np.sum(val<0))
                for k in np.flatnonzero(val<self.margin):append('terrain',a,int(ids[k]),val[k],d[k],pa[k],pb[k])
            for a,b in self.pairs:
                if np.any(hi[a]+self.padding<lo[b]) or np.any(hi[b]+self.padding<lo[a]):continue
                count['self_pairs_broadphase']+=1
                g,d,pa,pb=pair_projection(self.shapes[a],poses[a],self.shapes[b],poses[b])
                if g<0:
                    count['negative_self_projection_pairs']+=1
                    count['negative_self_same_link_pairs']+=int(self.shapes[a].link==self.shapes[b].link)
                append('self',a,b,g,d,pa,pb)
        if not jacobian:return np.array(gaps),records,count
        # Frozen separating planes/material witnesses have EXACT point derivatives.
        rows=[];cols=[];data=[]
        for row,r in enumerate(records):
            frame=r['frame'];d=np.array(r['normal']);j=np.zeros(35)
            for which,sign in [('first',1),('second',-1)]:
                if which=='second' and r['kind']=='terrain':continue
                link=self.shapes[r[which]].link;t=world[link][frame]
                point=t[:3,:3]@np.array(r[which+'_local'])+t[:3,3]
                j[:3]+=sign*d
                for axis in range(3):j[3+axis]+=sign*d@np.cross(np.eye(3)[axis],point-roots[frame,:3,3])
                for k,name in enumerate(model.names):
                    if name not in model.model.ancestors[link]:continue
                    pivot,axis=axes[name];wp=roots[frame,:3,:3]@pivot[frame]+roots[frame,:3,3];wa=roots[frame,:3,:3]@axis[frame]
                    j[6+k]+=sign*d@np.cross(wa,point-wp)
            nonzero=np.flatnonzero(abs(j)>1e-14);rows.extend([row]*len(nonzero));cols.extend((35*frame+nonzero).tolist());data.extend(j[nonzero].tolist())
        matrix=sparse.coo_matrix((data,(rows,cols)),shape=(len(gaps),n*35)).tocsr()
        return np.array(gaps),matrix,records,count

    def frozen_values(self,model,q,roots,records):
        local,_=model.model.fk(q);world={l:roots@p for l,p in local.items()};values=[]
        for r in records:
            f=r['frame'];t=world[self.shapes[r['first']].link][f];a=t[:3,:3]@np.array(r['first_local'])+t[:3,3]
            if r['kind']=='terrain':b=np.array(r['second_local'])
            else:
                t=world[self.shapes[r['second']].link][f];b=t[:3,:3]@np.array(r['second_local'])+t[:3,3]
            values.append(float(np.array(r['normal'])@(a-b)))
        return np.array(values)