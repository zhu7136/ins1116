"""Pelvis-local joint-pivot directions. Never uses NPZ root/quaternions.

The explicit correspondence below is a diagnostic anatomical proposal, not
reviewed palm/sole/knee material contacts or source exporter provenance.
"""
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

SOURCE_SHA='5e99d930af64f3f0bfd34235239e288cc380ba62c8b01fcfcf8cc9f0996e7fa4'
TARGET_SHA='865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7'
MOTION_SHA='56158e8679580910ccff375963d025b138636acfe70f5b1f88381837209b2056'
SEED_SHA='48f8d43e058420e5ff89861d2336ead3317dc18223c36a7305503f3df54a66eb'

def pairs(kind,wrist_endpoint='distal_chain'):
    if kind not in ('source','target'):raise ValueError('Explicit model role')
    if wrist_endpoint not in ('distal_chain','first_wrist_pivot'):raise ValueError('Explicit wrist endpoint')
    if wrist_endpoint=='distal_chain':wrist='wrist_yaw_link' if kind=='source' else 'wrist_roll_link'
    else:wrist='wrist_roll_link' if kind=='source' else 'wrist_yaw_link'
    return [(s+'_'+label,s+'_'+a,s+'_'+b) for s in ('left','right') for label,a,b in (
        ('thigh','hip_pitch_link','knee_link'),('shin','knee_link','ankle_pitch_link'),
        ('upper_arm','shoulder_pitch_link','elbow_link'),('forearm','elbow_link',wrist))]

class DirectionModel:
    def __init__(self,path,kind,*,wrist_endpoint='distal_chain'):
        if kind not in ('source','target'):raise ValueError('Model role')
        raw=path.read_bytes();digest=SOURCE_SHA if kind=='source' else TARGET_SHA
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Pinned model SHA256')
        self.kind=kind;xml=ET.fromstring(raw);nodes=xml.findall('joint')
        links={x.get('name') for x in xml.findall('link')}
        roots=links-{x.find('child').get('link') for x in nodes}
        root='torso_link' if kind=='source' else 'base_link'
        if roots!={root}:raise ValueError('Root graph')
        self.root=root;self.anchor='pelvis' if kind=='source' else 'base_link'
        movable=[j for j in nodes if j.get('type')!='fixed'];self.names=[j.get('name') for j in movable]
        if len(self.names)!=29 or len(set(self.names))!=29 or any('head' in n for n in self.names):raise ValueError('29 body DOF')
        self.lower,self.upper,self.vmax=[np.array([float(j.find('limit').get(k)) for j in movable]) for k in ('lower','upper','velocity')]
        if kind=='target' and not all(any(j.get('name')==n and j.get('type')=='fixed' for j in nodes) for n in ('head_yaw_joint','head_pitch_joint')):raise ValueError('Fixed head')
        self.joints=[];self.ancestors={root:set()};pending=list(nodes)
        while pending:
            ready=[j for j in pending if j.find('parent').get('link') in self.ancestors]
            if not ready:raise ValueError('Invalid kinematic graph')
            for j in ready:
                name=j.get('name');typ=j.get('type');a=j.find('parent').get('link');b=j.find('child').get('link')
                if typ not in ('fixed','revolute'):raise ValueError('Joint type')
                o=j.find('origin');t=np.eye(4)
                if o is not None:
                    t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
                    t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
                axis=np.fromstring(j.find('axis').get('xyz','1 0 0'),sep=' ') if typ!='fixed' else np.array([1.,0,0])
                if not np.isfinite(t).all() or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-12:raise ValueError('Finite geometry')
                axis=axis/np.linalg.norm(axis)
                self.joints.append((name,typ,a,b,t,axis));self.ancestors[b]=self.ancestors[a]|({name} if typ!='fixed' else set());pending.remove(j)
        self.pairs=pairs(kind,wrist_endpoint)
        self.wrist_endpoint=wrist_endpoint
        if not all(a in links and b in links for _,a,b in self.pairs):raise ValueError('Explicit anatomical links missing')

    def order(self,q,names):
        q=np.asarray(q,float);names=list(names)
        if len(names)!=29 or len(set(names))!=29 or set(names)!=set(self.names) or q.ndim!=2 or q.shape[1]!=29 or not 1<=len(q)<=997 or not np.isfinite(q).all():raise ValueError('Finite named joint sequence')
        return q[:,[names.index(n) for n in self.names]].copy()

    def fk(self,q):
        q=self.order(q,self.names);n=len(q);pose={self.root:np.tile(np.eye(4),(n,1,1))};axes={};values={name:q[:,i] for i,name in enumerate(self.names)}
        for name,typ,a,b,t,axis in self.joints:
            start=pose[a]@t;move=np.tile(np.eye(4),(n,1,1))
            if typ!='fixed':
                move[:,:3,:3]=Rotation.from_rotvec(values[name][:,None]*axis).as_matrix()
                axes[name]=(start[:,:3,3],np.einsum('nij,j->ni',start[:,:3,:3],axis))
            pose[b]=start@move
        return pose,axes

    def evaluate(self,q,jacobian=False):
        if jacobian and self.kind!='target':raise ValueError('Source Jacobian intentionally unsupported')
        pose,axes=self.fk(q);vectors=np.stack([pose[b][:,:3,3]-pose[a][:,:3,3] for _,a,b in self.pairs],axis=1)
        basis=pose[self.anchor][:,:3,:3]
        vectors=np.einsum('nji,nkj->nki',basis,vectors)
        lengths=np.linalg.norm(vectors,axis=2)
        if np.any(lengths<1e-8):raise ValueError('Degenerate anatomical direction')
        u=vectors/lengths[:,:,None]
        if not jacobian:return u
        # Target anchor is fixed base_link, so no omitted moving-basis term.
        if self.anchor!=self.root:raise ValueError('Fixed target anchor required')
        j=np.zeros((len(q),len(self.pairs),3,29))
        for k,(_,a,b) in enumerate(self.pairs):
            d=np.zeros((len(q),3,29))
            for c,name in enumerate(self.names):
                pivot,axis=axes[name]
                if name in self.ancestors[b]:d[:,:,c]+=np.cross(axis,pose[b][:,:3,3]-pivot)
                if name in self.ancestors[a]:d[:,:,c]-=np.cross(axis,pose[a][:,:3,3]-pivot)
            projection=(np.eye(3)[None]-u[:,k,:,None]*u[:,k,None,:])/lengths[:,k,None,None]
            j[:,k]=np.einsum('nij,njc->nic',projection,d)
        return u,j

def direction_statistics(actual,desired):
    a=np.asarray(actual,float);b=np.asarray(desired,float)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[1:]!=(8,3) or not all(np.isfinite(x).all() for x in (a,b)):raise ValueError('Direction shape')
    if not all(np.allclose(np.linalg.norm(x,axis=2),1,atol=1e-10,rtol=0) for x in (a,b)):raise ValueError('Unit directions')
    # atan2 avoids cosine clipping and is stable near zero angle.
    degrees=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=2),np.sum(a*b,axis=2)))
    return dict(mean_degrees=float(degrees.mean()),p95_degrees=float(np.percentile(degrees,95)),max_degrees=float(degrees.max()),
                sum_l1_chordal=float(np.abs(a-b).sum()),per_direction_mean_degrees=degrees.mean(axis=0).tolist())