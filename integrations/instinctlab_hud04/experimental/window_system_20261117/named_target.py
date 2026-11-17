"""Portable named 29-DoF target point kinematics, no source-frame assumptions."""
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

MODEL_SHA256='865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7'

def rigid(t):
    t=np.asarray(t,float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1],atol=1e-12,rtol=0) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-10,rtol=0) or not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-10,rtol=0):raise ValueError('SE3 required')
    return t

def perturb(root,d):
    root=rigid(root);d=np.asarray(d,float)
    if d.shape!=(6,) or not np.isfinite(d).all():raise ValueError('Six finite tangent entries')
    out=root.copy();out[:3,3]+=d[:3];out[:3,:3]=Rotation.from_rotvec(d[3:]).as_matrix()@root[:3,:3]
    return out

def origin(node):
    out=np.eye(4)
    if node is not None:
        out[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        out[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return rigid(out)

class NamedTarget:
    def __init__(self,path):
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=MODEL_SHA256:raise ValueError('Pinned target model checksum')
        xml=ET.fromstring(raw);links={l.get('name'):l for l in xml.findall('link')}
        joints=xml.findall('joint');roots=set(links)-{j.find('child').get('link') for j in joints}
        if roots!={'base_link'}:raise ValueError('Expected base_link root')
        movable=[j for j in joints if j.get('type')!='fixed']
        self.names=[j.get('name') for j in movable]
        if len(self.names)!=29 or len(set(self.names))!=29 or any('head' in n for n in self.names):raise ValueError('29 named body actions only')
        for n in ('head_yaw_joint','head_pitch_joint'):
            if not any(j.get('name')==n and j.get('type')=='fixed' for j in joints):raise ValueError('Head must remain fixed')
        self.lower=np.array([float(j.find('limit').get('lower')) for j in movable])
        self.upper=np.array([float(j.find('limit').get('upper')) for j in movable])
        self.vmax=np.array([float(j.find('limit').get('velocity')) for j in movable])
        self.joints=[];ready={'base_link'};self.ancestors={'base_link':set()}
        pending=list(joints)
        while pending:
            batch=[j for j in pending if j.find('parent').get('link') in ready]
            if not batch:raise ValueError('Invalid tree')
            for j in batch:
                name=j.get('name');kind=j.get('type');parent=j.find('parent').get('link');child=j.find('child').get('link')
                if kind not in ('fixed','revolute'):raise ValueError('Unsupported joint type')
                axis=np.fromstring(j.find('axis').get('xyz','1 0 0'),sep=' ') if kind!='fixed' else np.array([1.,0,0])
                axis/=np.linalg.norm(axis)
                self.joints.append((name,kind,parent,child,origin(j.find('origin')),axis))
                self.ancestors[child]=self.ancestors[parent]|({name} if kind!='fixed' else set())
                ready.add(child);pending.remove(j)
        self.features=[('chest_link_origin','waist_pitch_link',np.zeros(3))]
        for side in ('left','right'):
            self.features.extend([(side+'_knee_link_origin',side+'_knee_link',np.zeros(3)),
                                  (side+'_distal_wrist_link_origin',side+'_wrist_roll_link',np.zeros(3))])
            foot=side+'_ankle_roll_link';boxes=[c for c in links[foot].findall('collision') if c.find('geometry/box') is not None]
            if len(boxes)!=1:raise ValueError('One nominal foot box required')
            box=boxes[0];size=np.fromstring(box.find('geometry/box').get('size'),sep=' ')
            transform=origin(box.find('origin'))
            corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*size/2
            corners=corners@transform[:3,:3].T+transform[:3,3]
            bottom=corners[np.isclose(corners[:,2],corners[:,2].min(),atol=1e-12,rtol=0)]
            if len(bottom)!=4:raise ValueError('Four nominal foot bottom corners')
            self.features.extend((side+'_foot_bottom_corner_'+str(i),foot,p) for i,p in enumerate(bottom))

    def evaluate(self,q,root,jacobian=True):
        q=np.asarray(q,float);root=rigid(root)
        if q.shape!=(29,) or not np.isfinite(q).all():raise ValueError('29 finite named body joints required')
        values=dict(zip(self.names,q));world={'base_link':root};axes={}
        for name,kind,parent,child,t,axis in self.joints:
            start=world[parent]@t;move=np.eye(4)
            if kind!='fixed':
                move[:3,:3]=Rotation.from_rotvec(axis*values[name]).as_matrix();axes[name]=(start[:3,3],start[:3,:3]@axis)
            world[child]=start@move
        points=np.array([world[l][:3,:3]@p+world[l][:3,3] for _,l,p in self.features])
        if not jacobian:return points
        j=np.zeros((len(points),3,35));j[:,:,:3]=np.eye(3)
        for i,(_,link,_) in enumerate(self.features):
            for k in range(3):j[i,:,3+k]=np.cross(np.eye(3)[k],points[i]-root[:3,3])
            for k,name in enumerate(self.names):
                if name in self.ancestors[link]:
                    pivot,axis=axes[name];j[i,:,6+k]=np.cross(axis,points[i]-pivot)
        return points,j