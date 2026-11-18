"""Original-URDF batched FK and mass moments; NO contact or physics solver."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def origin(element):
    T=np.eye(4)
    if element is not None:
        T[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return T


class OriginalMassModel:
    def __init__(self,path):
        xml=ET.parse(path).getroot()
        self.links={l.get('name'):l for l in xml.findall('link')}
        self.joints=[];children=set();self.inertials={};self.physical_audit=[]
        for j in xml.findall('joint'):
            kind=j.get('type')
            if kind not in ['fixed','revolute']:raise ValueError('Unsupported joint type')
            parent=j.find('parent').get('link');child=j.find('child').get('link')
            if child in children:raise ValueError('Non-tree model')
            children.add(child)
            axis=np.fromstring(j.find('axis').get('xyz'),sep=' ') if kind=='revolute' else np.zeros(3)
            if kind=='revolute':
                if np.linalg.norm(axis)==0:raise ValueError('Zero joint axis')
                axis=axis/np.linalg.norm(axis) # URDF direction semantics
            self.joints.append((j.get('name'),parent,child,kind,origin(j.find('origin')),axis))
        roots=set(self.links)-children
        if len(roots)!=1:raise ValueError('Exactly one URDF root required')
        self.root=roots.pop()
        pending=self.joints.copy();ordered=[];seen={self.root}
        while pending:
            ready=[j for j in pending if j[1] in seen]
            if not ready:raise ValueError('Disconnected/cyclic graph')
            for j in ready:ordered.append(j);seen.add(j[2]);pending.remove(j)
        self.joints=ordered
        if seen!=set(self.links):raise ValueError('Disconnected links')
        self.movable=[j[0] for j in self.joints if j[3]=='revolute']
        for name,link in self.links.items():
            e=link.find('inertial')
            if e is None:raise ValueError('Missing inertia: '+name)
            m=float(e.find('mass').get('value'));v=e.find('inertia')
            I=np.array([[float(v.get('ixx')),float(v.get('ixy')),float(v.get('ixz'))],
                        [float(v.get('ixy')),float(v.get('iyy')),float(v.get('iyz'))],
                        [float(v.get('ixz')),float(v.get('iyz')),float(v.get('izz'))]])
            T=origin(e.find('origin'))
            if m<=0 or not np.isfinite(I).all():raise ValueError('Invalid inertial values')
            principal=np.linalg.eigvalsh(I)
            second=np.linalg.eigvalsh(.5*np.trace(I)*np.eye(3)-I)
            self.physical_audit.append({'link':name,'mass_kg':m,
                'min_principal_inertia_kg_m2':float(principal.min()),
                'min_second_moment_kg_m2':float(second.min()),
                'positive_inertia_and_triangle_inequality':bool(principal.min()>0 and second.min()>=0)})
            self.inertials[name]=(m,T,I)
        self.total_mass=sum(v[0] for v in self.inertials.values())

    def fk(self,q,names,roots):
        q=np.asarray(q,dtype=float);roots=np.asarray(roots,dtype=float)
        if (q.ndim!=2 or len(names)!=len(set(names)) or set(names)!=set(self.movable)
                or q.shape[1]!=len(names) or roots.shape!=(len(q),4,4)
                or not np.isfinite(q).all() or not np.isfinite(roots).all()):
            raise ValueError('Finite complete named joints and one root transform per frame required')
        if (np.max(np.abs(roots[:,3,:]-np.array([0,0,0,1])))>1e-9
                or np.max(np.abs(roots[:,:3,:3].transpose(0,2,1)@roots[:,:3,:3]-np.eye(3)))>1e-8
                or np.max(np.abs(np.linalg.det(roots[:,:3,:3])-1))>1e-8):
            raise ValueError('Rigid root transforms required')
        lookup={n:i for i,n in enumerate(names)};poses={self.root:roots}
        for name,parent,child,kind,T,axis in self.joints:
            motion=np.tile(np.eye(4),(len(q),1,1))
            if kind=='revolute':motion[:,:3,:3]=Rotation.from_rotvec(q[:,lookup[name],None]*axis).as_matrix()
            poses[child]=poses[parent]@T@motion
        return poses

    def properties(self,q,names,roots,inertia=False):
        poses=self.fk(q,names,roots);first=np.zeros((len(q),3));bodies=[]
        for name,(mass,T,I) in self.inertials.items():
            w=poses[name]@T;p=w[:,:3,3];first+=mass*p
            if inertia:bodies.append((mass,p,w[:,:3,:3]@I@w[:,:3,:3].transpose(0,2,1)))
        com=first/self.total_mass
        if not inertia:return com
        total=np.zeros((len(q),3,3))
        for mass,p,I in bodies:
            d=p-com
            total+=I+mass*((d*d).sum(axis=1)[:,None,None]*np.eye(3)-d[:,:,None]*d[:,None,:])
        return com,total