"""Hash-bound 43-shape inventory with exact primitive support functions.

ALL unordered shape pairs retained. Graph adjacency is descriptive, never an
automatic self-collision exclusion. Broadphase overlap means UNRESOLVED.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import ast
import hashlib
import sys
import numpy as np

sys.path.insert(0,'/workspace/hud04-runtime-preflight-20261117')
from task_kinematics import TARGET,K,S,R,check_rigid,TargetTasks
from convex_solid import Hull,points

CONFIG=Path('/workspace/hud04-recovery-bf4a057/flux/hud04_asset_camera_probe/asset_config.py')
CONFIG_SHA='4fdb08b4aa352893a6feef5afb6deb8ec273e1fbdcfc2d67ac0e691729321021'


@dataclass
class Shape:
    identity: str
    link: str
    kind: str
    local: np.ndarray
    parameters: dict
    vertices: object=None

    def support(self,directions,link_world):
        directions=points(directions,1,256);world=check_rigid(link_world)@check_rigid(self.local)
        d=directions@world[:3,:3];p=np.zeros_like(d)
        if self.kind=='box':p=np.sign(d)*np.asarray(self.parameters['size'])/2
        elif self.kind=='sphere':
            n=np.linalg.norm(d,axis=1);p=np.divide(d,n[:,None],out=p,where=n[:,None]>0)*self.parameters['radius']
        elif self.kind=='cylinder':
            n=np.linalg.norm(d[:,:2],axis=1)
            p[:,:2]=np.divide(d[:,:2],n[:,None],out=p[:,:2],where=n[:,None]>0)*self.parameters['radius']
            p[:,2]=np.sign(d[:,2])*self.parameters['length']/2
        elif self.kind=='mesh':p=self.vertices[np.argmax(self.vertices@d.T,axis=0)]
        else:raise ValueError('Unsupported exact shape')
        return p@world[:3,:3].T+world[:3,3]

    def aabb(self,link_world):
        d=np.vstack([np.eye(3),-np.eye(3)]);p=self.support(d,link_world)
        return np.diag(p[3:]),np.diag(p[:3])


def importer_policy():
    raw=CONFIG.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CONFIG_SHA:raise ValueError('Asset config hash mismatch')
    nodes=[n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='UrdfFileCfg']
    if len(nodes)!=1 or any(k.arg is None for k in nodes[0].keywords):raise ValueError('Ambiguous importer policy')
    kwargs={k.arg:k.value for k in nodes[0].keywords}
    expected=dict(merge_fixed_joints=False,replace_cylinders_with_capsules=False,self_collision=True)
    for key,value in expected.items():
        if key not in kwargs or ast.literal_eval(kwargs[key]) is not value:raise ValueError('Explicit policy changed: '+key)
    if 'collider_type' in kwargs:raise ValueError('Default hull assumption requires review')
    return dict(**expected,mesh_collider_type='convex_hull',mesh_default_isaaclab_commit='f73c33173801f5f8afea4142482e47b7710c2b75',
        actual_runtime_policy_observed=False,cooked_shape_identity_verified=False)


def load_shapes():
    policy=importer_policy();R.require_hash(TARGET,R.TARGET_MODEL_SHA256)
    tree=K.RobotTree.read(TARGET);shapes=[]
    for link,node in tree.links.items():
        for index,collision in enumerate(node.findall('collision')):
            geometry=list(collision.find('geometry'))
            if len(geometry)!=1:raise ValueError('One primitive per collision required')
            g=geometry[0];params={};vertices=None
            if g.tag=='box':params['size']=K.vector(g.get('size')).tolist()
            elif g.tag in ('sphere','cylinder'):
                params['radius']=float(g.get('radius'))
                if g.tag=='cylinder':params['length']=float(g.get('length'))
            elif g.tag=='mesh':
                path=(TARGET.parent/g.get('filename')).resolve()
                side=next((s for s in ['left','right'] if link==s+'_wrist_roll_link'),None)
                if side is None:raise ValueError('Unreviewed mesh link')
                R.require_hash(path,S.HAND_MESH_HASHES[side]);scale=K.vector(g.get('scale','1 1 1'))
                if np.any(scale<=0):raise ValueError('Positive mesh scale required')
                vertices=Hull.from_points(np.unique(K.read_triangles(path).reshape(-1,3),axis=0)*scale).vertices
                params=dict(mesh_sha256=S.HAND_MESH_HASHES[side],scale=scale.tolist(),nominal_convex_hull=True,
                    hull_vertices=len(vertices),physx_cooked=False)
            else:raise ValueError('Unreviewed collision geometry')
            if g.tag!='mesh':
                values=np.concatenate([np.asarray(v,float).reshape(-1) for v in params.values()])
                if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Positive finite dimensions required')
            shapes.append(Shape(link+'#'+str(index),link,g.tag,K.origin_transform(collision.find('origin')),params,vertices))
    if len(shapes)!=43 or len({s.link for s in shapes})!=19:raise ValueError('Pinned shape count changed')
    if Counter(s.kind for s in shapes)!=dict(cylinder=24,sphere=14,box=3,mesh=2):raise ValueError('Pinned shape types changed')
    return tree,shapes,policy


def all_pairs(tree,shapes,world):
    if len(shapes)>64:raise ValueError('Shape pair budget exceeded')
    boxes={s.identity:s.aabb(world[s.link]) for s in shapes}
    adjacency={frozenset([j['parent'],j['child']]):j['kind'] for j in tree.joints}
    by_child={j['child']:j for j in tree.joints};rows=[]
    def path_between(a,b):
        def ancestors(n):
            names=[n]
            while n in by_child:n=by_child[n]['parent'];names.append(n)
            return names
        aa,bb=ancestors(a),ancestors(b);common=next(x for x in aa if x in bb)
        return aa[:aa.index(common)+1]+list(reversed(bb[:bb.index(common)]))
    for a,b in combinations(shapes,2):
        alo,ahi=boxes[a.identity];blo,bhi=boxes[b.identity]
        gap=np.maximum(0,np.maximum(alo-bhi,blo-ahi));lower=float(np.linalg.norm(gap));same=a.link==b.link
        rows.append(dict(first=a.identity,second=b.identity,same_link=same,
            directly_joint_connected=frozenset([a.link,b.link]) in adjacency,
            link_graph_path=path_between(a.link,b.link),
            aabb_distance_lower_bound=lower,
            status='same_rigid_link_record_only' if same else ('separated_by_nominal_aabb' if lower>1e-12 else 'unresolved_narrowphase_required'),
            automatically_excluded=False,actual_physx_pair_filter_observed=False))
    return rows,boxes