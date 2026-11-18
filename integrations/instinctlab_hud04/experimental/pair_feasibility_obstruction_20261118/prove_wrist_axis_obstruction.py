"""Analytic all-angle nominal obstruction, not an exclusion/physics certificate.

A point on the revolute joint axis is invariant under EVERY joint angle.
If that point is strictly inside both the parent cylinder and child convex hull,
no joint pose or common world/root motion can separate this nominal shape pair.
"""
import hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull,Delaunay
ROOT=Path(__file__).parent
sys.path.insert(0,'/workspace/hud04-runtime-preflight-20261117')
import task_kinematics as TK
TARGET_SHA='865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7'
MESH_SHA={'left':'5aa0af4ccdaf46d498267b0fae247b8d547d8d72dd8db4f7691fe42b567eb5d6','right':'e4804755e9991f05ad5389340e1e83ee1ef687a403896c21c0f785c28ff871fc'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prove(side):
    if sha(TK.TARGET)!=TARGET_SHA:raise ValueError('Model identity')
    xml=ET.parse(TK.TARGET).getroot();joint=next(j for j in xml.findall('joint') if j.get('name')==side+'_wrist_roll_joint')
    origin=TK.K.origin_transform(joint.find('origin'));axis=TK.K.vector(joint.find('axis').get('xyz'))
    if not np.array_equal(axis,[1,0,0]) or not np.array_equal(origin[:3,:3],np.eye(3)):raise ValueError('Reviewed exact invariant axis required')
    parent=joint.find('parent').get('link');child=joint.find('child').get('link')
    links={l.get('name'):l for l in xml.findall('link')};c=links[parent].findall('collision')[1];m=links[child].findall('collision')[0]
    if list(c.find('geometry'))[0].tag!='cylinder' or list(m.find('geometry'))[0].tag!='mesh':raise ValueError('Original shape types')
    cylinder=TK.K.origin_transform(c.find('origin'));mesh=TK.K.origin_transform(m.find('origin'))
    if not np.array_equal(cylinder[:3,3],origin[:3,3]) or not np.array_equal(mesh,np.eye(4)):raise ValueError('Reviewed concentric original geometry required')
    g=m.find('geometry/mesh');path=(TK.TARGET.parent/g.get('filename')).resolve()
    if sha(path)!=MESH_SHA[side] or g.get('scale')!='1 1 1':raise ValueError('Original mesh identity/scale')
    vertices=np.unique(TK.K.read_triangles(path).reshape(-1,3),axis=0);hull=ConvexHull(vertices);planes=hull.equations
    radius=float(c.find('geometry/cylinder').get('radius'));half=float(c.find('geometry/cylinder').get('length'))/2
    # Inside an inscribed sphere of this cylinder suffices for cylinder membership.
    # This is a conservative proof only; it does NOT replace cylinder geometry.
    inscribed=min(radius,half)
    matrix=np.c_[planes[:,0],np.ones(len(planes))]
    matrix=np.vstack([matrix,[1,1],[-1,1]]);rhs=np.r_[-planes[:,3],inscribed,inscribed]
    lp=linprog([0,-1],A_ub=matrix,b_ub=rhs,bounds=[(-inscribed,inscribed),(0,None)],method='highs',options={'time_limit':1,'maxiter':2000})
    if not lp.success or lp.x[1]<=1e-8:raise ValueError('No invariant strict witness found; no impossibility claim')
    t,slack=lp.x;point=np.array([t,0.,0.]);parent_point=origin[:3,3]+point
    clocal=np.linalg.solve(cylinder[:3,:3],parent_point-cylinder[:3,3])
    cmargin=min(radius-np.linalg.norm(clocal[:2]),half-abs(clocal[2]));hmargin=float(-(planes[:,:3]@point+planes[:,3]).max())
    if min(cmargin,hmargin)<=1e-8 or Delaunay(vertices[hull.vertices]).find_simplex(point)<0:raise ValueError('Independent witness verification')
    limits=joint.find('limit');angles=np.linspace(float(limits.get('lower')),float(limits.get('upper')),1001)
    rotated=np.stack([TK.K.axis_rotation(axis,float(a))@point for a in angles])
    if not np.array_equal(rotated,np.broadcast_to(point,rotated.shape)):raise ValueError('Axis invariance')
    return dict(first=parent+'#1',second=child+'#0',joint=joint.get('name'),axis_child=axis.tolist(),point_child=point.tolist(),point_parent=parent_point.tolist(),
        cylinder_radial_margin_raw=float(radius-np.linalg.norm(clocal[:2])),cylinder_cap_margin_raw=float(half-abs(clocal[2])),hull_halfspace_margin_raw=hmargin,
        invariant_for_all_angles_analytic=True,supplementary_grid_angles=1001,original_joint_limits=[float(limits.get('lower')),float(limits.get('upper'))],
        Delaunay_point_membership=True,all_upstream_joints_and_world_rigid_motions_preserve_intersection=True,
        reason='R_x(theta) @ (t,0,0) == (t,0,0) for every real theta. Same strict common point remains in both original nominal solids.',
        witness_is_not_physical_penetration_depth=True,mesh_sha256=sha(path))
def main():
    out=ROOT/'INVARIANT_WRIST_OBSTRUCTION.json'
    if out.exists():raise FileExistsError('No replay')
    result=dict(stage='ANALYTIC_nominal_all_angle_obstruction_NOT_runtime_collision',model_sha256=sha(TK.TARGET),pairs=[prove(s) for s in ['left','right']],
        impossible_contract='Demand ALL different-link nominal shapes strictly disjoint, including both wrist joint-connected cylinder/hull pairs, at any pose',
        no_claim_genuine_task_infeasible=True,actual_runtime_filters_verified=False,automatic_pair_exclusions=0,
        source_motion_modified=False,new_motion_files=0,motion_solver_calls=0,gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    if json.loads(out.read_text())!=result:raise ValueError('Readback')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()