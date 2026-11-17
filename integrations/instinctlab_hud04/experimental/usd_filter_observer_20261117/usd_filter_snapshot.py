"""Read-only USD collision metadata collector. No simulation, import or solver.

USD-schema observations are NOT a PhysX pair-filter/cooked-geometry receipt.
Never infer exclusions from URDF adjacency, name suffixes or absent properties.
"""
import hashlib
import json
import math
from pathlib import Path
import struct


def _value(value):
    if value is None or isinstance(value,(bool,str,int)):return value
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('Nonfinite USD metadata')
        return value
    try:return [_value(x) for x in value]
    except TypeError:return str(value)


def _attribute(prim,name):
    attr=prim.GetAttribute(name)
    if not attr:return dict(present=False,authored=False,value=None)
    return dict(present=True,authored=attr.HasAuthoredValueOpinion(),value=_value(attr.Get()),
                time_samples=attr.GetNumTimeSamples())


def collect(stage,robot_root,*,max_prims=4096,max_points=65536):
    """Traverse composed instance proxies, retain schema fallbacks as unauthored."""
    from pxr import Usd,UsdGeom,UsdPhysics
    if not isinstance(robot_root,str) or not robot_root.startswith('/') or robot_root=='/':raise ValueError('Explicit robot root required')
    root=stage.GetPrimAtPath(robot_root)
    if not root:raise ValueError('Robot root not found')
    if not 1<=max_prims<=10000 or not 1<=max_points<=100000:raise ValueError('Bounded snapshot required')
    joints=[];colliders=[];filters=[];groups=[];articulations=[];prims=0;point_budget=0;index_budget=0
    xforms=UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        prims+=1
        if prims>max_prims:raise RuntimeError('USD prim budget exceeded; no partial success')
        path=str(prim.GetPath());inside=prim.GetPath().HasPrefix(root.GetPath())
        if inside and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulations.append(dict(path=path,self_collision=_attribute(prim,'physxArticulation:enabledSelfCollisions')))
        if inside and prim.IsA(UsdPhysics.Joint):
            joint=UsdPhysics.Joint(prim)
            joints.append(dict(path=path,type=prim.GetTypeName(),body0=[str(p) for p in joint.GetBody0Rel().GetTargets()],
                body1=[str(p) for p in joint.GetBody1Rel().GetTargets()],collision_enabled=_attribute(prim,'physics:collisionEnabled'),
                joint_enabled=_attribute(prim,'physics:jointEnabled'),exclude_from_articulation=_attribute(prim,'physics:excludeFromArticulation'),
                actual_physx_pair_filter_verified=False))
        # Global explicit relations can affect robot bodies even if owner is outside
        # robot root. Capture them, do not attempt undocumented filter inheritance.
        relation=prim.GetRelationship('physics:filteredPairs')
        if relation and relation.GetTargets():
            filters.append(dict(owner=path,targets=[str(p) for p in relation.GetTargets()]))
        if prim.IsA(UsdPhysics.CollisionGroup):
            groups.append(dict(path=path,relationships={r.GetName():[str(p) for p in r.GetTargets()] for r in prim.GetRelationships()},
                invert_filtered_groups=_attribute(prim,'physics:invertFilteredGroups')))
        if not inside or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        ancestor=prim;body=None
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):body=str(ancestor.GetPath());break
            ancestor=ancestor.GetParent()
        matrix=xforms.GetLocalToWorldTransform(prim)
        record=dict(path=path,type=prim.GetTypeName(),rigid_body_path=body,instance_proxy=prim.IsInstanceProxy(),
            collision_enabled=_attribute(prim,'physics:collisionEnabled'),
            world_transform_gf_row_vector=[[_value(float(x)) for x in row] for row in matrix],
            contact_offset=_attribute(prim,'physxCollision:contactOffset'),rest_offset=_attribute(prim,'physxCollision:restOffset'),
            approximation=_attribute(prim,'physics:approximation'),cooked_geometry_verified=False)
        for key in ['radius','height','axis','size','extent','xformOp:scale']:
            record[key]=_attribute(prim,key)
        if prim.IsA(UsdGeom.Mesh):
            mesh=UsdGeom.Mesh(prim);points=mesh.GetPointsAttr().Get();indices=mesh.GetFaceVertexIndicesAttr().Get();counts=mesh.GetFaceVertexCountsAttr().Get()
            n=0 if points is None else len(points);point_budget+=n
            if point_budget>max_points:raise RuntimeError('USD mesh point budget exceeded; no partial success')
            index_budget+=(0 if indices is None else len(indices))+(0 if counts is None else len(counts))
            if index_budget>12*max_points:raise RuntimeError('USD topology budget exceeded; no partial success')
            h=hashlib.sha256()
            if points is not None:
                for point in points:
                    p=tuple(float(v) for v in point)
                    if not all(math.isfinite(v) for v in p):raise ValueError('Nonfinite mesh vertex')
                    h.update(struct.pack('<3d',*p))
            record['mesh']=dict(point_count=n,point_hash_encoding='ordered points as <3d per vertex',points_sha256=h.hexdigest(),
                face_vertex_counts=[] if counts is None else [int(x) for x in counts],
                face_indices_sha256=hashlib.sha256(b'' if indices is None else b''.join(struct.pack('<q',int(x)) for x in indices)).hexdigest(),
                usd_points_are_not_physx_cooked_vertices=True)
        colliders.append(record)
    result=dict(schema_version=1,stage_observation_kind='composed_USD_metadata_ONLY',robot_root=robot_root,
        usd_library_version=list(Usd.GetVersion()),meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
        meters_per_unit_authored=stage.HasAuthoredMetadata('metersPerUnit'),up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        source_motion_units_or_alignment_proven=False,visited_prims=prims,articulations=articulations,joints=joints,
        colliders=colliders,global_filtered_pairs=filters,collision_groups=groups,
        cooked_geometry_verified=False,actual_physx_pair_filters_verified=False,contact_schedule_accepted=False,
        automatic_pair_exclusions=0,gpu_started=False,solver_calls=0,ppo_updates=0,accepted_reference=False,training_ready=False)
    if len(json.dumps(result,allow_nan=False).encode())>512000:raise RuntimeError('Snapshot output budget exceeded')
    return result


def compare_inventory(snapshot,inventory):
    """Compare exact rigid-body leaf names/counts; no collision-index/name guessing."""
    from collections import Counter
    expected=Counter(s['link'] for s in inventory['shapes']);actual=Counter();body_paths={}
    missing_body_paths=[]
    for row in snapshot['colliders']:
        if row['rigid_body_path'] is None:missing_body_paths.append(row['path'])
        else:
            leaf=row['rigid_body_path'].rsplit('/',1)[-1];actual[leaf]+=1
            body_paths.setdefault(leaf,set()).add(row['rigid_body_path'])
    ambiguous={leaf:sorted(paths) for leaf,paths in body_paths.items() if len(paths)>1}
    return dict(expected_shape_count=sum(expected.values()),observed_usd_collider_count=len(snapshot['colliders']),
        expected_per_link=dict(expected),observed_per_body_leaf=dict(actual),missing_rigid_body_paths=missing_body_paths,
        ambiguous_body_leaf_names=ambiguous,exact_body_leaf_counts_match=expected==actual and not missing_body_paths and not ambiguous,
        matching_counts_prove_cooked_geometry=False,automatic_pair_exclusions=0,training_ready=False)


def write_new(path,result):
    path=Path(path);raw=json.dumps(result,indent=2,allow_nan=False).encode()
    with path.open('xb') as stream:stream.write(raw)
    if path.read_bytes()!=raw:raise RuntimeError('Snapshot readback mismatch')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True);parser.add_argument('--robot-root',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    from pxr import Usd
    # Explicit local evidence only: this tool never opens a remote asset URL.
    source=Path(args.stage).resolve()
    if not source.is_file():raise FileNotFoundError('Local stage evidence required')
    stage=Usd.Stage.Open(str(source))
    if stage is None:raise RuntimeError('USD stage failed to open')
    result=collect(stage,args.robot_root)
    result['input_stage_file_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    result['input_stage_file_is_not_full_dependency_hash']=True
    write_new(args.out,result)