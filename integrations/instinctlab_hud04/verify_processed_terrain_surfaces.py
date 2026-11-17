"""Independent pinned-upstream terrain and segment/triangle checks, CPU only.

Executes only AST function bodies from SHA256-allowlisted upstream source files.
Does not import IsaacLab, submit tasks, or alter source assets.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh
import yaml
import terrain_surface_geometry as G

SOURCE_HASHES={'utils.py':'40d3427da3a56592367c61b4327647940b0a09360638b1102be092de9b269c01',
    'mesh_terrains.py':'9ac841818fd44d753bad0850a768d631339d0436b6542ca2af6cfad5d9fe8a20'}
RAW_REPORT_HASH='a2c4fe5c2ab03017552391973bd11851c02023b788dcfef18ef1366b344114e6'
TERRAIN_HASH='3aee33efea6793974d901dd0f7e46859e2d12d2dfd79f38f12c777686566c2de'
METADATA_HASH='cbd3010b21111f8fc69a354c37befaf72bd33508a93763edc024b3a29f887ea1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(path,name,scope):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[];node.returns=None
    for arg in node.args.args:arg.annotation=None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    return scope[name]


def independent_distance(triangles,a,b):
    """Convex per-triangle point-distance minimization, without custom broadphase.

    ALL triangles get separate 65-step ternary minimization on the segment;
    endpoint tests are included. Point distances use trimesh's own implementation.
    """
    left=np.zeros(len(triangles));right=np.ones(len(triangles));direction=b-a
    def distance(t):
        p=a+t[:,None]*direction
        nearest=trimesh.triangles.closest_point(triangles,p)
        return np.sum((p-nearest)**2,axis=1)
    for _ in range(65):
        t1=(2*left+right)/3;t2=(left+2*right)/3
        choose=distance(t1)<distance(t2)
        left=np.where(choose,left,t1);right=np.where(choose,t2,right)
    d=np.minimum(distance((left+right)/2),np.minimum(distance(np.zeros(len(left))),distance(np.ones(len(left)))))
    return float(np.sqrt(d.min()))


def verify(upstream_root,data_root,processed_report,raw_report):
    source=upstream_root/'source/instinctlab/instinctlab/terrains/trimesh'
    for name,digest in SOURCE_HASHES.items():
        if sha(source/name)!=digest:raise ValueError('Upstream source hash mismatch: '+name)
    terrain=data_root/'50cmboxes_19_36_49.stl';metadata=data_root/'metadata_kneelClimbStep1.yaml'
    if sha(terrain)!=TERRAIN_HASH or sha(metadata)!=METADATA_HASH:raise ValueError('Pinned original input hash mismatch')
    if sha(raw_report)!=RAW_REPORT_HASH:raise ValueError('Historical raw report hash mismatch')
    scope={'np':np,'trimesh':trimesh,'os':os,'yaml':yaml}
    extract(source/'utils.py','crop_terrain_mesh_aabb',scope)
    terrain_fn=extract(source/'mesh_terrains.py','motion_matched_terrain',scope)
    cfg=SimpleNamespace(metadata_yaml=str(metadata),path=str(data_root),size=(9.,12.))
    upstream_mesh,origin=terrain_fn(0.,cfg)
    mesh,shift,stats=G.processed_terrain(terrain,cfg.size)
    if not np.array_equal(upstream_mesh.faces,mesh.faces):raise ValueError('Crop faces mismatch')
    error=float(np.max(np.abs(upstream_mesh.vertices-(mesh.vertices+shift))))
    origin_error=float(np.max(np.abs(origin-shift)))
    if max(error,origin_error)>1e-12:raise ValueError('Upstream terrain replay mismatch')
    r=json.loads(processed_report.read_text())
    if r['terrain_sha256']!=TERRAIN_HASH or r['tile_size_raw']!=[9.,12.]:raise ValueError('Wrong processed report inputs')
    triangles=np.asarray(mesh.triangles)
    lateral=triangles[np.abs(mesh.face_normals[:,2])<=.1];idx=G.SurfaceIndex(lateral)
    results=[]
    witnesses=r['independent_verification_witnesses']
    if len(witnesses)!=24:raise ValueError('Expected exactly 24 bounded witnesses')
    for w in witnesses:
        a=np.array(w['start_raw']);b=np.array(w['end_raw'])
        if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([a,b]).all():raise ValueError('Invalid witness')
        independent=independent_distance(triangles,a,b)
        independent_side=independent_distance(lateral,a,b)
        side,_,_,_=idx.closest(a,b)
        delta=abs(independent-w['centerline_distance_raw']);side_delta=abs(independent_side-side)
        if max(delta,side_delta)>1e-8:raise ValueError('Independent distance mismatch')
        results.append(dict(anchor=w['anchor'],order=w['order'],link=w['link'],frame=w['frame'],
            all_faces_distance_error=delta,lateral_faces_distance_error=side_delta))
        print('verified',w['anchor'],w['order'],w['link'],flush=True)
    return dict(processed_report_sha256=sha(processed_report),raw_report_sha256=sha(raw_report),
        raw_report_byte_identical_to_historical_hash=True,
        upstream_ast_mesh_vertices_max_abs_error=error,upstream_ast_faces_equal=True,
        upstream_ast_origin_max_abs_error=origin_error,upstream_source_hashes=SOURCE_HASHES,
        independent_method='65 ternary steps minimizing trimesh point-triangle distance separately over EVERY triangle; endpoints included',
        witness_count=len(results),surface_queries_verified=2*len(results),
        max_all_faces_distance_error=max(x['all_faces_distance_error'] for x in results),
        max_lateral_distance_error=max(x['lateral_faces_distance_error'] for x in results),checks=results,
        exporter_provenance_verified=False,hypothesis_selected=False,accepted_reference=False,training_ready=False,
        limitations=['Independent numerical verification on 24 real diagnostic-minimum witnesses, not every one of 6696 queries',
            'Both terrain paths use installed trimesh 5.2.1; original GPU trimesh version remains unverified'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream-root','data-root','processed-report','raw-report','report'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists() or args.report.suffix!='.json':parser.error('New .json output required')
    result=verify(args.upstream_root.resolve(),args.data_root.resolve(),args.processed_report,args.raw_report)
    with args.report.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('independent_verification_passed',len(result['checks']),flush=True)


if __name__=='__main__':main()