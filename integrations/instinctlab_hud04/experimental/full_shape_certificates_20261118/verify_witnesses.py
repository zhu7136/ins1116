"""Independent membership recheck; never runs certificate-search functions.

Uses inverse linear solve, scalar primitive inequalities, and independent
Delaunay point-in-convex-hull checks for mesh certificates. Triangle witnesses
are checked by least-squares barycentric coordinates, not polygon clipping or
triangle projection. This validates positive witnesses, NOT all clearances.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import Delaunay
from audit_self_pairs import HERE,REPO,load_case,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-report',type=Path,required=True)
    parser.add_argument('--terrain-report',type=Path,required=True)
    parser.add_argument('--terrain',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--motion',type=Path);parser.add_argument('--motion-sha256')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Refuse verification overwrite')
    start=time.monotonic();K,pins,shapes,_=load_case(args.motion,args.motion_sha256)
    data={s.identity:(s,poses) for _,s,poses,_,_ in shapes}
    meshes={s.identity:Delaunay(s.vertices) for _,s,*_ in shapes if s.kind=='mesh'}
    reports=[json.loads(p.read_text()) for p in [args.self_report,args.terrain_report]]
    for report in reports:
        for p,digest in pins.items():
            key=str(p.relative_to(REPO))
            if report['input_sha256'].get(key)!=digest:
                raise ValueError('Report is not bound to the loaded motion/robot')
        for path,digest in report['input_sha256'].items():
            p=Path(path);p=p if p.is_absolute() else REPO/p
            if sha(p)!=digest:raise ValueError('Input binding mismatch')
        for name,digest in report['code_sha256'].items():
            if sha(HERE/name)!=digest:raise ValueError('Execution code binding mismatch: '+name)
    def inside(identity,frame,point,claimed):
        s,t=data[identity];p=np.linalg.solve(t[frame,:3,:3],np.array(point)-t[frame,:3,3])
        if s.kind=='sphere':margin=s.dimensions[0]-np.sqrt(float(p@p))
        elif s.kind=='box':margin=min((s.dimensions-abs(p)).tolist())
        elif s.kind=='cylinder':margin=min(s.dimensions[0]-np.hypot(p[0],p[1]),s.dimensions[1]-abs(p[2]))
        else:
            if meshes[identity].find_simplex(p,tol=1e-12)<0:raise ValueError('Delaunay hull membership failed')
            margin=-max((s.planes[:,:3]@p+s.planes[:,3]).tolist())
        if not margin>1e-8 or abs(margin-claimed)>1e-10:raise ValueError('Strict membership certificate failed')
        return margin
    ns=nt=nm=0;minimum=float('inf');nonadj=[]
    for row in reports[0]['pairs']:
        for witness in row.get('strict_common_point_certificates',[]):
            f=witness['frame'];p=witness['point']
            a=inside(row['first'],f,p,witness['first_inside_margin_raw'])
            b=inside(row['second'],f,p,witness['second_inside_margin_raw'])
            minimum=min(minimum,a,b);ns+=1
            nm+=sum(identity in meshes for identity in [row['first'],row['second']])
            if not row['directly_joint_connected']:
                nonadj.append(dict(first=row['first'],second=row['second'],frame=f,common_inside_margin_raw=min(a,b)))
    guard=HERE.parent/'nonlinear_guard_20261118/box_triangle_guard.py'
    spec=importlib.util.spec_from_file_location('crop_for_independent_membership',guard)
    G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
    tri,_=G.cropped_triangles(K.read_triangles(args.terrain),[9,12])
    max_residual=0.;min_bary=1.
    for row in reports[1]['per_shape']:
        for witness in row['strict_common_point_certificates']:
            f=witness['frame'];p=np.array(witness['point']);triangle=tri[witness['cropped_triangle_id']]
            margin=inside(row['shape'],f,p,witness['solid_inside_margin_raw'])
            basis=(triangle[1:]-triangle[0]).T
            uv,_,rank,_=np.linalg.lstsq(basis,p-triangle[0],rcond=None)
            bary=np.r_[1-uv.sum(),uv];residual=np.linalg.norm(triangle[0]+basis@uv-p)
            if rank!=2 or np.min(bary)<-1e-9 or residual>1e-10:raise ValueError('Finite triangle membership failed')
            max_residual=max(max_residual,float(residual));min_bary=min(min_bary,float(bary.min()))
            minimum=min(minimum,margin);nt+=1;nm+=row['shape'] in meshes
    if ns!=reports[0]['counts']['strict_common_point_pose_pairs'] or nt!=reports[1]['counts']['strict_surface_inside_solid']:
        raise ValueError('Incomplete positive witness recheck')
    result=dict(stage='independent_positive_witness_membership_NOT_motion_acceptance',
        self_report_sha256=sha(args.self_report),terrain_report_sha256=sha(args.terrain_report),
        self_strict_witnesses_verified=ns,terrain_strict_witnesses_verified=nt,mesh_Delaunay_membership_checks=nm,
        minimum_strict_inside_margin_raw=minimum,maximum_triangle_reconstruction_residual_raw=max_residual,
        minimum_triangle_barycentric_coordinate=min_bary,
        nonadjacent_strict_witnesses_verified=len(nonadj),
        strongest_nonadjacent_witnesses=sorted(nonadj,key=lambda r:r['common_inside_margin_raw'],reverse=True)[:10],
        all_positive_witnesses_verified=True,separation_certificates_independently_rechecked=False,
        cooked_geometry_verified=False,actual_pair_filters_verified=False,dynamics_checked=False,
        motion_optimizer_calls=0,new_motion_files=0,gpu_submissions=0,ppo_updates=0,
        accepted_reference=False,training_ready=False,elapsed_s=time.monotonic()-start)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()