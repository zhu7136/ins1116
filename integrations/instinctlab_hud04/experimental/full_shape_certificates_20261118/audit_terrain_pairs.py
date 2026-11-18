"""All997/all43 nominal shape vs cropped finite terrain surface certificates.

Raw coordinate hypothesis only. Does not label support, infer physical alignment,
close terrain, modify motion, invoke a motion optimizer, or authorize acceptance.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from audit_self_pairs import HERE,INTEGRATION,REPO,load_case,sha
from nominal_certificates import classify_triangles


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terrain',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--motion',type=Path);parser.add_argument('--motion-sha256')
    args=parser.parse_args()
    if not args.run:
        print(json.dumps(dict(plan_only=True,frames=997,shapes=43,optimizer_calls=0,gpu_submissions=0)));return
    if args.output.exists():raise FileExistsError('Refuse evidence overwrite')
    start=time.monotonic();K,pins,shapes,_=load_case(args.motion,args.motion_sha256)
    pins[args.terrain]='3aee33efea6793974d901dd0f7e46859e2d12d2dfd79f38f12c777686566c2de'
    guard=HERE.parent/'nonlinear_guard_20261118/box_triangle_guard.py'
    prior=json.loads((guard.parent/'FRESH_SAT_AUDIT.json').read_text())
    pins[guard]=prior['code_sha256'][guard.name]
    for p,digest in pins.items():
        if sha(p)!=digest:raise ValueError('Input changed: '+str(p))
    spec=importlib.util.spec_from_file_location('pinned_cropper',guard)
    G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
    tri,raw_ids=G.cropped_triangles(K.read_triangles(args.terrain),[9,12])
    if len(tri)!=8068:raise ValueError('Pinned crop changed')
    tl=tri.min(axis=1);th=tri.max(axis=1);rows=[];totals=Counter();all_frames=set()
    for link,s,poses,lo,hi in shapes:
        jobs=[]
        for first in range(0,997,64):
            last=min(997,first+64)
            overlap=np.all(th[None]>=lo[first:last,None,:]-1e-8,axis=2)&np.all(tl[None]<=hi[first:last,None,:]+1e-8,axis=2)
            f,t=np.nonzero(overlap);jobs.extend(zip((f+first).tolist(),t.tolist()))
        counts=Counter();hits=[];unresolved=[];strict_frames=set();separated=[]
        for first in range(0,len(jobs),512):
            batch=jobs[first:first+512];frames=np.array([f for f,t in batch]);ids=np.array([t for f,t in batch])
            r=classify_triangles(s,poses[frames],tri[ids])
            counts['support_separated']+=int(r['separated'].sum())
            counts['strict_surface_inside_solid']+=int(r['strict'].sum())
            counts['unresolved']+=int(r['unresolved'].sum())
            for k in np.flatnonzero(r['strict']):
                f=int(frames[k]);tid=int(ids[k]);strict_frames.add(f);all_frames.add(f)
                hits.append(dict(frame=f,cropped_triangle_id=tid,raw_triangle_id=int(raw_ids[tid]),
                    point=r['common_point'][k].tolist(),solid_inside_margin_raw=float(r['solid_inside_margin'][k]),
                    finite_triangle_projection_residual_raw=float(r['triangle_residual'][k])))
            unresolved.extend([list(batch[k]) for k in np.flatnonzero(r['unresolved'])])
            separated.extend([list(batch[k]) for k in np.flatnonzero(r['separated'])])
        if sum(counts.values())!=len(jobs):raise ArithmeticError('Incomplete candidate accounting')
        counts['aabb_candidates']=len(jobs);counts['aabb_separated']=997*len(tri)-len(jobs)
        totals.update(counts)
        rows.append(dict(shape=s.identity,kind=s.kind,counts=dict(counts),
            strict_hit_frames=sorted(strict_frames),strict_common_point_certificates=hits,
            unresolved_frame_triangle_pairs=unresolved,support_separated_frame_triangle_pairs=separated))
    for p,digest in pins.items():
        if sha(p)!=digest:raise ValueError('Input changed during audit')
    result=dict(stage='all997_all43_nominal_finite_terrain_surface_certificates_not_acceptance',
        base_commit='86f05dffa8696465147d7e3ebf99aff568a84c26',frames=997,shapes=43,terrain_triangles=len(tri),
        input_sha256={str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p):d for p,d in pins.items()},
        code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},counts=dict(totals),
        frames_with_strict_surface_intersection=sorted(all_frames),per_shape=rows,
        all_candidates_classified=True,all_narrowphase_resolved=totals['unresolved']==0,
        tolerance_raw=1e-8,projection_search_iteration_cap=64,
        root_world_units_inferred=False,contact_labels_inferred=False,closed_terrain_containment_excluded=False,
        cooked_geometry_verified=False,swept_geometry_checked=False,dynamics_checked=False,
        motion_optimizer_calls=0,new_motion_files=0,gpu_submissions=0,ppo_updates=0,
        accepted_reference=False,training_ready=False,elapsed_s=time.monotonic()-start)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['frames','shapes','terrain_triangles','counts','elapsed_s','accepted_reference']},indent=2),flush=True)


if __name__=='__main__':main()