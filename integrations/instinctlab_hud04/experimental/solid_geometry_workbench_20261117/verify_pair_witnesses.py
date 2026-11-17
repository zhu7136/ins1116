"""Independent primitive/halfspace membership of two GJK overlap witnesses."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from robot_shape_inventory import load_shapes,TargetTasks

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    output=ROOT/'INDEPENDENT_OVERLAP_WITNESSES_20261117.json'
    if output.exists():raise FileExistsError('Evidence already exists')
    raw=(ROOT/'ALL_SHAPE_PAIR_DETAIL_20261117.json').read_bytes();detail=json.loads(raw)
    supervision=json.loads((ROOT/'SOLID_PAIR_DIAGNOSTIC_20261117.json').read_text())
    assert supervision['status']=='completed'
    expected=supervision['windows'][0]['diagnostic']['detail_sha256']
    assert hashlib.sha256(raw).hexdigest()==expected
    tree,shapes,_=load_shapes();target=TargetTasks();q=np.array(detail['joint_positions'])
    assert all(target.limits[n]['lower']<=v<=target.limits[n]['upper'] for n,v in zip(target.names,q))
    world={n:t[0] for n,t in tree.fk(q[None],target.names).items()};lookup={s.identity:s for s in shapes}
    rows=[]
    for pair in detail['pairs']:
        query=pair.get('narrowphase',{})
        if query.get('status')!='within_numerical_tolerance':continue
        a,b=lookup[pair['first']],lookup[pair['second']]
        assert a.kind=='cylinder' and b.kind=='mesh' and pair['directly_joint_connected']
        first,second=np.array(query['first_witness']),np.array(query['second_witness'])
        ta=world[a.link]@a.local;tb=world[b.link]@b.local
        pa=(first-ta[:3,3])@ta[:3,:3];pb=(second-tb[:3,3])@tb[:3,:3]
        radial=a.parameters['radius']-float(np.linalg.norm(pa[:2]))
        cap=a.parameters['length']/2-abs(float(pa[2]))
        equations=ConvexHull(b.vertices).equations
        violation=float((equations[:,:3]@pb+equations[:,3]).max())
        assert min(radial,cap)>=-1e-12 and violation<=1e-12 and np.linalg.norm(first-second)<1e-12
        rows.append(dict(first=a.identity,second=b.identity,cylinder_radial_interior_margin=radial,
            cylinder_cap_interior_margin=cap,hull_max_halfspace_residual=violation,
            witness_separation=float(np.linalg.norm(first-second)),joint_connected=True,
            actual_runtime_pair_filter_known=False,pair_exclusion_authorized=False,
            physical_penetration_depth_computed=False))
    assert len(rows)==2
    result=dict(stage='independent_membership_of_nominal_static_pose_witnesses',detail_sha256=expected,
        rows=rows,all_29_joint_positions_in_limits=True,head_variables=0,
        source_motion_used=False,original_assets_changed=False,physical_collision_violation_concluded=False,
        retarget_solver_calls=0,gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    assert json.loads(output.read_text())==result
    print(json.dumps(result,indent=2))