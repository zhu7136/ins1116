"""Independent read-only recomputation of the 002 partial numerical result."""
import json
from pathlib import Path
import numpy as np
from material_problem import HERE, problem, feasibility, sha

OUT=HERE/'TRIAL_MATERIAL_002_HARD_CLEARANCE_PROBE'


def main():
    m,q,r,want,w,support,body,guard,kernel=problem()
    with np.load(OUT/'WINDOW0_14_UNACCEPTED.npz',allow_pickle=False) as z:
        assert z['joint_names'].tolist()==m.names
        assert z['frame_indices'].tolist()==list(range(15))
        assert not bool(z['accepted_reference']) and not bool(z['training_ready'])
        tq,tr=z['joint_pos'],z['root_transforms_w']
    assert tq.shape==(15,29) and tr.shape==(15,4,4)
    old=guard.scan(m,q[:15],r[:15],np.arange(15))
    new=guard.scan(m,tq,tr,np.arange(15))
    passed,check=guard.compare(old,new)
    assert passed and not check['newly_strict_pairs'] and not check['lost_certified_separation_pairs']
    v0=m.evaluate(q[:15],r[:15]);v1=m.evaluate(tq,tr)
    oldgap,_,_=body.evaluate(m,q[:15],r[:15],False)
    newgap,_,_=body.evaluate(m,tq,tr,False)
    initial=float(abs((v0-want[:15])*w[:15]).sum()+1000*np.maximum(body.margin-oldgap,0).sum())
    final=float(abs((v1-want[:15])*w[:15]).sum()+1000*np.maximum(body.margin-newgap,0).sum())
    report=json.loads((OUT/'RESULT.json').read_text())
    accepted=next(x for x in report['backtracking'] if x['engineering_step_accepted'])
    assert abs(initial-accepted['old_merit'])<1e-8 and abs(final-accepted['new_merit'])<1e-8
    fullq=q.copy();fullq[:15]=tq
    limits=feasibility(m,fullq)
    assert max(limits.values())<1e-6
    out=dict(endpoint_check_recomputed=check,objective_recomputed=dict(before=initial,after=final,
             relative_decrease_percent=100*(initial-final)/initial),original_joint_limits_and_neighbour_velocities=limits,
             material_position_mean_before_m=float(np.linalg.norm((v0[:,:18]-want[:15,:18]).reshape(15,6,3),axis=2).mean()),
             material_position_mean_after_m=float(np.linalg.norm((v1[:,:18]-want[:15,:18]).reshape(15,6,3),axis=2).mean()),
             root_and_body_are_joint_variables=True,partial_window_exported=True,full_motion_exported=False,
             full_997_frame_solve=False,accepted_reference=False,training_ready=False,
             strict_pairs_still_present=len(new.strict),unresolved_pairs_still_present=len(new.unresolved),
             physical_effort_verified=False,swept_motion_verified=False,
             input_sha256={p.name:sha(p) for p in [OUT/'RESULT.json',OUT/'WINDOW0_14_UNACCEPTED.npz']},
             software_source_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.py'))})
    guard.verify_inputs()
    target=OUT/'INDEPENDENT_RECOMPUTATION.json'
    with target.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()