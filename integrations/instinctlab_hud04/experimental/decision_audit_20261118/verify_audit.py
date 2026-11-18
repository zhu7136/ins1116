"""Independent legacy-FK checks of audit's surprising numerical claims."""
import json,sys
from pathlib import Path
import numpy as np
from audit import HERE,PRIOR,ARTIFACT,EXPECTED,sha,load
from inputs import SOURCE,TARGET,MOTION
from verify_full_output import RobotTree,directions,angles

def main():
    audit=json.loads((HERE/'DECISION_AUDIT.json').read_text())
    _,model,seed,want,_=load()
    with np.load(ARTIFACT,allow_pickle=False) as z:q=model.order(z['joint_pos'],z['joint_names'].tolist())
    target=RobotTree.read(TARGET);source=RobotTree.read(SOURCE)
    with np.load(MOTION,allow_pickle=False) as z:sq=z['joint_pos'].copy();sn=z['joint_names'].tolist()
    independent_want=directions(source,sq,sn,True)
    np.testing.assert_allclose(independent_want,want,atol=1e-12,rtol=0)
    baseline=directions(target,q,model.names,False)
    zero_errors=[]
    # Independent finite differences for every claimed zero column at 3 poses.
    for frame in (0,257,495):
        for name in audit['zero_sensitivity_joint_columns_over_saved_997_frames']:
            j=model.names.index(name);probe=q[frame:frame+1].copy();probe[0,j]+=1e-6
            changed=directions(target,probe,model.names,False)
            zero_errors.append(float(np.max(np.abs(changed-baseline[frame]))))
    if max(zero_errors)>1e-12:raise ValueError('Zero sensitivity independent check failed')
    # Verify a right-arm improvement at the WORST frame, not merely another limb.
    frame=257;entry=next(x for x in audit['sampled_best_probes']['257'] if x['joint']=='right_elbow_joint')
    j=model.names.index(entry['joint']);trial=q[frame].copy();trial[j]+=entry['step_rad']
    if np.any(trial<model.lower-1e-6) or np.any(trial>model.upper+1e-6):raise ValueError('Named position feasibility')
    for n in (frame-1,frame+1):
        if np.any(np.abs(trial-q[n])*90>model.vmax+1e-6):raise ValueError('Native neighbour speed feasibility')
    changed=directions(target,trial[None],model.names,False)[0]
    old=float(np.abs(baseline[frame]-independent_want[frame]).sum()+.001*np.abs(q[frame]-seed[frame]).sum())
    new=float(np.abs(changed-independent_want[frame]).sum()+.001*np.abs(trial-seed[frame]).sum())
    if not new<old:raise ValueError('Independent same-objective descent failed')
    before=angles(baseline[frame:frame+1],independent_want[frame:frame+1])[0,7]
    after=angles(changed[None],independent_want[frame:frame+1])[0,7]
    np.testing.assert_allclose([before,after,old-new],[entry['right_forearm_before'],entry['right_forearm_after'],entry['same_objective_decrease']],atol=1e-12,rtol=0)
    if sha(ARTIFACT)!=EXPECTED:raise ValueError('Baseline overwritten')
    result=dict(stage='independent_read_only_audit_verification',method='legacy RobotTree, separately explicit pivots, direct native position/neighbour speed checks',
        source_direction_max_disagreement=float(np.max(np.abs(independent_want-want))),
        zero_sensitivity_checks=len(zero_errors),max_independent_zero_direction_change=max(zero_errors),
        worst_frame=257,joint=entry['joint'],step_rad=entry['step_rad'],right_forearm_before=float(before),right_forearm_after=float(after),
        same_objective_decrease=old-new,native_position_and_neighbour_speed_pass=True,
        audit_sha256=sha(HERE/'DECISION_AUDIT.json'),artifact_sha256=EXPECTED,solver_calls=0,new_trajectory_generated=False,
        accepted_reference=False,training_ready=False)
    with (HERE/'INDEPENDENT_VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()