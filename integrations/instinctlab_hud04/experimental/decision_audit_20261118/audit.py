"""Read-only numerical audit; no optimizer, motion export or GPU calls.

Finite single-coordinate probes are independent counterfactual evaluations of
the SAME saved baseline. Never accumulate them into a trajectory.
"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=Path('/workspace/hud04-full-direction-20261118')
sys.path.insert(0,str(PRIOR))
from inputs import load,sha

ARTIFACT=PRIOR/'OVERLAP_V4/STEP1_FULL_LENGTH_JOINT_DIAGNOSTIC.npz'
EXPECTED='2aa620d3086ed5f1c260c210e38fcb3d3a94cbaea549911844b8e4c31127ddc0'

def angles(a,b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1)))

def feasible_box(model,q,frame):
    """Exact allowable one-frame coordinates with both temporal neighbours fixed."""
    lo=model.lower.copy();hi=model.upper.copy();delta=model.vmax/90
    for neighbour in (frame-1,frame+1):
        if 0<=neighbour<len(q):
            lo=np.maximum(lo,q[neighbour]-delta)
            hi=np.minimum(hi,q[neighbour]+delta)
    if np.max(lo-q[frame])>1e-6 or np.max(q[frame]-hi)>1e-6:
        raise ValueError('Baseline not feasible under unchanged limits')
    return lo,hi

def main():
    start=time.monotonic();source,target,seed,want,bindings=load()
    if sha(ARTIFACT)!=EXPECTED:raise ValueError('Saved baseline SHA mismatch')
    before_hashes={str(p):sha(p) for p in (ARTIFACT,PRIOR/'inputs.py',PRIOR/'overlap_solver.py')}
    with np.load(ARTIFACT,allow_pickle=False) as z:
        if set(z.files)!={'joint_pos','joint_names','frame_indices','framerate','role'}:raise ValueError('Diagnostic contract')
        q=target.order(z['joint_pos'],z['joint_names'].tolist())
        np.testing.assert_array_equal(z['frame_indices'],np.arange(997))
        if float(z['framerate'].item())!=90:raise ValueError('Native 90Hz required')
    actual,jac=target.evaluate(q,True);errors=angles(actual,want)
    flat=jac.reshape(997,24,29)
    column_max=np.max(np.abs(flat),axis=(0,1));zero=[n for n,v in zip(target.names,column_max) if v<1e-10]
    singular=np.linalg.svd(flat,compute_uv=False);ranks=(singular>1e-8).sum(axis=1)
    frames=list(range(244,346))+list(range(480,511))
    records=[];probes=[];metadata=[]
    for frame in frames:
        lo,hi=feasible_box(target,q,frame)
        for joint,name in enumerate(target.names):
            for sign,slack in ((-1,q[frame,joint]-lo[joint]),(1,hi[joint]-q[frame,joint])):
                # Choose an admissible probe step; do NOT clip a joint position.
                size=min(.01,float(slack))
                if size<1e-8:continue
                trial=q[frame].copy();trial[joint]+=sign*size
                if np.max(lo-trial)>1e-6 or np.max(trial-hi)>1e-6:raise ValueError('Probe feasibility')
                probes.append(trial);metadata.append((frame,joint,sign*size))
    improvements={frame:[] for frame in frames}
    for begin in range(0,len(probes),500):
        if time.monotonic()-start>25:raise RuntimeError('Finite audit time budget exceeded')
        batch=np.asarray(probes[begin:begin+500]);evaluated=target.evaluate(batch)
        for k,(frame,joint,step) in enumerate(metadata[begin:begin+500]):
            old=float(np.abs(actual[frame]-want[frame]).sum()+.001*np.abs(q[frame]-seed[frame]).sum())
            new=float(np.abs(evaluated[k]-want[frame]).sum()+.001*np.abs(batch[k]-seed[frame]).sum())
            if old-new>1e-8:
                ae=angles(evaluated[k],want[frame]);improvements[frame].append(dict(joint=target.names[joint],step_rad=step,
                    same_objective_decrease=old-new,mean_angle_before=float(errors[frame].mean()),mean_angle_after=float(ae.mean()),
                    right_forearm_before=float(errors[frame,7]),right_forearm_after=float(ae[7])))
    for first,last in ((244,345),(480,510)):
        subset=list(range(first,last+1));records.append(dict(first=first,last=last,frames=len(subset),
            frames_with_single_coordinate_feasible_descent=sum(bool(improvements[f]) for f in subset),
            sampled_frames_without_descent=[f for f in subset if not improvements[f]],
            no_descent_is_NOT_stationarity_or_infeasibility_proof=True))
    report=json.loads((PRIOR/'OVERLAP_V4/FULL_DIRECTION_REPORT.json').read_text())
    sample={str(f):sorted(improvements[f],key=lambda x:-x['same_objective_decrease'])[:3] for f in (244,257,283,325,345,480,495,510)}
    result=dict(stage='read_only_progress_and_observability_audit',artifact_sha256=EXPECTED,
        script_sha256=sha(Path(__file__)),input_bindings=bindings,full_frames=997,
        direction_observations_per_frame=8,maximum_independent_direction_constraints_per_frame=16,
        body_joint_variables=29,zero_sensitivity_joint_columns_over_saved_997_frames=zero,
        numerical_direction_jacobian_rank_tolerance=1e-8,numerical_rank_min=int(ranks.min()),numerical_rank_max=int(ranks.max()),
        minimum_local_linearized_nullity=int(29-ranks.max()),
        observation_limit='Direction-only objective cannot establish wrist/ankle orientation, contact position, world root or dynamics. Numerical rank is local evidence, not a global observability proof.',
        probe_method='Independent +/- one-coordinate feasible perturbations of unchanged saved baseline; max .01 rad, exact neighbour speed and named position bounds; same L1 direction + .001 original-seed regularizer.',
        probed_frames=len(frames),counterfactual_evaluations=len(probes),regions=records,sampled_best_probes=sample,
        all_84_updates_at_trust_boundary=sum(row.get('max_joint_step',0)>.149 for row in report['history']),
        last_recorded_update_objective_drop=report['history'][-1]['local_objective_before']-report['history'][-1]['local_objective_after'],
        interpretation='Available feasible descent disproves convergence at affected frames, NOT model incompatibility; it does not prove final attainability or correct anatomical/contact semantics.',
        solver_calls=0,new_trajectory_generated=False,baseline_overwritten=False,gpu_submissions=0,ppo_updates=0,
        accepted_reference=False,training_ready=False)
    for p,digest in before_hashes.items():
        if sha(Path(p))!=digest:raise ValueError('Prior evidence changed')
    result['prior_evidence_hashes_unchanged']=before_hashes
    result['elapsed_seconds']=time.monotonic()-start
    with (HERE/'DECISION_AUDIT.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()