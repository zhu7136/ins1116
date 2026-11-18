"""Algebraic derivative/observability verification, not a motion solve."""
import json
import numpy as np
from audit import HERE,ARTIFACT,EXPECTED,sha,load
from verify_full_output import RobotTree
from inputs import TARGET
from orientation_features import evaluate,LINKS

def main():
    _,model,_,_,_=load()
    with np.load(ARTIFACT,allow_pickle=False) as z:allq=model.order(z['joint_pos'],z['joint_names'].tolist())
    frames=[0,257,495,996];q=allq[frames]
    orientation,jac=evaluate(model,q);_,directions_jac=model.evaluate(q,True)
    legacy=RobotTree.read(TARGET).fk(q,model.names)
    independent=np.stack([legacy[l][:,:3,:3] for l in LINKS],axis=1)
    fk_error=float(np.max(np.abs(orientation-independent)))
    error=0.;epsilon=1e-6
    for c in range(29):
        plus=q.copy();minus=q.copy();plus[:,c]+=epsilon;minus[:,c]-=epsilon
        fd=(evaluate(model,plus,False)-evaluate(model,minus,False))/(2*epsilon)
        error=max(error,float(np.max(np.abs(fd-jac[:,:,:,:,c]))))
    combined=np.concatenate([directions_jac.reshape(len(q),24,29),jac.reshape(len(q),45,29)],axis=1)
    sv=np.linalg.svd(combined,compute_uv=False);ranks=(sv>1e-8).sum(axis=1)
    if fk_error>1e-12 or error>1e-8:raise ValueError('Orientation FK/derivative verification failed')
    if not np.all(ranks==29):raise ValueError('Expected full local joint rank in these algebraic configurations')
    if sha(ARTIFACT)!=EXPECTED:raise ValueError('Baseline changed')
    result=dict(stage='target_URDF_orientation_feature_algebraic_verification',frames=frames,features=list(LINKS),
        basis_shape=list(orientation.shape),jacobian_shape=list(jac.shape),independent_legacy_fk_max_error=fk_error,
        finite_difference_step=epsilon,finite_difference_max_error=error,
        combined_direction_and_orientation_rank=ranks.tolist(),rank_tolerance=1e-8,
        minimum_combined_singular_value=float(sv[:,-1].min()),
        finite_difference_perturbations_are_NOT_motion_or_feasibility_tests=True,
        source_target_orientation_alignment_calibrated=False,palm_or_sole_contact_material_calibrated=False,
        world_root_added=False,solver_called=False,new_motion_generated=False,accepted_reference=False,training_ready=False,
        source_code_sha256=sha(HERE/'orientation_features.py'),artifact_sha256=EXPECTED)
    with (HERE/'ORIENTATION_FEATURE_VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()