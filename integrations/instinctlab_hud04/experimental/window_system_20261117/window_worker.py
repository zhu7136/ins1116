"""Diagnostic matrix construction under the existing finite supervisor."""
import hashlib,os
from pathlib import Path
import numpy as np
from named_target import NamedTarget,MODEL_SHA256
from window_problem import build,named_order,summarize

def construct(gate,payload):
    # Operator-specified target model + already saved joint-only seed. No source
    # root/quaternion interpretation or source-to-world selection is performed.
    model=Path(payload['model']);fit=Path(payload['fit'])
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (model,fit)}
    if before[str(model)]!=MODEL_SHA256 or before[str(fit)]!='48f8d43e058420e5ff89861d2336ead3317dc18223c36a7305503f3df54a66eb':raise ValueError('Pinned input hash')
    target=NamedTarget(model)
    with np.load(fit,allow_pickle=False) as z:
        q=z['joint_pos'].astype(float);names=z['joint_names'].tolist()
        if q.shape!=(997,29) or z['framerate'].item()!=90 or any(n in z.files for n in ('base_pos_w','base_quat_w')):raise ValueError('Joint-only input contract')
    windows=((611,641),(659,689),(711,741));frames=np.concatenate([np.arange(a,b+1) for a,b in windows])
    # DELIBERATELY algebraic roots, not candidate motion. Unknown target world
    # placement remains unknown, rather than secretly copying source pelvis.
    roots=np.tile(np.eye(4),(len(frames),1,1));roots[:,:3,3]=[2.,-1.,1.]
    gate.step();system=build(target,q[frames],names,roots,frames,desired_points=None)
    if system.tracking_residual is not None:raise ValueError('Do not synthesize desired targets')
    report=summarize(system)
    if report['zero_joint_increment_max_violation']>1e-6:raise ValueError('Seed no longer satisfies hard joint rows')
    # Verify complete batched Jacobian action against a small finite step.
    rng=np.random.default_rng(173539);direction=rng.normal(size=(len(frames),35));eps=1e-6
    from named_target import perturb
    plus=[];minus=[];ordered=q[frames][:,named_order(names,target.names)]
    for row in range(len(frames)):
        if row%31==0:gate.step()
        d=direction[row]*eps
        plus.append(target.evaluate(ordered[row]+d[6:],perturb(roots[row],d[:6]),False))
        minus.append(target.evaluate(ordered[row]-d[6:],perturb(roots[row],-d[:6]),False))
    fd=(np.array(plus)-np.array(minus)).reshape(-1)/(2*eps)
    error=float(np.max(np.abs(fd-system.tracking_jacobian@direction.reshape(-1))))
    if error>2e-8:raise ValueError('Window derivative discrepancy')
    if before!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (model,fit)}:raise ValueError('Input changed')
    report.update(windows_zero_based_inclusive=[list(w) for w in windows],
        actual_joint_seed_frames=997,selected_construction_frames=93,actual_temporal_edges=90,
        independent_window_directional_finite_difference_max_error=error,
        root_seed_kind='algebraic_SE3_software_fixture_NOT_source_or_candidate_motion',
        anatomical_targets='NOT_SUPPLIED_no_implicit_zero_residual',
        no_source_coordinate_hypothesis_selected=True,no_support_labels_generated=True,
        input_sha256=before,inputs_unchanged=True,full_retarget_solver_calls=0,gpu_submissions=0,ppo_updates=0)
    return report