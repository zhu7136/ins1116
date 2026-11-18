"""Frame-index-only triage against CONDITIONED nominal G1 near-surface runs.

Not contact labels, not material mapping, not target collision/support tests.
"""
import json
from pathlib import Path
import numpy as np
from inputs import HERE,SOURCE,TARGET,MOTION,sha
from verify_full_output import directions,angles,RobotTree,runs

def main():
    existing=Path('/workspace/hud04-support-review-20261117/FULL_STEP1_MATERIAL_SUPPORT_REVIEW_20261117_v2.json')
    prior=json.loads(existing.read_text())
    if prior['source_frames']!=997 or prior['fps']!=90 or prior['accepted_support_intervals'] is not None:raise ValueError('Conditioned source support audit contract')
    result_path=HERE/'OVERLAP_V4'/'STEP1_FULL_LENGTH_JOINT_DIAGNOSTIC.npz'
    with np.load(MOTION,allow_pickle=False) as z:sq=z['joint_pos'].copy();sn=z['joint_names'].tolist()
    with np.load(result_path,allow_pickle=False) as z:q=z['joint_pos'].copy();names=z['joint_names'].tolist()
    actual=directions(RobotTree.read(TARGET),q,names,False);desired=directions(RobotTree.read(SOURCE),sq,sn,True);error=angles(actual,desired)
    rows=[]
    for row in prior['links']:
        link=row['link'];side=0 if link.startswith('left_') else 4
        # Only a same-limb risk association; never equal source/target contacts.
        indices=[side+2,side+3] if 'wrist' in link else [side,side+1]
        near=np.zeros(997,dtype=bool)
        for a,b in row['near_surface_intervals']:near[a:b+1]=True
        high=np.any(error[:,indices]>25,axis=1);common=near&high
        rows.append(dict(source_nominal_link=link,source_near_intervals=row['near_surface_intervals'],source_near_frames=int(near.sum()),
            inspected_direction_indices=indices,overlap_high_error_near_frames=int(common.sum()),overlap_runs=runs(common),
            near_interval_max_same_limb_direction_degrees=float(error[near][:,indices].max()) if near.any() else None,
            source_near_is_conditioned_on_official_consumer_contract=True,target_contact_or_collision_test=False,accepted_support_intervals=None))
    report=dict(stage='same_native_frame_risk_triage_NOT_support_acceptance',source_support_review_sha256=sha(existing),source_motion_sha256=sha(MOTION),
        target_joint_artifact_sha256=sha(result_path),direction_error_threshold_degrees=25,threshold_is_diagnostic_only=True,
        no_world_coordinate_mapping_or_unit_selection=True,no_solver_calls=True,links=rows,
        priority='Right forearm error at frame257 coincides with start of CONDITIONED source wrist-near interval; inspect before root/contact retargeting.',
        accepted_support_intervals=None,accepted_reference=False,training_ready=False)
    with (HERE/'SOURCE_SUPPORT_RISK_TRIAGE.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()