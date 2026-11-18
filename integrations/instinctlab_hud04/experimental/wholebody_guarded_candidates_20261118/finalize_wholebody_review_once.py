"""Assemble actual receipts, never alter a motion or mark it accepted."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def main():
    output=ROOT/'FINAL_WHOLEBODY_REVIEW.json'
    if output.exists():raise FileExistsError('No replay/overwrite')
    cases=[];total=0;updates=0
    for number in ['006','007','008']:
        trial=ROOT/('WHOLEBODY_TRIAL_'+number);r=load(trial/'RESULT.json');v=load(trial/'INDEPENDENT_WORLD_CANDIDATE_REVIEW.json');receipt=load(trial/'EXECUTION_RECEIPT.json')
        intent=load(trial/'INTENT.json');path=trial/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz'
        if sha(path)!=r['artifact_sha256'] or v['artifact_sha256']!=r['artifact_sha256']:raise ValueError('Saved candidate identity')
        if receipt['child_exit']!=0 or not receipt['child_reaped'] or receipt['parent_timeout']:raise ValueError('Execution not cleanly bounded')
        if r['actual_lp_calls']!=100 or r['visited_frames']!=997 or r['accepted_reference'] or r['training_ready']:raise ValueError('Full bounded unaccepted candidate contract')
        if v['position_bad_frames'] or v['overspeed_intervals'] or not v['nominal_footbox_geometry_rejection']:raise ValueError('Unexpected independent result')
        with np.load(path,allow_pickle=False) as z:
            if z['accepted_reference'].item() or z['training_ready'].item() or z['framerate'].item()!=90 or z['joint_pos'].shape!=(997,29):raise ValueError('Candidate contract')
            np.testing.assert_array_equal(z['frame_indices'],np.arange(997))
        for p,h in r['inputs_sha256'].items():
            if sha(Path(p))!=h:raise ValueError('Original input/seed changed')
        for name,h in load(trial/'CODE_HASHES.json').items():
            if sha(trial/'CODE_AT_EXECUTION'/name)!=h:raise ValueError('Historical execution code changed')
        rows=[json.loads(s) for s in (trial/'PROGRESS.jsonl').read_text().splitlines()]
        if len(rows)!=100 or sum(x['accepted'] for x in rows)!=r['accepted_updates'] or any(x['lp_status']!=0 for x in rows):raise ValueError('Actual progress count')
        if number=='008' and any(not x['backtracking'][-1]['per_pose_guard_passed'] or not x['backtracking'][-1]['foot_guard_passed'] for x in rows if x['accepted']):raise ValueError('Final per-pose guard log')
        cases.append(dict(trial=trial.name,artifact_sha256=r['artifact_sha256'],seed_sha256=intent['seed_sha256'],
            real_data_lp_calls=r['actual_lp_calls'],accepted_local_steps_NOT_reference_acceptance=r['accepted_updates'],
            visited_frames=r['visited_frames'],algorithm_seconds=r['elapsed_seconds'],algorithm_cap_seconds=intent['algorithm_seconds'],parent_cap_seconds=intent['parent_seconds'],parent_receipt=receipt,
            foot_interior_hit_frames=v['foot_box_surface_strict_interior_hit_frames'],position_bad_frames=v['position_bad_frames'],overspeed_intervals=v['overspeed_intervals'],
            max_quaternion_norm_error=v['quaternion_norm_max_error'],independent_metric_max_disagreement=v['independent_metric_max_disagreement'],
            root_speed_raw_per_s=v['root_linear_speed_raw_per_s'],root_acceleration_raw_per_s2=v['root_linear_acceleration_raw_per_s2'],
            left_right_nominal_sole_position_max_raw=[r['final']['position_max_raw'][i] for i in [3,6]],
            offline_wandb_run_id=r['offline_wandb_run_id'],online_verified=False,nominal_footbox_geometry_rejection=True,
            accepted_reference=False,training_ready=False,termination=r['termination']))
        total+=r['actual_lp_calls'];updates+=r['accepted_updates']
    for name,count in [('PER_POSE_FINAL_PREEXECUTION_TESTS.log',24),('REMOTE_ADVANCE_28_TESTS.log',28)]:
        text=(ROOT/name).read_text()
        if 'Ran '+str(count)+' tests' not in text or not text.rstrip().endswith('OK'):raise ValueError('Software tests')
    cross=load(ROOT/'FINAL_REMOTE_GUARD_WITH_TRIAL008_RECONCILIATION.json');strict=cross['stricter_remote_per_pose_guard_vs_seed005']['trial008']
    if any(r['reject'] or r['new_hit_pose_indices'] or r['worsened_pose_indices'] for r in strict.values()):raise ValueError('Fresh independent008 perpose comparison failed')
    audits={n:load(ROOT/('WHOLEBODY_TRIAL_'+n)/'ALL997_WHOLEBODY_AUDIT.json') for n in ['007','008']}
    review=dict(stage='300_real_data_LP_calls_all43_refresh_and_strict_per_pose_guard_BUT_NO_ACCEPTED_REFERENCE',
        publication_expected_parent='86f05dffa8696465147d7e3ebf99aff568a84c26',previous_feature_code_base='e50115ec3fdabc3583224ea060e3054d61631939',
        previous_world_series_real_data_lp_calls=347,new_stage_real_data_lp_calls=total,new_stage_local_updates_not_reference_acceptance=updates,
        trials=cases,all_three_start_same005_seed_not_chained=True,
        new_software_tests=24,concurrent_remote_tests_recovered_and_passed=28,historical_full_regression_rerun=False,
        remote_advance_review=load(ROOT/'REMOTE_ADVANCE_REVIEW_86f05/REMOTE_READBACK.json'),
        strict_final_trial008_vs_seed005={side:dict(reject=r['reject'],new_hit_frames=len(r['new_hit_pose_indices']),worse_per_pose_scores=len(r['worsened_pose_indices'])) for side,r in strict.items()},
        all997_nominal_geometry_audits=audits,all_frames_all_pairs_independent_narrowphase_complete=False,
        projection_values_are_not_signed_depth=True,software_tests_not_physics_acceptance=True,
        original_input_hashes_preserved=True,head_fixed_zero=True,body_actions=29,native_frames=997,fps=90,
        joint_angle_clipping=False,resampling=False,accepted_manifests_created=0,source_motion_files_uploaded=0,
        new_derived_unaccepted_motion_files=3,production_startup_gate_changed=False,existing_remote_guard_changed=False,
        only_offline_wandb_records=True,online_wandb_access_retested=False,live_gpu_status_checked_this_stage=False,
        new_gpu_submissions=0,ppo_updates=0,all_launched_direct_children_reaped=True,
        remaining_gates=['148left/84right nominal footbox strict terrain intersections in008',
            'Full-body projection score is soft, not a hard collision-free certificate; full independent narrowphase/runtime cooked pair filters remain incomplete',
            'Per-pose foot SAT nonregression does not guarantee no swept collision, solid containment, support or convergence',
            'Root/joint temporal and dynamic feasibility not accepted; root acceleration mean worsens vs005',
            'Auditable source exporter/root/quaternion/physical-unit/STL/world-alignment provenance still conditional',
            'Hand/foot/knee material-frame calibration/support schedule not accepted; zero-model offsets remain hypotheses',
            'Actual HUD04 terrain perception task/dynamic replay/reset/depth/cleanexit and eventual bounded learning-chain smoke still blocked'],
        current_trial_budgets_exhausted=True,automatic_retries=0,unlimited_budget_authorized=False,long_PPO_authorized=False,
        do_not_replay_launchers=True,accepted_reference=False,training_ready=False)
    with output.open('x') as f:json.dump(review,f,indent=2,allow_nan=False)
    if load(output)!=review:raise ValueError('Review local readback')
    print(json.dumps(dict(new_lp_calls=total,local_updates=updates,trials=len(cases),strict008=strict,accepted_reference=False,training_ready=False),indent=2))
if __name__=='__main__':main()