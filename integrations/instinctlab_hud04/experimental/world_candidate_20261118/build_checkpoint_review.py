"""Build/readback an honest checkpoint without training acceptance."""
import json,hashlib,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
TRIALS=['WORLD_TRIAL_001','WORLD_TRIAL_002_BLOCK31','WORLD_TRIAL_003_FOOT_TEMPORAL','WORLD_TRIAL_004_FOOT_TEMPORAL_BLOCK11','WORLD_TRIAL_005_FOOT_PRIORITY']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    output=ROOT/'FINAL_CANDIDATE_REVIEW.json'
    if output.exists():raise FileExistsError('Review already exists')
    trials=[]
    for name in TRIALS:
        d=ROOT/name;receipt=json.loads((d/'EXECUTION_RECEIPT.json').read_text())
        history=[json.loads(x) for x in (d/'PROGRESS.jsonl').read_text().splitlines()]
        for n,h in json.loads((d/'CODE_HASHES.json').read_text()).items():
            if sha(d/'CODE_AT_EXECUTION'/n)!=h:raise ValueError('Execution source snapshot changed')
        row=dict(trial=name,**receipt,actual_lp_calls=len(history),accepted_updates=sum(bool(r['accepted']) for r in history))
        if not receipt['child_reaped'] or receipt['parent_timeout']:raise ValueError('Child not cleanly reaped')
        if receipt['artifact_exists']:
            report=json.loads((d/'WORLD_CANDIDATE_RESULT.json').read_text());check=json.loads((d/'INDEPENDENT_WORLD_CANDIDATE_REVIEW.json').read_text())
            if sha(d/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz')!=report['artifact_sha256'] or check['artifact_sha256']!=report['artifact_sha256']:raise ValueError('Output changed')
            if check['position_bad_frames'] or check['overspeed_intervals'] or check['accepted_reference'] or check['training_ready']:raise ValueError('Incorrect candidate scope')
            for p,h in report['inputs_sha256'].items():
                if sha(p)!=h:raise ValueError('Original/pinned input changed')
            witnesses=json.loads((d/'FOOT_TERRAIN_INTERIOR_WITNESSES.json').read_text())
            row.update(artifact_sha256=report['artifact_sha256'],position_bad_frames=0,overspeed_intervals=0,
                root_speed=check['root_linear_speed_raw_per_s'],root_acceleration=check['root_linear_acceleration_raw_per_s2'],
                foot_interior_hit_frames=check['foot_box_surface_strict_interior_hit_frames'],
                foot_witnesses_with_lateral_normal_abs_z_lt_point5=sum(abs(r['triangle_normal'][2])<.5 for r in witnesses),
                sole_position_max_raw=[report['final']['position_max_raw'][i] for i in [3,6]],
                fixed_budget_is_not_convergence=True,termination=report['termination'])
        trials.append(row)
    full=json.loads((ROOT/TRIALS[-1]/'FULL_SHAPE_TRIAGE.json').read_text())
    tests=(ROOT/'FOOT_PRIORITY_PREEXECUTION_TESTS.log').read_text()
    if 'Ran 20 tests' not in tests or not tests.rstrip().endswith('OK'):raise ValueError('Tests not passed')
    review=dict(stage='Complete_world_candidates_generated_but_REJECTED_not_accepted_reference',
        original_base='70393c676c85888db0bdd92f8ab7e64fe83fa88d',parallel_advance=json.loads((ROOT/'PARALLEL_REMOTE_ADVANCE_READBACK.json').read_text()),
        trials=trials,total_real_data_lp_calls=sum(r['actual_lp_calls'] for r in trials),total_accepted_local_updates=sum(r['accepted_updates'] for r in trials),
        final_cpu_tests=20,tests_are_software_not_physical_acceptance=True,
        geometry_scope=dict(native_frames=997,all_shapes=43,all_pairs=903,all_frame_broadphase_complete=True,
            representative_gjk_counts=full['query_status_counts'],all_frame_narrowphase_complete=False,no_adjacency_auto_exclusion=True),
        latest_candidate='WORLD_TRIAL_005_FOOT_PRIORITY/STEP1_WORLD_CANDIDATE_UNACCEPTED.npz',
        important_regressions=['trial005_vs004 worstrootacceleration568.89->698.96 raw/s2',
            'trial005_vs004 maxleft/rightsolepositionerror .372/.381 ->.514/.599raw units',
            'meanerror and hitcount improvement NOT full acceptance; 167L/94R nominal foot interior hits remain'],
        remaining_gates=['Finite triangle contacts on sides/edges and unseen shapes must be included in nonlinear merit acceptance, not just frozen plane cuts',
            'All-shape terrain + full native temporal/selfcollision constraints, with actual runtime filter evidence',
            'Source-target material/hand-foot-knee calibration and auditable support schedule, not model-zero anatomical assumption',
            'Exporter/model/root/quaternion/physicalunits/alignment evidence remains conditional; no coordinate choice by smaller gap',
            'Root/joint temporal/dynamic feasibility, original complete env/reset/depth18x32/cleanexit gates remain unpassed'],
        original_input_hashes_preserved=True,head_fixed_zero=True,body_actions=29,frames=997,fps=90,
        no_joint_angle_clipping=True,no_resampling=True,source_raw_uploaded_this_stage=False,
        no_accepted_manifest_generated=True,do_not_use_candidates_for_training=True,
        offline_wandb_only=True,online_wandb_access_retested=False,live_gpu_status_checked=False,gpu_submissions=0,ppo_updates=0,
        background_processes_left=False,accepted_reference=False,training_ready=False)
    raw=json.dumps(review,indent=2,allow_nan=False)
    with output.open('x') as f:f.write(raw)
    if json.loads(output.read_text())!=review:raise ValueError('Local readback')
    print(json.dumps(dict(total_real_data_lp_calls=review['total_real_data_lp_calls'],total_accepted_local_updates=review['total_accepted_local_updates'],final_cpu_tests=20,accepted_reference=False),indent=2))
if __name__=='__main__':main()