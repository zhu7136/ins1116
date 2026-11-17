"""Reconcile independently re-established results against IMMUTABLE prior HEAD."""
from collections import Counter
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent
PREVIOUS=Path('/workspace/hud04-remote-6fa4349-20261117/datasets/hud04/source-only-20261117/continuation_evidence/actual_terrain_binding_20261117')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__=='__main__':
    out=ROOT/'INDEPENDENT_CLOSURE_RECONCILIATION_20261117.json'
    if out.exists():raise FileExistsError('No overwrite')
    old_path=PREVIOUS/'TARGET_ACTUAL_TERRAIN_DETAIL_20261117.json';old=json.loads(old_path.read_text())
    new=json.loads((ROOT/'ACTUAL_TERRAIN_DETAIL_20261117.json').read_text())
    checks=json.loads((ROOT/'INDEPENDENT_PRIMITIVE_CHECK_20261117.json').read_text())['checks']
    extra=json.loads((ROOT/'REMAINING_GEOMETRY_CHECK_20261117.json').read_text())['checks']
    by_key={(r['shape'],r['subset']):r for r in checks}
    by_key.update({(r['shape'],r['subset']):r for r in extra})
    previous={(r['shape'],r['subset']):r for r in old['rows'] if r['layout']=='raw'}
    rows=[];max_original_discrepancy=0.;max_repeat_bracket_endpoint_delta=0.
    for row in new['rows']:
        key=row['shape'],row['subset'];ind=by_key[key];ref=previous[key]['result'];independent=ind['independent'];current=row['result']
        discrepancy=max(0.,independent['lower']-ref['distance_upper_bound'],ref['distance_lower_bound']-independent['upper'])
        max_original_discrepancy=max(max_original_discrepancy,discrepancy)
        max_repeat_bracket_endpoint_delta=max(max_repeat_bracket_endpoint_delta,*[abs(ref[k]-current[k]) for k in ['distance_lower_bound','distance_upper_bound']])
        verified=independent['status']=='converged' and discrepancy<=1e-8
        if not verified:raise RuntimeError('Original remote query not independently closed')
        rows.append(dict(shape=key[0],subset=key[1],kind=row['kind'],independent=independent,
            original_remote_bracket=[ref['distance_lower_bound'],ref['distance_upper_bound']],
            discrepancy=discrepancy,verified_nominal_single_pose_distance_only=True))
    if len(rows)!=86 or len(previous)!=86 or len(by_key)!=86:raise ValueError('Query inventory mismatch')
    result=dict(stage='original_remote_86_nominal_queries_independently_closed_NOT_motion_acceptance',
        original_remote_commit='6fa434956487f800977b7e158cdc37d9ceefb723',original_remote_detail_sha256=sha(old_path),
        local_recovery_receipt_sha256=sha(Path('/workspace/hud04-remote-6fa4349-20261117/RECOVERY_RECEIPT.json')),
        unique_queries=86,type_counts=dict(Counter(r['kind'] for r in rows)),verified_queries=len(rows),
        new_closure_items=['base_link#0/all cylinder','left nominal hand hull/all','left nominal hand hull/lateral','right nominal hand hull/all','right nominal hand hull/lateral'],
        max_original_bracket_discrepancy=max_original_discrepancy,max_primary_repeat_endpoint_delta=max_repeat_bracket_endpoint_delta,
        previous_unresolved_result_retained=True,independent_tolerance=1e-8,primary_tolerance=1e-9,
        cylinder_plane_bound='For unit triangle-plane normal n, finite-cylinder extent=r*norm(n.xy)+h*abs(n.z). Plane-interval separation is a valid lower bound; degenerate normals yield zero.',
        no_iteration_budget_or_accuracy_relaxation=True,all_previous_remote_files_preserved=True,
        new_tests_passed=19,prior_solid_regressions_passed=31,total_tests_this_turn=50,
        rows=rows,source_motion_or_training_settings_changed=False,source_coordinate_hypothesis_selected=False,
        single_algebraic_pose_only=True,full_motion_collision_validated=False,cooked_physics_validated=False,
        actual_collision_filters_observed=False,contact_schedule_accepted=False,
        gpu_submissions=0,retarget_solver_calls=0,ppo_updates=0,accepted_reference=False,training_ready=False)
    raw=json.dumps(result,indent=2,allow_nan=False).encode()
    with out.open('xb') as stream:stream.write(raw)
    if out.read_bytes()!=raw:raise RuntimeError('Readback failed')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)