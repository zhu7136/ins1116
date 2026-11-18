"""Fail-closed measured detached-control report validation; no pair waiver."""
import math
CHANNELS={'detached_left','detached_right','articulated_left','articulated_right','box'}
def make_pair_view(sim_view,sensor_path,target_path):
    if not isinstance(sensor_path,str) or not isinstance(target_path,str) or not sensor_path.startswith('/World/') or not target_path.startswith('/World/') or sensor_path==target_path:raise ValueError('Exact distinct body paths')
    # Native overload: list[str] sensors pairs with list[list[str]] filters.
    # A str sensor pairs with list[str], NEVER this nested filter form.
    return sim_view.create_rigid_contact_view([sensor_path],filter_patterns=[[target_path]],max_contact_data_count=256)
def evaluate(samples,counts,trace,copies):
    if set(samples)!=CHANNELS or set(counts)!=CHANNELS:raise ValueError('Exact five channels')
    for k in CHANNELS:
        if len(samples[k])!=8 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in samples[k]):raise ValueError('Force samples')
        if len(counts[k])!=8 or any(type(v) is not int or v<0 or v>=256 for v in counts[k]):raise ValueError('Contact capacity/counts')
    if len(copies)!=4 or {(r['side'],r['role']) for r in copies}!={(s,r) for s in ('left','right') for r in ('Parent','Child')}:raise ValueError('Exact copied originals')
    for r in copies:
        if r['original']!=r['clone']:raise ValueError('Copied selected collider differs')
        link=r['side']+('_wrist_pitch_link' if r['role']=='Parent' else '_wrist_roll_link')
        suffix='/collisions/official_'+link+('_2/cylinder' if r['role']=='Parent' else '_1/node_STL_BINARY_')
        if r['source']!='/World/envs/env_0/Robot/'+link or r['target']!='/World/Detached_'+r['side']+'/'+r['role'] or r['selected_source']!=r['source']+suffix or r['selected_target']!=r['target']+suffix:raise ValueError('Original/copy exact binding')
    if len(trace)!=8 or [r['step'] for r in trace]!=list(range(8)):raise ValueError('Actual eight steps')
    for row in trace:
        for side,shift in [('left',5.),('right',10.)]:
            for role in ['Parent','Child']:
                r=row['before'][side+'_'+role];s=r['source_xyzw'];d=r['detached_xyzw']
                if len(s)!=7 or len(d)!=7 or any(not math.isfinite(x) for x in s+d):raise ValueError('Pose layout')
                expected=[s[0]+shift,s[1],s[2]+2.,*s[3:]]
                if max(abs(a-b) for a,b in zip(d,expected))>2e-6:raise ValueError('Exact translated source poses not read back')
        a=row['before']['box_A']['pose_xyzw'];b=row['before']['box_B']['pose_xyzw']
        if len(a)!=7 or len(b)!=7 or any(not math.isfinite(x) for x in a+b) or min(.1-abs(a[i]-b[i]) for i in range(3))<.019:raise ValueError('Box overlap not retained')
    positive=max(samples['box'])>1e-6 and max(counts['box'])>0
    supported={s:positive and max(samples['detached_'+s])>1e-6 and max(counts['detached_'+s])>0 and max(samples['articulated_'+s])<=1e-9 for s in ['left','right']}
    return dict(peak_force_norm_native={k:max(v) for k,v in samples.items()},positive_control_valid=positive,
        detached_original_shape_contact_positive={s:max(samples['detached_'+s])>1e-6 and max(counts['detached_'+s])>0 for s in ['left','right']},
        observed_articulation_vs_detached_context_difference=supported,
        exact_cooked_geometry_equivalence_proven=False,articulation_filter_causal_mechanism_proven=False,
        all_joint_pair_filters_verified=False,automatic_pair_exclusions=0,accepted_reference=False,training_ready=False)
def validate_report(d,commit,probe_sha):
    if d.get('stage')!='hud04_detached_wrist_control' or d.get('source_commit')!=commit or d.get('probe_sha256')!=probe_sha:raise ValueError('Identity')
    if d.get('explicit_steps')!=8 or d.get('dt')!=.005 or d.get('body_dof')!=29 or d.get('head_action_dof')!=0:raise ValueError('Scope')
    for k in ['accepted_reference','training_ready','all_joint_pairs_verified','source_motion_alignment_verified','cooked_hull_exact_vertices_verified']:
        if d.get(k) is not False:raise ValueError('No acceptance/extrapolation')
    if d['verdict']!=evaluate(d['samples'],d['counts'],d['trace'],d['copied_shapes']):raise ValueError('Measured verdict mismatch')
    return True