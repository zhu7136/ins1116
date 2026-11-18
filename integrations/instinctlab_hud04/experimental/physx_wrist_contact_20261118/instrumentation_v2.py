"""Pure-data review for v2 diagnostic instrumentation. Not reference acceptance."""
import math

STEPS=8
ROLES=('original_fallback','explicit_true','explicit_false')
CHANNELS={'box_positive_control'}|{r+'_'+s for r in ROLES for s in ('left','right')}

def vector(x,n):
    if not isinstance(x,list) or len(x)!=n or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in x):
        raise ValueError('Invalid measured vector')
    return x

def norm(x):return math.sqrt(sum(v*v for v in x))

def validate_instrumentation(d,samples):
    if d.get('revision')!=2:raise ValueError('v2 instrumentation required')
    p=d.get('contact_processing',{})
    if p.get('setting')!='/physics/disableContactProcessing' or p.get('after') is not False or d.get('disable_contact_processing_after_reset') is not False:
        raise ValueError('Explicit contact-processing readback required')
    if d.get('box_size')!=[.1,.1,.1] or d.get('box_threshold_zero_requested') is not True or d.get('contact_capacity_per_channel')!=256:
        raise ValueError('Box/contact-buffer contract mismatch')
    trace=d.get('box_state_trace',[]);sensor=d.get('official_sensor_force_vectors',[])
    counts=d.get('native_contact_counts',{})
    if len(trace)!=STEPS or len(sensor)!=STEPS or set(counts)!=CHANNELS or set(samples)!=CHANNELS:
        raise ValueError('Exact complete measured channels required')
    for k in CHANNELS:
        if len(counts[k])!=STEPS or any(isinstance(c,bool) or not isinstance(c,int) or c<0 for c in counts[k]):raise ValueError('Invalid contact counts')
        if len(samples[k])!=STEPS or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in samples[k]):raise ValueError('Invalid force magnitudes')
    margins=[];cross_match=[]
    for i,t in enumerate(trace):
        if t.get('step')!=i or set(t.get('before',{}))!={'A','B'} or set(t.get('after',{}))!={'A','B'}:raise ValueError('Box trace identity mismatch')
        for phase in ('before','after'):
            for side in ('A','B'):
                state=t[phase][side];pose=vector(state['pose_xyzw'],7);vector(state['velocity'],6)
                if abs(norm(pose[3:])-1.)>1e-4:raise ValueError('Non-unit native quaternion')
        a=t['before']['A'];b=t['before']['B']
        for s,x,vx in ((a,25.,.1),(b,25.08,-.1)):
            if max(abs(v-w) for v,w in zip(s['pose_xyzw'],[x,0.,1.,0.,0.,0.,1.]))>5e-5:raise ValueError('Repeated measured pre-step pose mismatch')
            if max(abs(v-w) for v,w in zip(s['velocity'],[vx,0.,0.,0.,0.,0.]))>1e-5:raise ValueError('Repeated measured pre-step velocity mismatch')
        margins.append([.1-abs(a['pose_xyzw'][j]-b['pose_xyzw'][j]) for j in range(3)])
        raw=norm(vector(t['native_force_vector'],3));official=norm(vector(sensor[i],3))
        magnitude=samples['box_positive_control'][i]
        tol=1e-5+1e-4*max(raw,official,magnitude)
        cross_match.append(abs(raw-official)<=tol and abs(raw-magnitude)<=tol)
    overlap=all(min(m)>1e-4 for m in margins)
    unsaturated=all(c<256 for cs in counts.values() for c in cs)
    sensor_positive=any(norm(v)>1e-6 for v in sensor)
    raw_positive=any(v>1e-6 for v in samples['box_positive_control'])
    count_positive=any(c>0 for c in counts['box_positive_control'])
    return dict(repeated_pre_step_overlap_verified=overlap,minimum_pre_step_axis_overlap=min(min(m) for m in margins),
        official_raw_box_force_agreement=all(cross_match),official_box_positive=sensor_positive,
        native_box_positive=raw_positive,box_contact_count_positive=count_positive,contact_buffers_below_capacity=unsaturated,
        positive_control_valid=overlap and all(cross_match) and sensor_positive and raw_positive and count_positive and unsaturated,
        gpu_setting_causal_effect_established=False,all_joint_pair_filters_verified=False,
        accepted_reference=False,training_ready=False)