"""Read-only singleton body-pair contact buffers. Not shape-specific proof.

Source API contract: IsaacSim v5.1.0, rigid_contact_view.py fixed blob
e8d9b2f8a0a1aed3db14fa65d79db1f153598a1f, lines 350-376 and 441-466.
No simulator creation, subscriptions, writes, steps or inference from zero points.
"""
import math
import numbers
import numpy as np
from native_contact_observer import exact_path,integer

SOURCE_BLOB='e8d9b2f8a0a1aed3db14fa65d79db1f153598a1f'


def cpu_copy(value):
    # Standard numpy / torch protocols, not unverified native shape APIs.
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.array(value,copy=True)


def unpack_singleton(buffers,*,channel,step,dt,capacity_limit=4096):
    """Return only populated contacts; unused native storage is NOT measurement.

    Channel paths are EXPECTED paths supplied by caller, not runtime binding.
    Missing API compatibility/readback/coverage/cooked proofs stay explicit false.
    Raises ValueError for ambiguous, saturated or invalid buffers; never truncates.
    """
    if type(channel) is not dict or set(channel)!={'name','expected_sensor_body','expected_filter_body'}:
        raise ValueError('Explicit expected singleton body-pair channel required')
    if not isinstance(channel['name'],str) or not channel['name']:raise ValueError('Channel name required')
    a=exact_path(channel['expected_sensor_body']);b=exact_path(channel['expected_filter_body'])
    if a==b:raise ValueError('Distinct expected body paths required')
    step=integer(step)
    if isinstance(dt,bool) or not isinstance(dt,numbers.Real) or not math.isfinite(dt) or dt<=0:
        raise ValueError('Positive finite physics dt required')
    if type(capacity_limit) is not int or not 1<=capacity_limit<=65536:
        raise ValueError('Bounded capacity required')
    if not isinstance(buffers,(list,tuple)) or len(buffers)!=6:raise ValueError('Exactly six native buffers required')
    # Inspect shape before CPU copying to reject unexpectedly large buffers.
    shapes=[]
    for value in buffers:
        if not hasattr(value,'shape'):raise ValueError('Array/tensor buffers required')
        shapes.append(tuple(value.shape))
    if len(shapes[0])!=2 or shapes[0][1]!=1:raise ValueError('Normal force column required')
    capacity=shapes[0][0]
    if not 1<=capacity<=capacity_limit:raise ValueError('Native buffer capacity outside budget')
    if shapes!=[(capacity,1),(capacity,3),(capacity,3),(capacity,1),(1,1),(1,1)]:
        raise ValueError('Exact singleton contact layout required; not multi-shape identity')
    forces,points,normals,separations,counts,starts=[cpu_copy(x) for x in buffers]
    if counts.dtype.kind not in 'iu' or starts.dtype.kind not in 'iu':raise ValueError('Integer count/index dtype required')
    count=integer(counts[0,0]);start=integer(starts[0,0])
    if start+count>capacity:raise ValueError('Contact buffer range exceeds allocated capacity')
    if count==capacity:raise ValueError('Saturated contact buffer: truncation cannot be excluded')
    used=[x[start:start+count] for x in [forces,points,normals,separations]]
    for x in used:
        if x.dtype.kind not in 'fiu' or not np.isfinite(x).all():raise ValueError('Populated contacts must be finite numeric values')
    # Normal force scalar signs and separation signs are preserved as returned.
    # No force->penetration conversion, no /dt operation applied a second time.
    contacts=[dict(normal_force_native=float(used[0][i,0]),position_native=used[1][i].astype(float).tolist(),
        normal_native=used[2][i].astype(float).tolist(),contact_separation_native=float(used[3][i,0])) for i in range(count)]
    distances=[c['contact_separation_native'] for c in contacts]
    return dict(schema='hud04.singleton_tensor_contact_observation.v1',channel=dict(channel),
        step=step,dt_argument_s=float(dt),source_api_blob=SOURCE_BLOB,
        allocated_capacity=capacity,populated_count=count,populated_start=start,
        contacts=contacts,minimum_reported_contact_separation_native=min(distances) if distances else None,
        negative_reported_separation_count=sum(x<0 for x in distances),
        field_identity_supported_by_pinned_source=True,target_runtime_api_compatibility_verified=False,
        runtime_channel_paths_independently_verified=False,shape_ids_present=False,
        native_units_and_offset_semantics_verified=False,global_signed_distance_query_verified=False,
        maximum_geometric_penetration_depth_verified=False,cooked_geometry_verified=False,
        actual_pair_filter_cause_verified=False,coverage_verified=False,
        zero_contacts_does_not_prove_separation_or_filtering=True,
        automatic_pair_exclusions=0,accepted_reference=False,training_ready=False)


def capture_singleton_view(view,*,channel,step,dt,capacity_limit=4096):
    """ONE documented read of an existing view; no view creation or simulation."""
    # Validate dt before touching the supplied view; rest is checked on returned data.
    if isinstance(dt,bool) or not isinstance(dt,numbers.Real) or not math.isfinite(dt) or dt<=0:
        raise ValueError('Positive finite physics dt required')
    return unpack_singleton(view.get_contact_data(dt=dt),channel=channel,step=step,dt=dt,capacity_limit=capacity_limit)