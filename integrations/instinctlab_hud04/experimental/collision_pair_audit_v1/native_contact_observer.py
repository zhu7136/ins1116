"""Read-only event ingestion for an ALREADY RUNNING simulation, not a launcher.

No Isaac/PhysX imports, subscriptions, filter writes or simulation steps here.
The caller supplies the runtime's documented integer-ID -> exact USD-path decoder.
Attribute availability is checked per event; missing fields never mean no contact.
Native `separation` values are retained raw, not promoted to verified penetration.
"""
import copy
import hashlib
import json
import math
import numbers
from itertools import islice
from pathlib import PurePosixPath
from contact_material_links_1073 import copy_contact_material_links


def exact_path(value):
    if not isinstance(value,str) or not value.startswith('/World/') or value.endswith('/') or any(c in value for c in '*?[]{}'):
        raise ValueError('Exact absolute /World path required')
    if str(PurePosixPath(value))!=value or '..' in value.split('/') or '.' in value.split('/'):
        raise ValueError('Noncanonical path')
    return value


def integer(value):
    if isinstance(value,bool) or not isinstance(value,numbers.Integral) or value<0:
        raise ValueError('Nonnegative integer required')
    return int(value)


def finite(value):
    if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(value):
        raise ValueError('Finite numeric contact field required')
    return float(value)


def vector(value):
    value=list(value)
    if len(value)!=3:raise ValueError('Three-vector required')
    return [finite(v) for v in value]


def registry_from_audit(audit):
    """Bind EXACT observed collision prim paths, not inferred mesh descendants."""
    out=[]
    for pair in audit['pairs']:
        observations=pair['runtime_observations']
        final=next(r for r in observations if r['phase']=='post_physics_init.json')
        a,b=final['colliders']
        out.append(dict(key=pair['joint'],actor0=final['body0'][0],actor1=final['body1'][0],
            collider0=a['path'],collider1=b['path']))
    return out


class NativeContactObserver:
    def __init__(self,registry,decode_path,*,max_events=256,max_points=4096,runtime_identity=None):
        if not callable(decode_path):raise ValueError('Explicit runtime path decoder required')
        if type(max_events) is not int or not 1<=max_events<=4096 or type(max_points) is not int or not 1<=max_points<=65536:
            raise ValueError('Bounded observer required')
        if not registry or len(registry)>32:raise ValueError('Bounded nonempty exact registry required')
        self.registry=copy.deepcopy(registry);seen=set();keys=set()
        for row in self.registry:
            key=row['key']
            if not isinstance(key,str) or not key or key in keys:raise ValueError('Unique pair keys required')
            keys.add(key)
            for k in ['actor0','actor1','collider0','collider1']:exact_path(row[k])
            if row['actor0']==row['actor1']:raise ValueError('Distinct original actors required')
            for k in ['0','1']:
                if not row['collider'+k].startswith(row['actor'+k]+'/'):raise ValueError('Collider owner mismatch')
            signature=frozenset([(row['actor0'],row['collider0']),(row['actor1'],row['collider1'])])
            if signature in seen:raise ValueError('Duplicate pair binding')
            seen.add(signature)
        self.decode_path=decode_path;self.max_events=max_events;self.max_points=max_points
        self.runtime_identity=copy.deepcopy(runtime_identity or {})
        self.events=[];self.points=0;self.error=None;self.callbacks_seen=0

    def _capture(self,headers,data):
        headers=list(islice(headers,self.max_events-len(self.events)+1))
        if len(self.events)+len(headers)>self.max_events:raise ValueError('Event budget exceeded; incomplete observation')
        for h in headers:
            paths={}
            for field in ['actor0','actor1','collider0','collider1']:
                if not hasattr(h,field):raise ValueError('Native contact header lacks '+field)
                paths[field]=exact_path(str(self.decode_path(getattr(h,field))))
            if not hasattr(h,'contact_data_offset') or not hasattr(h,'num_contact_data'):
                raise ValueError('Native contact buffer layout unavailable')
            start=integer(h.contact_data_offset);count=integer(h.num_contact_data)
            if start+count>len(data):raise ValueError('Contact buffer slice out of range')
            if self.points+count>self.max_points:raise ValueError('Point budget exceeded; incomplete observation')
            actual={(paths['actor0'],paths['collider0']),(paths['actor1'],paths['collider1'])}
            match=[r['key'] for r in self.registry if actual=={(r['actor0'],r['collider0']),(r['actor1'],r['collider1'])}]
            if len(match)>1:raise ValueError('Ambiguous registry')
            contacts=[]
            for index in range(start,start+count):
                p=data[index]
                record={}
                for field in ['position','normal','impulse']:
                    record[field]=vector(getattr(p,field)) if hasattr(p,field) else None
                record['native_separation_raw']=finite(p.separation) if hasattr(p,'separation') else None
                record['native_separation_semantics_verified']=False
                # Copy in this callback, while native point storage remains valid.
                # Preserve event-side ordering; material identity is NOT a tensor slot.
                record['contact_material_links']=copy_contact_material_links(p,self.decode_path)
                contacts.append(record)
            self.events.append(dict(**paths,matched_exact_pair=match[0] if match else None,
                unmatched_event_is_not_ignored_pair=True,contact_count=count,contacts=contacts,
                native_type=str(h.type) if hasattr(h,'type') else None,
                native_path_binding_only_not_cooked_shape_id_equivalence=True))
            self.points+=count

    def callback(self,headers,data):
        """Retain errors, do not throw into or change the user's simulation loop."""
        if self.error is not None:return
        self.callbacks_seen+=1
        try:self._capture(headers,data)
        except Exception as e:
            self.error=dict(type=type(e).__name__,message=str(e)[:1000])

    def report(self):
        counts={r['key']:sum(e['contact_count'] for e in self.events if e['matched_exact_pair']==r['key']) for r in self.registry}
        return copy.deepcopy(dict(schema='hud04.native_contact_event_observation.v1',
            runtime_identity=self.runtime_identity,registry=self.registry,events=self.events,
            callbacks_seen=self.callbacks_seen,contact_points_seen=self.points,error=self.error,
            ingestion_complete_within_budget=self.error is None,
            subscription_and_time_coverage_verified=False,matched_contact_point_counts=counts,
            zero_contacts_does_not_prove_filtering=True,exact_cooked_geometry_equivalence_verified=False,
            actual_pair_filter_cause_verified=False,signed_distance_query_verified=False,
            physical_penetration_depth_verified=False,automatic_pair_exclusions=0,
            accepted_reference=False,training_ready=False))


def evidence_digest(report):
    return hashlib.sha256(json.dumps(report,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()