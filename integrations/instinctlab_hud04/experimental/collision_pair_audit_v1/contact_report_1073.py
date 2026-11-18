"""107.3 documented contact-report adapter; no simulator imports or writes.

Caller provides an EXISTING simulation interface, pxr.PhysicsSchemaTools and
an independently obtained expected stage ID. No automatic subscription occurs.
Only non-instanced exact collider prim paths are supported. Source evidence is
not proof that the target image implements the documented contract.
"""
from itertools import islice
from native_contact_observer import NativeContactObserver,integer

UINT64_MAX=(1<<64)-1
NON_INSTANCED=(1<<32)-1
DOC_URL='https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html'


def uint64(value):
    value=integer(value)
    if value>UINT64_MAX:raise ValueError('Native uint64 ID overflow')
    return value


class ContactReport1073:
    def __init__(self,registry,schema_tools,*,expected_stage_id,max_events=256,max_points=4096):
        self.expected_stage_id=uint64(expected_stage_id)
        if self.expected_stage_id==0:raise ValueError('Nonzero independently obtained stage ID required')
        decoder=getattr(schema_tools,'intToSdfPath',None)
        if not callable(decoder):raise ValueError('Documented PhysicsSchemaTools.intToSdfPath required')
        self.observer=NativeContactObserver(registry,lambda value:decoder(uint64(value)),
            max_events=max_events,max_points=max_points,
            runtime_identity={'documented_contract':'omni.physx 107.3','actual_target_build_verified':False})
        self.error=None;self.callback_invocations=0;self.header_identity_records=[]
        self._subscription_attempted=False

    def subscribe_existing(self,simulation_interface):
        """Explicit opt-in ONLY. Return holder; caller must retain its lifetime.

        Does not enable contact reporting, change threshold/filter/scene state,
        start/step physics, or establish temporal coverage. A failed attempt is
        not retried implicitly, since native callback installation may be partial.
        """
        if self._subscription_attempted:raise RuntimeError('Subscription already attempted; no implicit retry')
        if self.error is not None:raise RuntimeError('Invalid observer cannot be subscribed')
        subscribe=getattr(simulation_interface,'subscribe_contact_report_events',None)
        if not callable(subscribe):raise ValueError('107.3 two-buffer subscription method unavailable')
        self._subscription_attempted=True
        return subscribe(self.callback)

    def callback(self,*args):
        """Copy populated data during native one-step buffer lifetime; never throw."""
        if self.error is not None or self.observer.error is not None:return
        self.callback_invocations+=1
        try:
            if len(args)!=2:raise ValueError('107.3 contact callback requires two buffers, not full/friction callback')
            headers,data=args
            remaining=self.observer.max_events-len(self.observer.events)
            headers=list(islice(headers,remaining+1))
            if len(headers)>remaining:raise ValueError('Event budget exceeded; observation incomplete')
            identities=[]
            for header in headers:
                if uint64(header.stage_id)!=self.expected_stage_id:raise ValueError('Native contact stage mismatch')
                for field in ('proto_index0','proto_index1'):
                    if integer(getattr(header,field))!=NON_INSTANCED:
                        raise ValueError('Point-instanced or invalid collider identity not supported')
                for field in ('actor0','actor1','collider0','collider1'):uint64(getattr(header,field))
                identities.append(dict(stage_id=self.expected_stage_id,
                    proto_index0=NON_INSTANCED,proto_index1=NON_INSTANCED,
                    native_actor0_id=int(header.actor0),native_actor1_id=int(header.actor1),
                    native_collider0_id=int(header.collider0),native_collider1_id=int(header.collider1)))
            self.observer.callback(headers,data)
            if self.observer.error is None:self.header_identity_records.extend(identities)
        except Exception as exc:
            self.error=dict(type=type(exc).__name__,message=str(exc)[:1000])

    def report(self):
        result=self.observer.report()
        result.update(schema='hud04.contact_report_1073.v1',documented_api_url=DOC_URL,
            expected_stage_id=self.expected_stage_id,callback_invocations=self.callback_invocations,
            header_identity_records=[dict(r) for r in self.header_identity_records],
            binding_error=dict(self.error) if self.error else None,
            ingestion_complete_within_budget=self.error is None and self.observer.error is None,
            documented_callback_and_header_contract_supported=True,
            actual_target_build_verified=False,actual_subscription_verified=False,
            existing_report_threshold_and_enabled_state_verified=False,
            subscription_holder_lifetime_verified=False,
            point_instanced_colliders_supported=False,
            cooked_mesh_request_executed=False,live_cooked_shape_equivalence_verified=False)
        return result