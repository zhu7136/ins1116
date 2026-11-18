"""Bounded observer plugin for an AUTHORIZED, ALREADY EXISTING host loop.

Not an executable scene. No simulator creation/import, stepping, reset, native
cooking, filtering/drive writes or process/task APIs. Host supplies step markers.
"""
import copy
import time
from contact_report_1073 import ContactReport1073
from native_contact_observer import integer


class ExistingRuntimeObservation(ContactReport1073):
    def __init__(self,*args,max_steps=8,wall_seconds=30,clock=time.monotonic,**kwargs):
        if type(max_steps) is not int or not 1<=max_steps<=8:raise ValueError('At most8 host steps')
        if isinstance(wall_seconds,bool) or not isinstance(wall_seconds,(int,float)) or not 0<wall_seconds<=30:
            raise ValueError('At most30 observer wall seconds')
        for key,cap in (('max_events',256),('max_points',4096)):
            value=kwargs.get(key,cap)
            if type(value) is not int or not 1<=value<=cap:raise ValueError('Hard observation budget: '+key)
        super().__init__(*args,**kwargs)
        self.max_steps=max_steps;self.wall_seconds=wall_seconds;self.clock=clock
        self.started=clock();self.steps=[];self.active=None;self.closed=False
        self.host_authorization_receipt=None

    def subscribe_existing(self,simulation_interface,*,authorization=None):
        if not isinstance(authorization,dict) or authorization.get('approved_existing_process') is not True or authorization.get('starts_new_process') is not False:
            raise PermissionError('Existing process authority must be supplied; no new process authorized')
        if not isinstance(authorization.get('approval_reference'),str) or not authorization['approval_reference'].strip():
            raise PermissionError('Explicit host approval reference required')
        self.host_authorization_receipt={k:authorization[k] for k in ('approved_existing_process','starts_new_process','approval_reference')}
        return super().subscribe_existing(simulation_interface)

    def _valid_window(self):
        if self.closed:raise ValueError('Observation already closed')
        if self.clock()-self.started>self.wall_seconds:raise ValueError('Observer deadline exceeded; stop recording, NOT physics')

    def _fail(self,exc):
        if self.error is None:self.error=dict(type=type(exc).__name__,message=str(exc)[:1000])

    def on_host_step_begin(self,step):
        """Host calls before its OWN next step; this method does not step physics."""
        try:
            self._valid_window();step=integer(step)
            if self.error is not None:raise ValueError('Prior observer error')
            if self.active is not None:raise ValueError('Prior host step not ended')
            if len(self.steps)>=self.max_steps:raise ValueError('Host step budget exhausted')
            if self.steps and step!=self.steps[-1]['host_step']+1:raise ValueError('Host marker gap or reversal')
            self.active=dict(host_step=step,begin_elapsed_s=self.clock()-self.started,
                event_begin=len(self.observer.events),callback_begin=self.callback_invocations)
        except Exception as exc:self._fail(exc)

    def callback(self,*args):
        if self.closed or self.error is not None:return
        try:
            self._valid_window()
            if self.active is None:raise ValueError('Callback outside marked host step')
            super().callback(*args)
        except Exception as exc:self._fail(exc)

    def on_host_step_end(self):
        try:
            self._valid_window()
            if self.active is None:raise ValueError('No active host step')
            record=dict(self.active,event_end=len(self.observer.events),
                callback_count=self.callback_invocations-self.active['callback_begin'],
                end_elapsed_s=self.clock()-self.started,
                native_buffer_capacity=None,native_buffer_overflow=None,
                native_buffer_health='UNKNOWN; budget checks are not native overflow telemetry',
                marker_is_host_supplied_not_independently_verified=True)
            record['callback_delivery']='OBSERVED' if record['callback_count'] else 'UNKNOWN_NO_CALLBACK'
            self.steps.append(record);self.active=None
        except Exception as exc:self._fail(exc)

    def finish(self):
        if self.active is not None:self._fail(ValueError('Unfinished host step; incomplete coverage'))
        self.closed=True
        return self.report()

    def report(self):
        result=super().report()
        result.update(schema='hud04.existing_runtime_observation.v1',
            host_step_records=copy.deepcopy(self.steps),unfinished_host_step=copy.deepcopy(self.active),
            observer_closed=self.closed,host_authorization_receipt=copy.deepcopy(self.host_authorization_receipt),
            host_authorization_independently_verified=False,
            declared_max_host_steps=self.max_steps,declared_observer_wall_seconds=self.wall_seconds,
            no_callback_host_steps=[r['host_step'] for r in self.steps if r['callback_count']==0],
            no_callback_is_not_proven_transport_drop=True,
            native_buffer_health_verified=False,host_step_coverage_independently_verified=False,
            callback_buffers_copied_in_callback=True,
            host_owns_subscription_holder_and_cleanup=True,
            no_new_process_creation_api=True,no_simulation_step_api=True,
            representation_evidence='UNRESOLVED',final_filter_evidence='UNRESOLVED',live_cooked_evidence='UNRESOLVED',
            reference_self_collision_acceptance='UNCHANGED_UNACCEPTED',
            reference_terrain_support_dynamics_acceptance='UNCHANGED_UNACCEPTED')
        return result