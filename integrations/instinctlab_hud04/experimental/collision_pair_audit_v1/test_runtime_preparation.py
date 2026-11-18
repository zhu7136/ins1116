from types import SimpleNamespace as S
import unittest
import test_contact_report_1073 as fixtures
from prepare_runtime_observation import preparation,existing_stage_snapshot,prim_metadata,scalar_or_small_array
from observe_existing_runtime import ExistingRuntimeObservation


class PreparationTests(unittest.TestCase):
    def test_historical_pins_do_not_become_runtime_identity(self):
        p=preparation();self.assertEqual(p['historical_image']['image_version'],'V000446')
        self.assertIsNone(p['historical_image']['immutable_image_digest'])
        self.assertIsNone(p['original_generated_usd_bytes_sha256'])
        self.assertFalse(p['can_submit_gpu']);self.assertFalse(p['existing_authorized_runtime_verified'])
        self.assertEqual(len(p['exact_expected_pairs']),2)
    def test_frozen_identity_mismatch_fails(self):
        from unittest.mock import patch
        with patch('prepare_runtime_observation.digest',return_value='bad'):
            with self.assertRaises(ValueError):preparation()
    def test_settings_missing_stay_unknown_not_false(self):
        r=existing_stage_snapshot(S(GetPrimAtPath=lambda p:None),S(get=lambda k:None),S(),paths=['/World/scene'])
        for v in r['settings'].values():self.assertEqual(v['status'],'UNKNOWN_OR_ERROR');self.assertIsNone(v['value'])
        self.assertEqual(r['prims']['/World/scene']['status'],'UNKNOWN_OR_ERROR')
    def test_setting_false_is_observed_but_not_solver_proof(self):
        r=existing_stage_snapshot(S(GetPrimAtPath=lambda p:None),S(get=lambda k:False),
            S(SETTING_COLLISION_APPROXIMATE_CYLINDERS='/example'),paths=['/World/scene'])
        self.assertIs(r['settings']['SETTING_COLLISION_APPROXIMATE_CYLINDERS']['value']['value'],False)
        self.assertFalse(r['final_solver_filter_verified']);self.assertFalse(r['cooked_equivalence_verified'])
    def test_metadata_keeps_authored_default_and_threshold(self):
        a=S(GetName=lambda:'physxContactReport:threshold',Get=lambda:0.,HasAuthoredValueOpinion=lambda:False,GetNumTimeSamples=lambda:0)
        p=S(IsValid=lambda:True,GetAttributes=lambda:[a],GetRelationships=lambda:[],GetTypeName=lambda:'Xform',IsInstanceProxy=lambda:True)
        r=prim_metadata(p);v=r['attributes']['physxContactReport:threshold']['value']
        self.assertEqual(v['value'],0.);self.assertFalse(v['authored']);self.assertFalse(r['final_solver_filter_verified'])
    def test_large_authored_mesh_arrays_not_read_as_cooked_data(self):
        def forbidden():raise AssertionError('Must not read points')
        p=S(IsValid=lambda:True,GetAttributes=lambda:[S(GetName=lambda:'points',Get=forbidden)],GetRelationships=lambda:[],GetTypeName=lambda:'Mesh',IsInstanceProxy=lambda:False)
        self.assertEqual(prim_metadata(p)['attributes']['points']['status'],'NOT_READ_LARGE_OR_NONCOOKED_ARRAY')
    def test_metadata_bounds_and_nonfinite(self):
        for v in (float('nan'),list(range(65)),[[[[[[0]]]]]],[[0]*64 for _ in range(64)]):
            with self.subTest(value_type=type(v).__name__):
                with self.assertRaises(ValueError):scalar_or_small_array(v)
    def test_exact_path_budget(self):
        for paths in ([],['/World/*'],['/World/a']*2,['/World/a'+str(i) for i in range(33)]):
            with self.assertRaises(ValueError):existing_stage_snapshot(None,None,None,paths=paths)


class HostObservationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.BindingTests();self.fixture.setUp();self.now=[0.]
        self.b=ExistingRuntimeObservation(self.fixture.registry,self.fixture.schema,expected_stage_id=42,clock=lambda:self.now[0])
    def event(self):self.b.callback([self.fixture.header],[self.fixture.point])
    def test_complete_identity_separation_and_step_mapping(self):
        self.b.on_host_step_begin(10);self.event();self.b.on_host_step_end();r=self.b.finish()
        self.assertEqual(r['host_step_records'][0]['event_begin'],0);self.assertEqual(r['host_step_records'][0]['event_end'],1)
        self.assertEqual(r['header_identity_records'][0]['proto_index0'],0xFFFFFFFF)
        self.assertEqual(r['events'][0]['native_type'],'CONTACT_FOUND')
        self.assertEqual(r['events'][0]['contacts'][0]['native_separation_raw'],-.002)
        self.assertIsNone(r['host_step_records'][0]['native_buffer_overflow'])
        self.assertFalse(r['native_buffer_health_verified']);self.assertFalse(r['training_ready'])
    def test_no_callback_unknown_not_zero_penetration(self):
        self.b.on_host_step_begin(0);self.b.on_host_step_end();r=self.b.finish()
        self.assertEqual(r['no_callback_host_steps'],[0]);self.assertFalse(r['physical_penetration_depth_verified'])
        self.assertEqual(r['final_filter_evidence'],'UNRESOLVED')
    def test_marker_gap_is_error(self):
        self.b.on_host_step_begin(0);self.b.on_host_step_end();self.b.on_host_step_begin(2)
        self.assertIsNotNone(self.b.report()['binding_error'])
    def test_callback_outside_step_never_escapes(self):
        self.event();self.assertEqual(self.b.report()['events'],[]);self.assertIsNotNone(self.b.report()['binding_error'])
    def test_partial_host_step_not_coverage_certificate(self):
        self.b.on_host_step_begin(0);r=self.b.finish()
        self.assertIsNotNone(r['binding_error']);self.assertFalse(r['subscription_and_time_coverage_verified'])
    def test_deadline_stops_recording_not_physics(self):
        self.b.on_host_step_begin(0);self.now[0]=31.;self.event()
        self.assertIsNotNone(self.b.report()['binding_error']);self.assertEqual(self.b.report()['events'],[])
    def test_max_step_budget(self):
        for step in range(8):self.b.on_host_step_begin(step);self.b.on_host_step_end()
        self.b.on_host_step_begin(8);self.assertIsNotNone(self.b.report()['binding_error'])
        self.assertEqual(len(self.b.report()['host_step_records']),8)
    def test_no_implicit_subscription_or_permission(self):
        calls=[];interface=S(subscribe_contact_report_events=lambda cb:calls.append(cb))
        for auth in (None,{},dict(approved_existing_process=True,starts_new_process=True,approval_reference='x')):
            with self.assertRaises(PermissionError):self.b.subscribe_existing(interface,authorization=auth)
        self.assertEqual(calls,[])
    def test_host_authorization_is_receipt_not_independent_verification(self):
        calls=[];holder=object();interface=S(subscribe_contact_report_events=lambda cb:(calls.append(cb),holder)[1])
        self.assertIs(self.b.subscribe_existing(interface,authorization=dict(approved_existing_process=True,starts_new_process=False,approval_reference='SYNTHETIC_TEST')),holder)
        self.assertEqual(len(calls),1);self.assertFalse(self.b.report()['host_authorization_independently_verified'])
        self.assertFalse(self.b.report()['actual_subscription_verified'])
    def test_finish_does_not_retain_or_mutate_late_buffers(self):
        self.b.finish();self.event();self.assertEqual(self.b.report()['events'],[]);self.assertIsNone(self.b.report()['binding_error'])
    def test_hard_limit_cannot_be_silently_extended(self):
        for kw in (dict(max_steps=9),dict(wall_seconds=31),dict(wall_seconds=float('nan')),dict(max_events=257),dict(max_points=4097),dict(max_events=True)):
            with self.assertRaises(ValueError):ExistingRuntimeObservation(self.fixture.registry,self.fixture.schema,expected_stage_id=42,**kw)

if __name__=='__main__':unittest.main()