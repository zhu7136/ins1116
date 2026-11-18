import copy
from types import SimpleNamespace as S
import unittest
from audit import HERE,read
from contact_report_1073 import ContactReport1073,NON_INSTANCED,UINT64_MAX


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.registry=read(HERE/'AUDIT.json')
        from native_contact_observer import registry_from_audit
        self.registry=registry_from_audit(self.registry)
        self.paths={i+1:self.registry[0][k] for i,k in enumerate(['actor0','actor1','collider0','collider1'])}
        self.schema=S(intToSdfPath=self.paths.__getitem__)
        self.header=S(actor0=1,actor1=2,collider0=3,collider1=4,stage_id=42,
            proto_index0=NON_INSTANCED,proto_index1=NON_INSTANCED,
            contact_data_offset=0,num_contact_data=1,type='CONTACT_FOUND')
        self.point=S(position=[0,0,0],normal=[0,0,1],impulse=[0,0,.1],separation=-.002)
    def binding(self,**kw):return ContactReport1073(self.registry,self.schema,expected_stage_id=42,**kw)
    def capture(self,binding=None,header=None):
        b=binding or self.binding();b.callback([header or self.header],[self.point]);return b.report()
    def test_exact_original_noninstanced_stage_header(self):
        r=self.capture();self.assertTrue(r['ingestion_complete_within_budget'])
        self.assertEqual(r['events'][0]['matched_exact_pair'],'left_wrist_roll_joint')
        self.assertEqual(r['header_identity_records'][0]['native_collider1_id'],4)
        self.assertEqual(r['events'][0]['contacts'][0]['native_separation_raw'],-.002)
    def test_same_paths_different_stage_rejected(self):
        h=copy.copy(self.header);h.stage_id=43;r=self.capture(header=h)
        self.assertFalse(r['ingestion_complete_within_budget']);self.assertEqual(r['events'],[])
    def test_instancing_not_silently_ignored(self):
        for f in ('proto_index0','proto_index1'):
            for v in (0,1,NON_INSTANCED+1,-1,True):
                with self.subTest(field=f,value=v):
                    h=copy.copy(self.header);setattr(h,f,v)
                    self.assertFalse(self.capture(header=h)['ingestion_complete_within_budget'])
    def test_missing_native_identity_rejected(self):
        for f in ('stage_id','proto_index0','proto_index1','collider0'):
            with self.subTest(field=f):
                h=copy.copy(self.header);delattr(h,f)
                self.assertFalse(self.capture(header=h)['ingestion_complete_within_budget'])
    def test_invalid_uint64_never_reaches_decoder(self):
        for v in (-1,True,1.0,UINT64_MAX+1):
            with self.subTest(value=v):
                h=copy.copy(self.header);h.actor0=v
                self.assertFalse(self.capture(header=h)['ingestion_complete_within_budget'])
    def test_three_buffer_callback_retained_as_error(self):
        b=self.binding();b.callback([self.header],[self.point],[])
        self.assertFalse(b.report()['ingestion_complete_within_budget']);self.assertEqual(b.report()['events'],[])
    def test_unknown_decoder_id_contained(self):
        h=copy.copy(self.header);h.collider0=99;r=self.capture(header=h)
        self.assertEqual(r['error']['type'],'KeyError');self.assertFalse(r['ingestion_complete_within_budget'])
    def test_no_contact_not_filter_or_coverage_proof(self):
        b=self.binding();b.callback([],[]);r=b.report()
        self.assertTrue(r['ingestion_complete_within_budget']);self.assertEqual(r['header_identity_records'],[])
        for k in ('actual_pair_filter_cause_verified','subscription_and_time_coverage_verified','actual_subscription_verified'):
            self.assertFalse(r[k])
    def test_transient_native_buffers_copied(self):
        b=self.binding();b.callback([self.header],[self.point]);self.point.position[0]=100;self.point.separation=1
        r=b.report();self.assertEqual(r['events'][0]['contacts'][0]['position'],[0.,0.,0.])
        self.assertEqual(r['events'][0]['contacts'][0]['native_separation_raw'],-.002)
    def test_fail_closed_latches(self):
        b=self.binding();h=copy.copy(self.header);h.stage_id=99
        b.callback([h],[self.point]);b.callback([self.header],[self.point])
        self.assertEqual(b.report()['events'],[]);self.assertEqual(b.report()['callback_invocations'],1)
    def test_total_event_budget_bounded(self):
        b=self.binding(max_events=1);b.callback([self.header],[self.point]);b.callback([self.header],[self.point]);r=b.report()
        self.assertEqual(len(r['events']),1);self.assertFalse(r['ingestion_complete_within_budget'])
    def test_stage_id_and_schema_are_explicit(self):
        for value in (0,-1,True,1.,UINT64_MAX+1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):ContactReport1073(self.registry,self.schema,expected_stage_id=value)
        with self.assertRaises(ValueError):ContactReport1073(self.registry,S(),expected_stage_id=42)
    def test_only_explicit_documented_subscription_no_simulation_setup(self):
        calls=[];holder=object();b=self.binding()
        interface=S(subscribe_contact_report_events=lambda cb:(calls.append(cb),holder)[1])
        self.assertEqual(calls,[])
        self.assertIs(b.subscribe_existing(interface),holder);self.assertEqual(len(calls),1)
        calls[0]([self.header],[self.point]);self.assertEqual(len(b.report()['events']),1)
        self.assertFalse(b.report()['actual_subscription_verified']) # Fake, not runtime evidence.
        with self.assertRaises(RuntimeError):b.subscribe_existing(interface)
        self.assertEqual(len(calls),1)
    def test_newer_api_not_guessed_as_fallback(self):
        b=self.binding();calls=[]
        interface=S(subscribe_physics_contact_report_events=lambda cb:calls.append(cb))
        with self.assertRaises(ValueError):b.subscribe_existing(interface)
        self.assertEqual(calls,[])
    def test_subscription_failure_not_automatically_retried(self):
        calls=[]
        def subscribe(cb):calls.append(cb);raise RuntimeError('synthetic native failure')
        b=self.binding();interface=S(subscribe_contact_report_events=subscribe)
        with self.assertRaises(RuntimeError):b.subscribe_existing(interface)
        with self.assertRaises(RuntimeError):b.subscribe_existing(interface)
        self.assertEqual(len(calls),1)
    def test_report_no_alias_and_no_false_physics_certificates(self):
        b=self.binding();r=self.capture(binding=b);r['header_identity_records'][0]['stage_id']=100
        self.assertEqual(b.report()['header_identity_records'][0]['stage_id'],42)
        for k in ('accepted_reference','training_ready','actual_target_build_verified',
                  'physical_penetration_depth_verified','exact_cooked_geometry_equivalence_verified',
                  'live_cooked_shape_equivalence_verified','cooked_mesh_request_executed'):
            self.assertFalse(b.report()[k])

if __name__=='__main__':unittest.main()