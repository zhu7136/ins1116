import copy
from types import SimpleNamespace as S
import unittest
from audit import HERE,read
from native_contact_observer import NativeContactObserver,registry_from_audit,evidence_digest

class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.registry=registry_from_audit(read(HERE/'AUDIT.json'))
        r=self.registry[0]
        self.paths={i+1:r[k] for i,k in enumerate(['actor0','actor1','collider0','collider1'])}
        self.header=S(actor0=1,actor1=2,collider0=3,collider1=4,contact_data_offset=0,num_contact_data=1,type='CONTACT_DATA')
        self.point=S(position=[0,0,0],normal=[0,0,1],impulse=[0,0,.1],separation=-.002)
    def observer(self,**kw):return NativeContactObserver(self.registry,self.paths.__getitem__,**kw)
    def test_exact_shape_paths_match_without_waiver(self):
        o=self.observer();o.callback([self.header],[self.point]);r=o.report()
        self.assertIsNone(r['error']);self.assertEqual(r['events'][0]['matched_exact_pair'],'left_wrist_roll_joint')
        self.assertEqual(r['events'][0]['contacts'][0]['native_separation_raw'],-.002)
        self.assertFalse(r['physical_penetration_depth_verified']);self.assertFalse(r['actual_pair_filter_cause_verified'])
        self.assertEqual(r['automatic_pair_exclusions'],0)
    def test_reverse_native_pair_order_supported(self):
        h=copy.copy(self.header);h.actor0,h.actor1=2,1;h.collider0,h.collider1=4,3
        o=self.observer();o.callback([h],[self.point]);self.assertEqual(o.report()['events'][0]['matched_exact_pair'],'left_wrist_roll_joint')
    def test_mesh_descendant_not_guessed(self):
        self.paths[4]+='/Mesh'
        o=self.observer();o.callback([self.header],[self.point]);self.assertIsNone(o.report()['events'][0]['matched_exact_pair'])
    def test_wrong_environment_not_matched(self):
        self.paths={k:v.replace('/env_0/','/env_1/') for k,v in self.paths.items()}
        o=self.observer();o.callback([self.header],[self.point]);self.assertIsNone(o.report()['events'][0]['matched_exact_pair'])
    def test_actor_shape_mismatch_not_matched(self):
        h=copy.copy(self.header);h.collider0,h.collider1=4,3
        o=self.observer();o.callback([h],[self.point]);self.assertIsNone(o.report()['events'][0]['matched_exact_pair'])
    def test_zero_event_not_filter_certificate(self):
        r=self.observer().report()
        self.assertFalse(r['subscription_and_time_coverage_verified']);self.assertFalse(r['actual_pair_filter_cause_verified'])
        self.assertTrue(r['zero_contacts_does_not_prove_filtering'])
    def test_missing_shape_fields_fail_closed(self):
        h=copy.copy(self.header);del h.collider0
        o=self.observer();o.callback([h],[self.point]);self.assertIsNotNone(o.report()['error'])
    def test_missing_separation_stays_unknown(self):
        p=copy.copy(self.point);del p.separation
        o=self.observer();o.callback([self.header],[p]);self.assertIsNone(o.report()['events'][0]['contacts'][0]['native_separation_raw'])
    def test_missing_force_vectors_not_fabricated(self):
        o=self.observer();o.callback([self.header],[S()]);self.assertIsNone(o.report()['events'][0]['contacts'][0]['impulse'])
    def test_nan_rejected(self):
        p=copy.copy(self.point);p.separation=float('nan')
        o=self.observer();o.callback([self.header],[p]);self.assertIsNotNone(o.report()['error'])
    def test_buffer_overflow_rejected(self):
        h=copy.copy(self.header);h.contact_data_offset=1
        o=self.observer();o.callback([h],[self.point]);self.assertFalse(o.report()['ingestion_complete_within_budget'])
    def test_point_budget_rejected(self):
        h=copy.copy(self.header);h.num_contact_data=2
        o=self.observer(max_points=1);o.callback([h],[self.point,self.point]);self.assertIsNotNone(o.report()['error'])
    def test_event_budget_rejected(self):
        o=self.observer(max_events=1);o.callback([self.header,self.header],[self.point]);self.assertIsNotNone(o.report()['error'])
    def test_unknown_decoder_id_does_not_escape_callback(self):
        h=copy.copy(self.header);h.actor0=100
        o=self.observer();o.callback([h],[self.point]);self.assertEqual(o.report()['error']['type'],'KeyError')
    def test_no_mutation_of_native_buffers(self):
        h=copy.deepcopy(self.header);p=copy.deepcopy(self.point)
        o=self.observer();o.callback([self.header],[self.point]);self.assertEqual(vars(h),vars(self.header));self.assertEqual(vars(p),vars(self.point))
    def test_report_not_alias_and_digest_stable(self):
        o=self.observer();o.callback([self.header],[self.point]);a=o.report();h=evidence_digest(a)
        a['events'].clear();self.assertEqual(len(o.report()['events']),1);self.assertEqual(h,evidence_digest(o.report()))
    def test_noncanonical_or_wildcard_registry_rejected(self):
        for value in ['/World/*/body','/World/a/../b','/World/a//b','/World/a/']:
            r=copy.deepcopy(self.registry);r[0]['actor0']=value
            with self.assertRaises(ValueError):NativeContactObserver(r,self.paths.__getitem__)
    def test_duplicate_pair_rejected(self):
        r=copy.deepcopy(self.registry);extra=copy.deepcopy(r[0]);extra['key']='other';r.append(extra)
        with self.assertRaises(ValueError):NativeContactObserver(r,self.paths.__getitem__)

if __name__=='__main__':unittest.main()