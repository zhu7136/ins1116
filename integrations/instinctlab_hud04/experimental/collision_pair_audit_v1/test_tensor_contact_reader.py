import copy
import unittest
import numpy as np
from tensor_contact_reader import unpack_singleton,capture_singleton_view


class TensorContactTests(unittest.TestCase):
    def setUp(self):
        self.channel=dict(name='original_left_body_pair',expected_sensor_body='/World/envs/env_0/Robot/left_wrist_pitch_link',expected_filter_body='/World/envs/env_0/Robot/left_wrist_roll_link')
        self.data=[np.full((4,1),np.nan),np.full((4,3),np.nan),np.full((4,3),np.nan),np.full((4,1),np.nan),np.array([[2]],dtype=np.int32),np.array([[1]],dtype=np.int32)]
        for a in self.data[:4]:a[1:3]=0.
        self.data[0][1:3,0]=[-.1,.2];self.data[3][1:3,0]=[-.003,.004]
    def read(self,data=None,**kw):
        return unpack_singleton(self.data if data is None else data,channel=self.channel,step=3,dt=.005,**kw)
    def test_signed_contact_field_and_no_second_dt_division(self):
        r=self.read();self.assertEqual(r['populated_count'],2)
        self.assertEqual(r['minimum_reported_contact_separation_native'],-.003)
        self.assertEqual(r['negative_reported_separation_count'],1)
        self.assertEqual(r['contacts'][0]['normal_force_native'],-.1)
        self.assertTrue(r['field_identity_supported_by_pinned_source'])
    def test_no_false_cooked_shape_or_filter_claim(self):
        r=self.read()
        for k in ['shape_ids_present','cooked_geometry_verified','actual_pair_filter_cause_verified','coverage_verified','training_ready','accepted_reference','maximum_geometric_penetration_depth_verified','target_runtime_api_compatibility_verified','runtime_channel_paths_independently_verified']:
            self.assertIs(r[k],False,k)
        self.assertEqual(r['automatic_pair_exclusions'],0)
    def test_unused_native_nan_storage_is_not_contact(self):
        r=self.read();self.assertEqual(len(r['contacts']),2)
    def test_populated_nonfinite_rejected(self):
        self.data[1][1,0]=np.inf
        with self.assertRaises(ValueError):self.read()
    def test_empty_contacts_not_zero_distance(self):
        self.data[4][0,0]=0
        r=self.read();self.assertIsNone(r['minimum_reported_contact_separation_native']);self.assertEqual(r['contacts'],[])
        self.assertTrue(r['zero_contacts_does_not_prove_separation_or_filtering'])
    def test_multi_body_buffer_not_accepted_as_shape_identity(self):
        self.data[4]=np.array([[1],[1]])
        with self.assertRaises(ValueError):self.read()
    def test_negative_count_rejected(self):
        self.data[4][0,0]=-1
        with self.assertRaises(ValueError):self.read()
    def test_negative_start_rejected(self):
        self.data[5][0,0]=-1
        with self.assertRaises(ValueError):self.read()
    def test_float_counts_rejected(self):
        self.data[4]=np.array([[2.]])
        with self.assertRaises(ValueError):self.read()
    def test_bool_counts_rejected(self):
        self.data[4]=np.array([[True]])
        with self.assertRaises(ValueError):self.read()
    def test_range_beyond_capacity_rejected(self):
        self.data[5][0,0]=3
        with self.assertRaises(ValueError):self.read()
    def test_saturation_not_silently_passed(self):
        self.data[4][0,0]=4;self.data[5][0,0]=0
        with self.assertRaises(ValueError):self.read()
    def test_shape_mismatch_rejected(self):
        self.data[3]=np.zeros((3,1))
        with self.assertRaises(ValueError):self.read()
    def test_capacity_budget_before_copy(self):
        with self.assertRaises(ValueError):self.read(capacity_limit=3)
    def test_tuple_size_rejected(self):
        with self.assertRaises(ValueError):self.read(self.data[:5])
    def test_non_numeric_populated_fields_rejected(self):
        self.data[3]=np.full((4,1),'bad',dtype=object)
        with self.assertRaises(ValueError):self.read()
    def test_source_arrays_are_not_modified(self):
        before=copy.deepcopy(self.data);self.read()
        for a,b in zip(before,self.data):np.testing.assert_array_equal(a,b)
    def test_report_data_not_aliased(self):
        r=self.read();r['contacts'][0]['position_native'][0]=12
        self.assertEqual(self.data[1][1,0],0.)
        self.channel['name']='modified';self.assertNotEqual(r['channel']['name'],self.channel['name'])
    def test_exact_paths_required(self):
        for p in ['/World/envs/*/Robot/body','/World/a/../b']:
            self.channel['expected_sensor_body']=p
            with self.assertRaises(ValueError):self.read()
    def test_same_body_pair_rejected(self):
        self.channel['expected_filter_body']=self.channel['expected_sensor_body']
        with self.assertRaises(ValueError):self.read()
    def test_existing_view_is_read_once_with_dt(self):
        data=self.data
        class View:
            def __init__(self):self.calls=[]
            def get_contact_data(self,*,dt):self.calls.append(dt);return data
        v=View();r=capture_singleton_view(v,channel=self.channel,step=0,dt=.005)
        self.assertEqual(v.calls,[.005]);self.assertEqual(r['populated_count'],2)
    def test_invalid_dt_rejected_before_native_read(self):
        class View:
            def get_contact_data(self,**kw):raise AssertionError('Native read must not happen')
        for dt in [0,-1,float('nan'),True]:
            with self.assertRaises(ValueError):capture_singleton_view(View(),channel=self.channel,step=0,dt=dt)

if __name__=='__main__':unittest.main()