"""Synthetic tests of proposed setting repair and interpretation; NOT physics."""
import unittest
from contact_processing_guard import enable_contact_processing,measured_classification
class Settings:
    def __init__(self,value=True,ignore=False):self.value=value;self.ignore=ignore;self.calls=[]
    def get_as_bool(self,key):self.calls.append(('get',key));return self.value
    def set_bool(self,key,value):
        self.calls.append(('set',key,value))
        if not self.ignore:self.value=value
class Tests(unittest.TestCase):
    def test_readback_and_no_physics_claim(self):
        s=Settings();d=enable_contact_processing(s)
        self.assertEqual([r[0] for r in s.calls],['get','set','get'])
        self.assertIs(d['before'],True);self.assertIs(d['after'],False)
        for k in ('gpu_effect_verified','positive_control_verified','accepted_reference','training_ready'):self.assertIs(d[k],False)
    def test_already_false(self):self.assertIs(enable_contact_processing(Settings(False))['after'],False)
    def test_ignored_setting_rejected(self):
        with self.assertRaises(RuntimeError):enable_contact_processing(Settings(True,True))
    def test_control_failure_overrides_wrist_claim(self):
        self.assertEqual(measured_classification(dict(independent_box_contact_positive=False,controlled_wrist_body_pair_filter_supported=dict(left=True,right=True))),'INCONCLUSIVE_POSITIVE_CONTROL_FAILED')
    def test_limited_support_never_acceptance(self):
        self.assertEqual(measured_classification(dict(independent_box_contact_positive=True,controlled_wrist_body_pair_filter_supported=dict(left=True,right=True))),'LIMITED_WRIST_PAIR_SUPPORT_ONLY')
    def test_unproven_side(self):
        self.assertEqual(measured_classification(dict(independent_box_contact_positive=True,controlled_wrist_body_pair_filter_supported=dict(left=True,right=False))),'INCONCLUSIVE_WRIST_TREATMENT_CONTROL_NOT_SATISFIED')
if __name__=='__main__':unittest.main()