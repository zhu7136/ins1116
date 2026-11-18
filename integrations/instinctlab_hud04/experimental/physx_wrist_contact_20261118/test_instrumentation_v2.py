"""Synthetic data only; these tests do not establish physical contact."""
import copy, math, unittest
from instrumentation_v2 import validate_instrumentation,CHANNELS,STEPS
from contact_probe import evaluate

def fixture(samples=None):
    samples=copy.deepcopy(samples) if samples is not None else {k:[1. if k=='box_positive_control' or k.startswith('explicit_true') else 0.]*STEPS for k in CHANNELS}
    def state(x,vx):return {'pose_xyzw':[x,0.,1.,0.,0.,0.,1.],'velocity':[vx,0.,0.,0.,0.,0.]}
    trace=[]
    for i in range(STEPS):
        before={'A':state(25.,.1),'B':state(25.08,-.1)}
        trace.append({'step':i,'before':before,'after':copy.deepcopy(before),'native_force_vector':[samples['box_positive_control'][i],0.,0.]})
    d={'revision':2,'contact_processing':{'setting':'/physics/disableContactProcessing','before':True,'after':False},
       'disable_contact_processing_after_reset':False,'box_size':[.1,.1,.1],'box_state_trace':trace,
       'official_sensor_force_vectors':[[x,0.,0.] for x in samples['box_positive_control']],
       'native_contact_counts':{k:[int(x>0) for x in values] for k,values in samples.items()},
       'contact_capacity_per_channel':256,'box_threshold_zero_requested':True}
    return d,samples

class InstrumentationTests(unittest.TestCase):
    def test_valid_synthetic_control(self):
        d,s=fixture();r=validate_instrumentation(d,s);self.assertTrue(r['positive_control_valid']);self.assertFalse(r['accepted_reference'])
    def test_zero_still_inconclusive(self):
        d,s=fixture({k:[0.]*STEPS for k in CHANNELS});r=validate_instrumentation(d,s)
        self.assertFalse(r['positive_control_valid']);self.assertFalse(evaluate(s,r)['independent_box_contact_positive'])
    def test_contact_count_required(self):
        d,s=fixture();d['native_contact_counts']['box_positive_control']=[0]*STEPS
        self.assertFalse(validate_instrumentation(d,s)['positive_control_valid'])
    def test_saturation_in_any_channel_rejects(self):
        d,s=fixture();d['native_contact_counts']['explicit_true_left'][0]=256
        self.assertFalse(validate_instrumentation(d,s)['positive_control_valid'])
    def test_independent_sensor_mismatch_rejects(self):
        d,s=fixture();d['official_sensor_force_vectors'][0]=[0.,0.,0.]
        self.assertFalse(validate_instrumentation(d,s)['positive_control_valid'])
    def test_raw_magnitude_mismatch_rejects(self):
        d,s=fixture();d['box_state_trace'][0]['native_force_vector']=[2.,0.,0.]
        self.assertFalse(validate_instrumentation(d,s)['positive_control_valid'])
    def test_pre_step_state_mandatory(self):
        d,s=fixture();d['box_state_trace'][3]['before']['A']['pose_xyzw'][0]=24.
        with self.assertRaises(ValueError):validate_instrumentation(d,s)
    def test_quaternion_xyzw_not_wxyz(self):
        d,s=fixture();d['box_state_trace'][0]['before']['A']['pose_xyzw'][3:]=[1.,0.,0.,0.]
        with self.assertRaises(ValueError):validate_instrumentation(d,s)
    def test_setting_must_read_back_false(self):
        d,s=fixture();d['disable_contact_processing_after_reset']=True
        with self.assertRaises(ValueError):validate_instrumentation(d,s)
    def test_missing_data_rejected(self):
        d,s=fixture();d['box_state_trace'].pop()
        with self.assertRaises(ValueError):validate_instrumentation(d,s)
    def test_nan_rejected(self):
        d,s=fixture();d['official_sensor_force_vectors'][0][0]=math.nan
        with self.assertRaises(ValueError):validate_instrumentation(d,s)
    def test_improper_velocity_rejected(self):
        d,s=fixture();d['box_state_trace'][0]['before']['B']['velocity'][0]=.1
        with self.assertRaises(ValueError):validate_instrumentation(d,s)
    def test_invalid_instrumentation_blocks_positive_raw_force_claim(self):
        d,s=fixture();d['native_contact_counts']['box_positive_control']=[0]*STEPS
        self.assertFalse(any(evaluate(s,validate_instrumentation(d,s))['controlled_wrist_body_pair_filter_supported'].values()))

if __name__=='__main__':unittest.main()