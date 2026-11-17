import importlib.util
import unittest
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('conversion_under_test',Path(__file__).with_name('prepare_source_conversion.py'))
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class ConversionStagingTests(unittest.TestCase):
    def setUp(self):
        self.names=['body_'+str(i) for i in range(29)]
        self.limits={n:dict(lower=-1.,upper=1.,velocity=10.) for n in self.names}
        self.source=dict(joint_names=np.array(self.names[::-1]),joint_pos=np.zeros((5,29),dtype=np.float32),
            base_pos_w=np.arange(15,dtype=np.float32).reshape(5,3),
            base_quat_w=np.tile(np.array([0.,0.,0.,1.],dtype=np.float32),(5,1)),framerate=np.array(90))

    def stage(self):return M.stage_arrays(self.source,self.names,self.limits)

    def test_lossless_column_roundtrip(self):
        self.source['joint_pos'][:]=np.arange(145).reshape(5,29)/100
        a=self.stage();M.verify_arrays(a,self.source,a)
        np.testing.assert_array_equal(a['source_joint_values_name_reordered_only'],self.source['joint_pos'][:,::-1])

    def test_root_quat_unchanged_without_normalization(self):
        self.source['base_quat_w']*=1.000001
        a=self.stage()
        np.testing.assert_array_equal(a['unverified_raw_root_quaternion'],self.source['base_quat_w'])
        np.testing.assert_array_equal(a['unverified_raw_root_position'],self.source['base_pos_w'])

    def test_no_training_consumer_keys(self):
        self.assertFalse({'joint_pos','joint_names','base_pos_w','base_quat_w','framerate'} & set(self.stage()))

    def test_no_clip_on_bad_frames(self):
        self.source['joint_pos'][2,0]=3
        a=self.stage();self.assertEqual(a['source_joint_values_name_reordered_only'][2,-1],3)
        self.assertTrue(a['source_vs_target_position_violation_mask'][2,-1])
        self.assertTrue(a['source_vs_target_speed_violation_mask'][1,-1])

    def test_time_uses_90_hz_intervals(self):
        a=self.stage();self.assertEqual(len(a['time_seconds']),5)
        self.assertEqual(a['time_seconds'][-1],4/90)

    def test_non_90_rejected(self):
        self.source['framerate']=np.array(50)
        with self.assertRaises(ValueError):self.stage()

    def test_nan_rejected(self):
        self.source['base_pos_w'][0,0]=np.nan
        with self.assertRaises(ValueError):self.stage()

    def test_head_channel_rejected(self):
        self.source['joint_names'][0]='head'
        with self.assertRaises(ValueError):self.stage()

    def test_duplicate_names_rejected(self):
        self.source['joint_names'][0]=self.source['joint_names'][1]
        with self.assertRaises(ValueError):self.stage()

    def test_readback_tampering_rejected(self):
        a=self.stage();b={k:v.copy() for k,v in a.items()};b['unverified_raw_root_position'][0,0]+=1
        with self.assertRaises(ValueError):M.verify_arrays(b,self.source,a)

    def test_dtype_change_rejected(self):
        a=self.stage();b={k:v.copy() for k,v in a.items()};b['unverified_raw_root_position']=b['unverified_raw_root_position'].astype(np.float64)
        with self.assertRaises(ValueError):M.verify_arrays(b,self.source,a)

    def test_no_sign_flip(self):
        self.source['joint_pos'][:,3]=-.35
        a=self.stage();np.testing.assert_array_equal(a['source_joint_values_name_reordered_only'][:,25],self.source['joint_pos'][:,3])


if __name__=='__main__':unittest.main()