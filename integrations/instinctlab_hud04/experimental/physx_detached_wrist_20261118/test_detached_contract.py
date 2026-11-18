"""Synthetic contract tests ONLY, never physical evidence."""
import copy,unittest
from detached_contract import evaluate,validate_report,CHANNELS
def fixture():
    samples={k:[2.]*8 if k in ['box','detached_left','detached_right'] else [0.]*8 for k in CHANNELS}
    counts={k:[1]*8 if k in ['box','detached_left','detached_right'] else [0]*8 for k in CHANNELS};copies=[]
    for side in ['left','right']:
        for role in ['Parent','Child']:
            link=side+('_wrist_pitch_link' if role=='Parent' else '_wrist_roll_link');source='/World/envs/env_0/Robot/'+link;target='/World/Detached_'+side+'/'+role
            suffix='/collisions/official_'+link+('_2/cylinder' if role=='Parent' else '_1/node_STL_BINARY_')
            copies.append(dict(side=side,role=role,source=source,target=target,selected_source=source+suffix,selected_target=target+suffix,original={'synthetic':True},clone={'synthetic':True}))
    trace=[]
    for i in range(8):
        before={'box_A':{'pose_xyzw':[20.,0.,1.,0.,0.,0.,1.]},'box_B':{'pose_xyzw':[20.08,0.,1.,0.,0.,0.,1.]}}
        for side,x in [('left',5.),('right',10.)]:
            for role in ['Parent','Child']:before[side+'_'+role]=dict(source_xyzw=[0.,0.,0.,0.,0.,0.,1.],detached_xyzw=[x,0.,2.,0.,0.,0.,1.])
        trace.append(dict(step=i,before=before))
    return samples,counts,trace,copies
class Contract(unittest.TestCase):
    def test_synthetic_positive_no_waiver(self):
        v=evaluate(*fixture());self.assertTrue(all(v['observed_articulation_vs_detached_context_difference'].values()));self.assertEqual(v['automatic_pair_exclusions'],0);self.assertFalse(v['articulation_filter_causal_mechanism_proven'])
    def test_box_zero(self):
        s,c,t,p=fixture();s['box']=[0.]*8;self.assertFalse(evaluate(s,c,t,p)['positive_control_valid'])
    def test_detached_zero(self):
        s,c,t,p=fixture();s['detached_left']=[0.]*8;self.assertFalse(evaluate(s,c,t,p)['observed_articulation_vs_detached_context_difference']['left'])
    def test_articulated_positive(self):
        s,c,t,p=fixture();s['articulated_left']=[1.]*8;self.assertFalse(evaluate(s,c,t,p)['observed_articulation_vs_detached_context_difference']['left'])
    def test_missing_channel(self):
        s,c,t,p=fixture();s.pop('box')
        with self.assertRaises(ValueError):evaluate(s,c,t,p)
    def test_nonfinite(self):
        s,c,t,p=fixture();s['box'][0]=float('nan')
        with self.assertRaises(ValueError):evaluate(s,c,t,p)
    def test_capacity(self):
        s,c,t,p=fixture();c['box'][0]=256
        with self.assertRaises(ValueError):evaluate(s,c,t,p)
    def test_transform_changed(self):
        s,c,t,p=fixture();t[0]['before']['left_Child']['detached_xyzw'][0]+=.001
        with self.assertRaises(ValueError):evaluate(s,c,t,p)
    def test_box_separated(self):
        s,c,t,p=fixture();t[0]['before']['box_B']['pose_xyzw'][0]=22.
        with self.assertRaises(ValueError):evaluate(s,c,t,p)
    def test_selected_shape_changed(self):
        s,c,t,p=fixture();p[0]['clone']={'changed':True}
        with self.assertRaises(ValueError):evaluate(s,c,t,p)
    def test_wrong_binding(self):
        s,c,t,p=fixture();p[0]['selected_source']='wrong'
        with self.assertRaises(ValueError):evaluate(s,c,t,p)
    def test_short_trace(self):
        s,c,t,p=fixture()
        with self.assertRaises(ValueError):evaluate(s,c,t[:-1],p)
if __name__=='__main__':unittest.main()