"""Synthetic schema/classification tests, NOT measurements or acceptance."""
import copy, math, unittest
from contact_probe import evaluate,ROLES,SIDES,STEPS
from launch_once import validate_report

class ContactContractTests(unittest.TestCase):
    def series(self):
        d={r+'_'+s:[0.]*STEPS for r in ROLES for s in SIDES}
        d['box_positive_control']=[1.]*STEPS
        for s in SIDES:d['explicit_true_'+s]=[.1]*STEPS
        return d
    def report(self):
        b=[]
        for i,r in enumerate(ROLES):
            for s in SIDES:
                root=f'/World/envs/env_{i}/Robot'
                b.append(dict(role=r,side=s,joint=root+'/joints/'+s+'_wrist_roll_joint',body0=[root+'/'+s+'_wrist_pitch_link'],body1=[root+'/'+s+'_wrist_roll_link'],
                    before=dict(value=False,authored=False),observed_value=r=='explicit_true',observed_authored=r!='original_fallback'))
        samples=self.series()
        return dict(stage='hud04_controlled_wrist_contact_observation',source_commit='a'*40,probe_sha256='b'*64,
            accepted_reference=False,training_ready=False,full_task_tested=False,graceful_full_cleanup_proven=False,
            body_dof=29,head_action_dof=0,explicit_physics_steps=STEPS,num_envs=3,exact_joint_body_bindings=b,
            measured_force_norms=samples,verdict=evaluate(samples))
    def test_positive_controls_required(self):
        s=self.series();s['box_positive_control']=[0.]*STEPS
        self.assertFalse(any(evaluate(s)['controlled_wrist_body_pair_filter_supported'].values()))
    def test_all_zero_not_filter_proof(self):
        s={k:[0.]*STEPS for k in self.series()}
        self.assertFalse(any(evaluate(s)['controlled_wrist_body_pair_filter_supported'].values()))
    def test_enabled_wrist_required_independently(self):
        s=self.series();s['explicit_true_left']=[0.]*STEPS
        self.assertEqual(evaluate(s)['controlled_wrist_body_pair_filter_supported'],dict(left=False,right=True))
    def test_original_contact_rejects_filter_claim(self):
        s=self.series();s['original_fallback_left'][3]=.01
        self.assertFalse(evaluate(s)['controlled_wrist_body_pair_filter_supported']['left'])
    def test_explicit_false_contact_rejects_filter_claim(self):
        s=self.series();s['explicit_false_right'][0]=.01
        self.assertFalse(evaluate(s)['controlled_wrist_body_pair_filter_supported']['right'])
    def test_positive_classification_never_full_pair_or_reference_acceptance(self):
        v=evaluate(self.series());self.assertTrue(all(v['controlled_wrist_body_pair_filter_supported'].values()))
        for k in ('all_joint_pair_filters_verified','cooked_geometry_verified','accepted_reference','training_ready','reference_acceptance'):self.assertIs(v[k],False)
    def test_nonfinite_negative_boolean_rejected(self):
        for x in (math.nan,math.inf,-1.,True):
            s=self.series();s['box_positive_control'][0]=x
            with self.assertRaises(ValueError):evaluate(s)
    def test_missing_extra_and_short_channels_rejected(self):
        s=self.series();s.pop('explicit_true_left')
        with self.assertRaises(ValueError):evaluate(s)
        s=self.series();s['extra']=[0.]*STEPS
        with self.assertRaises(ValueError):evaluate(s)
        s=self.series();s['explicit_true_left'].pop()
        with self.assertRaises(ValueError):evaluate(s)
    def test_exact_report_validates(self):self.assertTrue(validate_report(self.report(),'a'*40,'b'*64))
    def test_path_binding_not_suffix_based(self):
        d=self.report();d['exact_joint_body_bindings'][0]['body1']=['/Other/left_wrist_roll_link']
        with self.assertRaises(ValueError):validate_report(d,'a'*40,'b'*64)
    def test_policy_before_original_immutable(self):
        d=self.report();d['exact_joint_body_bindings'][0]['before']['authored']=True
        with self.assertRaises(ValueError):validate_report(d,'a'*40,'b'*64)
    def test_acceptance_cleanup_forgery_rejected(self):
        for k in ('accepted_reference','training_ready','graceful_full_cleanup_proven','full_task_tested'):
            d=self.report();d[k]=True
            with self.assertRaises(ValueError):validate_report(d,'a'*40,'b'*64)
    def test_identity_and_computed_verdict_rejected(self):
        d=self.report()
        with self.assertRaises(ValueError):validate_report(d,'c'*40,'b'*64)
        with self.assertRaises(ValueError):validate_report(d,'a'*40,'c'*64)
        d['verdict']['cooked_geometry_verified']=True
        with self.assertRaises(ValueError):validate_report(d,'a'*40,'b'*64)
if __name__=='__main__':unittest.main()