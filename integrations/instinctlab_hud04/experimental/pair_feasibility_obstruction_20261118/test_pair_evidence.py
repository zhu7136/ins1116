import copy,json,math,unittest
import numpy as np
import pair_policy_evidence as P
import prove_wrist_axis_obstruction as W

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=P.collect();s=json.loads((P.EVIDENCE/'post_physics_init.json').read_text())
        cls.root=s['robot_root'];cls.j=next(j for j in s['joints'] if j['path'].endswith('/left_wrist_roll_joint'))
    def test_actual_metadata_is_not_exclusion_permission(self):
        for phase in self.result['snapshots'].values():
            for row in phase['rows']:
                self.assertFalse(row['resolved_joint_collision_enabled']);self.assertFalse(row['collision_enabled_authored'])
                self.assertFalse(row['pair_exclusion_authorized']);self.assertFalse(row['actual_physx_pair_filter_verified'])
    def test_all_pairs_remain_and_failure_is_preserved(self):
        self.assertEqual(self.result['all_shape_pairs_retained'],903);self.assertEqual(self.result['collision_constraints_removed'],0)
        self.assertEqual(self.result['worker_return_code'],-11);self.assertFalse(self.result['observation_completed'])
    def test_exact_body_binding_not_suffix_match(self):
        j=copy.deepcopy(self.j);j['body0'][0]=j['body0'][0].replace('/env_0/','/env_1/')
        with self.assertRaises(ValueError):P.bind_row(j,self.root,'left_wrist_pitch_link','left_wrist_roll_link')
    def test_missing_collision_property_not_disabled(self):
        j=copy.deepcopy(self.j);j['collision_enabled']['present']=False
        with self.assertRaises(ValueError):P.bind_row(j,self.root,'left_wrist_pitch_link','left_wrist_roll_link')
    def test_disabled_joint_cannot_bind(self):
        j=copy.deepcopy(self.j);j['joint_enabled']['value']=False
        with self.assertRaises(ValueError):P.bind_row(j,self.root,'left_wrist_pitch_link','left_wrist_roll_link')
    def test_excluded_articulation_cannot_bind(self):
        j=copy.deepcopy(self.j);j['exclude_from_articulation']['value']=True
        with self.assertRaises(ValueError):P.bind_row(j,self.root,'left_wrist_pitch_link','left_wrist_roll_link')
    def test_explicit_true_is_not_rewritten_disabled(self):
        j=copy.deepcopy(self.j);j['collision_enabled'].update(value=True,authored=True)
        r=P.bind_row(j,self.root,'left_wrist_pitch_link','left_wrist_roll_link')
        self.assertEqual(r['metadata_intent'],'joint_body_collision_enabled');self.assertFalse(r['pair_exclusion_authorized'])
    def test_both_exact_original_geometry_invariant_witnesses(self):
        for side in ['left','right']:
            r=W.prove(side)
            self.assertTrue(r['invariant_for_all_angles_analytic']);self.assertGreater(r['hull_halfspace_margin_raw'],.02)
            self.assertGreater(r['cylinder_radial_margin_raw'],.02);self.assertGreater(r['cylinder_cap_margin_raw'],.02)
            self.assertTrue(r['witness_is_not_physical_penetration_depth'])
    def test_invariant_axis_for_angles_outside_joint_limits(self):
        point=np.array([.00029347,0,0])
        for a in [-200*math.pi,-2.1,0,1.571,1000]:
            np.testing.assert_array_equal(W.TK.K.axis_rotation(np.array([1.,0,0]),a)@point,point)
if __name__=='__main__':unittest.main()