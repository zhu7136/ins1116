import copy
import unittest
import numpy as np
from audit import HERE, REPO, MODEL, MODEL_SHA, digest, read, joint_metadata, finite_differences, one

class AuditTests(unittest.TestCase):
    def test_exact_model_identity(self):self.assertEqual(digest(MODEL),MODEL_SHA)
    def test_all_input_bytes_still_match(self):
        for name,h in read(HERE/'AUDIT.json')['input_files_sha256'].items():self.assertEqual(digest(REPO/name),h,name)
    def test_no_policy_or_training_promotion(self):
        a=read(HERE/'AUDIT.json')
        self.assertEqual(a['automatic_pair_exclusions'],0)
        self.assertEqual(a['optimizer_calls'],0)
        self.assertFalse(a['accepted_reference']);self.assertFalse(a['training_ready'])
        for p in a['pairs']:
            self.assertTrue(p['all_angle_nominal_intersection'])
            self.assertIsNone(p['nominal_penetration_depth_raw'])
            self.assertIsNone(p['physx_signed_separation_or_penetration_depth'])
            self.assertFalse(p['measured_shape_pair_cause_proven'])
            self.assertEqual(len(p['runtime_observations']),3)
    def test_metadata_does_not_grant_filter_causality(self):
        p=read(HERE/'AUDIT.json')['pairs'][0]
        for r in p['runtime_observations']:
            self.assertFalse(r['collision_enabled']['value'])
            self.assertFalse(r['actual_filter_cause_proven'])
            self.assertFalse(r['shape_ids_or_cooked_vertices_bound'])
    def fixture(self):
        return dict(path='/env0/j',body0=['/env0/a'],body1=['/env0/b'],
            collision_enabled=dict(present=True,authored=False,value=False),
            joint_enabled=dict(present=True,value=True),exclude_from_articulation=dict(present=True,value=False))
    def test_cross_environment_binding_rejected(self):
        j=self.fixture();j['body1']=['/env1/b']
        with self.assertRaises(ValueError):joint_metadata(j,'/env0','a','b')
    def test_missing_attribute_rejected(self):
        j=self.fixture();j['collision_enabled']['present']=False
        with self.assertRaises(ValueError):joint_metadata(j,'/env0','a','b')
    def test_explicit_enabled_not_rewritten(self):
        j=self.fixture();j['collision_enabled'].update(authored=True,value=True)
        self.assertTrue(joint_metadata(j,'/env0','a','b')['collision_enabled']['value'])
    def test_inactive_or_excluded_joint_rejected(self):
        for key,value in [('joint_enabled',False),('exclude_from_articulation',True)]:
            j=self.fixture();j[key]['value']=value
            with self.assertRaises(ValueError):joint_metadata(j,'/env0','a','b')
    def test_duplicate_binding_rejected(self):
        with self.assertRaises(ValueError):one([1,1])
    def test_derivatives_include_join_crossing_rows(self):
        x=np.arange(997,dtype=float)[:,None]**2/90**2
        v,a=finite_differences(x)
        self.assertEqual(a.shape,(995,1))
        np.testing.assert_allclose(a,2,atol=1e-9)
        joins=read(HERE/'BOUNDARY_DIAGNOSTICS.json')['window_joins']
        self.assertEqual(len(joins),16)
        for row in joins:self.assertTrue(row['derivative_differences_are_not_C2_discontinuity_proofs'])

if __name__=='__main__':unittest.main()