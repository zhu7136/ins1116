import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from context import models,features,diagnostic,dependency
from pose_tasks import (Feature,evaluate,rigid,rotation,stack_channels,
                        neutral_orientation_proposal,build_joint_pose_window)

class PoseTasksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.point_model=models();cls.features=features()
        cls.q=diagnostic(cls.model)[[0,257,495,996]]
        cls.roots=np.tile(np.eye(4),(4,1,1));cls.roots[:,:3,:3]=Rotation.from_rotvec([.2,-.1,.3]).as_matrix()
        cls.roots[:,:3,3]=[2,-1,.8]

    def test_shapes(self):
        p,r,j,k=evaluate(self.model,self.q,self.roots,self.features)
        self.assertEqual(p.shape,(4,5,3));self.assertEqual(r.shape,(4,5,3,3))
        self.assertEqual(j.shape,(4,5,3,35));self.assertEqual(k.shape,(4,5,3,3,35))
    def test_35_columns_no_head(self):
        self.assertEqual(len(self.model.names)+6,35);self.assertFalse(any('head' in n for n in self.model.names))
    def test_unknown_target_is_null(self):
        c=stack_channels(self.model,self.q,self.roots,self.features)
        self.assertIsNone(c['point_residual']);self.assertIsNone(c['basis_residual']);self.assertIsNone(c['support_intervals'])
    def test_explicit_targets_actual_residual(self):
        p,r=evaluate(self.model,self.q,self.roots,self.features,jacobian=False)
        d=Rotation.from_rotvec([.01,0,0]).as_matrix()@r
        c=stack_channels(self.model,self.q,self.roots,self.features,desired_points=p+.01,desired_bases=d)
        np.testing.assert_allclose(c['point_residual'],-.01,atol=1e-14)
        np.testing.assert_allclose(c['basis_residual'],(r-d).ravel(),atol=1e-14)
    def test_invalid_target_rotation_rejected(self):
        with self.assertRaises(ValueError):stack_channels(self.model,self.q,self.roots,self.features,desired_bases=np.zeros((4,5,3,3)))
    def test_target_shape_rejected(self):
        with self.assertRaises(ValueError):stack_channels(self.model,self.q,self.roots,self.features,desired_points=np.zeros((4,1,3)))
    def test_reflection_rejected(self):
        with self.assertRaises(ValueError):rotation(np.diag([-1,1,1]))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):rigid(np.full((4,4),np.nan))
    def test_bad_last_row_rejected(self):
        t=np.eye(4);t[3,0]=1
        with self.assertRaises(ValueError):rigid(t)
    def test_feature_transform_frozen(self):
        t=np.eye(4);f=Feature('n','base_link',t,'URDF_link_diagnostic');t[0,3]=1
        self.assertEqual(f.link_from_feature[0,3],0)
        with self.assertRaises(ValueError):f.link_from_feature[0,3]=2
    def test_fake_material_role_rejected(self):
        with self.assertRaises(ValueError):Feature('palm','left_wrist_roll_link',np.eye(4),'accepted_palm')
    def test_duplicate_feature_rejected(self):
        with self.assertRaises(ValueError):evaluate(self.model,self.q,self.roots,[self.features[0]]*2)
    def test_missing_link_rejected(self):
        f=Feature('unknown','unknown',np.eye(4),'URDF_link_diagnostic')
        with self.assertRaises(ValueError):evaluate(self.model,self.q,self.roots,[f])
    def test_missing_root_rejected(self):
        with self.assertRaises(ValueError):evaluate(self.model,self.q,self.roots[:1],self.features)
    def test_foot_frame_is_exact_box_bottom(self):
        for f in self.features[-2:]:np.testing.assert_allclose(f.link_from_feature[:3,3],[.018,0,-.0635],atol=1e-15)
    def test_no_fake_palm(self):
        for f in self.features[1:3]:
            self.assertEqual(f.role,'URDF_link_diagnostic');np.testing.assert_array_equal(f.link_from_feature,np.eye(4))
    def test_root_translation_has_no_rotation_derivative(self):
        _,_,jp,jr=evaluate(self.model,self.q,self.roots,self.features)
        np.testing.assert_array_equal(jr[...,:3],0)
        np.testing.assert_array_equal(jp[:,:,:,:3],np.broadcast_to(np.eye(3),(4,5,3,3)))
    def test_world_rigid_equivariance(self):
        p,r=evaluate(self.model,self.q,self.roots,self.features,jacobian=False)
        t=np.eye(4);t[:3,:3]=Rotation.from_rotvec([-.3,.2,.1]).as_matrix();t[:3,3]=[-3,4,2]
        p2,r2=evaluate(self.model,self.q,t@self.roots,self.features,jacobian=False)
        np.testing.assert_allclose(p2,np.einsum('ij,nkj->nki',t[:3,:3],p)+t[:3,3],atol=2e-15)
        np.testing.assert_allclose(r2,t[:3,:3]@r,atol=2e-15)
    def test_all35_derivative_columns(self):
        _,_,jp,jr=evaluate(self.model,self.q,self.roots,self.features);eps=1e-6
        perturb=dependency('named_target').perturb
        for c in range(35):
            d=np.zeros(35);d[c]=eps
            plus=evaluate(self.model,self.q+d[6:],np.array([perturb(r,d[:6]) for r in self.roots]),self.features,jacobian=False)
            minus=evaluate(self.model,self.q-d[6:],np.array([perturb(r,-d[:6]) for r in self.roots]),self.features,jacobian=False)
            np.testing.assert_allclose((plus[0]-minus[0])/(2*eps),jp[...,c],atol=2e-8,rtol=0)
            np.testing.assert_allclose((plus[1]-minus[1])/(2*eps),jr[...,c],atol=2e-8,rtol=0)
    def test_neutral_proposal_maps_neutral_exactly(self):
        s=Rotation.from_rotvec([[.1,0,0],[0,.2,0]]).as_matrix()
        t=Rotation.from_rotvec([[0,0,.3],[.4,0,0]]).as_matrix()
        result,_=neutral_orientation_proposal(s[None],s,t);np.testing.assert_allclose(result[0],t,atol=1e-15)
    def test_neutral_proposal_world_rotation_equivariance(self):
        s=Rotation.from_rotvec([[.1,0,0],[0,.2,0]]).as_matrix();t=np.tile(np.eye(3),(2,1,1))
        out,offset=neutral_orientation_proposal(s[None],s,t)
        r=Rotation.from_rotvec([.2,-.1,.4]).as_matrix()
        out2,offset2=neutral_orientation_proposal((r@s)[None],r@s,r@t)
        np.testing.assert_allclose(out2,r@out,atol=1e-15);np.testing.assert_allclose(offset,offset2,atol=1e-15)
    def test_bad_neutral_shape_rejected(self):
        with self.assertRaises(ValueError):neutral_orientation_proposal(np.eye(3)[None,None],np.eye(3),np.eye(3))
    def window(self,q=None,names=None):
        return build_joint_pose_window(dependency('window_problem').build,self.point_model,self.model,
            self.q if q is None else q,self.model.names if names is None else names,self.roots,
            [0,257,495,996],self.features)
    def test_existing_hard_rows_unchanged(self):
        got=self.window()['existing_window']
        old=dependency('window_problem').build(self.point_model,self.q,self.model.names,self.roots,[0,257,495,996])
        np.testing.assert_array_equal(got.hard_matrix.toarray(),old.hard_matrix.toarray())
        np.testing.assert_array_equal(got.hard_lower,old.hard_lower);np.testing.assert_array_equal(got.hard_upper,old.hard_upper)
        np.testing.assert_array_equal(got.step_upper,old.step_upper)
    def test_named_reversal_same_pose_channels(self):
        a=self.window();b=self.window(self.q[:,::-1],self.model.names[::-1])
        np.testing.assert_array_equal(a['pose_channels']['basis_jacobian'].toarray(),b['pose_channels']['basis_jacobian'].toarray())
    def test_no_gap_velocity_edges_or_false_training(self):
        w=self.window();self.assertEqual(w['existing_window'].edges,[])
        self.assertFalse(w['accepted_reference']);self.assertFalse(w['training_ready']);self.assertFalse(w['optimizer_implemented'])
    def test_no_inputs_changed(self):
        before=self.q.copy();roots=self.roots.copy();self.window()
        np.testing.assert_array_equal(self.q,before);np.testing.assert_array_equal(self.roots,roots)
    def test_bone_direction_unknown_is_null(self):
        c=stack_channels(self.model,self.q,self.roots,self.features)
        self.assertIsNone(c['direction_residual']);self.assertEqual(c['direction_jacobian'].shape,(96,140))
    def test_bone_direction_root_columns_are_zero(self):
        c=stack_channels(self.model,self.q,self.roots,self.features)
        j=c['direction_jacobian'].toarray()
        for frame in range(4):np.testing.assert_array_equal(j[:,35*frame:35*frame+6],0)
    def test_bone_direction_explicit_residual(self):
        want=self.model.evaluate(self.q)
        c=stack_channels(self.model,self.q,self.roots,self.features,desired_directions=want)
        np.testing.assert_array_equal(c['direction_residual'],0)
    def test_bone_direction_nonunit_rejected(self):
        with self.assertRaises(ValueError):stack_channels(self.model,self.q,self.roots,self.features,desired_directions=2*self.model.evaluate(self.q))
    def test_pose_only_rank_not_35_but_combined_local_rank35(self):
        _,_,p,r=evaluate(self.model,self.q,self.roots,self.features)
        _,d=self.model.evaluate(self.q,True)
        d=np.pad(d.reshape(4,24,29),((0,0),(0,0),(6,0)))
        pose=np.concatenate([p.reshape(4,15,35),r.reshape(4,45,35)],axis=1)
        self.assertTrue(np.all(np.linalg.matrix_rank(pose,tol=1e-8)<=30))
        np.testing.assert_array_equal(np.linalg.matrix_rank(np.concatenate([pose,d],axis=1),tol=1e-8),35)

if __name__=='__main__':unittest.main()