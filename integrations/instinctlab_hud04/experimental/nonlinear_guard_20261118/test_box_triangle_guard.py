import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from box_triangle_guard import BoxTriangleGuard,sat_gaps,clip_polygon,cropped_triangles,reject_new_or_worse


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.h=np.array([.4,.3,.2]);self.t=np.eye(4)[None]
        self.tri=np.array([[[-2,-2,0],[2,-2,0],[0,2,0]]],float)
        self.g=BoxTriangleGuard(self.tri,tolerance=1e-9)

    def scan(self,guard=None,pose=None,ids=None):
        return (guard or self.g).scan(self.t if pose is None else pose,self.h,pose_ids=['box:0'] if ids is None else ids)

    def test_strict_crossing(self):self.assertEqual(self.scan()['strict_hit_frames'],[0])
    def test_crossing_depth_along_sat_axis(self):self.assertAlmostEqual(self.scan()['maximum_negative_sat_raw'],.2)
    def test_face_touch_not_strict(self):
        t=self.t.copy();t[:,2,3]=.2;self.assertEqual(self.scan(pose=t)['hit_frame_count'],0)
    def test_clear_above(self):
        t=self.t.copy();t[:,2,3]=.3;self.assertEqual(self.scan(pose=t)['fresh_triangle_queries'],0)
    def test_finite_triangle_not_infinite_plane(self):
        tri=self.tri+np.array([20,0,0]);self.assertGreater(sat_gaps(tri,self.h)[0],0)
        self.assertEqual(self.scan(BoxTriangleGuard(tri,tolerance=1e-9))['hit_frame_count'],0)
    def test_winding_invariant(self):np.testing.assert_allclose(sat_gaps(self.tri,self.h),sat_gaps(self.tri[:,::-1],self.h))
    def test_solid_containment_not_excluded(self):self.assertIs(self.scan()['solid_containment_excluded'],False)
    def test_not_motion_acceptance(self):self.assertIs(self.scan()['accepted_reference'],False)
    def test_new_sidewall_is_rejected(self):
        tri=np.array([[[2,-2,-2],[2,2,-2],[2,0,2]]],float);guard=BoxTriangleGuard(tri,tolerance=1e-9)
        old=self.scan(guard);pose=self.t.copy();pose[:,0,3]=2
        new=self.scan(guard,pose);r=reject_new_or_worse(old,new,numerical_slack=1e-12)
        self.assertTrue(r['reject']);self.assertEqual(r['new_hit_pose_indices'],[0]);self.assertEqual(old['fresh_triangle_queries'],0)
    def test_increased_same_frame_overlap_rejected(self):
        p=self.t.copy();p[:,2,3]=.15;a=self.scan(pose=p);b=self.scan()
        self.assertTrue(reject_new_or_worse(a,b,numerical_slack=1e-12)['reject'])
    def test_reducing_overlap_does_not_reject(self):
        p=self.t.copy();p[:,2,3]=.15
        self.assertFalse(reject_new_or_worse(self.scan(),self.scan(pose=p),numerical_slack=1e-12)['reject'])
    def test_rigid_equivariance(self):
        r=Rotation.from_rotvec([.2,-.8,.3]).as_matrix();c=np.array([3.,-.5,2.])
        guard=BoxTriangleGuard(self.tri@r.T+c,tolerance=1e-9);p=self.t.copy();p[0,:3,:3]=r;p[0,:3,3]=c
        self.assertAlmostEqual(self.scan()['maximum_negative_sat_raw'],self.scan(guard,p)['maximum_negative_sat_raw'])
    def test_bad_rotation_rejected(self):
        p=self.t.copy();p[0,0,0]=2
        with self.assertRaises(ValueError):self.scan(pose=p)
    def test_degenerate_rejected(self):
        with self.assertRaises(ValueError):BoxTriangleGuard(np.zeros((1,3,3)),tolerance=1e-9)
    def test_nan_rejected(self):
        tri=self.tri.copy();tri[0,0,0]=np.nan
        with self.assertRaises(ValueError):BoxTriangleGuard(tri,tolerance=1e-9)
    def test_explicit_numerical_tolerance(self):
        with self.assertRaises(ValueError):BoxTriangleGuard(self.tri,tolerance=.02)
    def test_pose_binding_rejected(self):
        a=self.scan();b=self.scan(ids=['other:0'])
        with self.assertRaises(ValueError):reject_new_or_worse(a,b,numerical_slack=0)
    def test_geometry_binding_rejected(self):
        a=self.scan();b=self.scan(BoxTriangleGuard(self.tri[:,::-1],tolerance=1e-9))
        with self.assertRaises(ValueError):reject_new_or_worse(a,b,numerical_slack=0)
    def test_mutated_geometry_refused(self):
        with self.assertRaises(ValueError):self.g.triangles[0,0,0]=20
    def test_incomplete_report_rejected(self):
        a=self.scan();b=copy.deepcopy(a);b['full_surface_scan']=False
        with self.assertRaises(ValueError):reject_new_or_worse(a,b,numerical_slack=0)
    def test_source_not_modified(self):
        before=self.tri.copy();self.scan();np.testing.assert_array_equal(self.tri,before)
    def test_all_poses_not_just_first(self):
        p=np.repeat(self.t,3,axis=0);p[:2,2,3]=1
        self.assertEqual(self.g.scan(p,self.h,pose_ids=['0','1','2'])['strict_hit_frames'],[2])
    def test_random500_against_independent_polygon_clipping(self):
        rng=np.random.default_rng(278);tri=rng.uniform(-1,1,(500,3,3));gaps=sat_gaps(tri,self.h)
        for triangle,gap in zip(tri,gaps):
            p=triangle.copy()
            for axis in range(3):
                for sign in [-1,1]:p=clip_polygon(p,axis,sign*self.h[axis],sign)
            strict=bool(len(p) and np.min(self.h-abs(p.mean(axis=0)))>1e-9)
            self.assertEqual(bool(gap < -1e-9),strict)
    def test_crop_area_and_originating_ids(self):
        tri=np.array([[[-2,-2,0],[2,-2,0],[2,2,0]],[[-2,-2,0],[2,2,0],[-2,2,0]]],float)
        out,ids=cropped_triangles(tri,[2,2]);area=np.linalg.norm(np.cross(out[:,1]-out[:,0],out[:,2]-out[:,0]),axis=1).sum()/2
        self.assertAlmostEqual(area,4);self.assertTrue(set(ids)=={0,1});self.assertLessEqual(abs(out[:,:,:2]).max(),1)
    def test_crop_invalid_rejected(self):
        with self.assertRaises(ValueError):cropped_triangles(self.tri,[-1,2])


if __name__=='__main__':unittest.main(verbosity=2)