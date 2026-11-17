import unittest
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from convex_solid import Hull
from robot_shape_inventory import Shape,load_shapes,all_pairs
from support_distance import convex_distance


def cube():return np.array(list(product([-1.,1.],repeat=3)))


class ConvexSolidTests(unittest.TestCase):
    def setUp(self):self.hull=Hull.from_points(cube())

    def test_contained_segment_zero_solid_distance(self):
        r=self.hull.segment_clearance([0,0,0],[.2,0,0]);self.assertEqual(r['unsigned_solid_distance'],0.)
        self.assertTrue(r['overlap_or_within_tolerance']);self.assertFalse(r['penetration_depth_computed'])

    def test_contained_triangle(self):
        r=self.hull.triangle_clearance([[0,0,0],[.1,0,0],[0,.1,0]])
        self.assertEqual(r['unsigned_solid_distance'],0.)

    def test_crossing_triangle_no_inside_vertices(self):
        tri=np.array([[-3,-3,0],[3,-3,0],[0,3,0]])
        self.assertFalse(self.hull.contains(tri).any())
        self.assertTrue(self.hull.triangle_clearance(tri)['overlap_or_within_tolerance'])

    def test_crossing_segment(self):
        r=self.hull.segment_clearance([-3,0,0],[3,0,0]);self.assertEqual(r['unsigned_solid_distance'],0.)
        self.assertTrue(self.hull.contains([r['first_witness']])[0])

    def test_exact_parallel_segment(self):
        r=self.hull.segment_clearance([3,-.5,0],[3,.5,0]);self.assertAlmostEqual(r['unsigned_solid_distance'],2.)

    def test_sidewall_triangle_distance(self):
        r=self.hull.triangle_clearance([[3,-.5,-.5],[3,.5,-.5],[3,0,.5]])
        self.assertAlmostEqual(r['unsigned_solid_distance'],2.)
        self.assertAlmostEqual(np.linalg.norm(np.array(r['first_witness'])-r['second_witness']),2.)

    def test_degenerate_point_triangle(self):
        self.assertAlmostEqual(self.hull.triangle_clearance([[3,0,0]]*3)['unsigned_solid_distance'],2.)

    def test_cube_point_distances_independent_formula_96(self):
        for p in np.random.default_rng(7141).uniform(-3,3,(96,3)):
            exact=np.linalg.norm(np.maximum(abs(p)-1.,0.))
            self.assertAlmostEqual(self.hull.segment_clearance(p,p)['unsigned_solid_distance'],exact,places=10)

    def test_capsule_axis_inside(self):
        r=self.hull.capsule_overlap([0,0,0],[.1,0,0],.01)
        self.assertTrue(r['overlap_or_within_tolerance']);self.assertEqual(r['unsigned_solid_distance'],0.)

    def test_capsule_disjoint_and_touch(self):
        for radius,expected in [(.5,1.5),(2.,0.)]:
            self.assertAlmostEqual(self.hull.capsule_overlap([3,0,0],[3,.5,0],radius)['unsigned_solid_distance'],expected)

    def test_rigid_equivariance(self):
        rot=Rotation.from_rotvec([.3,.7,-.2]).as_matrix();shift=np.array([3,-4,2])
        moved=Hull.from_points(cube()@rot.T+shift);a,b=np.array([[3,-.3,0],[3,.3,0]])@rot.T+shift
        self.assertAlmostEqual(moved.segment_clearance(a,b)['unsigned_solid_distance'],2.,places=10)

    def test_invalid_hull_and_points(self):
        for p in [np.zeros((4,3)),[[0,0,0],[1,0,0],[0,1,0],[1,1,0]],[[float('nan')]*3]*4]:
            with self.assertRaises(ValueError):Hull.from_points(p)
        with self.assertRaises(ValueError):self.hull.segment_clearance([0,0],[1,0])
        with self.assertRaises(ValueError):self.hull.capsule_overlap([0,0,0],[1,0,0],-1)


class ShapeSupportTests(unittest.TestCase):
    def shape(self,kind,params):return Shape('test','test',kind,np.eye(4),params)

    def test_exact_cylinder_not_capsule(self):
        s=self.shape('cylinder',dict(radius=.4,length=2.));d=np.array([[1,0,1],[0,0,1],[0,2,0],[0,0,0]])
        p=s.support(d,np.eye(4));expected=.4*np.linalg.norm(d[:,:2],axis=1)+abs(d[:,2])
        np.testing.assert_allclose(np.sum(p*d,axis=1),expected)
        self.assertNotAlmostEqual(float(p[0]@d[0]),1.+.4*np.sqrt(2))

    def test_sphere_box_support(self):
        d=np.random.default_rng(2).normal(size=(64,3))
        for kind,params,expected in [('sphere',dict(radius=.5),.5*np.linalg.norm(d,axis=1)),
            ('box',dict(size=[2,4,6]),abs(d)@np.array([1,2,3]))]:
            p=self.shape(kind,params).support(d,np.eye(4));np.testing.assert_allclose(np.sum(p*d,axis=1),expected)

    def test_transformed_support_aabb(self):
        world=np.eye(4);world[:3,:3]=Rotation.from_rotvec([.2,-.4,.3]).as_matrix();world[:3,3]=[2,3,-1]
        s=self.shape('box',dict(size=[2,2,2]));lo,hi=s.aabb(world);p=cube()@world[:3,:3].T+world[:3,3]
        np.testing.assert_allclose(lo,p.min(axis=0));np.testing.assert_allclose(hi,p.max(axis=0))

    def test_mesh_support(self):
        s=Shape('mesh','test','mesh',np.eye(4),{},cube());d=np.random.default_rng(8).normal(size=(80,3))
        np.testing.assert_allclose(np.sum(s.support(d,np.eye(4))*d,axis=1),abs(d).sum(axis=1))

    def test_invalid_direction_or_pose(self):
        s=self.shape('sphere',dict(radius=1))
        with self.assertRaises(ValueError):s.support([[float('nan'),0,0]],np.eye(4))
        with self.assertRaises(ValueError):s.support([[1,0,0]],np.zeros((4,4)))

    def test_actual_inventory_no_pairs_omitted(self):
        tree,shapes,policy=load_shapes();self.assertEqual(len(shapes),43)
        self.assertFalse(policy['replace_cylinders_with_capsules']);self.assertTrue(policy['self_collision'])
        self.assertIn('head_pitch_link',{s.link for s in shapes})
        world={n:p[0] for n,p in tree.fk(np.zeros((1,29)),tree.movable).items()}
        rows,_=all_pairs(tree,shapes,world);self.assertEqual(len(rows),903)
        self.assertEqual(len({(r['first'],r['second']) for r in rows}),903)
        self.assertFalse(any(r['automatically_excluded'] for r in rows))
        self.assertEqual(len({tuple(sorted([r['first'].split('#')[0],r['second'].split('#')[0]])) for r in rows if not r['same_link']}),171)


class SupportDistanceTests(unittest.TestCase):
    def pose(self,shift):
        t=np.eye(4);t[:3,3]=shift;return t

    def shape(self,kind,params):return Shape('test','test',kind,np.eye(4),params)

    def test_boxes_exact_gap_and_witness(self):
        box=self.shape('box',dict(size=[2,2,2]))
        r=convex_distance(box,np.eye(4),box,self.pose([4,5,0]));self.assertTrue(r['converged'])
        exact=np.sqrt(13);self.assertAlmostEqual(r['distance_upper_bound'],exact,places=8)
        self.assertLessEqual(r['distance_lower_bound'],exact+1e-10)
        self.assertAlmostEqual(np.linalg.norm(np.array(r['first_witness'])-r['second_witness']),r['distance_upper_bound'])

    def test_contained_sphere(self):
        box=self.shape('box',dict(size=[4,4,4]));sphere=self.shape('sphere',dict(radius=.1))
        r=convex_distance(box,np.eye(4),sphere,np.eye(4));self.assertTrue(r['converged'])
        self.assertEqual(r['status'],'within_numerical_tolerance')

    def test_sphere_distances_32_analytic(self):
        sphere=self.shape('sphere',dict(radius=.5))
        for shift in np.random.default_rng(34).normal(size=(32,3))*3:
            exact=max(0.,np.linalg.norm(shift)-1);r=convex_distance(sphere,np.eye(4),sphere,self.pose(shift))
            self.assertTrue(r['converged']);self.assertLessEqual(r['distance_lower_bound'],exact+1e-10)
            self.assertGreaterEqual(r['distance_upper_bound']+1e-10,exact)
            self.assertLess(r['distance_upper_bound']-r['distance_lower_bound'],1.1e-9)

    def test_exact_cylinder_caps_and_sides(self):
        c=self.shape('cylinder',dict(radius=.5,length=2.))
        for shift,distance in [([0,0,4],2.),([3,0,0],2.),([3,0,4],np.sqrt(8)),([.1,0,0],0.)]:
            r=convex_distance(c,np.eye(4),c,self.pose(shift));self.assertTrue(r['converged'])
            self.assertAlmostEqual(r['distance_upper_bound'],distance,places=8)

    def test_rotated_equivariance(self):
        box=self.shape('box',dict(size=[2,1,3]));sphere=self.shape('sphere',dict(radius=.3))
        a=np.eye(4);b=self.pose([3,4,2]);r=convex_distance(box,a,sphere,b)
        common=self.pose([4,2,-1]);common[:3,:3]=Rotation.from_rotvec([.3,.2,-.5]).as_matrix()
        moved=convex_distance(box,common@a,sphere,common@b)
        self.assertTrue(r['converged'] and moved['converged'])
        self.assertAlmostEqual(r['distance_upper_bound'],moved['distance_upper_bound'],places=8)

    def test_nonconvergence_is_unresolved(self):
        box=self.shape('box',dict(size=[2,1,3]));sphere=self.shape('sphere',dict(radius=.3))
        r=convex_distance(box,np.eye(4),sphere,self.pose([3,4,2]),iterations=1)
        self.assertFalse(r['converged']);self.assertEqual(r['status'],'unresolved_iteration_or_stagnation')

    def test_invalid_iteration_budget(self):
        b=self.shape('box',dict(size=[1,1,1]))
        with self.assertRaises(ValueError):convex_distance(b,np.eye(4),b,np.eye(4),iterations=129)


if __name__=='__main__':unittest.main()