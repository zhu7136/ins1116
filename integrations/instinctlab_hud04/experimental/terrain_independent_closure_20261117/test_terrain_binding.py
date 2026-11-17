import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from terrain_binding import ActualTerrain
from robot_shape_inventory import Shape
from independent_primitives import sphere_distance,box_distance,cylinder_bracket,project_cylinder
from remaining_geometry_worker import hand_bracket


class Gate:
    def step(self):pass


def shape(kind,**parameters):return Shape('test','test',kind,np.eye(4),parameters)


def plane(axis,value):
    other=[i for i in range(3) if i!=axis];p=np.zeros((3,3));p[:,axis]=value
    p[:,other]=[[-10,-10],[10,-10],[0,10]]
    return p[None]


class IndependentGeometryTests(unittest.TestCase):
    def test_projection_side_cap_and_inside(self):
        p=project_cylinder(np.array([[2.,0,2],[0,0,0],[0,0,-3]]),1,1)
        np.testing.assert_allclose(p,[[1,0,1],[0,0,0],[0,0,-1]])

    def test_sphere_plane_all_triangles(self):
        t=np.concatenate([plane(2,4),plane(2,2)])
        self.assertAlmostEqual(sphere_distance(shape('sphere',radius=.5),np.eye(4),t),1.5)

    def test_sphere_surface_intersection_zero(self):
        self.assertEqual(sphere_distance(shape('sphere',radius=1),np.eye(4),plane(2,0)),0)

    def test_box_plane(self):
        self.assertAlmostEqual(box_distance(shape('box',size=[2,2,2]),np.eye(4),plane(0,3)),2)

    def test_box_face_crossing_without_contained_vertices(self):
        self.assertEqual(box_distance(shape('box',size=[2,2,2]),np.eye(4),plane(2,0)),0)

    def test_box_contains_entire_triangle(self):
        self.assertEqual(box_distance(shape('box',size=[2,2,2]),np.eye(4),np.array([[[0,0,0],[.1,0,0],[0,.1,0]]])),0)

    def test_box_degenerate_point_analytic_32_cases(self):
        rng=np.random.default_rng(37);box=shape('box',size=[2,4,6]);half=np.array([1,2,3])
        for p in rng.uniform(-5,5,(32,3)):
            t=np.tile(p,(1,3,1));expected=np.linalg.norm(np.maximum(0,abs(p)-half))
            self.assertAlmostEqual(box_distance(box,np.eye(4),t),expected,places=10)

    def test_cylinder_side_and_cap(self):
        cylinder=shape('cylinder',radius=1.,length=2.)
        for axis,value,expected in [(0,3,2),(2,2,1),(2,.5,0)]:
            r=cylinder_bracket(cylinder,np.eye(4),plane(axis,value),include_face_plane_lower=True)
            self.assertEqual(r['status'],'converged');self.assertLessEqual(r['lower'],expected+1e-8);self.assertGreaterEqual(r['upper'],expected-1e-8)

    def test_cylinder_diagonal_point(self):
        r=cylinder_bracket(shape('cylinder',radius=1.,length=2.),np.eye(4),np.tile([2.,0,2],(1,3,1)))
        self.assertLessEqual(r['lower'],np.sqrt(2)+1e-8);self.assertGreaterEqual(r['upper'],np.sqrt(2)-1e-8)

    def test_local_lower_may_exceed_global_upper(self):
        r=cylinder_bracket(shape('cylinder',radius=1.,length=2.),np.eye(4),np.concatenate([plane(2,2),plane(2,8)]),include_face_plane_lower=True)
        self.assertEqual(r['status'],'converged');self.assertAlmostEqual(r['upper'],1,places=7)

    def test_primitive_rigid_equivariance(self):
        transform=np.eye(4);transform[:3,:3]=Rotation.from_rotvec([.2,-.5,.1]).as_matrix();transform[:3,3]=[2,4,-1]
        t=plane(2,3);shifted=t@transform[:3,:3].T+transform[:3,3]
        for s,f in [(shape('sphere',radius=1),sphere_distance),(shape('box',size=[2,2,2]),box_distance)]:
            self.assertAlmostEqual(f(s,np.eye(4),t),f(s,transform,shifted),places=9)
        a=cylinder_bracket(shape('cylinder',radius=1.,length=2.),np.eye(4),t,include_face_plane_lower=True)
        b=cylinder_bracket(shape('cylinder',radius=1.,length=2.),transform,shifted,include_face_plane_lower=True)
        self.assertAlmostEqual(a['upper'],b['upper'],places=8)

    def test_independent_queries_do_not_call_gjk(self):
        with patch('support_distance.convex_distance',side_effect=AssertionError('GJK forbidden')):
            sphere_distance(shape('sphere',radius=1),np.eye(4),plane(2,3))
            box_distance(shape('box',size=[2,2,2]),np.eye(4),plane(2,3))
            cylinder_bracket(shape('cylinder',radius=1.,length=2.),np.eye(4),plane(2,3),include_face_plane_lower=True)

    def test_hand_hull_contains_triangle(self):
        import itertools
        hand=Shape('hand','hand','mesh',np.eye(4),{},np.array(list(itertools.product([-1.,1.],repeat=3))))
        r=hand_bracket(hand,np.eye(4),np.array([[[0,0,0],[.1,0,0],[0,.1,0]]]),Gate())
        self.assertEqual(r['status'],'converged');self.assertEqual(r['upper'],0)


class ActualBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.terrain=ActualTerrain()

    def test_actual_crop_counts(self):
        self.assertEqual(len(self.terrain.triangles),8068);self.assertEqual(len(self.terrain.lateral_ids),2958)

    def test_nonflat_border_preserved_not_flattened(self):
        stats=self.terrain.stats;self.assertGreater(stats['border_z_std_raw'],.19)
        self.assertAlmostEqual(self.terrain.shift[2],-.13514180794761088,places=12)

    def test_shift_is_explicit(self):
        with self.assertRaises(ValueError):self.terrain.transform('automatic')
        np.testing.assert_array_equal(self.terrain.transform('raw'),np.eye(4))

    def test_side_faces_have_original_face_identity(self):
        for i in [0,100,len(self.terrain.lateral_ids)-1]:
            original=self.terrain.original_triangle_id('lateral',i)
            np.testing.assert_array_equal(self.terrain.indices['raw','lateral'].triangles[i],self.terrain.triangles[original])

    def test_no_random_wall_mixin(self):
        self.assertTrue(self.terrain.config_proof['wall_mixin_not_inherited'])
        self.assertFalse(self.terrain.evidence()['random_decorator_walls_included'])

    def test_acceptance_stays_false(self):
        for key in ['source_root_or_units_inferred','actual_terrain_runtime_cooking_verified','accepted_reference','training_ready']:
            self.assertIs(self.terrain.evidence()[key],False)


if __name__=='__main__':unittest.main(verbosity=2)