import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from terrain_shape_query import TerrainShapeIndex
from robot_shape_inventory import Shape


class TerrainShapeTests(unittest.TestCase):
    def box(self):return Shape('box','body','box',np.eye(4),dict(size=[2,2,2]))

    def plane(self,axis=2,value=-3.):
        u,v=[i for i in range(3) if i!=axis];p=np.zeros((4,3));p[:,axis]=value
        p[:,u]=[-10,10,10,-10];p[:,v]=[-10,-10,10,10]
        return p[[[0,1,2],[0,2,3]]]

    def test_floor_and_sidewall(self):
        for axis in [0,1,2]:
            r=TerrainShapeIndex(self.plane(axis),terrain_to_world=np.eye(4)).closest(self.box(),np.eye(4))
            self.assertEqual(r['status'],'surface_distance_bracket_converged')
            self.assertAlmostEqual(r['distance_upper_bound'],2.,places=8)
            self.assertFalse(r['terrain_solid_containment_excluded'])

    def test_triangle_inside_shape(self):
        t=np.array([[[0,0,0],[.1,0,0],[0,.1,0]]])
        r=TerrainShapeIndex(t,terrain_to_world=np.eye(4)).closest(self.box(),np.eye(4))
        self.assertLess(r['distance_upper_bound'],1e-9)

    def test_rigid_equivariance(self):
        moved=np.eye(4);moved[:3,:3]=Rotation.from_rotvec([.3,-.1,.6]).as_matrix();moved[:3,3]=[4,2,-3]
        r=TerrainShapeIndex(self.plane(),terrain_to_world=moved).closest(self.box(),moved)
        self.assertAlmostEqual(r['distance_upper_bound'],2.,places=8)

    def test_query_budget_does_not_accept_unseen_triangles(self):
        triangle=np.array([[2,0,0],[0,2,0],[2,2,0]])
        index=TerrainShapeIndex(np.tile(triangle,(20,1,1)),terrain_to_world=np.eye(4))
        sphere=Shape('sphere','body','sphere',np.eye(4),dict(radius=.1))
        r=index.closest(sphere,np.eye(4),max_queries=1)
        self.assertTrue(r['triangle_budget_exhausted']);self.assertEqual(r['status'],'unresolved_surface_query')

    def test_near_surface_not_support_label(self):
        r=TerrainShapeIndex(self.plane(value=-1),terrain_to_world=np.eye(4)).closest(self.box(),np.eye(4))
        self.assertLess(r['distance_upper_bound'],1e-9);self.assertFalse(r['contact_or_support_label'])

    def test_transform_mandatory_and_invalid_inputs(self):
        with self.assertRaises(TypeError):TerrainShapeIndex(self.plane())
        with self.assertRaises(ValueError):TerrainShapeIndex([],terrain_to_world=np.eye(4))
        with self.assertRaises(ValueError):TerrainShapeIndex(self.plane(),terrain_to_world=np.zeros((4,4)))


if __name__=='__main__':unittest.main()