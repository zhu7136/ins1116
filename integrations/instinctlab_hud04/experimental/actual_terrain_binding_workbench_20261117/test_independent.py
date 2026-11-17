import unittest
import numpy as np
from independent_primitives import distance
from robot_shape_inventory import Shape


class IndependentTests(unittest.TestCase):
    def box(self):return Shape('test','link','box',np.eye(4),dict(size=[2,2,2]))

    def test_triangle_contained(self):
        t=np.array([[[0,0,0],[.1,0,0],[0,.1,0]]])
        self.assertEqual(distance(self.box(),np.eye(4),t),0.)

    def test_crossing_without_inside_vertices(self):
        t=np.array([[[-8,0,0],[8,-8,0],[8,8,0]]])
        self.assertEqual(distance(self.box(),np.eye(4),t),0.)

    def test_sphere_planes(self):
        t=np.array([[[-8,-8,3],[8,-8,3],[0,8,3]]])
        s=Shape('sphere','link','sphere',np.eye(4),dict(radius=.2))
        self.assertAlmostEqual(distance(s,np.eye(4),t),2.8)

    def test_box_plane(self):
        t=np.array([[[-8,-8,3],[8,-8,3],[0,8,3]]])
        self.assertAlmostEqual(distance(self.box(),np.eye(4),t),2.)

    def test_cube_point_distances_96(self):
        rng=np.random.default_rng(22)
        for p in rng.uniform(-3,3,(96,3)):
            t=np.array([[p,p,p]])
            expected=float(np.linalg.norm(np.maximum(np.abs(p)-1,0)))
            self.assertAlmostEqual(distance(self.box(),np.eye(4),t),expected,places=11)


if __name__=='__main__':unittest.main()