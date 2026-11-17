import unittest
import numpy as np
from independent_cylinder import cylinder_surface_bracket
from robot_shape_inventory import Shape
from independent_cylinder_refined import refined_bracket


class CylinderTests(unittest.TestCase):
    def cylinder(self):return Shape('cylinder','body','cylinder',np.eye(4),dict(radius=1.,length=2.))

    def test_caps_and_sides(self):
        for axis in range(3):
            tri=np.zeros((1,3,3));tri[:,:,axis]=3
            u,v=[x for x in range(3) if x!=axis];tri[0,:,u]=[-8,8,0];tri[0,:,v]=[-8,-8,8]
            r=cylinder_surface_bracket(self.cylinder(),np.eye(4),tri)
            self.assertEqual(r['status'],'converged');self.assertAlmostEqual(r['upper'],2.,places=7)

    def test_contained_triangle(self):
        r=cylinder_surface_bracket(self.cylinder(),np.eye(4),np.array([[[0,0,0],[.1,0,0],[0,.1,0]]]))
        self.assertEqual(r['status'],'converged');self.assertEqual(r['upper'],0.)

    def test_point_cases_64(self):
        rng=np.random.default_rng(14)
        for p in rng.uniform(-3,3,(64,3)):
            expected=np.hypot(max(0,np.linalg.norm(p[:2])-1),max(0,abs(p[2])-1))
            r=cylinder_surface_bracket(self.cylinder(),np.eye(4),np.array([[p,p,p]]))
            self.assertAlmostEqual(r['upper'],expected,places=12)

    def test_refinement_local_bound_not_global_bound_regression(self):
        from terrain_binding import ActualTerrain
        from robot_shape_inventory import load_shapes,TargetTasks
        terrain=ActualTerrain();tree,shapes,_=load_shapes();target=TargetTasks()
        q=np.array([(target.limits[n]['lower']+target.limits[n]['upper'])/2 for n in target.names])
        shape=next(s for s in shapes if s.identity=='left_hip_pitch_link#0');pose=tree.fk(q[None],target.names)[shape.link][0]
        r=refined_bracket(shape,pose,terrain.triangles)
        self.assertLess(r['lower'],r['upper']+1e-12)
        self.assertLess(r['upper']-r['lower'],1e-8)
        self.assertTrue(.034<r['upper']<.035)


if __name__=='__main__':unittest.main()