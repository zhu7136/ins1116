"""Analytic CPU geometry tests, not simulation or motion acceptance."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import terrain_surface_geometry as G


class SegmentTriangleTests(unittest.TestCase):
    def setUp(self):
        self.tri=np.array([[[0.,0,0],[2,0,0],[0,2,0]]])

    def distance(self,a,b,tri=None):
        d,ps,pt=G.segment_triangles(a,b,self.tri if tri is None else tri)
        np.testing.assert_allclose(np.linalg.norm(ps-pt,axis=1),d,atol=1e-12)
        return d[0]

    def test_crossing_face_interior(self):self.assertAlmostEqual(self.distance([.2,.2,-1],[.2,.2,1]),0)
    def test_parallel_above_face(self):self.assertAlmostEqual(self.distance([.2,.2,1],[.8,.2,1]),1)
    def test_endpoint_face(self):self.assertAlmostEqual(self.distance([.2,.2,1],[.2,.2,2]),1)
    def test_zero_length_segment(self):self.assertAlmostEqual(self.distance([.2,.2,1],[.2,.2,1]),1)
    def test_vertex_closest(self):self.assertAlmostEqual(self.distance([-1,-1,0],[-2,-2,0]),np.sqrt(2))
    def test_coplanar_crossing(self):self.assertAlmostEqual(self.distance([-.5,.5,0],[1,.5,0]),0)
    def test_edge_interior_to_segment_interior(self):
        self.assertAlmostEqual(self.distance([.5,-1,-1],[.5,-1,1]),1)
    def test_degenerate_line_triangle(self):
        tri=np.array([[[0.,0,0],[1,0,0],[2,0,0]]])
        self.assertAlmostEqual(self.distance([.5,-1,-1],[.5,-1,1],tri),1)
    def test_degenerate_point_triangle(self):
        self.assertAlmostEqual(self.distance([0,0,1],[0,0,2],np.zeros((1,3,3))),1)
    def test_lateral_wall_detected(self):
        wall=np.array([[[0.,0,0],[0,2,0],[0,0,2]]])
        # A world-down XY envelope is degenerate, but true surface distance is .01.
        self.assertAlmostEqual(self.distance([.01,.2,.2],[.01,.2,.8],wall),.01)
    def test_rigid_transform_invariance(self):
        rng=np.random.default_rng(17)
        for _ in range(25):
            a,b=rng.normal(size=(2,3));tri=rng.normal(size=(1,3,3))
            rot=Rotation.random(random_state=rng).as_matrix();shift=rng.normal(size=3)
            self.assertAlmostEqual(self.distance(a,b,tri),self.distance(a@rot.T+shift,b@rot.T+shift,tri@rot.T+shift),places=10)
    def test_broadphase_matches_exhaustive(self):
        rng=np.random.default_rng(18);tri=rng.normal(size=(100,3,3));index=G.SurfaceIndex(tri)
        for _ in range(20):
            a,b=rng.normal(size=(2,3))
            expected=G.segment_triangles(a,b,tri)[0].min()
            self.assertAlmostEqual(index.closest(a,b)[0],expected,places=12)
    def test_nans_rejected(self):
        with self.assertRaises(ValueError):G.segment_triangles([np.nan,0,0],[0,0,0],self.tri)
    def test_empty_triangles_rejected(self):
        with self.assertRaises(ValueError):G.SurfaceIndex(np.empty((0,3,3)))
    def test_endpoint_reversal(self):
        self.assertEqual(self.distance([3,0,1],[3,0,2]),self.distance([3,0,2],[3,0,1]))


class IntervalTests(unittest.TestCase):
    def test_never_bridges_windows(self):
        self.assertEqual(G.candidate_intervals([1,2,5,6],[0,0,0,0],.02),[[1,2],[5,6]])
    def test_threshold_and_breaks(self):
        self.assertEqual(G.candidate_intervals([1,2,3,4],[0,.02,.03,-.02],.02),[[1,2],[4,4]])
    def test_no_candidates(self):self.assertEqual(G.candidate_intervals([1,2],[1,2],.02),[])
    def test_unordered_rejected(self):
        with self.assertRaises(ValueError):G.candidate_intervals([2,1],[0,0],.02)


class IndependentKernelTests(unittest.TestCase):
    def check(self,triangles,a,b,expected):
        from verify_processed_terrain_surfaces import independent_distance
        triangles=np.asarray(triangles,float);a=np.asarray(a,float);b=np.asarray(b,float)
        result=independent_distance(triangles,a,b)
        self.assertAlmostEqual(result,expected,places=8)
        self.assertAlmostEqual(result,G.SurfaceIndex(triangles).closest(a,b)[0],places=8)

    def test_independent_parallel(self):
        self.check([[[0,0,0],[2,0,0],[0,2,0]]],[.1,.1,.4],[.5,.1,.4],.4)

    def test_independent_face_crossing(self):
        self.check([[[0,0,0],[2,0,0],[0,2,0]]],[.2,.2,-1],[.2,.2,1],0)

    def test_independent_lateral(self):
        self.check([[[0,0,0],[0,2,0],[0,0,2]]],[.3,.2,.2],[.3,.5,.5],.3)

    def test_independent_global_triangle_choice(self):
        self.check([[[0,0,8],[2,0,8],[0,2,8]],[[0,0,0],[2,0,0],[0,2,0]]],
            [.1,.1,.4],[.5,.1,.4],.4)


class CropTests(unittest.TestCase):
    def test_crop_border_and_no_source_mutation(self):
        import trimesh
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            path=Path(d)/'synthetic.stl'
            mesh=trimesh.Trimesh(vertices=[[-2,-2,3],[2,-2,3],[2,2,3],[-2,2,3]],faces=[[0,1,2],[0,2,3]],process=False)
            mesh.export(path);raw=path.read_bytes()
            cropped,shift,report=G.processed_terrain(path,[2,2])
            self.assertTrue(np.all(np.abs(cropped.vertices[:,:2])<=1+1e-12))
            np.testing.assert_allclose(shift,[1,1,-3])
            self.assertEqual(report['border_height_raw'],3)
            self.assertFalse(report['border_fallback_used'])
            self.assertEqual(path.read_bytes(),raw)
    def test_missing_border_reported(self):
        import trimesh
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            path=Path(d)/'synthetic.stl'
            trimesh.Trimesh(vertices=[[0,0,4],[.1,0,4],[0,.1,4]],faces=[[0,1,2]]).export(path)
            _,_,report=G.processed_terrain(path,[2,2])
            self.assertTrue(report['border_fallback_used'])
    def test_invalid_tile_rejected(self):
        with self.assertRaises(ValueError):G.processed_terrain('missing',[-1,2])


if __name__=='__main__':unittest.main()