import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from terrain_binding import ActualTerrain,SourceCapsule,translation,layout_transform


class BindingTests(unittest.TestCase):
    def test_actual_input_inventory(self):
        t=ActualTerrain();self.assertEqual(len(t.triangles),8068);self.assertEqual(len(t.lateral_ids),2958)
        self.assertFalse(t.mesh.is_watertight);self.assertEqual(len(t.historical['independent_verification_witnesses']),24)
        t.verify_unchanged()

    def test_layout_chain_36_tiles(self):
        for row in range(6):
            for col in range(6):
                tf,pieces=layout_transform([9.,12.],.135,row=row,col=col,num_rows=6,num_cols=6)
                self.assertEqual(len(pieces),4)
                np.testing.assert_allclose(tf[:3,3],[(row-2.5)*9,(col-2.5)*12,-.135],atol=1e-12)

    def test_single_tile_not_function_shift(self):
        tf,_=layout_transform([9.,12.],.135,row=0,col=0,num_rows=1,num_cols=1)
        np.testing.assert_allclose(tf[:3,3],[0,0,-.135])

    def test_invalid_layout(self):
        for kwargs in [dict(row=-1,col=0,num_rows=1,num_cols=1),dict(row=1,col=0,num_rows=1,num_cols=1),dict(row=True,col=0,num_rows=1,num_cols=1)]:
            with self.assertRaises(ValueError):layout_transform([9.,12.],.135,**kwargs)

    def test_capsule_support_exact_128_directions(self):
        rng=np.random.default_rng(178);directions=rng.normal(size=(128,3));a=np.array([1,-2,.3]);b=np.array([2,1,4])
        capsule=SourceCapsule(a,b,.23);got=np.sum(capsule.support(directions,np.eye(4))*directions,axis=1)
        expected=np.maximum(directions@a,directions@b)+.23*np.linalg.norm(directions,axis=1)
        np.testing.assert_allclose(got,expected,atol=1e-12)

    def test_capsule_rigid_support(self):
        capsule=SourceCapsule([0,0,0],[0,0,2],.1);moved=translation([1,2,3]);moved[:3,:3]=Rotation.from_rotvec([.2,.4,-.1]).as_matrix()
        dirs=np.random.default_rng(123).normal(size=(48,3))
        expected=capsule.support(dirs@moved[:3,:3],np.eye(4))@moved[:3,:3].T+moved[:3,3]
        np.testing.assert_allclose(capsule.support(dirs,moved),expected,atol=1e-12)

    def test_fail_closed_subset_or_transform(self):
        t=ActualTerrain()
        with self.assertRaises(ValueError):t.index(subset='guess',transform=np.eye(4))
        with self.assertRaises(ValueError):t.index(subset='all',transform=np.zeros((4,4)))


if __name__=='__main__':unittest.main()