import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import full_sequence_003 as F
from full_sequence_003 import windows,root_smooth_rows,save,load


class FullSequenceTests(unittest.TestCase):
    def test_failed_window_must_block_resume(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            p=Path(d);(p/'INTENT.json').write_text('{}');w=p/'WINDOW_00';w.mkdir()
            (w/'SUPERVISION.json').write_text(json.dumps(dict(child_exit=1,child_reaped=True,timed_out=False)))
            with patch.object(F,'OUT',p):
                with self.assertRaisesRegex(ValueError,'Failed prior window'):F.batch(1)

    def test_exact_native_coverage(self):
        w=windows();self.assertEqual(len(w),17)
        self.assertEqual(sorted(set(f for a,b in w for f in range(a,b+1))),list(range(997)))
        self.assertTrue(all(b-a==60 for a,b in w))

    def test_root_rows_with_fixed_neighbours(self):
        rng=np.random.default_rng(86);r=np.tile(np.eye(4),(997,1,1));r[:,:3,3]=rng.normal(size=(997,3))
        for a,b in [(0,60),(400,460),(936,996)]:
            m,v=root_smooth_rows(r,a,b);delta=rng.normal(size=(61,35))*1e-5
            rr=r.copy();rr[a:b+1,:3,3]+=delta[:,:3]
            _,vv=root_smooth_rows(rr,a,b)
            np.testing.assert_allclose(vv-v,m@delta.ravel(),atol=3e-15)

    def test_fail_closed_export_load(self):
        r=np.tile(np.eye(4),(997,1,1));q=np.zeros((997,29));names=[str(i) for i in range(29)]
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            p=Path(d)/'motion.npz';save(p,q,r,names)
            a,b=load(p,names);np.testing.assert_array_equal(a,q);np.testing.assert_array_equal(b,r)
            with self.assertRaises(FileExistsError):save(p,q,r,names)
            with self.assertRaises(ValueError):load(p,list(reversed(names)))


if __name__=='__main__':unittest.main()