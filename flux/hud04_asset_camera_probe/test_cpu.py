"""Offline package tests. These are NOT Isaac simulation tests."""
import contextlib,io,json,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
import preflight
from worker import deadline_reason
from probe_checks import HEAD_ZERO_TRANSLATIONS,check_head_zero,require_finite_values
from evidence_transport import encode_evidence,decode_evidence

ROOT=Path(__file__).resolve().parent

class PackageTests(unittest.TestCase):
    def test_integrity_and_robot(self):
        with contextlib.redirect_stdout(io.StringIO()):report=preflight.audit(ROOT)
        self.assertEqual(report['body_dof'],29);self.assertEqual(report['mesh_count'],32)
        self.assertTrue(report['head_fixed']);self.assertFalse(report['Isaac_runtime_verified'])

    def test_tamper_rejected(self):
        with patch.object(preflight,'sha',return_value='invalid'):
            with self.assertRaisesRegex(ValueError,'Integrity mismatch'):preflight.audit(ROOT)

    def test_probe_default_no_gpu(self):
        proc=subprocess.run([sys.executable,str(ROOT/'probe.py')],capture_output=True,text=True,timeout=5,check=True)
        report=json.loads(proc.stdout);self.assertFalse(report['gpu_started']);self.assertEqual(report['training_updates'],0)

    def test_worker_default_no_gpu(self):
        proc=subprocess.run([sys.executable,str(ROOT/'worker.py')],capture_output=True,text=True,timeout=5,check=True)
        report=json.loads(proc.stdout);self.assertFalse(report['gpu_started']);self.assertFalse(report['launch_authorized'])

    def test_initialization_deadline(self):
        self.assertIsNone(deadline_reason(299,False))
        self.assertEqual(deadline_reason(300,False),'asset_initialization_deadline')
        self.assertIsNone(deadline_reason(300,True))

    def test_total_deadline(self):
        self.assertIsNone(deadline_reason(539,True))
        self.assertEqual(deadline_reason(540,True),'worker_child_wall_limit')
        self.assertEqual(deadline_reason(540,False),'worker_child_wall_limit')

    def test_no_full_training_import(self):
        text=(ROOT/'probe.py').read_text()
        self.assertNotIn('import instinct_rl',text)
        self.assertNotIn('gym.make(',text)
        self.assertIn('cfg.spawn.fix_base=True',text)
        self.assertIn('training_ready=False',text)

    def test_head_zero_pose_and_quaternion_sign(self):
        self.assertTrue(check_head_zero(HEAD_ZERO_TRANSLATIONS,[[1,0,0,0],[-1,0,0,0]]))

    def test_wrong_fixed_head_pose_rejected(self):
        with self.assertRaises(ValueError):check_head_zero([[0,0,0],[0,0,.0395]],[[1,0,0,0]]*2)
        with self.assertRaises(ValueError):check_head_zero(HEAD_ZERO_TRANSLATIONS,[[.9,.1,0,0],[1,0,0,0]])

    def test_nonfinite_head_rejected(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            check_head_zero([[float('nan'),0,0],[0,0,.0395]],[[1,0,0,0]]*2)

    def test_nonfinite_metrics_rejected(self):
        for values in [[],[float('nan')],[float('inf')]]:
            with self.assertRaises(ValueError):require_finite_values(values,'test')

    def sample_evidence(self):
        return {'report.json':b'{"synthetic_transport_test":true}',
                'depth_samples.npz':b'synthetic-not-physics','worker_result.json':b'{}'}

    def test_evidence_roundtrip(self):
        files=self.sample_evidence();self.assertEqual(decode_evidence(encode_evidence(files)),files)

    def test_evidence_missing_chunk_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing'):decode_evidence(encode_evidence(self.sample_evidence())[:-1])

    def test_evidence_duplicate_rejected(self):
        events=encode_evidence(self.sample_evidence())
        with self.assertRaisesRegex(ValueError,'duplicate'):decode_evidence(events+[events[-1]])

    def test_evidence_wrong_hash_rejected(self):
        events=encode_evidence(self.sample_evidence());events[0]['sha256']='invalid'
        with self.assertRaises(ValueError):decode_evidence(events)

    def test_evidence_unexpected_file_rejected(self):
        files=self.sample_evidence();files['anything_else']=b'notallowed'
        with self.assertRaises(ValueError):encode_evidence(files)

if __name__=='__main__':unittest.main(verbosity=2)