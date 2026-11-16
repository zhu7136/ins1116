"""Offline package tests. These are NOT Isaac simulation tests."""
import contextlib,io,json,subprocess,sys,tempfile,unittest
from types import SimpleNamespace
import worker
from pathlib import Path
from unittest.mock import patch
import preflight
from worker import deadline_reason,diagnostic_tail
from probe_checks import HEAD_ZERO_TRANSLATIONS,check_head_zero,require_finite_values
from evidence_transport import encode_evidence,decode_evidence,parse_evidence_log

ROOT=Path(__file__).resolve().parent

class PackageTests(unittest.TestCase):
    def test_warp_depth_probe_does_not_enable_rtx_camera(self):
        text=(ROOT/'probe.py').read_text()
        self.assertIn("os.environ['ENABLE_CAMERAS']='0'",text)
        self.assertIn('AppLauncher(headless=True,enable_cameras=False',text)
        self.assertIn('sim.step(render=False)',text)
        self.assertIn('camera.update(.005)',text)

    def test_diagnostic_tail_bounded(self):
        self.assertEqual(len(diagnostic_tail('x'*40000)),8192)

    def test_diagnostic_tail_redaction(self):
        text=diagnostic_tail('api_key=synthetic_secret https://example.org/private?sig=synthetic_secret password=synthetic_secret')
        self.assertNotIn('synthetic_secret',text)
        self.assertNotIn('https://',text)
        self.assertIn('[REDACTED]',text)

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

    def test_shutdown_deadline(self):
        self.assertIsNone(deadline_reason(60,True,29.9))
        self.assertEqual(deadline_reason(60,True,30),'simulator_shutdown_deadline')
        self.assertEqual(deadline_reason(540,True,30),'worker_child_wall_limit')

    def test_documented_shutdown_option(self):
        text=(ROOT/'probe.py').read_text()
        self.assertIn('app.close(wait_for_replicator=False)',text)
        self.assertIn("event('simulator_shutdown_started'",text)
        self.assertIn("event('simulator_shutdown_completed'",text)

    def test_evidence_preserved_on_lifecycle_failure(self):
        text=(ROOT/'worker.py').read_text()
        self.assertIn('if simulation_passed:',text)
        self.assertIn('probe_lifecycle_passed=passed',text)

    def test_sdk_concatenated_log_roundtrip(self):
        files=self.sample_evidence()
        text='[SDK] '+'x'*40000+''.join(json.dumps(e) for e in encode_evidence(files))
        self.assertEqual(decode_evidence(parse_evidence_log(text)),files)

    def test_log_parse_budget(self):
        with self.assertRaisesRegex(ValueError,'bounded'):
            parse_evidence_log('x'*(10*1024*1024+1))

    def synthetic_worker(self,hang):
        # CPU-only stub. Deliberately not Isaac data, and never a W&B experiment.
        root=Path(tempfile.mkdtemp(prefix='hud04-synthetic-wrapper-',dir='/workspace'))
        script="""import json,pathlib,sys,time
out=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir()
(out/'report.json').write_text(json.dumps(dict(probe_passed=True,synthetic_cpu_test=True)))
(out/'depth_samples.npz').write_bytes(b'synthetic-not-physics')
print(json.dumps(dict(stage='asset_initialized')),flush=True)
print(json.dumps(dict(stage='simulator_shutdown_started')),flush=True)
"""
        if hang:script+='time.sleep(5)\n'
        (root/'probe.py').write_text(script)
        buffer=io.StringIO()
        def deadline(age,initialized,shutdown_age=None):
            return 'simulator_shutdown_deadline' if shutdown_age is not None and shutdown_age>=.2 else None
        with patch.object(worker,'ROOT',root),patch.object(worker,'audit'),patch.object(worker,'deadline_reason',side_effect=deadline),contextlib.redirect_stdout(buffer):
            if hang:
                with self.assertRaises(SystemExit) as error:
                    worker.main(SimpleNamespace(execute_probe=True,isaac_launcher=sys.executable))
                self.assertEqual(error.exception.code,1)
            else:worker.main(SimpleNamespace(execute_probe=True,isaac_launcher=sys.executable))
        files=decode_evidence(parse_evidence_log(buffer.getvalue()))
        self.assertTrue(json.loads(files['report.json'])['synthetic_cpu_test'])
        return json.loads(files['worker_result.json'])

    def test_failed_shutdown_exports_checksums_but_still_fails(self):
        result=self.synthetic_worker(True)
        self.assertTrue(result['simulation_checks_passed'])
        self.assertFalse(result['probe_lifecycle_passed'])
        self.assertFalse(result['probe_passed'])
        self.assertEqual(result['stop_reason'],'simulator_shutdown_deadline')

    def test_clean_shutdown_exports_success(self):
        result=self.synthetic_worker(False)
        self.assertTrue(result['simulation_checks_passed'])
        self.assertTrue(result['probe_lifecycle_passed'])
        self.assertTrue(result['probe_passed'])
        self.assertIsNone(result['stop_reason'])

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