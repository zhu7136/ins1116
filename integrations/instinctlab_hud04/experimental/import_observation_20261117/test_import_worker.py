import contextlib,io,json,sys,unittest,uuid
from pathlib import Path
from bounded_import_runner import run_child,emit_evidence
from import_observation_probe import plan,verify_bundle

ROOT=Path(__file__).resolve().parent
class ImportWorkerTests(unittest.TestCase):
    def run_fixture(self,mode,seconds=2):
        out=ROOT/'test_outputs'/uuid.uuid4().hex
        result=run_child([sys.executable,str(ROOT/'fixture_child.py'),str(out),mode],out,seconds=seconds)
        self.assertTrue(result['direct_child_reaped']);self.assertFalse(result['training_ready'])
        return result,out
    def test_normal_fixture_is_not_physics_acceptance(self):
        r,out=self.run_fixture('normal');self.assertTrue(r['observation_completed'])
        self.assertFalse(r['platform_deadline_verified']);self.assertEqual(r['solver_calls'],0)
        self.assertTrue(json.loads((out/'runtime_report.json').read_text())['synthetic_fixture_only'])
    def test_noncooperative_sleep_stopped(self):
        r,_=self.run_fixture('sleep',.1);self.assertEqual(r['stop_reason'],'child_wall_deadline');self.assertFalse(r['observation_completed'])
    def test_report_before_hang_not_success(self):
        r,_=self.run_fixture('result_then_hang',.15);self.assertFalse(r['observation_completed']);self.assertIsNotNone(r['stop_reason'])
    def test_partial_phases_rejected(self):
        r,_=self.run_fixture('partial');self.assertFalse(r['observation_completed'])
    def test_promoted_acceptance_rejected(self):
        r,out=self.run_fixture('promoted');self.assertFalse(r['observation_completed'])
        with self.assertRaises(ValueError):emit_evidence(out)
    def test_nonzero_after_report_rejected(self):
        r,_=self.run_fixture('nonzero');self.assertFalse(r['observation_completed']);self.assertEqual(r['return_code'],2)
    def test_bad_json_rejected(self):
        r,_=self.run_fixture('bad_json');self.assertFalse(r['observation_completed'])
    def test_exception_stops_without_report(self):
        r,_=self.run_fixture('error');self.assertFalse(r['observation_completed']);self.assertFalse(r['local_report_present'])
    def test_log_bound_enforced(self):
        r,_=self.run_fixture('flood');self.assertLessEqual(r['log_bytes'],8*1024*1024);self.assertFalse(r['observation_completed'])
    def test_output_not_overwritten(self):
        _,out=self.run_fixture('normal')
        with self.assertRaises(FileExistsError):run_child([sys.executable,'-V'],out)
    def test_invalid_budgets_rejected(self):
        for value in [0,-1,301,float('inf'),float('nan'),True]:
            with self.assertRaises(ValueError):run_child([sys.executable,'-V'],ROOT/'not_created',seconds=value)
    def test_evidence_roundtrip_hashes_no_raw_log(self):
        import base64,hashlib,zipfile
        _,out=self.run_fixture('normal');buf=io.StringIO()
        with contextlib.redirect_stdout(buf):emit_evidence(out)
        rows=[json.loads(l) for l in buf.getvalue().splitlines()];m=rows[0];chunks=rows[1:]
        self.assertEqual(len(chunks),m['chunks']);data=base64.b64decode(''.join(x['data'] for x in chunks))
        self.assertEqual(hashlib.sha256(data).hexdigest(),m['sha256'])
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            self.assertNotIn('child.log',z.namelist());self.assertEqual(len(z.namelist()),5)
            for name in z.namelist():self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),m['file_sha256'][name])
    def test_plan_no_gpu_or_promotion(self):
        p=plan();self.assertFalse(p['gpu_started']);self.assertFalse(p['training_ready']);self.assertEqual(p['explicit_physics_steps'],0)
    def test_pinned_bundle_integrity_offline(self):
        import os
        repo=os.environ.get('HUD04_TEST_REPO_ROOT')
        if not repo:self.skipTest('HUD04_TEST_REPO_ROOT must name a recovered checkout for offline integrity test')
        asset,model=verify_bundle(Path(repo))
        self.assertTrue(model.is_file());self.assertTrue((asset/'manifest.json').is_file())
if __name__=='__main__':unittest.main()