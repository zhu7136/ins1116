"""CPU fixtures only. Not robot/physics acceptance."""
import contextlib,io,json,os,sys,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from failure_evidence import emit_failure_evidence,decode_failure_records,check_files,strict_json

ROOT=Path(__file__).parent;COMMIT='a'*40
class FailureEvidenceTests(unittest.TestCase):
    def fixture(self,partial=False):
        out=ROOT/'test_outputs'/uuid.uuid4().hex;out.mkdir(parents=True)
        data={'worker_result.json':dict(stage='bounded_import_worker_result',observation_completed=False,direct_child_reaped=True,return_code=-11,accepted_reference=False,training_ready=False)}
        if not partial:
            data['runtime_report.json']=dict(stage='hud04_import_observed',source_commit=COMMIT,accepted_reference=False,training_ready=False)
            for phase in ['post_import','post_clone','post_physics_init']:
                data[phase+'.json']=dict(stage_observation_kind='composed_USD_metadata_ONLY',runtime_phase=phase,runtime_execution=True,colliders=[],joints=[],articulations=[],global_filtered_pairs=[],collision_groups=[],accepted_reference=False,training_ready=False)
        for n,d in data.items():(out/n).write_text(json.dumps(d))
        (out/'child.log').write_text('This raw log must not be emitted')
        return out,data
    def records(self,out):
        s=io.StringIO()
        with contextlib.redirect_stdout(s):emit_failure_evidence(out,COMMIT)
        return [json.loads(l) for l in s.getvalue().splitlines()]
    def test_complete_failure_roundtrip_remains_failure(self):
        out,data=self.fixture();r=self.records(out);d,files=decode_failure_records(r,COMMIT)
        self.assertEqual(d,data);self.assertNotIn('child.log',files)
        self.assertIs(d['worker_result.json']['observation_completed'],False)
        self.assertEqual(d['worker_result.json']['return_code'],-11)
    def test_partial_failure_explicit_missing(self):
        out,_=self.fixture(True);r=self.records(out);d,_=decode_failure_records(r,COMMIT)
        self.assertEqual(len(r[0]['missing_files']),4);self.assertEqual(set(d),{'worker_result.json'})
    def test_worker_still_exits_nonzero_and_no_success_envelope(self):
        import worker
        out,data=self.fixture();s=io.StringIO()
        args=SimpleNamespace(execute=True,repo_root=str(ROOT),expected_commit=COMMIT,output=str(out),isaac_launcher=sys.executable)
        with patch.object(worker.subprocess,'check_output',return_value=COMMIT+'\n'),patch.object(worker,'verify_bundle'),patch.object(worker,'run_child',return_value=data['worker_result.json']),contextlib.redirect_stdout(s):
            with self.assertRaises(SystemExit) as e:worker.main(args)
        self.assertEqual(e.exception.code,1);self.assertNotIn('hud04_import_evidence_manifest',s.getvalue())
        self.assertIn('hud04_import_failure_manifest',s.getvalue())
    def test_malformed_or_nonfinite_rejected(self):
        for value in ['{"x":NaN}','{"x":Infinity}','{"x":1e999}','{"x":1,"x":2}']:
            with self.assertRaises(ValueError):strict_json(value)
    def test_nested_acceptance_claim_rejected(self):
        with self.assertRaises(ValueError):strict_json('{"nested":{"training_ready":true}}')
    def test_success_worker_not_exported_as_failure(self):
        out,data=self.fixture();data['worker_result.json']['observation_completed']=True
        (out/'worker_result.json').write_text(json.dumps(data['worker_result.json']))
        with self.assertRaises(ValueError):self.records(out)
    def test_wrong_source_commit_rejected(self):
        out,_=self.fixture();r=self.records(out)
        with self.assertRaises(ValueError):decode_failure_records(r,'b'*40)
    def test_missing_duplicate_chunks_rejected(self):
        out,_=self.fixture();r=self.records(out)
        for changed in [r[:-1],r+[r[-1]],r+[r[0]]]:
            with self.assertRaises(ValueError):decode_failure_records(changed,COMMIT)
    def test_archive_corruption_rejected(self):
        out,_=self.fixture();r=self.records(out);r[1]['data']='AAAA'+r[1]['data'][4:]
        with self.assertRaises(ValueError):decode_failure_records(r,COMMIT)
    def test_individual_file_hash_rejected(self):
        out,_=self.fixture();r=self.records(out);r[0]['file_sha256']['worker_result.json']='0'*64
        with self.assertRaises(ValueError):decode_failure_records(r,COMMIT)
    def test_expanded_budget_and_allowlist(self):
        for files in [{'../bad.json':b'{}'},{'worker_result.json':b' '*1048577}]:
            with self.assertRaises(ValueError):check_files(files,COMMIT)
    def test_manifest_budget_rejected_before_decode(self):
        out,_=self.fixture();r=self.records(out);r[0]['zip_bytes']=1024*1024
        with self.assertRaises(ValueError):decode_failure_records(r,COMMIT)
    def test_secret_rejected_no_partial_output(self):
        out,_=self.fixture();p=out/'runtime_report.json';d=json.loads(p.read_text());d['note']='fixtureCredential123';p.write_text(json.dumps(d));s=io.StringIO()
        with patch.dict(os.environ,{'FLUX_API_KEY':'fixtureCredential123'}),contextlib.redirect_stdout(s):
            with self.assertRaises(ValueError):emit_failure_evidence(out,COMMIT)
        self.assertEqual(s.getvalue(),'')
    def test_symlink_rejected(self):
        out,_=self.fixture(True);(out/'runtime_report.json').symlink_to(out/'worker_result.json')
        with self.assertRaises(ValueError):self.records(out)
    def test_file_budget_rejected_before_read(self):
        out,_=self.fixture(True);(out/'runtime_report.json').write_bytes(b' '*32769)
        with self.assertRaises(ValueError):self.records(out)
    def test_no_runtime_still_cannot_pass_success_emitter(self):
        from bounded_import_runner import emit_evidence
        out,_=self.fixture()
        with self.assertRaises(ValueError):emit_evidence(out)
if __name__=='__main__':unittest.main()