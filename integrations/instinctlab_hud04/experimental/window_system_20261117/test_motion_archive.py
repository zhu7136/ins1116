import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from motion_archive import freeze_file

class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.p=self.root/'synthetic-not-motion.npz'
        with self.p.open('xb') as f:np.savez(f,software_fixture=np.array([1,2,3]))
        self.kw=dict(role='diagnostic_not_motion',source_sha256='1'*64,model_sha256='2'*64)
    def save(self):return freeze_file(self.p,self.root/'archive',**self.kw)
    def test_roundtrip(self):
        r=self.save();self.assertEqual(Path(r['backup_path']).read_bytes(),self.p.read_bytes());self.assertTrue(r['readback_verified'])
        self.assertFalse(r['accepted_reference']);self.assertEqual(r['sha256'],hashlib.sha256(self.p.read_bytes()).hexdigest())
    def test_same_content_idempotent(self):self.assertEqual(self.save(),self.save())
    def test_two_versions_do_not_overwrite(self):
        a=self.save();old=Path(a['backup_path']).read_bytes()
        with self.p.open('wb') as f:np.savez(f,software_fixture=np.array([4,5,6]))
        b=self.save();self.assertNotEqual(a['backup_path'],b['backup_path']);self.assertEqual(Path(a['backup_path']).read_bytes(),old)
    def test_corrupt_existing_blob_rejected(self):
        r=self.save();Path(r['backup_path']).write_bytes(b'SYNTHETIC CORRUPTION')
        with self.assertRaises(ValueError):self.save()
    def test_conflicting_metadata_rejected(self):
        self.save();self.kw['source_sha256']='3'*64
        with self.assertRaises(ValueError):self.save()
    def test_reject_fake_accepted_role(self):
        self.kw['role']='accepted_reference'
        with self.assertRaises(ValueError):self.save()
    def test_bad_bindings_rejected(self):
        self.kw['model_sha256']='not-a-hash'
        with self.assertRaises(ValueError):self.save()
    def test_symlink_backup_rejected(self):
        archive=self.root/'archive';archive.mkdir();digest=hashlib.sha256(self.p.read_bytes()).hexdigest()
        (archive/(digest+'.npz')).symlink_to(self.p)
        with self.assertRaises(ValueError):self.save()

if __name__=='__main__':unittest.main(verbosity=2)