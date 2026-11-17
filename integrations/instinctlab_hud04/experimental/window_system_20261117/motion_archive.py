"""Additive local content-addressed archive. Not an off-site backup service."""
import datetime,hashlib,json
from pathlib import Path

def freeze_file(path,archive_root,*,role,source_sha256,model_sha256):
    path=Path(path);root=Path(archive_root)
    if role not in ('joint_temporal_fit_not_reference','unaccepted_full_body_candidate','diagnostic_not_motion'):raise ValueError('Explicit nonaccepted output role required')
    for digest in (source_sha256,model_sha256):
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Explicit source/model SHA256 required')
    if path.suffix!='.npz' or not path.is_file() or path.is_symlink():raise ValueError('Regular NPZ only')
    raw=path.read_bytes()
    if not 0<len(raw)<=32_000_000:raise ValueError('Bounded artifact size')
    digest=hashlib.sha256(raw).hexdigest();root.mkdir(parents=True,exist_ok=True)
    copy=root/(digest+'.npz');manifest=root/(digest+'.json')
    if copy.exists():
        if copy.is_symlink() or copy.read_bytes()!=raw:raise ValueError('Archive object conflict')
    else:
        with copy.open('xb') as stream:stream.write(raw)
    if hashlib.sha256(copy.read_bytes()).hexdigest()!=digest:raise ValueError('Backup readback failed')
    bindings=dict(schema_version=1,sha256=digest,bytes=len(raw),role=role,source_sha256=source_sha256,
        model_sha256=model_sha256,accepted_reference=False,training_ready=False,
        backup_scope='local_same_sandbox_NOT_independent_offsite')
    if manifest.exists():
        if manifest.is_symlink() or json.loads(manifest.read_text())!=bindings:raise ValueError('Archive metadata conflict')
    else:
        with manifest.open('x') as stream:json.dump(bindings,stream,indent=2)
    if json.loads(manifest.read_text())!=bindings:raise ValueError('Metadata readback failed')
    return dict(**bindings,backup_path=str(copy),manifest_path=str(manifest),readback_verified=True)