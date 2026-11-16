"""Standard-library offline integrity checks; never imports Isaac or CUDA."""
import ast,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('Path escapes bundle')
        if sha(path)!=digest:raise ValueError('Integrity mismatch:'+name)
    urdf=root/'humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
    tree=ET.parse(urdf).getroot();joints=tree.findall('joint')
    moving=[j.get('name') for j in joints if j.get('type') in ['revolute','continuous','prismatic']]
    assert len(moving)==29 and not any('head' in n for n in moving)
    for name in ['head_yaw_joint','head_pitch_joint']:
        item=next(j for j in joints if j.get('name')==name);assert item.get('type')=='fixed'
    from probe_checks import HEAD_ZERO_TRANSLATIONS
    for name,expected in zip(['head_yaw_joint','head_pitch_joint'],HEAD_ZERO_TRANSLATIONS):
        item=next(j for j in joints if j.get('name')==name);origin=item.find('origin')
        assert tuple(map(float,origin.get('xyz').split()))==expected
        assert tuple(map(float,origin.get('rpy').split()))==(0.,0.,0.)
    meshes=set()
    for node in tree.findall('.//mesh'):
        mesh=(urdf.parent/node.get('filename')).resolve()
        assert mesh.is_relative_to(root) and mesh.is_file()
        assert str(mesh.relative_to(root)) in manifest['files'];meshes.add(mesh)
    for name in ['probe.py','worker.py','preflight.py','asset_config.py','camera_mount.py','probe_checks.py','evidence_transport.py']:
        ast.parse((root/name).read_text(),filename=name)
    report=dict(passed=True,body_dof=len(moving),head_fixed=True,mesh_count=len(meshes),
        mesh_bytes=sum(p.stat().st_size for p in meshes),manifest_files=len(manifest['files']),
        gpu_initialized=False,Isaac_runtime_verified=False,training_ready=False)
    print(json.dumps(dict(stage='cpu_preflight',**report)),flush=True)
    return report

if __name__=='__main__':audit(Path(__file__).resolve().parent)