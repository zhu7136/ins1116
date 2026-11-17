"""Read-only remote verification plus additive content-addressed local backup."""
import base64,datetime,hashlib,json,os
from pathlib import Path
import requests

ROOT=Path(__file__).parent
BASE='2e2eee2832230586b5bca492a75063058be38acb'
PREFIX='integrations/instinctlab_hud04/'
BACKUP=Path('/workspace/hud04-backups/sha256')

def sha(raw):return hashlib.sha256(raw).hexdigest()
def gitsha(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
def main():
    dest=ROOT/'BACKUP_AUDIT.json'
    if dest.exists():raise FileExistsError('Audit exists; inspect rather than overwrite')
    s=requests.Session();s.headers['Authorization']='Bearer '+os.environ['github1117']
    def get(path):
        r=s.get('https://api.github.com/repos/zhu7136/ins1116'+path,timeout=20,allow_redirects=False)
        if r.status_code!=200:raise RuntimeError('GitHub HTTP '+str(r.status_code))
        return r.json()
    head=get('/git/ref/heads/aria/hud04-asset-camera-probe')['object']['sha']
    if head!=BASE:raise RuntimeError('Remote advanced; reconcile before changes')
    commit=get('/git/commits/'+head);tree=get('/git/trees/'+commit['tree']['sha']+'?recursive=1')
    if tree.get('truncated'):raise ValueError('Incomplete remote tree')
    by_sha={}
    for row in tree['tree']:
        if row['type']=='blob':by_sha.setdefault(row['sha'],[]).append(row['path'])
    files=[*sorted(Path('/workspace/hud04-original-drive-20261117').glob('*.npz')),
           *sorted(Path('/workspace/hud04-source-staging-20261117').glob('*.npz')),
           Path('/workspace/hud04-implementation-20261117/step1.joint_fit.npz')]
    if len(files)!=9:raise ValueError('Expected four original, four staged and one joint-only file')
    BACKUP.mkdir(parents=True,exist_ok=True);rows=[]
    for path in files:
        raw=path.read_bytes()
        if not 0<len(raw)<20_000_000:raise ValueError('Unexpected motion size')
        digest=sha(raw);target=BACKUP/(digest+'.npz')
        if target.exists():
            if target.read_bytes()!=raw:raise ValueError('Content-addressed backup collision')
        else:
            with target.open('xb') as f:f.write(raw)
        if sha(target.read_bytes())!=digest:raise ValueError('Backup readback')
        remote=by_sha.get(gitsha(raw),[])
        if remote:
            blob=get('/git/blobs/'+gitsha(raw))
            if base64.b64decode(blob['content'])!=raw:raise ValueError('Remote bytes differ')
        kind='joint_temporal_fit_not_reference' if path.name=='step1.joint_fit.npz' else ('source_staging_not_HUD04_reference' if 'source-stage' in path.name else 'original_G1_source_not_HUD04_reference')
        rows.append(dict(name=path.name,local_original=str(path),bytes=len(raw),sha256=digest,backup=str(target),
            backup_readback_verified=True,remote_identical_blob_paths=remote,remote_readback_verified=bool(remote),kind=kind,accepted_HUD04_reference=False))
    report=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_head=head,
        file_records=len(rows),unique_backup_objects=len({r['sha256'] for r in rows}),files=rows,
        local_only_count=sum(not r['remote_readback_verified'] for r in rows),
        all_remote_verified=all(r['remote_readback_verified'] for r in rows),
        notes=['Only these nine motion/source-staging files audited; other NPZ geometry/matrices are not motion outputs.',
            'No claim that all historical temporary files were backed up.',
            'Local backups share this sandbox fault domain; remote byte readback is separately stated per file.',
            'Content hash names and exclusive writes prevent accidental replacement by this tool.'],
        generated_motion_this_turn=False,gpu_submissions=0,accepted_reference=False,training_ready=False)
    with dest.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()