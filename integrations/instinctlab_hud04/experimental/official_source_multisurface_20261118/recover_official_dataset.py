"""Bounded public release reads; no credentials, extraction, code execution or mutation."""
from pathlib import Path,PurePosixPath
import hashlib,io,json,stat,zipfile
import requests,numpy as np
ROOT=Path(__file__).parent/'OFFICIAL_RELEASE'
BASE='https://github.com/project-instinct/InstinctMJ/releases/download/supermimic-v0.1/'
ZIP='20251116_50cm_kneeClimbStep1.zip'
def sha(b):return hashlib.sha256(b).hexdigest()
def download(name,cap):
    p=ROOT/name
    if p.exists():return p.read_bytes()
    b=bytearray()
    with requests.get(BASE+name,stream=True,timeout=(15,40)) as r:
        r.raise_for_status()
        for chunk in r.iter_content(65536):
            b.extend(chunk)
            if len(b)>cap:raise ValueError('Download cap')
    with p.open('xb') as f:f.write(b)
    return bytes(b)
def main():
    ROOT.mkdir(exist_ok=True)
    checksum=download('SHA256SUMS',8192)
    expected={line.split()[-1].lstrip('*'):line.split()[0] for line in checksum.decode().splitlines() if line.strip()}
    blob=download(ZIP,4000000)
    if sha(blob)!=expected[ZIP]:raise ValueError('Release checksum mismatch')
    print('CHECKSUM_VERIFIED',sha(blob),len(blob),flush=True)
    originals=json.loads(Path('/workspace/project-instinct-data-search-20261117/motion_source_comparison.json').read_text())['originals']
    records=[];yaml_text={}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        infos=z.infolist()
        if len(infos)>2000 or sum(i.file_size for i in infos)>50000000:raise ValueError('Archive cap')
        for i in infos:
            p=PurePosixPath(i.filename)
            if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(i.external_attr>>16):raise ValueError('Unsafe member')
            if i.is_dir():continue
            b=z.read(i);r=dict(name=i.filename,size=len(b),sha256=sha(b))
            if p.name in originals:r.update(original_sha256=originals[p.name]['sha256'],matches_original=sha(b)==originals[p.name]['sha256'])
            if p.suffix in ('.yaml','.yml','.json','.md','.txt') and len(b)<100000:yaml_text[i.filename]=b.decode('utf-8')
            if p.suffix=='.npz':
                with np.load(io.BytesIO(b),allow_pickle=False) as n:
                    r['schema']={k:{'shape':list(n[k].shape),'dtype':str(n[k].dtype)} for k in n.files}
                    r['framerate']=n['framerate'].item() if 'framerate' in n.files else None
                    r['extra_metadata_keys']=[k for k in n.files if k not in ('framerate','joint_names','joint_pos','base_pos_w','base_quat_w')]
            records.append(r)
    result=dict(release=BASE.rsplit('/',1)[0],archive_sha256=sha(blob),checksum_verified=True,members=records,text_metadata=yaml_text,code_executed=False,extracted=False,originals_modified=False,accepted_reference=False,training_ready=False)
    out=ROOT/'CONTENT_AUDIT.json'
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    assert json.loads(out.read_text())==result
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()