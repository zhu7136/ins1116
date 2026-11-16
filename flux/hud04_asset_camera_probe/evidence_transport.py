"""Bounded, checksummed output evidence over application logs; no upload key."""
import base64,hashlib,io,json,zipfile

NAMES={'report.json','depth_samples.npz','worker_result.json'}
MAX_BYTES=131072

def encode_evidence(files):
    if set(files)!=NAMES:raise ValueError('Only allowlisted probe outputs may leave worker')
    if sum(map(len,files.values()))>MAX_BYTES:raise ValueError('Evidence exceeds output budget')
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):archive.writestr(name,files[name])
    payload=memory.getvalue()
    if len(payload)>MAX_BYTES:raise ValueError('ZIP exceeds output budget')
    digest=hashlib.sha256(payload).hexdigest();text=base64.b64encode(payload).decode('ascii')
    chunks=[text[n:n+3072] for n in range(0,len(text),3072)]
    manifest=dict(stage='hud04_evidence_manifest',sha256=digest,zip_bytes=len(payload),chunks=len(chunks),
        file_sha256={n:hashlib.sha256(v).hexdigest() for n,v in files.items()})
    return [manifest]+[dict(stage='hud04_evidence_chunk',sha256=digest,index=i,data=chunk) for i,chunk in enumerate(chunks)]

def decode_evidence(events):
    manifests=[x for x in events if x.get('stage')=='hud04_evidence_manifest']
    if len(manifests)!=1:raise ValueError('Need exactly one evidence manifest')
    manifest=manifests[0];count=manifest['chunks']
    if not isinstance(count,int) or not 1<=count<=64:raise ValueError('Bad chunk count')
    chunks=[x for x in events if x.get('stage')=='hud04_evidence_chunk']
    if len(chunks)!=count or sorted(x['index'] for x in chunks)!=list(range(count)):
        raise ValueError('Missing or duplicate evidence chunks')
    if any(x['sha256']!=manifest['sha256'] for x in chunks):raise ValueError('Evidence identity mismatch')
    text=''.join(x['data'] for x in sorted(chunks,key=lambda x:x['index']))
    if len(text)>MAX_BYTES*2:raise ValueError('Oversized evidence')
    payload=base64.b64decode(text,validate=True)
    if len(payload)!=manifest['zip_bytes'] or len(payload)>MAX_BYTES:raise ValueError('Evidence size mismatch')
    if hashlib.sha256(payload).hexdigest()!=manifest['sha256']:raise ValueError('Evidence checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.infolist())!=3 or set(archive.namelist())!=NAMES:raise ValueError('Unexpected output file')
        if sum(x.file_size for x in archive.infolist())>MAX_BYTES:raise ValueError('Expanded evidence too large')
        files={n:archive.read(n) for n in NAMES}
    if {n:hashlib.sha256(v).hexdigest() for n,v in files.items()}!=manifest['file_sha256']:
        raise ValueError('Per-file evidence checksum mismatch')
    return files

if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Decode exported Flux application log evidence; refuses overwrite')
    parser.add_argument('log',type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();events=[]
    with args.log.open() as stream:
        for line in stream:
            start=line.find('{')
            if start<0 or len(line)>16384:continue
            try:event=json.loads(line[start:])
            except ValueError:continue
            if isinstance(event,dict) and str(event.get('stage','')).startswith('hud04_evidence_'):events.append(event)
    files=decode_evidence(events);args.output.mkdir(exist_ok=False,parents=True)
    for name,value in files.items():(args.output/name).write_bytes(value)
    print(json.dumps(dict(verified_files=len(files),training_ready=False)))