"""Bounded failure-only evidence transport. NEVER changes run/success status."""
import base64,hashlib,io,json,math,os,re,zipfile
from pathlib import Path

NAMES=('worker_result.json','runtime_report.json','post_import.json','post_clone.json','post_physics_init.json')
EXPANDED=1024*1024
COMPRESSED=512*1024
FALSE_FLAGS={'accepted_reference','training_ready','cooked_geometry_verified','actual_physx_pair_filters_verified','contact_schedule_accepted','source_motion_units_or_alignment_proven'}

def digest(raw):return hashlib.sha256(raw).hexdigest()
def strict_json(raw):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('Duplicate JSON key')
            d[k]=v
        return d
    def bad(_):raise ValueError('Nonfinite JSON')
    d=json.loads(raw,parse_constant=bad,object_pairs_hook=pairs)
    stack=[(d,0)];count=0
    while stack:
        v,depth=stack.pop();count+=1
        if depth>60 or count>300000:raise ValueError('JSON structure budget')
        if isinstance(v,float) and not math.isfinite(v):raise ValueError('Nonfinite JSON number')
        if isinstance(v,dict):
            for k,x in v.items():
                if k in FALSE_FLAGS and x is not False:raise ValueError('Unexpected acceptance claim')
                if re.search(r'(?i)(api[-_]?key|authorization|password|access[-_]?token|secret)',k):raise ValueError('Sensitive field')
                stack.append((x,depth+1))
        elif isinstance(v,list):stack.extend((x,depth+1) for x in v)
    return d

def check_files(files,expected_commit):
    if not re.fullmatch('[0-9a-f]{40}',expected_commit):raise ValueError('Explicit expected commit')
    if not set(files)<=set(NAMES) or 'worker_result.json' not in files:raise ValueError('Evidence allowlist')
    if sum(len(r) for r in files.values())>EXPANDED:raise ValueError('Expanded budget')
    parsed={}
    for n,raw in files.items():
        if len(raw)>(512000 if n.startswith('post_') else 32768):raise ValueError('File budget')
        for key in ['github1117','flux1117','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY']:
            val=os.environ.get(key)
            if val and val.encode() in raw:raise ValueError('Credential in evidence')
        if re.search(rb'(?:gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|gm_sk_[A-Za-z0-9_]+|https?://)',raw):raise ValueError('Sensitive token or URL')
        d=strict_json(raw)
        if not isinstance(d,dict) or d.get('accepted_reference') is not False or d.get('training_ready') is not False:raise ValueError('Missing conservative flags')
        if n=='worker_result.json':
            if d.get('stage')!='bounded_import_worker_result' or d.get('observation_completed') is not False or d.get('direct_child_reaped') is not True:raise ValueError('Failure worker required')
            if type(d.get('return_code')) is not int:raise ValueError('Missing child exit')
        elif n=='runtime_report.json':
            if d.get('stage')!='hud04_import_observed' or d.get('source_commit')!=expected_commit:raise ValueError('Runtime source mismatch')
        else:
            if d.get('stage_observation_kind')!='composed_USD_metadata_ONLY' or d.get('runtime_phase')!=n[:-5] or d.get('runtime_execution') is not True:raise ValueError('Snapshot schema/phase')
            for k in ['colliders','joints','articulations','global_filtered_pairs','collision_groups']:
                if not isinstance(d.get(k),list):raise ValueError('Snapshot lists required')
        parsed[n]=d
    return parsed

def emit_failure_evidence(output,expected_commit):
    root=Path(output);files={}
    for n in NAMES:
        p=root/n
        if p.is_symlink():raise ValueError('Symlink not allowed')
        if p.exists():
            if not p.is_file() or p.stat().st_size>(512000 if n.startswith('post_') else 32768):raise ValueError('File type/budget')
            files[n]=p.read_bytes()
    check_files(files,expected_commit)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for n,raw in files.items():z.writestr(n,raw)
    data=buf.getvalue()
    if len(data)>COMPRESSED:raise ValueError('Compressed budget')
    encoded=base64.b64encode(data).decode();chunks=[encoded[i:i+8192] for i in range(0,len(encoded),8192)]
    m=dict(stage='hud04_import_failure_manifest',schema_version=1,expected_commit=expected_commit,
        observation_completed=False,accepted_reference=False,training_ready=False,
        missing_files=sorted(set(NAMES)-set(files)),sha256=digest(data),zip_bytes=len(data),chunks=len(chunks),
        file_sha256={n:digest(raw) for n,raw in files.items()})
    print(json.dumps(m),flush=True)
    for i,v in enumerate(chunks):print(json.dumps(dict(stage='hud04_import_failure_chunk',sha256=m['sha256'],index=i,data=v)),flush=True)

def decode_failure_records(records,expected_commit):
    manifests=[r for r in records if r.get('stage')=='hud04_import_failure_manifest']
    if len(manifests)!=1:raise ValueError('Exactly one failure manifest required')
    m=manifests[0]
    if m.get('schema_version')!=1 or m.get('expected_commit')!=expected_commit or any(m.get(k) is not False for k in ['observation_completed','accepted_reference','training_ready']):raise ValueError('Failure manifest identity/flags')
    if type(m.get('chunks')) is not int or not 1<=m['chunks']<=86 or type(m.get('zip_bytes')) is not int or not 0<m['zip_bytes']<=COMPRESSED:raise ValueError('Manifest budget')
    chunks=[r for r in records if r.get('stage')=='hud04_import_failure_chunk']
    if len(chunks)!=m['chunks'] or any(type(r.get('index')) is not int or r.get('sha256')!=m['sha256'] or not isinstance(r.get('data'),str) or len(r['data'])>8192 for r in chunks):raise ValueError('Chunk metadata')
    if {r['index'] for r in chunks}!=set(range(m['chunks'])):raise ValueError('Missing/duplicate chunk')
    data=base64.b64decode(''.join(r['data'] for r in sorted(chunks,key=lambda r:r['index'])),validate=True)
    if len(data)!=m['zip_bytes'] or digest(data)!=m['sha256']:raise ValueError('Archive integrity')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or not set(names)<=set(NAMES) or 'worker_result.json' not in names or sum(i.file_size for i in z.infolist())>EXPANDED:raise ValueError('Archive allowlist/budget')
        files={n:z.read(n) for n in names}
    if {n:digest(raw) for n,raw in files.items()}!=m.get('file_sha256') or sorted(set(NAMES)-set(files))!=m.get('missing_files'):raise ValueError('File hash/missing list mismatch')
    return check_files(files,expected_commit),files