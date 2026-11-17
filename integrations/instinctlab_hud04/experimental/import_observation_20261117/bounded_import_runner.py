"""Bound a trusted diagnostic child. This is NOT a platform billing cap."""
import hashlib,json,math,os,signal,subprocess,time
from pathlib import Path

MAX_LOG=8*1024*1024

def run_child(argv,output,*,seconds=240,env=None):
    if isinstance(seconds,bool) or not isinstance(seconds,(float,int)) or not math.isfinite(seconds) or not 0<seconds<=300:raise ValueError('Child deadline must be <=300 seconds')
    if not isinstance(argv,list) or not argv or any(not isinstance(x,str) for x in argv):raise ValueError('Explicit trusted argv required')
    out=Path(output);out.mkdir(exist_ok=False,parents=True);log=out/'child.log';start=time.monotonic();reason=None
    def limits():
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE,(MAX_LOG,MAX_LOG))
    # POSIX trusted supervisor; no process discovery, daemon sockets or host APIs.
    with log.open('xb') as stream:
        proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            env=env,start_new_session=True,preexec_fn=limits)
        try:
            while proc.poll() is None:
                if time.monotonic()-start>=seconds:reason='child_wall_deadline';break
                if log.stat().st_size>=MAX_LOG:reason='child_log_budget';break
                time.sleep(.02)
        finally:
            # The process group belongs to THIS child only. Never discover/kill
            # other tasks. Clean surviving owned descendants even if leader exits.
            for sig in (signal.SIGTERM,signal.SIGKILL):
                try:os.killpg(proc.pid,sig)
                except ProcessLookupError:pass
                if sig==signal.SIGTERM:
                    try:proc.wait(timeout=.3)
                    except subprocess.TimeoutExpired:pass
            try:code=proc.wait(timeout=.5)
            except subprocess.TimeoutExpired:raise RuntimeError('Owned child cleanup failed') from None
    report_path=out/'runtime_report.json';report=None
    if report_path.is_file() and report_path.stat().st_size<=32768:
        try:report=json.loads(report_path.read_text())
        except (ValueError,UnicodeError):pass
    normal=(reason is None and code==0 and isinstance(report,dict) and report.get('stage')=='hud04_import_observed'
        and report.get('cleanup_completed') is True and report.get('snapshot_phases')==['post_import','post_clone','post_physics_init']
        and all(report.get(k) is False for k in ['accepted_reference','training_ready','actual_physx_pair_filters_verified','cooked_geometry_verified']))
    result=dict(stage='bounded_import_worker_result',observation_completed=normal,stop_reason=reason,return_code=code,
        direct_child_reaped=True,cleanup_grace_seconds_max=.8,child_wall_seconds=time.monotonic()-start,
        log_bytes=log.stat().st_size,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        local_report_present=report is not None,platform_deadline_verified=False,gpu_submissions=0,
        solver_calls=0,ppo_updates=0,accepted_reference=False,training_ready=False)
    with (out/'worker_result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    return result

def emit_evidence(output):
    """Exact allowlist, no raw child logs, no credentials or arbitrary files."""
    import base64,io,zipfile
    root=Path(output);names=['runtime_report.json','post_import.json','post_clone.json','post_physics_init.json','worker_result.json']
    if any(not (root/n).is_file() for n in names):raise ValueError('Complete observation files required')
    files={n:(root/n).read_bytes() for n in names}
    if sum(map(len,files.values()))>1024*1024:raise ValueError('Evidence exceeds 1MB expanded budget')
    for n,raw in files.items():json.loads(raw)
    if json.loads(files['worker_result.json']).get('observation_completed') is not True:raise ValueError('Failed observation is not complete evidence')
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for n,v in files.items():z.writestr(n,v)
    data=memory.getvalue()
    if len(data)>512*1024:raise ValueError('Compressed evidence exceeds budget')
    encoded=base64.b64encode(data).decode('ascii');chunks=[encoded[i:i+8192] for i in range(0,len(encoded),8192)]
    digest=hashlib.sha256(data).hexdigest()
    print(json.dumps(dict(stage='hud04_import_evidence_manifest',sha256=digest,zip_bytes=len(data),chunks=len(chunks),
        file_sha256={n:hashlib.sha256(v).hexdigest() for n,v in files.items()})),flush=True)
    for i,value in enumerate(chunks):print(json.dumps(dict(stage='hud04_import_evidence_chunk',sha256=digest,index=i,data=value)),flush=True)