"""Bounded subprocess wrapper. Default=plan only; no installs or network IO.

Proposed limits are NOT spending approval. Flux task must be separately
authorized and launched. Never runs train.py or any Flux API/CLI command.
"""
import argparse,json,os,re,signal,subprocess,sys,time
import importlib.util
from pathlib import Path
from preflight import audit

ROOT=Path(__file__).resolve().parent

def diagnostic_tail(text):
    text=re.sub(r'https?://\S+','[URL omitted]',text)
    text=re.sub(r'(?i)(api[-_ ]?key|authorization|password|secret|access[-_]?token)\s*[:=]\s*[^\s,]+',r'\1=[REDACTED]',text)
    text=re.sub(r'\b(?:gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|gm_sk_[A-Za-z0-9]+)\b','[REDACTED]',text)
    return text[-8192:]

def deadline_reason(age,initialized):
    if age>=540:return 'worker_child_wall_limit'
    if age>=300 and not initialized:return 'asset_initialization_deadline'
    return None

def stop_child(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)

def main(args):
    if not args.execute_probe:
        print(json.dumps(dict(mode='plan_only',gpu_started=False,training_updates=0,
            proposed_worker_cap_s=600,proposed_child_cap_s=540,proposed_asset_initialization_cap_s=300,
            automatic_retries=0,launch_authorized=False)),flush=True);return
    started=time.monotonic()
    print(json.dumps(dict(stage='worker_started',python_version=sys.version.split()[0],training_updates=0)),flush=True)
    audit(ROOT)
    # Both paths are documented image candidates, not guaranteed locations.
    # We inspect only known installation paths, not runtime/control interfaces.
    if args.isaac_launcher:
        launcher=Path(args.isaac_launcher)
        if not launcher.is_file():raise FileNotFoundError('Requested Isaac Python launcher absent')
    else:
        candidates=[]
        if importlib.util.find_spec('isaaclab') is not None:candidates.append(Path(sys.executable))
        candidates.extend([Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')])
        launcher=next((p for p in candidates if p.is_file()),None)
        if launcher is None:
            raise RuntimeError('Set --isaac-launcher to the verified image Python executable/python.sh; no environment is guessed or installed')
    output=ROOT/'output'
    if output.exists():raise FileExistsError('Output already exists: refuse retry/overwrite')
    logpath=ROOT/'isaac_probe.log'
    command=[str(launcher),str(ROOT/'probe.py'),'--execute-probe','--output',str(output),'--steps','120']
    reason=None;initialized=False;proc=None;last_heartbeat=-30
    print(json.dumps(dict(stage='probe_child_starting',launcher=str(launcher))),flush=True)
    try:
        with logpath.open('x') as logfile:
            proc=subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
            while proc.poll() is None:
                age=time.monotonic()-started
                with logpath.open('rb') as reader:
                    reader.seek(max(0,logpath.stat().st_size-32768));tail=reader.read(32768)
                if b'"stage": "asset_initialized"' in tail:initialized=True
                if age-last_heartbeat>=30:
                    print(json.dumps(dict(stage='probe_heartbeat',age_s=round(age,1),asset_initialized=initialized)),flush=True)
                    last_heartbeat=age
                reason=deadline_reason(age,initialized)
                if reason:break
                time.sleep(.2)
            if reason:stop_child(proc)
            code=proc.wait()
        report_path=output/'report.json'
        result=json.loads(report_path.read_text()) if report_path.is_file() else None
        passed=(reason is None and code==0 and result is not None and result.get('probe_passed') is True)
        final=dict(stage='worker_finished',probe_passed=passed,return_code=code,stop_reason=reason,
            worker_wall_s=time.monotonic()-started,training_updates=0,training_ready=False,
            actual_output_upload_verified=False,output_transport='checksummed_allowlisted_log_chunks',
            flux_log_transport_end_to_end_verified=False)
        with (ROOT/'worker_result.json').open('x') as f:json.dump(final,f,indent=2)
        print(json.dumps(final),flush=True)
        if not passed:
            with logpath.open('rb') as reader:
                reader.seek(max(0,logpath.stat().st_size-32768))
                print(json.dumps(dict(stage='probe_child_log_tail',text=diagnostic_tail(reader.read(32768).decode('utf-8',errors='replace')))),flush=True)
        if passed:
            from evidence_transport import NAMES,encode_evidence
            files={name:(ROOT/'worker_result.json' if name=='worker_result.json' else output/name).read_bytes() for name in NAMES}
            for item in encode_evidence(files):print(json.dumps(item),flush=True)
        if not passed:raise SystemExit(1)
    finally:
        if proc is not None:stop_child(proc)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-probe',action='store_true')
    parser.add_argument('--isaac-launcher')
    main(parser.parse_args())