"""Bounded local child wrapper. NOT a Flux submission/client or GPU authorization.

Default: plan only. Before real service submission, separately inspect live
tasks, duplicates, verified image/launcher and the platform billing/stop path.
No imports/execution of old workers, no auto-install, retries or overwrites.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
INPUTS=('contact_instrument.py','terrain_contact_bridge.py','contact_instrument_worker.py')
MAX_LOG=8*1024*1024


def deadline_reason(age,initialized,shutdown_age=None):
    if not math.isfinite(age) or age<0:raise ValueError('Invalid worker clock')
    if age>=300:return 'child_wall_limit'
    if shutdown_age is not None:
        if not math.isfinite(shutdown_age) or shutdown_age<0:raise ValueError('Invalid shutdown clock')
        if shutdown_age>=20:return 'shutdown_limit'
    if not initialized and age>=180:return 'initialization_limit'
    return None


def evidence(report):
    raw=json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>32768:raise ValueError('Report exceeds evidence cap')
    return dict(stage='independent_contact_instrument_evidence',report=report,
                sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def verify_evidence(item):
    if item!=evidence(item['report']):raise ValueError('Evidence checksum/size mismatch')
    return item['report']


def audit_sources(root=ROOT):
    root=Path(root)
    manifest=json.loads((root/'CONTACT_INSTRUMENT_MANIFEST.json').read_text())
    if set(manifest['files'])!=set(INPUTS):raise ValueError('Unexpected input allowlist')
    for name in INPUTS:
        path=root/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['files'][name]:
            raise ValueError('Instrument source mismatch: '+name)
    return manifest


def stop_child(proc):
    """Only the child started in our own new session, never external tasks."""
    if proc.poll() is not None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait(timeout=3)


def tail(path):
    with Path(path).open('rb') as reader:
        reader.seek(max(0,Path(path).stat().st_size-65536))
        return reader.read(65536)


def overflow_warning(text):
    # Conservative fail-closed patterns; retained full bounded log is also reviewed.
    lowered=text.lower()
    return any(pattern in lowered for pattern in [b'contact buffer overflow',
        b'contact buffer is full',b'contacts have been dropped',b'patch buffer overflow',
        b'gpu contact buffer',b'gpu collision stack size',b'foundlostpairscapacity'])


def event(stage,**values):
    print(json.dumps(dict(stage=stage,**values),allow_nan=False),flush=True)


def main(args):
    if not args.execute_instrument:
        event('worker_plan_only',gpu_started=False,child_wall_cap_s=300,
              initialization_cap_s=180,shutdown_cap_s=20,termination_grace_total_s=6,
              physics_steps_max=260,automatic_retries=0,ppo_updates=0,
              platform_task_status_checked=False,platform_billing_cap_verified=False)
        return 0
    if not args.expected_commit or not re.fullmatch('[0-9a-f]{40}',args.expected_commit):
        raise ValueError('Literal verified commit required before any child')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if commit!=args.expected_commit:raise ValueError('Checkout is not the expected commit')
    repo=Path(subprocess.check_output(['git','-C',str(ROOT),'rev-parse','--show-toplevel'],text=True).strip()).resolve()
    relative=(ROOT/'CONTACT_INSTRUMENT_MANIFEST.json').relative_to(repo).as_posix()
    committed=subprocess.check_output(['git','-C',str(repo),'show','HEAD:'+relative])
    if committed!=(ROOT/'CONTACT_INSTRUMENT_MANIFEST.json').read_bytes():
        raise ValueError('Instrument manifest is not committed at the verified HEAD')
    audit_sources()
    if args.output is None or args.isaac_launcher is None:
        raise ValueError('Explicit new output and verified image launcher required')
    launcher=Path(args.isaac_launcher).resolve()
    if not launcher.is_file():raise ValueError('Verified launcher is not a file')
    output=Path(args.output).resolve()
    output.mkdir(exist_ok=False)
    log=output/'instrument.log';proc=None;reason=None;initialized=False;shutdown=None
    start=time.monotonic();last_heartbeat=-30.
    try:
        with log.open('xb') as stream:
            proc=subprocess.Popen([str(launcher),str(ROOT/'contact_instrument.py'),
                '--execute-instrument','--output',str(output/'child')],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                age=time.monotonic()-start;current=tail(log)
                initialized |= b'"stage": "instrument_initialized"' in current
                if shutdown is None and b'"stage": "instrument_shutdown_started"' in current:shutdown=age
                reason=deadline_reason(age,initialized,None if shutdown is None else age-shutdown)
                if log.stat().st_size>MAX_LOG:reason='log_size_limit'
                if overflow_warning(current):reason='possible_physics_buffer_overflow'
                if age-last_heartbeat>=30:
                    event('instrument_heartbeat',age_s=round(age,1),initialized=initialized)
                    last_heartbeat=age
                if reason:
                    stop_child(proc);break
                time.sleep(.2)
            code=proc.wait(timeout=3)
        # Read the whole log once only after exit; size is bounded above by the
        # polling guard plus one producer burst, never print arbitrary raw logs.
        log_size=log.stat().st_size
        with log.open('rb') as stream:bounded_log=stream.read(MAX_LOG+1)
        if log_size>MAX_LOG:reason=reason or 'log_size_limit'
        if overflow_warning(bounded_log):reason=reason or 'possible_physics_buffer_overflow'
        report_path=output/'child'/'report.json';report=None
        if report_path.is_file():
            if report_path.stat().st_size>32768:reason=reason or 'report_size_limit'
            else:report=json.loads(report_path.read_text())
        if report is not None:print(json.dumps(evidence(report),allow_nan=False),flush=True)
        cleanup=bool(report and report.get('normal_cleanup_completed') is True and
                     b'"stage": "instrument_cleanup_completed", "completed": true' in bounded_log)
        passed=reason is None and code==0 and cleanup and report is not None and report.get('instrument_passed') is True
        result=dict(instrument_passed=passed,return_code=code,stop_reason=reason,
                    source_commit=commit,wall_seconds=time.monotonic()-start,
                    report_present=report is not None,normal_cleanup_verified=cleanup,
                    accepted_reference=False,training_ready=False,ppo_updates=0,
                    platform_billing_cap_verified=False)
        with (output/'worker_result.json').open('x') as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
        event('instrument_worker_finished',**result)
        return 0 if passed else 1
    finally:
        if proc is not None:stop_child(proc)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-instrument',action='store_true')
    parser.add_argument('--expected-commit')
    parser.add_argument('--isaac-launcher')
    parser.add_argument('--output',type=Path)
    raise SystemExit(main(parser.parse_args()))