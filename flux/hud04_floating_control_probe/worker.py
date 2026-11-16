"""Plan-only by default. One bounded child, no Flux API calls or retries."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ASSET = ROOT.parent / 'hud04_asset_camera_probe'


def evidence(report):
    raw = json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > 32768:
        raise ValueError('Report exceeds bounded log evidence size')
    return dict(stage='hud04_floating_report_evidence', report=report,
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def verify_evidence(item):
    expected = evidence(item['report'])
    if item != expected:
        raise ValueError('Report evidence identity/checksum/size mismatch')
    return item['report']


def main(args):
    if not args.execute_probe:
        print(json.dumps(dict(mode='plan_only', gpu_started=False, automatic_retries=0,
            child_wall_cap_s=540, initialization_cap_s=300, shutdown_cap_s=30,
            physics_steps_max=400, training_updates=0, lp_solves=0, qp_solves=0,
            optimizer_setup_calls=0, platform_billing_cap_verified=False)), flush=True)
        return
    sys.path.insert(0, str(ASSET))
    source_commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if not args.expected_commit or source_commit != args.expected_commit:
        raise RuntimeError('Explicit expected commit required and must match checkout before GPU initialization')
    print(json.dumps(dict(stage='floating_source_verified', source_commit=source_commit)), flush=True)
    old = runpy.run_path(str(ASSET / 'worker.py'))
    old['audit'](ASSET)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('New probe source integrity mismatch: ' + name)
    if args.isaac_launcher:
        launcher = Path(args.isaac_launcher)
    elif importlib.util.find_spec('isaaclab') is not None:
        launcher = Path(sys.executable)
    else:
        launcher = next((p for p in [Path('/isaac-sim/python.sh'),
            Path('/workspace/isaaclab/_isaac_sim/python.sh')] if p.is_file()), None)
    if launcher is None or not launcher.is_file():
        raise RuntimeError('Verified Isaac launcher required; no installation attempted')
    output, log = ROOT / 'output', ROOT / 'floating_probe.log'
    if output.exists() or log.exists() or (ROOT / 'worker_result.json').exists():
        raise FileExistsError('Prior outputs exist; refuse retry/overwrite')
    started, initialized, shutdown_at, reason, proc = time.monotonic(), False, None, None, None
    try:
        with log.open('x') as stream:
            proc = subprocess.Popen([str(launcher), str(ROOT/'probe.py'), '--execute-probe',
                '--output', str(output)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
            last_heartbeat = -30.
            while proc.poll() is None:
                age = time.monotonic() - started
                with log.open('rb') as reader:
                    reader.seek(max(0, log.stat().st_size-32768))
                    tail = reader.read(32768)
                initialized |= b'"stage": "asset_initialized"' in tail
                if shutdown_at is None and b'"stage": "simulator_shutdown_started"' in tail:
                    shutdown_at = age
                if age-last_heartbeat >= 30:
                    print(json.dumps(dict(stage='floating_probe_heartbeat', age_s=round(age, 1),
                                          asset_initialized=initialized)), flush=True)
                    last_heartbeat = age
                reason = old['deadline_reason'](age, initialized,
                    None if shutdown_at is None else age-shutdown_at)
                if reason:
                    old['stop_child'](proc)
                    break
                time.sleep(.2)
            code = proc.wait()
        path = output / 'report.json'
        report = json.loads(path.read_text()) if path.is_file() else None
        if report is not None:
            print(json.dumps(evidence(report), allow_nan=False), flush=True)
        with log.open('rb') as reader:
            reader.seek(max(0, log.stat().st_size-32768))
            final_tail = reader.read(32768)
        cleanup_complete = b'"stage": "simulation_context_cleanup_completed"' in final_tail
        normal_exit = reason is None and code == 0 and cleanup_complete
        passed = normal_exit and report is not None and report.get('probe_passed') is True
        result = dict(stage='floating_worker_finished', probe_passed=passed, return_code=code,
            source_commit=source_commit,
            stop_reason=reason, worker_wall_s=time.monotonic()-started, training_ready=False,
            training_updates=0, report_present=report is not None, normal_exit_verified=normal_exit,
            simulation_context_cleanup_completed=cleanup_complete)
        with (ROOT/'worker_result.json').open('x') as f:
            json.dump(result, f, indent=2, allow_nan=False)
        print(json.dumps(result), flush=True)
        if not passed:
            with log.open('rb') as reader:
                reader.seek(max(0, log.stat().st_size-32768))
                text = old['diagnostic_tail'](reader.read(32768).decode(errors='replace'))
            print(json.dumps(dict(stage='floating_child_log_tail', text=text)), flush=True)
            raise SystemExit(1)
    finally:
        if proc is not None:
            old['stop_child'](proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-probe', action='store_true')
    parser.add_argument('--isaac-launcher')
    parser.add_argument('--expected-commit')
    main(parser.parse_args())