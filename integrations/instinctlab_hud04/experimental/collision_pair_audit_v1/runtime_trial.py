"""One bounded NEW runtime observation; not a reference or PPO training run."""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import selectors
import signal
import subprocess
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ASSET = REPO / 'flux/hud04_asset_camera_probe'
MAX_STEPS = 8
CHILD_SECONDS = 285


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def emit(phase, **kw):
    print(json.dumps(dict(stage='hud04_runtime_trial', phase=phase, **kw)), flush=True)


def verify(commit):
    actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True, timeout=5).strip()
    if actual != commit:
        raise ValueError('Expected full commit mismatch')
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True, timeout=5).strip():
        raise ValueError('Tracked modifications; do not reset')
    # Existing verified model checker; never execute the old worker/probe.
    sys.path.insert(0, str(HERE.parent / 'import_observation_20261117'))
    from import_observation_probe import verify_bundle
    verify_bundle(REPO)


def child(out, commit):
    verify(commit)
    os.environ['HUMANOID_DESCRIPTION_DIR'] = str(ASSET / 'humanoid-description')
    os.environ['ENABLE_CAMERAS'] = '0'
    from isaaclab.app import AppLauncher
    emit('initializing', explicit_physics_steps_max=MAX_STEPS)
    app = AppLauncher(headless=True, enable_cameras=False, device='cuda:0', fast_shutdown=True).app
    sim = None
    holder = None
    try:
        import carb
        import numpy as np
        import omni.physx
        from omni.physx.bindings import _physx
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass
        from pxr import PhysicsSchemaTools, UsdUtils, Usd, Gf
        from prepare_runtime_observation import preparation, existing_stage_snapshot, configured_scene_path, attempt
        from observe_existing_runtime import ExistingRuntimeObservation

        prepared = preparation()
        cfg = runpy.run_path(str(ASSET / 'asset_config.py'))['HUD04_CFG'].replace(prim_path='{ENV_REGEX_NS}/Robot')
        # NEW conversion only; no original drive/filter/gravity/threshold changes.
        cfg.spawn.usd_dir = str(out / 'new_conversion_cache')
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device='cuda:0'))

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = cfg

        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=4., replicate_physics=False, filter_collisions=True))
        sim.reset()
        scene.reset()
        robot = scene['robot']
        names = list(robot.joint_names)
        if robot.is_fixed_base or len(names) != 29 or len(set(names)) != 29 or any('head' in n for n in names):
            raise ValueError('Floating 29-body/zero-head policy contract failed')
        stage = scene.stage
        stage_id = int(UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
        registry = prepared['exact_expected_pairs']
        scene_selection = attempt(lambda: configured_scene_path(sim.cfg))
        paths = ['/World/envs/env_0/Robot']
        if scene_selection['status'] == 'OBSERVED':
            paths.insert(0, scene_selection['value'])
        for pair in registry:
            paths.extend(pair[k] for k in ('actor0', 'actor1', 'collider0', 'collider1'))
            paths.append('/World/envs/env_0/Robot/joints/' + pair['key'])
        paths = list(dict.fromkeys(paths))
        snapshot = existing_stage_snapshot(stage, carb.settings.get_settings(), _physx, paths=paths,
            quaternion_types=(Gf.Quatd, Gf.Quatf, Gf.Quath))
        snapshot['physics_scene_selection'] = dict(source='HOST_sim.cfg.physics_prim_path',
            read=scene_selection,hardcoded_fallback_used=False,solver_scene_ownership_verified=False)
        write(out / 'metadata.json', snapshot)
        # Hash the NEW generated layers, never call these historical USD bytes.
        generated = []
        for layer in stage.GetUsedLayers():
            path = Path(layer.realPath) if layer.realPath else None
            generated.append(dict(identifier=str(layer.identifier)[:300], real_file=bool(path and path.is_file()),
                sha256=digest(path) if path and path.is_file() and path.stat().st_size <= 50_000_000 else None))
        observer = ExistingRuntimeObservation(registry, PhysicsSchemaTools, expected_stage_id=stage_id,
            max_steps=MAX_STEPS, wall_seconds=30, max_events=64, max_points=512)
        holder = observer.subscribe_existing(omni.physx.get_physx_simulation_interface(), authorization=dict(
            approved_existing_process=True, starts_new_process=False,
            approval_reference='Host created by user-authorized runtime_trial at 2026-11-18T08:11:46Z; subscription attaches only after reset'))
        emit('runtime_ready', body_dof=29, head_policy_dof=0, stage_id=stage_id)
        traces = []
        for step in range(MAX_STEPS):
            if not app.is_running():
                raise RuntimeError('Application stopped before step budget')
            observer.on_host_step_begin(step)
            robot.set_joint_position_target(robot.data.default_joint_pos)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(.005)
            observer.on_host_step_end()
            q = robot.data.joint_pos.clone().cpu().numpy()
            root = robot.data.root_state_w.clone().cpu().numpy()
            if not np.isfinite(q).all() or not np.isfinite(root).all():
                raise ValueError('Nonfinite runtime state')
            traces.append(dict(step=step, root_state_w=root.tolist(), joint_pos=q.tolist()))
            if observer.error or observer.observer.error:
                break
        contacts = observer.finish()
        write(out / 'contacts.json', contacts)
        identity = dict(python=sys.version.split()[0], torch_version=torch.__version__, cuda=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0), usd_version=list(Usd.GetVersion()),
            physx_version=getattr(_physx, '__version__', None),
            physics_simulation_source_sha256=digest(Path(sim_utils.__file__)),
            native_physics_library_sha256=digest(Path(_physx.__file__)) if getattr(_physx, '__file__', None) and Path(_physx.__file__).is_file() else None)
        report = dict(stage='hud04_new_runtime_observed', source_commit=commit, runtime_identity=identity,
            image_reference='BJX00000178/V000446', immutable_image_digest=None, stage_id=stage_id,
            new_generated_layers=generated, historical_usd_recovered=False,
            body_dof=29, head_policy_dof=0, joint_names=names, num_envs=1,
            explicit_physics_steps=len(traces), initialization_may_step_physics=True, physics_dt_s=.005,
            camera_enabled=False, ground_or_terrain_added=False, original_gravity_drive_filter_thresholds_unchanged=True,
            root_and_joint_trace=traces, callback_invocations=contacts['callback_invocations'],
            contact_points=contacts['contact_points_seen'], observer_error=contacts['binding_error'] or contacts['error'],
            observation_completed=len(traces) == MAX_STEPS and contacts['binding_error'] is None and contacts['error'] is None,
            zero_contacts_does_not_prove_filtering=True, cooked_geometry_verified=False,
            full_task_tested=False, reference_motion_used=False, solver_calls=0, ppo_updates=0,
            accepted_reference=False, training_ready=False, cleanup_requested=False)
        write(out / 'runtime_report.json', report)
        emit('observation_saved', explicit_steps=len(traces), callbacks=report['callback_invocations'], ppo_updates=0)
    except BaseException as exc:
        write(out / 'runtime_error.json', dict(type=type(exc).__name__, message=str(exc)[:1500],
            frames=traceback.format_tb(exc.__traceback__)[-8:]))
        raise
    finally:
        # Release our holder before cleaning our own simulation context.
        holder = None
        emit('cleanup_started')
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        app.close(wait_for_replicator=False)
        emit('cleanup_returned')


def supervisor(out, commit):
    verify(commit)
    out.mkdir(exist_ok=False)
    if importlib.util.find_spec('isaaclab') is not None:
        launcher = Path(sys.executable)
    else:
        launcher = next((p for p in [Path('/isaac-sim/python.sh'), Path('/workspace/isaaclab/_isaac_sim/python.sh')] if p.is_file()), None)
    if launcher is None:
        raise RuntimeError('No local Isaac interpreter; no install/retry')
    env = os.environ.copy()
    for key in ('ins1116', 'flux_key1103', 'limxanzhuangbao', 'FLUX_API_KEY', 'WANDB_API_KEY', 'OPENAI_API_KEY'):
        env.pop(key, None)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PXR_WORK_THREAD_LIMIT'):
        env[key] = '2'
    env['TMPDIR'] = str(out)
    start = time.monotonic()
    proc = subprocess.Popen([str(launcher), str(Path(__file__).resolve()), '--child', '--expected-commit', commit,
        '--output', str(out)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    reason = None
    written = 0
    heartbeat = -30
    with (out / 'child.log').open('xb') as stream:
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            eof = False
            while not eof or proc.poll() is None:
                age = time.monotonic() - start
                if age - heartbeat >= 20:
                    emit('heartbeat', elapsed_s=round(age, 1)); heartbeat = age
                if age >= CHILD_SECONDS:
                    reason = 'child_wall_deadline'; break
                if not selector.select(.05):
                    continue
                data = os.read(proc.stdout.fileno(), 65536)
                if not data:
                    eof = True; selector.unregister(proc.stdout); continue
                if written + len(data) > 8 * 1024 * 1024:
                    reason = 'log_budget'; break
                stream.write(data); written += len(data)
        finally:
            selector.close()
            proc.stdout.close()
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    pass
                if sig == signal.SIGTERM:
                    try:
                        proc.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        pass
            code = proc.wait(timeout=1)
    result = dict(stage='hud04_runtime_trial_worker_finished', source_commit=commit, return_code=code,
        stop_reason=reason, child_wall_s=time.monotonic()-start, child_reaped=True,
        automatic_retries=0, platform_hard_cap_verified=False, ppo_updates=0, training_ready=False)
    write(out / 'worker_result.json', result)
    # Only allowlisted JSON, not the raw child/SDK log or arbitrary worker files.
    names = ['runtime_report.json', 'metadata.json', 'contacts.json', 'runtime_error.json', 'worker_result.json']
    files = {n: (out/n).read_bytes() for n in names if (out/n).is_file()}
    if sum(map(len, files.values())) > 1_000_000:
        raise ValueError('Expanded result budget exceeded')
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name, raw in files.items():
            json.loads(raw); bundle.writestr(name, raw)
    raw = memory.getvalue()
    if len(raw) > 512_000:
        raise ValueError('Compressed result budget exceeded')
    encoded = base64.b64encode(raw).decode()
    chunks = [encoded[i:i+8192] for i in range(0, len(encoded), 8192)]
    sha = hashlib.sha256(raw).hexdigest()
    emit('evidence_manifest', sha256=sha, zip_bytes=len(raw), chunks=len(chunks),
        file_sha256={n: hashlib.sha256(v).hexdigest() for n, v in files.items()})
    for index, data in enumerate(chunks):
        emit('evidence_chunk', sha256=sha, index=index, data=data)
    print(json.dumps(result), flush=True)
    if code or reason or 'runtime_report.json' not in files or not json.loads(files['runtime_report.json'])['observation_completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--expected-commit')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.child or args.execute:
        if not args.expected_commit or not args.output or not args.output.is_absolute():
            parser.error('Explicit commit and absolute output required')
        (child if args.child else supervisor)(args.output, args.expected_commit)
    else:
        emit('plan_only', child_seconds_max=CHILD_SECONDS, explicit_physics_steps_max=MAX_STEPS,
            automatic_retries=0, platform_hard_cap_verified=False, ppo_updates=0)