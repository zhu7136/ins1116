"""Fresh bounded CPU-only experiment. No old launcher or task is replayed."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
EXP = HERE.parent
SOURCE = Path('/workspace/InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf')
TARGET = REPO / 'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
RAW = Path('/workspace/hud04-original-drive-20261118/kneelClimbStep1-x-0.1-ziwen-retargeted.npz')
INPUT = EXP / 'full_direction_20261118/OVERLAP_V4/STEP1_FULL_LENGTH_JOINT_DIAGNOSTIC.npz'
MODEL_CODE = EXP / 'direction_solve_20261118/anatomical_directions.py'
KD = REPO / 'integrations/instinctlab_hud04/kinematic_diagnostics.py'
PINS = {SOURCE: '5e99d930af64f3f0bfd34235239e288cc380ba62c8b01fcfcf8cc9f0996e7fa4',
        TARGET: '865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7',
        RAW: '56158e8679580910ccff375963d025b138636acfe70f5b1f88381837209b2056',
        INPUT: '2aa620d3086ed5f1c260c210e38fcb3d3a94cbaea549911844b8e4c31127ddc0',
        MODEL_CODE: '23660e03d2fbe7b1e0e334c47b506e9c07e4c8931430bd6f6c3afc762159587d',
        KD: '943b63605c4e837451a47eee87fa754ac04d72afcef02fe8904c77252b438696'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
    if json.loads(path.read_text()) != value:
        raise ValueError('JSON readback mismatch')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def summary(e):
    import numpy as np
    return dict(mean_degrees=float(e.mean()), p95_degrees=float(np.percentile(e, 95)),
                max_degrees=float(e.max()), over25_observations=int((e > 25).sum()),
                over25_frames=int(np.any(e > 25, axis=1).sum()))


def worker(out):
    import numpy as np
    from tail_solver import solve, angles
    for path, digest in PINS.items():
        if sha(path) != digest:
            raise ValueError('Pinned input differs: '+str(path))
    D = load_module('tail_anatomical', MODEL_CODE)
    source = D.DirectionModel(SOURCE, 'source', wrist_endpoint='first_wrist_pivot')
    target = D.DirectionModel(TARGET, 'target', wrist_endpoint='first_wrist_pivot')
    with np.load(RAW, allow_pickle=False) as z:
        sq = source.order(z['joint_pos'], z['joint_names'].tolist())
        if float(z['framerate']) != 90.:
            raise ValueError('Original native FPS')
    with np.load(INPUT, allow_pickle=False) as z:
        q = target.order(z['joint_pos'], z['joint_names'].tolist())
        np.testing.assert_array_equal(z['frame_indices'], np.arange(997))
    want = source.evaluate(sq)
    # Verify the analytic Jacobian before spending the real-data solve budget.
    probe = q[[0, 257, 495, 996]]
    _, jac = target.evaluate(probe, True)
    worst = 0.
    for j in range(29):
        plus = probe.copy(); minus = probe.copy(); plus[:, j] += 1e-6; minus[:, j] -= 1e-6
        numeric = (target.evaluate(plus)-target.evaluate(minus))/2e-6
        worst = max(worst, float(np.max(np.abs(numeric-jac[:, :, :, j]))))
    if worst > 2e-7:
        raise ValueError('Actual model Jacobian mismatch')
    with (out/'PROGRESS.jsonl').open('x') as log:
        def progress(row):
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
        result, report = solve(target, q, want, progress=progress, seconds=90.)
    filename = 'STEP1_TAIL_JOINT_DIAGNOSTIC_UNACCEPTED.npz'
    np.savez_compressed(out/filename, joint_pos=result, joint_names=np.array(target.names),
                        frame_indices=np.arange(997), framerate=np.array(90),
                        role=np.array('joint_direction_diagnostic_NOT_reference_NO_WORLD_ROOT'),
                        accepted_reference=np.array(False), training_ready=np.array(False))
    report.update(stage='angular_tail_joint_only_diagnostic_not_reference',
                  output=filename, output_sha256=sha(out/filename),
                  input_sha256={str(p): digest for p, digest in PINS.items()},
                  code_sha256={p.name: sha(p) for p in (HERE/'tail_solver.py', HERE/'run_tail_once.py')},
                  actual_model_jacobian_max_absolute_error=worst,
                  initial_summary=summary(np.array(report['initial_angles_degrees'])),
                  final_summary=summary(np.array(report['final_angles_degrees'])),
                  geometry_changes=0, automatic_pair_exclusions=0, gpu_submissions=0, ppo_updates=0)
    save_new(out/'REPORT.json', report)
    # Independent saved-file FK, separate Rodrigues implementation and names.
    K = load_module('tail_independent_fk', KD)
    def directions(tree, q, names, source):
        f = tree.fk(q, names)
        basis = f['pelvis' if source else 'base_link'][:, :3, :3]
        wrist = 'wrist_roll_link' if source else 'wrist_yaw_link'
        values = []
        for side in ('left', 'right'):
            for a, b in [('hip_pitch_link', 'knee_link'), ('knee_link', 'ankle_pitch_link'),
                         ('shoulder_pitch_link', 'elbow_link'), ('elbow_link', wrist)]:
                v = f[side+'_'+b][:, :3, 3]-f[side+'_'+a][:, :3, 3]
                v = np.einsum('nji,nj->ni', basis, v)
                values.append(v/np.linalg.norm(v, axis=1)[:, None])
        return np.stack(values, axis=1)
    with np.load(out/filename, allow_pickle=False) as z:
        saved = z['joint_pos'].copy(); names = z['joint_names'].tolist()
        if saved.shape != (997, 29) or float(z['framerate']) != 90. or bool(z['accepted_reference']):
            raise ValueError('Saved diagnostic contract differs')
        np.testing.assert_array_equal(z['frame_indices'], np.arange(997))
    independent_want = directions(K.RobotTree.read(SOURCE), sq, source.names, True)
    independent_actual = directions(K.RobotTree.read(TARGET), saved, names, False)
    independent_errors = np.rad2deg(angles(independent_actual, independent_want))
    discrepancy = float(np.max(np.abs(independent_errors-np.asarray(report['final_angles_degrees']))))
    if discrepancy > 1e-8:
        raise ValueError('Independent Rodrigues FK disagrees with saved output report')
    xml = ET.parse(TARGET).getroot()
    joints = {j.get('name'): j for j in xml.findall('joint') if j.get('type') != 'fixed'}
    if set(names) != set(joints) or len(names) != 29 or any('head' in n for n in names):
        raise ValueError('Independent exact named body space')
    lower, upper, vmax = [np.array([float(joints[n].find('limit').get(k)) for n in names])
                         for k in ('lower', 'upper', 'velocity')]
    position = float(max(0., np.max(lower-saved), np.max(saved-upper)))
    velocity = float(max(0., np.max(np.abs(np.diff(saved, axis=0))*90-vmax)))
    for name in ('head_yaw_joint', 'head_pitch_joint'):
        j = next(j for j in xml.findall('joint') if j.get('name') == name)
        if j.get('type') != 'fixed':
            raise ValueError('Head not fixed')
        if j.find('origin') is not None and np.linalg.norm(np.fromstring(j.find('origin').get('rpy', '0 0 0'), sep=' ')) > 1e-12:
            raise ValueError('Head not fixed at model zero')
    if max(position, velocity) > 1e-6:
        raise ValueError('Independent original joint/velocity bound failure')
    before = np.asarray(report['initial_angles_degrees'])
    verification = dict(stage='independent_saved997_native_rodrigues_fk_xml_limit_review',
        saved_output_sha256=sha(out/filename), max_angular_disagreement_degrees=discrepancy,
        max_position_violation_rad=position, max_velocity_violation_rad_s=velocity,
        final_summary=summary(independent_errors),
        worsened_direction_observations=int((independent_errors > before+1e-7).sum()),
        worsened_mean_direction_frames=int((independent_errors.mean(axis=1) > before.mean(axis=1)+1e-7).sum()),
        max_not_regressed=bool(independent_errors.max() <= before.max()+1e-7),
        root_fields_present=False, world_collision_or_support_validated=False,
        accepted_reference=False, training_ready=False)
    save_new(out/'INDEPENDENT_VERIFICATION.json', verification)
    for p, digest in PINS.items():
        if sha(p) != digest:
            raise ValueError('Input changed during solve')
    print(json.dumps(dict(initial=report['initial_summary'], final=report['final_summary'],
                         calls=report['actual_lp_calls'], accepted=report['accepted_updates'],
                         elapsed=report['elapsed_seconds'], independent=verification)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    out = HERE/'TRIAL_TAIL_001'
    if args.worker:
        worker(out); return
    if not args.execute:
        print(json.dumps(dict(plan_only=True, max_lp_calls=144, algorithm_seconds=90,
                              child_seconds=110, cpu_threads=2, gpu_submissions=0,
                              accepted_reference=False))); return
    out.mkdir(exist_ok=False)
    save_new(out/'INTENT.json', dict(source_parent_commit=subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        goal='reduce remaining997frame angular tails without joint/velocity relaxation',
        max_lp_calls=144, algorithm_seconds=90, child_seconds=110, automatic_retries=0,
        gpu_submissions=0, root_motion_or_contact_generation=False, accepted_reference=False))
    archive = out/'CODE_AT_EXECUTION'; archive.mkdir()
    for p in (HERE/'tail_solver.py', Path(__file__), MODEL_CODE, KD):
        shutil.copy2(p, archive/p.name)
    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '2'
    start = time.monotonic()
    timed_out = False
    with (out/'EXECUTION.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__)), '--worker'],
                                 stdout=log, stderr=subprocess.STDOUT, env=env, cwd='/workspace')
        try:
            code = child.wait(timeout=110)
        except subprocess.TimeoutExpired:
            timed_out = True; child.kill(); code = child.wait(timeout=10)
    save_new(out/'SUPERVISION.json', dict(return_code=code, timeout=timed_out, child_reaped=True,
                                         elapsed_seconds=time.monotonic()-start, automatic_retries=0))
    print((out/'EXECUTION.log').read_text()[-12000:], flush=True)
    if code != 0:
        raise SystemExit('Bounded child failed; no retry. Preserved evidence.')
    import wandb
    report = json.loads((out/'REPORT.json').read_text())
    agent = {k: os.environ[v] for k, v in [('thread_id', 'WANDB_ARIA_THREAD_ID'), ('turn_id', 'WANDB_ARIA_TURN_ID')] if os.environ.get(v)}
    run = wandb.init(entity='zhul136-university-of-science-and-technology-of-china', project='ins1116',
                     name='hud04-tail-joint-fit-001-NOT-reference', mode='offline', dir=str(out),
                     config={'_wb_agent': agent, 'scope': 'CPU joint-direction diagnostic; NOT accepted reference',
                             'input_sha256': sha(INPUT), 'gpu_submissions': 0, 'ppo_updates': 0})
    run.summary.update({'initial_mean_degrees': report['initial_summary']['mean_degrees'],
                        'final_mean_degrees': report['final_summary']['mean_degrees'],
                        'initial_max_degrees': report['initial_summary']['max_degrees'],
                        'final_max_degrees': report['final_summary']['max_degrees'],
                        'actual_lp_calls': report['actual_lp_calls'], 'accepted_reference': False,
                        'training_ready': False})
    rid = run.id; run.finish()
    save_new(out/'OFFLINE_WANDB_RECEIPT.json', dict(run_id=rid, mode='offline', server_readback_verified=False))


if __name__ == '__main__':
    main()