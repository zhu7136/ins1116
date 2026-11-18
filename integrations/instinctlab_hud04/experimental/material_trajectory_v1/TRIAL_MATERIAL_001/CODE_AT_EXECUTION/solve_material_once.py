"""One bounded CPU 35D material+geometry experiment. Never an acceptance writer."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, OptimizeWarning
from scipy.spatial.transform import Rotation
from material_problem import (HERE, REPO, CONTRACT, DIAGNOSTICS, SEED, SEED_SHA, TERRAIN,
                              problem, perturb, hard_rows, feasibility, sha)

DIRECTORY = HERE/'TRIAL_MATERIAL_001'


def write(path, data):
    with Path(path).open('x') as f: json.dump(data, f, indent=2, allow_nan=False)


def schedule():
    starts = np.rint(np.linspace(0, 936, 17)).astype(int).tolist()
    return [(s, s+60) for s in starts]+[(s, s+60) for s in starts[::-1]]


def material_stats(model, q, roots, desired, support):
    value = model.evaluate(q, roots)
    error = np.linalg.norm((value[:, :18]-desired[:, :18]).reshape(997, 6, 3), axis=2)
    normal = np.rad2deg(np.arccos(np.clip(np.einsum('nki,nki->nk', value[:, 18:].reshape(997, 6, 3),
                             desired[:, 18:].reshape(997, 6, 3)), -1, 1)))
    return dict(all_material_position_mean_m=float(error.mean()), all_material_position_max_m=float(error.max()),
                proposed_support_position_mean_m=float(error[support].mean()),
                proposed_support_position_max_m=float(error[support].max()),
                proposed_support_normal_mean_deg=float(normal[support].mean()),
                proposed_support_normal_max_deg=float(normal[support].max()))


def solve(model, q, roots, want, weights, support, body, guard, progress):
    q = q.copy(); roots = roots.copy(); seed = q.copy()
    coverage = np.zeros(997, int); history = []; start = time.monotonic()
    for number, (first, last) in enumerate(schedule()):
        if time.monotonic()-start > 140: break
        frames = slice(first, last+1); piece, r = q[frames], roots[frames]
        n = len(piece); nv = 35*n; nq = 29*n
        value, j = model.evaluate(piece, r, True)
        res = ((value-want[frames])*weights[frames]).reshape(-1)
        a = sparse.block_diag([sparse.csr_matrix(v*w[:, None]) for v, w in zip(j, weights[frames])], format='csr')
        hard, rhs = hard_rows(model, q, first, last)
        select = sparse.coo_matrix((np.ones(nq), (np.arange(nq), [35*i+6+k for i in range(n) for k in range(29)])), shape=(nq, nv)).tocsr()
        # Fixed original joint regularizer; root continuity is an additional soft
        # first-difference term against the existing full trajectory, not a limit.
        departure = (piece-seed[frames]).reshape(-1)
        sm = sparse.diags([-np.ones(n-1), np.ones(n-1)], [0, 1], shape=(n-1, n), format='csr')
        translate = sparse.hstack([sparse.eye(3), sparse.csr_matrix((3, 32))], format='csr')
        smooth = sparse.kron(sm, translate, format='csr')*.1
        sr = np.diff(r[:, :3, 3], axis=0).reshape(-1)*.1
        all_a = sparse.vstack([a, .001*select, smooth], format='csr')
        all_res = np.r_[res, .001*departure, sr]; na = len(all_res)
        gaps, gj, records, counts = body.evaluate(model, piece, r, True)
        same_link = np.array([row['kind']=='self' and body.shapes[row['first']].link==body.shapes[row['second']].link for row in records], dtype=bool)
        movable = np.asarray(gj.getnnz(axis=1)>0).ravel() & ~same_link
        gl, jl = gaps[movable], gj[movable]; ng = len(gl)
        matrix = sparse.vstack([
            sparse.hstack([hard, sparse.csr_matrix((len(rhs), na+ng))]),
            sparse.hstack([all_a, -sparse.eye(na), sparse.csr_matrix((na, ng))]),
            sparse.hstack([-all_a, -sparse.eye(na), sparse.csr_matrix((na, ng))]),
            sparse.hstack([-1000*jl, sparse.csr_matrix((ng, na)), -sparse.eye(ng)])], format='csr')
        b = np.r_[rhs, -all_res, all_res, 1000*(gl-body.margin)]
        trust = np.tile(np.r_[np.full(3, .015), np.full(3, .04), np.full(29, .08)], n)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=OptimizeWarning, message='Unrecognized options detected.*')
            lp = linprog(np.r_[np.zeros(nv), np.ones(na+ng)], A_ub=matrix, b_ub=b,
                         bounds=np.c_[-trust, trust].tolist()+[(0, None)]*(na+ng), method='highs',
                         options={'time_limit': 1.5, 'maxiter': 12000, 'threads': 2, 'parallel': False})
        coverage[frames] += 1
        row = dict(update=number, first=first, last=last, lp_status=int(lp.status), accepted=False,
                   geometry_records=len(gaps), old_geometry_counts=counts, backtracking=[])
        if not lp.success:
            row['stop_reason'] = 'LP failure; no automatic retry'; history.append(row); progress(row); break
        step = lp.x[:nv].reshape(n, 35)
        old_merit = float(abs(all_res).sum()+1000*np.maximum(body.margin-gaps, 0).sum())
        old_state = None
        for alpha in [1., .5, .25, .125, .0625]:
            tq, tr = piece+alpha*step[:, 6:], perturb(r, alpha*step[:, :6])
            if np.max(hard@(alpha*step.reshape(-1))-rhs) > 1e-6:
                raise ValueError('Original hard constraints lost; no clipping')
            newgaps, newrecords, newcounts = body.evaluate(model, tq, tr, False)
            newres = (model.evaluate(tq, tr)-want[frames])*weights[frames]
            merit = float(abs(newres).sum()+.001*abs(tq-seed[frames]).sum()+
                          .1*abs(np.diff(tr[:, :3, 3], axis=0)).sum()+1000*np.maximum(body.margin-newgaps, 0).sum())
            detail = dict(alpha=alpha, merit=merit, old_merit=old_merit, fresh_all43_all903=True)
            row['backtracking'].append(detail)
            if merit >= old_merit-1e-9: continue
            if old_state is None: old_state = guard.scan(model, piece, r, np.arange(first, last+1))
            state = guard.scan(model, tq, tr, np.arange(first, last+1))
            passed, check = guard.compare(old_state, state)
            detail.update(endpoint_guard=check)
            if not passed: continue
            q[frames], roots[frames] = tq, tr
            row.update(accepted=True, alpha=alpha, old_merit=old_merit, merit=merit,
                       max_root_translation_step_m=float(abs(alpha*step[:, :3]).max()),
                       max_root_rotation_step_rad=float(abs(alpha*step[:, 3:6]).max()),
                       max_joint_step_rad=float(abs(alpha*step[:, 6:]).max()), new_geometry_counts=newcounts)
            break
        history.append(row); progress(row)
    limits = feasibility(model, q)
    if max(limits.values()) > 1e-6: raise ValueError('Full native limit feasibility lost')
    return q, roots, dict(actual_lp_calls=len(history), accepted_updates=sum(r['accepted'] for r in history),
                         visited_frames=int((coverage>0).sum()), coverage_min=int(coverage.min()),
                         algorithm_wall_s=time.monotonic()-start, termination='finite_budget_NOT_convergence', **limits)


def worker():
    import wandb
    intent = json.loads((DIRECTORY/'INTENT.json').read_text())
    m, q, roots, want, weights, support, body, guard, _ = problem()
    config = dict(intent)
    provenance = {k: os.environ[e] for k, e in [('thread_id', 'WANDB_ARIA_THREAD_ID'), ('turn_id', 'WANDB_ARIA_TURN_ID')] if e in os.environ}
    if provenance: config['_wb_agent'] = provenance
    run = wandb.init(entity='zhul136-university-of-science-and-technology-of-china', project='ins1116', id=intent['wandb_run_id'],
                     name='hud04-material-trajectory-001-NOT-ACCEPTED', job_type='cpu-material-retarget',
                     dir=str(DIRECTORY), config=config, settings=wandb.Settings(disable_git=True, save_code=False))
    initial = material_stats(m, q, roots, want, support)
    try:
        with (DIRECTORY/'PROGRESS.jsonl').open('x') as f:
            def progress(row):
                f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush()
                print(json.dumps({k: row[k] for k in ['update', 'first', 'last', 'lp_status', 'accepted']}), flush=True)
                run.log({'lp_call': row['update']+1, 'update_accepted': row['accepted']})
            q, roots, report = solve(m, q, roots, want, weights, support, body, guard, progress)
        final = material_stats(m, q, roots, want, support)
        # Independent original-solid classifications of every native frame.
        final_state = guard.scan(m, q, roots, np.arange(997))
        counts = {kind+'_'+status: sum(k[0]==kind for k in getattr(final_state, status))
                  for kind in ['self', 'terrain'] for status in ['strict', 'unresolved']}
        exported = report['visited_frames'] == 997
        if exported:
            quat = Rotation.from_matrix(roots[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
            for i in range(1, 997):
                if quat[i]@quat[i-1] < 0: quat[i] *= -1
            np.savez_compressed(DIRECTORY/'STEP1_MATERIAL_CANDIDATE_UNACCEPTED.npz', joint_pos=q,
                                joint_names=np.array(m.names), base_pos_w=roots[:, :3, 3], base_quat_w=quat,
                                framerate=np.array(90), frame_indices=np.arange(997), accepted_reference=np.array(False),
                                training_ready=np.array(False), artifact_role=np.array('UNACCEPTED_MATERIAL_DESIGN_NOT_TRAINING_INPUT'))
        report.update(initial=initial, final=final, final_geometry=counts, motion_exported=exported,
                      unassigned_support_frames=346, contact_or_flight_inferred=False, physical_effort_verified=False,
                      swept_motion_verified=False, original_geometry_unchanged=True, runtime_pair_policy_verified=False,
                      source_commit=intent['source_commit'], seed_sha256=SEED_SHA, accepted_reference=False,
                      training_ready=False, gpu_submissions=0, ppo_updates=0)
        guard.verify_inputs()
        write(DIRECTORY/'RESULT.json', report)
        run.summary.update({k: v for k, v in report.items() if isinstance(v, (int, float, bool, str))})
        run.finish()
    except BaseException:
        run.finish(exit_code=1); raise


def main():
    if len(sys.argv) == 2 and sys.argv[1] == '--worker': return worker()
    tests = Path('/workspace/hud04-material-trajectory-pretests11.log').read_text()
    if 'Ran 11 tests' not in tests or not tests.rstrip().endswith('OK'): raise ValueError('Pretests required')
    DIRECTORY.mkdir(exist_ok=False)
    intent = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  user_authority='2026-11-18T03:57:02Z continue toward acceptable reference; finite CPU trial only',
                  source_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                  seed_sha256=SEED_SHA, contract_sha256=sha(CONTRACT), diagnostics_sha256=sha(DIAGNOSTICS),
                  frames=997, fps=90, body_dof=29, head_dof=0, maximum_lp_calls=34, algorithm_wall_limit_s=140,
                  worker_wall_limit_s=190, lp_wall_limit_s=1.5, lp_maxiter=12000, automatic_retries=0,
                  original_limits_and_geometry_unchanged=True, no_automatic_collision_exemptions=True,
                  physical_effort_limit_acceptance_pending=True, gpu_submissions=0, ppo_updates=0,
                  accepted_reference=False, training_ready=False, wandb_run_id=uuid.uuid4().hex[:8])
    write(DIRECTORY/'INTENT.json', intent)
    snap = DIRECTORY/'CODE_AT_EXECUTION'; snap.mkdir()
    for file in sorted(HERE.glob('*.py')):
        (snap/file.name).write_bytes(file.read_bytes())
    write(DIRECTORY/'CODE_SHA256.json', {f.name: sha(f) for f in snap.iterdir()})
    env = os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    start = time.monotonic(); timed_out = False
    with (DIRECTORY/'EXECUTION.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__)), '--worker'], env=env, stdout=log, stderr=subprocess.STDOUT)
        try: child.wait(timeout=190)
        except subprocess.TimeoutExpired:
            timed_out = True; child.terminate()
            try: child.wait(timeout=2)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=2)
    receipt = dict(child_exit=child.returncode, child_reaped=child.poll() is not None, timed_out=timed_out,
                   wall_s=time.monotonic()-start, result_exists=(DIRECTORY/'RESULT.json').exists(),
                   gpu_submissions=0, ppo_updates=0, accepted_reference=False)
    write(DIRECTORY/'SUPERVISION.json', receipt); print(json.dumps(receipt, indent=2), flush=True)
    if timed_out or child.returncode: raise SystemExit(1)


if __name__ == '__main__': main()