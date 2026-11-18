"""Finite native-frame angular-tail refinement; NOT a reference-motion solver.

No world root, terrain, contact labels or collision exemptions. Original target
joint and native-edge velocity bounds are hard constraints, not penalties.
"""
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog


THRESHOLD = np.deg2rad(25.)  # Tail diagnostic, never motion acceptance.


def angles(u, desired):
    return np.arctan2(np.linalg.norm(np.cross(u, desired), axis=2), np.sum(u * desired, axis=2))


def objective(errors, departure):
    return float(errors.sum() + 10. * np.maximum(errors - THRESHOLD, 0.).sum()
                 + 100. * errors.max() + .001 * np.abs(departure).sum())


def window_rows(model, current, first, last):
    if current.shape != (997, 29) or not 0 <= first <= last < 997 or last-first >= 93:
        raise ValueError('Complete native sequence and bounded contiguous window required')
    n = last-first+1
    rows, cols, values, offsets = [], [], [], []
    for edge in range(max(0, first-1), min(996, last+1)):
        for j in range(29):
            row, const = len(offsets), 0.
            for frame, sign in ((edge, -1.), (edge+1, 1.)):
                if first <= frame <= last:
                    rows.append(row); cols.append((frame-first)*29+j); values.append(sign)
                else:
                    const += sign * current[frame, j]
            offsets.append(const)
    d = sparse.coo_matrix((values, (rows, cols)), shape=(len(offsets), n*29)).tocsr()
    offset = np.asarray(offsets)
    speed = np.tile(model.vmax/90., len(offset)//29)
    eye = sparse.eye(n*29, format='csr')
    return sparse.vstack([eye, -eye, d, -d], format='csr'), np.r_[
        np.tile(model.upper, n), -np.tile(model.lower, n), speed-offset, speed+offset]


def plan(initial_errors):
    starts = np.rint(np.linspace(0, 904, 17)).astype(int).tolist()
    base = [(s, s+92) for s in starts]
    # Fixed once from the input; not an adaptive unlimited retry schedule.
    frame_score = initial_errors.max(axis=1)
    chosen = []
    for f in np.argsort(-frame_score):
        start = min(904, max(0, int(f)-46))
        if all(abs(start-a) >= 46 for a, _ in chosen):
            chosen.append((start, start+92))
        if len(chosen) == 4:
            break
    schedule = []
    for sweep in range(8):
        schedule.extend((sweep, 'coverage', a, b) for a, b in (base if sweep % 2 == 0 else base[::-1]))
        if sweep in (3, 7):
            schedule.extend((sweep, 'initial_tail', a, b) for a, b in chosen)
    if len(schedule) > 144:
        raise ValueError('Finite plan overflow')
    return schedule


def solve(model, initial, desired, *, progress=None, seconds=90.):
    if not 0 < seconds <= 90.:
        raise ValueError('Finite algorithm wall limit required')
    initial = np.asarray(initial, dtype=float)
    desired = np.asarray(desired, dtype=float)
    if initial.shape != (997, 29) or desired.shape != (997, 8, 3):
        raise ValueError('All997 native frames and29 named body joints required')
    if not np.isfinite(initial).all() or not np.isfinite(desired).all():
        raise ValueError('Finite inputs required')
    if not np.allclose(np.linalg.norm(desired, axis=2), 1., atol=1e-10, rtol=0):
        raise ValueError('Explicit unit anatomical directions required')
    if max(0., np.max(model.lower-initial), np.max(initial-model.upper),
           np.max(np.abs(np.diff(initial, axis=0))*90-model.vmax)) > 1e-6:
        raise ValueError('Hard feasible input required; no seed clipping')
    q = initial.copy()
    errors = angles(model.evaluate(q), desired)
    original_errors = errors.copy()
    schedule = plan(errors)
    value = objective(errors, q-initial)
    initial_objective = value
    history = []
    visited = np.zeros(997, int)
    start_time = time.monotonic()
    termination = 'fixed_plan_completed'
    for index, (sweep, role, first, last) in enumerate(schedule):
        if time.monotonic()-start_time >= seconds-1.5:
            termination = 'finite_algorithm_wall_budget'; break
        piece, want = q[first:last+1].copy(), desired[first:last+1]
        n, w, m = len(piece), len(piece)*29, len(piece)*8
        u, du = model.evaluate(piece, True)
        theta = angles(u, want)
        sine = np.linalg.norm(np.cross(u, want), axis=2)
        # d acos(dot(u,d))/dq; at zero choose zero subgradient.
        grad = -np.einsum('nkj,nkjc->nkc', want, du) / np.maximum(sine[..., None], 1e-8)
        grad[sine < 1e-8] = 0.
        a = sparse.block_diag([sparse.csr_matrix(x) for x in grad], format='csr')
        g, h = window_rows(model, q, first, last)
        eye, tail_eye = sparse.eye(w, format='csr'), sparse.eye(m, format='csr')
        z_wm, z_mw = sparse.csr_matrix((w, m)), sparse.csr_matrix((m, w))
        z_w1, z_m1 = sparse.csr_matrix((w, 1)), sparse.csr_matrix((m, 1))
        z_h = sparse.csr_matrix((g.shape[0], m+w+1))
        # Variables: delta, tail epigraphs, |q-input| epigraphs, GLOBAL max angle.
        hard = sparse.hstack([g, z_h], format='csr')
        tail = sparse.hstack([a, -tail_eye, z_mw, z_m1], format='csr')
        maximum = sparse.hstack([a, sparse.csr_matrix((m, m+w)), -np.ones((m, 1))], format='csr')
        nonnegative_angle = sparse.hstack([-a, sparse.csr_matrix((m, m+w+1))], format='csr')
        reg = sparse.vstack([
            sparse.hstack([eye, z_wm, -eye, z_w1]),
            sparse.hstack([-eye, z_wm, -eye, z_w1])], format='csr')
        departure = (piece-initial[first:last+1]).reshape(-1)
        matrix = sparse.vstack([hard, tail, maximum, nonnegative_angle, reg], format='csr')
        rhs = np.r_[h-g@piece.reshape(-1), THRESHOLD-theta.reshape(-1),
                    -theta.reshape(-1), theta.reshape(-1), -departure, departure]
        outside = np.r_[errors[:first].reshape(-1), errors[last+1:].reshape(-1)]
        outside_max = float(outside.max()) if len(outside) else 0.
        cost = np.r_[np.asarray(a.sum(axis=0)).reshape(-1), np.full(m, 10.), np.full(w, .001), 100.]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=OptimizeWarning, message='Unrecognized options detected.*')
            result = linprog(cost, A_ub=matrix, b_ub=rhs,
                bounds=[(-.15, .15)]*w + [(0., None)]*(m+w) + [(outside_max, None)],
                method='highs', options={'time_limit': 1., 'maxiter': 10000, 'threads': 2, 'parallel': False})
        row = dict(index=index, sweep=sweep, role=role, first=first, last=last,
                   lp_status=int(result.status), accepted=False)
        visited[first:last+1] += 1
        if not result.success:
            history.append(row); termination = 'LP_stopped_status_'+str(result.status)
            if progress: progress(row)
            break
        delta = result.x[:w].reshape(n, 29)
        for alpha in (1., .5, .25, .125, .0625):
            trial = piece + alpha*delta
            if np.max(g@trial.reshape(-1)-h) > 1e-6:
                raise ValueError('Hard edge/position feasibility lost; no clipping allowed')
            local_errors = angles(model.evaluate(trial), want)
            combined = errors.copy(); combined[first:last+1] = local_errors
            depart_sum = float(np.abs(q-initial).sum()-np.abs(piece-initial[first:last+1]).sum()
                               + np.abs(trial-initial[first:last+1]).sum())
            trial_value = float(combined.sum()+10*np.maximum(combined-THRESHOLD, 0.).sum()
                                +100*combined.max()+.001*depart_sum)
            # Never trade a worse global angular maximum for a better aggregate.
            if trial_value < value-1e-9 and combined.max() <= errors.max()+1e-10:
                row.update(accepted=True, alpha=alpha, objective_before=value, objective_after=trial_value,
                           max_before_degrees=float(np.rad2deg(errors.max())),
                           max_after_degrees=float(np.rad2deg(combined.max())))
                q[first:last+1], errors, value = trial, combined, trial_value
                break
        history.append(row)
        if progress: progress(row)
    position = float(max(0., np.max(model.lower-q), np.max(q-model.upper)))
    speed = float(max(0., np.max(np.abs(np.diff(q, axis=0))*90-model.vmax)))
    if max(position, speed) > 1e-6 or abs(objective(angles(model.evaluate(q), desired), q-initial)-value) > 1e-7:
        raise ValueError('Final direct full-sequence constraint/objective verification failed')
    return q, dict(planned_lp_calls=len(schedule), actual_lp_calls=len(history),
                   accepted_updates=sum(r['accepted'] for r in history), history=history,
                   termination=termination, elapsed_seconds=time.monotonic()-start_time,
                   initial_objective=initial_objective, final_objective=value,
                   initial_angles_degrees=np.rad2deg(original_errors).tolist(),
                   final_angles_degrees=np.rad2deg(errors).tolist(),
                   position_violation_rad=position, velocity_violation_rad_s=speed,
                   visited_frames=int((visited > 0).sum()),
                   changed_frames=int(np.any(np.abs(q-initial)>1e-9, axis=1).sum()),
                   native_frames=997, native_framerate=90, native_velocity_edges=996,
                   numerical_threads=2, per_lp_seconds=1., trust_radius_rad=.15,
                   world_root_optimized=False, collision_checked=False, material_support_verified=False,
                   accepted_reference=False, training_ready=False)