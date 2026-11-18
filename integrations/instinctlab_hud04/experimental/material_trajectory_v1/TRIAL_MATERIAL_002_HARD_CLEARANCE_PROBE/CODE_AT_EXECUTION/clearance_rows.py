"""Preserve nearby separating planes as hard SQP rows, not a collision waiver.

Nonlinear endpoint guard remains mandatory. Positive support gap is a valid
separation witness; negative gap is NOT penetration depth. Rows only constrain
currently separated near pairs. Strict/unresolved starting pairs stay rejected
by physical acceptance and recorded by the independent guard.
"""
import numpy as np
from scipy import sparse


def build(body, kernel, model, q, roots, *, padding=.12):
    if padding != .12: raise ValueError('Fixed prototype activation distance, NOT an acceptance tolerance')
    fk, axes = model.model.fk(q)
    world = {link: roots@t for link, t in fk.items()}
    rows, cols, data, rhs, records = [], [], [], [], []

    def add(frame, a, b, kind, gap, direction, pa, pb):
        if not 1e-8 < gap < padding: return
        j = np.zeros(35)
        for number, point, sign in [(a, pa, 1), (b, pb, -1)]:
            if sign == -1 and kind == 'terrain': continue
            link = body.shapes[number].link
            j[:3] += sign*direction
            for c in range(3): j[3+c] += sign*direction@np.cross(np.eye(3)[c], point-roots[frame, :3, 3])
            for c, name in enumerate(model.names):
                if name not in model.model.ancestors[link]: continue
                pivot, axis = axes[name]
                pivot = roots[frame, :3, :3]@pivot[frame]+roots[frame, :3, 3]
                axis = roots[frame, :3, :3]@axis[frame]
                j[6+c] += sign*direction@np.cross(axis, point-pivot)
        nz = np.flatnonzero(abs(j)>1e-14)
        rows.extend([len(rhs)]*len(nz)); cols.extend((35*frame+nz).tolist()); data.extend((-j[nz]).tolist())
        # Retain an existing positive gap (or 0.1 mm if it is larger), rather
        # than allowing the soft collision penalty to trade it away.
        rhs.append(gap-min(gap, .0001))
        ta = world[body.shapes[a].link][frame]
        second_local = pb if kind == 'terrain' else (pb-world[body.shapes[b].link][frame,:3,3])@world[body.shapes[b].link][frame,:3,:3]
        records.append(dict(frame=frame, kind=kind, first=a, second=b, gap=gap, normal=direction.tolist(),
                            first_local=((pa-ta[:3,3])@ta[:3,:3]).tolist(), second_local=second_local.tolist()))

    for f in range(len(q)):
        poses = [world[s.link][f]@s.local for s in body.shapes]
        low, high = [], []
        for s, pose in zip(body.shapes, poses):
            v = kernel['supports'](s, pose, np.r_[np.eye(3), -np.eye(3)])
            high.append(np.diag(v[:3])); low.append(np.diag(v[3:]))
        low, high = np.array(low), np.array(high)
        for a, s in enumerate(body.shapes):
            ids = np.flatnonzero(np.all(body.hi >= low[a]-padding, axis=1)&np.all(body.lo <= high[a]+padding, axis=1))
            gaps, d, pa, pb = kernel['triangle_projection'](s, poses[a], body.tri[ids])
            for k in np.flatnonzero((gaps>1e-8)&(gaps<padding)):
                add(f, a, int(ids[k]), 'terrain', float(gaps[k]), d[k], pa[k], pb[k])
        for a, b in body.pairs:
            if body.shapes[a].link == body.shapes[b].link: continue
            if np.any(high[a]+padding < low[b]) or np.any(high[b]+padding < low[a]): continue
            gap, d, pa, pb = kernel['pair_projection'](body.shapes[a], poses[a], body.shapes[b], poses[b])
            add(f, a, b, 'self', gap, d, pa, pb)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(rhs), len(q)*35)).tocsr()
    return matrix, np.array(rhs), records