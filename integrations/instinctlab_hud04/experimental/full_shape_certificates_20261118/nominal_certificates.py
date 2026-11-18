"""Nominal finite certificates, not a motion optimizer or physics acceptance.

Separation is certified by disjoint support intervals. Strict intersection is
certified ONLY by a common point inside both original solids by > tolerance.
Failure to find either is UNRESOLVED, never clearance. Cylinders remain cylinders.
"""
from dataclasses import dataclass
import numpy as np


def finite(value):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite geometry')
    return a


def rigid(value):
    t = finite(value)
    if t.ndim != 3 or t.shape[1:] != (4, 4) or not 1 <= len(t) <= 1000:
        raise ValueError('Bounded batched rigid transforms required')
    if not np.allclose(t[:, 3], [0, 0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError('Homogeneous transform required')
    r = t[:, :3, :3]
    if not np.allclose(r.swapaxes(1, 2) @ r, np.eye(3), atol=1e-10, rtol=0) or not np.allclose(np.linalg.det(r), 1, atol=1e-10, rtol=0):
        raise ValueError('SO3 rotation required')
    return t


@dataclass
class Solid:
    identity: str
    kind: str
    dimensions: np.ndarray
    vertices: object = None
    planes: object = None

    def __post_init__(self):
        self.dimensions = finite(self.dimensions)
        count = {'sphere': 1, 'cylinder': 2, 'box': 3, 'mesh': 0}.get(self.kind)
        if count is None or self.dimensions.shape != (count,) or np.any(self.dimensions <= 0):
            raise ValueError('Explicit positive nominal solid dimensions required')
        if self.kind == 'mesh':
            self.vertices, self.planes = finite(self.vertices), finite(self.planes)
            if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not 4 <= len(self.vertices) <= 20000:
                raise ValueError('Bounded convex hull vertices required')
            if self.planes.ndim != 2 or self.planes.shape[1] != 4 or not 4 <= len(self.planes) <= 2000:
                raise ValueError('Bounded convex hull planes required')
            if not np.allclose(np.linalg.norm(self.planes[:, :3], axis=1), 1, atol=1e-10, rtol=0):
                raise ValueError('Normalized hull planes required')
            # Ensure every supplied vertex is within every supplied halfspace.
            if np.max(self.vertices @ self.planes[:, :3].T + self.planes[:, 3]) > 1e-9:
                raise ValueError('Inconsistent hull certificate')

    def interval(self, directions, poses):
        t = rigid(poses); d = finite(directions)
        if d.ndim != 3 or d.shape[0] != len(t) or d.shape[2] != 3 or not 1 <= d.shape[1] <= 40:
            raise ValueError('Batched bounded support directions required')
        local = np.einsum('nki,nij->nkj', d, t[:, :3, :3])
        center = np.einsum('nki,ni->nk', d, t[:, :3, 3])
        if self.kind == 'sphere':
            ext = np.linalg.norm(local, axis=2) * self.dimensions[0]
        elif self.kind == 'box':
            ext = abs(local) @ self.dimensions
        elif self.kind == 'cylinder':
            ext = np.linalg.norm(local[:, :, :2], axis=2) * self.dimensions[0] + abs(local[:, :, 2]) * self.dimensions[1]
        else:
            values = np.einsum('nki,vi->nkv', local, self.vertices)
            return center + values.min(axis=2), center + values.max(axis=2)
        return center - ext, center + ext

    def margin(self, points, poses):
        t = rigid(poses); p = finite(points)
        if p.shape != (len(t), 3):
            raise ValueError('One common point per rigid pose required')
        x = np.einsum('ni,nij->nj', p - t[:, :3, 3], t[:, :3, :3])
        if self.kind == 'sphere':
            return self.dimensions[0] - np.linalg.norm(x, axis=1)
        if self.kind == 'box':
            return (self.dimensions - abs(x)).min(axis=1)
        if self.kind == 'cylinder':
            return np.minimum(self.dimensions[0] - np.linalg.norm(x[:, :2], axis=1), self.dimensions[1] - abs(x[:, 2]))
        return -(x @ self.planes[:, :3].T + self.planes[:, 3]).max(axis=1)

    def project_inset(self, points, poses, inset):
        # A witness-search heuristic only. Its convergence is NOT a certificate.
        t = rigid(poses); p = finite(points)
        if p.shape != (len(t), 3) or not np.isfinite(inset) or inset <= 0:
            raise ValueError('Finite positive inset and batched points required')
        if self.kind != 'mesh' and np.any(self.dimensions <= inset):
            raise ValueError('Inset exceeds primitive dimensions')
        x = np.einsum('ni,nij->nj', p - t[:, :3, 3], t[:, :3, :3])
        if self.kind == 'sphere':
            x *= np.minimum(1, (self.dimensions[0] - inset) / np.maximum(np.linalg.norm(x, axis=1), 1e-300))[:, None]
        elif self.kind == 'box':
            x = np.clip(x, -self.dimensions + inset, self.dimensions - inset)
        elif self.kind == 'cylinder':
            x[:, :2] *= np.minimum(1, (self.dimensions[0] - inset) / np.maximum(np.linalg.norm(x[:, :2], axis=1), 1e-300))[:, None]
            x[:, 2] = np.clip(x[:, 2], -self.dimensions[1] + inset, self.dimensions[1] - inset)
        else:
            for plane in self.planes:
                excess = np.maximum(0, x @ plane[:3] + plane[3] + inset)
                x -= excess[:, None] * plane[:3]
        return np.einsum('nij,nj->ni', t[:, :3, :3], x) + t[:, :3, 3]


def classify(first, first_poses, second, second_poses, *, tolerance=1e-8, iterations=64):
    if not np.isfinite(tolerance) or not 1e-10 <= tolerance <= 1e-6 or type(iterations) is not int or not 1 <= iterations <= 128:
        raise ValueError('Explicit bounded numerical tolerances/iterations required')
    a, b = rigid(first_poses), rigid(second_poses)
    if a.shape != b.shape:
        raise ValueError('Both shape poses required for every frame')
    n = len(a)
    axes_a, axes_b = a[:, :3, :3].swapaxes(1, 2), b[:, :3, :3].swapaxes(1, 2)
    delta = b[:, :3, 3] - a[:, :3, 3]
    cross = np.cross(axes_a[:, :, None], axes_b[:, None]).reshape(n, 9, 3)
    d = np.concatenate([np.broadcast_to(np.eye(3), (n, 3, 3)), axes_a, axes_b, cross, delta[:, None]], axis=1)
    d /= np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-300)
    alo, ahi = first.interval(d, a); blo, bhi = second.interval(d, b)
    signed = np.maximum(blo - ahi, alo - bhi)
    best_axis = signed.argmax(axis=1)
    gap = signed[np.arange(n), best_axis]
    separated = gap > tolerance
    strict = np.zeros(n, bool)
    p = (a[:, :3, 3] + b[:, :3, 3]) / 2
    ma, mb = first.margin(p, a), second.margin(p, b)
    strict = (ma > tolerance) & (mb > tolerance)
    steps = 0
    for steps in range(1, iterations + 1):
        ids = np.flatnonzero(~separated & ~strict)
        if not len(ids):
            break
        x = first.project_inset(p[ids], a[ids], tolerance * 16)
        p[ids] = second.project_inset(x, b[ids], tolerance * 16)
        ma[ids] = first.margin(p[ids], a[ids]); mb[ids] = second.margin(p[ids], b[ids])
        strict[ids] = (ma[ids] > tolerance) & (mb[ids] > tolerance)
    if np.any(separated & strict):
        raise ArithmeticError('Contradictory certificates')
    return dict(separated=separated, strict=strict, unresolved=~separated & ~strict,
                separating_gap=gap, separating_direction=d[np.arange(n), best_axis],
                common_point=p, first_inside_margin=ma, second_inside_margin=mb,
                search_iterations=steps, tolerance=tolerance)


def project_triangle(points, triangles):
    p, tri = finite(points), finite(triangles)
    if p.ndim != 2 or p.shape[1] != 3 or tri.shape != (len(p), 3, 3) or not 1 <= len(p) <= 1000:
        raise ValueError('Bounded matched points/triangles required')
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac); norm2 = (normal*normal).sum(axis=1)
    if np.any(norm2 <= 1e-24):
        raise ValueError('Degenerate triangle is unresolved, not clear')
    plane_point = p - (np.sum((p-a)*normal, axis=1)/norm2)[:, None]*normal
    v = plane_point-a
    beta = np.sum(np.cross(v, ac)*normal, axis=1)/norm2
    gamma = np.sum(np.cross(ab, v)*normal, axis=1)/norm2
    inside = (beta >= 0) & (gamma >= 0) & (beta+gamma <= 1)
    candidates = [plane_point]
    distances = [np.where(inside, np.sum((p-plane_point)**2, axis=1), np.inf)]
    for start, end in [(a,b),(b,c),(c,a)]:
        edge=end-start
        s=np.clip(np.sum((p-start)*edge,axis=1)/np.sum(edge*edge,axis=1),0,1)
        q=start+s[:,None]*edge
        candidates.append(q);distances.append(np.sum((p-q)**2,axis=1))
    k=np.argmin(np.stack(distances,axis=1),axis=1)
    return np.stack(candidates,axis=1)[np.arange(len(p)),k]


def classify_triangles(solid, poses, triangles, *, tolerance=1e-8, iterations=64):
    if not np.isfinite(tolerance) or not 1e-10 <= tolerance <= 1e-6 or type(iterations) is not int or not 1 <= iterations <= 128:
        raise ValueError('Explicit bounded numerical tolerances/iterations required')
    t, tri = rigid(poses), finite(triangles)
    if tri.shape != (len(t),3,3): raise ValueError('One triangle per pose required')
    n=len(t); axes=t[:,:3,:3].swapaxes(1,2)
    edges=np.roll(tri,-1,axis=1)-tri
    normal=np.cross(edges[:,0],-edges[:,2])
    if np.any(np.linalg.norm(normal,axis=1)<=1e-12): raise ValueError('Degenerate triangle')
    cross=np.cross(axes[:,:,None],edges[:,None]).reshape(n,9,3)
    delta=tri.mean(axis=1)-t[:,:3,3]
    d=np.concatenate([np.broadcast_to(np.eye(3),(n,3,3)),axes,cross,normal[:,None],delta[:,None]],axis=1)
    d/=np.maximum(np.linalg.norm(d,axis=2,keepdims=True),1e-300)
    lo,hi=solid.interval(d,t); v=np.einsum('nki,nvi->nkv',d,tri)
    gaps=np.maximum(v.min(axis=2)-hi,lo-v.max(axis=2));best=gaps.argmax(axis=1)
    gap=gaps[np.arange(n),best];separated=gap>tolerance
    p=project_triangle(t[:,:3,3],tri);margin=solid.margin(p,t);strict=margin>tolerance
    steps=0
    for steps in range(1,iterations+1):
        ids=np.flatnonzero(~separated & ~strict)
        if not len(ids):break
        p[ids]=project_triangle(solid.project_inset(p[ids],t[ids],tolerance*16),tri[ids])
        margin[ids]=solid.margin(p[ids],t[ids]);strict[ids]=margin[ids]>tolerance
    # Polyhedral fallback: finite polygon clipping supplies a stronger witness
    # than alternating point projections. Never replace a cylinder by this hull.
    if solid.kind in ('box','mesh'):
        if solid.kind=='box':
            planes=np.concatenate([np.eye(3),-np.eye(3)])
            planes=np.column_stack([planes,-np.tile(solid.dimensions,2)])
        else:planes=solid.planes
        for k in np.flatnonzero(~separated & ~strict):
            polygon=(tri[k]-t[k,:3,3])@t[k,:3,:3]
            for plane in planes:
                if not len(polygon):break
                signed=polygon@plane[:3]+plane[3];output=[]
                for j,start in enumerate(polygon):
                    end=polygon[(j+1)%len(polygon)];a=signed[j];b=signed[(j+1)%len(polygon)]
                    if a<=0:output.append(start)
                    if (a<=0)!=(b<=0):output.append(start+a/(a-b)*(end-start))
                polygon=np.asarray(output,float).reshape(-1,3)
                if len(polygon)>len(planes)+3:raise ArithmeticError('Clipped polygon budget exceeded')
            if len(polygon):
                candidate=polygon.mean(axis=0)@t[k,:3,:3].T+t[k,:3,3]
                m=solid.margin(candidate[None],t[k:k+1])[0]
                if m>tolerance:p[k]=candidate;margin[k]=m;strict[k]=True
    if np.any(separated & strict):raise ArithmeticError('Contradictory surface certificates')
    residual=np.linalg.norm(p-project_triangle(p,tri),axis=1)
    if np.any(residual[strict]>1e-10):raise ArithmeticError('Common point not on finite triangle')
    return dict(separated=separated,strict=strict,unresolved=~separated & ~strict,
                separating_gap=gap,common_point=p,solid_inside_margin=margin,triangle_residual=residual,
                search_iterations=steps,tolerance=tolerance)