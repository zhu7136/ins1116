"""Euclidean shared-node quintics. NOT SO(3), dynamics, or acceptance."""
import math
import numpy as np


def coefficients(position, velocity, acceleration, dt):
    """Return ascending coefficients in normalized segment time u in [0,1].

    Each global node has ONE position/velocity/acceleration, reused by both
    neighboring segments. No window writes or duplicated interface states.
    Derivatives supplied in physical seconds, not normalized u.
    """
    p, v, a = (np.asarray(x, dtype=float) for x in (position, velocity, acceleration))
    if p.ndim != 2 or p.shape != v.shape or p.shape != a.shape or len(p) < 2:
        raise ValueError('Expected matching [global_node, Euclidean_dimension] arrays')
    if not np.isfinite(dt) or dt <= 0 or not all(np.isfinite(x).all() for x in (p, v, a)):
        raise ValueError('Finite states and positive finite dt required')
    c0, c1, c2 = p[:-1], dt * v[:-1], .5 * dt**2 * a[:-1]
    r0 = p[1:] - c0 - c1 - c2
    r1 = dt * v[1:] - c1 - 2 * c2
    r2 = dt**2 * a[1:] - 2 * c2
    return np.stack([c0, c1, c2,
                     10*r0 - 4*r1 + .5*r2,
                     -15*r0 + 7*r1 - r2,
                     6*r0 - 3*r1 + .5*r2], axis=1)


def derivative_coefficients(c, dt, order):
    c = np.asarray(c, dtype=float)
    if not isinstance(order, int) or not 0 <= order <= 5 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Derivative order 0..5 and positive dt required')
    if c.ndim != 3 or c.shape[1] != 6 or not np.isfinite(c).all():
        raise ValueError('Expected finite [segment, 6, dimension] coefficients')
    factors = np.array([math.factorial(j) / math.factorial(j-order)
                        for j in range(order, 6)]) / dt**order
    return c[:, order:, :] * factors[None, :, None]


def evaluate(c, dt, u, order=0):
    if not np.isfinite(u) or not 0 <= u <= 1:
        raise ValueError('Segment u must be in [0,1]')
    dc = derivative_coefficients(c, dt, order)
    out = np.zeros((dc.shape[0], dc.shape[2]))
    for j in range(dc.shape[1]-1, -1, -1):
        out = out*u + dc[:, j, :]
    return out


def component_extrema(c, dt, order=0):
    """Numerical stationary-root extrema for each segment/component.

    Includes endpoints; uses polynomial roots, NOT interval-certified bounds.
    Returns min,max,argmin,argmax. Never a collision/dynamics certificate.
    """
    dc = derivative_coefficients(c, dt, order)
    shape = (len(dc), dc.shape[2])
    low, high, arglow, arghigh = (np.empty(shape) for _ in range(4))
    for i in range(shape[0]):
        for j in range(shape[1]):
            poly = dc[i, :, j]
            deriv = np.arange(1, len(poly))*poly[1:]
            deriv = np.trim_zeros(deriv, 'b')
            roots = np.polynomial.polynomial.polyroots(deriv) if len(deriv) > 1 else []
            us = [0., 1.] + [float(z.real) for z in roots
                            if abs(z.imag) < 1e-9 and 0 < z.real < 1]
            vals = np.polynomial.polynomial.polyval(us, poly)
            lo, hi = int(np.argmin(vals)), int(np.argmax(vals))
            low[i,j], high[i,j] = vals[lo], vals[hi]
            arglow[i,j], arghigh[i,j] = us[lo], us[hi]
    return low, high, arglow, arghigh