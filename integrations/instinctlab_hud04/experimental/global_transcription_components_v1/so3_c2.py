"""Local-chart SO(3) quintics with shared WORLD angular velocity/acceleration.

R(u)=R0 Exp(phi(u)); omega_world=R0 J_left(phi) phi_dot.
alpha_world=R0 (J_left phi_ddot + D J_left[phi_dot] phi_dot).
This is a kinematic component, NOT a global solver or a feasibility certificate.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from euclidean_c2 import coefficients, evaluate


def vector(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError('Finite length-three vector required')
    return x


def skew(x):
    x,y,z = vector(x)
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def left_jacobian_and_directional(phi, direction):
    """Analytic J_l(phi) and D J_l(phi)[direction], stable at zero."""
    phi, direction = vector(phi), vector(direction)
    s = float(phi@phi)
    dot = float(phi@direction)
    if s < 1e-4:
        A = .5-s/24+s*s/720-s**3/40320+s**4/3628800
        B = 1/6-s/120+s*s/5040-s**3/362880+s**4/39916800
        Ad = 2*dot*(-1/24+s/360-s*s/13440+s**3/907200)
        Bd = 2*dot*(-1/120+s/2520-s*s/120960+s**3/9979200)
    else:
        t = np.sqrt(s)
        A = (1-np.cos(t))/s
        B = (t-np.sin(t))/(s*t)
        Ad = (t*np.sin(t)-2*(1-np.cos(t)))/(s*s)*dot
        Bd = (t*(1-np.cos(t))-3*(t-np.sin(t)))/(s*s*t)*dot
    K,D = skew(phi),skew(direction)
    J = np.eye(3)+A*K+B*(K@K)
    dJ = Ad*K+A*D+Bd*(K@K)+B*(D@K+K@D)
    return J,dJ


class SharedSO3Quintic:
    def __init__(self, rotations, omega_world, alpha_world, dt):
        R,w,a = (np.array(x,dtype=float,copy=True) for x in (rotations,omega_world,alpha_world))
        if (R.ndim != 3 or R.shape[1:] != (3,3) or len(R)<2
                or w.shape != (len(R),3) or a.shape != w.shape
                or not all(np.isfinite(x).all() for x in (R,w,a))):
            raise ValueError('Finite [N,3,3] SO3 nodes and [N,3] WORLD rates required')
        if not np.isfinite(dt) or dt<=0:
            raise ValueError('Positive finite dt required')
        if (np.max(np.abs(R.transpose(0,2,1)@R-np.eye(3)))>1e-8
                or np.max(np.abs(np.linalg.det(R)-1))>1e-8):
            raise ValueError('Input matrices must be rotations; no silent projection')
        c = []
        for i in range(len(R)-1):
            phi = Rotation.from_matrix(R[i].T@R[i+1]).as_rotvec()
            if np.linalg.norm(phi)>=np.pi-1e-6:
                raise ValueError('Near-pi endpoint chart requires explicit recharting')
            J,_ = left_jacobian_and_directional(phi,np.zeros(3))
            pd = np.linalg.solve(J,R[i].T@w[i+1])
            _,dJ = left_jacobian_and_directional(phi,pd)
            pdd = np.linalg.solve(J,R[i].T@a[i+1]-dJ@pd)
            c.append(coefficients(np.array([np.zeros(3),phi]),
                       np.array([R[i].T@w[i],pd]),
                       np.array([R[i].T@a[i],pdd]),dt)[0])
        self.coefficients = np.array(c)
        self.rotations = R
        self.dt = float(dt)

    def state(self, segment, u):
        if not isinstance(segment,(int,np.integer)) or not 0<=segment<len(self.coefficients):
            raise ValueError('Invalid segment')
        c = self.coefficients[segment:segment+1]
        p,pd,pdd = (evaluate(c,self.dt,u,k)[0] for k in range(3))
        # This is a pointwise chart guard, NOT a certified segment-wide bound.
        if np.linalg.norm(p)>=np.pi-1e-6:
            raise ValueError('Interior near-pi chart excursion requires recharting')
        J,dJ = left_jacobian_and_directional(p,pd)
        R0 = self.rotations[segment]
        R = R0@Rotation.from_rotvec(p).as_matrix()
        return R,R0@(J@pd),R0@(J@pdd+dJ@pd)