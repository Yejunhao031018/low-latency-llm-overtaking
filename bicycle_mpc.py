#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic bicycle-model LTV-MPC (QP) for the ego lateral/longitudinal control.

All lateral quantities are RIGHT-positive (CARLA positive steer = right turn),
so the MPC steering maps directly to the CARLA steer with no sign flip.

State  x = [e, psi, vy, r, v]
    e    lateral error to the desired line (right-positive)
    psi  heading error (right-positive)
    vy   lateral velocity (body frame, right-positive)
    r    yaw rate (right-positive)
    v    speed
Control u = [delta, a]
    delta steering (right-positive, rad)
    a     longitudinal acceleration

Continuous linear bicycle model (small slip), reference curvature kappa:
    e_dot   = v*psi + vy
    psi_dot = r - v*kappa
    vy_dot  = -(Cf+Cr)/(m v) vy + ((b Cr - a Cf)/(m v) - v) r + Cf/m delta
    r_dot   = (b Cr - a Cf)/(Iz v) vy - (a^2 Cf + b^2 Cr)/(Iz v) r + a Cf/Iz delta
    v_dot   = a

The tyre-slip state vy damps the kinematic double integrator, which was the
instability source in the kinematic QP.
"""
import math

import numpy as np
import osqp
from scipy import sparse


class BicycleMPC:
    def __init__(self, N=20, dt=0.05, m=1600.0, a=1.45, b=1.45, Iz=2200.0,
                 Cf=1.2e5, Cr=1.2e5, a_lat=10.5, a_max=3.0, off_max=5.5,
                 verbose=False, r_delta=1.0, q_e=4.0, mu=1.0):
        self.N = int(N)
        self.dt = float(dt)
        self.m = float(m)
        self.a = float(a)
        self.b = float(b)
        self.L = a + b
        self.Iz = float(Iz)
        self.Cf = float(Cf)
        self.Cr = float(Cr)
        self.a_lat = float(a_lat)
        self.a_max = float(a_max)
        self.off_max = float(off_max)
        self.verbose = verbose
        self.r_delta = float(r_delta)
        self.q_e = float(q_e)
        self.mu = float(mu)
        self.last_u = np.zeros(2 * self.N)
        self.last_ok = False
        self.n_solve = 0

    def solve(self, e0, psi0, vy0, r0, v0, kappa_seq, desired_off, v_ref):
        N, dt = self.N, self.dt
        m, a, b, Iz = self.m, self.a, self.b, self.Iz
        v = max(float(v0), 0.5)
        # successive linearization of the saturated tyre model:
        #   Fy = Fy_max * tanh(C*alpha / Fy_max)  ->  C_eff = C * sech^2(...)
        d_prev = float(self.last_u[0]) if self.last_u.size else 0.0
        g = 9.81
        Fyf_max = self.mu * m * g * b / (a + b)
        Fyr_max = self.mu * m * g * a / (a + b)
        af = d_prev - (vy0 + a * r0) / v
        ar = -(vy0 - b * r0) / v
        Cf = self.Cf / (math.cosh(self.Cf * af / Fyf_max) ** 2)
        Cr = self.Cr / (math.cosh(self.Cr * ar / Fyr_max) ** 2)

        A = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -(Cf + Cr) / (m * v), (b * Cr - a * Cf) / (m * v) - v, 0.0],
            [0.0, 0.0, (b * Cr - a * Cf) / (Iz * v), -(a * a * Cf + b * b * Cr) / (Iz * v), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        B = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [Cf / m, 0.0],
            [a * Cf / Iz, 0.0],
            [0.0, 1.0],
        ])
        # continuous x_dot = A x + B u + c, where c only couples v->e/psi
        # We discretize by Euler and add the coupling terms e_dot, psi_dot, v_dot.
        Ad = np.eye(5) + dt * A
        Bd = dt * B

        nx = 5
        nz = nx * (N + 1)
        # Rebuild with explicit coupling (speed frozen at v):
        Sx = np.zeros((nz, nx))
        Su = np.zeros((nz, 2 * N))
        Sc = np.zeros(nz)
        Sx[0:nx, :] = np.eye(nx)
        for k in range(N):
            r0_ = nx * k
            r1_ = nx * (k + 1)
            kap = float(kappa_seq[k]) if kappa_seq is not None else 0.0
            xk = Sx[r0_:r0_ + nx, :]
            uk = Su[r0_:r0_ + nx, :]
            ck = Sc[r0_:r0_ + nx]
            x_next = Ad @ xk
            u_next = Ad @ uk
            # coupling terms (explicit Euler):
            Sx[r1_ + 0, :] = x_next[0, :] + dt * (v * xk[1, :] + xk[2, :])
            Su[r1_ + 0, :] = u_next[0, :] + dt * (v * uk[1, :] + uk[2, :])
            Sx[r1_ + 1, :] = x_next[1, :] + dt * xk[3, :]
            Su[r1_ + 1, :] = u_next[1, :] + dt * uk[3, :]
            Sc[r1_ + 1] = ck[1] - dt * v * kap
            # vy, r, v rows unchanged from Ad
            Sx[r1_ + 2, :] = x_next[2, :]
            Su[r1_ + 2, :] = u_next[2, :]
            Sc[r1_ + 2] = ck[2]
            Sx[r1_ + 3, :] = x_next[3, :]
            Su[r1_ + 3, :] = u_next[3, :]
            Sc[r1_ + 3] = ck[3]
            Sx[r1_ + 4, :] = x_next[4, :]
            Su[r1_ + 4, :] = u_next[4, :]
            Sc[r1_ + 4] = ck[4]
            # control-input contributions B_d u_k
            for m in range(nx):
                Su[r1_ + m, 2 * k] += Bd[m, 0]
                Su[r1_ + m, 2 * k + 1] += Bd[m, 1]

        x0 = np.array([e0, psi0, vy0, r0, v])
        base = Sx @ x0 + Sc
        Xref = np.zeros(nz)
        for k in range(N + 1):
            kap = float(kappa_seq[k]) if (kappa_seq is not None and k < N) else \
                  (float(kappa_seq[-1]) if kappa_seq is not None else 0.0)
            Xref[nx * k:nx * k + 5] = [0.0, 0.0, 0.0, v * kap, v_ref]

        Q = np.diag([self.q_e, 1.0, 0.5, 0.2, 0.5])
        Qf = np.diag([2 * self.q_e, 2.0, 0.5, 0.2, 0.5])
        R = np.diag([self.r_delta, 0.25])
        Qbar = sparse.block_diag([Q] * N + [Qf], format="csc")
        Rbar = sparse.block_diag([R] * N, format="csc")
        SuS = sparse.csc_matrix(Su)
        H = 2.0 * (SuS.T @ Qbar @ SuS + Rbar)
        H = (H + H.T) * 0.5
        f = 2.0 * (SuS.T @ Qbar @ (base - Xref))

        dmax = min(0.30, math.atan(self.a_lat * self.L / max(v * v, 1.0)))
        rmax = self.a_lat / max(v, 0.5)
        lu, uu = [], []
        for _ in range(N):
            lu += [-dmax, -self.a_max]
            uu += [dmax, self.a_max]
        l_state, u_state = [], []
        for k in range(N + 1):
            l_state += [desired_off - self.off_max, -0.6, -5.0, -rmax, 0.0]
            u_state += [desired_off + self.off_max, 0.6, 5.0, rmax, 60.0]
        A_state = sparse.csc_matrix(Su)
        l_s = np.array(l_state) - base
        u_s = np.array(u_state) - base
        A_cons = sparse.vstack([A_state, sparse.eye(2 * N, format="csc")],
                               format="csc")
        l = np.concatenate([l_s, np.array(lu)])
        u = np.concatenate([u_s, np.array(uu)])

        try:
            prob = osqp.OSQP()
            prob.setup(P=H, q=f, A=A_cons, l=l, u=u, verbose=self.verbose,
                       eps_abs=1e-4, eps_rel=1e-4, max_iter=4000,
                       polishing=False, warm_start=True)
            res = prob.solve()
            self.n_solve += 1
            if res.x is not None and res.info.status in ("solved", "solved inaccurate"):
                self.last_u = np.asarray(res.x)
                self.last_ok = True
                return float(res.x[0]), float(res.x[1]), True
        except Exception:
            pass
        self.last_ok = False
        # fallback: small proportional steer (right-positive) + curvature
        d_ff = math.atan(self.L * (kappa_seq[0] if kappa_seq is not None else 0.0))
        d = d_ff - 0.25 * e0 - 0.5 * psi0
        d = max(min(d, dmax), -dmax)
        a = max(min(0.5 * (v_ref - v), self.a_max), -self.a_max)
        return d, a, False
