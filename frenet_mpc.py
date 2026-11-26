#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frenet-frame linear time-varying MPC (QP) for the ego lateral/longitudinal control.

State  x = [e, psi, v]:
    e   lateral error to the desired line (left-positive)
    psi heading error to the desired line tangent (left-positive)
    v   speed
Control u = [delta, a]:
    delta steering angle (rad, left-positive)
    a     acceleration (m/s^2)

Linear model (frozen v0, road curvature kappa):
    e_{k+1}   = e_k + dt*v0*psi_k
    psi_{k+1} = psi_k + dt*(v0/L*delta_k - v0*kappa)
    v_{k+1}   = v_k + dt*a_k

The QP enforces |v^2 tan(delta)/L| <= a_lat, road boundaries on the absolute
offset, and control bounds.  It is solved with OSQP at 20 Hz.
"""
import math

import numpy as np
import osqp
from scipy import sparse


class FrenetMPC:
    def __init__(self, N=15, dt=0.05, L=2.90, a_lat=10.5, a_max=3.0,
                 off_max=5.5, verbose=False):
        self.N = int(N)
        self.dt = float(dt)
        self.L = float(L)
        self.a_lat = float(a_lat)
        self.a_max = float(a_max)
        self.off_max = float(off_max)
        self.verbose = verbose
        self.last_u = np.zeros(2 * self.N)
        self.last_ok = False
        self.n_solve = 0

    def _fallback(self, e0, psi0, v0, kappa, v_ref):
        dmax = min(0.30, math.atan(self.a_lat * self.L / max(v0 * v0, 1.0)))
        d = math.atan(self.L * kappa) - 0.25 * e0 - 0.8 * psi0
        d = max(min(d, dmax), -dmax)
        a = max(min(0.5 * (v_ref - v0), self.a_max), -self.a_max)
        return d, a, False

    def solve(self, e0, psi0, v0, kappa, desired_off, v_ref, kappa_seq=None):
        N, dt, L = self.N, self.dt, self.L
        v0 = max(float(v0), 0.1)
        # linearized dynamics
        A = np.array([[1.0, dt * v0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        B = np.array([[0.0, 0.0],
                      [dt * v0 / L, 0.0],
                      [0.0, dt]])
        c = np.array([0.0, -dt * v0 * kappa, 0.0])

        nz = 3 * (N + 1)
        Sx = np.zeros((nz, 3))
        Su = np.zeros((nz, 2 * N))
        Sc = np.zeros(nz)
        Sx[0:3, :] = np.eye(3)
        for k in range(N):
            r0 = 3 * k
            r1 = 3 * (k + 1)
            kap = float(kappa_seq[k]) if kappa_seq is not None else kappa
            Sx[r1:r1 + 3, :] = A @ Sx[r0:r0 + 3, :]
            Su[r1:r1 + 3, :] = A @ Su[r0:r0 + 3, :]
            Su[r1:r1 + 3, 2 * k:2 * k + 2] += B
            Sc[r1:r1 + 3] = A @ Sc[r0:r0 + 3] + np.array(
                [0.0, -dt * v0 * kap, 0.0])

        x0 = np.array([e0, psi0, v0])
        base = Sx @ x0 + Sc
        Xref = np.tile(np.array([0.0, 0.0, v_ref]), N + 1)

        Q = np.diag([1.5, 1.0, 0.35])
        Qf = np.diag([4.0, 2.0, 0.5])
        R = np.diag([2.5, 0.25])
        Qbar = sparse.block_diag([Q] * N + [Qf], format="csc")
        Rbar = sparse.block_diag([R] * N, format="csc")
        SuS = sparse.csc_matrix(Su)
        H = 2.0 * (SuS.T @ Qbar @ SuS + Rbar)
        H = (H + H.T) * 0.5
        f = 2.0 * (SuS.T @ Qbar @ (base - Xref))

        dmax = min(0.30, math.atan(self.a_lat * L / max(v0 * v0, 1.0)))
        lu = []
        uu = []
        for _ in range(N):
            lu += [-dmax, -self.a_max]
            uu += [dmax, self.a_max]

        # state constraints: e in [desired-off_max, desired+off_max],
        # psi in [-0.6, 0.6], v in [0, 60]
        l_state = []
        u_state = []
        for k in range(N + 1):
            r = 3 * k
            l_state += [desired_off - self.off_max,
                        -0.6,
                        0.0]
            u_state += [desired_off + self.off_max,
                        0.6,
                        60.0]
        # convert to Su*u bounds by subtracting base
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
        return self._fallback(e0, psi0, v0, kappa, v_ref)
