#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline closed-loop simulation of BicycleMPC vs a dynamic bicycle plant.

No CARLA.  Uses the same right-positive convention.  The reference path is a
straight then a constant-curvature bend (mimicking the T2 entry); the desired
lateral offset ramps to +3.5 m inside the bend.  This validates the QP and
lets us tune Cf/Cr/Iz and the cost weights before touching CARLA.

    /home/yjh/venv-racingllm/bin/python mpc_sim.py
"""
import math

import numpy as np
from bicycle_mpc import BicycleMPC


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def smooth01(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def curvature(s):
    # straight -> bend(radius 120 m) -> straight, smooth transitions
    return (1.0 / 120.0) * smooth01((s - 200.0) / 60.0) * \
           (1.0 - smooth01((s - 350.0) / 60.0))


def desired_offset(s):
    # racing line: 0 on straight, +3.5 m inside, smooth 70 m transitions
    return 3.5 * smooth01((s - 180.0) / 70.0) * \
           (1.0 - smooth01((s - 350.0) / 70.0))


def plant_step(x, delta, a, kappa, dt, p):
    """Dynamic bicycle plant with tyre saturation (tanh force cap)."""
    e, psi, vy, r, v = x
    m, aa, bb, Iz, Cf, Cr, mu = p
    v = max(v, 0.5)
    g = 9.81
    af = delta - (vy + aa * r) / v          # front slip angle
    ar = -(vy - bb * r) / v                 # rear slip angle
    Fyf_max = mu * m * g * bb / (aa + bb)
    Fyr_max = mu * m * g * aa / (aa + bb)
    Fyf = Fyf_max * math.tanh(Cf * af / Fyf_max)
    Fyr = Fyr_max * math.tanh(Cr * ar / Fyr_max)
    e_dot = v * psi + vy
    psi_dot = r - v * kappa
    vy_dot = (Fyf + Fyr) / m - v * r
    r_dot = (aa * Fyf - bb * Fyr) / Iz
    v_dot = a
    return np.array([e + e_dot * dt, psi + psi_dot * dt, vy + vy_dot * dt,
                     r + r_dot * dt, max(v + v_dot * dt, 0.1)])


def simulate(ctrl, plant_p, v_ref=30.0, T=25.0, dt=0.05, noise=0.0,
             speed_profile=False):
    N = ctrl.N
    x = np.array([0.0, 0.0, 0.0, 0.0, v_ref])
    s = 0.0
    hist = []
    for k in range(int(T / dt)):
        kappa = curvature(s)
        v_target = (min(v_ref, math.sqrt(ctrl.a_lat / max(abs(kappa), 1e-4)) * 0.95)
                    if speed_profile else v_ref)
        ks = [curvature(s + x[4] * dt * j) for j in range(N)]
        off = desired_offset(s)
        d, a, ok = ctrl.solve(x[0], x[1], x[2], x[3], x[4], ks, off, v_target)
        if noise > 0:
            d += np.random.normal(0, noise)
        x = plant_step(x, d, a, kappa, dt, plant_p)
        s += x[4] * dt
        hist.append([k * dt, x[0], x[1], x[2], x[3], x[4], d, off, 1 if ok else 0])
    return np.array(hist)


def stats(h):
    e = h[:, 1]
    return dict(max_abs_e=float(np.max(np.abs(e))), rms_e=float(np.sqrt(np.mean(e ** 2))),
                final_e=float(e[-1]))


if __name__ == "__main__":
    m, aa, bb, Iz = 1600.0, 1.45, 1.45, 2200.0
    plant = (m, aa, bb, Iz, 1.2e5, 1.2e5, 1.0)  # saturated plant, Cf=1.2e5, mu=1.0
    print("%-14s %-6s %10s %10s" % ("ctrl Cf", "r_delta", "max|e|", "final_e"))
    for ctrl_cf in (2e4, 3e4, 4e4, 5e4, 6e4, 8e4):
        for r_delta in (1.0, 10.0):
            ctrl = BicycleMPC(N=20, dt=0.05, m=m, a=aa, b=bb, Iz=Iz,
                              Cf=ctrl_cf, Cr=ctrl_cf, r_delta=r_delta)
            h = simulate(ctrl, plant, v_ref=30.0, T=25.0, speed_profile=True)
            st = stats(h)
            flag = "OK" if st["max_abs_e"] < 3.0 else "DIVERGE"
            print("%-14.0e %-6.1f %10.3f %10.3f  %s" % (
                ctrl_cf, r_delta, st["max_abs_e"], st["final_e"], flag))
