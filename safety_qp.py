#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based CBF-QP safety filter at the tactical layer.

State used:
    e        : ego actual extra-inside offset (m, signed; 0 = racing line)
    e_l      : lead actual extra-inside offset
    e_dot    : ego lateral rate (m/s)
    e_l_dot  : lead lateral rate (m/s)
    gap      : |ahead| (m)
    v_e/v_l  : speeds

Decision variables:
    e_ref : commanded extra inside offset
    v_ref : speed target

Constraints:
    lateral CBF (first-order lateral response, time constant tau):
        d = e - e_l, h = |d| - d_min
        if d >= 0: e_ref >= e + tau * (e_l_dot - alpha*h)
        if d <  0: e_ref <= e + tau * (e_l_dot + alpha*h)
    longitudinal CBF:
        if behind: v_ref <= v_l + beta*(gap - d_gap)
        if ahead : v_ref >= v_l - beta*(ahead - d_gap)
    TTC: if closing, v_ref <= v_l + gap/ttc_min
    bounds + rate limits: e_ref, v_ref.

The objective is separable quadratic, so the exact solution is interval
projection.  This is a discrete-time CBF approximation with rate limiting, not
a formal continuous-time CBF; the paper should state that and add the delayed
input bound.
"""
import math


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class ShieldQP:
    def __init__(self, d_min=2.2, d_gap=3.0, ttc_min=0.8,
                 tau=0.4, alpha=1.0, beta=0.8,
                 e_lim=3.5, v_max=55.0,
                 e_rate=1.0, v_acc_rate=3.0):
        self.d_min = float(d_min)
        self.d_gap = float(d_gap)
        self.ttc_min = float(ttc_min)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.e_lim = float(e_lim)
        self.v_max = float(v_max)
        self.e_rate = float(e_rate)
        self.v_acc_rate = float(v_acc_rate)

    def filter(self, state, nominal_extra, nominal_v, e_ref_prev, v_ref_prev, dt=0.05):
        e = float(state.get("actual_extra_m", 0.0))
        e_l = float(state.get("actual_extra_lead_m", 0.0))
        e_dot = float(state.get("actual_extra_rate_mps", 0.0))
        e_l_dot = float(state.get("lead_extra_rate_mps", 0.0))
        ahead = float(state.get("ahead_m", 0.0))
        gap = abs(ahead)
        v_e = float(state.get("ego_v", 0.0))
        v_l = float(state.get("lead_v", 0.0))
        d = e - e_l
        h = abs(d) - self.d_min

        e_lo, e_hi = -self.e_lim, self.e_lim
        lat_active = False
        # CBF only near the lateral boundary and near the opponent longitudinally
        if gap < 12.0 and abs(d) < self.d_min + 1.0:
            lat_active = True
            if d >= 0.0:
                e_lo = max(e_lo, e + self.tau * (e_l_dot - self.alpha * h))
            else:
                e_hi = min(e_hi, e + self.tau * (e_l_dot + self.alpha * h))

        # rate limits are treated separately from the safety bounds.  If the
        # CBF wants a faster lateral move than the comfort rate allows, take the
        # maximum achievable step and mark a lateral slack; do NOT declare the
        # whole QP infeasible and stop the car (that was the naive failure).
        e_lo_safe, e_hi_safe = e_lo, e_hi
        e_lo_rate, e_hi_rate = -1e9, 1e9
        if e_ref_prev is not None:
            step = self.e_rate * dt
            e_lo_rate = float(e_ref_prev) - step
            e_hi_rate = float(e_ref_prev) + step

        v_lo_safe, v_hi_safe = 0.0, self.v_max
        long_active = False
        if ahead < 0.0:
            long_active = True
            v_hi_safe = min(v_hi_safe, v_l + self.beta * (gap - self.d_gap))
            if v_e > v_l:
                v_hi_safe = min(v_hi_safe, v_l + gap / self.ttc_min)
        else:
            long_active = True
            v_lo_safe = max(v_lo_safe, v_l - self.beta * (ahead - self.d_gap))

        v_lo_rate, v_hi_rate = -1e9, 1e9
        if v_ref_prev is not None:
            step_v = self.v_acc_rate * dt
            v_lo_rate = float(v_ref_prev) - step_v
            v_hi_rate = float(v_ref_prev) + step_v

        lat_slack = False
        lat_infeasible = e_lo_safe > e_hi_safe + 1e-9
        if lat_infeasible:
            e_ref = _clip(float(e_ref_prev if e_ref_prev is not None else e),
                          -self.e_lim, self.e_lim)
            lat_slack = True
        elif e_lo_safe > e_hi_rate:
            e_ref = e_hi_rate
            lat_slack = True
        elif e_hi_safe < e_lo_rate:
            e_ref = e_lo_rate
            lat_slack = True
        else:
            e_ref = _clip(float(nominal_extra),
                          max(e_lo_safe, e_lo_rate), min(e_hi_safe, e_hi_rate))

        v_slack = False
        long_infeasible = v_lo_safe > v_hi_safe + 1e-9
        if long_infeasible:
            v_ref = _clip(min(float(nominal_v), v_l - 2.0, max(0.0, v_e - 2.0)),
                          max(0.0, v_lo_safe), self.v_max)
            v_slack = True
        elif v_lo_safe > v_hi_rate:
            v_ref = v_hi_rate
            v_slack = True
        elif v_hi_safe < v_lo_rate:
            v_ref = v_lo_rate
            v_slack = True
        else:
            v_ref = _clip(float(nominal_v),
                          max(v_lo_safe, v_lo_rate), min(v_hi_safe, v_hi_rate))

        infeasible = lat_infeasible or long_infeasible
        e_lo, e_hi = max(e_lo_safe, e_lo_rate), min(e_hi_safe, e_hi_rate)
        v_lo, v_hi = max(v_lo_safe, v_lo_rate), min(v_hi_safe, v_hi_rate)

        d_e = abs(e_ref - float(nominal_extra))
        d_v = abs(v_ref - float(nominal_v))
        return dict(
            extra=round(e_ref, 3), v_ref=round(v_ref, 3),
            intervened=bool(d_e > 1e-3 or d_v > 1e-3 or infeasible),
            infeasible=bool(infeasible),
            lat_slack=bool(lat_slack), v_slack=bool(v_slack),
            lat_infeasible=bool(lat_infeasible),
            long_infeasible=bool(long_infeasible),
            lat_active=bool(lat_active), long_active=bool(long_active),
            e=round(e, 3), e_l=round(e_l, 3), e_dot=round(e_dot, 3),
            e_l_dot=round(e_l_dot, 3), d=round(d, 3), h=round(h, 3),
            e_lo=round(e_lo, 3), e_hi=round(e_hi, 3),
            v_lo=round(v_lo, 3), v_hi=round(v_hi, 3),
            nominal_extra=round(float(nominal_extra), 3),
            nominal_v=round(float(nominal_v), 3),
        )
