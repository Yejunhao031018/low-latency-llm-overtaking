#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configurable CARLA episode runner for the paper data engine.

Core loop:
    state -> policy action -> optional counterfactual candidate rollouts
          -> short-horizon outcome -> cost-to-go label

It reuses the ring / racing line / speed profile / pure-pursuit controller
from race_overtake.py, and writes:
    <out>/<episode_id>/decisions.jsonl
    <out>/<episode_id>/ticks.csv
    <out>/<episode_id>/summary.json

Policies:
    stub     : rule policy + current safety filter (safe baseline)
    hold     : always hold
    random   : deliberately unsafe exploration (hard negatives / near misses)

Lead policies:
    constant : fixed speed factor, racing line
    variable : slow sinusoidal speed variation
    defensive: shifts toward the ego's side when the ego is close (blocks the line)

Counterfactual labeling:
    At selected states near T2, snapshot both cars, run each candidate action
    for a few seconds, restore, and record collision / min separation / pass.
    The CARLA state restore is approximate (transform + target velocities), which
    is adequate for short-horizon preference labels but not a bit-exact snapshot.

Usage:
    # one safe episode with labels
    python dataset_runner.py --episodes 1 --policy stub --label --max-labels 4

    # use from collect_dataset.py for randomized batches
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np

import carla

from race_overtake import (
    A_ACC, A_BRAKE_EGO, A_BRAKE_LEAD, A_LAT, DT, MAX_STEER, ROTATE_S,
    V_STRAIGHT, WHEELBASE, Ring, attack_profile, build_ring, build_ring_canonical,
    corner_offsets,
    smooth01, speed_profile_field, wrap,
)
from llm_decision import DecisionEngine, StubBackend
from safety_qp import ShieldQP

OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

CANDIDATES = [
    dict(name="hold", extra_inside_m=0.0, speed_factor=1.00),
    dict(name="inside_3.0", extra_inside_m=3.0, speed_factor=1.00),
    dict(name="inside_3.5", extra_inside_m=3.5, speed_factor=1.05),
    dict(name="inside_late_brake", extra_inside_m=3.5, speed_factor=1.10),
    dict(name="outside_3.5", extra_inside_m=-3.5, speed_factor=1.05),
]

RANDOM_CANDIDATES = [
    # name, extra, speed_factor, weight
    ("hold", 0.0, 1.00, 0.10),
    ("inside_2.0", 2.0, 1.05, 0.20),
    ("inside_3.0", 3.0, 1.05, 0.20),
    ("inside_3.5", 3.5, 1.10, 0.25),
    ("outside_2.5", -2.5, 1.05, 0.15),
    ("outside_3.5", -3.5, 1.10, 0.10),
]


def _destroy_all(world):
    for a in world.get_actors():
        if a.type_id.startswith(("vehicle.", "walker.", "sensor.")):
            try:
                a.destroy()
            except Exception:
                pass


def _pick(bp_lib, patterns):
    for pat in patterns:
        found = bp_lib.filter(pat)
        if len(found) > 0:
            return found[0]
    raise RuntimeError("no blueprint for %s" % patterns)


class Episode:
    def __init__(self, world, cfg, episode_id):
        self.world = world
        self.cfg = cfg
        self.episode_id = episode_id
        self.rng = random.Random(cfg.get("seed", 0))
        self.policy = cfg.get("policy", "stub")
        self.lead_policy = cfg.get("lead_policy", "constant")
        self.in_rollout = False
        self.coll = {"ego": 0, "lead": 0}
        self.coll_total = {"ego": 0, "lead": 0}
        self.coll_sensors = []
        self._setup()

    # ------------------------------------------------------------------ setup
    def _setup(self):
        w = self.world
        _destroy_all(w)
        try:
            w.tick()
        except Exception:
            pass
        st = w.get_settings()
        st.synchronous_mode = True
        st.fixed_delta_seconds = DT
        st.no_rendering_mode = self.cfg.get("no_rendering", True)
        w.apply_settings(st)
        w.tick()
        weather = self.cfg.get("weather", "ClearNoon")
        try:
            w.set_weather(getattr(carla.WeatherParameters, weather))
        except Exception:
            w.set_weather(carla.WeatherParameters.ClearNoon)

        self.ring = Ring(build_ring_canonical(w))
        self.ring.rotate(self.ring.rot_shift(ROTATE_S))
        radius = float(self.cfg.get("corner_radius", 0.0))
        if radius > 0.0:
            self.t2 = min(self.ring.corners, key=lambda c: abs(c["R"] - radius))
        else:
            self.t2 = self.ring.corners[int(self.cfg.get("corner_index", 1))]
        self.t2_sign = float(self.t2.get("inside_sign", 1.0))
        self.i_apex = self.ring.s_of(self.t2["s_apex"])
        self.racing_off = corner_offsets(
            self.ring,
            off_out=float(self.cfg.get("racing_off_out", 2.0)),
            off_in=float(self.cfg.get("racing_off_in", 3.3)))
        self.attack_s = float(self.cfg.get("attack_s", self.t2["s_apex"]))
        self.attack_corner = dict(s_apex=self.attack_s)
        self.attack_w = attack_profile(
            self.ring, self.attack_corner,
            before=float(self.cfg.get("attack_before", 240.0)),
            hold=float(self.cfg.get("attack_hold", 120.0)),
            after=float(self.cfg.get("attack_after", 30.0)),
            ramp_out=float(self.cfg.get("attack_ramp_out", 50.0)))

        lead_f = float(self.cfg.get("lead_speed_factor", 0.90))
        ego_f = float(self.cfg.get("ego_speed_factor", 1.05))
        boost_s = float(self.cfg.get("boost_s", self.attack_s))
        boost_before = float(self.cfg.get("boost_before", 190.0))
        boost_after = float(self.cfg.get("boost_after", 90.0))
        n = self.ring.n
        fac_lead = np.full(n, lead_f)
        fac_ego = np.full(n, lead_f)
        for i in range(n):
            s = i * 2.0
            d = (s - boost_s) % self.ring.L
            if d > self.ring.L / 2:
                d -= self.ring.L
            if -boost_before <= d <= boost_after:
                wgt = 1.0
                if d < -boost_before + 40.0:
                    wgt = smooth01((d + boost_before) / 40.0)
                elif d > boost_after - 40.0:
                    wgt = 1.0 - smooth01((d - (boost_after - 40.0)) / 40.0)
                fac_ego[i] = lead_f + (ego_f - lead_f) * wgt
        a_lat = float(self.cfg.get("a_lat", A_LAT))
        v_max = float(self.cfg.get("v_max", V_STRAIGHT))
        lead_v_max = float(self.cfg.get("lead_v_max", v_max))
        ego_v_max = float(self.cfg.get("ego_v_max", v_max))
        self.v_lead = speed_profile_field(self.ring, fac_lead, A_BRAKE_LEAD,
                                          a_lat=a_lat, v_max=lead_v_max)
        self.v_ego = speed_profile_field(self.ring, fac_ego, A_BRAKE_EGO,
                                         a_lat=a_lat, v_max=ego_v_max)

        self.s_ego0 = (self.t2["s0"] - 360.0) % self.ring.L
        self.s_lead0 = (self.s_ego0 + float(self.cfg.get("start_gap_m", 26.0))) % self.ring.L

        bl = w.get_blueprint_library()
        lead_bp = _pick(bl, ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.*"])
        ego_bp = _pick(bl, ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.*"])
        for bp, col in ((lead_bp, "255,0,0"), (ego_bp, "0,0,255")):
            try:
                if bp.has_attribute("color"):
                    bp.set_attribute("color", col)
            except Exception:
                pass

        self.lead = self._spawn(lead_bp, self.s_lead0)
        self.ego = self._spawn(ego_bp, self.s_ego0)

        coll_bp = bl.find("sensor.other.collision")
        for name, actor in (("ego", self.ego), ("lead", self.lead)):
            cs = w.spawn_actor(coll_bp, carla.Transform(), attach_to=actor)

            def cb(event, name=name):
                self.coll[name] += 1
                if not self.in_rollout:
                    self.coll_total[name] += 1

            cs.listen(cb)
            self.coll_sensors.append(cs)

        self.s_e_prev = self.s_ego0
        self.s_l_prev = self.s_lead0
        self.extra_cmd = 0.0
        self.t = 0.0
        self.step = 0
        self.lap_frac = 0.0
        self.rows = []
        self.decisions = []
        self.history = []
        self.min_center = 1e9
        self.min_sep_close = 1e9
        self.pass_t = None
        self.pass_s = None
        self.v_peak = 0.0
        self.last_action = {"action": "hold", "extra_inside_m": 0.0,
                            "speed_factor": 1.0, "reason": "init"}
        self.stub = StubBackend()
        self.qp = ShieldQP() if self.cfg.get("shield_mode") == "qp" else None
        self.mpc = None
        self.mpc_every = int(self.cfg.get("mpc_every", 1))
        self._mpc_i = 0
        self._mpc_cmd = (0.0, 0.0)
        self._use_mpc = False
        self._in_fallback = False
        self.frenet = None
        self.bicycle = None
        self._prev_heading = None
        self._prev_t = 0.0
        self._last_bike_steer = 0.0
        self._pp_prev_heading = {}
        mode = self.cfg.get("controller", "pure")
        if mode == "mpc":
            from overtake_mpc import MPC
            self.mpc = MPC()
        elif mode in ("frenet", "hybrid"):
            from frenet_mpc import FrenetMPC
            self.frenet = FrenetMPC(
                N=int(self.cfg.get("frenet_N", 15)),
                dt=DT, L=WHEELBASE,
                a_lat=float(self.cfg.get("a_lat", 10.5)),
                a_max=3.0,
                off_max=float(self.cfg.get("road_off_max", 5.5)))
        elif mode == "bicycle":
            from bicycle_mpc import BicycleMPC
            self.bicycle = BicycleMPC(
                N=int(self.cfg.get("bicycle_N", 20)), dt=DT,
                a_lat=float(self.cfg.get("a_lat", 10.5)),
                a_max=3.0, off_max=float(self.cfg.get("road_off_max", 5.5)),
                Cf=float(self.cfg.get("bicycle_Cf", 1.2e5)),
                Cr=float(self.cfg.get("bicycle_Cr", 1.2e5)),
                Iz=float(self.cfg.get("bicycle_Iz", 2200.0)))
        self.prev_extra = None
        self.prev_extra_l = None
        self.prev_v_target = None
        self.engine = None
        if self.policy == "openai":
            self.engine = DecisionEngine("openai")
            print("[dataset] openai backend:", self.engine.backend_name)
        self.labels_done = 0
        self.last_label_t = -1e9
        self.coll_grace = 0.0
        self.latency_events = []
        self.latency_stats = {}
        self.pass_done = False
        self.sep_latched = False
        self._last_live = 0.0
        self._last_label_t = -1e9
        self._live_decisions = []
        self._live_history = []
        self._last_resp = None
        self.shield_count = 0
        self.tick_shield = bool(self.cfg.get("tick_shield", self.policy != "random"))

    def _spawn(self, bp, s):
        p, yaw = self._path_pt(s, self.racing_off)
        # Spawn relative to the actual road surface: z=0.4 can be far below
        # the road at some Town04 sections and the car then falls / gains speed.
        z0 = 0.0
        try:
            wp = self.world.get_map().get_waypoint(
                carla.Location(x=float(p[0]), y=float(p[1]), z=0.0),
                project_to_road=True, lane_type=carla.LaneType.Driving)
            z0 = float(wp.transform.location.z)
        except Exception:
            z0 = 0.0
        for dz in (0.35, 0.7, 1.2, 2.0):
            a = self.world.try_spawn_actor(bp, carla.Transform(
                carla.Location(x=p[0], y=p[1], z=z0 + dz),
                carla.Rotation(yaw=math.degrees(yaw))))
            if a is not None:
                v0 = float(self.cfg.get("initial_speed_mps", 0.0))
                if v0 > 0.0:
                    i = self.ring.s_of(s)
                    tang = self.ring.tang[i]
                    a.set_target_velocity(carla.Vector3D(
                        float(tang[0]) * v0, float(tang[1]) * v0, 0.0))
                return a
        raise RuntimeError("spawn failed")

    def _path_pt(self, s, offs):
        i = self.ring.s_of(s)
        p = self.ring.P[i] + self.ring.nright[i] * offs[i]
        yaw = math.atan2(self.ring.tang[i][1], self.ring.tang[i][0])
        return p, yaw

    def _pure_pursuit(self, actor, s_u, offs, v_ref, attack, loc, speed):
        """Pure-pursuit steering + speed control (also the MPC fallback)."""
        if attack:
            # at high speed a very short lookahead makes the lateral offset
            # oscillate and scrubs speed; use a longer, damped lookahead.
            look = max(8.0, min(22.0, speed * 0.50))
        else:
            look = max(6.0, min(20.0, speed * 0.55))
        i = self.ring.s_of(s_u + look)
        tgt = self.ring.P[i] + self.ring.nright[i] * offs[i]
        tf = actor.get_transform()
        fwd = tf.rotation.get_forward_vector()
        heading = math.atan2(fwd.y, fwd.x)
        alpha = wrap(math.atan2(tgt[1] - loc.y, tgt[0] - loc.x) - heading)
        dist = math.hypot(tgt[0] - loc.x, tgt[1] - loc.y)
        delta = math.atan2(2.0 * WHEELBASE * math.sin(alpha), max(dist, 0.5))
        k_damp = float(self.cfg.get("pp_yaw_damp", 0.0))
        if k_damp > 0.0:
            prev_h = self._pp_prev_heading.get(actor.id)
            if prev_h is not None:
                r = wrap(heading - prev_h) / DT   # right-positive yaw rate
                delta = delta - k_damp * r        # yaw damper
            self._pp_prev_heading[actor.id] = heading
        delta = max(min(delta, MAX_STEER), -MAX_STEER)
        err = v_ref - speed
        accel = max(min(1.5 * err, A_ACC), -A_BRAKE_EGO)
        if accel >= 0:
            thr = min(1.0, accel / 4.0)
            brk = 0.0
        else:
            thr = 0.0
            brk = min(1.0, -accel / 8.0)
        return delta, thr, brk

    # --------------------------------------------------------------- control
    def drive(self, actor, s_u, offs, v_ref, attack):
        loc = actor.get_location()
        v = actor.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        if self.mpc is not None and actor is self.ego and self._use_mpc:
            # Coupled MPC replaces pure-pursuit for the ego.  The reference is
            # the current desired lateral offset (racing line + attack offset).
            tf = actor.get_transform()
            fwd = tf.rotation.get_forward_vector()
            heading = math.atan2(fwd.y, fwd.x)
            i = self.ring.s_of(s_u)
            tang = self.ring.tang[i]
            yaw_road = math.atan2(tang[1], tang[0])
            e_p = np.array([loc.x, loc.y])
            actual_off = float((e_p - self.ring.P[i]) @ self.ring.nright[i])
            desired = float(offs[i])
            e0 = desired - actual_off          # left-positive error to target
            psi0 = wrap(heading - yaw_road)    # left-positive heading error
            if self._mpc_i % max(1, self.mpc_every) == 0:
                kappa = float(self.ring.k[i])
                d0, a0, ok = self.mpc.solve(
                    e0, psi0, speed, s_u, s_u, 0.0, v_ref,
                    kappa=kappa)
                if not ok:
                    # Optimizer fallback: curvature feedforward + PD.  A failed
                    # solve must never mean "steer straight" at 30 m/s.
                    d_ff = math.atan(WHEELBASE * kappa)
                    d0 = max(min(d_ff - 0.25 * e0 - 0.8 * psi0, 0.30), -0.30)
                    a0 = 0.5 * (v_ref - speed)
                self._mpc_cmd = (d0, a0)
            d0, a0 = self._mpc_cmd
            self._mpc_i += 1
            # friction-limited steering: the kinematic MPC can request a steer
            # that exceeds tyre grip (a_lat = v^2 tan(d)/L), which makes the car
            # slide off the road.  Clamp to the same a_lat used by the profile.
            a_lat_max = float(self.cfg.get("a_lat", 10.5))
            d_max = math.atan(a_lat_max * WHEELBASE / max(speed ** 2, 1.0))
            d0 = max(min(d0, d_max), -d_max)
            if self.cfg.get("mpc_debug", False) and self.step % 20 == 0:
                print("[mpc] t=%.1f s=%.0f e=%.2f psi=%.3f d=%.3f a=%.2f "
                      "ok=%s v=%.1f off=%.2f" % (
                          self.t, s_u, e0, psi0, d0, a0, ok, speed, actual_off),
                      flush=True)
            delta = max(min(d0, MAX_STEER), -MAX_STEER)
            thr = max(min(a0 / 4.0, 1.0), 0.0)
            brk = max(min(-a0 / 8.0, 1.0), 0.0)
            # speed envelope: the short-horizon kinematic MPC can run away
            if speed > v_ref + 2.0:
                thr = 0.0
                brk = min(1.0, (speed - v_ref) / 8.0)
            elif speed < v_ref - 2.0:
                thr = max(thr, min(1.0, (v_ref - speed) / 10.0))
                brk = 0.0
            actor.apply_control(carla.VehicleControl(
                steer=-delta / MAX_STEER, throttle=thr, brake=brk))
            return

        if self.frenet is not None and actor is self.ego and self._use_mpc:
            tf = actor.get_transform()
            fwd = tf.rotation.get_forward_vector()
            heading = math.atan2(fwd.y, fwd.x)
            i = self.ring.s_of(s_u)
            def dpoint(ss):
                j = self.ring.s_of(ss)
                return self.ring.P[j] + self.ring.nright[j] * offs[j]
            p0 = dpoint(s_u - 6.0)
            p1 = dpoint(s_u)
            p2 = dpoint(s_u + 6.0)
            desired_yaw = math.atan2(p2[1] - p0[1], p2[0] - p0[0])
            yaw1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            yaw2 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            kappa = wrap(yaw2 - yaw1) / 6.0
            e_p = np.array([loc.x, loc.y])
            actual_off = float((e_p - self.ring.P[i]) @ self.ring.nright[i])
            desired = float(offs[i])
            e0 = desired - actual_off
            psi0 = wrap(heading - desired_yaw)
            if self._mpc_i % max(1, self.mpc_every) == 0:
                ks = []
                for kk in range(self.frenet.N):
                    ss = s_u + speed * self.frenet.dt * kk
                    q0 = dpoint(ss - 6.0)
                    q1 = dpoint(ss)
                    q2 = dpoint(ss + 6.0)
                    yy1 = math.atan2(q1[1] - q0[1], q1[0] - q0[0])
                    yy2 = math.atan2(q2[1] - q1[1], q2[0] - q1[0])
                    ks.append(wrap(yy2 - yy1) / 6.0)
                d0, a0, ok = self.frenet.solve(
                    e0, psi0, speed, kappa, desired, v_ref,
                    kappa_seq=ks)
                self._mpc_cmd = (d0, a0)
            d0, a0 = self._mpc_cmd
            self._mpc_i += 1
            # Safety fallback: if the QP is infeasible or the lateral error is
            # already large, hand control back to pure-pursuit for this tick.
            fallback_on = float(self.cfg.get("pp_fallback_err", 1.0))
            fallback_off = float(self.cfg.get("pp_fallback_resume", 0.5))
            if self._in_fallback and abs(e0) < fallback_off:
                self._in_fallback = False
            if (not self._in_fallback) and (
                    (not self.frenet.last_ok) or abs(e0) > fallback_on):
                self._in_fallback = True
            if self._in_fallback:
                delta, thr, brk = self._pure_pursuit(
                    actor, s_u, offs, v_ref, attack, loc, speed)
                if self.cfg.get("mpc_debug", False) and self.step % 20 == 0:
                    print("[frenet] FALLBACK t=%.1f s=%.0f e=%.2f "
                          "ok=%s" % (self.t, s_u, e0,
                                     self.frenet.last_ok), flush=True)
                actor.apply_control(carla.VehicleControl(
                    steer=delta / MAX_STEER, throttle=thr, brake=brk))
                return
            a_lat_max = float(self.cfg.get("a_lat", 10.5))
            d_max = math.atan(a_lat_max * WHEELBASE / max(speed ** 2, 1.0))
            d0 = max(min(d0, d_max), -d_max)
            if self.cfg.get("mpc_debug", False) and self.step % 20 == 0:
                print("[frenet] t=%.1f s=%.0f e=%.2f psi=%.3f d=%.3f a=%.2f "
                      "ok=%s v=%.1f off=%.2f" % (
                          self.t, s_u, e0, psi0, d0, a0,
                          self.frenet.last_ok, speed, actual_off), flush=True)
            delta = max(min(d0, MAX_STEER), -MAX_STEER)
            thr = max(min(a0 / 4.0, 1.0), 0.0)
            brk = max(min(-a0 / 8.0, 1.0), 0.0)
            if speed > v_ref + 2.0:
                thr = 0.0
                brk = min(1.0, (speed - v_ref) / 8.0)
            elif speed < v_ref - 2.0:
                thr = max(thr, min(1.0, (v_ref - speed) / 10.0))
                brk = 0.0
            actor.apply_control(carla.VehicleControl(
                steer=-delta / MAX_STEER, throttle=thr, brake=brk))
            return

        if self.bicycle is not None and actor is self.ego and self._use_mpc:
            tf = actor.get_transform()
            fwd = tf.rotation.get_forward_vector()
            heading = math.atan2(fwd.y, fwd.x)
            i = self.ring.s_of(s_u)
            def dpt(ss):
                j = self.ring.s_of(ss)
                return self.ring.P[j] + self.ring.nright[j] * offs[j]
            p0 = dpt(s_u - 6.0)
            p1 = dpt(s_u)
            p2 = dpt(s_u + 6.0)
            desired_yaw = math.atan2(p2[1] - p0[1], p2[0] - p0[0])
            e_p = np.array([loc.x, loc.y])
            actual_off = float((e_p - self.ring.P[i]) @ self.ring.nright[i])
            desired = float(offs[i])
            e0 = actual_off - desired            # right-positive error
            psi0 = wrap(heading - desired_yaw)   # right-positive
            vel = actor.get_velocity()
            vy0 = vel.x * self.ring.nright[i][0] + vel.y * self.ring.nright[i][1]
            dtc = self.t - self._prev_t if self._prev_heading is not None else DT
            r0 = (wrap(heading - self._prev_heading) / dtc
                  if self._prev_heading is not None else 0.0)
            self._prev_heading = heading
            self._prev_t = self.t
            if self._mpc_i % max(1, self.mpc_every) == 0:
                ks = []
                for kk in range(self.bicycle.N):
                    ss = s_u + speed * self.bicycle.dt * kk
                    q0 = dpt(ss - 6.0); q1 = dpt(ss); q2 = dpt(ss + 6.0)
                    yy1 = math.atan2(q1[1] - q0[1], q1[0] - q0[0])
                    yy2 = math.atan2(q2[1] - q1[1], q2[0] - q1[0])
                    ks.append(wrap(yy2 - yy1) / 6.0)
                d0, a0, ok = self.bicycle.solve(
                    e0, psi0, vy0, r0, speed, ks, desired, v_ref)
                self._mpc_cmd = (d0, a0)
            d0, a0 = self._mpc_cmd
            self._mpc_i += 1
            slew = float(self.cfg.get("steer_slew", 0.0))
            if slew > 0.0:
                step = slew * DT
                d0 = max(min(d0, self._last_bike_steer + step),
                         self._last_bike_steer - step)
            self._last_bike_steer = d0
            delta = max(min(d0, MAX_STEER), -MAX_STEER)
            thr = max(min(a0 / 4.0, 1.0), 0.0)
            brk = max(min(-a0 / 8.0, 1.0), 0.0)
            if speed > v_ref + 2.0:
                thr = 0.0; brk = min(1.0, (speed - v_ref) / 8.0)
            elif speed < v_ref - 2.0:
                thr = max(thr, min(1.0, (v_ref - speed) / 10.0)); brk = 0.0
            if self.cfg.get("mpc_debug", False) and self.step % 20 == 0:
                print("[bike] t=%.1f s=%.0f e=%.2f psi=%.3f vy=%.2f r=%.3f "
                      "d=%.3f ok=%s v=%.1f" % (
                          self.t, s_u, e0, psi0, vy0, r0, d0,
                          self.bicycle.last_ok, speed), flush=True)
            actor.apply_control(carla.VehicleControl(
                steer=delta / MAX_STEER, throttle=thr, brake=brk))
            return

        delta, thr, brk = self._pure_pursuit(actor, s_u, offs, v_ref, attack,
                                             loc, speed)
        actor.apply_control(carla.VehicleControl(
            steer=delta / MAX_STEER, throttle=thr, brake=brk))

    def lead_control(self, state, t_sim=None):
        i_l = state["i_l"]
        v = float(self.v_lead[i_l])
        t_sim = self.t if t_sim is None else t_sim
        if self.lead_policy == "variable":
            v *= 0.97 + 0.06 * math.sin(0.25 * t_sim)
        off = self.racing_off
        if self.lead_policy == "defensive":
            ahead = state["ahead_m"]
            if -20.0 < ahead < 5.0:
                side = 1.0 if state["actual_off_m"] > state["actual_off_l_m"] else -1.0
                bias = side * min(1.5, 0.05 * (20.0 - abs(ahead)))
                off = self.racing_off + bias
                v *= 0.99
        self.drive(self.lead, state["s_l_u"], off, v, False)

    def read_state(self, update=True):
        e_loc = self.ego.get_location()
        l_loc = self.lead.get_location()
        ev = self.ego.get_velocity()
        lv = self.lead.get_velocity()
        v_e = math.sqrt(ev.x ** 2 + ev.y ** 2 + ev.z ** 2)
        v_l = math.sqrt(lv.x ** 2 + lv.y ** 2 + lv.z ** 2)

        s_e = self.ring.project(e_loc.x, e_loc.y, s_hint=self.s_e_prev, window=100.0)
        s_l = self.ring.project(l_loc.x, l_loc.y, s_hint=self.s_l_prev, window=100.0)

        def unwrap(s, prev):
            d = (s - prev) % self.ring.L
            if d > self.ring.L / 2:
                d -= self.ring.L
            return prev + d

        s_e_u = unwrap(s_e, self.s_e_prev)
        s_l_u = unwrap(s_l, self.s_l_prev)
        if update:
            self.s_e_prev = s_e_u
            self.s_l_prev = s_l_u
        i_e = self.ring.s_of(s_e_u)
        i_l = self.ring.s_of(s_l_u)
        e_p = np.array([e_loc.x, e_loc.y])
        l_p = np.array([l_loc.x, l_loc.y])
        actual_off = float((e_p - self.ring.P[i_e]) @ self.ring.nright[i_e])
        actual_off_l = float((l_p - self.ring.P[i_l]) @ self.ring.nright[i_l])
        actual_extra = self.t2_sign * (actual_off - float(self.racing_off[i_e]))
        actual_extra_l = self.t2_sign * (actual_off_l - float(self.racing_off[i_l]))
        if self.prev_extra is None:
            extra_rate = 0.0
        else:
            extra_rate = (actual_extra - self.prev_extra) / DT
        if self.prev_extra_l is None:
            lead_extra_rate = 0.0
        else:
            lead_extra_rate = (actual_extra_l - self.prev_extra_l) / DT
        if update:
            self.prev_extra = actual_extra
            self.prev_extra_l = actual_extra_l
        lat_sep = actual_off - actual_off_l
        gap = s_e_u - s_l_u
        d_apex = (s_e_u - self.t2["s_apex"]) % self.ring.L
        if d_apex > self.ring.L / 2:
            d_apex -= self.ring.L
        d_tactical = (s_e_u - self.attack_s) % self.ring.L
        if d_tactical > self.ring.L / 2:
            d_tactical -= self.ring.L
        closing = (v_e - v_l) if gap <= 0 else (v_l - v_e)
        ttc = abs(gap) / max(closing, 0.5) if abs(gap) > 0 else 1e9
        lap_frac = max(0.0, (s_e_u - self.s_ego0) / self.ring.L)
        if update:
            self.lap_frac = lap_frac
            self.v_peak = max(self.v_peak, v_e)
            self.min_center = min(self.min_center, math.hypot(e_loc.x - l_loc.x, e_loc.y - l_loc.y))
            if abs(gap) < 8.0:
                self.min_sep_close = min(self.min_sep_close, abs(lat_sep))
            s_mod = s_e_u % self.ring.L
            in_attack = (self.attack_s - 120.0 <= s_mod <= self.attack_s + 120.0)
            if gap > 2.0 and in_attack and self.pass_t is None:
                self.pass_t = self.t
                self.pass_s = s_e_u % self.ring.L
        state = dict(
            s_e_u=s_e_u, s_l_u=s_l_u, i_e=i_e, i_l=i_l,
            ego_v=v_e, lead_v=v_l, gap_m=abs(gap), ahead_m=gap,
            d_apex_m=d_apex, d_tactical_m=d_tactical,
            kappa=float(self.ring.k[i_e]), ttc=ttc,
            actual_off_m=actual_off, actual_off_l_m=actual_off_l,
            actual_extra_m=actual_extra,
            actual_extra_lead_m=actual_extra_l,
            actual_extra_rate_mps=extra_rate,
            lead_extra_rate_mps=lead_extra_rate,
            lateral_sep_m=lat_sep,
            racing_off_m=float(self.racing_off[i_e]),
            inside_side="right" if self.t2_sign > 0 else "left",
            lap_frac=lap_frac,
        )
        return state

    # ---------------------------------------------------------------- policy
    def _random_action(self, state):
        d = state["d_apex_m"]
        if -260.0 <= d <= 150.0 and state["gap_m"] < 60.0:
            extra, sf = None, None
            r = self.rng.random()
            acc = 0.0
            for name, e, s, wgt in RANDOM_CANDIDATES:
                acc += wgt
                if r <= acc:
                    extra, sf = e, s
                    break
            if extra is None:
                extra, sf = RANDOM_CANDIDATES[-1][1], RANDOM_CANDIDATES[-1][2]
            return {"action": "overtake" if abs(extra) > 0.5 else "hold",
                    "extra_inside_m": extra, "speed_factor": sf,
                    "reason": "random exploration"}
        return {"action": "hold", "extra_inside_m": 0.0,
                "speed_factor": 1.0, "reason": "random: outside window"}

    def decide(self, t, state):
        if self.policy == "stub":
            raw = self.stub.decide(state)
            dm = DecisionEngine._filter(state, raw)
            return dm, raw, dm, True
        if self.policy == "hold":
            raw = {"action": "hold", "extra_inside_m": 0.0,
                   "target_offset_m": 0.0, "speed_factor": 1.0, "reason": "hold policy"}
            dm = DecisionEngine._filter(state, raw)
            return dm, raw, dm, True
        if self.policy == "openai":
            dm = self.engine.decide(self.t, state)
            return dm, dm, dm, True
        raw = self._random_action(state)
        return raw, raw, raw, False

    def apply_action(self, state, action):
        mode = self.cfg.get("controller", "pure")
        have = ((self.mpc is not None) or (self.frenet is not None)
                or (self.bicycle is not None))
        self._use_mpc = (have and (
            mode in ("mpc", "frenet", "bicycle")
            or (mode == "hybrid" and -260.0 <= state["d_apex_m"] <= 150.0)))
        i_e = state["i_e"]
        i_l = state["i_l"]
        attack = action.get("action") == "overtake"
        if "_extra_ref" in action:
            target = float(action["_extra_ref"])
        else:
            target = float(action.get("extra_inside_m", 0.0)) if attack else 0.0
        if (action.get("blocked") and abs(state["ahead_m"]) < 15.0
                and not attack and not action.get("_qp")):
            target = self.extra_cmd
        # Pass commitment: once the ego is alongside/just ahead, do not retract
        # the inside line just because the tactical layer switched to hold.
        # Keep the lateral offset until the pass is clearly complete.
        if (not attack and self.extra_cmd > 0.5
                and -5.0 < state["ahead_m"] < float(self.cfg.get("pass_clear_m", 8.0))
                and state["gap_m"] < 12.0):
            target = self.extra_cmd
        if attack and target > 0.5:
            bias = float(self.cfg.get("extra_bias_m", 0.0))
            if bias > 0.0:
                target = min(4.5, target + bias)
        step = 6.0 * DT
        if target > self.extra_cmd:
            self.extra_cmd = min(target, self.extra_cmd + step)
        elif target < self.extra_cmd:
            self.extra_cmd = max(target, self.extra_cmd - step)
        # Hold the inside line until the pass actually completes.  The
        # attack_profile window ramps down after the apex and the racing line
        # returns to the outside; on tight corners the ego has not cleared the
        # lead yet at that point, so retracting the offset steers it back onto
        # the lead's line.  Pin attack_w=1 while still alongside the lead.
        aw = self.attack_w
        if (self.extra_cmd > 0.5
                and -5.0 < state["ahead_m"] < float(self.cfg.get("pass_clear_m", 8.0))
                and state["gap_m"] < 12.0):
            aw = np.ones_like(self.attack_w)
        off = self.racing_off + self.t2_sign * aw * self.extra_cmd
        off_cap = float(self.cfg.get("off_cap_m", 0.0))
        if off_cap > 0.0:
            off = np.clip(off, -off_cap, off_cap)
        sf = float(action.get("speed_factor", 1.05))
        if "_v_ref" in action:
            v_target = float(action["_v_ref"])
        elif attack:
            v_target = float(self.v_ego[i_e]) * sf / 1.05
        else:
            v_target = float(self.v_lead[i_l])
            if (action.get("blocked") and state["gap_m"] < 10.0
                    and state["ahead_m"] < 2.0):
                # only brake when behind/level; braking while clearly ahead
                # invites the opponent to rear-end us
                v_target = min(v_target, state["ego_v"] - 3.0)

        # ---- control-layer safety envelope for the pass ----
        if attack:
            # Cap the speed advantage: a late tactical decision must not turn
            # into a 12 m/s dive-bomb at the lead car.
            dv_cap = float(self.cfg.get("attack_dv_max", 0.0))
            if dv_cap > 0.0:
                v_target = min(v_target, float(state["lead_v"]) + dv_cap)
            # Build lateral clearance before closing.  Latch once the offset is
            # established so the car commits to the pass instead of oscillating
            # alongside the lead for seconds.
            gate = float(self.cfg.get("sep_gate_m", 0.0))
            if (gate > 0.0 and self.extra_cmd > 0.5
                    and state["ahead_m"] < 2.0):
                if abs(state["lateral_sep_m"]) >= gate:
                    self.sep_latched = True
                if not self.sep_latched:
                    bias = float(self.cfg.get("sep_gate_bias", 0.5))
                    v_target = min(v_target, float(state["lead_v"]) + bias)
        else:
            self.sep_latched = False

        # per-tick shield, independent of the 0.5s policy cadence
        if (not action.get("_qp") and self.tick_shield
                and self.extra_cmd > 0.5 and (
                (state["gap_m"] < 8.0 and abs(state["lateral_sep_m"]) < 2.2
                 and -5.0 < state["ahead_m"] < 1.0) or
                (state["gap_m"] < 4.0 and abs(state["lateral_sep_m"]) < 2.5
                 and state["ahead_m"] < 0.5))):
            new_v = min(v_target, state["ego_v"] - 3.0)
            if new_v < v_target - 1e-6:
                self.shield_count += 1
            v_target = new_v
        self.prev_v_target = v_target
        self.drive(self.ego, state["s_e_u"], off, v_target, attack)
        self.lead_control(state)
        e_loc = self.ego.get_location()
        l_loc = self.lead.get_location()
        self.rows.append([
            round(self.t, 2), round(state["s_e_u"], 1), round(state["s_l_u"], 1),
            round(state["ego_v"], 2), round(state["lead_v"], 2), round(state["ahead_m"], 2),
            round(float(self.racing_off[i_e]), 2), round(float(off[i_e]), 2),
            round(state["actual_off_m"], 2), round(state["actual_off_l_m"], 2),
            round(state["lateral_sep_m"], 2), round(self.extra_cmd, 2),
            round(self.lap_frac, 3), "INSIDE" if self.t2_sign * (off[i_e] - self.racing_off[i_e]) > 0.5 else "",
            action.get("action", ""),
            round(e_loc.x, 2), round(e_loc.y, 2), round(l_loc.x, 2), round(l_loc.y, 2),
        ])
        return dict(extra_cmd=round(self.extra_cmd, 2), speed_target=round(v_target, 2),
                    attack=attack)

    # ------------------------------------------------------------ snapshot
    def snapshot(self):
        def s(a):
            return dict(tf=a.get_transform(), v=a.get_velocity(), av=a.get_angular_velocity())
        return dict(ego=s(self.ego), lead=s(self.lead),
                    extra_cmd=self.extra_cmd, s_e_prev=self.s_e_prev,
                    s_l_prev=self.s_l_prev, t=self.t)

    def restore(self, snap):
        for actor, key in ((self.ego, "ego"), (self.lead, "lead")):
            d = snap[key]
            actor.set_transform(d["tf"])
            actor.set_target_velocity(d["v"])
            actor.set_target_angular_velocity(d["av"])
            actor.apply_control(carla.VehicleControl())
        self.extra_cmd = snap["extra_cmd"]
        self.s_e_prev = snap["s_e_prev"]
        self.s_l_prev = snap["s_l_prev"]
        self.coll = {"ego": 0, "lead": 0}

    @staticmethod
    def _tf_json(tf):
        return dict(loc=[tf.location.x, tf.location.y, tf.location.z],
                    rot=[tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll])

    @staticmethod
    def _vec_json(v):
        return [v.x, v.y, v.z]

    def snapshot_json(self):
        snap = self.snapshot()
        out = dict(extra_cmd=snap["extra_cmd"], s_e_prev=snap["s_e_prev"],
                   s_l_prev=snap["s_l_prev"], t=snap["t"])
        for key in ("ego", "lead"):
            d = snap[key]
            out[key] = dict(tf=self._tf_json(d["tf"]),
                            v=self._vec_json(d["v"]),
                            av=self._vec_json(d["av"]))
        return out

    def restore_json(self, data):
        def tf_from(d):
            return carla.Transform(
                carla.Location(x=float(d["loc"][0]), y=float(d["loc"][1]), z=float(d["loc"][2])),
                carla.Rotation(pitch=float(d["rot"][0]), yaw=float(d["rot"][1]), roll=float(d["rot"][2])))

        for actor, key in ((self.ego, "ego"), (self.lead, "lead")):
            d = data[key]
            actor.set_transform(tf_from(d["tf"]))
            actor.set_target_velocity(carla.Vector3D(*[float(x) for x in d["v"]]))
            actor.set_target_angular_velocity(carla.Vector3D(*[float(x) for x in d["av"]]))
            actor.apply_control(carla.VehicleControl())
        self.extra_cmd = float(data.get("extra_cmd", 0.0))
        self.s_e_prev = float(data.get("s_e_prev", self.s_ego0))
        self.s_l_prev = float(data.get("s_l_prev", self.s_lead0))
        self.t = float(data.get("t", self.t))
        self.coll = {"ego": 0, "lead": 0}

    # --------------------------------------------------------- counterfactual
    @staticmethod
    def _cost(out, extra, initial_gap=None):
        c = 1000.0 * float(bool(out["collision"]))
        if out["min_sep_close"] < 1e8:
            c += 200.0 * max(0.0, 2.2 - out["min_sep_close"])
        c += 50.0 * max(0.0, 3.0 - out["min_center"])
        if out["pass_"]:
            c += 20.0 * max(0.0, out["t_pass"] - 1.0)
        else:
            c += 60.0
        # reward longitudinal progress: a 4-6s rollout from 20m behind may not
        # complete the pass yet, so "no pass" alone would wrongly prefer hold.
        if initial_gap is not None and out["final_gap"] is not None:
            progress = out["final_gap"] - initial_gap
            c -= 0.8 * max(-20.0, min(20.0, progress))
        c += 5.0 * abs(extra)
        return c

    def simulate(self, cand, horizon_s, t0):
        self.in_rollout = True
        self.coll = {"ego": 0, "lead": 0}
        min_center = 1e9
        min_sep_close = 1e9
        pass_t = None
        final_gap = None
        extra = float(cand["extra_inside_m"])
        sf = float(cand["speed_factor"])
        attack = abs(extra) > 0.5
        try:
            for k in range(int(round(horizon_s / DT))):
                self.world.tick()
                state = self.read_state(update=False)
                final_gap = state["ahead_m"]
                min_center = min(min_center, math.hypot(
                    self.ego.get_location().x - self.lead.get_location().x,
                    self.ego.get_location().y - self.lead.get_location().y))
                if abs(state["ahead_m"]) < 8.0:
                    min_sep_close = min(min_sep_close, abs(state["lateral_sep_m"]))
                if self.coll["ego"] or self.coll["lead"]:
                    break
                if state["ahead_m"] > 2.0 and pass_t is None and (
                        self.t2["s0"] - 25.0 <= (state["s_e_u"] % self.ring.L) <= self.t2["s1"] + 80.0):
                    pass_t = (k + 1) * DT
                off = self.racing_off + self.t2_sign * self.attack_w * extra
                if attack:
                    v_target = float(self.v_ego[state["i_e"]]) * sf / 1.05
                else:
                    v_target = float(self.v_lead[state["i_l"]])
                self.drive(self.ego, state["s_e_u"], off, v_target, attack)
                self.lead_control(state, t0 + (k + 1) * DT)
        finally:
            self.in_rollout = False
        return dict(collision=bool(self.coll["ego"] or self.coll["lead"]),
                    min_center=min_center, min_sep_close=min_sep_close,
                    pass_=bool(pass_t is not None), t_pass=(pass_t if pass_t is not None else 1e9),
                    final_gap=final_gap)

    def label_state(self, state):
        snap = self.snapshot()
        results = []
        for cand in CANDIDATES:
            self.restore(snap)
            out = self.simulate(cand, self.cfg.get("label_horizon_s", 4.0), self.t)
            results.append(dict(
                name=cand["name"],
                action=dict(action="overtake" if abs(cand["extra_inside_m"]) > 0.5 else "hold",
                            extra_inside_m=cand["extra_inside_m"],
                            speed_factor=cand["speed_factor"]),
                outcome=dict(collision=out["collision"],
                             min_center=round(out["min_center"], 3),
                             min_sep_close=(None if out["min_sep_close"] >= 1e8
                                            else round(out["min_sep_close"], 3)),
                             pass_=out["pass_"],
                             t_pass=(None if out["t_pass"] >= 1e8 else round(out["t_pass"], 3)),
                             final_gap=(None if out["final_gap"] is None else round(out["final_gap"], 2))),
                cost=round(self._cost(out, cand["extra_inside_m"], state["ahead_m"]), 2),
            ))
        self.restore(snap)
        results.sort(key=lambda x: x["cost"])
        for k, r in enumerate(results):
            r["rank"] = k
        return results

    # ------------------------------------------------------------------ run
    def run(self):
        next_dec = 0.0
        max_steps = int(self.cfg.get("max_time_s", 120.0) / DT)
        while self.step < max_steps:
            self.world.tick()
            self.step += 1
            self.t += DT
            state = self.read_state(update=True)
            if self.t + 1e-9 >= next_dec:
                action, raw, filtered, shielded = self.decide(self.t, state)
                # Snapshot only near T2; the separate label_dataset.py script
                # replays them.  Never run counterfactual rollouts inside the
                # data-collection episode: CARLA's state restore is approximate
                # and would corrupt the actual trajectory.
                snapshot = None
                if self.cfg.get("record_snapshot", True) and -260.0 <= state["d_apex_m"] <= 150.0:
                    snapshot = self.snapshot_json()
                executed = self.apply_action(state, action)
                compact = {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in state.items() if k not in ("i_e", "i_l")}
                rec = dict(episode_id=self.episode_id, t=round(self.t, 3),
                           decision=len(self.decisions),
                           state=compact,
                           history=list(self.history[-5:]),
                           policy=self.policy, lead_policy=self.lead_policy,
                           raw=raw, filtered=filtered, shielded=shielded,
                           executed=executed, snapshot=snapshot, labels=[])
                self.decisions.append(rec)
                self.history.append(compact)
                self.last_action = action
                next_dec += float(self.cfg.get("decision_dt", 0.5))
            else:
                self.apply_action(state, self.last_action)

            if any(self.coll_total.values()):
                self.coll_grace += DT
                if self.cfg.get("stop_on_collision", False) and self.coll_grace > 1.0:
                    break
            if self.lap_frac >= 1.0:
                break
        return self.summary()

    # ------------------------------------------------------------- viz hooks
    def _draw_routes(self, state):
        """Draw the lead's racing line and the ego's planned line in CARLA."""
        if not self.cfg.get("view_debug", False):
            return
        z = self.ego.get_location().z + 0.3

        def line(s0, offsets, color):
            pts = []
            for ds in range(0, 88, 4):
                i = self.ring.s_of(s0 + ds)
                p = self.ring.P[i] + self.ring.nright[i] * offsets[i]
                pts.append(carla.Location(x=float(p[0]), y=float(p[1]), z=z))
            for a, b in zip(pts[:-1], pts[1:]):
                self.world.debug.draw_line(a, b, thickness=0.18,
                                           color=color, life_time=0.25)
            return pts[0]

        lead_p0 = line(state["s_l_u"], self.racing_off,
                       carla.Color(255, 150, 40))
        plan = self.racing_off + self.t2_sign * self.attack_w * self.extra_cmd
        ego_p0 = line(state["s_e_u"], plan, carla.Color(0, 230, 255))
        if self.t - self._last_label_t >= 1.0:
            self._last_label_t = self.t
            self.world.debug.draw_string(
                carla.Location(x=lead_p0.x, y=lead_p0.y, z=z + 1.4),
                "lead racing line", draw_shadow=True,
                color=carla.Color(255, 150, 40), life_time=1.0)
            self.world.debug.draw_string(
                carla.Location(x=ego_p0.x, y=ego_p0.y, z=z + 1.4),
                "ego plan", draw_shadow=True,
                color=carla.Color(0, 230, 255), life_time=1.0)

    def _write_live(self, state, action, option):
        """Write a small JSON snapshot for the browser decision HUD."""
        path = self.cfg.get("live_status_path")
        if not path:
            return
        now = time.time()
        if now - self._last_live < 0.2:
            return
        self._last_live = now
        resp = self._last_resp or {}
        lat = resp.get("wall_latency_s", resp.get("latency_s"))
        act = action.get("action", "hold")
        if self.pass_done:
            phase = "pass complete"
        elif act == "overtake":
            phase = "overtake"
        elif -300.0 <= state["d_apex_m"] <= 150.0:
            phase = "window"
        else:
            phase = "cruise"
        self._live_history.append([
            round(self.t, 2), round(state["gap_m"], 2),
            round(state["lateral_sep_m"], 2), round(state["ego_v"], 2),
            round(state["lead_v"], 2), act])
        self._live_history = self._live_history[-150:]
        st = dict(
            t=round(self.t, 2), phase=phase,
            d_apex=round(state["d_apex_m"], 1),
            gap=round(state["gap_m"], 2), ahead=round(state["ahead_m"], 2),
            sep=round(state["lateral_sep_m"], 2),
            v_ego=round(state["ego_v"], 2), v_lead=round(state["lead_v"], 2),
            extra_cmd=round(self.extra_cmd, 2),
            action=act, reason=str(action.get("reason", ""))[:140],
            speed_factor=action.get("speed_factor"),
            latency_s=(round(float(lat), 3) if lat is not None else None),
            stale=bool(resp.get("stale", False)) if resp else False,
            pass_done=self.pass_done, pass_t=self.pass_t,
            shield=self.shield_count,
            decisions=self._live_decisions[-8:],
            history=self._live_history,
            updated_at=time.strftime("%H:%M:%S"))
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            pass

    def run_latency(self, client, decision_hz=5.0, max_time_s=None):
        """Run with an asynchronous tactical client + action chunks.

        client.request(t, state) -> bool (accepted)
        client.poll() -> None or dict(option) with:
            action, extra_inside_m, speed_factor, valid_for_s,
            request_t, ready_t, latency_s, reason
        """
        max_time_s = max_time_s or self.cfg.get("max_time_s", 90.0)
        max_steps = int(max_time_s / DT)
        realtime = bool(self.cfg.get("realtime", False))
        wall0 = time.monotonic()
        next_req = 0.0
        option = None
        pending_queue = []   # FIFO of (available_t, expire_t, option)
        n_req = 0
        n_resp = 0
        n_expired = 0
        n_stale = 0
        n_options = 0
        n_env = 0
        latencies = []
        while self.step < max_steps:
            self.world.tick()
            self.step += 1
            self.t += DT
            if realtime:
                # Keep simulated time aligned with wall time so a real 1 s
                # inference delay is a real 1 s delay.  Without this the
                # headless sim runs several times faster than real time and
                # every real response would look stale.
                lag = wall0 + self.t - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
            state = self.read_state(update=True)

            # Once the ego is clearly ahead, the pass is over: command a hold
            # and return to the racing line instead of holding the inside line
            # alongside/just ahead of the lead.
            pass_clear = float(self.cfg.get("pass_clear_m", 0.0))
            if (pass_clear > 0.0 and not self.pass_done
                    and self.pass_t is not None
                    and state["ahead_m"] > pass_clear):
                self.pass_done = True
                option = None
                pending_queue.clear()

            if self.t + 1e-9 >= next_req:
                if client.request(self.t, state):
                    n_req += 1
                    next_req = self.t + 1.0 / max(1e-3, decision_hz)
                    self.latency_events.append({
                        "event": "request", "t": round(self.t, 3),
                        "state_age_s": 0.0,
                        "state": {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in state.items() if k not in ("i_e", "i_l")},
                    })

            resp = client.poll()
            if resp is not None:
                n_resp += 1
                # Tactical envelope: the LLM may only request an inside pass
                # inside the legal window and with at least the minimum offset.
                # This mirrors the prompt constraint and the DecisionEngine
                # filter, and is separate from the per-tick signed shield.
                env_block = False
                if resp.get("action") == "overtake":
                    d_ap = float(state.get("d_apex_m", 1e9))
                    win_lo, win_hi = self.cfg.get("decision_window", (-220.0, 150.0))
                    try:
                        req_extra = float(resp.get("extra_inside_m", 0.0))
                    except Exception:
                        req_extra = 0.0
                    if not (win_lo <= d_ap <= win_hi):
                        env_block = True
                        env_reason = "outside window (d_apex=%.0f)" % d_ap
                    elif req_extra < 3.0:
                        env_block = True
                        env_reason = "extra %.1f < 3.0" % req_extra
                    if env_block:
                        n_env += 1
                        resp = dict(resp, action="hold", extra_inside_m=0.0,
                                    speed_factor=1.0,
                                    reason="envelope: " + env_reason)
                valid_for = float(resp.get("valid_for_s", 0.8))
                req_t = float(resp.get("request_t", self.t))
                expire_t = req_t + valid_for
                latency_sim = float(resp.get("latency_sim", 0.0))
                available_t = req_t + latency_sim
                wall_latency = float(resp.get("wall_latency_s", resp.get("latency_s", 0.0)))
                latencies.append(latency_sim)
                # A plan is generated at request_t and is useful until expire_t.
                # If its simulated inference latency pushes availability past the
                # validity horizon, it is already stale when it arrives.
                stale = available_t >= expire_t
                self.latency_events.append({
                    "event": "response", "t": round(self.t, 3),
                    "request_t": round(req_t, 3),
                    "ready_t": round(float(resp.get("ready_t", self.t)), 3),
                    "latency_sim": round(latency_sim, 3),
                    "wall_latency_s": round(wall_latency, 3),
                    "available_t": round(available_t, 3),
                    "expire_t": round(expire_t, 3),
                    "state_age_s": round(max(0.0, self.t - req_t), 3),
                    "stale": bool(stale),
                    "option": {k: resp.get(k) for k in
                               ("action", "extra_inside_m", "speed_factor", "reason")},
                })
                self._last_resp = dict(resp, stale=bool(stale),
                                       wall_latency_s=wall_latency)
                self._live_decisions.append({
                    "t": round(self.t, 2),
                    "d_apex": round(float(state.get("d_apex_m", 0.0)), 1),
                    "action": resp.get("action"),
                    "extra": resp.get("extra_inside_m"),
                    "latency": round(wall_latency, 2),
                    "stale": bool(stale),
                    "reason": str(resp.get("reason", ""))[:90],
                })
                self._live_decisions = self._live_decisions[-8:]
                if stale:
                    n_stale += 1
                else:
                    opt = dict(resp)
                    opt["_expire_t"] = expire_t
                    opt["_available_t"] = available_t
                    pending_queue.append((available_t, expire_t, opt))

            # Apply the newest option whose availability time has arrived.  A FIFO
            # queue is essential: at 5 Hz, a 300 ms inference delay means a new
            # response arrives before the previous one becomes available; a single
            # pending slot would silently drop every option.
            while pending_queue and pending_queue[0][0] <= self.t + 1e-9:
                av, ex, opt = pending_queue.pop(0)
                if ex <= self.t:
                    n_expired += 1
                    continue
                if option is None or av >= option.get("_available_t", -1.0):
                    option = opt
                    n_options += 1

            active = (option is not None and self.t <= option["_expire_t"]
                      and option.get("action") == "overtake"
                      and not self.pass_done)
            if option is not None and self.t > option["_expire_t"]:
                n_expired += 1
                option = None
            if active:
                action = dict(action="overtake",
                              extra_inside_m=float(option.get("extra_inside_m", 0.0)),
                              speed_factor=float(option.get("speed_factor", 1.05)),
                              reason=option.get("reason", "chunk"))
            else:
                action = dict(action="hold", extra_inside_m=0.0,
                              speed_factor=1.0, reason="no active option")
                if abs(state["ahead_m"]) < 15.0 and self.extra_cmd > 0.5:
                    action["blocked"] = True

            if self.qp is not None:
                if active:
                    nom_extra = float(option.get("extra_inside_m", 0.0))
                    nom_v = (float(self.v_ego[state["i_e"]])
                             * float(option.get("speed_factor", 1.05)) / 1.05)
                else:
                    nom_extra = 0.0
                    nom_v = float(self.v_lead[state["i_l"]])
                safe = self.qp.filter(state, nom_extra, nom_v,
                                      self.extra_cmd, self.prev_v_target, DT)
                if safe["intervened"]:
                    self.shield_count += 1
                action = dict(action="overtake" if abs(safe["extra"]) > 0.5 else "hold",
                              extra_inside_m=safe["extra"], speed_factor=1.0,
                              reason="qp shield",
                              _extra_ref=safe["extra"], _v_ref=safe["v_ref"],
                              _qp=True, qp=safe)
            executed = self.apply_action(state, action)
            self._draw_routes(state)
            self._write_live(state, action, option)
            self.decisions.append({
                "t": round(self.t, 3), "event": "apply",
                "state": {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in state.items() if k not in ("i_e", "i_l")},
                "active": bool(active), "executed": executed,
                "option_remaining_s": (round(option["_expire_t"] - self.t, 3)
                                       if active else 0.0),
            })
            if any(self.coll_total.values()):
                self.coll_grace += DT
                if self.cfg.get("stop_on_collision", False) and self.coll_grace > 1.0:
                    break
            stop_after = float(self.cfg.get("stop_after_pass_s", 0.0))
            if (stop_after > 0 and self.pass_t is not None
                    and self.t - self.pass_t >= stop_after):
                break
            stop_after_m = float(self.cfg.get("stop_after_attack_m", 0.0))
            if stop_after_m > 0 and state["d_tactical_m"] > stop_after_m:
                break
            if self.lap_frac >= 1.0:
                break

        lat_sorted = sorted(latencies)
        p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else None
        self.latency_stats = {
            "decision_hz": decision_hz,
            "requests": n_req,
            "responses": n_resp,
            "options_used": n_options,
            "expired": n_expired,
            "stale_discarded": n_stale,
            "envelope_blocks": n_env,
            "latency_mean_s": (round(sum(latencies) / len(latencies), 3) if latencies else None),
            "latency_p95_s": (round(p95, 3) if p95 is not None else None),
            "shield_interventions": self.shield_count,
        }
        summary = self.summary()
        summary.update(self.latency_stats)
        return summary

    def summary(self):
        return dict(
            episode_id=self.episode_id, config=self.cfg,
            outcome=dict(
                collision=bool(any(self.coll_total.values())),
                collisions=self.coll_total,
                pass_=self.pass_t is not None,
                pass_t=(None if self.pass_t is None else round(self.pass_t, 2)),
                pass_s=(None if self.pass_s is None else round(self.pass_s, 1)),
                min_center=(None if self.min_center >= 1e8 else round(self.min_center, 2)),
                min_sep_close=(None if self.min_sep_close >= 1e8 else round(self.min_sep_close, 2)),
                max_speed=round(self.v_peak, 2),
                lap_completed=bool(self.lap_frac >= 1.0),
                lap_time=(round(self.t, 2) if self.lap_frac >= 1.0 else None),
            ),
            n_decisions=len(self.decisions),
            n_labels=0,
            shield_interventions=self.shield_count,
            rows=len(self.rows),
        )

    def write(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "decisions.jsonl"), "w") as f:
            for r in self.decisions:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "ticks.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "s_ego", "s_lead", "v_ego", "v_lead", "gap",
                        "off_racing", "off_cmd", "off_actual", "off_actual_lead",
                        "lat_sep", "extra_cmd", "lap_frac", "phase", "policy_action",
                        "x_ego", "y_ego", "x_lead", "y_lead"])
            w.writerows(self.rows)
        if self.latency_events:
            with open(os.path.join(out_dir, "latency_events.jsonl"), "w") as f:
                for r in self.latency_events:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(self.summary(), f, indent=2)

    def close(self):
        for cs in self.coll_sensors:
            try:
                cs.stop()
                cs.destroy()
            except Exception:
                pass
        self.coll_sensors = []
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:
                pass


def run_episode(world, cfg, out_root=OUT_DEFAULT, episode_id=None):
    eid = episode_id or "ep_%s_%s_seed%d" % (
        cfg.get("policy", "stub"), cfg.get("lead_policy", "constant"), cfg.get("seed", 0))
    out_dir = os.path.join(out_root, eid)
    ep = Episode(world, cfg, eid)
    try:
        summary = ep.run()
        ep.write(out_dir)
        return summary, out_dir
    finally:
        ep.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--policy", default="stub", choices=["stub", "hold", "random", "openai"])
    ap.add_argument("--lead-policy", default="constant",
                    choices=["constant", "variable", "defensive"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-gap", type=float, default=26.0)
    ap.add_argument("--lead-factor", type=float, default=0.90)
    ap.add_argument("--ego-factor", type=float, default=1.05)
    ap.add_argument("--weather", default="ClearNoon")
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--max-labels", type=int, default=5)
    ap.add_argument("--label-every-s", type=float, default=8.0)
    ap.add_argument("--label-horizon-s", type=float, default=4.0)
    ap.add_argument("--stop-on-collision", action="store_true")
    ap.add_argument("--max-time-s", type=float, default=120.0)
    ap.add_argument("--no-tick-shield", action="store_true")
    args = ap.parse_args()

    port = int(os.environ.get("CARLA_PORT", "2000"))
    client = carla.Client("127.0.0.1", port)
    client.set_timeout(30.0)
    world = client.get_world()
    print("connected:", client.get_server_version(), "| map:", world.get_map().name)

    for k in range(args.episodes):
        cfg = dict(policy=args.policy, lead_policy=args.lead_policy, seed=args.seed + k,
                   start_gap_m=args.start_gap, lead_speed_factor=args.lead_factor,
                   ego_speed_factor=args.ego_factor, weather=args.weather,
                   label=args.label, max_labels=args.max_labels,
                   label_every_s=args.label_every_s, label_horizon_s=args.label_horizon_s,
                   stop_on_collision=args.stop_on_collision, max_time_s=args.max_time_s,
                   tick_shield=(args.policy != "random" and not args.no_tick_shield),
                   no_rendering=True)
        summary, out_dir = run_episode(world, cfg, args.out)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("saved:", out_dir)


if __name__ == "__main__":
    main()
