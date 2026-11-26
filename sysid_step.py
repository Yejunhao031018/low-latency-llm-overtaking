#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust step-steer identification of the CARLA bicycle model.

Fits the symmetric (a=b=L/2, Cf=Cr=C) bicycle model from a step response:
    r_dot  = -a1 r + b1 d,   a1 = L^2 C/(2 Iz v),  b1 = L C/(2 Iz)
    vy_dot = -2C/(m v) vy - v r + C/m d
So C is identified from the vy equation, Iz from b1.  Includes smoothing and a
consistency check (a1 ~ (L/v) b1).  Prints raw samples to verify signs.

    ./run_py.sh sysid_step.py --out results/sysid_step
"""
import argparse
import json
import math
import os

import numpy as np
import carla

from race_overtake import Ring, build_ring, wrap


def pick_vehicle(bl, patterns):
    for pat in patterns:
        f = bl.filter(pat)
        if len(f) > 0:
            return f[0]
    raise RuntimeError("no vehicle blueprint")


def spawn_ego(world, ring, s0):
    bl = world.get_blueprint_library()
    bp = pick_vehicle(bl, ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.*"])
    try:
        m = float(bp.get_attribute("mass").as_float())
    except Exception:
        m = 1600.0
    i = ring.s_of(s0)
    p = ring.P[i]
    yaw = math.degrees(math.atan2(ring.tang[i][1], ring.tang[i][0]))
    loc = carla.Location(x=float(p[0]), y=float(p[1]), z=0.0)
    wp = world.get_map().get_waypoint(loc, project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    z0 = float(wp.transform.location.z) + 0.3
    a = world.try_spawn_actor(bp, carla.Transform(
        carla.Location(x=float(p[0]), y=float(p[1]), z=z0),
        carla.Rotation(yaw=yaw)))
    if a is None:
        raise RuntimeError("spawn failed")
    return a, m


def smooth(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sysid_step")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    c = carla.Client("127.0.0.1", args.port)
    c.set_timeout(30.0)
    world = c.get_world()
    ring = Ring(build_ring(world))
    s0 = 470.0
    st = world.get_settings()
    st.synchronous_mode = True
    st.fixed_delta_seconds = 0.05
    st.no_rendering_mode = True
    world.apply_settings(st)
    world.tick()

    L = 2.90
    dt = 0.05
    results = {}

    for v_ref in (20.0, 30.0):
        ego, m = spawn_ego(world, ring, s0)
        for _ in range(40):
            world.tick()
            v = math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2)
            ego.apply_control(carla.VehicleControl(
                throttle=max(0.0, min(1.0, 0.25 + 0.12 * (v_ref - v)))))
        rows = []
        prev_heading = None
        prev_t = 0.0
        for k in range(int(4.0 / dt)):
            world.tick()
            t = k * dt
            tf = ego.get_transform()
            vel = ego.get_velocity()
            v = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            fwd = tf.rotation.get_forward_vector()
            heading = math.atan2(fwd.y, fwd.x)
            r = wrap(heading - prev_heading) / dt if prev_heading else 0.0
            prev_heading, prev_t = heading, t
            vy = vel.x * math.sin(heading) - vel.y * math.cos(heading)
            thr = max(0.0, min(1.0, 0.25 + 0.12 * (v_ref - v)))
            ego.apply_control(carla.VehicleControl(throttle=thr, steer=0.03))
            rows.append([t, v, r, vy, 0.03])
        ego.destroy()

        A = np.array(rows)
        t, v, r, vy, d = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]
        r_s, vy_s = smooth(r), smooth(vy)
        r_dot = np.gradient(r_s, dt)
        vy_dot = np.gradient(vy_s, dt)
        # print raw (smoothed) samples to verify signs
        print("v=%.0f  r[10]=%.4f r_ss=%.4f vy[10]=%.3f d=%.3f" % (
            v_ref, r_s[10], np.mean(r_s[-30:]), vy_s[10], d[10]))
        # fit C from vy equation: vy_dot + v r = C * ( -(2/(m v)) vy + (1/m) d )
        X1 = -(2.0 / (m * v)) * vy_s + (1.0 / m) * d
        y1 = vy_dot + v * r_s
        # fit b1 from r equation: r_dot = -a1 r + b1 d
        X2 = d  # for b1
        y2 = r_dot + 0.0  # will do full 2-param fit below
        # robust: use samples in steady+transient window (skip first 0.3s)
        keep = np.arange(len(t)) > 6
        mask = (v > 3.0) & keep
        # 2-param fit for r: [r_dot] = [-r, d] @ [a1, b1]
        Rm = np.vstack([-r_s[mask], d[mask]]).T
        rr = r_dot[mask]
        a1, b1 = np.linalg.lstsq(Rm, rr, rcond=None)[0]
        # C from vy
        C = float(np.dot(X1[mask], y1[mask]) / max(1e-9, np.dot(X1[mask], X1[mask])))
        Iz = L * C / (2.0 * b1) if abs(b1) > 1e-9 else float("nan")
        consistency = (L / np.mean(v[mask])) * b1
        print("  a1=%.4f b1=%.4f  C=%.0f N/rad  Iz=%.0f kg*m^2  "
              "consist(a1~L/v*b1)=%.4f" % (a1, b1, C, Iz, consistency))
        results[int(v_ref)] = dict(C=C, Iz=Iz, a1=a1, b1=b1,
                                   consistency=consistency, mass=m)

    with open(os.path.join(args.out, "sysid_step.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved:", os.path.join(args.out, "sysid_step.json"))


if __name__ == "__main__":
    main()
