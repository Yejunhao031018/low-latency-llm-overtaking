#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Open-loop sign / gain identification (fresh spawn per run)."""
import argparse
import json
import math
import os

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
    i = ring.s_of(s0)
    p = ring.P[i]
    yaw = math.degrees(math.atan2(ring.tang[i][1], ring.tang[i][0]))
    loc = carla.Location(x=float(p[0]), y=float(p[1]), z=0.0)
    wp = world.get_map().get_waypoint(loc, project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    z0 = float(wp.transform.location.z) + 0.3
    tf = carla.Transform(carla.Location(x=float(p[0]), y=float(p[1]), z=z0),
                         carla.Rotation(yaw=yaw))
    a = world.try_spawn_actor(bp, tf)
    if a is None:
        raise RuntimeError("spawn failed")
    a.set_target_velocity(carla.Vector3D(ring.tang[i][0] * 8.0,
                                         ring.tang[i][1] * 8.0, 0.0))
    return a, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sysid")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    c = carla.Client("127.0.0.1", args.port)
    c.set_timeout(30.0)
    world = c.get_world()
    ring = Ring(build_ring(world))
    s0 = 470.0
    i0 = ring.s_of(s0)
    print("straight kappa:", round(ring.k[i0], 5))

    st = world.get_settings()
    st.synchronous_mode = True
    st.fixed_delta_seconds = 0.05
    st.no_rendering_mode = True
    world.apply_settings(st)
    world.tick()

    def run(steer, v_ref, T=3.0, dt=0.05):
        ego, i = spawn_ego(world, ring, s0)
        rows = []
        prev_heading = None
        prev_t = 0.0
        for k in range(int(T / dt)):
            world.tick()
            t = k * dt
            tf = ego.get_transform()
            loc = tf.location
            vel = ego.get_velocity()
            v = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            fwd = tf.rotation.get_forward_vector()
            heading = math.atan2(fwd.y, fwd.x)
            r = wrap(heading - prev_heading) / (t - prev_t) if prev_heading else 0.0
            prev_heading = heading
            prev_t = t
            s_car = ring.project(loc.x, loc.y, s_hint=s0, window=80.0)
            ii = ring.s_of(s_car)
            e = (loc.x - ring.P[ii][0]) * ring.nright[ii][0] + \
                (loc.y - ring.P[ii][1]) * ring.nright[ii][1]
            accel = v_ref - v
            thr = max(0.0, min(1.0, 0.3 + 0.15 * accel))
            brk = min(1.0, -accel / 8.0) if accel < -1.0 else 0.0
            ego.apply_control(carla.VehicleControl(throttle=thr, brake=brk,
                                                   steer=steer))
            rows.append(dict(t=t, v=v, r=r, e=e, steer=steer))
        ego.destroy()
        return rows

    results = []
    for steer in (0.04, -0.04, 0.02, -0.02):
        rows = run(steer, 10.0, T=3.0)
        win = rows[30:]
        e_slope = (win[-1]["e"] - win[0]["e"]) / (win[-1]["t"] - win[0]["t"])
        r_ss = sum(x["r"] for x in win) / len(win)
        v_ss = sum(x["v"] for x in win) / len(win)
        results.append(dict(steer=steer, e_slope=e_slope, r_ss=r_ss, v_ss=v_ss))
        print("steer %+.2f -> e_slope %+.3f m/s  r_ss %+.3f rad/s  v %.1f" % (
            steer, e_slope, r_ss, v_ss))

    gains = []
    for v_ref in (8.0, 12.0, 16.0):
        rows = run(0.03, v_ref, T=3.0)
        win = rows[30:]
        r_ss = sum(x["r"] for x in win) / len(win)
        v_ss = sum(x["v"] for x in win) / len(win)
        gains.append(dict(v=v_ss, r=r_ss, steer=0.03))
        print("v %.1f -> r_ss %+.4f  (G=%.4f)" % (v_ss, r_ss, r_ss / 0.03))

    L = 2.90
    kus = []
    for g in gains:
        if abs(g["r"]) > 1e-4 and g["v"] > 1.0:
            kus.append((g["steer"] / g["r"] - L / g["v"]) / g["v"])
    print("Kus estimates:", [round(x, 5) for x in kus])
    with open(os.path.join(args.out, "sysid.json"), "w") as f:
        json.dump(dict(sign_results=results, gains=gains, kus=kus), f, indent=2)
    print("saved:", os.path.join(args.out, "sysid.json"))


if __name__ == "__main__":
    main()
