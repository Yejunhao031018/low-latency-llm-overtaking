#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coupled lateral+longitudinal MPC for a high-speed straight overtake.

Unlike the tactical QP, this MPC directly optimizes steering and acceleration
with a kinematic bicycle model; the lateral reference is a lane change around
the lead.  Town04 ring straight s=202..562 (~360 m), lead 38 m/s, ego 45 m/s.

Run (CARLA venv):
    ./run_py.sh coupled_mpc.py --out results/coupled_mpc
"""
import argparse
import csv
import json
import math
import os

import numpy as np
import carla

from race_overtake import Ring, build_ring, ROTATE_S
from overtake_mpc import Path, MPC, spawn_vehicle, wrap


class WP:
    def __init__(self, x, y, yaw_deg):
        self.transform = carla.Transform(
            carla.Location(x=float(x), y=float(y), z=0.0),
            carla.Rotation(yaw=float(yaw_deg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/coupled_mpc")
    ap.add_argument("--v-ref", type=float, default=45.0)
    ap.add_argument("--lead-speed", type=float, default=38.0)
    ap.add_argument("--lane-off", type=float, default=3.5)
    ap.add_argument("--start-gap", type=float, default=36.0)
    ap.add_argument("--max-time", type=float, default=30.0)
    ap.add_argument("--initial-speed", type=float, default=38.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    c = carla.Client("127.0.0.1", int(os.environ.get("CARLA_PORT", "2000")))
    c.set_timeout(30.0)
    world = c.get_world()
    print("connected:", c.get_server_version(), "| map:", world.get_map().name)

    for a in world.get_actors():
        if a.type_id.startswith(("vehicle.", "walker.", "sensor.")):
            try:
                a.destroy()
            except Exception:
                pass
    st = world.get_settings()
    st.synchronous_mode = True
    st.fixed_delta_seconds = 0.05
    st.no_rendering_mode = False
    world.apply_settings(st)
    world.tick()
    world.set_weather(carla.WeatherParameters.ClearNoon)

    ring = Ring(build_ring(world))
    ring.rotate(ring.rot_shift(ROTATE_S))
    # use the long straight before T2 (s=202..562)
    s0, s1 = 202.0, 562.0
    i0, i1 = ring.s_of(s0), ring.s_of(s1)
    wps = []
    i = i0
    while True:
        yaw = math.degrees(math.atan2(ring.tang[i][1], ring.tang[i][0]))
        wps.append(WP(ring.P[i][0], ring.P[i][1], yaw))
        if i == i1:
            break
        i = (i + 1) % ring.n
    path = Path(wps)
    print("path length incl. extrapolation:", round(path.s[-1], 1))

    bl = world.get_blueprint_library()
    ego_bp = bl.filter("vehicle.tesla.model3")[0]
    lead_bp = bl.filter("vehicle.tesla.model3")[0]
    for bp, col in ((ego_bp, "0,0,255"), (lead_bp, "255,0,0")):
        try:
            if bp.has_attribute("color"):
                bp.set_attribute("color", col)
        except Exception:
            pass
    p0x, p0y, yaw0 = path.sample(0.0)
    p1x, p1y, yaw1 = path.sample(args.start_gap)
    m = world.get_map()
    z0 = m.get_waypoint(carla.Location(x=p0x, y=p0y, z=0.0),
                        project_to_road=True).transform.location.z + 0.5
    z1 = m.get_waypoint(carla.Location(x=p1x, y=p1y, z=0.0),
                        project_to_road=True).transform.location.z + 0.5
    ego = spawn_vehicle(world, ego_bp, carla.Transform(
        carla.Location(x=p0x, y=p0y, z=z0), carla.Rotation(yaw=math.degrees(yaw0))))
    lead = spawn_vehicle(world, lead_bp, carla.Transform(
        carla.Location(x=p1x, y=p1y, z=z1), carla.Rotation(yaw=math.degrees(yaw1))))
    # rolling start, otherwise a 38 m/s lead runs away from a standing ego
    ego.set_target_velocity(carla.Vector3D(
        math.cos(yaw0) * args.initial_speed, math.sin(yaw0) * args.initial_speed, 0.0))
    lead.set_target_velocity(carla.Vector3D(
        math.cos(yaw1) * args.initial_speed, math.sin(yaw1) * args.initial_speed, 0.0))

    coll_bp = bl.find("sensor.other.collision")
    coll = {"ego": 0, "lead": 0}

    def ccb(event, name):
        coll[name] += 1
        if sum(coll.values()) <= 4:
            print("[collide]", name, event.other_actor.id)

    sensors = []
    for name, actor in (("ego", ego), ("lead", lead)):
        cs = world.spawn_actor(coll_bp, carla.Transform(), attach_to=actor)
        cs.listen(lambda e, n=name: ccb(e, n))
        sensors.append(cs)

    spectator = world.get_spectator()

    def chase():
        et = ego.get_transform()
        fw = et.rotation.get_forward_vector()
        loc = et.location - fw * 14.0 + carla.Location(z=6.0)
        spectator.set_transform(carla.Transform(
            loc, carla.Rotation(pitch=-14.0, yaw=et.rotation.yaw)))

    chase()
    mpc = MPC()
    rows = []
    t = 0.0
    step = 0
    max_steps = int(args.max_time / 0.05)
    done_hold = 0
    passed = False
    min_gap = 1e9
    max_e = 0.0
    v_max = 0.0
    phase_hist = []
    try:
        while step < max_steps and done_hold < 10:
            world.tick()
            step += 1
            t = step * 0.05
            chase()
            et = ego.get_transform()
            ev = ego.get_velocity()
            lt = lead.get_transform()
            lv = lead.get_velocity()
            speed = math.sqrt(ev.x ** 2 + ev.y ** 2 + ev.z ** 2)
            v_max = max(v_max, speed)
            lspeed = math.sqrt(lv.x ** 2 + lv.y ** 2)

            s_e, yaw_road, e_ego = path.project(et.location.x, et.location.y)
            s_l, yaw_road_l, e_lead = path.project(lt.location.x, lt.location.y)
            psi_e = wrap(math.radians(et.rotation.yaw) - yaw_road)
            psi_l = wrap(math.radians(lt.rotation.yaw) - yaw_road_l)

            # lead: lane keep + constant speed 38 m/s
            steer_l = max(min(-0.06 * e_lead - 0.45 * psi_l, 0.3), -0.3)
            err_l = args.lead_speed - lspeed
            thr_l = max(min(0.45 * err_l, 0.9), 0.0)
            brk_l = max(min(-0.15 * err_l, 0.6), 0.0)
            lead.apply_control(carla.VehicleControl(
                throttle=thr_l, brake=brk_l, steer=steer_l))

            # coupled MPC: steering + acceleration
            d0, a0, ok = mpc.solve(e_ego, psi_e, speed, s_e, s_l, args.lane_off, args.v_ref)
            dd = s_e - s_l
            if dd < -30.0:
                phase = "APPROACH"
            elif dd < -5.0:
                phase = "LANE_CHANGE_LEFT"
            elif dd <= 10.0:
                phase = "PASS"
            elif dd < 25.0:
                phase = "LANE_CHANGE_RIGHT"
            else:
                phase = "DONE"
            gap = math.hypot(et.location.x - lt.location.x, et.location.y - lt.location.y)
            min_gap = min(min_gap, gap)
            if gap < 3.0 and e_ego < args.lane_off * 0.5:
                d0, a0 = 0.0, -4.0
            steer_n = max(min(d0 / math.radians(70.0), 1.0), -1.0)
            thr = max(min(a0 / 2.5, 1.0), 0.0)
            brk = max(min(-a0 / 4.0, 1.0), 0.0) if a0 < 0 else 0.0
            # hard speed envelope around v_ref: the kinematic MPC alone can run
            # away at high speed because the horizon is short.
            if speed > args.v_ref + 2.0:
                thr = 0.0
                brk = min(1.0, (speed - args.v_ref) / 8.0)
            elif speed < args.v_ref - 2.0:
                thr = max(thr, min(1.0, (args.v_ref - speed) / 10.0))
                brk = 0.0
            ego.apply_control(carla.VehicleControl(
                steer=steer_n, throttle=thr, brake=brk))
            max_e = max(max_e, e_ego)
            if dd > 0 and not passed:
                passed = True
            if phase != (phase_hist[-1] if phase_hist else None):
                phase_hist.append(phase)
            if phase == "DONE":
                done_hold += 1
            rows.append([round(t, 2), round(s_e, 2), round(s_l, 2), round(speed, 2),
                         round(e_ego, 3), round(psi_e, 4), round(steer_n, 3),
                         round(thr, 3), round(brk, 3), phase, round(gap, 2)])
            if step % 20 == 0:
                print("t=%5.1fs v=%5.1f e=%5.2f d=%6.1f gap=%5.1f %s" % (
                    t, speed, e_ego, dd, gap, phase))
    finally:
        try:
            st2 = world.get_settings()
            st2.synchronous_mode = False
            world.apply_settings(st2)
            world.tick()
        except Exception:
            pass
        for cs in sensors:
            try:
                cs.stop(); cs.destroy()
            except Exception:
                pass

    with open(os.path.join(args.out, "telemetry.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "s_ego", "s_lead", "speed", "e", "psi", "steer",
                    "throttle", "brake", "phase", "gap"])
        w.writerows(rows)
    result = dict(
        map=world.get_map().name,
        v_ref=args.v_ref, lead_speed=args.lead_speed, lane_off=args.lane_off,
        duration=round(t, 2), passed=passed, phases=phase_hist,
        max_lateral_offset=round(max_e, 3), min_center_gap=round(min_gap, 2),
        max_speed=round(v_max, 2), collisions=coll,
    )
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
