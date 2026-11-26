#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 CARLA 场景中放入两辆车：
  - 超车的车(自车)：蓝色 Lincoln MKZ
  - 被超的车(前车)：红色 Tesla Model3，位于自车前方约 36m 同一车道
放在 Town04 直道(road 40)上，观察者相机对准自车。脚本退出后两车保留在场景中。
用法：CARLA_PORT=2000 python spawn_two_cars.py
"""
import os
import sys
import time

import carla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overtake_mpc as om  # 复用 Path / find_straight_start / spawn_vehicle

PORT = int(os.environ.get("CARLA_PORT", "2000"))


def pick_vehicle(bl, patterns):
    for pat in patterns:
        found = bl.filter(pat)
        if len(found) > 0:
            return found[0]
    raise RuntimeError("no vehicle blueprint found")


def main():
    c = carla.Client("127.0.0.1", PORT)
    c.set_timeout(30.0)
    world = c.get_world()
    wmap = world.get_map()
    print("connected:", c.get_server_version(), "| map:", wmap.name)

    # 若上次演示异常中断处于同步模式，先恢复正常
    st = world.get_settings()
    if st.synchronous_mode:
        st.synchronous_mode = False
        world.apply_settings(st)
        print("synchronous mode disabled")
    time.sleep(0.3)

    # 清掉旧的车辆/传感器，避免重复
    for a in world.get_actors():
        t = a.type_id
        if t.startswith(("vehicle.", "walker.", "sensor.")):
            try:
                a.destroy()
            except Exception:
                pass
    time.sleep(0.6)

    # 选直道（Town04 road 40），复用 demo 的查找逻辑
    w0, chain, off = om.find_straight_start(wmap)
    print("straight road: road_id=%d lane=%d straight=%.0fm lane_offset=%.2fm (map=%s)" % (
        w0.road_id, w0.lane_id, (len(chain) - 1) * 2.0, off, wmap.name))

    bl = world.get_blueprint_library()
    ego_bp = pick_vehicle(bl, ["vehicle.lincoln.mkz_2017", "vehicle.lincoln*", "vehicle.*"])
    lead_bp = pick_vehicle(bl, ["vehicle.tesla.model3", "vehicle.audi.a2",
                                "vehicle.mercedes.coupe_2020", "vehicle.*"])
    for bp, color in ((ego_bp, "0,0,255"), (lead_bp, "255,0,0")):
        try:
            if bp.has_attribute("color"):
                bp.set_attribute("color", color)
        except Exception as e:
            print("color set skipped:", e)

    ego = om.spawn_vehicle(world, ego_bp, w0.transform)
    lead = om.spawn_vehicle(world, lead_bp, chain[18].transform)  # 36m 前方
    time.sleep(0.5)

    spectator = world.get_spectator()
    et = ego.get_transform()
    fwd = et.rotation.get_forward_vector()
    spectator.set_transform(carla.Transform(
        et.location - fwd * 14.0 + carla.Location(z=6.0),
        carla.Rotation(pitch=-14.0, yaw=et.rotation.yaw)))

    print("overtaker (blue Lincoln) id=%d loc=%s" % (ego.id, ego.get_location()))
    print("overtaken (red Tesla)  id=%d loc=%s" % (lead.id, lead.get_location()))
    print(">>> 两车已放入场景并永久保留；观察者相机已对准自车。")
    print(">>> 想看 MPC 自动超车时，运行 overtake_mpc.py 即可。")


if __name__ == "__main__":
    main()
