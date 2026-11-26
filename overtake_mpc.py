#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 直道自动超车演示 (MPC 类控制)
- 在 Town04 上寻找一段长直道（>=140m，无路口，有左邻车道；本机实测 road 40）
- 自车(ego)以 MPC 跟踪“车道中心 + 超车横向偏移”参考轨迹
- 前车(lead)以约 7 m/s 匀速行驶，自车从右侧车道变到左侧车道超越后回到右侧车道
- 用鸟瞰相机 + 前视相机录帧，输出 CSV/JSON/曲线图
"""
import csv
import json
import math
import os
import time

import numpy as np
import carla
from scipy.optimize import minimize

OUT = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(OUT, "frames")
os.makedirs(FRAMES, exist_ok=True)

DT = 0.1          # MPC 内部离散步长
N = 12            # 预测步数 (1.2s)
MAX_STEER_RAD = 0.30
WHEELBASE = 2.90
V_REF = 12.5      # 自车巡航速度 m/s
LEAD_SPEED = 7.0  # 前车速度 m/s
LANE_OFF = 3.5    # 超车时相对本车道中心的左向偏移(m)，实际用实测值覆盖


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class Path:
    """由 waypoint 链生成的 2m 采样折线 (s 坐标 + 切线 + 左法向)。"""

    def __init__(self, wps):
        pts = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
        yaw = np.array([math.radians(w.transform.rotation.yaw) for w in wps])
        # 两端各外推 500m 虚拟直线，保证 ego 冲出参考路径后投影不饱和
        d0 = pts[1] - pts[0]; d0 /= np.linalg.norm(d0)
        dN = pts[-1] - pts[-2]; dN /= np.linalg.norm(dN)
        self.pts = np.vstack([pts[0] - d0 * 500.0, pts, pts[-1] + dN * 500.0])
        self.yaw = np.concatenate([[yaw[0]], yaw, [yaw[-1]]])
        d = np.diff(self.pts, axis=0)
        self.seg_len = np.linalg.norm(d, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])

    def project(self, x, y):
        """返回 (s, tangent_yaw, e_leftward) —— 投影到折线；e 为相对中心线左向偏移。"""
        p = np.array([x, y])
        a = self.pts[:-1]
        b = self.pts[1:]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / (self.seg_len ** 2), 0.0, 1.0)
        proj = a + t[:, None] * ab
        dist = np.linalg.norm(p - proj, axis=1)
        i = int(np.argmin(dist))
        s = self.s[i] + t[i] * self.seg_len[i]
        tang = ab[i] / self.seg_len[i]
        n_left = np.array([-tang[1], tang[0]])
        e = float(np.dot(p - proj[i], n_left))
        yaw = math.atan2(tang[1], tang[0])
        return s, yaw, e

    def sample(self, s):
        """按弧长 s 取 (x, y, tangent_yaw)。"""
        s = min(max(s, 0.0), self.s[-1])
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        i = min(i, len(self.pts) - 2)
        frac = (s - self.s[i]) / max(self.seg_len[i], 1e-9)
        x, y = self.pts[i] + frac * (self.pts[i + 1] - self.pts[i])
        tang = (self.pts[i + 1] - self.pts[i]) / self.seg_len[i]
        return float(x), float(y), math.atan2(tang[1], tang[0])


def find_straight_start(world_map, want_len=220.0):
    """在整张图上寻找最长的一段长直道：同一车道连续(next链)，无路口，方向偏差<3°，
    起点左邻车道可用(间距2.8-4.6m)；优先返回 >= want_len 的，否则返回最长的可用段。"""
    cands = world_map.generate_waypoints(10.0)
    best = None
    for w0 in cands:
        if w0.is_junction:
            continue
        left0 = w0.get_left_lane()
        if left0 is None or left0.is_junction:
            continue
        if left0.lane_type != carla.LaneType.Driving:
            continue
        off = math.hypot(left0.transform.location.x - w0.transform.location.x,
                         left0.transform.location.y - w0.transform.location.y)
        if not (2.8 < off < 4.6):
            continue
        n1 = w0.next(10.0)
        if not n1:
            continue
        if abs(wrap(math.radians(n1[0].transform.rotation.yaw) - math.radians(w0.transform.rotation.yaw))) > math.radians(3.0):
            continue
        chain = [w0]
        cur = w0
        yaw0 = math.radians(w0.transform.rotation.yaw)
        maxdev = 0.0
        for _ in range(140):
            nxt = cur.next(2.0)
            if not nxt:
                break
            nxt = nxt[0]
            if nxt.is_junction:
                break
            maxdev = max(maxdev, abs(wrap(math.radians(nxt.transform.rotation.yaw) - yaw0)))
            if maxdev > math.radians(3.0):
                break
            chain.append(nxt)
            cur = nxt
        length = (len(chain) - 1) * 2.0
        if length < 100.0:
            continue
        if best is None or length > best[3]:
            best = (w0, chain, off, length, maxdev)
        if length >= want_len:
            return w0, chain, off
    if best is not None and best[3] >= 120.0:
        return best[0], best[1], best[2]
    raise RuntimeError("未找到合适直道")


def spawn_vehicle(world, bp, transform):
    for dz in [0.3, 0.6, 1.0, 1.6, 2.2]:
        tf = carla.Transform(transform.location + carla.Location(z=dz), transform.rotation)
        a = world.try_spawn_actor(bp, tf)
        if a is not None:
            return a
    raise RuntimeError("spawn failed")


class MPC:
    def __init__(self):
        self.last = np.zeros(2 * N)

    def solve(self, e0, psi0, v0, s0, lead_s, off, v_ref, kappa=0.0):
        def predict(z):
            e, psi, v = e0, psi0, v0
            s = s0
            es, psis, vs = [], [], []
            for k in range(N):
                d = z[2 * k]
                a = z[2 * k + 1]
                e = e + DT * v * psi
                # road curvature feedforward: the path yaw rotates by v*kappa
                psi = psi + DT * (v * math.tan(d) / WHEELBASE - v * kappa)
                v = max(v + DT * a, 0.0)
                s = s + DT * v
                es.append(e); psis.append(psi); vs.append(v)
            return es, psis, vs

        def cost(z):
            es, psis, vs = predict(z)
            c = 0.0
            for k in range(N):
                s_k = s0 + DT * sum(vs[:k]) if k > 0 else s0
                # 参考横向偏移（相对当前前车位置）
                dd = s_k - lead_s
                if dd < -30.0:
                    er = 0.0
                elif dd < -5.0:
                    er = off * smoothstep((dd + 30.0) / 25.0)
                elif dd <= 10.0:
                    er = off
                elif dd < 25.0:
                    er = off * (1.0 - smoothstep((dd - 10.0) / 15.0))
                else:
                    er = 0.0
                c += 4.0 * (es[k] - er) ** 2 + 2.0 * psis[k] ** 2 + 0.35 * (vs[k] - v_ref) ** 2
                c += 0.8 * z[2 * k] ** 2 + 0.25 * z[2 * k + 1] ** 2
            erN = 0.0
            c += 12.0 * (es[-1] - erN) ** 2 + 6.0 * psis[-1] ** 2
            return c

        bounds = []
        for k in range(N):
            bounds += [(-MAX_STEER_RAD, MAX_STEER_RAD), (-5.0, 3.0)]
        x0 = self.last.copy()
        res = minimize(cost, x0, method="SLSQP", bounds=bounds,
                       options={"maxiter": 100, "ftol": 1e-3})
        if res.success or res.nit > 3:
            self.last = res.x
            return float(res.x[0]), float(res.x[1]), True
        return float(self.last[0]), float(self.last[1]), False


def main():
    port = int(os.environ.get("CARLA_PORT", "2000"))
    c = carla.Client("127.0.0.1", port)
    c.set_timeout(30.0)
    world = c.get_world()
    world_map = world.get_map()
    print("connected:", c.get_server_version(), "| map:", world_map.name)

    # 清理动态 actors，保持干净场景
    for a in world.get_actors():
        t = a.type_id
        if t.startswith(("vehicle.", "walker.")) and a.attributes.get("role_name") != "static.prop.parkingmeter":
            a.destroy()
        elif t.startswith("sensor."):
            a.destroy()
    time.sleep(0.5)

    # 同步模式
    st = world.get_settings()
    st.synchronous_mode = True
    st.fixed_delta_seconds = 0.05
    st.no_rendering_mode = False
    world.apply_settings(st)
    world.tick()

    world.set_weather(carla.WeatherParameters.ClearNoon)

    # 找直道
    w0, chain, off = find_straight_start(world_map)
    path = Path(chain)
    print("straight road: road_id=%d lane=%d  length=%.0fm  lane_offset=%.2fm" %
          (w0.road_id, w0.lane_id, path.s[-1], off))

    bl = world.get_blueprint_library()
    def pick_vehicle(patterns):
        for pat in patterns:
            found = bl.filter(pat)
            if len(found) > 0:
                return found[0]
        raise RuntimeError("no vehicle blueprint found")
    ego_bp = pick_vehicle(["vehicle.lincoln.mkz_2017", "vehicle.lincoln*", "vehicle.*"])
    lead_bp = pick_vehicle(["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.mercedes.coupe_2020", "vehicle.*"])
    for bp, color in [(ego_bp, "0,0,255"), (lead_bp, "255,0,0")]:
        try:
            if bp.has_attribute("color"):
                bp.set_attribute("color", color)
        except Exception as e:
            print("color attr skipped:", e)
    try:
        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", "ego")
        if lead_bp.has_attribute("role_name"):
            lead_bp.set_attribute("role_name", "lead")
    except Exception:
        pass

    ego = spawn_vehicle(world, ego_bp, w0.transform)
    lead_s0 = 36.0
    lead_wp = chain[int(lead_s0 / 2.0)]
    lead = spawn_vehicle(world, lead_bp, lead_wp.transform)
    print("ego spawned:", ego.get_location(), "| lead spawned:", lead.get_location())

    # 相机：鸟瞰 + 前视
    cam_bird = None
    cam_front = None
    frames = {"bird": 0, "front": 0}

    def attach_cam(name, tf, w=800, h=600, fov=90):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(w))
        bp.set_attribute("image_size_y", str(h))
        bp.set_attribute("fov", str(fov))
        cam = world.spawn_actor(bp, tf, attach_to=ego)

        def cb(image, name=name):
            if image.frame % 16 != 0:
                return
            idx = frames[name]
            image.save_to_disk(os.path.join(FRAMES, "%s_%03d.png" % (name, idx)))
            frames[name] += 1
        cam.listen(cb)
        return cam

    # 观察者相机跟随自车（用户窗口里直接能看到）
    spectator = world.get_spectator()

    def follow_spectator():
        et = ego.get_transform()
        fwd = et.rotation.get_forward_vector()
        loc = et.location - fwd * 14.0 + carla.Location(z=6.0)
        spectator.set_transform(carla.Transform(loc, carla.Rotation(pitch=-14.0, yaw=et.rotation.yaw)))

    follow_spectator()

    cam_bird = attach_cam("bird", carla.Transform(carla.Location(z=55.0), carla.Rotation(pitch=-90.0)), 640, 480, 90)
    cam_front = attach_cam("front", carla.Transform(carla.Location(x=-6.5, z=2.6), carla.Rotation(pitch=-6.0)), 800, 450, 90)

    mpc = MPC()
    rows = []
    t = 0.0
    done_hold = 0
    passed = False
    min_gap = 1e9
    max_e = 0.0
    v_max = 0.0
    phase_hist = []

    max_steps = int(30.0 / 0.05)
    last_print = -1
    step = 0
    try:
        while step < max_steps and done_hold < 60:
            world.tick()
            step += 1
            t = step * 0.05

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

            # 前车：速度保持 + 车道保持（小增益）
            psi_lead = wrap(math.radians(lt.rotation.yaw) - yaw_road_l)
            steer_l = max(min(-0.06 * e_lead - 0.45 * psi_lead, 0.3), -0.3)
            err = LEAD_SPEED - lspeed
            thr_l = max(min(0.45 * err, 0.9), 0.0)
            brk_l = max(min(-0.15 * err, 0.6), 0.0)
            lead.apply_control(carla.VehicleControl(throttle=thr_l, brake=brk_l, steer=steer_l))

            # MPC
            d0, a0, okm = mpc.solve(e_ego, psi_e, speed, s_e, s_l, off, V_REF)
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

            # 安全兜底
            gap = math.hypot(et.location.x - lt.location.x, et.location.y - lt.location.y)
            min_gap = min(min_gap, gap)
            if gap < 3.0 and e_ego < off * 0.5:
                d0, a0 = 0.0, -4.0

            steer_n = max(min(d0 / math.radians(70.0), 1.0), -1.0)
            thr = max(min(a0 / 2.5, 1.0), 0.0)
            brk = max(min(-a0 / 4.0, 1.0), 0.0) if a0 < 0 else 0.0
            ego.apply_control(carla.VehicleControl(steer=steer_n, throttle=thr, brake=brk))

            follow_spectator()
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

            if step - last_print >= 20:
                last_print = step
                print("t=%5.1fs v=%5.1f m/s e=%5.2f d=%6.1f gap=%5.1f steer=%6.3f %s" %
                      (t, speed, e_ego, dd, gap, steer_n, phase))
    finally:
        try:
            st = world.get_settings()
            st.synchronous_mode = False
            world.apply_settings(st)
            world.tick()
        except Exception as e:
            print("restore settings failed:", e)
        for cam in (cam_bird, cam_front):
            if cam is not None:
                try:
                    cam.stop()
                    cam.destroy()
                except Exception:
                    pass

    follow_spectator()  # 循环结束后仍停在自车附近
    print(">>> 演示结束。车辆保留在场景中(ego/lead)，观察者相机已停在自车附近。")

    # 输出 csv / json
    with open(os.path.join(OUT, "telemetry.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "s_ego", "s_lead", "speed", "e", "psi", "steer", "throttle", "brake", "phase", "gap"])
        w.writerows(rows)

    result = {
        "map": world_map.name,
        "road_id": w0.road_id,
        "lane": w0.lane_id,
        "lane_offset_m": off,
        "duration_s": t,
        "passed": passed,
        "max_lateral_offset_m": round(max_e, 3),
        "min_center_gap_m": round(min_gap, 2),
        "max_speed_mps": round(v_max, 2),
        "phases": phase_hist,
        "frames": frames,
    }
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("=" * 70)
    print("RESULT:", json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70)

    # 曲线图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tt = np.array([r[0] for r in rows])
        fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        ax[0].plot(tt, [r[3] for r in rows], "b", label="ego speed")
        ax[0].plot(tt, [r[2] - r[1] for r in rows], "0.3", label="lead-ego s (rel)")
        ax[0].set_ylabel("m/s / m"); ax[0].legend(); ax[0].grid(True)
        ax[1].plot(tt, [r[4] for r in rows], "g")
        ax[1].axhline(off, ls="--", c="r", label="lane offset")
        ax[1].set_ylabel("lateral e (m)"); ax[1].legend(); ax[1].grid(True)
        ax[2].plot(tt, [r[10] for r in rows], "m")
        ax[2].set_ylabel("center gap (m)"); ax[2].set_xlabel("t (s)"); ax[2].grid(True)
        fig.suptitle("CARLA MPC overtake demo - Town04 road 40")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "telemetry.png"), dpi=120)
        print("plot saved")
    except Exception as e:
        print("plot failed:", e)

    # 保留 ego/lead 在场景中，便于继续观察；如需要可再清理
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
