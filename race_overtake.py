#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环形赛道竞速超车演示（Town04 高速环路）
闭环 3028m。赛段起点设在 T2 之前约 360m 的直线段上（旋转后的 s≈202）：
  - T2 = R151 右弯（s=562..814, apex=688）<- 超车点
  - 前车(红): 沿竞速线（外-内-外），常规刹车点
  - 后车(蓝): 竞速线 + 在 T2 向弯心额外偏移、晚刹车，在弯中完成超越

坐标系说明：CARLA 是左手系（x 前、y 右、z 上），本文件的 ring.nright
沿车辆右侧；横向偏移为正 = 车道中心线右侧。右弯的内侧就是右侧。
旧版本把 (-ty, tx) 误当左法向，导致“内线”实际跑成了外线；本版已修正。

运行: CARLA_PORT=2000 python race_overtake.py
      USE_LLM_DECISION=1 LLM_BACKEND=stub python race_overtake.py
输出: frames_race/*.png, telemetry_race.csv, result_race.json
"""
import csv
import json
import math
import os

import numpy as np

import carla

from llm_decision import DecisionEngine

OUT = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(OUT, "frames_race")
os.makedirs(FRAMES, exist_ok=True)

DT = 0.05
WHEELBASE = 2.9
MAX_STEER = math.radians(70.0)
V_STRAIGHT = 40.0
A_LAT = 6.5
A_BRAKE_LEAD = 5.0
A_BRAKE_EGO = 7.5
A_ACC = 4.0

# 旋转起点：原环 s=1160 附近（T3 之后的长直线）
ROTATE_S = 1160


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def smooth01(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def smooth_list(v, w, loop):
    n = len(v)
    out = []
    for i in range(n):
        if loop:
            acc = sum(v[(i + o) % n] for o in range(-w, w + 1)) / (2 * w + 1)
        else:
            a = max(0, i - w); b = min(n, i + w + 1)
            acc = sum(v[a:b]) / (b - a)
        out.append(acc)
    return out


def build_ring(world):
    """从 road40 的 -2 车道出发，闭环 next() 链。"""
    m = world.get_map()
    wps = m.generate_waypoints(10.0)
    # deterministic start: pick the road-40 waypoint with the smallest (x,y)
    r40 = [x for x in wps if x.road_id == 40]
    w = min(r40, key=lambda x: (x.transform.location.x, x.transform.location.y))
    for _ in range(2):
        left = w.get_left_lane()
        if left is None:
            break
        w = left
    pos0 = w.transform.location
    prev = w
    pts = [(pos0.x, pos0.y)]
    closed = False
    for step in range(1, 3500):
        nxts = prev.next(2.0)
        if not nxts:
            break
        pyaw = math.radians(prev.transform.rotation.yaw)
        best = None
        for cand in nxts:
            ny = math.radians(cand.transform.rotation.yaw)
            d = abs(wrap(ny - pyaw))
            if best is None or d < best[0]:
                best = (d, cand)
        prev = best[1]
        p = prev.transform.location
        pts.append((p.x, p.y))
        if step > 1200 and math.hypot(p.x - pos0.x, p.y - pos0.y) < 25.0:
            closed = True
            break
    assert closed, "ring not closed"
    return pts


class Ring:
    def __init__(self, pts):
        P = np.array(pts)
        n = len(P)
        d = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
        tang = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        self.P = P
        self.n = n
        self.L = n * 2.0
        self.tang = tang
        # CARLA uses a left-handed frame (x forward, y right, z up).
        # (-ty, tx) is therefore the RIGHT normal, not the left one.
        # All lateral offsets below are expressed along this right normal,
        # so positive offset = to the right of the lane centreline.
        self.nright = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        # 稳健曲率（周期性）
        w = 5
        k = np.zeros(n)
        for i in range(n):
            a = P[(i - w) % n]; b = P[i]; c = P[(i + w) % n]
            ab = np.linalg.norm(b - a); bc = np.linalg.norm(c - b); ca = np.linalg.norm(a - c)
            area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
            R = (ab * bc * ca) / (4 * area) if area > 1e-9 else 1e9
            cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            k[i] = (1.0 / R) * (1 if cross > 0 else -1)
        self.k = np.array(smooth_list(list(k), 4, True))
        self.corners = self._corners()

    def _corners(self):
        k = self.k
        n = self.n
        thr = 1.0 / 300.0
        corners = []
        i = 0
        while i < n:
            if abs(k[i]) > thr:
                j = i
                while j < n and abs(k[j]) > thr:
                    j += 1
                if (j - i) >= 20:  # 仅保留跨度 >= 40m 的弯
                    kmax = max(k[i:j], key=abs)
                    # In CARLA's left-handed (x,y) plane a positive cross
                    # product means a right-hand turn.  inside_sign is the
                    # multiplier that moves an offset toward the corner
                    # inside along ring.nright (+1 right, -1 left).
                    right = kmax > 0
                    corners.append(dict(
                        s0=i * 2, s1=(j - 1) * 2, s_apex=((i + j) // 2) * 2,
                        R=1.0 / abs(kmax), dir="R" if right else "L",
                        inside_sign=1.0 if right else -1.0))
                i = j
            else:
                i += 1
        return corners

    def rot_shift(self, s0):
        return int(round(s0 / 2.0)) % self.n

    def rotate(self, idx):
        self.P = np.roll(self.P, -idx, axis=0)
        self.k = np.roll(self.k, -idx)
        self.nright = np.roll(self.nright, -idx, axis=0)
        self.tang = np.roll(self.tang, -idx, axis=0)
        self.corners = self._corners()

    def pt(self, i):
        """索引 i 处：位置/切线/右法向 (返回 numpy)"""
        i = i % self.n
        return self.P[i], self.tang[i], self.nright[i]

    def s_of(self, s):
        return int(round(s / 2.0)) % self.n

    def project(self, x, y, s_hint=None, window=100.0):
        p = np.array([x, y])
        if s_hint is None:
            idx = np.arange(self.n)
        else:
            # local projection: the global nearest point can jump to a parallel
            # road/overpass in Town04 at high speed, so search near the previous s
            ci = int(round((float(s_hint) % self.L) / 2.0)) % self.n
            half = max(1, int(round(window / 2.0)))
            idx = (np.arange(ci - half, ci + half + 1)) % self.n
        d = self.P[idx] - p
        i_local = int(np.argmin((d ** 2).sum(1)))
        i = int(idx[i_local])
        a = self.P[i]; b = self.P[(i + 1) % self.n]
        ab = b - a
        L2 = float(ab @ ab)
        t = float(np.clip(((p - a) @ ab) / max(L2, 1e-9), 0.0, 1.0))
        return (i * 2.0 + t * math.sqrt(L2)) % self.L


def build_ring_canonical(world):
    """build_ring with a canonical traversal direction.

    The next() chain direction can flip across CARLA restarts, which flips the
    sign of k and thus the left/right of every corner.  The training data uses
    corner index 1 (T2) as a RIGHT turn, so we pin on that and reverse the ring
    if it came out left-handed.
    """
    pts = build_ring(world)
    r = Ring(pts)
    t2 = min(r.corners, key=lambda c: abs(c["R"] - 188.0))
    if float(t2.get("inside_sign", 1.0)) < 0.0:
        pts = pts[::-1]
    return pts


def corner_offsets(ring, off_out=2.0, off_in=3.3):
    """全圈竞速线：入弯/出弯外侧，弯心内侧。

    偏移沿 ring.nright（CARLA 的 +y / 右侧）表达：
      - 右弯 inside_sign=+1：入弯/出弯 -off_out（左/外侧），弯心 +off_in（右/内侧）
      - 左弯 inside_sign=-1：入弯/出弯 +off_out（右/外侧），弯心 -off_in（左/内侧）

    返回只含竞速线的数组；超车时的额外内线偏移由 attack_profile() 在运行时叠加。
    """
    n = ring.n
    L = ring.L
    racing = np.zeros(n)
    for c in ring.corners:
        s0, s1, sa = c['s0'], c['s1'], c['s_apex']
        sgn = c.get('inside_sign', 1.0 if c['dir'] == 'R' else -1.0)
        out = -off_out * sgn          # outside = opposite of inside
        inn = off_in * sgn            # inside
        span_enter = 55.0
        span_exit = 55.0
        for i in range(n):
            s = i * 2.0
            # 相对弯起点的有符号距离
            d = (s - (s0 - span_enter)) % L
            if d > L / 2:
                d -= L
            # NOTE: every branch must also test the lower bound.  The old
            # code used `elif d <= ...`, so negative d (before the corner)
            # fell into the interpolation branch with t<0 and overwrote the
            # inside line with the outside value for every corner except the
            # last one in the list.
            if 0.0 <= d <= span_enter:
                racing[i] = out
            elif span_enter < d <= span_enter + (sa - s0):
                t = (d - span_enter) / max(1e-9, sa - s0)
                racing[i] = out + (inn - out) * smooth01(t)
            elif span_enter + (sa - s0) < d <= span_enter + (sa - s0) + (s1 - sa):
                t = (d - span_enter - (sa - s0)) / max(1e-9, s1 - sa)
                racing[i] = inn + (out - inn) * smooth01(t)
            elif (span_enter + (sa - s0) + (s1 - sa)
                  < d <= span_enter + (sa - s0) + (s1 - sa) + span_exit):
                racing[i] = out
    racing = np.array(smooth_list(list(racing), 6, True))
    return racing


def attack_profile(ring, corner, before=240.0, hold=120.0, after=30.0, ramp_out=50.0):
    """T2 内线攻击的 0..1 空间窗：apex 前 before 米渐入，后 after 米保持，再 ramp_out 米退出。"""
    n = ring.n
    L = ring.L
    w = np.zeros(n)
    for i in range(n):
        s = i * 2.0
        d = (s - corner['s_apex']) % L
        if d > L / 2:
            d -= L
        if -before <= d < -hold:
            w[i] = smooth01((d + before) / max(1e-9, before - hold))
        elif -hold <= d <= after:
            w[i] = 1.0
        elif after < d <= after + ramp_out:
            w[i] = 1.0 - smooth01((d - after) / max(1e-9, ramp_out))
    return w


def speed_profile_field(ring, factor_arr, a_brake, a_acc=A_ACC, a_lat=A_LAT, v_max=V_STRAIGHT):
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(ring.k), 1e-4)) * factor_arr)
    n = ring.n
    ds = 2.0
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            v[i] = min(v[i], math.sqrt(v[(i + 1) % n] ** 2 + 2 * a_brake * ds))
        for i in range(n):
            v[i] = min(v[i], math.sqrt(v[(i + 1) % n] ** 2 + 2 * a_brake * ds))
    for i in range(n):
        v[i] = min(v[i], math.sqrt(v[(i - 1) % n] ** 2 + 2 * a_acc * ds))
    return v


def main():
    # only clear old frames when actually running the demo (not on import)
    for _f in os.listdir(FRAMES):
        try:
            os.remove(os.path.join(FRAMES, _f))
        except OSError:
            pass
    port = int(os.environ.get("CARLA_PORT", "2000"))
    c = carla.Client("127.0.0.1", port)
    c.set_timeout(30.0)
    world = c.get_world()
    print("connected:", c.get_server_version(), "| map:", world.get_map().name)

    for a in world.get_actors():
        if a.type_id.startswith(("vehicle.", "walker.", "sensor.")):
            a.destroy()
    st = world.get_settings()
    st.synchronous_mode = True
    st.fixed_delta_seconds = DT
    world.apply_settings(st)
    world.tick()
    world.set_weather(carla.WeatherParameters.ClearNoon)

    print("building ring ...")
    ring = Ring(build_ring(world))
    ring.rotate(ring.rot_shift(ROTATE_S))
    print("ring: %.0fm, %d pts" % (ring.L, ring.n))
    for num, cn in enumerate(ring.corners, 1):
        print("  turn %d: s=%4d..%4d  minR=%3dm %s  apex=%4d" % (
            num, cn['s0'], cn['s1'], round(cn['R']), cn['dir'], cn['s_apex']))
    t2 = ring.corners[1]
    print("T2 (overtake corner): s=%d..%d apex=%d R=%dm %s" % (t2['s0'], t2['s1'], t2['s_apex'], round(t2['R']), t2['dir']))

    t2_sign = float(t2.get('inside_sign', 1.0 if t2['dir'] == 'R' else -1.0))
    i_apex = ring.s_of(t2['s_apex'])

    use_llm = os.environ.get("USE_LLM_DECISION") == "1"
    llm_engine = None
    if use_llm:
        try:
            llm_engine = DecisionEngine(os.environ.get("LLM_BACKEND", "stub"))
        except Exception as e:  # configuration errors must never abort the drive
            print("[LLM] init failed, fallback to hold:", e)
            llm_engine = DecisionEngine("hold")
        print("LLM decision backend:", llm_engine.backend_name)

    racing_off = corner_offsets(ring)
    attack_w = attack_profile(ring, t2)
    # 固定内线量（无 LLM 基线）；LLM 模式下由决策动态给出 0..3m
    extra_cmd = 0.0 if llm_engine is not None else 3.5
    n = ring.n
    fac_lead = np.full(n, 0.90)
    fac_ego = np.full(n, 0.90)
    # T2 区域内后车提速（超车窗口），其余全圈同速编队
    for i in range(n):
        s_pos = i * 2.0
        d = (s_pos - t2['s_apex']) % ring.L
        if d > ring.L / 2:
            d -= ring.L
        if -190.0 <= d <= 90.0:
            w = 1.0
            if d < -150.0:
                w = smooth01((d + 190.0) / 40.0)
            elif d > 50.0:
                w = 1.0 - smooth01((d - 50.0) / 40.0)
            fac_ego[i] = 0.90 + (1.05 - 0.90) * w
    v_lead = speed_profile_field(ring, fac_lead, A_BRAKE_LEAD)
    v_ego = speed_profile_field(ring, fac_ego, A_BRAKE_EGO)

    # 起点：T2 前 ~360m
    s_ego0 = (t2['s0'] - 360.0) % ring.L
    s_lead0 = (s_ego0 + 26.0) % ring.L

    def path_pt(s, offs):
        i = ring.s_of(s)
        p = ring.P[i] + ring.nright[i] * offs[i]
        yaw = math.atan2(ring.tang[i][1], ring.tang[i][0])
        return p, yaw

    bl = world.get_blueprint_library()
    lead_bp = bl.filter("vehicle.tesla.model3")[0]
    ego_bp = bl.filter("vehicle.tesla.model3")[0]
    for bp, col in ((lead_bp, "255,0,0"), (ego_bp, "0,0,255")):
        try:
            if bp.has_attribute("color"):
                bp.set_attribute("color", col)
        except Exception:
            pass

    def spawn(bp, s, offs):
        p, yaw = path_pt(s, offs)
        for dz in (0.4, 0.8, 1.5, 2.5):
            a = world.try_spawn_actor(bp, carla.Transform(
                carla.Location(x=p[0], y=p[1], z=dz), carla.Rotation(yaw=math.degrees(yaw))))
            if a:
                return a
        raise RuntimeError("spawn failed")

    lead = spawn(lead_bp, s_lead0, racing_off)
    ego = spawn(ego_bp, s_ego0, racing_off)
    print("spawn lead s=%.0f, ego s=%.0f (ego %.0fm behind)" % (s_lead0, s_ego0, s_lead0 - s_ego0 if s_lead0 > s_ego0 else 0))

    spectator = world.get_spectator()
    bird_bp = bl.find("sensor.camera.rgb")
    bird_bp.set_attribute("image_size_x", "800")
    bird_bp.set_attribute("image_size_y", "600")
    bird_bp.set_attribute("fov", "110")
    bird_cam = world.spawn_actor(bird_bp, carla.Transform(
        carla.Location(z=65.0), carla.Rotation(pitch=-90.0)), attach_to=ego)
    front_bp = bl.find("sensor.camera.rgb")
    front_bp.set_attribute("image_size_x", "800")
    front_bp.set_attribute("image_size_y", "450")
    front_bp.set_attribute("fov", "90")
    front_cam = world.spawn_actor(front_bp, carla.Transform(
        carla.Location(x=-6.5, z=2.6), carla.Rotation(pitch=-6.0)), attach_to=ego)
    frames = {"bird": 0, "front": 0}

    def make_cb(name):
        def _cb(image):
            if image.frame % 6 != 0:
                return
            image.save_to_disk(os.path.join(FRAMES, "%s_%03d.png" % (name, frames[name])))
            frames[name] += 1
        return _cb

    bird_cam.listen(make_cb("bird"))
    front_cam.listen(make_cb("front"))

    # 碰撞检测
    coll_bp = bl.find("sensor.other.collision")
    coll_counts = {"ego": 0, "lead": 0}
    coll_sensors = []
    for name, actor in (("ego", ego), ("lead", lead)):
        cs = world.spawn_actor(coll_bp, carla.Transform(), attach_to=actor)
        def ccb(event, name=name):
            coll_counts[name] += 1
            if coll_counts[name] <= 4:
                oa = event.other_actor
                try:
                    d = math.hypot(event.actor.get_location().x - oa.get_location().x,
                                   event.actor.get_location().y - oa.get_location().y)
                except Exception:
                    d = -1
                print("[collide] %s hits id=%s dist=%.2fm impulse=(%.0f,%.0f)" % (
                    name, oa.id, d, event.normal_impulse.x, event.normal_impulse.y))
        cs.listen(ccb)
        coll_sensors.append(cs)

    def drive(actor, s_u, offs, v_ref, look_gain=0.55, look_min=6.0, look_max=20.0):
        loc = actor.get_location()
        v = actor.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        look = max(look_min, min(look_max, speed * look_gain))
        i = ring.s_of(s_u + look)
        tgt = ring.P[i] + ring.nright[i] * offs[i]
        tf = actor.get_transform()
        fwd = tf.rotation.get_forward_vector()
        heading = math.atan2(fwd.y, fwd.x)
        alpha = wrap(math.atan2(tgt[1] - loc.y, tgt[0] - loc.x) - heading)
        dist = math.hypot(tgt[0] - loc.x, tgt[1] - loc.y)
        delta = math.atan2(2.0 * WHEELBASE * math.sin(alpha), max(dist, 0.5))
        delta = max(min(delta, MAX_STEER), -MAX_STEER)
        err = v_ref - speed
        accel = max(min(1.5 * err, A_ACC), -A_BRAKE_EGO)
        if accel >= 0:
            thr = min(1.0, accel / 4.0); brk = 0.0
        else:
            thr = 0.0; brk = min(1.0, -accel / 8.0)
        actor.apply_control(carla.VehicleControl(
            steer=delta / MAX_STEER, throttle=thr, brake=brk))

    s_e_prev = s_ego0
    s_l_prev = s_lead0
    s_e = s_ego0
    s_l = s_lead0
    t = 0.0
    rows = []
    passed_at = None
    passed_s = None
    min_gap = 1e9
    min_dist = 1e9
    v_peak = 0.0
    extra_peak = 0.0
    shield_count = 0
    lap_done = False
    lap_frac = 0.0
    lap_time = None

    def chase():
        et = ego.get_transform()
        fw = et.rotation.get_forward_vector()
        loc = et.location - fw * 12.0 + carla.Location(z=5.0)
        spectator.set_transform(carla.Transform(loc, carla.Rotation(
            pitch=-12.0, yaw=et.rotation.yaw)))

    chase()
    try:
        for step in range(int(150.0 / DT)):
            world.tick()
            t += DT
            chase()
            e_loc = ego.get_location(); l_loc = lead.get_location()
            ev = ego.get_velocity(); lv = lead.get_velocity()
            v_e = math.sqrt(ev.x ** 2 + ev.y ** 2 + ev.z ** 2)
            v_l = math.sqrt(lv.x ** 2 + lv.y ** 2 + lv.z ** 2)
            v_peak = max(v_peak, v_e)

            s_e = ring.project(e_loc.x, e_loc.y)
            s_l = ring.project(l_loc.x, l_loc.y)
            def unwrap(s, prev):
                d = (s - prev) % ring.L
                if d > ring.L / 2:
                    d -= ring.L
                return prev + d
            s_e_u = unwrap(s_e, s_e_prev)
            s_l_u = unwrap(s_l, s_l_prev)
            s_e_prev = s_e_u
            s_l_prev = s_l_u

            gap = s_e_u - s_l_u
            min_gap = min(min_gap, abs(gap))
            min_dist = min(min_dist, math.hypot(e_loc.x - l_loc.x, e_loc.y - l_loc.y))

            i_e = ring.s_of(s_e_u)
            i_l = ring.s_of(s_l_u)

            # 实测横向位置（相对车道中心线与竞速线），供安全过滤使用
            e_p = np.array([e_loc.x, e_loc.y])
            l_p = np.array([l_loc.x, l_loc.y])
            actual_off = float((e_p - ring.P[i_e]) @ ring.nright[i_e])
            actual_off_l = float((l_p - ring.P[i_l]) @ ring.nright[i_l])
            lat_sep = actual_off - actual_off_l
            actual_extra = t2_sign * (actual_off - float(racing_off[i_e]))

            # ---- LLM 上层决策（可插拔，带安全过滤） ----
            attack = True
            llm_action = ""
            target_extra = 3.5
            if llm_engine is not None:
                d_apex = (s_e_u - t2['s_apex']) % ring.L
                if d_apex > ring.L / 2:
                    d_apex -= ring.L
                ahead_m = s_e_u - s_l_u
                gap_abs = abs(ahead_m)
                closing = (v_e - v_l) if ahead_m <= 0 else (v_l - v_e)
                ttc = gap_abs / max(closing, 0.5) if gap_abs > 0 else 1e9
                state = dict(gap_m=gap_abs, d_apex_m=d_apex, ego_v=v_e, lead_v=v_l,
                             kappa=float(ring.k[i_e]), ttc=ttc,
                             ahead_m=float(ahead_m),
                             actual_off_m=actual_off,
                             actual_extra_m=actual_extra,
                             lateral_sep_m=lat_sep,
                             racing_off_m=float(racing_off[i_e]),
                             inside_side="right" if t2_sign > 0 else "left")
                dm = llm_engine.decide(t, state)
                llm_action = dm["action"]
                attack = llm_action == "overtake"
                blocked = bool(dm.get("blocked", False))
                if attack:
                    target_extra = float(dm.get("extra_inside_m", 0.0))
                elif blocked and abs(ahead_m) < 15.0:
                    # 已被安全层拦下且人在旁边：保持当前横向位置并跟车，
                    # 不要突然向竞速线回打方向（那会撞向对手）
                    target_extra = extra_cmd
                else:
                    target_extra = 0.0
                if llm_action == "hold" and abs(t - getattr(llm_engine, "_last_log", -9)) > 1.0:
                    llm_engine._last_log = t
                    print("[LLM] hold: %s" % dm.get("reason", "")[:100])

            # 对额外内线量做速率限制，避免决策切换时方向盘突变
            step_lim = 6.0 * DT
            if target_extra > extra_cmd:
                extra_cmd = min(target_extra, extra_cmd + step_lim)
            elif target_extra < extra_cmd:
                extra_cmd = max(target_extra, extra_cmd - step_lim)
            extra_peak = max(extra_peak, extra_cmd)

            # 真正把“额外内线量”接入横向指令：竞速线 + inside_sign * 攻击窗 * extra
            off_ego_cmd = racing_off + t2_sign * attack_w * extra_cmd

            if llm_engine is not None:
                if attack:
                    v_target = float(v_ego[i_e]) * float(dm.get("speed_factor", 1.05)) / 1.05
                else:
                    v_target = float(v_lead[i_l])
            else:
                v_target = float(v_ego[i_e])

            # Per-tick safety shield, independent of the 0.5s LLM cadence.
            # A decision made at gap=9m can become a 0.5s/15m collision before the
            # next LLM decision; the shield must run every tick.
            tick_shield = (llm_engine is not None and extra_cmd > 0.5 and (
                (abs(gap) < 8.0 and abs(lat_sep) < 2.2 and -5.0 < gap < 1.0) or
                (abs(gap) < 4.0 and abs(lat_sep) < 2.5 and gap < 0.5)))
            if tick_shield:
                new_v = min(v_target, v_e - 3.0)
                if new_v < v_target - 1e-6:
                    shield_count += 1
                v_target = new_v

            ego_look = dict(look_gain=0.38, look_min=5.0, look_max=12.0) \
                if extra_cmd > 0.5 else {}
            drive(ego, s_e_u, off_ego_cmd, v_target, **ego_look)
            drive(lead, s_l_u, racing_off, float(v_lead[i_l]))

            s_mod = s_e_u % ring.L
            in_t2 = t2['s0'] - 25.0 <= s_mod <= t2['s1'] + 80.0
            inside = in_t2 and t2_sign * (float(off_ego_cmd[i_e]) - float(racing_off[i_e])) > 0.5
            if (s_e_u - s_l_u) > 2.0 and passed_at is None and in_t2:
                passed_at = t
                passed_s = s_mod
            lap_frac = max(0.0, (s_e_u - s_ego0) / ring.L)
            rows.append([round(t, 2), round(s_e_u, 1), round(s_l_u, 1), round(v_e, 2),
                         round(v_l, 2), round(gap, 2), round(float(racing_off[i_e]), 2),
                         round(float(off_ego_cmd[i_e]), 2), round(actual_off, 2),
                         round(actual_off_l, 2), round(lat_sep, 2),
                         round(extra_cmd, 2), round(lap_frac, 3),
                         "INSIDE" if inside else "", llm_action])
            if step % 20 == 0:
                print("t=%5.1f lap=%5.1f%% v_e=%5.1f v_l=%5.1f gap=%6.1f s=%6.1f inside=%s" % (
                    t, 100.0 * lap_frac, v_e, v_l, gap, s_mod, inside))

            if lap_frac >= 1.0:
                lap_done = True
                lap_time = t
                break
            if t > 145.0:
                break
    finally:
        try:
            st2 = world.get_settings()
            st2.synchronous_mode = False
            world.apply_settings(st2)
            world.tick()
        except Exception:
            pass
        for cam in (bird_cam, front_cam):
            try:
                cam.stop(); cam.destroy()
            except Exception:
                pass
        for cs in coll_sensors:
            try:
                cs.stop(); cs.destroy()
            except Exception:
                pass
        if llm_engine is not None:
            try:
                llm_engine.close()
            except Exception:
                pass

    with open(os.path.join(OUT, "telemetry_race.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "s_ego", "s_lead", "v_ego", "v_lead", "gap", "off_racing", "off_ego",
                    "off_actual", "off_actual_lead", "lat_sep", "extra_cmd",
                    "lap_frac", "phase", "llm_action"])
        w.writerows(rows)
    result = {
        "map": world.get_map().name,
        "ring_len_m": round(ring.L),
        "turn2": {"s0": t2['s0'], "s1": t2['s1'], "apex": t2['s_apex'],
                  "R": round(t2['R']), "dir": t2['dir'], "inside_sign": t2_sign},
        "shield_interventions": int(shield_count),
        "line_at_apex": {
            "racing_off_m": round(float(racing_off[i_apex]), 2),
            "ego_off_m": round(float(racing_off[i_apex] + t2_sign * attack_w[i_apex] * extra_peak), 2),
            "extra_inside_m": round(extra_peak, 2),
        },
        "pass_at_turn2": passed_at is not None,
        "pass_time_s": round(passed_at, 2) if passed_at else None,
        "pass_s": round(passed_s, 1) if passed_s else None,
        "final_gap_m": round((s_e_u - s_l_u), 1),
        "max_ego_speed": round(v_peak, 2),
        "min_gap_m": round(min_gap, 2),
        "min_center_dist_m": round(min_dist, 2),
        "collisions": coll_counts,
        "llm": {"enabled": llm_engine is not None,
                "backend": llm_engine.backend_name if llm_engine else None,
                "decisions": [d for d in (llm_engine.history if llm_engine else [])]},
        "lap_completed": bool(lap_done),
        "lap_time_s": round(lap_time, 2) if lap_time else None,
        "lap_frac": round(lap_frac, 3),
        "frames": frames,
    }
    with open(os.path.join(OUT, "result_race.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("=" * 70)
    print(json.dumps(result, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
