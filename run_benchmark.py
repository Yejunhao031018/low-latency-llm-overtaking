#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch benchmark for the latent-tactical overtaking paper.

Runs headless CARLA episodes for E1 (gap x lead policy x seed) or E2
(baselines / ablations), with the real local LoRA server and real-time pacing.
Writes one summary per episode and an aggregate CSV.

Examples:
    ./run_py.sh run_benchmark.py --configs e1 --seeds 3 --out results/bench_e1
    ./run_py.sh run_benchmark.py --configs e2 --seeds 3 --out results/bench_e2
"""
import argparse
import csv
import json
import os
import statistics
import time

import carla

from dataset_runner import Episode
from latency_overtake import AsyncLLMClient
from llm_decision import StubBackend


class RuleClient:
    """Synchronous rule/hold client for baselines."""

    def __init__(self, kind="stub"):
        self.kind = kind
        self.stub = StubBackend()
        self.pending = None

    def request(self, t, state):
        if self.kind == "stub":
            a = self.stub.decide(state)
        else:
            a = {"action": "hold", "extra_inside_m": 0.0,
                 "speed_factor": 1.0, "reason": "hold baseline"}
        a = dict(a)
        a["valid_for_s"] = 0.8 if a.get("action") == "overtake" else 1.0
        self.pending = (t, a)
        return True

    def poll(self):
        if self.pending is None:
            return None
        t, a = self.pending
        self.pending = None
        a["request_t"] = t
        a["ready_t"] = t
        a["latency_s"] = 0.0
        a["latency_sim"] = 0.0
        a["wall_latency_s"] = 0.0
        return a

    def close(self):
        pass


def base_cfg(args, gap, lead, seed, policy):
    return dict(
        policy="latency", lead_policy=lead, seed=seed,
        start_gap_m=gap, initial_speed_mps=args.initial_speed,
        lead_speed_factor=args.lead_factor, ego_speed_factor=1.05,
        weather="ClearNoon", max_time_s=args.max_time,
        a_lat=args.a_lat, v_max=args.v_max, lead_v_max=args.lead_v_max,
        ego_v_max=args.v_max,
        attack_after=80.0, attack_ramp_out=60.0,
        decision_window=(-300.0, 150.0),
        sep_gate_m=args.sep_gate, sep_gate_bias=args.gate_bias,
        attack_dv_max=args.attack_dv, pass_clear_m=8.0,
        extra_bias_m=args.bias, off_cap_m=args.off_cap,
        corner_index=args.corner_index,
        racing_off_in=args.racing_off_in,
        corner_radius=args.corner_radius,
        stop_after_pass_s=3.0, stop_after_attack_m=120.0,
        controller=args.controller, mpc_every=args.mpc_every,
        mpc_debug=args.mpc_debug,
        bicycle_Cf=args.bike_cf, bicycle_Cr=args.bike_cf, bicycle_Iz=args.bike_iz,
        steer_slew=args.steer_slew,
        pp_fallback_err=args.pp_fallback_err,
        pp_yaw_damp=args.pp_yaw_damp,
        tick_shield=not args.no_shield, stop_on_collision=True,
        no_rendering=True, record_snapshot=False,
        view_debug=False, live_status_path="",
        realtime=args.realtime,
    )


def make_client(args, policy):
    if policy == "lora":
        return AsyncLLMClient(args.url, timeout=20.0, latency_sim=0.0,
                              latency_from_wall=True)
    return RuleClient("stub" if policy == "stub" else "hold")


def run_one(world, args, gap, lead, seed, policy, bias, shield):
    eid = "%s_gap%02.0f_%s_s%d_b%.1f_%s" % (
        policy, gap, lead, seed, bias, "sh" if shield else "nosh")
    cfg = base_cfg(args, gap, lead, seed, policy)
    cfg["extra_bias_m"] = bias
    cfg["tick_shield"] = shield
    client = make_client(args, policy)
    ep = Episode(world, cfg, eid)
    try:
        t0 = time.time()
        summary = ep.run_latency(client, decision_hz=args.hz,
                                 max_time_s=args.max_time)
        summary.update(policy=policy, lead_policy=lead, start_gap=gap,
                       seed=seed, bias=bias, shield=shield,
                       wall_time_s=round(time.time() - t0, 1))
        ep.write(os.path.join(args.out, eid))
        return summary, eid
    finally:
        ep.close()
        client.close()


def aggregate(rows, key):
    groups = {}
    for r in rows:
        k = tuple(r.get(x) for x in key)
        groups.setdefault(k, []).append(r)
    out = []
    for k, rs in sorted(groups.items()):
        n = len(rs)
        succ = sum(1 for r in rs if r["outcome"].get("pass_"))
        coll = sum(1 for r in rs if r["outcome"].get("collision"))
        seps = [r["outcome"].get("min_sep_close") for r in rs
                if r["outcome"].get("min_sep_close") is not None]
        times = [r["outcome"].get("pass_t") for r in rs
                 if r["outcome"].get("pass_t") is not None]
        lats = [r.get("latency_mean_s") for r in rs
                if r.get("latency_mean_s") is not None]
        out.append(dict(zip(key, k), n=n, success=succ / n,
                        collision=coll / n,
                        minsep_median=(statistics.median(seps) if seps else None),
                        minsep_min=(min(seps) if seps else None),
                        pass_t_median=(statistics.median(times) if times else None),
                        latency_median=(statistics.median(lats) if lats else None)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench")
    ap.add_argument("--configs", default="e1", choices=["e1", "e2"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gaps", default="15,25,35")
    ap.add_argument("--leads", default="constant,variable,defensive")
    ap.add_argument("--policy", default="lora", choices=["lora", "stub", "hold"])
    ap.add_argument("--bias", type=float, default=1.2)
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--realtime", action="store_true", default=True)
    ap.add_argument("--no-realtime", dest="realtime", action="store_false")
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--max-time", type=float, default=35.0)
    ap.add_argument("--initial-speed", type=float, default=30.0)
    ap.add_argument("--lead-factor", type=float, default=0.90)
    ap.add_argument("--sep-gate", type=float, default=0.0,
                    help="hold speed until this lateral separation is established")
    ap.add_argument("--gate-bias", type=float, default=-1.5,
                    help="speed relative to lead while the gate is closed")
    ap.add_argument("--attack-dv", type=float, default=0.0,
                    help="max speed advantage over lead during the pass")
    ap.add_argument("--controller", default="pure", choices=["pure", "mpc", "hybrid", "frenet", "bicycle"])
    ap.add_argument("--mpc-every", type=int, default=1,
                    help="solve the MPC every N ticks (1 = 20 Hz)")
    ap.add_argument("--bike-cf", type=float, default=1.2e5)
    ap.add_argument("--bike-iz", type=float, default=2200.0)
    ap.add_argument("--steer-slew", type=float, default=0.0,
                    help="steering rate limit (rad/s), 0 = disabled")
    ap.add_argument("--off-cap", type=float, default=0.0,
                    help="clip lateral offset reference to +/-off_cap (m)")
    ap.add_argument("--corner-index", type=int, default=1,
                    help="which ring corner to use as the overtake apex (0..5)")
    ap.add_argument("--corner-radius", type=float, default=188.0,
                    help="select corner by radius (T2=188); overrides index")
    ap.add_argument("--racing-off-in", type=float, default=3.3,
                    help="racing line inside offset (default 3.3m)")
    ap.add_argument("--mpc-debug", action="store_true")
    ap.add_argument("--pp-fallback-err", type=float, default=1.0,
                    help="hand back to pure-pursuit if |e| exceeds this (m)")
    ap.add_argument("--v-max", type=float, default=50.0)
    ap.add_argument("--lead-v-max", type=float, default=30.0)
    ap.add_argument("--a-lat", type=float, default=10.5)
    ap.add_argument("--pp-yaw-damp", type=float, default=0.0,
                    help="pure-pursuit yaw-rate damping gain (s)")
    ap.add_argument("--url", default="http://127.0.0.1:8765/v1/chat/completions")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    print("connected:", client.get_server_version(), "| map:", world.get_map().name)

    gaps = [float(x) for x in args.gaps.split(",")]
    leads = [x for x in args.leads.split(",") if x]
    if args.configs == "e1":
        configs = [(gap, lead, seed, args.policy, args.bias, not args.no_shield)
                   for gap in gaps for lead in leads
                   for seed in range(args.seeds)]
    else:
        # baselines / ablations at a fixed gap, all lead policies
        configs = []
        for seed in range(args.seeds):
            for lead in leads:
                configs.append((25.0, lead, seed, "lora", args.bias, not args.no_shield))
                configs.append((25.0, lead, seed, "stub", 0.0, not args.no_shield))
                configs.append((25.0, lead, seed, "hold", 0.0, not args.no_shield))
                if not args.no_shield:
                    configs.append((25.0, lead, seed, "lora", args.bias, False))

    index_path = os.path.join(args.out, "index.jsonl")
    open(index_path, "w").close()
    rows = []
    t_start = time.time()
    for k, (gap, lead, seed, policy, bias, shield) in enumerate(configs, 1):
        print("=" * 78)
        print("[%d/%d] %s gap=%.0f lead=%s seed=%d bias=%.1f shield=%s" % (
            k, len(configs), policy, gap, lead, seed, bias, shield), flush=True)
        try:
            summary, eid = run_one(world, args, gap, lead, seed, policy, bias, shield)
        except Exception as e:
            print("  FAILED:", e, flush=True)
            continue
        rows.append(summary)
        with open(index_path, "a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        o = summary["outcome"]
        print("  pass=%s t=%s minC=%s minSep=%s coll=%s shield=%s stale=%s exp=%s "
              "lat=%.3f wall=%.1fs" % (
                  o.get("pass_"), o.get("pass_t"), o.get("min_center"),
                  o.get("min_sep_close"), o.get("collision"),
                  summary.get("shield_interventions"),
                  summary.get("stale_discarded"), summary.get("expired"),
                  summary.get("latency_mean_s") or 0.0,
                  summary.get("wall_time_s") or 0.0), flush=True)

    keys = (["policy", "lead_policy", "start_gap"] if args.configs == "e1"
            else ["policy", "lead_policy", "start_gap", "bias", "shield"])
    agg = aggregate(rows, keys)
    with open(os.path.join(args.out, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    keys_csv = keys + ["n", "success", "collision", "minsep_median",
                       "minsep_min", "pass_t_median", "latency_median"]
    with open(os.path.join(args.out, "aggregate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys_csv)
        w.writeheader()
        for r in agg:
            w.writerow({k: r.get(k) for k in keys_csv})
    print("=" * 78)
    print("done %d runs in %.1f min" % (len(rows), (time.time() - t_start) / 60.0))
    for r in agg:
        extra = (" bias=%.1f shield=%s" % (r["bias"], r["shield"])
                 if "bias" in r else "")
        print("%-5s %-10s gap=%4.0f n=%2d success=%.2f coll=%.2f "
              "minSep med=%s min=%s%s" % (
                  r["policy"], r["lead_policy"], r["start_gap"], r["n"],
                  r["success"], r["collision"],
                  ("%.2f" % r["minsep_median"]) if r["minsep_median"] else "–",
                  ("%.2f" % r["minsep_min"]) if r["minsep_min"] else "–",
                  extra))
    print("aggregate:", os.path.join(args.out, "aggregate.csv"))


if __name__ == "__main__":
    main()
