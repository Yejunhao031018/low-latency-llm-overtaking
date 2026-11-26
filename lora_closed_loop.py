#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closed-loop CARLA test with the local Qwen2.5-1.5B LoRA tactical policy.

Runs in the CARLA py3.8 venv.  The LLM itself runs in finetune/lora_server.py
(py3.11 + torch cu128) and is reached over local HTTP.

    ./run_py.sh lora_closed_loop.py --runs 1 --hz 1.0 --out results/lora_closed_loop
"""
import argparse
import json
import os
import threading
import time

import carla

from dataset_runner import Episode
from latency_overtake import AsyncLLMClient


class SpectatorFollower:
    """Chase camera for live visualization in the CARLA window."""

    def __init__(self, world, actor, hz=20.0):
        self.actor = actor
        self.spectator = world.get_spectator()
        self.dt = 1.0 / max(1.0, hz)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                tf = self.actor.get_transform()
                f = tf.rotation.get_forward_vector()
                loc = carla.Location(
                    x=tf.location.x - f.x * 14.0,
                    y=tf.location.y - f.y * 14.0,
                    z=tf.location.z + 5.0)
                self.spectator.set_transform(carla.Transform(
                    loc, carla.Rotation(pitch=-14.0, yaw=tf.rotation.yaw)))
            except Exception:
                pass
            time.sleep(self.dt)


def run_one(world, out_root, name, url, hz, shield, realtime, args):
    cfg = dict(
        policy="latency", lead_policy="constant", seed=0,
        start_gap_m=args.start_gap, initial_speed_mps=args.initial_speed,
        lead_speed_factor=args.lead_factor, ego_speed_factor=1.05,
        weather="ClearNoon", max_time_s=args.max_time,
        a_lat=10.5, v_max=50.0, lead_v_max=30.0, ego_v_max=50.0,
        attack_after=80.0, attack_ramp_out=60.0,
        decision_window=(-300.0, 150.0),
        sep_gate_m=(0.0 if args.no_gate else 2.5),
        sep_gate_bias=(0.5 if args.no_gate else -1.5),
        attack_dv_max=(0.0 if args.no_gate else 6.0), pass_clear_m=8.0,
        extra_bias_m=args.extra_bias,
        stop_after_pass_s=3.0, stop_after_attack_m=120.0,
        tick_shield=shield, stop_on_collision=True,
        no_rendering=not args.view, record_snapshot=False,
        view_debug=args.view,
        live_status_path=(os.path.join(out_root, "live_status.json")
                          if args.view else ""),
        realtime=realtime,
    )
    out_dir = os.path.join(out_root, name)
    client = AsyncLLMClient(url, timeout=20.0, latency_sim=0.0,
                            latency_from_wall=True)
    ep = Episode(world, cfg, name)
    follower = SpectatorFollower(world, ep.ego) if args.view else None
    if follower is not None:
        follower.start()
    try:
        summary = ep.run_latency(client, decision_hz=hz)
        ep.write(out_dir)
        summary.update(dict(mode="lora", decision_hz=hz, shield=shield,
                            realtime=realtime, url=url))
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary, out_dir
    finally:
        if follower is not None:
            follower.stop()
        ep.close()
        client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/lora_closed_loop")
    ap.add_argument("--url", default="http://127.0.0.1:8765/v1/chat/completions")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--start-gap", type=float, default=20.0)
    ap.add_argument("--initial-speed", type=float, default=35.0)
    ap.add_argument("--lead-factor", type=float, default=0.75)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--realtime", action="store_true",
                    help="pace the sim to wall time (needed for real LLM latency)")
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--no-gate", action="store_true",
                    help="disable separation gate / speed cap (raw LLM + shield)")
    ap.add_argument("--extra-bias", type=float, default=0.0,
                    help="additional inside offset added to the LLM request (m)")
    ap.add_argument("--view", action="store_true",
                    help="render the CARLA window and follow the ego with the spectator")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    print("connected:", client.get_server_version(), "| map:", world.get_map().name)

    rows = []
    for k in range(args.runs):
        name = "lora_%.1fhz_shield%d_run%d" % (args.hz, 0 if args.no_shield else 1, k)
        print("=" * 72)
        print("RUN", name)
        summary, out_dir = run_one(world, args.out, name, args.url, args.hz,
                                   not args.no_shield, args.realtime, args)
        rows.append(summary)
        o = summary["outcome"]
        print("pass=%s t=%s s=%s lap=%s minC=%s minSep=%s coll=%s shield=%s "
              "lat_mean=%s lat_p95=%s stale=%s expired=%s env=%s req=%s resp=%s" % (
                  o.get("pass_"), o.get("pass_t"), o.get("pass_s"),
                  o.get("lap_time"), o.get("min_center"), o.get("min_sep_close"),
                  o.get("collision"), summary.get("shield_interventions"),
                  summary.get("latency_mean_s"), summary.get("latency_p95_s"),
                  summary.get("stale_discarded"), summary.get("expired"),
                  summary.get("envelope_blocks"),
                  summary.get("requests"), summary.get("responses")))
        with open(os.path.join(args.out, "index.jsonl"), "a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print("done:", args.out)


if __name__ == "__main__":
    main()
