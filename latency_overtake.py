#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latency-aware tactical overtaking experiment.

Compares:
  - a fast "student" rule at 20 Hz (zero latency)
  - an asynchronous "mock LLM" at 5 Hz with injected latency 0/100/300/500 ms
  - action chunks with an explicit validity horizon
  - per-tick safety shield on/off

The LLM here is a local OpenAI-compatible mock with a real HTTP delay; it can be
swapped for a real model by pointing the client at vLLM/DeepSeek later.  The
paper question is about the architecture (latency, staleness, action chunks,
shield), not about this mock's language ability.

Run:
    python latency_overtake.py --out results/latency_overtake
Requires CARLA Town04 on port 2000.
"""
import argparse
import http.server
import json
import math
import os
import threading
import time
import urllib.request
from collections import OrderedDict

import carla

from dataset_runner import Episode

# Extra inside offset used by the mock tactical policy. 3.5 m is the safe
# nominal line; set TACTICAL_EXTRA=2.0 to stress the safety shield.
TACTICAL_EXTRA = float(os.environ.get("TACTICAL_EXTRA", "3.5"))


def tactical_rule(state):
    d = float(state.get("d_tactical_m", state.get("d_apex_m", 9999.0)))
    ahead = float(state.get("ahead_m", -999.0))
    gap = float(state.get("gap_m", 999.0))
    attack = (-220.0 <= d <= 150.0) and (ahead < 10.0) and (gap < 60.0)
    if attack:
        return dict(action="overtake", extra_inside_m=TACTICAL_EXTRA, speed_factor=1.05,
                    valid_for_s=0.8,
                    reason="gap %.0f d_apex %.0f: inside %.1f" % (gap, d, TACTICAL_EXTRA))
    return dict(action="hold", extra_inside_m=0.0, speed_factor=1.0,
                valid_for_s=0.4, reason="hold")


class MockLLMServer:
    def __init__(self, port, delay_s):
        self.port = port
        self.delay_s = float(delay_s)

        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                try:
                    content = body["messages"][-1]["content"]
                    state = json.loads(content)
                except Exception:
                    state = {}
                out = tactical_rule(state)
                time.sleep(outer.delay_s)
                raw = json.dumps({"choices": [{"message": {"content": json.dumps(out)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass


class FastClient:
    def __init__(self):
        self.pending = None

    def request(self, t, state):
        self.pending = (t, tactical_rule(state))
        return True

    def poll(self):
        if self.pending is None:
            return None
        t, opt = self.pending
        self.pending = None
        opt = dict(opt)
        opt["request_t"] = t
        opt["ready_t"] = t
        opt["latency_s"] = 0.0
        opt["latency_sim"] = 0.0
        opt["wall_latency_s"] = 0.0
        return opt

    def close(self):
        pass


class AsyncLLMClient:
    def __init__(self, url, timeout=10.0, latency_sim=0.0, latency_from_wall=False):
        self.url = url
        self.timeout = timeout
        self.latency_sim = float(latency_sim)
        # Real model: use the measured wall latency as the simulated latency.
        # With cfg["realtime"]=True one wall second is one simulated second.
        self.latency_from_wall = bool(latency_from_wall)
        self.busy = False
        self.response = None
        self.request_t = 0.0
        self._lock = threading.Lock()

    def request(self, t, state):
        with self._lock:
            if self.busy:
                return False
            self.busy = True
            self.request_t = t
        th = threading.Thread(target=self._work, args=(state,), daemon=True)
        th.start()
        return True

    def _work(self, state):
        t0 = time.monotonic()
        obj = None
        try:
            body = json.dumps({
                "model": "mock",
                "temperature": 0.0,
                "max_tokens": 120,
                "messages": [{"role": "user", "content": json.dumps(state)}],
            }).encode()
            req = urllib.request.Request(self.url, data=body, headers={
                "Content-Type": "application/json", "Authorization": "Bearer dummy"})
            # localhost must bypass Clash/HTTP proxies
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=self.timeout) as r:
                resp = json.loads(r.read().decode())
            text = resp["choices"][0]["message"]["content"]
            obj = json.loads(text)
        except Exception as e:
            obj = dict(action="hold", extra_inside_m=0.0, speed_factor=1.0,
                       valid_for_s=0.3, reason="client error: %s" % str(e)[:80])
        obj["request_t"] = self.request_t
        obj["latency_s"] = time.monotonic() - t0
        obj["latency_sim"] = (obj["latency_s"] if self.latency_from_wall
                              else self.latency_sim)
        obj["wall_latency_s"] = obj["latency_s"]
        with self._lock:
            self.response = obj
            self.busy = False

    def poll(self):
        with self._lock:
            r = self.response
            self.response = None
            return r

    def close(self):
        pass


def run_one(world, out_root, name, mode, delay, hz, shield, cfg_extra):
    cfg = dict(
        policy="latency", lead_policy="constant", seed=0,
        start_gap_m=20.0, initial_speed_mps=35.0,
        lead_speed_factor=0.75, ego_speed_factor=1.05,
        weather="ClearNoon", max_time_s=30.0,
        a_lat=10.5, v_max=50.0, lead_v_max=38.0, ego_v_max=50.0,
        attack_s=380.0,
        attack_before=300.0, attack_hold=180.0,
        attack_after=80.0, attack_ramp_out=60.0,
        stop_after_pass_s=3.0, stop_after_attack_m=120.0,
        tick_shield=shield, stop_on_collision=True,
        no_rendering=True, record_snapshot=False,
    )
    cfg.update(cfg_extra or {})
    eid = name
    out_dir = os.path.join(out_root, eid)
    server = None
    if mode == "mock":
        port = 8800 + abs(hash(name)) % 200
        server = MockLLMServer(port, 0.0)   # wall delay 0; latency is injected in sim time
        server.start()
        client = AsyncLLMClient("http://127.0.0.1:%d/v1/chat/completions" % port,
                                latency_sim=delay)
    else:
        client = FastClient()

    ep = Episode(world, cfg, eid)
    try:
        summary = ep.run_latency(client, decision_hz=hz)
        ep.write(out_dir)
        summary.update(dict(mode=mode, delay_s=delay, decision_hz=hz, shield=shield,
                            tactical_extra=TACTICAL_EXTRA))
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary, out_dir
    finally:
        ep.close()
        client.close()
        if server is not None:
            server.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/latency_overtake")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    print("connected:", client.get_server_version(), "| map:", world.get_map().name)

    runs = [
        ("student_20hz", "student", 0.0, 20.0, True),
        ("mock_5hz_L0", "mock", 0.0, 5.0, True),
        ("mock_5hz_L100", "mock", 0.1, 5.0, True),
        ("mock_5hz_L300", "mock", 0.3, 5.0, True),
        ("mock_5hz_L500", "mock", 0.5, 5.0, True),
        ("mock_5hz_L800", "mock", 0.8, 5.0, True),
        ("mock_5hz_L300_noshield", "mock", 0.3, 5.0, False),
    ]
    rows = []
    index_path = os.path.join(args.out, "index.jsonl")
    open(index_path, "w").close()
    for name, mode, delay, hz, shield in runs:
        print("=" * 72)
        print("RUN", name, "| mode", mode, "| delay", delay, "| hz", hz, "| shield", shield)
        summary, out_dir = run_one(world, args.out, name, mode, delay, hz, shield,
                                   cfg_extra={"start_gap_m": 20.0})
        rows.append(summary)
        with open(index_path, "a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        o = summary["outcome"]
        print("pass=%s t=%s s=%s lap=%s minC=%s minSep=%s coll=%s shield=%s lat=%s" % (
            o.get("pass_"), o.get("pass_t"), o.get("pass_s"), o.get("lap_time"),
            o.get("min_center"), o.get("min_sep_close"), o.get("collision"),
            summary.get("shield_interventions"), summary.get("latency_mean_s")))

    # CSV
    import csv
    keys = ["episode_id", "mode", "delay_s", "decision_hz", "shield",
            "pass", "pass_t", "pass_s", "lap", "min_center", "min_sep",
            "collision", "collisions", "shield_interventions",
            "latency_mean_s", "latency_p95_s", "expired", "stale_discarded",
            "requests", "responses", "options_used"]
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            o = r["outcome"]
            w.writerow([r.get("episode_id"), r.get("mode"), r.get("delay_s"),
                        r.get("decision_hz"), r.get("shield"),
                        o.get("pass_"), o.get("pass_t"), o.get("pass_s"),
                        o.get("lap_time"), o.get("min_center"), o.get("min_sep_close"),
                        o.get("collision"), json.dumps(o.get("collisions")),
                        r.get("shield_interventions"), r.get("latency_mean_s"),
                        r.get("latency_p95_s"), r.get("expired"), r.get("stale_discarded"),
                        r.get("requests"), r.get("responses"), r.get("options_used")])

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [r["episode_id"] for r in rows]
        x = range(len(rows))
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].bar(x, [1 if r["outcome"].get("pass_") else 0 for r in rows], color="tab:green")
        ax[0].set_ylabel("pass (1/0)")
        ax[1].bar(x, [r["outcome"].get("min_center") or 0 for r in rows], color="tab:blue")
        ax[1].axhline(2.2, ls="--", c="r", label="min sep target")
        ax[1].set_ylabel("min center (m)")
        ax[2].bar(x, [r.get("expired", 0) for r in rows], color="tab:orange")
        ax[2].set_ylabel("expired options")
        for a in ax:
            a.set_xticks(list(x))
            a.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            a.grid(True, axis="y", alpha=0.3)
        fig.suptitle("Latency-aware tactical overtaking (CARLA Town04, high speed)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "latency_tradeoff.png"), dpi=130)
        print("plot:", os.path.join(args.out, "latency_tradeoff.png"))
    except Exception as e:
        print("plot skipped:", e)

    print("done:", args.out)


if __name__ == "__main__":
    main()
