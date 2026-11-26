#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CBF-QP / if-else / no-shield stress matrix at 5 Hz + 300 ms.

Extras: 2.0 m (marginal), 1.5 m (unsafe), 1.0 m (clearly unsafe).
For each extra: model-based QP, signed if-else shield, no shield.
"""
import json
import os

import carla

import latency_overtake as L


def main():
    out = "results/latency_stress"
    os.makedirs(out, exist_ok=True)
    client = carla.Client("127.0.0.1", int(os.environ.get("CARLA_PORT", "2000")))
    client.set_timeout(30.0)
    world = client.get_world()
    print("connected:", client.get_server_version())

    modes = [
        ("qp", True, {"shield_mode": "qp", "tick_shield": False}),
        ("ifelse", True, {"tick_shield": True}),
        ("none", False, {"tick_shield": False}),
    ]
    rows = []
    for extra in (2.0, 1.5, 1.0):
        L.TACTICAL_EXTRA = extra
        for mode, shield, cfg_extra in modes:
            name = "extra%.1f_%s" % (extra, mode)
            cfg = {"start_gap_m": 20.0}
            cfg.update(cfg_extra)
            print("=" * 70)
            print("RUN", name, "| shield", shield, "| cfg", cfg)
            summary, out_dir = L.run_one(
                world, out, name, "mock", 0.3, 5.0, shield, cfg)
            summary["stress_extra"] = extra
            summary["stress_mode"] = mode
            with open(os.path.join(out_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            rows.append(summary)
            o = summary["outcome"]
            print("pass=%s minC=%s minSep=%s coll=%s shield=%s" % (
                o.get("pass_"), o.get("min_center"), o.get("min_sep_close"),
                o.get("collision"), summary.get("shield_interventions")))
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
