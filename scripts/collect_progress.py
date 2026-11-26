#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watch multi10 data collection progress."""
import glob
import json
import os
import time

OUT = "results/collect_status.json"
ROOT = "dataset/multi10"


def main():
    while True:
        eps = [d for d in glob.glob(ROOT + "/ep*") if os.path.isdir(d)]
        labels = 0
        p = ROOT + "/labels.jsonl"
        if os.path.exists(p):
            labels = sum(1 for _ in open(p))
        # per-scenario breakdown
        by_scene = {}
        for d in eps:
            s = os.path.join(d, "summary.json")
            if not os.path.exists(s):
                continue
            cfg = json.load(open(s)).get("config", {})
            key = "R=%s %s" % (cfg.get("corner_radius"), cfg.get("lead_policy"))
            by_scene[key] = by_scene.get(key, 0) + 1
        st = dict(episodes=len(eps), target=20, labels=labels,
                  scenes=by_scene, updated_at=time.strftime("%H:%M:%S"))
        try:
            json.dump(st, open(OUT + ".tmp", "w"))
            os.replace(OUT + ".tmp", OUT)
        except Exception:
            pass
        time.sleep(3.0)


if __name__ == "__main__":
    main()
