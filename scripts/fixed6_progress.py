#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watch the 6-corner re-run and write a progress JSON for the dashboard."""
import glob
import json
import os
import time

OUT = "results/fixed6_status.json"
CORNERS = [45, 83, 138, 151, 188, 208]
DIR = {"45": "L(紧)", "83": "R", "138": "L", "151": "R", "188": "R(T2)", "208": "R(缓)"}


def main():
    while True:
        status = []
        for R in CORNERS:
            idx = "results/fixed6_%d/index.jsonl" % R
            n = passn = coll = 0
            seps = []
            if os.path.exists(idx):
                for l in open(idx):
                    r = json.loads(l)
                    n += 1
                    o = r.get("outcome", {})
                    if o.get("pass_"):
                        passn += 1
                    if o.get("collision"):
                        coll += 1
                    if o.get("min_sep_close") is not None:
                        seps.append(o["min_sep_close"])
            status.append(dict(R=R, dir=DIR[str(R)], n=n,
                               pass_=passn, coll=coll,
                               minsep=(round(min(seps), 2) if seps else None)))
        total = sum(s["n"] for s in status)
        done = sum(1 for s in status if s["n"] >= 2)
        st = dict(done_corners=done, total_corners=6, runs=total,
                  corners=status, updated_at=time.strftime("%H:%M:%S"))
        try:
            json.dump(st, open(OUT + ".tmp", "w"))
            os.replace(OUT + ".tmp", OUT)
        except Exception:
            pass
        time.sleep(2.0)


if __name__ == "__main__":
    main()
