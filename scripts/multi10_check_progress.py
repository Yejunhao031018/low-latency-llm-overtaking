#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watch the 10-scenario validation of the multi model."""
import json
import os
import time

OUT = "results/zeroshot10_status.json"
SCENES = [("208", "constant"), ("208", "variable"), ("208", "defensive"),
          ("151", "constant"), ("151", "variable"), ("151", "defensive"),
          ("188", "constant"), ("188", "defensive"),
          ("83", "constant"), ("138", "variable")]


def main():
    while True:
        rows = []
        for R, L in SCENES:
            p = "results/zeroshot10_r%s_%s/index.jsonl" % (R, L)
            n = passn = coll = 0
            seps = []
            if os.path.exists(p):
                for l in open(p):
                    r = json.loads(l)
                    n += 1
                    o = r.get("outcome", {})
                    if o.get("pass_"):
                        passn += 1
                    if o.get("collision"):
                        coll += 1
                    if o.get("min_sep_close") is not None:
                        seps.append(o["min_sep_close"])
            rows.append(dict(R=R, L=L, n=n, pass_=passn, coll=coll,
                             minsep=(round(min(seps), 2) if seps else None)))
        st = dict(runs=sum(r["n"] for r in rows), target=100, rows=rows,
                  updated_at=time.strftime("%H:%M:%S"))
        try:
            json.dump(st, open(OUT + ".tmp", "w"))
            os.replace(OUT + ".tmp", OUT)
        except Exception:
            pass
        time.sleep(2.0)


if __name__ == "__main__":
    main()
