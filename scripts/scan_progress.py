#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watch the stub feasibility scan (6 corners x 3 leads x 3 seeds)."""
import json
import os
import time

OUT = "results/scan_status.json"
CORNERS = [("45", "左·极紧"), ("83", "右·紧"), ("138", "左·紧"),
           ("151", "左·紧"), ("188", "右·T2"), ("208", "右·缓")]
LEADS = ["constant", "variable", "defensive"]


def main():
    while True:
        per = []
        for tag, label in CORNERS:
            p = "results/scanlora_r%s/index.jsonl" % tag
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
            per.append(dict(tag=tag, label=label, n=n, pass_=passn, coll=coll,
                            minsep=(round(min(seps), 2) if seps else None)))
        st = dict(runs=sum(p["n"] for p in per), target=54, per=per,
                  updated_at=time.strftime("%H:%M:%S"))
        try:
            json.dump(st, open(OUT + ".tmp", "w"))
            os.replace(OUT + ".tmp", OUT)
        except Exception:
            pass
        time.sleep(2.0)


if __name__ == "__main__":
    main()
