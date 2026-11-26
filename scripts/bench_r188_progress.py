#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watch the 20-seed x 3-lead right-turn benchmark."""
import json
import os
import time

OUT = "results/bench_r188_status.json"
IDX = "results/bench_r188_20s/index.jsonl"
LEADS = ["constant", "variable", "defensive"]


def main():
    while True:
        rows = []
        if os.path.exists(IDX):
            for l in open(IDX):
                rows.append(json.loads(l))
        per = {}
        for lead in LEADS:
            g = [r for r in rows if r.get("lead_policy") == lead]
            n = len(g)
            npass = sum(1 for r in g if r["outcome"].get("pass_"))
            ncoll = sum(1 for r in g if r["outcome"].get("collision"))
            seps = [r["outcome"].get("min_sep_close") for r in g
                    if r["outcome"].get("min_sep_close") is not None]
            per[lead] = dict(n=n, pass_=npass, coll=ncoll,
                             minsep=(round(min(seps), 2) if seps else None))
        total = sum(v["n"] for v in per.values())
        st = dict(total=total, target=60, per=per,
                  updated_at=time.strftime("%H:%M:%S"))
        try:
            json.dump(st, open(OUT + ".tmp", "w"))
            os.replace(OUT + ".tmp", OUT)
        except Exception:
            pass
        time.sleep(2.0)


if __name__ == "__main__":
    main()
