#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watch left/right regression check and write progress JSON."""
import json
import os
import time

OUT = "results/lr_check_status.json"
CHECKS = [("138", "左弯 R=138", "results/lr_check_138"),
          ("188", "右弯 R=188", "results/lr_check_188")]


def main():
    while True:
        rows = []
        for tag, label, idx in CHECKS:
            p = idx + "/index.jsonl"
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
            rows.append(dict(tag=tag, label=label, n=n, pass_=passn, coll=coll,
                             minsep=(round(min(seps), 2) if seps else None)))
        st = dict(runs=sum(r["n"] for r in rows), done=sum(1 for r in rows if r["n"] >= 2),
                  total=2, rows=rows, updated_at=time.strftime("%H:%M:%S"))
        try:
            json.dump(st, open(OUT + ".tmp", "w"))
            os.replace(OUT + ".tmp", OUT)
        except Exception:
            pass
        time.sleep(2.0)


if __name__ == "__main__":
    main()
