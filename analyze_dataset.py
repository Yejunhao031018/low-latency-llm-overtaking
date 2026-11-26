#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate episode summaries / labels into a small table for the paper.

    python analyze_dataset.py --dataset dataset/pilot
    python analyze_dataset.py --dataset dataset/pilot --csv dataset/pilot/summary.csv
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/pilot")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.dataset, "*"))):
        sp = os.path.join(d, "summary.json")
        if not os.path.isfile(sp):
            continue
        with open(sp) as f:
            s = json.load(f)
        c = s.get("config", {})
        o = s.get("outcome", {})
        rows.append(dict(
            episode=os.path.basename(d),
            policy=c.get("policy"), lead=c.get("lead_policy"),
            gap=c.get("start_gap_m"), lead_f=c.get("lead_speed_factor"),
            ego_f=c.get("ego_speed_factor"),
            pass_=o.get("pass_"), pass_t=o.get("pass_t"), pass_s=o.get("pass_s"),
            lap=o.get("lap_time"), min_center=o.get("min_center"),
            min_sep=o.get("min_sep_close"),
            collision=o.get("collision"), collisions=o.get("collisions"),
            shield=s.get("shield_interventions"),
        ))

    print("%-38s %-7s %-9s %5s %6s %6s %7s %7s %5s" % (
        "episode", "lead", "pass", "t", "s", "lap", "minC", "minSep", "coll"))
    for r in rows:
        print("%-38s %-7s %-9s %5s %6s %6s %7s %7s %5s" % (
            r["episode"][:38], r["lead"], r["pass_"], r["pass_t"], r["pass_s"],
            r["lap"], r["min_center"], r["min_sep"], r["collision"]))

    if rows:
        n = len(rows)
        ok = sum(1 for r in rows if r["pass_"] and not r["collision"])
        print("\nepisodes=%d success=%d (%.0f%%) collisions=%d" % (
            n, ok, 100.0 * ok / n, sum(1 for r in rows if r["collision"])))
        for key in ("policy", "lead"):
            groups = defaultdict(list)
            for r in rows:
                groups[r[key]].append(r)
            for k, v in groups.items():
                print("  %s=%s: n=%d success=%d" % (
                    key, k, len(v), sum(1 for r in v if r["pass_"] and not r["collision"])))

    label_path = os.path.join(args.dataset, "labels.jsonl")
    if os.path.exists(label_path):
        acts = Counter()
        n = 0
        with open(label_path) as f:
            for line in f:
                r = json.loads(line)
                acts[r["best"]["name"]] += 1
                n += 1
        print("\nlabels=%d best actions=%s" % (n, dict(acts)))

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("csv:", args.csv)


if __name__ == "__main__":
    main()
