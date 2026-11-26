#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate latency_overtake results into a markdown table + CSV."""
import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/latency_overtake")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = os.path.join(args.dir, "index.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    out = args.out or os.path.join(args.dir, "summary.md")
    cols = ["run", "mode", "delay_ms", "hz", "shield", "pass", "pass_t",
            "min_center", "min_sep", "collision", "shield_int", "expired",
            "stale", "options", "lat_mean"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    csv_rows = []
    for r in rows:
        o = r.get("outcome", {})
        delay = r.get("delay_s", 0.0)
        row = [
            r.get("episode_id"), r.get("mode"),
            int(round(float(delay) * 1000)), r.get("decision_hz"),
            r.get("shield"), o.get("pass_"), o.get("pass_t"),
            o.get("min_center"), o.get("min_sep_close"),
            o.get("collision"), r.get("shield_interventions"),
            r.get("expired"), r.get("stale_discarded"),
            r.get("options_used"), r.get("latency_mean_s"),
        ]
        lines.append("| " + " | ".join("" if x is None else str(x) for x in row) + " |")
        csv_rows.append(dict(zip(cols, row)))
    text = "\n".join(lines) + "\n"
    with open(out, "w") as f:
        f.write(text)
    with open(os.path.join(args.dir, "summary_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(csv_rows)
    print(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
