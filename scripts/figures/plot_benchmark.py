#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot a batch benchmark from run_benchmark.py index.jsonl files.

    /home/yjh/venv-racingllm/bin/python plot_benchmark.py \
        --dirs results/bench_e1_pilot,results/bench_e1_stub \
        --labels LoRA,stub --out results/bench_e1_pilot/summary.png
"""
import argparse
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(dir_path):
    path = os.path.join(dir_path, "index.jsonl")
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        o = r.get("outcome", {})
        r["_pass"] = bool(o.get("pass_"))
        r["_coll"] = bool(o.get("collision"))
        r["_sep"] = o.get("min_sep_close")
        r["_tpass"] = o.get("pass_t")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--out", default="results/benchmark_summary.png")
    args = ap.parse_args()
    dirs = [x for x in args.dirs.split(",") if x]
    labels = ([x for x in args.labels.split(",")] if args.labels
              else [os.path.basename(x.rstrip("/")) for x in dirs])
    assert len(labels) == len(dirs)

    data = []
    for d, lab in zip(dirs, labels):
        for r in load(d):
            r["_label"] = lab
            data.append(r)
    leads = sorted({r.get("lead_policy", "?") for r in data})
    groups = [(lab, lead) for lab in labels for lead in leads]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_ok, ax_sep, ax_t, ax_n = axes.ravel()
    x = np.arange(len(groups))
    w = 0.38
    succ = []; coll = []; ns = []
    for g in groups:
        rs = [r for r in data if r["_label"] == g[0] and r.get("lead_policy") == g[1]]
        ns.append(len(rs))
        succ.append(np.mean([r["_pass"] for r in rs]) if rs else 0.0)
        coll.append(np.mean([r["_coll"] for r in rs]) if rs else 0.0)
    ax_ok.bar(x - w / 2, succ, w, label="pass rate", color="#22c98a")
    ax_ok.bar(x + w / 2, coll, w, label="collision rate", color="#ff6b6b")
    ax_ok.set_ylim(0, 1.05)
    ax_ok.set_ylabel("rate")
    ax_ok.set_title("Pass / collision rate")
    ax_ok.legend(fontsize=8)
    ax_ok.grid(True, axis="y", alpha=0.3)

    sep_data = []
    for g in groups:
        vals = [r["_sep"] for r in data
                if r["_label"] == g[0] and r.get("lead_policy") == g[1]
                and r["_sep"] is not None]
        sep_data.append(vals if vals else [np.nan])
    bp = ax_sep.boxplot(sep_data, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black"))
    for patch in bp["boxes"]:
        patch.set_facecolor("#6ea8fe")
        patch.set_alpha(0.75)
    for i, vals in enumerate(sep_data):
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            ax_sep.scatter([i + 1] * len(vals), vals, color="#16305a", s=16, zorder=3)
    ax_sep.axhline(2.2, color="red", ls="--", lw=1, label="target 2.2 m")
    ax_sep.set_ylabel("min lateral separation (m)")
    ax_sep.set_title("Minimum lateral separation")
    ax_sep.legend(fontsize=8)
    ax_sep.grid(True, axis="y", alpha=0.3)

    tp_data = []
    for g in groups:
        vals = [r["_tpass"] for r in data
                if r["_label"] == g[0] and r.get("lead_policy") == g[1]
                and r["_tpass"] is not None]
        tp_data.append(vals if vals else [np.nan])
    bp2 = ax_t.boxplot(tp_data, patch_artist=True, showfliers=False,
                       medianprops=dict(color="black"))
    for patch in bp2["boxes"]:
        patch.set_facecolor("#ffb454")
        patch.set_alpha(0.75)
    ax_t.set_ylabel("pass time (s)")
    ax_t.set_title("Pass time")
    ax_t.grid(True, axis="y", alpha=0.3)

    ax_n.bar(x, ns, color="#8b9dff")
    ax_n.set_ylabel("N")
    ax_n.set_title("Samples per group")
    ax_n.grid(True, axis="y", alpha=0.3)

    names = ["%s\n%s" % g for g in groups]
    for a in (ax_ok, ax_sep, ax_t, ax_n):
        a.set_xticks(x)
        a.set_xticklabels(names, fontsize=7)
    fig.suptitle("Closed-loop benchmark: local LoRA tactical overtaking")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150)
    print("saved:", args.out)

    print("\n%-8s %-10s %4s %7s %7s %10s %10s" % (
        "policy", "lead", "N", "pass", "coll", "minSep med", "minSep min"))
    for g in groups:
        rs = [r for r in data if r["_label"] == g[0] and r.get("lead_policy") == g[1]]
        seps = [r["_sep"] for r in rs if r["_sep"] is not None]
        print("%-8s %-10s %4d %7.2f %7.2f %10s %10s" % (
            g[0], g[1], len(rs),
            np.mean([r["_pass"] for r in rs]) if rs else 0,
            np.mean([r["_coll"] for r in rs]) if rs else 0,
            ("%.2f" % statistics.median(seps)) if seps else "–",
            ("%.2f" % min(seps)) if seps else "–"))


if __name__ == "__main__":
    main()
