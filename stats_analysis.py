#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statistics for the E1 benchmark: Wilson CI for rates, bootstrap CI for
minSep, Mann-Whitney U between methods.

    /home/yjh/carla-venv/bin/python stats_analysis.py \
        --dirs results/bench20_pure,results/bench20_gate,results/bench20_stub,results/bench20_opt \
        --labels pure,gate,stub,opt-pure --out results/bench20_stats.md
"""
import argparse
import json
import math
import os
import statistics

import numpy as np


def load(d):
    p = os.path.join(d, "index.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(vals, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    if not vals:
        return (None, None, None)
    meds = []
    a = np.asarray(vals, dtype=float)
    for _ in range(n_boot):
        s = rng.choice(a, size=len(a), replace=True)
        meds.append(float(np.median(s)))
    meds = np.sort(meds)
    return (float(np.median(a)), float(meds[int(0.025 * len(meds))]),
            float(meds[int(0.975 * len(meds))]))


def mannwhitney(a, b):
    try:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return u, float(p)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results/bench_stats.md")
    args = ap.parse_args()
    dirs = args.dirs.split(",")
    labels = args.labels.split(",")
    assert len(dirs) == len(labels)

    leads = ["constant", "variable", "defensive"]
    data = []
    for d, lab in zip(dirs, labels):
        for r in load(d):
            r["_label"] = lab
            data.append(r)

    lines = ["# E1 statistics", ""]
    lines.append("| method | lead | N | pass rate [95% CI] | collision [95% CI] "
                "| minSep med [boot CI] | minSep min |")
    lines.append("|---|---|---|---|---|---|---|---|")
    groups = {}
    for lab in labels:
        for lead in leads:
            g = [r for r in data if r["_label"] == lab and r.get("lead_policy") == lead]
            n = len(g)
            npass = sum(1 for r in g if r["outcome"].get("pass_"))
            ncoll = sum(1 for r in g if r["outcome"].get("collision"))
            seps = [r["outcome"]["min_sep_close"] for r in g
                    if r["outcome"].get("min_sep_close") is not None]
            pc = wilson(npass, n)
            cc = wilson(ncoll, n)
            med, lo, hi = bootstrap_ci(seps)
            mn = min(seps) if seps else None
            groups[(lab, lead)] = dict(n=n, pass_=npass, coll=ncoll, seps=seps,
                                       med=med, mn=mn)
            lines.append("| %s | %s | %d | %.2f [%.2f, %.2f] | %.2f [%.2f, %.2f] "
                         "| %s | %s |" % (
                             lab, lead, n, npass / max(n, 1), pc[0], pc[1],
                             ncoll / max(n, 1), cc[0], cc[1],
                             ("%.2f [%.2f, %.2f]" % (med, lo, hi)) if med else "–",
                             ("%.2f" % mn) if mn is not None else "–"))

    # Mann-Whitney: opt-pure vs pure on minSep, per lead
    lines += ["", "## minSep 显著性（Mann-Whitney U，opt-pure vs pure）", ""]
    for lead in leads:
        a = groups.get(("opt-pure", lead), {}).get("seps", [])
        b = groups.get(("pure", lead), {}).get("seps", [])
        if a and b:
            u, p = mannwhitney(a, b)
            lines.append("- %s: U=%.1f p=%.4f%s" % (
                lead, u or 0, p or 1.0,
                " (significant)" if p is not None and p < 0.05 else ""))

    text = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
