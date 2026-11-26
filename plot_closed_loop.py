#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static multi-panel figure for one closed-loop episode.

    /home/yjh/venv-racingllm/bin/python plot_closed_loop.py \
        --dir results/lora_closed_loop/lora_1.0hz_shield1_run0

Reads ticks.csv / decisions.jsonl / latency_events.jsonl / summary.json and
writes closed_loop.png next to them.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=0.0):
    try:
        return float(row[key])
    except Exception:
        return default


def contiguous(flags):
    """Yield (start_idx, end_idx) runs of True."""
    start = None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            yield start, i
            start = None
    if start is not None:
        yield start, len(flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    d = args.dir
    ticks = read_csv(os.path.join(d, "ticks.csv"))
    latency = []
    lp = os.path.join(d, "latency_events.jsonl")
    if os.path.exists(lp):
        latency = [json.loads(l) for l in open(lp)]
    summary = {}
    sp = os.path.join(d, "summary.json")
    if os.path.exists(sp):
        summary = json.load(open(sp))
    o = summary.get("outcome", {})

    t = np.array([fnum(r, "t") for r in ticks])
    v_e = np.array([fnum(r, "v_ego") for r in ticks])
    v_l = np.array([fnum(r, "v_lead") for r in ticks])
    gap = np.array([fnum(r, "gap") for r in ticks])
    sep = np.array([fnum(r, "lat_sep") for r in ticks])
    extra = np.array([fnum(r, "extra_cmd") for r in ticks])
    act = [r.get("policy_action", "") for r in ticks]
    overt = np.array([a == "overtake" for a in act])

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1])
    ax_xy = fig.add_subplot(gs[0, :])
    ax_v = fig.add_subplot(gs[1, 0])
    ax_g = fig.add_subplot(gs[1, 1])
    ax_l = fig.add_subplot(gs[2, :])

    # ---- top: track-relative trajectory ----
    s_e = np.array([fnum(r, "s_ego") for r in ticks])
    s_l = np.array([fnum(r, "s_lead") for r in ticks])
    off_e = np.array([fnum(r, "off_actual") for r in ticks])
    off_l = np.array([fnum(r, "off_actual_lead") for r in ticks])
    ax_xy.plot(s_e, off_e, color="tab:blue", lw=2.2, label="ego")
    ax_xy.plot(s_l, off_l, "--", color="tab:orange", lw=1.6, label="lead")
    if o.get("pass_s"):
        ax_xy.axvline(o["pass_s"], color="k", ls=":", lw=1)
        ax_xy.text(o["pass_s"], ax_xy.get_ylim()[1], " pass", va="top", fontsize=8)
    ax_xy.set_xlabel("track s (m)")
    ax_xy.set_ylabel("lateral offset (m)")
    ax_xy.set_title("Track-relative trajectory (overtake decisions shaded green below)")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best", fontsize=8)

    # ---- speed ----
    ax_v.plot(t, v_e, color="tab:blue", lw=2, label="ego")
    ax_v.plot(t, v_l, color="tab:orange", lw=1.5, label="lead")
    for a, b in contiguous(overt):
        ax_v.axvspan(t[a], t[min(b, len(t) - 1)], color="tab:green", alpha=0.15)
    if o.get("pass_t"):
        ax_v.axvline(o["pass_t"], color="k", ls=":", lw=1)
        ax_v.text(o["pass_t"], ax_v.get_ylim()[1], " pass", va="top", fontsize=8)
    ax_v.set_xlabel("time (s)")
    ax_v.set_ylabel("speed (m/s)")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(fontsize=8)
    ax_v.set_title("Speed")

    # ---- gap + lateral separation ----
    ax_g.plot(t, gap, color="tab:red", lw=2, label="longitudinal gap")
    ax_g.plot(t, np.abs(sep), color="tab:purple", lw=1.5, label="|lateral sep|")
    ax_g.axhline(2.5, color="tab:purple", ls="--", lw=0.8, alpha=0.6)
    for a, b in contiguous(overt):
        ax_g.axvspan(t[a], t[min(b, len(t) - 1)], color="tab:green", alpha=0.15)
    ax_g.set_xlabel("time (s)")
    ax_g.set_ylabel("distance (m)")
    ax_g.grid(True, alpha=0.3)
    ax_g.legend(fontsize=8)
    ax_g.set_title("Gap and lateral separation (dashed: 2.5 m)")

    # ---- latency timeline ----
    resp = [e for e in latency if e.get("event") == "response"]
    if resp:
        rt = [e["request_t"] for e in resp]
        wl = [e.get("wall_latency_s", 0.0) for e in resp]
        age = [e.get("state_age_s", 0.0) for e in resp]
        stale = [bool(e.get("stale")) for e in resp]
        ax_l.plot(rt, wl, "o-", color="tab:blue", ms=4, lw=1, label="inference wall latency")
        ax_l.plot(rt, age, "s--", color="tab:gray", ms=3, lw=1, label="state age when consumed")
        if any(stale):
            ax_l.scatter(np.array(rt)[stale], np.array(wl)[stale], marker="x",
                         color="red", s=60, label="stale (discarded)")
        acts = [e.get("option", {}).get("action", "") for e in resp]
        for i, (x, a) in enumerate(zip(rt, acts)):
            if a == "overtake":
                ax_l.axvspan(x, x + 1.0, color="tab:green", alpha=0.10)
    ax_l.set_xlabel("request time (s)")
    ax_l.set_ylabel("seconds")
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(fontsize=8, loc="upper left")
    ax_l.set_title("Tactical latency / staleness (green spans: overtake decisions)")

    txt = ("pass=%s   min center=%.2f m   min sep=%.2f m   collision=%s\n"
           "shield=%s   latency mean=%s s   p95=%s s   stale=%s   expired=%s"
           % (o.get("pass_"), o.get("min_center") or 0, o.get("min_sep_close") or 0,
              o.get("collision"), summary.get("shield_interventions"),
              summary.get("latency_mean_s"), summary.get("latency_p95_s"),
              summary.get("stale_discarded"), summary.get("expired")))
    fig.text(0.01, 0.005, txt, fontsize=9, family="monospace")

    title = args.title or summary.get("episode_id", os.path.basename(d))
    fig.suptitle("Local Qwen2.5-1.5B LoRA closed loop - %s" % title, fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = args.out or os.path.join(d, "closed_loop.png")
    fig.savefig(out, dpi=150)
    print("saved:", out)


if __name__ == "__main__":
    main()
