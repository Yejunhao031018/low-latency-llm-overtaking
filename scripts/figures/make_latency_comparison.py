#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latency comparison figure: horizontal bars + distance-traveled annotation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for p in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
    fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Liberation Sans"

# (label, latency_s, is_ours, note)
rows = [
    ("Cloud LLM (GPT-4,\nLanguageMPC-style)", 3.5, False, "~105 m  (estimated)"),
    ("Qwen2.5-3B\n(4-bit)", 2.02, False, "61 m"),
    ("Qwen2.5-7B\n(4-bit, zero-shot)", 1.41, False, "42 m"),
    ("Qwen2.5-1.5B", 1.08, False, "32 m"),
    ("Qwen2.5-0.5B", 0.86, False, "26 m"),
    ("Qwen2.5-1.5B-short\n(ours)", 0.59, True, "18 m"),
]
rows = rows[::-1]  # ascending for a clean bar chart

labels = [r[0] for r in rows]
vals   = [r[1] for r in rows]
notes  = [r[3] for r in rows]
colors = ["#0ea5e9" if r[2] else "#94a3b8" for r in rows]

fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=160)
y = range(len(rows))
bars = ax.barh(y, vals, color=colors, edgecolor="white", height=0.62)

for yi, v, note, col in zip(y, vals, notes, colors):
    ax.text(v + 0.06, yi, f"{v:.2f} s   ·   {note}",
            va="center", ha="left", fontsize=9.5,
            fontweight="bold" if col == "#0ea5e9" else "normal",
            color="#0f172a" if col == "#0ea5e9" else "#475569")

ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0, 4.6)
ax.set_xlabel("Decision latency (s)   —   lower is better", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.2)
ax.set_title("Decision latency and distance traveled at 30 m/s",
             fontweight="bold", fontsize=12, pad=12)

# highlight our bar region
ax.axvspan(0, 0.7, color="#0ea5e9", alpha=0.06, zorder=0)
fig.tight_layout()
import os
os.makedirs("results/media", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"results/media/latency_comparison.{ext}",
                bbox_inches="tight", facecolor="white")
print("saved latency_comparison.png/.pdf/.svg")
