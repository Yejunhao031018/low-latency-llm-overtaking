#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the core "low-latency" superiority figure (English labels)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

models = [
    ("Qwen2.5-0.5B", 75.7, 0.86),
    ("Qwen2.5-1.5B", 87.8, 1.08),
    ("Qwen2.5-1.5B-short\n(ours)", 86.1, 0.59),
    ("Qwen2.5-3B\n(4-bit)", 71.3, 2.02),
    ("Qwen2.5-7B\n(4-bit zero-shot)", 69.6, 1.41),
]
names = [m[0] for m in models]
acc = [m[1] for m in models]
lat = [m[2] for m in models]
colors = ["#8a9bb8", "#8a9bb8", "#0ea5e9", "#8a9bb8", "#8a9bb8"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=130)

bars = ax1.bar(names, lat, color=colors, edgecolor="white", linewidth=0.5)
for b, v in zip(bars, lat):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.2f}s",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_ylabel("Decision latency (s)")
ax1.set_title("(a) Latency (lower is better)", fontweight="bold")
ax1.set_ylim(0, 2.4)
ax1.axhline(0.59, color="#0ea5e9", linestyle="--", alpha=0.5)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", alpha=0.2)
ax1.tick_params(axis="x", labelsize=8)

for (n, a, l), c in zip(models, colors):
    ax2.scatter(l, a, s=130, color=c, zorder=3, edgecolor="white", linewidth=0.6)
    ax2.annotate(n.replace("\n", " "), (l, a), textcoords="offset points",
                 xytext=(8, 6), fontsize=8)
ax2.set_xlabel("Decision latency (s)")
ax2.set_ylabel("Decision accuracy (%)")
ax2.set_title("(b) Accuracy–latency Pareto (lower-left better)", fontweight="bold")
ax2.set_xlim(0.4, 2.3)
ax2.set_ylim(65, 92)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(alpha=0.2)

fig.suptitle("Low-latency LLM tactical decisions: the 1.5B-short model is the\n"
             "Pareto-optimal point (fast and accurate)",
             fontweight="bold", fontsize=12.5, y=1.04)
fig.tight_layout()
os.makedirs("results/media", exist_ok=True)
fig.savefig("results/media/latency_superiority.png", bbox_inches="tight",
            facecolor="white")
fig.savefig("results/media/latency_superiority.pdf", bbox_inches="tight",
            facecolor="white")
fig.savefig("results/media/latency_superiority.svg", bbox_inches="tight",
            facecolor="white")
print("saved latency_superiority.png/.pdf/.svg")
