#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distance-traveled-during-decision figure: a top-down road with car positions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, FancyArrow

for p in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
    fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Liberation Sans"

fig, ax = plt.subplots(figsize=(9.6, 3.6), dpi=160)
ax.set_xlim(-4, 124)
ax.set_ylim(-3, 8)
ax.axis("off")

# road
ax.add_patch(Rectangle((-2, -0.9), 122, 2.4, facecolor="#e2e8f0", edgecolor="none", zorder=0))
ax.plot([-2, 120], [0.3, 0.3], color="#94a3b8", lw=2, ls=(0, (6, 5)), zorder=1)

# start car (t=0)
ax.add_patch(Rectangle((-0.6, -0.45), 1.2, 0.9, facecolor="#0f172a", edgecolor="none", zorder=2))
ax.text(0, -1.5, "t = 0\nsituation observed", ha="center", va="top", fontsize=8, color="#0f172a", fontweight="bold")

# car positions at decision arrival
cars = [
    (17.7, "0.59 s\n(ours, 1.5B-short)", "#0ea5e9", True),
    (42.3, "1.41 s\n(7B)", "#64748b", False),
    (60.6, "2.02 s\n(3B)", "#64748b", False),
    (105.0, "3.5 s\n(cloud GPT-4)", "#b91c1c", False),
]
for x, label, color, ours in cars:
    ax.add_patch(Rectangle((x - 0.55, -0.42), 1.1, 0.84, facecolor=color,
                           edgecolor="white", lw=1.2, zorder=2))
    ax.text(x, -1.5, "%.0f m" % x, ha="center", va="top", fontsize=9,
            color=color, fontweight="bold")
    ax.text(x, 1.1, label, ha="center", va="bottom", fontsize=8.2,
            color=color, fontweight="bold" if ours else "normal", zorder=3)

# football field reference
ax.annotate("", xy=(105, 3.0), xytext=(0, 3.0),
            arrowprops=dict(arrowstyle="<->", color="#94a3b8", lw=1.0))
ax.text(52.5, 3.5, "105 m ≈ a football field", ha="center", va="bottom",
        fontsize=8.5, color="#475569", style="italic")

# scale
ax.text(17.7, -3.0, "", fontsize=1)
ax.set_title("Distance traveled before the decision arrives  ·  at 108 km/h (30 m/s)",
             fontweight="bold", fontsize=12.5, pad=14)

fig.tight_layout()
import os
os.makedirs("results/media", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"results/media/distance_traveled.{ext}", bbox_inches="tight", facecolor="white")
print("saved distance_traveled.png/.pdf/.svg")
