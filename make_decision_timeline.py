#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decision timeline figure: the LLM's real-time decisions over the overtake window."""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

for p in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
    fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Liberation Sans"

rows = list(csv.DictReader(open("telemetry_race.csv")))
t = np.array([float(r["t"]) for r in rows if float(r["t"]) <= 25.0])
extra = np.array([float(r["extra_cmd"]) for r in rows if float(r["t"]) <= 25.0])
act = [r["llm_action"] for r in rows if float(r["t"]) <= 25.0]
inside = [1 if r.get("phase") == "INSIDE" else 0 for r in rows if float(r["t"]) <= 25.0]

fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=160)

# decision bands: overtake region
overtake = np.array([1 if a == "overtake" else 0 for a in act], dtype=float)
ax.fill_between(t, 0, 3.8, where=overtake > 0.5, color="#0ea5e9", alpha=0.10, zorder=0)

ax.plot(t, extra, color="#0f766e", lw=2.2, zorder=3, label="attack offset  extra_inside_m")
ax.plot(t, inside, color="#f59e0b", lw=1.4, ls="--", zorder=2, alpha=0.8,
        label="inside-line flag")

# key events
ax.axvline(11.7, color="#2563eb", lw=1.0, ls=":", alpha=0.7)
ax.text(11.7, 3.95, "LLM: overtake\n(t=11.7 s)", ha="center", va="bottom",
        fontsize=9, color="#2563eb", fontweight="bold")
ax.axvline(18.7, color="#16a34a", lw=1.0, ls=":", alpha=0.7)
ax.text(18.7, 3.95, "pass complete\n(t=18.7 s)", ha="center", va="bottom",
        fontsize=9, color="#16a34a", fontweight="bold")

ax.set_xlim(0, 25)
ax.set_ylim(0, 4.3)
ax.set_xlabel("time (s)", fontsize=10)
ax.set_ylabel("offset / flag", fontsize=10)
ax.set_title("Real-time LLM decisions during the overtake",
             fontweight="bold", fontsize=12, pad=10)
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
os.makedirs("results/media", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"results/media/decision_timeline.{ext}",
                bbox_inches="tight", facecolor="white")
print("saved decision_timeline.png/.pdf/.svg")
