#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-quality architecture figure (IEEE-style, Liberation Sans)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import font_manager as fm

# ---- fonts ----
for path in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
    fm.fontManager.addfont(path)
plt.rcParams["font.family"] = "Liberation Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ---- palette (muted, professional) ----
C_ENV      = "#64748b"   # slate
C_OBS      = "#475569"
C_LLM      = "#2563eb"   # blue
C_LLM_DARK = "#1d4ed8"
C_CTRL     = "#0d9488"   # teal
C_CTRL_DARK= "#0f766e"
C_ACT      = "#334155"
C_LOOP     = "#b45309"   # amber
GRAY_TXT   = "#334155"
LIGHT_TXT  = "#64748b"

fig, ax = plt.subplots(figsize=(9.6, 7.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def rbox(x, y, w, h, fc, ec, lw=1.1, r=2.4):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p); return p

def title(x, y, s, color, size=10.5, bold=True, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal",
            color=color, ha=ha, va=va, zorder=4)

def body(x, y, s, color=GRAY_TXT, size=8.2, ha="center", va="center", italic=False):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=4,
            style="italic" if italic else "normal")

def badge(x, y, s, fc, tc="white", size=7.2):
    ax.text(x, y, s, fontsize=size, color=tc, ha="center", va="center",
            fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec="none"))

def arrow(x1, y1, x2, y2, color=GRAY_TXT, lw=1.5, style="-|>", ms=14):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                        linewidth=lw, color=color, zorder=1)
    ax.add_patch(a); return a

# ================= outer environment =================
rbox(6, 3, 88, 94, "#f8fafc", "#cbd5e1", lw=1.0, r=3)
title(10.5, 94.2, "CARLA Environment (Town04)", "#475569", size=9.5, ha="left")

# ================= perception / observation =================
rbox(22, 82, 56, 10, "#eef2f7", C_OBS)
title(50, 89.2, "Observation", C_OBS, size=10.5)
body(50, 84.8, "ego_v · lead_v · gap_m · d_apex_m · kappa · ttc · lateral_sep_m · inside_side",
     LIGHT_TXT, size=7.4)

arrow(50, 82, 50, 76.6, color=C_OBS)
badge(56.5, 79.3, "1 Hz", C_OBS)

# ================= tactical layer (LLM) =================
rbox(18, 58, 64, 18.6, "#eff6ff", C_LLM, lw=1.4)
# accent bar
rbox(18, 58, 1.6, 18.6, C_LLM, C_LLM, lw=0)
title(50, 73.4, "Tactical Layer  ·  LLM", C_LLM_DARK, size=11)
body(50, 69.4, "Qwen2.5-1.5B  +  LoRA   (short output)", GRAY_TXT, size=8.6)
body(50, 65.6, "→  {  action: overtake | hold,   extra_inside_m: 0–3.5 m,   speed_factor: 0.9–1.1  }",
     LIGHT_TXT, size=7.6)
badge(82.4, 73.4, "0.59 s", C_LLM, size=8.0)
badge(82.4, 63.0, "edge 8 GB", "#94a3b8", size=6.6)

arrow(50, 58, 50, 53.2, color=C_LLM_DARK)
badge(58.2, 55.6, "intent (held)", C_LLM_DARK, size=7.0)

# ================= control layer =================
rbox(18, 36, 64, 17.2, "#f0fdfa", C_CTRL, lw=1.4)
rbox(18, 36, 1.6, 17.2, C_CTRL, C_CTRL, lw=0)
title(50, 50.0, "Control Layer", C_CTRL_DARK, size=11)
body(50, 45.9, "pure-pursuit racing-line tracking  +  attack offset", GRAY_TXT, size=8.4)
body(50, 42.0, "signed shield:  brake if  gap < 8 m  &  |sep| < 2.2 m  (alongside)",
     GRAY_TXT, size=8.4)
badge(82.4, 50.0, "20 Hz", C_CTRL, size=8.0)

arrow(50, 36, 50, 31.4, color=C_CTRL_DARK)

# ================= actuation =================
rbox(22, 20, 56, 11.4, "#f1f5f9", C_ACT)
title(50, 28.2, "Vehicle Actuation", C_ACT, size=10)
body(50, 23.6, "throttle · brake · steering   (CARLA physics)", LIGHT_TXT, size=8.0)

# ================= feedback loop =================
arrow(82, 23.0, 82, 86.6, color=C_LOOP, lw=1.6)
arrow(82, 86.6, 78.5, 86.6, color=C_LOOP, lw=1.6)
ax.text(86.5, 54.8, "state\nfeedback", fontsize=7.4, color=C_LOOP,
        ha="center", va="center", fontweight="bold", rotation=90, zorder=4)

# ================= left timescale rail =================
ax.plot([15.5, 15.5], [58, 76.6], color="#cbd5e1", lw=1.0, zorder=1)
ax.plot([15.5, 15.5], [36, 53.2], color="#cbd5e1", lw=1.0, zorder=1)
ax.text(11.2, 68.0, "decision\ncadence\n(≈ 1 Hz)", fontsize=7.0, color=LIGHT_TXT,
        ha="center", va="center", rotation=90)
ax.text(11.2, 44.6, "control loop\n(20 Hz)", fontsize=7.0, color=LIGHT_TXT,
        ha="center", va="center", rotation=90)

fig.tight_layout(pad=0.4)
fig.savefig("results/media/architecture.png", bbox_inches="tight", facecolor="white")
fig.savefig("results/media/architecture.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("results/media/architecture.svg", bbox_inches="tight", facecolor="white")
print("saved architecture.png/.pdf/.svg")
