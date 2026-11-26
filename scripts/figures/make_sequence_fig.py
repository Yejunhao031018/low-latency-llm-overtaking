#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a publication-quality overtaking-sequence figure from dense frames.

Reads frames_race/bird_*.png (saved every 6 sim frames = 0.3 s at 20 Hz),
locates the overtake window from the pass time, and stitches a horizontal
time-lapse strip with per-frame labels.
"""
import glob
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image

for p in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
    fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Liberation Sans"

FRAME_DT = 0.3  # seconds per saved frame (every 6 sim ticks at 20 Hz)

def load_pass_time():
    # result_race.json is written by race_overtake.py
    p = "result_race.json"
    if os.path.exists(p):
        d = json.load(open(p))
        return d.get("pass_time_s")
    return None

def main():
    birds = sorted(glob.glob("frames_race/bird_*.png"))
    if not birds:
        print("no frames"); return
    pass_t = load_pass_time()
    n = len(birds)
    print(f"frames={n}, pass_time={pass_t}")

    # Choose 6 frames: 2 before the pass, the pass, 3 after.
    if pass_t is None:
        pass_t = 18.7
    center = int(pass_t / FRAME_DT)
    offsets = [-5, -2, 0, 2, 5, 8]  # in frame indices
    idx = [max(0, min(n - 1, center + o)) for o in offsets]
    labels = ["%.1f s" % (i * FRAME_DT) for i in idx]
    phases = ["approach", "approach", "alongside", "pass", "clear", "clear"]

    imgs = [Image.open(birds[i]).convert("RGB") for i in idx]

    # 2 rows x 3 columns (balanced aspect), with phase labels
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), dpi=180)
    for ax, im, lab, phase in zip(axes.flat, imgs, labels, phases):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(f"t = {lab}   ·   {phase}", fontsize=10.5,
                     fontweight="bold", color="#0f172a", pad=4)
    fig.tight_layout(pad=0.5)
    os.makedirs("results/media", exist_ok=True)
    fig.savefig("results/media/overtake_sequence.png", bbox_inches="tight",
                facecolor="white", dpi=180)
    fig.savefig("results/media/overtake_sequence.pdf", bbox_inches="tight",
                facecolor="white")
    fig.savefig("results/media/overtake_sequence.svg", bbox_inches="tight",
                facecolor="white")
    print("saved overtake_sequence.png/.pdf/.svg (2x3 grid)")

if __name__ == "__main__":
    main()
