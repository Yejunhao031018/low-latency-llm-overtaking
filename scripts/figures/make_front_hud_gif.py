#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Front-view (driver perspective) overtake GIF with live decision HUD."""
import csv
import glob
import os
from PIL import Image, ImageDraw, ImageFont

FRAME_DT = 0.3

decisions = []
with open("telemetry_race.csv") as f:
    for r in csv.DictReader(f):
        decisions.append((float(r["t"]), r.get("llm_action", "hold"),
                          float(r.get("extra_cmd", 0.0))))

def decision_at(t):
    cur = ("hold", 0.0)
    for tt, a, e in decisions:
        if tt <= t + 1e-6:
            cur = (a, e)
        else:
            break
    return cur

try:
    font_big = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 26)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 18)
except Exception:
    font_big = ImageFont.load_default()
    font_sm = ImageFont.load_default()

frames = sorted(glob.glob("frames_race/front_*.png"))
start_i = int(8 / FRAME_DT)
end_i = int(24 / FRAME_DT)
selected = frames[start_i:end_i:2]

imgs = []
for i, f in enumerate(selected):
    t = (start_i + 2 * i) * FRAME_DT
    im = Image.open(f).convert("RGB").resize((480, 360))
    d = ImageDraw.Draw(im)
    a, e = decision_at(t)
    d.rectangle([0, 300, 480, 360], fill=(10, 18, 34, 235))
    col = (56, 189, 248) if a == "overtake" else (148, 163, 184)
    d.text((10, 306), "LLM decision  ·  t=%.1f s" % t, font=font_big, fill=(255, 255, 255))
    d.text((10, 332), "%s   extra=%.1f m" % (a.upper(), e), font=font_big, fill=col)
    d.text((10, 349), "latency 0.59 s", font=font_sm, fill=(143, 163, 200))
    imgs.append(im)

out = "results/media/overtake_front_hud.gif"
imgs[0].save(out, save_all=True, append_images=imgs[1:],
             duration=140, loop=0, optimize=False)
print("saved", out, f"({len(imgs)} frames)")
