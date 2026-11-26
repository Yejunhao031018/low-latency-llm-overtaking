#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watch the 3B model download and write a progress JSON for the dashboard."""
import glob
import json
import os
import time

SNAP = "models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
OUT = "results/dl_3b_status.json"
TOTAL = 6.09e9  # ~6 GB safetensors
BNB = "/home/yjh/venv-racingllm/lib/python3.11/site-packages/bitsandbytes/__init__.py"


def main():
    os.makedirs("results", exist_ok=True)
    prev_size = 0.0
    prev_t = time.time()
    while True:
        files = glob.glob(os.path.join(SNAP, "*"))
        files = [f for f in files if os.path.isfile(f)]
        size = sum(os.path.getsize(f) for f in files)
        now = time.time()
        dt = max(1e-6, now - prev_t)
        speed = (size - prev_size) / dt
        eta = (TOTAL - size) / speed if speed > 0 else None
        st = dict(
            bytes_done=int(size), bytes_total=int(TOTAL),
            percent=round(min(100.0, size / TOTAL * 100.0), 2),
            speed_mb=round(speed / 1e6, 2),
            eta_s=round(eta, 0) if eta is not None else None,
            bitsandbytes="installed" if os.path.exists(BNB) else "installing",
            n_files=len(files),
            updated_at=time.strftime("%H:%M:%S"),
        )
        try:
            tmp = OUT + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, OUT)
        except Exception:
            pass
        prev_size, prev_t = size, now
        time.sleep(2.0)


if __name__ == "__main__":
    main()
