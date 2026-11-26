#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mirror train_status.json into results/ for the training dashboard."""
import json
import os
import time

SRC = "checkpoints/qwen1.5b-sft-short-multi2/train_status.json"
OUT = "results/train_lr_status.json"


def main():
    while True:
        try:
            if os.path.exists(SRC):
                st = json.load(open(SRC))
                json.dump(st, open(OUT + ".tmp", "w"))
                os.replace(OUT + ".tmp", OUT)
            else:
                json.dump({"phase": "pending", "updated_at": time.strftime("%H:%M:%S")},
                          open(OUT + ".tmp", "w"))
                os.replace(OUT + ".tmp", OUT)
        except Exception:
            pass
        time.sleep(3.0)


if __name__ == "__main__":
    main()
