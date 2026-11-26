#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the SFT split used by the closed-loop LoRA policy.

Two data problems showed up in the first closed-loop attempt:
  1. the observation had no memory of the previous tactical action, so
     "attack entry" and "attack exit" looked identical;
  2. only a handful of samples showed an attack being initiated from the
     racing line, so the model always answered hold there.

convert.py now adds prev_action.  This script additionally creates a
counterfactual "pre-attack" variant for every overtake state that lies inside
the legal window: lateral offset reset to the racing line, prev_action=hold,
same overtake target.  That is the state the car is actually in when the
tactical layer has to decide to start the pass.

    python finetune/augment_init.py --data dataset/sft_v2 --out dataset/sft_aug
"""
import argparse
import json
import os


def load(path):
    return [json.loads(l) for l in open(path)]


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/sft_v2")
    ap.add_argument("--out", default="dataset/sft_aug")
    ap.add_argument("--overtake-copies", type=int, default=2)
    ap.add_argument("--init-copies", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    splits = {s: load(os.path.join(args.data, s + ".jsonl"))
              for s in ("train", "val", "test")}

    extra = []
    n_init = 0
    for r in splits["train"]:
        state = json.loads(r["messages"][1]["content"])
        target = json.loads(r["messages"][-1]["content"])
        if target.get("action") != "overtake":
            continue
        if not (-220.0 <= state.get("d_apex_m", 1e9) <= 150.0):
            continue
        if not (6.0 <= state.get("gap_m", 1e9) <= 60.0):
            continue
        init = json.loads(r["messages"][1]["content"])
        init["actual_extra_m"] = 0.0
        init["actual_off_m"] = init.get("racing_off_m", 0.0)
        init["lateral_sep_m"] = 0.0
        init["prev_action"] = "hold"
        out = json.loads(r["messages"][-1]["content"])
        out["action"] = "overtake"
        out["extra_inside_m"] = 3.5
        out["speed_factor"] = max(1.0, min(1.1, out.get("speed_factor", 1.05)))
        out["reason"] = "gap=%.1f m d_apex=%.0f m; initiate inside 3.5 m" % (
            state.get("gap_m", 0.0), state.get("d_apex_m", 0.0))
        row = {"messages": [
            {"role": "system", "content": r["messages"][0]["content"]},
            {"role": "user", "content": json.dumps(init, ensure_ascii=False,
                                                   separators=(",", ":"))},
            {"role": "assistant", "content": json.dumps(out, ensure_ascii=False,
                                                        separators=(",", ":"))},
        ], "meta": dict(r.get("meta", {}), source="init_aug")}
        for _ in range(args.init_copies):
            extra.append(row)
            n_init += 1

    train = []
    for r in splits["train"]:
        target = json.loads(r["messages"][-1]["content"])
        n = args.overtake_copies if target.get("action") == "overtake" else 1
        train.extend([r] * n)
    train.extend(extra)

    dump(os.path.join(args.out, "train.jsonl"), train)
    dump(os.path.join(args.out, "val.jsonl"), splits["val"])
    dump(os.path.join(args.out, "test.jsonl"), splits["test"])
    print("train %d (base %d, init-aug %d) | val %d | test %d" % (
        len(train), len(splits["train"]), n_init,
        len(splits["val"]), len(splits["test"])))


if __name__ == "__main__":
    main()
