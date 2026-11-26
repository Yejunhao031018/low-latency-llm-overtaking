#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a short-output SFT variant: drop the `reason` field from the
assistant answer (and the system prompt).  The decision fields (action /
extra_inside_m / speed_factor) are what drive the controller; reason is only an
explanation and costs ~40 of the ~64 generated tokens.

    /home/yjh/venv-racingllm/bin/python finetune/make_short.py --in dataset/sft_aug --out dataset/sft_short
"""
import argparse
import json
import os

SYS_LONG = ("You are a racing tactical planner for a car approaching a corner. "
            "Answer ONLY a JSON object with keys: action (overtake|hold), "
            "extra_inside_m (float, 0..3.5; 0 = racing line, positive = further toward the corner inside), "
            "speed_factor (float, 0.9..1.1), reason (short text). "
            "Never request an inside pass with less than 3.0 m extra offset.")

SYS_SHORT = ("You are a racing tactical planner for a car approaching a corner. "
             "Answer ONLY a JSON object with keys: action (overtake|hold), "
             "extra_inside_m (float, 0..3.5; 0 = racing line, positive = further toward the corner inside), "
             "speed_factor (float, 0.9..1.1). "
             "Never request an inside pass with less than 3.0 m extra offset.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="dataset/sft_aug")
    ap.add_argument("--out", default="dataset/sft_short")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in open(os.path.join(args.inp, split + ".jsonl"))]
        for r in rows:
            msgs = r["messages"]
            if msgs[0]["content"] == SYS_LONG:
                msgs[0]["content"] = SYS_SHORT
            a = json.loads(msgs[-1]["content"])
            a.pop("reason", None)
            msgs[-1]["content"] = json.dumps(a, ensure_ascii=False, separators=(",", ":"))
        with open(os.path.join(args.out, split + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(rows))
    # quick token length check
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "models/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master", trust_remote_code=True)
    lens = []
    for l in open(os.path.join(args.out, "train.jsonl")):
        r = json.loads(l); m = r["messages"]
        p = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
        lens.append(len(tok(p + m[-1]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]))
    print("mean full seq len:", round(sum(lens) / len(lens), 1))


if __name__ == "__main__":
    main()
