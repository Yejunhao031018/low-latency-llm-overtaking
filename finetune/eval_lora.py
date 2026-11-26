#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a LoRA adapter on a chat JSONL split: parse rate, action accuracy, latency."""
import argparse
import json
import statistics
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_json(text):
    start = text.find("{")
    if start < 0:
        return None
    depth, ins, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if ins:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                ins = False
        else:
            if c == '"':
                ins = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        return None
    return None


def load_base(path):
    kw = dict(device_map={"": 0}, trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kw)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default="dataset/sft/test.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(load_base(args.model), args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    results, lat = [], []
    n_ok = n_parsed = n_action = 0
    conf = {}
    for r in rows:
        msgs = r["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                         add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        lat.append(dt)
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_json(text)
        target = parse_json(msgs[-1]["content"])
        n_ok += 1
        if pred is not None:
            n_parsed += 1
        if pred is not None and target is not None:
            key = (target.get("action"), pred.get("action"))
            conf[key] = conf.get(key, 0) + 1
            if pred.get("action") == target.get("action"):
                n_action += 1
        results.append({"target": target, "pred": pred, "raw": text[:300],
                        "latency_s": dt})

    lat_sorted = sorted(lat)
    def pct(p):
        return lat_sorted[min(len(lat_sorted) - 1, int(p * len(lat_sorted)))]
    summary = {
        "n": n_ok,
        "json_parse_rate": n_parsed / max(1, n_ok),
        "action_accuracy": n_action / max(1, n_ok),
        "latency_mean_s": statistics.mean(lat) if lat else 0.0,
        "latency_p50_s": pct(0.50),
        "latency_p95_s": pct(0.95),
        "max_new_tokens": args.max_new,
        "confusion": {f"{t}->{p}": v for (t, p), v in sorted(conf.items())},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "results": results}, f,
                      indent=2, ensure_ascii=False)
        print("saved:", args.out)


if __name__ == "__main__":
    main()
