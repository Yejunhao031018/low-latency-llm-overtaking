#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-scale evaluation (E4): JSON / action accuracy + latency for base models
(and optionally a LoRA adapter), on the SFT test split.

    /home/yjh/venv-racingllm/bin/python finetune/eval_models.py \
        --model models/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master \
        --data dataset/sft_aug/test.jsonl --limit 60 --out results/e4_05b.json
"""
import argparse
import json
import time

import torch
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data", default="dataset/sft_aug/test.jsonl")
    ap.add_argument("--out", default="results/eval_models.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--load-4bit", action="store_true",
                    help="load base in 4-bit (for 3B+ on 8GB)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map={"": 0}, trust_remote_code=True)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, **kw)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    lats = []
    n_parsed = n_action = n = 0
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
        lats.append(time.perf_counter() - t0)
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_json(text)
        target = parse_json(msgs[-1]["content"])
        n += 1
        if pred is not None:
            n_parsed += 1
        if pred is not None and target is not None and pred.get("action") == target.get("action"):
            n_action += 1
    lats = sorted(lats)
    p50 = lats[int(0.50 * (len(lats) - 1))] if lats else None
    p95 = lats[int(0.95 * (len(lats) - 1))] if lats else None
    out = dict(model=args.model, adapter=args.adapter, n=n,
               json_parse_rate=n_parsed / max(1, n),
               action_accuracy=n_action / max(1, n),
               latency_mean=(sum(lats) / n if n else 0.0),
               latency_p50=p50, latency_p95=p95)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
