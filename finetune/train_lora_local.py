#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local bf16 LoRA SFT for the tactical JSON policy.

Designed for RTX 5060 Laptop 8 GB + Qwen2.5-1.5B-Instruct (no 4-bit needed).
Uses transformers + peft directly (no TRL version drift).

    /home/yjh/venv-racingllm/bin/python finetune/train_lora_local.py \
        --model models/Qwen/Qwen2.5-1.5B-Instruct \
        --data dataset/sft --out checkpoints/qwen1.5b-sft

Data: chat JSONL with messages[system,user,assistant] + meta.
Loss: completion-only (prompt tokens masked to -100).
"""
import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class ChatSFT(Dataset):
    def __init__(self, rows, tok, max_len):
        self.examples = []
        eos = tok.eos_token or ""
        for r in rows:
            msgs = r["messages"]
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                             add_generation_prompt=True)
            answer = msgs[-1]["content"] + eos
            full = prompt + answer
            ids = tok(full, add_special_tokens=False)["input_ids"]
            pids = tok(prompt, add_special_tokens=False)["input_ids"]
            labels = list(ids)
            n = min(len(pids), len(labels))
            labels[:n] = [-100] * n
            if len(ids) > max_len:
                ids = ids[:max_len]
                labels = labels[:max_len]
            if all(x == -100 for x in labels):
                continue
            self.examples.append((ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    max_len = max(len(x[0]) for x in batch)
    input_ids, attention, labels = [], [], []
    for ids, labs in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [-100] * pad)
    return (torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for ids, att, lab in loader:
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            total += float(out.loss)
            n += 1
    return total / max(1, n)


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
    ap.add_argument("--model", default="models/Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="dataset/sft")
    ap.add_argument("--out", default="checkpoints/qwen1.5b-sft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--eval-samples", type=int, default=4)
    ap.add_argument("--load-4bit", action="store_true",
                    help="load the base model in 4-bit (QLoRA) for 3B+ on 8GB")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map={"": 0},
            trust_remote_code=True)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16, device_map={"": 0},
                trust_remote_code=True)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
                trust_remote_code=True)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    train_rows = load_jsonl(os.path.join(args.data, "train.jsonl"))
    val_rows = load_jsonl(os.path.join(args.data, "val.jsonl"))
    test_rows = load_jsonl(os.path.join(args.data, "test.jsonl"))
    train_ds = ChatSFT(train_rows, tok, args.max_len)
    val_ds = ChatSFT(val_rows, tok, args.max_len)
    test_ds = ChatSFT(test_rows, tok, args.max_len)
    print("samples train/val/test:", len(train_ds), len(val_ds), len(test_ds))

    coll = lambda b: collate(b, tok.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=coll)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    device = "cuda:0"
    model.train()
    step = 0
    history = []
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    t_start = time.time()
    status_path = os.path.join(args.out, "train_status.json")
    loss_points = []
    last_metrics = {}

    def write_status(phase="train", epoch=0, **kw):
        elapsed = time.time() - t_start
        eta = (elapsed / step * (total_steps - step)) if step > 0 else None
        for k in ("loss", "val_loss"):
            if k in kw:
                last_metrics[k] = kw[k]
        st = dict(phase=phase, epoch=epoch, epochs=args.epochs,
                  opt_step=step, total_steps=total_steps,
                  fraction=(step / total_steps if total_steps else 0.0),
                  elapsed_s=round(elapsed, 1),
                  eta_s=(round(eta, 1) if eta is not None else None),
                  train_samples=len(train_ds), val_samples=len(val_ds),
                  loss_points=loss_points[-300:],
                  updated_at=time.strftime("%H:%M:%S"))
        st.update(last_metrics)
        st.update(kw)
        try:
            tmp = status_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, status_path)
        except Exception:
            pass

    write_status(phase="start")
    pbar = tqdm(total=total_steps, desc="LoRA", unit="step", dynamic_ncols=True)
    for epoch in range(args.epochs):
        total, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (ids, att, lab) in enumerate(train_loader):
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            loss = out.loss / args.grad_accum
            loss.backward()
            total += float(out.loss)
            n += 1
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                avg = total / max(1, n)
                loss_points.append([step, round(avg, 4)])
                pbar.update(1)
                pbar.set_postfix(ep="%d/%d" % (epoch + 1, args.epochs),
                                 loss="%.4f" % avg)
                print("epoch %d step %d/%d loss %.4f" % (
                    epoch, step, total_steps, avg), flush=True)
                history.append({"epoch": epoch, "step": step, "loss": avg})
                write_status(epoch=epoch, loss=round(avg, 4))
                total, n = 0.0, 0.0
        # final partial accumulation
        if len(train_loader) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            if n > 0:
                avg = total / n
                loss_points.append([step, round(avg, 4)])
                write_status(epoch=epoch, loss=round(avg, 4))
        val_loss = evaluate(model, val_loader, device)
        print("epoch %d val_loss %.4f" % (epoch, val_loss), flush=True)
        history.append({"epoch": epoch, "val_loss": val_loss})
        pbar.set_postfix(ep="%d/%d" % (epoch + 1, args.epochs),
                         val="%.4f" % val_loss)
        write_status(phase="val", epoch=epoch, val_loss=round(val_loss, 4))
        model.train()
    pbar.close()
    write_status(phase="done", val_loss=round(val_loss, 4))

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # qualitative generation on a few test samples
    model.eval()
    examples = []
    for r in test_rows[:args.eval_samples]:
        msgs = r["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                         add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_json(text)
        target = parse_json(msgs[-1]["content"])
        examples.append({"target": target, "pred": pred, "raw": text[:200]})
        print("PRED", pred, "| TARGET", target)
    with open(os.path.join(args.out, "eval_samples.json"), "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print("saved adapter:", args.out)


if __name__ == "__main__":
    main()
