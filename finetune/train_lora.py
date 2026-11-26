#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA SFT for the tactical JSON policy.

This script is intentionally not runnable inside the CARLA venv: it needs a
modern Python (3.10+), torch, transformers, peft, trl, datasets and bitsandbytes.
Create a separate training environment:

    python3.10 -m venv ~/venv-llm
    source ~/venv-llm/bin/activate
    pip install -r finetune/requirements-finetune.txt

Then:
    python finetune/train_lora.py \
      --model Qwen/Qwen2.5-3B-Instruct \
      --data dataset/sft \
      --out checkpoints/qwen3b-sft

8 GB VRAM recommendation: 3B + 4-bit QLoRA, max_seq_len 768-1024,
per_device_batch 1, grad_accum 8.  7B is possible but tight.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="dataset/sft")
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer

    train = load_dataset("json", data_files=os.path.join(args.data, "train.jsonl"),
                         split="train")
    val = load_dataset("json", data_files=os.path.join(args.data, "val.jsonl"),
                       split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        max_length=args.max_seq_len,
        packing=False,
        assistant_only_loss=True,   # TRL >= 0.12; if unsupported, remove this line
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train,
        eval_dataset=val,
        peft_config=lora,
        processing_class=tokenizer,   # older TRL: use tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("saved adapter to", args.out)


if __name__ == "__main__":
    main()
