#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO/ORPO on counterfactual preference pairs.

The pairs come from dataset/sft/dpo.jsonl:
    prompt   = system + observation
    chosen   = safe / faster action JSON
    rejected = unsafe / slower action JSON

    python finetune/train_dpo.py --model Qwen/Qwen2.5-3B-Instruct \
        --data dataset/sft/dpo.jsonl --out checkpoints/qwen3b-dpo

Use the same training venv as train_lora.py.  DPO is more memory-hungry than
SFT; keep 4-bit, batch 1, max_length 768 on 8 GB.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="dataset/sft/dpo.jsonl")
    ap.add_argument("--out", default="checkpoints/dpo")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    ds = load_dataset("json", data_files=args.data, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    cfg = DPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_len,
        max_prompt_length=args.max_len // 2,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model, ref_model=None, args=cfg,
        train_dataset=ds, tokenizer=tokenizer, peft_config=lora)
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    with open(args.out + "/train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("saved DPO adapter to", args.out)


if __name__ == "__main__":
    main()
