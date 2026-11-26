#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG=/tmp/opencode/local_lora.log
exec > >(tee -a "$LOG") 2>&1

echo "[1/3] installing LLM env (torch cu128) ..."
./install_llm_env.sh

echo "[2/3] downloading model from ModelScope ..."
MODEL_PATH=$(./run_llm.sh finetune/download_model.py --cache models | tee /tmp/opencode/model_download.log | sed -n 's/^MODEL_PATH=//p' | tail -1)
if [ -z "${MODEL_PATH:-}" ]; then
  echo "model download failed"; exit 1
fi
echo "MODEL_PATH=$MODEL_PATH"

echo "[3/3] training bf16 LoRA ..."
./run_llm.sh finetune/train_lora_local.py \
  --model "$MODEL_PATH" \
  --data dataset/sft \
  --out checkpoints/qwen1.5b-sft \
  --epochs 3 --max-len 512 --batch 1 --grad-accum 8 \
  --eval-samples 4
echo "DONE"
