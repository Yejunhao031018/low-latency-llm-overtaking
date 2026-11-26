#!/usr/bin/env bash
# Minimal data-engine pipeline for the tactical-LLM paper.
# CARLA Town04 must already be running on port 2000.
#
#   ./run_pipeline.sh                 # 3 episodes -> pilot -> SFT -> MLP
#   ./run_pipeline.sh dataset/pilot2  # custom output dir
#
set -euo pipefail

cd "$(dirname "$0")"
DATA="${1:-dataset/pilot}"
PY=/home/yjh/carla-venv/bin/python

export PYTHONPATH=/home/yjh/carla/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg
export LD_LIBRARY_PATH="$PWD/deps:/snap/dsh-linux/1:${LD_LIBRARY_PATH:-}"

echo "== 1) collect clean episodes (no inline rollout) =="
"$PY" collect_dataset.py --episodes 3 --policy stub --out "$DATA"

echo "== 2) counterfactual labeling (separate pass) =="
"$PY" label_dataset.py --dataset "$DATA" --max-states 4 --horizon 6

echo "== 3) convert to SFT / DPO =="
"$PY" finetune/convert.py --dataset "$DATA" --out "$DATA/sft"

echo "== 4) small-model baseline =="
"$PY" finetune/train_mlp.py --data "$DATA/sft" --out "$DATA/sft/mlp"

echo
echo "done. Next:"
echo "  - inspect $DATA/labels.jsonl"
echo "  - restart CARLA before collecting more (labeler moves actors)"
echo "  - train 3B QLoRA in a Python 3.10+ venv: finetune/README.md"
