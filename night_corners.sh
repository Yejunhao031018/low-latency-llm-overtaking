#!/usr/bin/env bash
# 6-corner generalization: run the opt-pure + LoRA closed loop at every corner.
set -euo pipefail
cd /home/yjh/snap/dsh-linux/1/dsh-haness-ws/overtake_demo
COMMON="-u run_benchmark.py --configs e1 --seeds 2 --gaps 20 --leads constant \
  --policy lora --bias 1.5 --lead-factor 0.75 --controller pure \
  --sep-gate 2.5 --gate-bias -1.5 --attack-dv 6 --pp-yaw-damp 0.2"
for R in 45 83 138 151 188 208; do
  echo "=== corner R=$R ==="
  ./run_py.sh $COMMON --corner-radius $R --out results/night_corner_$R 2>&1 | grep -E "pass=" | tail -4
done
echo "ALL_CORNERS_DONE"
