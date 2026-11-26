#!/usr/bin/env bash
# Step 2: 采集 10 场景联合数据集（stub 驾驶 + 反事实规划标注）
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUT=dataset/multi10
rm -rf "$OUT"; mkdir -p "$OUT"

# (corner_radius:lead_policy) —— 锁定 10 场景
SCENES=(
  "208:constant" "208:variable" "208:defensive"
  "151:constant" "151:variable" "151:defensive"
  "188:constant" "188:defensive"
  "83:constant" "138:variable"
)

seed=100
for entry in "${SCENES[@]}"; do
  R="${entry%%:*}"; L="${entry##*:}"
  for gap in 20 28; do
    echo "=== scene R=$R lead=$L gap=$gap seed=$seed ==="
    ./run_py.sh -u collect_dataset.py --episodes 1 --policy stub --fixed \
      --lead-policy "$L" --corner-radius "$R" \
      --racing-off-in 1.0 --off-cap 4.5 \
      --lead-factor 0.9 --start-gap "$gap" --seed "$seed" \
      --max-time-s 45 --out "$OUT" 2>&1 | grep -E "summary:" | tail -1
    seed=$((seed+1))
  done
done

echo "=== 反事实规划标注（一次性） ==="
./run_py.sh -u label_dataset.py --dataset "$OUT" --max-states 4 --horizon 4.0 2>&1 | tail -4
echo "MULTI10_DONE"
