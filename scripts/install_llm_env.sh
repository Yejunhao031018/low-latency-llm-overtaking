#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
PIP=/home/yjh/venv-racingllm/bin/pip
PY=/home/yjh/venv-racingllm/bin/python
LOG=/tmp/opencode/pip_llm.log
exec > >(tee -a "$LOG") 2>&1

WHEEL=/tmp/opencode/torch-2.9.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
URL="https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.9.1%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl"

echo "[1/3] downloading torch 2.9.1+cu128 (901 MB) with curl ..."
if [ ! -s "$WHEEL" ]; then
  curl -L --retry 5 -C - -x http://127.0.0.1:7890 -o "$WHEEL" "$URL"
fi
ls -lh "$WHEEL"

echo "[2/3] installing local torch wheel (deps from Tsinghua) ..."
"$PIP" install "$WHEEL" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --timeout 180 --retries 5

echo "[3/3] installing other deps from Tsinghua ..."
"$PIP" install transformers peft accelerate modelscope sentencepiece safetensors datasets \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --timeout 180 --retries 5

"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
echo "ENV_DONE"
