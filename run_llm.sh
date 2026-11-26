#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
exec /home/yjh/venv-racingllm/bin/python "$@"
