#!/usr/bin/env bash
# Source this file: source ./env_carla.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="/home/yjh/carla/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg"
export LD_LIBRARY_PATH="$ROOT/deps:/snap/dsh-linux/1:${LD_LIBRARY_PATH:-}"
echo "CARLA python env ready: PYTHONPATH=$PYTHONPATH"
