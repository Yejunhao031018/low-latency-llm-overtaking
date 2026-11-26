#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source ./env_carla.sh
# check the CARLA RPC port
if ! /home/yjh/carla-venv/bin/python - <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(1.0)
ok = s.connect_ex(("127.0.0.1", 2000)) == 0
s.close(); sys.exit(0 if ok else 1)
PY
then
  echo "CARLA is not listening on 127.0.0.1:2000."
  echo "Start it first:  ./start_carla.sh"
  exit 1
fi
exec /home/yjh/carla-venv/bin/python latency_overtake.py "$@"
