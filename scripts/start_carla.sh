#!/usr/bin/env bash
set -euo pipefail
if /home/yjh/carla-venv/bin/python - <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(1.0)
ok = s.connect_ex(("127.0.0.1", 2000)) == 0
s.close(); sys.exit(0 if ok else 1)
PY
then
  echo "CARLA already listening on 2000."
  exit 0
fi
cd /home/yjh/carla/Unreal/CarlaUE4
export HOME=/home/yjh
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
LOG=/tmp/carla_overtake.log
setsid nohup /home/yjh/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor \
  /home/yjh/carla/Unreal/CarlaUE4/CarlaUE4.uproject \
  -game -RenderOffScreen -nosound -noloadingscreen \
  -DEFAULTMAP=/Game/Carla/Maps/Town04.Town04 \
  -carla-rpc-port=2000 -carla-streaming-port=2001 \
  > "$LOG" 2>&1 < /dev/null &
echo "launched UE4Editor, waiting for port 2000 (log: $LOG) ..."
for i in $(seq 1 120); do
  if /home/yjh/carla-venv/bin/python - <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(0.5)
ok = s.connect_ex(("127.0.0.1", 2000)) == 0
s.close(); sys.exit(0 if ok else 1)
PY
  then
    echo "CARLA ready on 2000."
    exit 0
  fi
  sleep 1
done
echo "CARLA did not become ready within 120s; see $LOG"
exit 1
