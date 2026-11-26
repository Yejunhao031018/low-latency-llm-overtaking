#!/usr/bin/env bash
# 录制超车 demo 视频 + 提取关键帧图片（成果展示用）
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MEDIA=results/media
rm -rf "$MEDIA"; mkdir -p "$MEDIA"

# 1) 跑 demo（LLM 决策驱动，指向本地 lora server）
echo "=== 运行超车 demo（LLM） ==="
USE_LLM_DECISION=1 LLM_BACKEND=openai \
  LLM_BASE_URL=http://127.0.0.1:8765/v1 \
  ./run_py.sh -u race_overtake.py 2>&1 | tail -25

# 2) 合成视频（鸟瞰 + 前视）
echo "=== 合成视频 ==="
if ls frames_race/bird_*.png >/dev/null 2>&1; then
  n_bird=$(ls frames_race/bird_*.png | wc -l)
  n_front=$(ls frames_race/front_*.png | wc -l)
  echo "帧数: bird=$n_bird front=$n_front"
  # 鸟瞰视频（约 10 fps）
  ffmpeg -y -framerate 10 -i frames_race/bird_%03d.png -c:v libx264 -pix_fmt yuv420p \
    -vf "scale=800:600" "$MEDIA/overtake_bird.mp4" 2>/dev/null
  # 前视视频
  ffmpeg -y -framerate 10 -i frames_race/front_%03d.png -c:v libx264 -pix_fmt yuv420p \
    -vf "scale=800:450" "$MEDIA/overtake_front.mp4" 2>/dev/null
else
  echo "没有帧！demo 可能失败或相机未输出"
fi

# 3) 提取关键帧图片（鸟瞰均匀采样 8 张）
echo "=== 提取关键帧图片 ==="
if ls frames_race/bird_*.png >/dev/null 2>&1; then
  n=$(ls frames_race/bird_*.png | wc -l)
  /home/yjh/venv-racingllm/bin/python - "$n" "$MEDIA" <<'PY'
import glob, os, shutil, sys
n = int(sys.argv[1]); media = sys.argv[2]
frames = sorted(glob.glob('frames_race/bird_*.png'))
# 均匀采样最多 8 张
k = min(8, len(frames))
idx = [int(i*(len(frames)-1)/max(1,k-1)) for i in range(k)]
for j, i in enumerate(idx):
    shutil.copy(frames[i], os.path.join(media, f"keyframe_{j:02d}.png"))
print(f"提取 {k} 张关键帧（共 {len(frames)} 帧）")
PY
fi

echo "=== 成果 ==="
ls -la "$MEDIA"
echo "MEDIA_DONE"
