# 数据引擎与微调管线

这份目录把“LLM 战术决策”论文需要的最小闭环拆成了可执行步骤：

```
CARLA episode
  → decisions.jsonl（state / raw / filtered / executed / snapshot）
  → label_dataset.py（同一状态跑多个候选动作 → cost-to-go）
  → labels.jsonl（best action / 候选 / 代价）
  → finetune/convert.py（SFT chat 数据 + DPO 偏好对）
  → finetune/train_mlp.py（小模型 baseline）
  → finetune/train_lora.py（QLoRA SFT）
  → finetune/train_dpo.py（安全偏好优化）
  → vLLM 服务 + race_overtake.py / dataset_runner.py 闭环评测
```

## 1. 采集 clean episode

先启动 CARLA Town04，然后：

```bash
cd /home/yjh/snap/dsh-linux/1/dsh-haness-ws/overtake_demo
export PYTHONPATH=/home/yjh/carla/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg
export LD_LIBRARY_PATH=$PWD/deps:/snap/dsh-linux/1:$LD_LIBRARY_PATH

# 3 条随机化 stub episode（推荐先采 clean，再单独 label）
python collect_dataset.py --episodes 3 --policy stub --out dataset/pilot

# 单条 nominal（gap 26m, constant lead）
python collect_dataset.py --episodes 1 --fixed --start-gap 26 \
  --lead-policy constant --policy stub --out dataset/pilot
```

每条 episode 输出：

```
dataset/pilot/<episode_id>/
├── decisions.jsonl   # 每 0.5s 一条：state / raw / filtered / executed / snapshot
├── ticks.csv         # 20Hz 遥测
└── summary.json      # 碰撞/最小中心距/超车点/圈速/配置
```

`snapshot` 只在 T2 前后 `d_apex ∈ [-260, 150]` 记录，包含两车的
transform / velocity / angular velocity / extra_cmd / s 参考。

## 2. 反事实标注

**不要在主 episode 里 inline 跑 rollout**：CARLA 的 transform/velocity
restore 是近似的，边跑边 restore 会污染主轨迹（我们在开发时确实复现了
8706 次连续碰撞）。采集和标注必须分开。

```bash
python label_dataset.py --dataset dataset/pilot --max-states 4 --horizon 6
```

每个 state 跑 5 个候选（hold / inside 3.0 / inside 3.5 / late brake / outside），
记录碰撞、最小中心距、近距离横向净空、是否 pass、pass 时间，并计算 cost：

```
collision*1000
+ 200*max(0, 2.2 - min_sep_close)
+ 50*max(0, 3.0 - min_center)
+ 60 if no pass
- 0.8*clamp(final_gap - initial_gap, -20, 20)
+ 5*|extra_inside|
```

> 重要：labeler 会移动车辆做 rollout。跑完 labeler 后如果还要继续采集，
> 建议重启 CARLA，避免残留物理状态影响后续 episode。

`dataset/pilot_invalid/` 里有一条 `gap≈44m / constant lead` 的 stub 追尾样本：
不是 restore 污染，而是后车较晚 pass 后并线/减速时，前车（没有 car-following
避撞的 NPC）撞上来。它说明**对手也需要安全控制**；这条留作 hard negative，
不进入 clean SFT。

## 3. 转 SFT / DPO

```bash
python finetune/convert.py --dataset dataset/pilot --out dataset/sft
```

输出：

```
dataset/sft/
├── train.jsonl / val.jsonl / test.jsonl   # chat messages + meta
├── dpo.jsonl                              # prompt / chosen / rejected
└── stats.json
```

规则：
- 有反事实标签的 state 用 `best_action`（planner 标签）。
- 没有标签的 stub state 只在 T2 附近或 overtake 时保留；远离 T2 的 hold 下采样。
- random policy 只在有标签时使用（原始动作本身不安全）。
- 默认跳过 `best.cost > 500`、`min_center < 2.5`、`min_sep < 1.8` 的标签，
  避免把近失/不稳定 replay 当成专家。

## 4. 小模型 baseline

```bash
python finetune/train_mlp.py --data dataset/sft --out dataset/sft/mlp
```

只用 numpy，能在 CARLA 的 py3.8 venv 里跑。它回答一个关键问题：
**这个低维战术任务是否真的需要 LLM？** 如果 MLP 已经达到同样安全/圈速，
论文的卖点应转向“解释/泛化/新场景”，而不是“LLM 更准”。

当前 pilot（3 条 clean episode、12 个标签、11 个可用）的 MLP 结果：

```
train acc=0.625  safety_cost=100
val   acc=0.600  safety_cost=92
test  acc=0.671  safety_cost=67
```

这个数字**没有论文意义**：数据只有 3 条 episode，而且类别极不平衡。
它只证明管线能跑通。

## 5. QLoRA SFT / DPO

训练环境必须是 Python 3.10+，不要用 CARLA 的 py3.8 venv：

```bash
python3.10 -m venv ~/venv-llm
source ~/venv-llm/bin/activate
pip install -r finetune/requirements-finetune.txt

python finetune/train_lora.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data dataset/sft --out checkpoints/qwen3b-sft

python finetune/train_dpo.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data dataset/sft/dpo.jsonl --out checkpoints/qwen3b-dpo
```

8 GB 显存建议：3B + 4-bit QLoRA，seq 768~1024，batch 1，grad accumulation 8。
7B 可以试，但需要更小的 seq / 更严的显存设置。

## 6. 闭环评测

用 vLLM 起 OpenAI 兼容服务（见 `serve_vllm.md`），然后：

```bash
export USE_LLM_DECISION=1 LLM_BACKEND=openai
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_API_KEY=dummy LLM_MODEL=sft LLM_PROXY= LLM_TIMEOUT=10
/home/yjh/carla-venv/bin/python dataset_runner.py --policy openai --episodes 20 \
  --out dataset/eval_sft
```

对比：no-LLM / stub / 零样本 API LLM / SFT / SFT+DPO / MLP。
每个策略至少 100 episodes（论文），pilot 阶段 20 条即可看趋势。
必须同时报告 **raw policy** 和 **shielded policy**，否则安全层会掩盖失败。

## 7. 下一步（数据规模）

- 500+ clean episodes（约 10 万条决策），其中 1~2 万条带反事实标签。
- 场景随机化：gap 8~60m、lead factor 0.85~0.95、ego 1.00~1.06、
  constant/variable/defensive、湿滑/低摩擦、多弯。
- 失败/近失：random 或低 offset 策略 + `--stop-on-collision`，
  每类保留 10~20%；需要真实失败时用 `--no-tick-shield` 关闭每 tick shield。
- 前车也要有 car-following / 避撞；否则 44m 这类失败会被误记成策略问题。
- 平衡 hold/overtake：当前 pilot 的 hold 占比仍高，训练时用 class weight
  或对 hold 下采样。
- 先跑 MLP，再跑 3B SFT，最后 DPO；不要一上来就 7B。


---

## 8. 本机 bf16 LoRA（Qwen2.5-1.5B，8GB）

当前环境：PyPI 走清华镜像可用，ModelScope 可用，HuggingFace 被重置。
因此用 ModelScope 下载 Qwen2.5-1.5B-Instruct，bf16 LoRA（1.5B 不需要 4-bit）。

```bash
# 1) 独立训练 venv（不要用 CARLA 的 py3.8 venv）
python3.11 -m venv /home/yjh/venv-racingllm
source /home/yjh/venv-racingllm/bin/activate
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890
pip install -r finetune/requirements-local-lora.txt   -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 从 ModelScope 下载模型（HF 不可达）
cd /home/yjh/snap/dsh-linux/1/dsh-haness-ws/overtake_demo
./run_llm.sh finetune/download_model.py --cache models
# 输出 MODEL_PATH=models/Qwen/Qwen2.5-1.5B-Instruct

# 3) 本机 LoRA SFT
./run_llm.sh finetune/train_lora_local.py   --model models/Qwen/Qwen2.5-1.5B-Instruct   --data dataset/sft   --out checkpoints/qwen1.5b-sft   --epochs 3 --max-len 512 --batch 1 --grad-accum 8
```

说明：
- `train_lora_local.py` 用 completion-only loss（prompt mask），不依赖 TRL 版本；
- pilot 只有 210 条 SFT / 3 条 episode，训练会过拟合，只验证本机链路；
- 3B 可改用 4-bit QLoRA（`finetune/train_lora.py`），但 8GB 更紧；
- 训练后可用 `finetune/serve_vllm.md` 起服务，或先用 transformers 生成几个样例。
