# Model Weights — Qwen2.5-1.5B-short (LoRA adapter)

The tactical decision model used in the paper. This is a **LoRA adapter** on top
of [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
(the base model is on Hugging Face; we only distribute the adapter).

| Metric | Value |
|---|---|
| Decision latency | **0.59 s** (8 GB GPU, greedy) |
| Decision accuracy | **86.1%** |
| Output | short JSON (no `reason` field) |
| Fine-tune | LoRA r=16, α=32, bf16, 2 epochs |

## Files

```
weights/qwen1.5b-sft-short-lr/
├── adapter_model.safetensors   # LoRA weights (74 MB)
└── adapter_config.json         # LoRA config
```

## How to load

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "Qwen/Qwen2.5-1.5B-Instruct"   # base model from Hugging Face
base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(model_id)
model = PeftModel.from_pretrained(base, "weights/qwen1.5b-sft-short-lr")
model.eval()
```

Or serve it with the bundled server:

```bash
python src/finetune/lora_server.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter weights/qwen1.5b-sft-short-lr \
  --port 8765 --short
```

## Input / output format

The model maps a JSON observation (ego/lead speeds, gap, distance-to-apex,
curvature, TTC, lateral offset/separation, inside side) to a structured intent:

```json
{"action": "overtake", "extra_inside_m": 3.5, "speed_factor": 1.05}
```

See `paper/paper_ieee.tex` §3 for the full method.
