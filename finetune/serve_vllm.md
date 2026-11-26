# Serving the fine-tuned adapter for closed-loop evaluation

The existing `llm_decision.py` `openai` backend speaks the OpenAI
`/chat/completions` protocol, so the simplest path is vLLM's OpenAI server.
No code change is needed in `race_overtake.py` / `dataset_runner.py`.

## 1. Install vLLM in the training env

```bash
python -m venv ~/venv-llm
source ~/venv-llm/bin/activate
pip install -r finetune/requirements-finetune.txt
pip install vllm
```

## 2. Serve the base model + LoRA adapter

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --enable-lora \
  --lora-modules sft=checkpoints/qwen3b-sft \
  --max-model-len 2048 \
  --port 8000
```

If you trained with DPO on top of the SFT adapter, merge first or register the
DPO adapter as another module (`--lora-modules dpo=checkpoints/qwen3b-dpo`).

Merge manually (optional):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base, "checkpoints/qwen3b-sft")
model = model.merge_and_unload()
model.save_pretrained("checkpoints/qwen3b-sft-merged")
```

## 3. Run the closed-loop evaluator

```bash
export USE_LLM_DECISION=1 LLM_BACKEND=openai
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_API_KEY=dummy
export LLM_MODEL=sft
export LLM_PROXY=          # disable the Clash proxy for localhost
export LLM_TIMEOUT=10

/home/yjh/carla-venv/bin/python race_overtake.py
# or dataset_runner.py --policy openai --episodes N ...
```

The safety shield still runs independently of the model.  Report both
`raw policy` and `shielded policy` metrics in the paper.

## 4. Notes

- 8 GB VRAM: 3B 4-bit is comfortable; 7B 4-bit + LoRA is tight but possible
  with `--max-model-len 1024`, `--gpu-memory-utilization 0.90`.
- Keep `temperature=0` for evaluation reproducibility.
- vLLM's OpenAI server does not guarantee JSON-only output. The safety layer
  parses the first balanced JSON object and falls back to `hold` on failure;
  for cleaner data, also use a grammar/JSON schema at inference time.
