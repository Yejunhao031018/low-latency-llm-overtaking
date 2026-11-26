#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local LoRA decision server (OpenAI-compatible) for the CARLA closed loop.

Runs in the training venv (py3.11 + torch cu128).  CARLA runs in its own py3.8
venv and talks to this server over HTTP, which is also the intended deployment
shape for the 1/10 car (LLM on a companion compute unit).

    /home/yjh/venv-racingllm/bin/python finetune/lora_server.py \
        --model models/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master \
        --adapter checkpoints/qwen1.5b-sft --port 8765

The server rebuilds the exact training prompt: SYSTEM + compact JSON
observation (see finetune/convert.py).  It does not reuse the verbose English
prompt from llm_decision.py, which would be out of distribution.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_SHORT = (
    "You are a racing tactical planner for a car approaching a corner. "
    "Answer ONLY a JSON object with keys: action (overtake|hold), "
    "extra_inside_m (float, 0..3.5; 0 = racing line, positive = further toward the corner inside), "
    "speed_factor (float, 0.9..1.1). "
    "Never request an inside pass with less than 3.0 m extra offset."
)

SYSTEM = (
    "You are a racing tactical planner for a car approaching a corner. "
    "Answer ONLY a JSON object with keys: action (overtake|hold), "
    "extra_inside_m (float, 0..3.5; 0 = racing line, positive = further toward the corner inside), "
    "speed_factor (float, 0.9..1.1), reason (short text). "
    "Never request an inside pass with less than 3.0 m extra offset."
)

OBS_KEYS = ["ego_v", "lead_v", "ahead_m", "gap_m", "d_apex_m", "kappa", "ttc",
            "actual_off_m", "racing_off_m", "actual_extra_m", "lateral_sep_m",
            "inside_side"]


def _f(x, n=2):
    try:
        return round(float(x), n)
    except Exception:
        return x


def observation(state, lead_policy="constant", prev_action="hold"):
    obs = {k: (_f(state.get(k), 4) if isinstance(state.get(k), float) else state.get(k))
           for k in OBS_KEYS}
    # Tactical state abstraction: while not attacking (prev_action=hold), a
    # small offset from the racing line is the tracking controller's error, not
    # a tactical command.  Feeding it raw makes "attack entry" look identical
    # to "attack exit" (the v2 failure mode).  Clear it when safely behind.
    if (prev_action == "hold"
            and float(state.get("gap_m", 0.0) or 0.0) > 8.0
            and abs(float(obs.get("actual_extra_m") or 0.0)) < 2.0):
        obs["actual_extra_m"] = 0.0
        obs["actual_off_m"] = obs.get("racing_off_m", 0.0)
        obs["lateral_sep_m"] = 0.0
    obs["prev_action"] = prev_action
    obs["scenario"] = {"track": "Town04", "corner": "T2",
                       "lead_policy": state.get("lead_policy", lead_policy)}
    return json.dumps(obs, ensure_ascii=False, separators=(",", ":"))


def parse_json(text):
    start = text.find("{")
    if start < 0:
        return None
    depth, ins, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if ins:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                ins = False
        else:
            if c == '"':
                ins = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        return None
    return None


class LoRAPolicy:
    def __init__(self, model_path, adapter_path, max_new=64,
                 valid_overtake=1.5, valid_hold=1.0, short=False):
        self.short = bool(short)
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            base = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map={"": 0},
                trust_remote_code=True)
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
                trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self.max_new = max_new
        self.valid_overtake = float(valid_overtake)
        self.valid_hold = float(valid_hold)
        self.lock = threading.Lock()
        self.n = 0
        self.prev_action = "hold"

    def decide(self, state):
        # New episode: the car is far before the attack window, so no attack
        # can be in progress.  Reset the feedback state for repeated demos.
        if float(state.get("d_apex_m", 0.0)) < -450.0:
            self.prev_action = "hold"
        # A client may send prev_action explicitly (debug/replay); otherwise
        # use the server's own last decision, which is the deployment path.
        prev = state.get("prev_action") or self.prev_action
        sys = SYSTEM_SHORT if self.short else SYSTEM
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": observation(
                    state, prev_action=prev)}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt",
                       add_special_tokens=False).to("cuda:0")
        t0 = time.perf_counter()
        with self.lock, torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=self.max_new,
                                      do_sample=False,
                                      pad_token_id=self.tok.pad_token_id,
                                      eos_token_id=self.tok.eos_token_id)
        latency = time.perf_counter() - t0
        text = self.tok.decode(gen[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        obj = parse_json(text)
        if obj is None:
            obj = {"action": "hold", "extra_inside_m": 0.0,
                   "speed_factor": 1.0, "reason": "non-JSON: " + text[:80]}
        obj["action"] = str(obj.get("action", "hold")).lower()
        if obj["action"] not in ("overtake", "hold"):
            obj["action"] = "hold"
        try:
            obj["extra_inside_m"] = max(0.0, min(3.5, float(obj.get("extra_inside_m", 0.0))))
        except Exception:
            obj["extra_inside_m"] = 0.0
        try:
            obj["speed_factor"] = max(0.9, min(1.1, float(obj.get("speed_factor", 1.0))))
        except Exception:
            obj["speed_factor"] = 1.0
        obj["valid_for_s"] = (self.valid_overtake if obj["action"] == "overtake"
                              else self.valid_hold)
        obj["inference_s"] = round(latency, 3)
        self.prev_action = obj["action"]
        self.n += 1
        return obj, latency, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--valid-overtake", type=float, default=1.5)
    ap.add_argument("--valid-hold", type=float, default=1.0)
    ap.add_argument("--short", action="store_true",
                    help="use the short system prompt (no reason field)")
    args = ap.parse_args()

    print("[lora-server] loading model ...", flush=True)
    policy = LoRAPolicy(args.model, args.adapter, args.max_new,
                        args.valid_overtake, args.valid_hold, short=args.short)
    print("[lora-server] READY on http://127.0.0.1:%d/v1/chat/completions"
          % args.port, flush=True)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                state = json.loads(body["messages"][-1]["content"])
            except Exception:
                state = {}
            obj, latency, text = policy.decide(state)
            content = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            raw = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "model": "qwen1.5b-sft-lora",
                "usage": {"inference_s": round(latency, 3)},
            }, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            print("[lora-server] #%d %.3fs %s %s" % (
                policy.n, latency, obj["action"], obj.get("reason", "")[:80]), flush=True)

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
