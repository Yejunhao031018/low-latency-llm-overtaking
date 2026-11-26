#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上层超车决策引擎（LLM 可插拔）

- backend = "stub": 内置规则策略（带滞回），离线验证接口与安全过滤
- backend = "hold": 永远 hold（用于配置失败时的安全回退）
- backend = "openai": OpenAI 兼容 API（OpenAI / DeepSeek / 本地 vLLM）
    环境变量: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
              LLM_PROXY(可选，默认 http://127.0.0.1:7890；设为空串或 "none" 可关闭),
              LLM_TIMEOUT(可选，默认 20s)
- 决策输出统一为 dict:
    {"action": "overtake"|"hold",
     "extra_inside_m": 0.0..3.5,      # 在竞速线基础上再向弯心偏移多少米
     "target_offset_m": -3.5..0.0,    # 兼容旧接口的别名 = -extra_inside_m
     "speed_factor": 0.9..1.1,
     "reason": str}
- 安全过滤：窗口/横向净空/TTC 不满足时强制 hold，参数越界被钳制。
  openai 后端默认异步执行（后台线程），仿真主循环不会被 HTTP 超时阻塞。
"""
import json
import os
import threading
import time
import urllib.request

DECISION_WINDOW = (-220.0, 150.0)  # 相对 T2 apex 的 s 范围（米）
MIN_DECISION_GAP = 0.5             # 决策最小间隔（秒，仿真时间）
MAX_EXTRA_INSIDE = 3.5             # 最大额外内线偏移（米，约一个车道）
MIN_ATTACK_EXTRA = 3.0             # 发起超车必须请求的最小内线偏移（米）
MIN_CLOSE_EXTRA = 2.2              # 8m 内并排所需的最小实测横向净空（米）
MIN_VERY_CLOSE_EXTRA = 2.5         # 5m 内并排所需的最小实测横向净空（米）


def _extract_json(text):
    """提取第一个平衡的 {...}，正确处理字符串里的花括号/转义。"""
    if not isinstance(text, str):
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except Exception:
                            break
        start = text.find("{", start + 1)
    return None


class HoldBackend:
    """配置失败/无 API 时的安全后端：永不发起超车。"""

    name = "hold"

    def decide(self, state):
        return {"action": "hold", "extra_inside_m": 0.0,
                "target_offset_m": 0.0, "speed_factor": 1.0,
                "reason": "LLM unavailable, hold"}


class StubBackend:
    """规则策略（带滞回）：接近 T2 且间距合适时发起超车，并持续攻击直到超过前车。"""

    name = "stub"

    def __init__(self):
        self.attacking = False

    def decide(self, state):
        gap = state["gap_m"]                 # 绝对间距
        ahead = state.get("ahead_m", -gap)   # 正 = 自车已在前
        d_apex = state["d_apex_m"]
        # 攻击中：持续到超过前车 15m 或离开 T2 区域
        if self.attacking:
            if ahead > 10.0 or d_apex > 150.0 or d_apex < -300.0:
                self.attacking = False
                return {"action": "hold", "extra_inside_m": 0.0,
                        "target_offset_m": 0.0, "speed_factor": 1.0,
                        "reason": "attack finished (ahead=%.0fm)" % ahead}
            return {"action": "overtake", "extra_inside_m": 3.5,
                    "target_offset_m": -3.5, "speed_factor": 1.05,
                    "reason": "attacking, ahead=%.0fm d_apex=%.0fm" % (ahead, d_apex)}
        # 非攻击状态
        if d_apex < -220.0:
            return {"action": "hold", "extra_inside_m": 0.0,
                    "target_offset_m": 0.0, "speed_factor": 1.0,
                    "reason": "too far from turn 2"}
        if gap > 60.0:
            return {"action": "hold", "extra_inside_m": 0.0,
                    "target_offset_m": 0.0, "speed_factor": 1.0,
                    "reason": "gap too large, catch up first"}
        if gap < 6.0:
            return {"action": "hold", "extra_inside_m": 0.0,
                    "target_offset_m": 0.0, "speed_factor": 1.0,
                    "reason": "too close, avoid contact"}
        if self._curve(state):
            return {"action": "hold", "extra_inside_m": 0.0,
                    "target_offset_m": 0.0, "speed_factor": 1.0,
                    "reason": "curvature too high before attack"}
        self.attacking = True
        return {"action": "overtake", "extra_inside_m": 3.5,
                "target_offset_m": -3.5, "speed_factor": 1.05,
                "reason": "gap %.0fm, d_apex %.0fm: inside line + late brake" % (gap, d_apex)}

    @staticmethod
    def _curve(state):
        return abs(state.get("kappa", 0.0)) > 1.0 / 120.0


class OpenAIBackend:
    """OpenAI 兼容 chat/completions 后端（可走本地 Clash 代理）。"""

    name = "openai"

    def __init__(self):
        self.base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.proxy = os.environ.get("LLM_PROXY", "http://127.0.0.1:7890")
        if self.proxy.strip().lower() in ("", "none", "off"):
            self.proxy = ""
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "20"))
        if not self.key:
            raise RuntimeError("LLM_API_KEY 未设置")

    def _post(self, url, data):
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.key,
        })
        if self.proxy:
            handlers = [urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy})]
        else:
            handlers = []
        opener = urllib.request.build_opener(*handlers)
        with opener.open(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def decide(self, state):
        prompt = self._prompt(state)
        url = self.base + "/chat/completions"
        resp = self._post(url, {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 300,
            "messages": [
                {"role": "system",
                 "content": "You are a racing strategy decision maker. "
                 "Answer ONLY a JSON object with keys: "
                 "action (overtake|hold), "
                 "extra_inside_m (float, 0..3.5; 0 = stay on the racing line, "
                 "positive = further toward the inside of the corner), "
                 "speed_factor (float, 0.9..1.1), "
                 "reason (short text)."},
                {"role": "user", "content": prompt},
            ],
        })
        try:
            text = resp["choices"][0]["message"]["content"]
        except Exception:
            return {"action": "hold", "reason": "LLM response malformed"}
        obj = _extract_json(text)
        if not obj:
            return {"action": "hold", "reason": "LLM non-JSON output: %s" % str(text)[:80]}
        return obj

    @staticmethod
    def _prompt(state):
        return (
            "Scenario: highway section approaching turn 2, a %s-hand corner; "
            "the inside of the corner is to the %s.\n"
            "ego speed %.1f m/s, lead speed %.1f m/s.\n"
            "signed gap ahead_m=%.1f m (positive = ego is AHEAD of the lead), "
            "absolute gap %.1f m.\n"
            "distance to turn2 apex %.0f m; track curvature here %.5f (1/m); "
            "TTC %.2f s.\n"
            "current lateral offset from lane centre %.2f m; racing-line offset here %.2f m; "
            "current extra-inside offset %.2f m; measured lateral separation to the lead %.2f m.\n"
            "overtake window: -220 <= distance_to_apex <= +150 m. "
            "For a safe pass within 8 m longitudinally, request at least 3.0 m extra_inside_m (requests below 3.0 m are rejected).\n"
            "If ahead_m > 10 m, hold. "
            "decide overtake or hold; if overtake, give extra_inside_m and speed_factor "
            "relative to the corner speed profile."
        ) % (state.get("inside_side", "right"), state.get("inside_side", "right"),
             state["ego_v"], state["lead_v"],
             state.get("ahead_m", -state["gap_m"]), state["gap_m"],
             state["d_apex_m"], state["kappa"], state.get("ttc", 1e9),
             state.get("actual_off_m", 0.0), state.get("racing_off_m", 0.0),
             state.get("actual_extra_m", 0.0), state.get("lateral_sep_m", 0.0))


class LoraBackend:
    """本地 LoRA server 后端：直接把 JSON state 作为 messages[-1].content 发送。

    lora_server.py 的 HTTP 端点会 `json.loads(body["messages"][-1]["content"])`
    拿到 state 后自己组 observation，因此这里不能发自然语言 prompt，必须发 JSON state。
    """

    name = "lora"

    def __init__(self):
        self.base = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8765/v1").rstrip("/")
        self.model = os.environ.get("LLM_MODEL", "qwen1.5b-sft-lora")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "5.0"))

    def _post(self, url, data):
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
        })
        opener = urllib.request.build_opener()
        with opener.open(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def decide(self, state):
        url = self.base + "/chat/completions"
        resp = self._post(url, {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 64,
            "messages": [
                {"role": "system", "content": "racing tactical planner"},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
            ],
        })
        try:
            text = resp["choices"][0]["message"]["content"]
        except Exception:
            return {"action": "hold", "reason": "LLM response malformed"}
        obj = _extract_json(text)
        if not obj:
            return {"action": "hold", "reason": "non-JSON: %s" % str(text)[:80]}
        return obj


class DecisionEngine:
    """决策引擎：stub/hold 同步执行；openai 默认后台线程异步执行。"""

    def __init__(self, backend="stub", async_mode=None):
        requested = backend
        self.backend_name = requested
        try:
            if requested == "openai":
                self.backend = OpenAIBackend()
            elif requested == "lora":
                self.backend = LoraBackend()
            elif requested == "hold":
                self.backend = HoldBackend()
            elif requested == "stub":
                self.backend = StubBackend()
            else:
                raise ValueError("unknown LLM backend: %s" % requested)
        except Exception as e:
            print("[LLM] backend '%s' init failed, fallback to hold: %s" % (requested, e))
            self.backend = HoldBackend()
            self.backend_name = "hold(fallback)"
        if async_mode is None:
            self.async_mode = self.backend_name == "openai"
        else:
            self.async_mode = bool(async_mode) and self.backend_name == "openai"

        self.last_t = -1e9
        self.last_decision = {"action": "hold", "extra_inside_m": 0.0,
                              "target_offset_m": 0.0, "speed_factor": 1.0,
                              "reason": "not yet"}
        self.history = []
        self._cv = threading.Condition()
        self._pending = False
        self._state = None
        self._stop = False
        self._thread = None
        if self.async_mode:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def decide(self, t, state):
        """返回生效决策（含安全过滤）。state 必须包含 gap_m/d_apex_m/ego_v/lead_v/kappa/ttc。"""
        if t - self.last_t < MIN_DECISION_GAP:
            return self.last_decision
        self.last_t = t
        if self.async_mode:
            st = dict(state)
            st["_t"] = t
            with self._cv:
                self._state = st
                self._pending = True
                self._cv.notify()
            return self.last_decision
        raw = self._call_backend(state)
        dm = self._filter(state, raw)
        dm["backend"] = self.backend_name
        dm["t"] = round(t, 2)
        self.last_decision = dm
        self.history.append(dm)
        return dm

    def _call_backend(self, state):
        try:
            return self.backend.decide(state)
        except Exception as e:
            return {"action": "hold", "reason": "backend error: %s" % str(e)[:120]}

    def _worker(self):
        while True:
            with self._cv:
                while not self._pending and not self._stop:
                    self._cv.wait(timeout=0.5)
                if self._stop:
                    return
                state = self._state
                self._pending = False
            t0 = time.monotonic()
            raw = self._call_backend(state)
            latency = time.monotonic() - t0
            dm = self._filter(state, raw)
            dm["backend"] = self.backend_name
            dm["t"] = round(state.get("_t", -1.0), 2)
            dm["latency_s"] = round(latency, 3)
            self.last_decision = dm
            self.history.append(dm)

    def close(self):
        if self._thread is None:
            return
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=1.0)

    @staticmethod
    def _filter(state, raw):
        d = dict(raw) if isinstance(raw, dict) else {}
        action = str(d.get("action", "hold")).lower()
        reason = str(d.get("reason", ""))[:160]
        if action not in ("overtake", "hold"):
            action = "hold"
            reason = "invalid action -> hold; " + reason
        requested_action = action

        extra = d.get("extra_inside_m", None)
        if extra is None:
            try:
                extra = -float(d.get("target_offset_m", 0.0))
            except Exception:
                extra = 0.0
        try:
            extra = float(extra)
        except Exception:
            extra = 0.0
        extra = max(0.0, min(MAX_EXTRA_INSIDE, extra))

        try:
            fac = float(d.get("speed_factor", 1.05))
        except Exception:
            fac = 1.05
        fac = max(0.90, min(1.10, fac))

        d_apex = state.get("d_apex_m", 1e9)
        ahead = state.get("ahead_m", -1e9)
        gap = state.get("gap_m", 1e9)
        ttc = state.get("ttc", 1e9)

        if action == "overtake":
            if not (DECISION_WINDOW[0] <= d_apex <= DECISION_WINDOW[1]):
                action = "hold"
                reason = "outside decision window -> hold; " + reason
            elif extra < MIN_ATTACK_EXTRA:
                action = "hold"
                reason = "requested offset < %.1fm is unsafe -> hold; " % MIN_ATTACK_EXTRA + reason
            elif gap < 8.0:
                sep = abs(state.get("lateral_sep_m", 0.0))
                if sep < MIN_CLOSE_EXTRA:
                    action = "hold"
                    reason = "close gap without lateral separation -> hold; " + reason
                elif gap < 5.0 and sep < MIN_VERY_CLOSE_EXTRA:
                    action = "hold"
                    reason = "very close, need >=2.5m separation -> hold; " + reason
                elif ahead < -1.5 and ttc < 0.6:
                    action = "hold"
                    reason = "TTC too small (close, behind) -> hold; " + reason
            elif ttc < 1.5:
                action = "hold"
                reason = "TTC too small -> hold; " + reason

        return {"action": action,
                "blocked": bool(requested_action == "overtake" and action == "hold"),
                "extra_inside_m": round(extra, 2),
                "target_offset_m": round(0.0 if extra == 0.0 else -extra, 2),
                "speed_factor": round(fac, 3),
                "reason": reason}
