#!/usr/bin/env python3
"""OpenAI-compatible mock for exercising llm_decision.OpenAIBackend.

MOCK_EXTRA controls the requested extra-inside offset (default 3.5).
MOCK_DELAY adds an artificial response delay in seconds (default 0.2).
"""
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

EXTRA = float(os.environ.get("MOCK_EXTRA", "3.5"))
DELAY = float(os.environ.get("MOCK_DELAY", "0.2"))


def decide(prompt):
    m = re.search(r"signed gap ahead_m=(-?[\d.]+) m", prompt)
    d = re.search(r"distance to turn2 apex (-?[\d.]+) m", prompt)
    ahead = float(m.group(1)) if m else -999.0
    d_apex = float(d.group(1)) if d else 9999.0
    attack = (-220.0 <= d_apex <= 150.0) and ahead < 10.0 and EXTRA >= 1.0
    return {
        "action": "overtake" if attack else "hold",
        "extra_inside_m": EXTRA if attack else 0.0,
        "speed_factor": 1.05,
        "reason": "mock: ahead=%.0f d_apex=%.0f extra=%.1f" % (ahead, d_apex, EXTRA),
    }


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        prompt = body.get("messages", [{}, {}])[-1].get("content", "")
        if DELAY > 0:
            time.sleep(DELAY)
        out = decide(prompt)
        raw = json.dumps({"choices": [{"message": {"content": json.dumps(out)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8765), H).serve_forever()
