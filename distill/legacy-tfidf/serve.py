"""Step 3 — serve the distilled head locally as a plain HTTP endpoint.

This student is non-autoregressive (TF-IDF + linear heads): no prefill/decode,
so it does not need SGLang. One `/decide` call = one forward pass.
"""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer

from common import DEPARTMENTS, vectorize

MODEL = None
VOCAB_INDEX = None

def load(path):
    global MODEL, VOCAB_INDEX
    with open(path, "r", encoding="utf-8") as f:
        MODEL = json.load(f)
    VOCAB_INDEX = {t: i for i, t in enumerate(MODEL["vocab"])}

def dot(x, w):
    return sum(a * b for a, b in zip(x, w))

def clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def predict(text):
    v = vectorize(text, VOCAB_INDEX, MODEL["idf"])
    dim = len(MODEL["idf"])

    # department: softmax head with temperature
    W = MODEL["dept"]["W"]; b = MODEL["dept"]["b"]; temp = MODEL["dept"]["temp"]
    C = len(b)
    logits = [b[c] + sum(v[i] * W[i][c] for i in range(dim)) for c in range(C)]
    logits = [l / temp for l in logits]
    mx = max(logits)
    ex = [math.exp(l - mx) for l in logits]
    ps = [e / sum(ex) for e in ex]
    choice = DEPARTMENTS[max(range(C), key=lambda c: ps[c])]

    # urgency: linear 0..2
    u = clip(dot(v, MODEL["urgency"]["w"]) + MODEL["urgency"]["b"], 0.0, 2.0)

    # churn: logistic head with temperature
    z = (dot(v, MODEL["churn"]["w"]) + MODEL["churn"]["b"]) / MODEL["churn"]["temp"]
    z = max(-30.0, min(30.0, z))
    churn = 1.0 / (1.0 + math.exp(-z))

    return {
        "department": {
            "choice": choice,
            "confidence": round(max(ps), 4),
            "probs": {DEPARTMENTS[c]: round(ps[c], 4) for c in range(C)},
        },
        "urgency": round(u, 3),
        "churn": round(churn, 4),
    }

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "model": "distilled-triage-head"})
        else:
            self._send(200, {
                "usage": 'POST /decide with {"text": "..."}',
                "example": {"text": "I was charged twice and want to cancel."},
            })

    def do_POST(self):
        if self.path != "/decide":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            self._send(200, predict(req.get("text", "")))
        except Exception as e:
            self._send(400, {"error": str(e)})

    def log_message(self, *args):
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distill_model.json")
    ap.add_argument("--port", type=int, default=8642)
    args = ap.parse_args()
    load(args.model)
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print("serving distilled head on http://127.0.0.1:%d (no prefill/decode)" % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()