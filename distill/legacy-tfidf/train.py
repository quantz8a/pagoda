"""Step 2 — distill the teacher's calibrated soft labels into a tiny
non-autoregressive decision head (TF-IDF + linear heads) in numpy.

The student never generates text; it predicts the three typed answers
(department probs, urgency score, churn probability). No prefill/decode.
"""
import argparse
import json
import math
import random

import numpy as np

from common import DEPARTMENTS, vectorize, tokenize

class Scaler:
    # logistic regression + temperature, trained against *soft* targets
    def __init__(self, dim, out=1, kind="logistic"):
        rng = np.random.default_rng(3)
        self.kind = kind
        if kind == "softmax":
            self.W = rng.normal(0, 0.02, (dim, out)).astype(np.float32)
            self.b = np.zeros(out, dtype=np.float32)
        else:
            self.w = rng.normal(0, 0.02, (dim,)).astype(np.float32)
            self.b = np.float32(0.0)
        self.temp = np.float32(1.0)

    def logits(self, X):
        if self.kind == "softmax":
            return X @ self.W + self.b
        return X @ self.w + self.b

    def fit(self, X, Y, epochs=400, lr=0.5, l2=1e-3):
        n = X.shape[0]
        if self.kind == "softmax":
            for _ in range(epochs):
                z = X @ self.W + self.b
                z = z - z.max(axis=1, keepdims=True)
                p = np.exp(z)
                p = p / p.sum(axis=1, keepdims=True)
                g = (p - Y) / n
                gW = X.T @ g + l2 * self.W
                gb = g.sum(axis=0)
                self.W -= lr * gW
                self.b -= lr * gb
        elif self.kind == "logistic":
            y = Y.reshape(-1)
            for _ in range(epochs):
                z = X @ self.w + self.b
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                gw = (X.T @ (p - y)) / n + l2 * self.w
                gb = np.mean(p - y)
                self.w -= lr * gw
                self.b -= lr * gb
        else:  # linear score 0..2
            y = Y.reshape(-1)
            for _ in range(epochs):
                z = X @ self.w + self.b
                gw = (X.T @ (z - y)) / n + l2 * self.w
                gb = np.mean(z - y)
                self.w -= lr * gw
                self.b -= lr * gb

        if self.kind in ("softmax", "logistic"):
            self.temp = self._fit_temp(X, Y)
        return self

    def _fit_temp(self, X, Y):
        L = self.logits(X)
        best_t, best_loss = 1.0, 1e18
        for t in np.arange(0.5, 2.01, 0.1):
            if self.kind == "softmax":
                z = (L - L.max(axis=1, keepdims=True)) / t
                p = np.exp(z)
                p = p / p.sum(axis=1, keepdims=True)
                loss = -float(np.mean(np.sum(Y * np.log(np.clip(p, 1e-9, 1.0)), axis=1)))
            else:
                z = np.clip(L / t, -30, 30)
                p = 1.0 / (1.0 + np.exp(-z))
                y = Y.reshape(-1)
                loss = -float(np.mean(y * np.log(np.clip(p, 1e-9, 1.0)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1.0))))
            if loss < best_loss:
                best_loss, best_t = loss, t
        return np.float32(best_t)

    def to_json(self):
        if self.kind == "softmax":
            return {"kind": self.kind, "W": self.W.tolist(), "b": self.b.tolist(), "temp": float(self.temp)}
        return {"kind": self.kind, "w": self.w.tolist(), "b": float(self.b), "temp": float(self.temp)}

def build_features(texts, max_features=2000):
    docs = []
    df = {}
    for t in texts:
        toks = list(dict.fromkeys(tokenize(t)))
        docs.append(toks)
        for tok in toks:
            df[tok] = df.get(tok, 0) + 1
    n = len(texts)
    vocab = sorted(df, key=lambda t: (-df[t], t))[:max_features]
    vocab_index = {t: i for i, t in enumerate(vocab)}
    idf = [math.log((1 + n) / (1 + df[t])) for t in vocab]
    X = np.array([vectorize(t, vocab_index, idf) for t in texts], dtype=np.float32)
    return X, vocab, idf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="train.jsonl")
    ap.add_argument("--output", default="distill_model.json")
    ap.add_argument("--metrics", default="metrics.json")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    texts = [r["text"] for r in rows]
    idx = list(range(len(rows)))
    random.seed(7)
    random.shuffle(idx)
    cut = max(1, int(len(rows) * 0.8)) if len(rows) >= 5 else len(rows)
    tr_idx, ev_idx = idx[:cut], idx[cut:]

    X, vocab, idf = build_features(texts)

    Y_dept = np.array([[r["y_department"][d] for d in DEPARTMENTS] for r in rows], dtype=np.float32)
    Y_urg = np.array([r["y_urgency"] for r in rows], dtype=np.float32).reshape(-1, 1)
    Y_churn = np.array([r["y_churn"] for r in rows], dtype=np.float32).reshape(-1, 1)

    dept = Scaler(X.shape[1], len(DEPARTMENTS), "softmax").fit(X[tr_idx], Y_dept[tr_idx])
    urg = Scaler(X.shape[1], 1, "linear").fit(X[tr_idx], Y_urg[tr_idx])
    churn = Scaler(X.shape[1], 1, "logistic").fit(X[tr_idx], Y_churn[tr_idx])

    metrics = {"train_rows": len(tr_idx), "eval_rows": len(ev_idx), "vocab": len(vocab)}
    if ev_idx:
        z = dept.logits(X[ev_idx])
        metrics["department_accuracy"] = float(np.mean(np.argmax(z, axis=1) == np.argmax(Y_dept[ev_idx], axis=1)))
        cz = churn.logits(X[ev_idx]).reshape(-1)
        churn_p = 1.0 / (1.0 + np.exp(-np.clip(cz / float(churn.temp), -30, 30)))
        metrics["churn_mae"] = float(np.mean(np.abs(churn_p - Y_churn[ev_idx].reshape(-1))))
        metrics["urgency_mae"] = float(np.mean(np.abs(urg.logits(X[ev_idx]).reshape(-1) - Y_urg[ev_idx].reshape(-1))))
    print("metrics:", json.dumps(metrics))

    model = {
        "departments": DEPARTMENTS,
        "vocab": vocab,
        "idf": idf,
        "dept": dept.to_json(),
        "urgency": urg.to_json(),
        "churn": churn.to_json(),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(model, f)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote student -> {args.output}")

if __name__ == "__main__":
    main()