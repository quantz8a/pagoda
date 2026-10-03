"""Step 1 — label raw tickets with any OpenAI-compatible teacher LLM.

Env:
  TEACHER_BASE_URL  https://api.openai.com/v1  (or your private gateway)
  TEACHER_API_KEY   sk-...
  TEACHER_MODEL     gpt-4o-mini

The teacher is only asked to emit *typed judgments with probabilities*
(choice / score / noul), not free-form prose. Those calibrated outputs are
the soft labels the student will fit.
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

from common import DEPARTMENTS

SYSTEM = (
    "You are a support-ticket triage assistant. Reply with ONLY a JSON object "
    "and no other text. The JSON must be exactly:\n"
    '{"department":{"choice":"<label>","probs":{dept to 1}},'
    '"urgency":<0..2>,"churn":<0..1>}\n'
    "department labels: " + ", ".join(DEPARTMENTS) + ".\n"
    'probs: a probability distribution over the FOUR labels that sums to 1.\n'
    "urgency: 0 = not urgent, 1 = normal, 2 = blocking.\n"
    "churn: probability 0..1 that this customer is about to cancel or leave."
)

def call_teacher(base, key, model, text, retries=3):
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Ticket: " + text},
        ],
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return extract_json(content)
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")

def extract_json(content):
    content = re.sub(r"```(?:json)?|```", "", content).strip()
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError("no JSON found in teacher reply: " + content[:200])
    return json.loads(m.group(0))

def to_row(tid, text, out):
    dp = out.get("department", {}) or {}
    choice = dp.get("choice", DEPARTMENTS[0])
    probs = dp.get("probs", {}) or {}
    probs = {k: max(0.0, min(1.0, float(probs.get(k, 0.0)))) for k in DEPARTMENTS}
    s = float(sum(probs.values())) or 1.0
    probs = {k: v / s for k, v in probs.items()}
    if choice not in DEPARTMENTS or probs.get(choice, 0.0) == 0.0:
        # normalize choice to argmax if the teacher used an unexpected label
        choice = max(probs, key=probs.get)
    return {
        "id": tid,
        "text": text,
        "y_department": probs,
        "y_department_choice": choice,
        "y_urgency": max(0.0, min(2.0, float(out.get("urgency", 1.0)))),
        "y_churn": max(0.0, min(1.0, float(out.get("churn", 0.0)))),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="seed_tickets.jsonl")
    ap.add_argument("--output", default="train.jsonl")
    args = ap.parse_args()

    base = os.environ.get("TEACHER_BASE_URL", "https://api.openai.com/v1")
    key = os.environ.get("TEACHER_API_KEY", "")
    model = os.environ.get("TEACHER_MODEL", "gpt-4o-mini")
    if not key:
        raise SystemExit("TEACHER_API_KEY is not set")

    with open(args.input, "r", encoding="utf-8") as f:
        tickets = [json.loads(l) for l in f if l.strip()]

    rows = []
    for t in tickets:
        out = call_teacher(base, key, model, t["text"])
        rows.append(to_row(t["id"], t["text"], out))
        print(f"[teacher] {t['id']}: dept={rows[-1]['y_department_choice']} "
              f"urgency={rows[-1]['y_urgency']:.2f} churn={rows[-1]['y_churn']:.2f}")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} labels -> {args.output}")

if __name__ == "__main__":
    main()