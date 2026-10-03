#!/usr/bin/env python3
"""Teacher data engine: seed tickets -> teacher-labeled JSON dataset.

Calls any OpenAI-compatible chat API (paid: OpenAI/DeepSeek/private; local:
SGLang's OpenAI server). Two phases:

1. answer: teacher extracts structured fields + drafts a reply for each seed
2. vary:   teacher rewrites each seed K times (new wording / order id /
           amount, same underlying issue), then answers each variation

Every teacher output is validated against the business schema; failures are
dropped and counted. Output: dataset.jsonl with {email, gold, seed_id, split}.
Splits are by seed group so variations never leak across train/eval.

Env:
  TEACHER_BASE_URL  e.g. https://api.openai.com/v1 or http://127.0.0.1:30001/v1
  TEACHER_API_KEY   paid key, or EMPTY for a local server
  TEACHER_MODEL     e.g. gpt-4o-mini, deepseek-chat, Qwen/Qwen2.5-3B-Instruct-AWQ
"""
import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

SCHEMA_HINT = """{
  "department": "billing|shipping|technical|product|other",
  "urgency": 0,
  "sentiment": "angry|neutral|positive",
  "order_id": "订单号字符串或null",
  "refund_amount": "数字或null（仅客户明确要求退赔的具体金额，元）",
  "needs_human": false,
  "reply": "不超过80字的中文客服回复草稿，共情+行动方案+时限"
}"""

DEPARTMENTS = {"billing", "shipping", "technical", "product", "other"}
SENTIMENTS = {"angry", "neutral", "positive"}

ANSWER_SYS = (
    "你是电商客服工单结构化专家。从客户消息中提取信息，只输出 JSON，不要任何解释。\n"
    "字段定义与取值范围严格遵守：\n" + SCHEMA_HINT + "\n"
    "规则：urgency 0=不紧急 1=一般 2=紧急；客户在气头上/有拒付、投诉、曝光威胁时 angry；"
    "涉及纠纷、投诉、人身威胁或客户明确要求人工时 needs_human=true；"
    "没有提到订单号/退款金额时用 null。"
)

VARY_SYS = (
    "你是电商客服工单改写器。把给定的客户工单改写成 K 个不同版本：换措辞、换订单号"
    "（格式 ORD-xxxxx，数字随机）、换具体金额/日期/姓名等细节，但**保持业务问题类型不变**"
    "（该投诉账单还是投诉账单，该问物流还是问物流）。只输出 JSON 数组，每个元素是一封"
    "改写后的工单文本字符串，不要任何解释。"
)


def extract_json(text: str):
    """Best-effort JSON extraction: strip fences, take the first {...} or [...]."""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"[\{\[].*[\}\]]", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"no JSON found in: {text[:120]!r}")


def validate_gold(g: dict) -> dict:
    """Return a normalized gold record, or raise ValueError."""
    out = {}
    out["department"] = str(g["department"])
    if out["department"] not in DEPARTMENTS:
        raise ValueError(f"bad department {out['department']!r}")
    out["urgency"] = int(g["urgency"])
    if out["urgency"] not in (0, 1, 2):
        raise ValueError("urgency must be 0|1|2")
    out["sentiment"] = str(g["sentiment"])
    if out["sentiment"] not in SENTIMENTS:
        raise ValueError(f"bad sentiment {out['sentiment']!r}")
    oid = g.get("order_id")
    out["order_id"] = str(oid) if oid not in (None, "", "null") else None
    amt = g.get("refund_amount")
    out["refund_amount"] = float(amt) if amt not in (None, "", "null") else None
    out["needs_human"] = bool(g["needs_human"])
    reply = str(g["reply"]).strip()
    if not reply or len(reply) > 120:
        raise ValueError(f"bad reply length {len(reply)}")
    out["reply"] = reply
    return out


class Teacher:
    def __init__(self):
        base_url = os.environ.get("TEACHER_BASE_URL", "http://127.0.0.1:30001/v1")
        api_key = os.environ.get("TEACHER_API_KEY", "EMPTY")
        self.model = os.environ.get("TEACHER_MODEL")
        if not self.model:
            raise SystemExit("set TEACHER_MODEL (e.g. gpt-4o-mini, deepseek-chat, or a local repo id)")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=180)

    def chat(self, sys_prompt: str, user: str, max_tokens: int = 700, temperature: float = 0.3) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def answer(self, email: str) -> dict:
        raw = self.chat(ANSWER_SYS, "客户消息：\n" + email, temperature=0.0)
        return validate_gold(extract_json(raw))

    def variations(self, email: str, k: int) -> list:
        raw = self.chat(VARY_SYS, f"K={k}。原工单：\n{email}", max_tokens=1500, temperature=0.9)
        arr = extract_json(raw)
        if not isinstance(arr, list):
            raise ValueError("variations must be a JSON array")
        out = []
        for v in arr:
            v = str(v).strip()
            if 30 <= len(v) <= 600:
                out.append(v)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="seeds/tickets.jsonl")
    ap.add_argument("--out", default="dataset.jsonl")
    ap.add_argument("--variations", type=int, default=3)
    ap.add_argument("--eval-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seeds = [json.loads(l) for l in open(args.seeds, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    seed_ids = [s["id"] for s in seeds]
    rng.shuffle(seed_ids)
    n_eval = max(1, round(len(seed_ids) * args.eval_ratio))
    eval_seeds = set(seed_ids[:n_eval])

    # SGLang batches concurrent requests, which hides its per-token CPU
    # dispatch cost on shared boxes; a few parallel seeds speed the data
    # engine up several-fold without changing any output content.
    teacher = Teacher()
    workers = int(os.environ.get("GEN_WORKERS", "6"))
    n_written = n_dropped = 0
    lock = threading.Lock()

    def process_seed(i, seed):
        split = "eval" if seed["id"] in eval_seeds else "train"
        candidates = [seed["email"]]
        try:
            vars_ = teacher.variations(seed["email"], args.variations)
            candidates += vars_
            print(f"[{i+1}/{len(seeds)}] {seed['id']}: {len(vars_)} variations", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(seeds)}] {seed['id']}: variation failed: {e}", flush=True)
        records, dropped = [], 0
        for email in candidates:
            try:
                gold = teacher.answer(email)
            except Exception as e:
                dropped += 1
                print(f"    answer dropped: {e}", flush=True)
                continue
            records.append({"email": email, "gold": gold,
                            "seed_id": seed["id"], "split": split})
        return records, dropped

    with open(args.out, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_seed, i, s) for i, s in enumerate(seeds)]
            for fut in as_completed(futures):
                records, dropped = fut.result()
                with lock:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    n_written += len(records)
                    n_dropped += dropped
    print(f"\ndone: {n_written} examples written, {n_dropped} dropped -> {args.out}")


if __name__ == "__main__":
    main()
