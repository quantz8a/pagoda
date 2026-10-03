#!/usr/bin/env python3
"""Field-level evaluation: base student vs distilled student vs teacher gold.

Runs generation in-process with transformers (no server needed). Metrics:
  - exact match per structured field (department/urgency/sentiment/order_id/
    refund_amount/needs_human)
  - reply format compliance (non-empty, <=120 chars)
  - full-JSON exact match (all fields at once)

Usage:
  python eval.py --base Qwen/Qwen2.5-0.5B-Instruct --merged out/student-merged \
      --data dataset.jsonl --report eval_report.json
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gen_data import extract_json, validate_gold
from train_lora import PROMPT

FIELDS = ["department", "urgency", "sentiment", "order_id", "refund_amount", "needs_human"]


def gen_json(model, tok, email, max_new=320):
    messages = [{"role": "user", "content": PROMPT.replace("{email}", email)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    raw = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        return validate_gold(extract_json(raw)), raw
    except Exception:
        return None, raw


def score(pred, gold):
    per = {}
    for f in FIELDS:
        per[f] = int(pred is not None and pred.get(f) == gold.get(f))
    per["reply_ok"] = int(pred is not None and isinstance(pred.get("reply"), str)
                          and 0 < len(pred["reply"]) <= 120)
    per["json_exact"] = int(all(per[f] for f in FIELDS) and per["reply_ok"])
    return per


def run_model(path, eval_rows, tag):
    print(f"\n== evaluating {tag}: {path}")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    agg = {k: 0 for k in list(FIELDS) + ["reply_ok", "json_exact"]}
    shown = 0
    for i, r in enumerate(eval_rows):
        pred, raw = gen_json(model, tok, r["email"])
        s = score(pred, r["gold"])
        for k, v in s.items():
            agg[k] += v
        if shown < 3:
            print(f"  [{tag} sample] dept: pred={pred and pred['department']} gold={r['gold']['department']}")
            shown += 1
    del model
    torch.cuda.empty_cache()
    n = len(eval_rows)
    return {k: round(v / n, 4) for k, v in agg.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--merged", required=True)
    ap.add_argument("--data", default="dataset.jsonl")
    ap.add_argument("--report", default="eval_report.json")
    args = ap.parse_args()

    eval_rows = [json.loads(l) for l in open(args.data, encoding="utf-8")
                 if json.loads(l)["split"] == "eval"]
    print(f"eval examples: {len(eval_rows)}")

    base_acc, n = run_model(args.base, eval_rows, "base")
    dist_acc, _ = run_model(args.merged, eval_rows, "distilled")

    keys = list(FIELDS) + ["reply_ok", "json_exact"]
    report = {"n_eval": n,
              "base": base_acc, "distilled": dist_acc,
              "lift": {k: round(dist_acc[k] - base_acc[k], 4) for k in keys}}
    print("\nfield                base   distilled   lift")
    for k in keys:
        print(f"  {k:<16} {base_acc[k]:>6}  {dist_acc[k]:>9}  {report['lift'][k]:+>7}")
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()
