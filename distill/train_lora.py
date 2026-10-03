#!/usr/bin/env python3
"""LoRA SFT for the student model on teacher-labeled ticket data.

Completion-only loss: prompt tokens are masked with -100, the model learns
to emit exactly the gold JSON (chat-template aware, works with Qwen2.5 and
similar instruct models).

Usage:
  python train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --data dataset.jsonl \
      --out out/student --epochs 3
"""
import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model

PROMPT = (
    "你是电商客服工单结构化助手。阅读客户消息，提取信息并只输出 JSON：\n"
    '{"department": "billing|shipping|technical|product|other", '
    '"urgency": 0|1|2, "sentiment": "angry|neutral|positive", '
    '"order_id": "订单号或null", "refund_amount": "数字或null", '
    '"needs_human": true|false, "reply": "不超过80字的中文回复草稿"}\n'
    "客户消息：\n{email}"
)


class TicketData(Dataset):
    def __init__(self, path, tokenizer, max_len=768):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["split"] != "train":
                    continue
                self.rows.append(r)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        messages = [{"role": "user", "content": PROMPT.replace("{email}", r["email"])}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = json.dumps(r["gold"], ensure_ascii=False) + self.tok.eos_token
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_len]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * (width - n))
        labels.append(b["labels"] + [-100] * (width - n))
        attn.append([1] * n + [0] * (width - n))
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="dataset.jsonl")
    ap.add_argument("--out", default="out/student")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    data = TicketData(args.data, tok)
    print(f"train examples: {len(data)}")

    targs = TrainingArguments(
        output_dir=args.out + "-ckpt",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to=[],
        seed=42,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=data,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()

    adapter_dir = args.out + "-lora"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"adapter saved -> {adapter_dir}")

    print("merging LoRA into base weights ...")
    merged = model.merge_and_unload()
    merged_dir = args.out + "-merged"
    merged.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    print(f"merged model saved -> {merged_dir}")


if __name__ == "__main__":
    main()
