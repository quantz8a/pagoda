# Copyright (C) 2026  quantz8a
# SPDX-License-Identifier: Apache-2.0
"""Benchmark harness: HuggingFace transformers (python/torch) CPU reference,
same scenarios as pagoda-hf/examples/bench.rs. Prints one JSON object.

Usage:
    python bench_hf.py --repo TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --threads 6 --prompt-tokens 128 --gen-tokens 32 --batch 8 --repeat 3
"""

import argparse
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SENTENCE = "The quick brown fox jumps over the lazy dog. "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(args.repo)
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.repo, torch_dtype=dtype)
    model.eval()
    model.to(args.device)

    def count(text: str) -> int:
        return len(tok.encode(text))

    prompt = ""
    while count(prompt) < args.prompt_tokens:
        prompt += SENTENCE
    prompt_tokens = count(prompt)

    gen_kwargs = dict(
        max_new_tokens=args.gen_tokens,
        min_new_tokens=args.gen_tokens,  # full-length runs, no early EOS
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )

    def run(texts):
        if isinstance(texts, str):
            texts = [texts]
        tok.padding_side = "left"
        inputs = tok(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        ms = (time.perf_counter() - t0) * 1e3
        return ms, out.shape[1] - inputs["input_ids"].shape[1]

    # Cold.
    cold_ms, _ = run(prompt)

    # Warm: HF transformers has no cross-request prefix cache — warm runs
    # recompute the prompt (an honest contrast to radix/APC engines).
    warm_ms = [run(prompt)[0] for _ in range(args.repeat)]

    # Batch: K distinct prompts (distinct first tokens => no shared trunk).
    prompts = [f"Request number {i}. {prompt}" for i in range(args.batch)]
    batch_ms, per_new = run(prompts)

    warm_sorted = sorted(warm_ms)
    result = {
        "engine": f"hf transformers (torch {torch.__version__}, {args.device} {str(dtype).split('.')[-1]})",
        "torch_threads": torch.get_num_threads(),
        "repo": args.repo,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": args.gen_tokens,
        "cold_ms": round(cold_ms, 2),
        "warm_median_ms": round(warm_sorted[len(warm_sorted) // 2], 2),
        "warm_all_ms": [round(v, 2) for v in warm_ms],
        "batch": {
            "requests": args.batch,
            "wall_ms": round(batch_ms, 2),
            "output_tokens": args.batch * per_new,
            "output_tok_per_s": round(args.batch * per_new / (batch_ms / 1e3), 1),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
