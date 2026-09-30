# Copyright (C) 2026  quantz8a
# SPDX-License-Identifier: AGPL-3.0-only
"""Benchmark harness: SGLang (python) reference, same scenarios as
pagoda-hf/examples/bench.rs. Prints one JSON object to stdout.

Usage:
    python bench_sglang.py --repo hf-internal-testing/tiny-random-LlamaForCausalLM \
        --device cpu --threads 6 --prompt-tokens 128 --gen-tokens 32 --batch 8 --repeat 3
    python bench_sglang.py --repo TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda
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
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--mem-fraction", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import torch

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    import sglang as sgl

    engine_kwargs = dict(
        model_path=args.repo,
        device=args.device,
        log_level="error",
        random_seed=0,
        skip_tokenizer_init=False,
        # The overlap scheduler JIT-compiles a CUDA bookkeeping kernel at
        # startup (needs a modern nvcc); keep runs on the plain scheduler.
        disable_overlap_schedule=True,
        mem_fraction_static=args.mem_fraction,
    )
    if args.attention_backend:
        engine_kwargs["attention_backend"] = args.attention_backend
    if args.device == "cpu":
        engine_kwargs["attention_backend"] = "torch_native"
        # sglang 0.5 touches the CUDA graph memory pool unconditionally;
        # CPU runs must opt out of CUDA graphs explicitly.
        engine_kwargs["disable_cuda_graph"] = True
        engine_kwargs["disable_piecewise_cuda_graph"] = True
    else:
        # CUDA-graph capture JIT-compiles kernels via the system nvcc; this
        # box ships CUDA 10.1 which is too old, so graphs stay off (noted in
        # the benchmark doc as a concession).
        engine_kwargs["disable_cuda_graph"] = True
        engine_kwargs["disable_piecewise_cuda_graph"] = True
    try:
        engine = sgl.Engine(**engine_kwargs)
    except Exception:
        # Fall back to the portable attention backend if the default one
        # (e.g. flashinfer) is unavailable on this GPU/driver combo.
        engine_kwargs["attention_backend"] = "torch_native"
        engine = sgl.Engine(**engine_kwargs)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.repo)

    def count(text: str) -> int:
        return len(tok.encode(text))

    prompt = ""
    while count(prompt) < args.prompt_tokens:
        prompt += SENTENCE
    prompt_tokens = count(prompt)

    # min_new_tokens forces full-length runs (no early EOS) so latency is
    # comparable with pagoda's length-bounded runs.
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": args.gen_tokens,
        "min_new_tokens": args.gen_tokens,
    }

    # Cold: first run of this prompt on a fresh engine.
    t0 = time.perf_counter()
    cold = engine.generate(prompt, sampling)
    cold_ms = (time.perf_counter() - t0) * 1e3
    completion = cold["meta_info"]["completion_tokens"] if isinstance(cold, dict) else None

    # Warm: same prompt, radix cache hot.
    warm_ms = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        engine.generate(prompt, sampling)
        warm_ms.append((time.perf_counter() - t0) * 1e3)

    # Batch: K distinct prompts (distinct first tokens => no shared trunk).
    prompts = [f"Request number {i}. {prompt}" for i in range(args.batch)]
    t0 = time.perf_counter()
    outs = engine.generate(prompts, sampling)
    batch_ms = (time.perf_counter() - t0) * 1e3
    total_out = sum(o["meta_info"]["completion_tokens"] for o in outs)

    warm_sorted = sorted(warm_ms)
    result = {
        "engine": f"sglang {sgl.__version__} (torch {torch.__version__}, {args.device})",
        "torch_threads": torch.get_num_threads(),
        "repo": args.repo,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": args.gen_tokens,
        "cold_ms": round(cold_ms, 2),
        "cold_completion_tokens": completion,
        "warm_median_ms": round(warm_sorted[len(warm_sorted) // 2], 2),
        "warm_all_ms": [round(v, 2) for v in warm_ms],
        "batch": {
            "requests": args.batch,
            "wall_ms": round(batch_ms, 2),
            "output_tokens": total_out,
            "output_tok_per_s": round(total_out / (batch_ms / 1e3), 1),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    engine.shutdown()


if __name__ == "__main__":
    main()
