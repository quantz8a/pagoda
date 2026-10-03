# Pagoda

An SGLang-inspired LLM serving runtime, implemented from scratch in Rust with
**zero external dependencies** — it builds and runs fully offline.

> **License: Apache-2.0.** Free to use, study, modify, and redistribute —
> including commercial and closed-source use. Pagoda is built in the same open
> ecosystem as upstream SGLang, with attribution preserved in `NOTICE`.

It distills the core of [SGLang](https://github.com/sgl-project/sglang) into a
compact, testable reference implementation:

- **RadixAttention** prefix cache (compute-skip with hit metrics)
- **APC block cache**: vLLM-style chained block hashing, switchable via
  `EngineConfig.cache_backend`
- **KV checkpoints**: pin a shared prefix (system prompt / agent trunk) and
  branch many generations off it with a guaranteed full hit — the primitive for
  agent-cluster / tree-search workloads (`POST /checkpoint` + friends)
- **Tensor-level prefix grafting** (true RadixAttention): finished sessions' KV
  snapshots live in a bounded cross-request vault; a later request whose prompt
  shares a prefix grafts the cached tensors instead of recomputing them
- **Batched decode**: same-shape sessions advance in one `[B, 1]` forward
  (`decode_batch_factor` observable in `/stats` and the CLI)
- **Continuous batching** scheduler with chunked prefill and a waiting queue
- **Paged KV cache** with reference counting and copy-on-write
- **Sampling**: temperature / top-k / top-p / frequency & presence penalties
- **Constrained decoding**: byte-level regex and JSON grammar `Grammar`, wired
  into the sampler as a logit mask, with zero dependencies
- **SGLang-style frontend language**: `gen` / `select` / `fork`
- **Serving frontend**: OpenAI-compatible chat endpoint + native `/generate`
- **Toy model & tokenizer** stand-ins so the whole pipeline runs without GPU
  (real HuggingFace tokenizer + Candle weights live in the companion `pagoda-hf` crate)

## Quick start

```powershell
cd pagoda

# one-click: build + 93 tests + demos
powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1   # bash: scripts/quickstart.sh

# run the test suite
cargo test --offline

# offline generation — repeat a prompt to watch the prefix cache pay off
cargo run --offline --bin pagoda -- sample -p "SGLang is a serving framework" --max-tokens 40 --repeat 3

# run the DSL demo (gen / select / fork)
cargo run --offline --bin pagoda -- program

# start the HTTP server
cargo run --offline --bin pagoda -- serve --port 8080
```

Then in another shell:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/generate -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"the quick brown fox","sampling_params":{"max_tokens":20}}'
Invoke-RestMethod http://127.0.0.1:8080/stats
```

Checkpoint (agent-style branching):

```powershell
Invoke-RestMethod http://127.0.0.1:8080/checkpoint -Method Post `
  -ContentType "application/json" -Body '{"text":"very long shared system prompt"}'
Invoke-RestMethod http://127.0.0.1:8080/checkpoint/generate -Method Post `
  -ContentType "application/json" `
  -Body '{"checkpoint_id":0,"text":" user turn","sampling_params":{"max_tokens":20}}'
```

## Documentation

- **`docs/guide/`** — beginner series (小白文档): from "what is LLM serving" to
  KV paging, radix/APC caches, checkpoints, constrained decoding, and scheduling
  metrics. Start at `docs/guide/README.md`.
- **`docs/SELLING-POINTS.md`** — what makes pagoda different (卖点).
- **`docs/SGLANG-COMPAT.md`** — SGLang fusion guide: API/DSL/architecture
  compatibility matrix.
- **`LICENSE` / `NOTICE`** — Apache-2.0 licensing and SGLang provenance /
  attribution.
- **`docs/DESIGN.md`** — architecture, SGLang mapping, parity roadmap.
- **`docs/REQUIREMENTS.md`** — requirement analysis and acceptance criteria.
- **`docs/BENCHMARK.md`** — measured comparison vs SGLang / HF transformers
  (same model, same prompts, on a shared box).

## Real backends

`pagoda` itself is dependency-free and model-agnostic: the engine, scheduler,
radix prefix cache, paged KV and HTTP server are generic over the
`ModelEngine` + `Tokenizer` traits. The sibling crate
[`../pagoda-hf`](../pagoda-hf) plugs **real** artifacts back in:

- `HfTokenizer` — a HuggingFace BPE / Unigram tokenizer (`tokenizer.json`).
- `CandleModel<M>` — a `ModelEngine` adapter loading Llama-family `config.json`
  + safetensors weights via Candle.

That crate requires network access to fetch its dependencies and weights, so it
is not compiled by this repo's offline build/test loop.

## Layout

```
pagoda/
├── Cargo.toml
├── docs/DESIGN.md      # architecture, SGLang mapping, parity roadmap
├── docs/guide/         # beginner-friendly documentation series (小白文档)
├── src/
│   ├── engine.rs       # offline engine + continuous-batching scheduler
│   ├── radix_cache.rs  # RadixAttention-style prefix cache
│   ├── apc.rs          # vLLM-style block-level chained-hash prefix cache
│   ├── kv_cache.rs     # paged KV cache + copy-on-write
│   ├── sampler.rs      # temperature / top-k / top-p / penalties
│   ├── grammar.rs      # regex + JSON constrained-decoding logit mask
│   ├── dsl.rs          # SGLang-style frontend (gen/select/fork)
│   ├── model.rs        # ModelEngine trait + deterministic n-gram toy LM
│   ├── tokenizer.rs    # Tokenizer trait + byte-level tokenizer
│   ├── json.rs         # minimal dependency-free JSON
│   ├── server.rs       # minimal HTTP serving frontend
│   └── bin/pagoda.rs
├── scripts/            # one-click quickstart + agent-cluster demo (ps1/sh)
└── tests/              # engine / fault / admission / apc / checkpoint /
                        # session / batch / graft / property / grammar /
                        # system tests (93 total)
```

## Design notes

The reference engine swaps the heavy parts for deterministic stand-ins while
keeping the architecture honest:

- `ModelEngine::forward` can be backed by Candle / GGML / CUDA / an external
  server; the shipped `NGramModel` is a byte n-gram trained on an embedded corpus.
- `PagedKvCache` stores token ids per page instead of tensors; swap the slot type
  to `[f32; head_dim]` for a real paged-attention pool.
- Prefix caching performs *physical zero-copy reuse*: shared prompt blocks are
  reference-counted, the tail block is copy-on-written on divergence, and requests
  release their references on completion. Repeated prompts cost one forward per
  token instead of re-materializing the whole prefix.

See `docs/DESIGN.md` for the full architecture, the mapping to upstream SGLang,
and the parity roadmap.


