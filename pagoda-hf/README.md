# pagoda-hf

Real HuggingFace tokenizer and Candle weight backends for
[`pagoda`](../pagoda). It plugs into the same trait surface and replaces
the toy n-gram + byte stand-ins with real artifacts:

- `HfTokenizer` — wraps `tokenizers::Tokenizer` (`tokenizer.json`).
- `CandleModel<M>` — a `ModelEngine` adapter with a Candle Llama-family model
  loading real `config.json` + safetensors weights.
- `Laya` — the Laya System-1 decision model (ModernBERT encoder + decision
  head): typed answers (choice / score / noul) with calibrated probabilities
  in one non-autoregressive forward pass. See `examples/e2e_laya.rs`.

> **Status**: this crate requires network access to download its dependencies and
> model weights. It is **not** compiled or exercised by the offline
> `cargo build --offline` / `cargo test --offline` loop inside `pagoda`.

## Build

```powershell
cd pagoda-hf
cargo build
```

## Verify (P1)

```powershell
cd pagoda-hf
cargo run --release --example e2e_tiny_llama
```

Downloads the ungated `hf-internal-testing/tiny-random-LlamaForCausalLM`
(random weights — output text is meaningless by design) and asserts the full
serving path: tokenizer roundtrip, finite vocab-sized logits, cold-miss /
warm-hit prefix caching, greedy determinism across cache states, zero faults.
See `../pagoda/docs/P1-VERIFICATION.md` for the full runbook.

## Verify (P5): a real distilled student, pure Rust

```powershell
cargo run --release --example e2e_qwen_student -- /path/to/student-merged
```

Loads a Qwen2.5 LoRA-merged student (from ../distill) and runs a real
customer-ticket prompt through prefill+decode on the pagoda engine.
Verified 2026-10-03 on Qwen2.5-0.5B: stops cleanly at <|im_end|> (85 tokens,
12.9s on a shared CPU, f32), feeds every token exactly once, and emits
schema-complete ticket JSON — no Python in the loop. Qwen2 checkpoints need
q/k/v bias, which the loader probes from the checkpoint itself.

## Usage

```rust
use candle_core::Device;
use pagoda_hf::{CandleModel, HfTokenizer};
use pagoda::{Engine, EngineConfig};

fn main() -> anyhow::Result<()> {
    // Llama-family repo; meta-llama/* is gated (needs HF_TOKEN), TinyLlama is not.
    let tokenizer = HfTokenizer::from_hub("TinyLlama/TinyLlama-1.1B-Chat-v1.0")?;
    let model = CandleModel::llama_from_hub("TinyLlama/TinyLlama-1.1B-Chat-v1.0")?;

    let mut engine = Engine::new(tokenizer, model, EngineConfig::default());
    let out = engine.generate(&pagoda::spec::WriteRequest::new(
        "Once upon a time",
        Default::default(),
    ));
    println!("{}", out.text);
    Ok(())
}
```

Notes:

- `llama_from_hub` assumes a single-file `model.safetensors`; sharded repos need
  the full list of weight files (pass `&[PathBuf]` to `llama_from_safetensors`).
- The engine currently feeds the full token context each step, so the adapter
  runs full-context forwards without incremental KV reuse. See the crate-level
  docs for the production path forward.

## Laya decision server (System 1)

One-binary Rust deployment of `convaiinnovations/laya` (ModernBERT encoder +
decision head): typed decisions (choice / score / noul) in a single
non-autoregressive forward, Jev-compatible JSON, no Python / PyTorch needed.

```powershell
# one click: build + serve + health-check + smoke-test
powershell -ExecutionPolicy Bypass -File scripts\serve-laya.ps1 -Smoke   # Windows
bash scripts/serve-laya.sh --smoke                                       # Linux/macOS

curl -X POST http://127.0.0.1:8081/decide -H "Content-Type: application/json" -d '{
  "state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
  "questions": {"department": {"type": "choice", "instructions": "Which department should handle this?",
    "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}}}
}'
# -> "choice": "billing", probabilities identical to the reference Python API
```

Verified bit-compatible with the official `rl_agent_api.py` responses
(billing 0.9865 / confidence 0.9267 / churn 0.879 / act 1.0 on the README
scenario). Same-machine benchmark vs the PyTorch reference:
`../pagoda/docs/BENCHMARK-LAYA.md`; beginner guide:
`../pagoda/docs/guide/14-laya-one-click-deploy.md`. End-to-end assertion suite:
`cargo run --release --example e2e_laya`.
