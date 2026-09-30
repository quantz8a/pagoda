# pagoda-hf

Real HuggingFace tokenizer and Candle weight backends for
[`pagoda`](../pagoda). It plugs into the same trait surface and replaces
the toy n-gram + byte stand-ins with real artifacts:

- `HfTokenizer` — wraps `tokenizers::Tokenizer` (`tokenizer.json`).
- `CandleModel<M>` — a `ModelEngine` adapter with a Candle Llama-family model
  loading real `config.json` + safetensors weights.

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

