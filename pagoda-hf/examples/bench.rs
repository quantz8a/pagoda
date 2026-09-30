// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Benchmark harness: pagoda engine + real Candle weights, CPU.
//!
//! Prints one JSON object to stdout; methodology and comparison against
//! SGLang (python) live in `../pagoda/docs/BENCHMARK.md`.
//!
//! ```text
//! cargo run --release --example bench -- \
//!     --repo hf-internal-testing/tiny-random-LlamaForCausalLM \
//!     --prompt-tokens 128 --gen-tokens 32 --batch 8 --repeat 3
//! ```

use anyhow::Result;
use pagoda::{Engine, EngineConfig, SamplingParams, Tokenizer, WriteRequest};
use pagoda_hf::{CandleModel, HfTokenizer};
use std::sync::atomic::Ordering;
use std::time::Instant;

struct Args {
    repo: String,
    prompt_tokens: usize,
    gen_tokens: usize,
    batch: usize,
    repeat: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            repo: "hf-internal-testing/tiny-random-LlamaForCausalLM".to_string(),
            prompt_tokens: 128,
            gen_tokens: 32,
            batch: 8,
            repeat: 3,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else { break };
        match flag.as_str() {
            "--repo" => args.repo = value,
            "--prompt-tokens" => args.prompt_tokens = value.parse().expect("--prompt-tokens"),
            "--gen-tokens" => args.gen_tokens = value.parse().expect("--gen-tokens"),
            "--batch" => args.batch = value.parse().expect("--batch"),
            "--repeat" => args.repeat = value.parse().expect("--repeat"),
            other => eprintln!("ignoring unknown flag {other}"),
        }
    }
    args
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn main() -> Result<()> {
    let args = parse_args();
    eprintln!("==> loading tokenizer + weights from {}", args.repo);
    let tokenizer = HfTokenizer::from_hub(&args.repo)?;
    let device = CandleModel::device_from_env()?;
    eprintln!("==> device: {device:?}");
    let model = CandleModel::llama_from_hub_on(&args.repo, device.clone())?;
    let fed = model.tokens_fed_handle();

    // Same prompt text on both engines: a fixed sentence repeated until the
    // target token count is reached.
    let sentence = "The quick brown fox jumps over the lazy dog. ";
    let mut prompt = String::new();
    while tokenizer.encode(&prompt).len() < args.prompt_tokens {
        prompt.push_str(sentence);
    }
    let prompt_tokens = tokenizer.encode(&prompt).len();

    let greedy = SamplingParams {
        max_tokens: args.gen_tokens,
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let mut engine = Engine::new(
        tokenizer,
        model,
        EngineConfig {
            num_kv_blocks: 8192,
            block_size: 16,
            max_running_requests: 64,
            ..EngineConfig::default()
        },
    );

    // Cold: first run of this prompt on a fresh engine.
    let t = Instant::now();
    let cold = engine.generate(&WriteRequest::new(&prompt, greedy.clone()));
    let cold_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(
        cold.finish_reason,
        pagoda::FinishReason::Length,
        "bench expects length-bounded runs"
    );

    // Warm: same prompt, radix cache hot.
    let mut warm_ms = Vec::new();
    for _ in 0..args.repeat {
        let t = Instant::now();
        engine.generate(&WriteRequest::new(&prompt, greedy.clone()));
        warm_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }

    // Batch: K distinct prompts (distinct first tokens => no shared trunk),
    // one continuous-batching drain.
    let reqs: Vec<WriteRequest> = (0..args.batch)
        .map(|i| WriteRequest::new(format!("Request number {i}. {prompt}"), greedy.clone()))
        .collect();
    let fed_before = fed.load(Ordering::Relaxed);
    let t = Instant::now();
    let outs = engine.generate_batch(&reqs);
    let batch_ms = t.elapsed().as_secs_f64() * 1e3;
    let fed_batch = fed.load(Ordering::Relaxed) - fed_before;
    let total_out: usize = outs.iter().map(|o| o.output_token_ids.len()).sum();

    let out = serde_json::json!({
        "engine": format!("pagoda-hf (candle, {} {})",
            if device.is_cuda() { "cuda" } else { "cpu" },
            std::env::var("PAGODA_DTYPE").unwrap_or_else(|_| "f32".to_string())),
        "repo": args.repo,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": args.gen_tokens,
        "cold_ms": (cold_ms * 100.0).round() / 100.0,
        "warm_median_ms": (median(&warm_ms) * 100.0).round() / 100.0,
        "warm_all_ms": warm_ms.iter().map(|v| (*v * 100.0).round() / 100.0).collect::<Vec<_>>(),
        "batch": {
            "requests": args.batch,
            "wall_ms": (batch_ms * 100.0).round() / 100.0,
            "output_tokens": total_out,
            "output_tok_per_s": (total_out as f64 / (batch_ms / 1e3)).round(),
            "model_tokens_fed": fed_batch,
        },
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
