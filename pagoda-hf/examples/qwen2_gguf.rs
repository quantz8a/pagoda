// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Greedy Qwen2.5 GGUF on the pagoda engine, with a second pass that must
//! hit the prefix cache.
//!
//! ```text
//! PAGODA_DEVICE=metal cargo run --release --features metal --example qwen2_gguf
//! ```
//!
//! Defaults to Qwen2.5-0.5B-Instruct Q4_K_M. Pass `--repo` / `--file` for
//! another Qwen2 GGUF (Qwen2.5-7B-Instruct Q4_K_M uses the same loader).

use anyhow::Result;
use pagoda::{Engine, EngineConfig, SamplingParams, Tokenizer, WriteRequest};
use pagoda_hf::{CandleModel, HfTokenizer};
use std::sync::atomic::Ordering;
use std::time::Instant;

struct Args {
    repo: String,
    file: String,
    tokenizer: String,
    gguf: Option<String>,
    tokenizer_file: Option<String>,
    prompt: String,
    max_tokens: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF".into(),
            file: "qwen2.5-0.5b-instruct-q4_k_m.gguf".into(),
            tokenizer: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            gguf: None,
            tokenizer_file: None,
            prompt: "The quick brown fox jumps over the lazy dog. ".repeat(8),
            max_tokens: 32,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else {
            eprintln!("flag {flag} needs a value");
            break;
        };
        match flag.as_str() {
            "--repo" => args.repo = value,
            "--file" => args.file = value,
            "--tokenizer" => args.tokenizer = value,
            "--gguf" => args.gguf = Some(value),
            "--tokenizer-file" => args.tokenizer_file = Some(value),
            "--prompt" => args.prompt = value,
            "--max-tokens" => args.max_tokens = value.parse().expect("--max-tokens"),
            other => eprintln!("ignoring unknown flag {other}"),
        }
    }
    args
}

fn main() -> Result<()> {
    let args = parse_args();
    let device = CandleModel::device_from_env()?;
    eprintln!("==> device {device:?}");
    let tokenizer = match &args.tokenizer_file {
        Some(path) => {
            eprintln!("==> tokenizer file {path}");
            HfTokenizer::from_file(path)?
        }
        None => {
            eprintln!("==> tokenizer {}", args.tokenizer);
            HfTokenizer::from_hub(&args.tokenizer)?
        }
    };
    let weights = match &args.gguf {
        Some(path) => {
            eprintln!("==> weights {path}");
            std::path::PathBuf::from(path)
        }
        None => {
            eprintln!("==> weights {}/{}", args.repo, args.file);
            HfTokenizer::download(&args.repo, &args.file)?
        }
    };
    let t_load = Instant::now();
    let model = CandleModel::qwen2_from_gguf(&weights, device.clone())?;
    let load_s = t_load.elapsed().as_secs_f64();
    let fed = model.tokens_fed_handle();
    let prompt_tokens = tokenizer.encode(&args.prompt).len();

    let greedy = SamplingParams {
        max_tokens: args.max_tokens,
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let mut engine = Engine::new(
        tokenizer,
        model,
        EngineConfig {
            num_kv_blocks: 8192,
            block_size: 16,
            max_running_requests: 8,
            ..EngineConfig::default()
        },
    );

    let t = Instant::now();
    let cold = engine.generate(&WriteRequest::new(&args.prompt, greedy.clone()));
    let cold_ms = t.elapsed().as_secs_f64() * 1e3;
    let fed_cold = fed.load(Ordering::Relaxed);

    let t = Instant::now();
    let warm = engine.generate(&WriteRequest::new(&args.prompt, greedy));
    let warm_ms = t.elapsed().as_secs_f64() * 1e3;
    let fed_warm = fed.load(Ordering::Relaxed) - fed_cold;

    anyhow::ensure!(
        cold.finish_reason != pagoda::FinishReason::Fault,
        "model returned non-finite logits"
    );
    anyhow::ensure!(
        cold.output_token_ids == warm.output_token_ids,
        "warm greedy output diverged from cold"
    );
    anyhow::ensure!(
        fed_cold >= prompt_tokens as u64,
        "cold pass did not feed the prompt ({fed_cold} tokens fed, prompt {prompt_tokens})"
    );
    anyhow::ensure!(
        fed_warm <= (args.max_tokens + 1) as u64,
        "prefix cache missed: warm fed {fed_warm} tokens, cold fed {fed_cold}, prompt {prompt_tokens}"
    );

    let out_tokens = cold.output_token_ids.len().max(1);
    let report = serde_json::json!({
        "engine": "pagoda-hf candle-qwen2-gguf",
        "device": format!("{device:?}"),
        "repo": args.repo,
        "file": args.gguf.clone().unwrap_or(args.file),
        "load_s": (load_s * 100.0).round() / 100.0,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": out_tokens,
        "cold_ms": (cold_ms * 100.0).round() / 100.0,
        "warm_ms": (warm_ms * 100.0).round() / 100.0,
        "cold_tokens_fed": fed_cold,
        "warm_tokens_fed": fed_warm,
        "cold_tok_per_s": ((out_tokens as f64) / (cold_ms / 1e3)).round(),
        "warm_tok_per_s": ((out_tokens as f64) / (warm_ms / 1e3)).round(),
        "text": cold.text,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
