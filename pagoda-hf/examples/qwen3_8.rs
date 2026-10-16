// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Greedy text generation for an MLX Qwen3.5 / Qwen3.8 directory.
//!
//! ```text
//! PAGODA_DEVICE=metal cargo run --release --features metal --example qwen3_8 -- \
//!   --model ~/.mlx-serve/models/mlx-community/Qwen3.8-27B-8bit \
//!   --prompt "Hello" --max-tokens 4
//! ```
//!
//! Text only. The vision tower in that directory is ignored.

use anyhow::Result;
use pagoda::Tokenizer;
use pagoda_hf::{CandleModel, HfTokenizer, Qwen35};

fn main() -> Result<()> {
    let mut model_dir = std::env::var("HOME").unwrap_or_default()
        + "/.mlx-serve/models/mlx-community/Qwen3.8-27B-8bit";
    let mut prompt = "Hello".to_string();
    let mut max_tokens: usize = 4;
    let mut check = false;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--check" {
            check = true;
            continue;
        }
        let Some(value) = it.next() else { break };
        match flag.as_str() {
            "--model" => model_dir = value,
            "--prompt" => prompt = value,
            "--max-tokens" => max_tokens = value.parse().expect("--max-tokens"),
            other => eprintln!("ignoring {other} {value}"),
        }
    }
    let device = CandleModel::device_from_env()?;
    eprintln!("==> device {device:?}");
    eprintln!("==> load {model_dir}");
    let model = Qwen35::load(&model_dir, device)?;
    let tokenizer = HfTokenizer::from_file(format!("{model_dir}/tokenizer.json"))?;
    let ids = tokenizer.encode(&prompt);
    eprintln!("==> prompt tokens {}", ids.len());
    let mut cache = model.new_cache();
    let mut next = ids;
    let mut produced = Vec::new();
    for step in 0..max_tokens {
        let t = std::time::Instant::now();
        let (token, best) = model.forward_greedy(&next, &mut cache)?;
        let secs = t.elapsed().as_secs_f64();
        eprintln!(
            "==> step {step} {:.3}s ({:.1} tok/s) token {token} logit {best:.3} {:?}",
            secs,
            1.0 / secs.max(1e-9),
            tokenizer.decode(&[token])
        );
        if token == model.eos() {
            break;
        }
        produced.push(token);
        next = vec![token];
    }
    println!("{}", tokenizer.decode(&produced));
    if check {
        // Greedy `Hi` on mlx-community/Qwen3.8-27B-8bit, matched to mlx-serve
        // temperature 0 on 2026-10-17. Same fp32 accumulation order.
        const ANCHOR: [u32; 16] = [
            11, 353, 2688, 264, 220, 17, 15, 4514, 6086, 8330, 11, 321, 353, 2908, 978, 23330,
        ];
        let n = produced.len().min(ANCHOR.len());
        if prompt != "Hi" || produced.len() < n || produced[..n] != ANCHOR[..n] {
            eprintln!("check failed, got {produced:?}");
            anyhow::bail!("greedy tokens diverged from the mlx-serve anchor");
        }
        eprintln!("check ok ({n} tokens)");
    }
    Ok(())
}
