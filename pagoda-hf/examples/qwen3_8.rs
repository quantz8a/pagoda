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

#[cfg(feature = "metal")]
fn fill_prefix(model: &Qwen35, cache: &mut pagoda_hf::Qwen35Cache, ids: &[u32]) -> Result<()> {
    for &id in ids {
        let _ = model.forward(&[id], cache)?;
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn run_compare(model: &Qwen35, tokenizer: &HfTokenizer, prompt: &str) -> Result<()> {
    let ids = tokenizer.encode(prompt);
    anyhow::ensure!(!ids.is_empty(), "empty prompt");
    let prefix = &ids[..ids.len() - 1];
    let last = ids[ids.len() - 1];
    eprintln!("==> prompt tokens {}", ids.len());
    eprintln!("==> warmup");
    {
        let mut cache = model.new_cache();
        let _ = model.forward_greedy(&[last], &mut cache)?;
    }
    eprintln!("==> allocate verify");
    {
        let mut cache = model.new_cache();
        let _ = model.forward_verify(&[last, last, last, last], &mut cache)?;
    }
    eprintln!("==> serial 4, prefix {}", prefix.len());
    let mut cache = model.new_cache();
    fill_prefix(model, &mut cache, prefix)?;
    let mut input = last;
    let mut produced = Vec::new();
    let mut serial_logits = Vec::new();
    let t_serial = std::time::Instant::now();
    for _ in 0..4 {
        let (token, logit) = model.forward_greedy(&[input], &mut cache)?;
        produced.push(token);
        serial_logits.push(logit);
        input = token;
    }
    let serial_s = t_serial.elapsed().as_secs_f64();
    eprintln!(
        "==> serial {:.3}s ({:.2} tok/s) {produced:?} {:?}",
        serial_s,
        4.0 / serial_s,
        tokenizer.decode(&produced)
    );
    let inputs = [last, produced[0], produced[1], produced[2]];
    eprintln!("==> verify 4, prefix {}", prefix.len());
    let mut cache = model.new_cache();
    fill_prefix(model, &mut cache, prefix)?;
    let t_verify = std::time::Instant::now();
    let got = model.forward_verify(&inputs, &mut cache)?;
    let verify_s = t_verify.elapsed().as_secs_f64();
    let got_ids: Vec<u32> = got.iter().map(|p| p.0).collect();
    eprintln!(
        "==> verify {:.3}s ({:.2} tok/s) speedup {:.2}x {got_ids:?}",
        verify_s,
        4.0 / verify_s,
        serial_s / verify_s
    );
    if got_ids != produced {
        anyhow::bail!("verify tokens {got_ids:?} diverged from serial {produced:?}");
    }
    let mut max_abs = 0.0f32;
    for (i, ((sid, slogit), (vid, vlogit))) in produced
        .iter()
        .zip(serial_logits.iter())
        .zip(got.iter())
        .enumerate()
    {
        let d = (vlogit - slogit).abs();
        max_abs = max_abs.max(d);
        eprintln!("==> pos {i} token {sid}=={vid} serial {slogit:.6} verify {vlogit:.6} abs {d:.6}");
    }
    eprintln!("==> tokens match, max logit abs {max_abs:.6}");
    Ok(())
}

fn main() -> Result<()> {
    let mut model_dir = std::env::var("HOME").unwrap_or_default()
        + "/.mlx-serve/models/mlx-community/Qwen3.8-27B-8bit";
    let mut prompt = "Hello".to_string();
    let mut max_tokens: usize = 4;
    let mut check = false;
    let mut compare = false;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--check" {
            check = true;
            continue;
        }
        if flag == "--compare" {
            compare = true;
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
    if compare {
        #[cfg(feature = "metal")]
        {
            run_compare(&model, &tokenizer, &prompt)?;
            return Ok(());
        }
        #[cfg(not(feature = "metal"))]
        {
            anyhow::bail!("--compare needs --features metal (forward_verify is Metal-only)");
        }
    }
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
