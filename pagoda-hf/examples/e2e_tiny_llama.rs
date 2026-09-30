// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: AGPL-3.0-only
//! P1+P2 end-to-end verification: a real HuggingFace tokenizer + real Candle
//! weights driving the full pagoda engine (prefix cache, batching, sampling),
//! with incremental KV sessions (each token fed to the model exactly once)
//! and checkpoint forking (branch over a shared trunk without recompute).
//!
//! Run on a networked machine:
//!
//! ```text
//! cd pagoda-hf
//! cargo run --release --example e2e_tiny_llama
//! ```
//!
//! Uses `hf-internal-testing/tiny-random-LlamaForCausalLM` (ungated, small):
//! the weights are random, so the generated text is meaningless — what this
//! verifies is the *plumbing*: tokenization, forward shapes, prefix-cache
//! reuse, determinism, and fault-freedom. Every step asserts; a clean exit
//! means the real backend passed.

use anyhow::Result;
use pagoda::{
    Engine, EngineConfig, FinishReason, ModelEngine, SamplingParams, Tokenizer, WriteRequest,
};
use pagoda_hf::{CandleModel, HfTokenizer};
use std::sync::atomic::Ordering;

fn main() -> Result<()> {
    const REPO: &str = "hf-internal-testing/tiny-random-LlamaForCausalLM";

    println!("==> [1/4] download + load HF tokenizer from {REPO}");
    let tokenizer = HfTokenizer::from_hub(REPO)?;
    println!(
        "    vocab_size={} eos={} bos={:?}",
        tokenizer.vocab_size(),
        tokenizer.eos_token_id(),
        tokenizer.bos_token_id()
    );
    let ids = tokenizer.encode("The capital of France is");
    let back = tokenizer.decode(&ids);
    println!("    encode -> {ids:?}");
    println!("    decode -> {back:?}");
    assert!(!ids.is_empty(), "tokenizer produced no tokens");
    assert!(!back.is_empty(), "tokenizer decode produced nothing");

    // Computed before the engine takes ownership of the tokenizer.
    let continuation = " the largest city";
    let continuation_len = tokenizer.encode(continuation).len();

    println!("==> [2/4] download config + safetensors weights, load on CPU (F32)");
    let model = CandleModel::llama_from_hub(REPO)?;
    println!("    model loaded: {}", model.name());
    // Ground-truth model compute, shared before the engine takes ownership.
    let fed = model.tokens_fed_handle();
    // Smoke: logits must be finite and vocab-sized before the engine sees them.
    let probe = model.forward(&ids);
    assert_eq!(probe.len(), model.vocab_size(), "logits width mismatch");
    assert!(
        probe.iter().all(|v| v.is_finite()),
        "logits contain NaN/inf"
    );
    println!(
        "    forward ok: vocab={} min={:.3} max={:.3}",
        probe.len(),
        probe.iter().cloned().fold(f32::INFINITY, f32::min),
        probe.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    println!("==> [3/5] engine generation, cold cache (incremental KV session)");
    let mut engine = Engine::new(
        tokenizer,
        model,
        EngineConfig {
            num_kv_blocks: 4096,
            block_size: 16,
            ..EngineConfig::default()
        },
    );
    let params = SamplingParams {
        max_tokens: 16,
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let fed_before = fed.load(Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    let out1 = engine.generate(&WriteRequest::new("The capital of France is", params.clone()));
    let fed_cold = fed.load(Ordering::Relaxed) - fed_before;
    println!(
        "    {:?} {:?} prefix_hit={}/{} forward={}",
        t0.elapsed(),
        out1.finish_reason,
        out1.prefix_hit_tokens,
        out1.prompt_tokens,
        out1.forward_count
    );
    println!("    text: {:?}", out1.text);
    let decode_steps = out1.output_token_ids.len();
    let replay_cost: usize = (0..decode_steps).map(|i| ids.len() + i).sum();
    println!(
        "    model tokens fed: {fed_cold} (full replay would be {replay_cost}, {:.1}x saved)",
        replay_cost as f64 / fed_cold.max(1) as f64
    );
    assert_eq!(
        fed_cold as usize,
        ids.len() + decode_steps - 1,
        "session must feed the prompt once, then one token per step"
    );
    assert_ne!(
        out1.finish_reason,
        FinishReason::Fault,
        "real model must produce sane logits"
    );
    assert_ne!(
        out1.finish_reason,
        FinishReason::Rejected,
        "request must not be rejected"
    );
    assert_eq!(out1.prefix_hit_tokens, 0, "cold cache must miss");

    println!("==> [4/5] identical request, warm cache");
    let fed_before = fed.load(Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    let out2 = engine.generate(&WriteRequest::new("The capital of France is", params));
    let fed_warm = fed.load(Ordering::Relaxed) - fed_before;
    println!(
        "    {:?} {:?} prefix_hit={}/{} forward={}",
        t0.elapsed(),
        out2.finish_reason,
        out2.prefix_hit_tokens,
        out2.prompt_tokens,
        out2.forward_count
    );
    assert_eq!(
        out2.prefix_hit_tokens, out2.prompt_tokens,
        "warm cache must serve the whole prompt"
    );
    assert!(
        out2.forward_count < out1.forward_count,
        "cache hit must skip prefill compute"
    );
    assert_eq!(
        out1.output_token_ids, out2.output_token_ids,
        "greedy decoding must be deterministic across cache hit/miss"
    );
    println!(
        "    model tokens fed: {fed_warm} (session recomputes the hit prefix once; cross-sequence KV sharing is the P3 milestone)"
    );

    println!("==> [5/5] checkpoint fork: shared trunk, branched continuation");
    let trunk = "The capital of France is";
    let trunk_len = ids.len();
    let fed_before = fed.load(Ordering::Relaxed);
    let ckpt = engine.create_checkpoint(trunk);
    let fed_ckpt = fed.load(Ordering::Relaxed) - fed_before;
    assert_eq!(
        fed_ckpt as usize, trunk_len,
        "checkpoint creation prefills the trunk exactly once"
    );

    let fed_before = fed.load(Ordering::Relaxed);
    let out3 = engine
        .generate_from_checkpoint(
            ckpt,
            continuation,
            SamplingParams {
                max_tokens: 8,
                temperature: 0.0,
                ..SamplingParams::default()
            },
        )
        .expect("checkpoint exists");
    let fed_branch = fed.load(Ordering::Relaxed) - fed_before;
    println!(
    "    {:?} prefix_hit={}/{} branch_fed={} (continuation {} + {} decode)",
        out3.finish_reason,
        out3.prefix_hit_tokens,
        out3.prompt_tokens,
        fed_branch,
        continuation_len,
        out3.output_token_ids.len().saturating_sub(1),
    );
    assert_ne!(out3.finish_reason, FinishReason::Fault);
    assert_eq!(
        out3.prefix_hit_tokens, trunk_len,
        "checkpoint branch must hit the whole trunk"
    );
    assert_eq!(
        fed_branch as usize,
        continuation_len + out3.output_token_ids.len() - 1,
        "branch feeds the continuation + decode steps only, never the trunk"
    );

    // Forking the same trunk twice must be deterministic and independent.
    let out4 = engine
        .generate_from_checkpoint(
            ckpt,
            continuation,
            SamplingParams {
                max_tokens: 8,
                temperature: 0.0,
                ..SamplingParams::default()
            },
        )
        .expect("checkpoint exists");
    assert_eq!(
        out3.output_token_ids, out4.output_token_ids,
        "branches of one checkpoint must be deterministic"
    );

    let stats = engine.stats();
    println!(
        "==> stats: saved={} skip={:.2} faults={} rejected={} kv_util={:.2}",
        stats.compute_saved_tokens(),
        stats.prefill_skip_ratio(),
        stats.faulted_requests,
        stats.rejected_requests,
        stats.kv_utilization()
    );
    assert_eq!(stats.faulted_requests, 0);
    assert_eq!(stats.rejected_requests, 0);

    println!();
    println!("P1 VERIFICATION OK — real tokenizer + real weights drive pagoda end to end");
    println!("P2 VERIFICATION OK — incremental KV sessions + checkpoint fork verified");
    Ok(())
}
