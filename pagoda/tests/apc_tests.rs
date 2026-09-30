// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: AGPL-3.0-only
//! Engine-level tests for the APC (block-level chained-hash) cache backend:
//! cross-request reuse at block granularity, chain semantics on divergence,
//! and eviction under memory pressure.

use pagoda::{CacheBackend, EngineConfig, FinishReason, SamplingParams, ToyEngine, WriteRequest};

fn apc_config(num_kv_blocks: usize, block_size: usize) -> EngineConfig {
    EngineConfig {
        num_kv_blocks,
        block_size,
        cache_backend: CacheBackend::Apc,
        ..EngineConfig::default()
    }
}

fn req(text: &str, max_tokens: usize) -> WriteRequest {
    WriteRequest::new(
        text,
        SamplingParams {
            max_tokens,
            ..SamplingParams::default()
        },
    )
}

#[test]
fn apc_reuses_full_blocks_across_requests() {
    let mut e = ToyEngine::toy(apc_config(64, 4));
    let prompt = "abcdefghijklmnop"; // 16 bytes = exactly 4 full blocks
    let first = e.generate(&req(prompt, 4));
    assert_eq!(first.prefix_hit_tokens, 0, "cold cache must miss");

    let second = e.generate(&req(prompt, 4));
    assert_eq!(
        second.prefix_hit_tokens, 16,
        "identical request must hit every full block"
    );
    assert!(
        second.forward_count < first.forward_count,
        "cache hit must skip prefill compute"
    );

    let stats = e.stats();
    assert!(stats.apc_blocks >= 4, "prompt blocks must be indexed");
    assert!(stats.apc_hit_tokens >= 16);
    assert_eq!(stats.compute_saved_tokens(), stats.apc_hit_tokens);
    assert!(stats.prefill_skip_ratio() > 0.0);
}

#[test]
fn apc_tail_partial_block_is_not_reused() {
    let mut e = ToyEngine::toy(apc_config(64, 4));
    let prompt = "abcdefghijklmnopq"; // 17 bytes: 4 full blocks + 1 tail token
    e.generate(&req(prompt, 2));
    let out = e.generate(&req(prompt, 2));
    assert_eq!(
        out.prefix_hit_tokens, 16,
        "APC is block-granular: the 1-token partial tail must be re-prefilled"
    );
}

#[test]
fn apc_divergent_first_block_misses_everything() {
    let mut e = ToyEngine::toy(apc_config(64, 4));
    e.generate(&req("abcdefghijklmnop", 2));
    // Same length, same suffix, but block 0 differs: the chained hash must
    // break the chain immediately instead of reusing later blocks.
    let out = e.generate(&req("Xbcdefghijklmnop", 2));
    assert_eq!(out.prefix_hit_tokens, 0);
}

#[test]
fn apc_eviction_under_pressure_keeps_serving() {
    let mut e = ToyEngine::toy(EngineConfig {
        num_kv_blocks: 10,
        block_size: 4,
        evict_on_pressure: true,
        cache_backend: CacheBackend::Apc,
        ..EngineConfig::default()
    });
    for i in 0..10 {
        // ~18 bytes = 5 prompt blocks + 1 output block per request: the pool
        // can hold the live request plus one cached predecessor, so nearly
        // every allocation forces an LRU eviction.
        let prompt = format!("prompt number {i}!");
        let out = e.generate(&req(&prompt, 4));
        assert!(
            matches!(out.finish_reason, FinishReason::Stop | FinishReason::Length),
            "request {i} failed: {:?}",
            out.finish_reason
        );
    }
    let stats = e.stats();
    assert_eq!(stats.total_requests, 10);
}

#[test]
fn radix_and_apc_backends_agree_on_outputs() {
    // The toy model is deterministic given the same logical tokens, so both
    // cache backends must produce identical text for the same workload.
    let prompts = ["hello world, hello", "hello world, hello", "goodbye"];
    let mut radix_engine = ToyEngine::toy(EngineConfig {
        num_kv_blocks: 64,
        block_size: 4,
        ..EngineConfig::default()
    });
    let mut apc_engine = ToyEngine::toy(apc_config(64, 4));
    for p in prompts {
        let a = radix_engine.generate(&req(p, 8));
        let b = apc_engine.generate(&req(p, 8));
        assert_eq!(a.text, b.text, "backend disagreement on {p:?}");
        assert_eq!(a.output_token_ids, b.output_token_ids);
    }
}
