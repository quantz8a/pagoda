// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! KV checkpoints: pinned, shareable prefixes for agent-tree / branching
//! workloads (create once, branch many times, never re-prefill the trunk).

use pagoda::{EngineConfig, FinishReason, SamplingParams, ToyEngine, WriteRequest};

fn cfg() -> EngineConfig {
    EngineConfig {
        num_kv_blocks: 64,
        block_size: 8,
        ..EngineConfig::default()
    }
}

fn params(max_tokens: usize, seed: u64) -> SamplingParams {
    SamplingParams {
        max_tokens,
        seed,
        ..SamplingParams::default()
    }
}

#[test]
fn checkpoint_branch_gets_full_prefix_hit() {
    let mut e = ToyEngine::toy(cfg());
    let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(4);
    let cp = e.create_checkpoint(&prompt);
    let trunk_tokens = e.checkpoint_tokens(cp).expect("checkpoint exists");
    assert_eq!(trunk_tokens, prompt.len(), "byte tokenizer: 1 byte = 1 token");

    let continuation = "And then?";
    let out = e
        .generate_from_checkpoint(cp, continuation, params(8, 7))
        .expect("branch generation");
    assert_eq!(out.prefix_hit_tokens, trunk_tokens);
    assert_eq!(out.prompt_tokens, trunk_tokens + continuation.len());
    // Compute charged = continuation prefill + decode only; the trunk is free.
    assert!(
        out.forward_count <= continuation.len() + 8,
        "branch must not re-prefill the trunk: forward_count={}",
        out.forward_count
    );
    // Checkpoint savings are first-class revenue metrics, not invisible.
    let stats = e.stats();
    assert_eq!(stats.checkpoint_hit_tokens, trunk_tokens as u64);
    assert!(stats.compute_saved_tokens() >= trunk_tokens as u64);
}

#[test]
fn unknown_checkpoint_returns_none() {
    let mut e = ToyEngine::toy(cfg());
    assert!(e
        .generate_from_checkpoint(pagoda::CheckpointId(999), "hi", params(4, 1))
        .is_none());
}

#[test]
fn checkpoint_pin_survives_eviction_pressure() {
    let mut e = ToyEngine::toy(EngineConfig {
        num_kv_blocks: 24,
        block_size: 4,
        evict_on_pressure: true,
        ..EngineConfig::default()
    });
    let prompt = "pinned shared system prompt!!"; // 29 bytes -> 8 blocks
    let cp = e.create_checkpoint(prompt);

    // Flood the engine with uncached traffic; LRU eviction will reclaim every
    // unpinned block, but the checkpoint's pin must keep its blocks resident.
    for i in 0..12 {
        let noise = format!("noise request number {i} with padding");
        let out = e.generate(&WriteRequest::new(noise, params(4, i as u64)));
        assert!(matches!(
            out.finish_reason,
            FinishReason::Stop | FinishReason::Length
        ));
    }

    let out = e
        .generate_from_checkpoint(cp, "?", params(4, 3))
        .expect("branch after pressure");
    assert_eq!(
        out.prefix_hit_tokens,
        prompt.len(),
        "pinned checkpoint must keep a full hit even after cache eviction"
    );
}

#[test]
fn drop_checkpoint_releases_the_pin() {
    let mut e = ToyEngine::toy(cfg());
    assert_eq!(e.num_checkpoints(), 0);
    let cp = e.create_checkpoint(&"x".repeat(40));
    assert_eq!(e.num_checkpoints(), 1);
    assert!(e.drop_checkpoint(cp));
    assert_eq!(e.num_checkpoints(), 0);
    assert!(!e.drop_checkpoint(cp), "double drop must report false");
    assert!(
        e.generate_from_checkpoint(cp, "hi", params(4, 1)).is_none(),
        "dropped checkpoint can no longer be branched from"
    );
    // The engine keeps serving normally afterwards.
    let out = e.generate(&WriteRequest::new("still alive", params(4, 1)));
    assert!(matches!(
        out.finish_reason,
        FinishReason::Stop | FinishReason::Length
    ));
}

#[test]
fn branch_respects_per_request_token_cap() {
    let mut e = ToyEngine::toy(EngineConfig {
        max_total_tokens: 32,
        ..cfg()
    });
    let cp = e.create_checkpoint(&"y".repeat(30));
    let out = e
        .generate_from_checkpoint(cp, "zzzz", params(16, 1))
        .expect("branch");
    assert_eq!(out.finish_reason, FinishReason::Rejected);
    assert_eq!(out.rejection, Some(pagoda::RejectReason::TooLong));
}

#[test]
fn branches_do_not_contaminate_each_other() {
    // The toy model is deterministic over logical tokens, so a branch run after
    // sibling branches must produce byte-identical output to the same branch
    // run on a fresh engine.
    let prompt = "shared trunk of the conversation. ".repeat(3);
    let mut e1 = ToyEngine::toy(cfg());
    let cp = e1.create_checkpoint(&prompt);
    let _a = e1
        .generate_from_checkpoint(cp, "branch A says hello", params(12, 11))
        .unwrap();
    let b_after_a = e1
        .generate_from_checkpoint(cp, "branch B", params(12, 22))
        .unwrap();

    let mut e2 = ToyEngine::toy(cfg());
    let cp2 = e2.create_checkpoint(&prompt);
    let b_clean = e2
        .generate_from_checkpoint(cp2, "branch B", params(12, 22))
        .unwrap();

    assert_eq!(b_after_a.text, b_clean.text);
    assert_eq!(b_after_a.output_token_ids, b_clean.output_token_ids);
}

#[test]
fn checkpoint_works_with_apc_backend() {
    // Branching off a checkpoint whose tail block is partially filled forces a
    // copy-on-write fork; publishing the branch into the APC cache then
    // exercises the mixed-block chunk path.
    let mut e = ToyEngine::toy(EngineConfig {
        num_kv_blocks: 32,
        block_size: 4,
        cache_backend: pagoda::CacheBackend::Apc,
        ..EngineConfig::default()
    });
    let prompt = "agent trunk with a partial tail"; // 30 bytes, tail = 2 tokens
    let cp = e.create_checkpoint(prompt);

    let a = e
        .generate_from_checkpoint(cp, " + more", params(8, 5))
        .expect("branch A");
    assert_eq!(a.prefix_hit_tokens, prompt.len());
    assert!(matches!(
        a.finish_reason,
        FinishReason::Stop | FinishReason::Length
    ));

    let b = e
        .generate_from_checkpoint(cp, " + again", params(8, 6))
        .expect("branch B");
    assert_eq!(b.prefix_hit_tokens, prompt.len());
    assert!(matches!(
        b.finish_reason,
        FinishReason::Stop | FinishReason::Length
    ));
}
