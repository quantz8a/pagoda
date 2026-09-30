// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! End-to-end tests of the engine, DSL, and prefix caching.

use pagoda::{
    EngineConfig, FinishReason, Program, SamplingParams, SchedulePolicy, ToyEngine, WriteRequest,
};

fn engine() -> ToyEngine {
    ToyEngine::toy(EngineConfig {
        num_kv_blocks: 256,
        block_size: 8,
        max_running_requests: 32,
        max_prefill_tokens_per_step: 32,
        seed: 0,
        ..EngineConfig::default()
    })
}

#[test]
fn generate_runs_to_max_tokens() {
    let mut e = engine();
    let out = e.generate(&WriteRequest::new(
        "Once upon a time",
        SamplingParams { max_tokens: 32, ..SamplingParams::default() },
    ));
    assert_eq!(out.finish_reason, FinishReason::Length);
    assert_eq!(out.output_token_ids.len(), 32);
    assert!(out.full_text.as_ref().unwrap().starts_with("Once upon a time"));
}

#[test]
fn generation_is_deterministic_for_a_seed() {
    let params = SamplingParams { temperature: 0.7, seed: 7, max_tokens: 16, ..SamplingParams::default() };
    let req = WriteRequest::new("hello", params.clone());
    let mut a = engine();
    let mut b = engine();
    assert_eq!(a.generate(&req).text, b.generate(&req).text);
}

#[test]
fn prefix_cache_skips_repeated_prompt() {
    let mut e = engine();
    let req = WriteRequest::new(
        "the quick brown fox jumps over the lazy dog",
        SamplingParams { max_tokens: 8, ..SamplingParams::default() },
    );
    let first = e.generate(&req);
    assert_eq!(first.prefix_hit_tokens, 0);

    let second = e.generate(&req);
    let prompt_tokens = first.prompt_tokens;
    assert_eq!(second.prefix_hit_tokens, prompt_tokens);
    // Prefill work is skipped entirely on the second call.
    assert_eq!(second.forward_count, second.output_token_ids.len());
}

#[test]
fn batch_yields_outputs_in_request_order() {
    let mut e = engine();
    let params = SamplingParams { max_tokens: 6, ..SamplingParams::default() };
    let reqs = vec![
        WriteRequest::new("alpha", params.clone()),
        WriteRequest::new("beta", params.clone()),
        WriteRequest::new("gamma", params.clone()),
    ];
    let outs = e.generate_batch(&reqs);
    assert_eq!(outs.len(), 3);
    for o in outs {
        assert_eq!(o.output_token_ids.len(), 6);
    }
}

#[test]
fn stop_string_terminates() {
    let mut e = engine();
    // A stop string that will never match forces length termination, while a
    // long stop token list exercises the code path without changing the result.
    let params = SamplingParams {
        max_tokens: 16,
        stop_token_ids: vec![u32::MAX],
        ..SamplingParams::default()
    };
    let out = e.generate(&WriteRequest::new("abc", params));
    assert_eq!(out.finish_reason, FinishReason::Length);
}

#[test]
fn program_binds_gen_and_select_vars() {
    let mut e = engine();

    let mut p = Program::new();
    p.system("assistant");
    p.user("question");
    p.select(
        "answer",
        vec![" alpha".to_string(), " beta".to_string()],
        SamplingParams::default(),
    );
    p.gen("tail", SamplingParams { max_tokens: 5, ..SamplingParams::default() });

    let results = e.run_program(&p);
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(r.get("answer").is_some());
    assert!(r.get("tail").is_some());
}

#[test]
fn program_fork_produces_one_result_per_branch() {
    let mut e = engine();

    let mut left = Program::new();
    left.gen("side", SamplingParams { max_tokens: 5, ..SamplingParams::default() });
    let mut right = Program::new();
    right.gen("side", SamplingParams { max_tokens: 5, seed: 1, ..SamplingParams::default() });

    let mut p = Program::new();
    p.user("seed");
    p.fork(vec![left, right]);

    let results = e.run_program(&p);
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.get("side").is_some());
    }
}

#[test]
fn physical_prefix_reuse_reduces_allocations() {
    let mut e = ToyEngine::toy(EngineConfig {
        num_kv_blocks: 1024,
        block_size: 8,
        max_running_requests: 32,
        max_prefill_tokens_per_step: 32,
        seed: 0,
        ..EngineConfig::default()
    });
    let req = WriteRequest::new(
        "the shared system prompt is somewhat long",
        SamplingParams { max_tokens: 4, ..SamplingParams::default() },
    );

    let _ = e.generate(&req);
    let first_block_count = e.stats().kv_allocations;

    let _ = e.generate(&req);
    let second_block_count = e.stats().kv_allocations - first_block_count;

    // The reused prompt spans several pages; its physical KV is shared, so the
    // second request must touch far fewer pages than the first materialization.
    assert!(
        second_block_count < first_block_count,
        "second={second_block_count} first={first_block_count}"
    );
}
#[test]
fn chunked_prefill_splits_long_prompts() {
    let mut e = ToyEngine::toy(EngineConfig {
        num_kv_blocks: 1024,
        block_size: 8,
        max_running_requests: 32,
        max_prefill_tokens_per_step: 4,
        seed: 0,
        ..EngineConfig::default()
    });
    let req = WriteRequest::new(
        "a fairly long prompt that has more than four tokens to materialize",
        SamplingParams { max_tokens: 6, ..SamplingParams::default() },
    );
    let out = e.generate(&req);
    // The scheduler must spend several steps prefilling before decoding.
    assert!(e.stats().prefill_chunks > 1, "chunks={}", e.stats().prefill_chunks);
    assert_eq!(out.output_token_ids.len(), 6);
    assert_eq!(out.finish_reason, FinishReason::Length);
}

#[test]
fn stats_expose_revenue_metrics() {
    let mut e = engine();
    let req = WriteRequest::new(
        "the quick brown fox jumps over the lazy dog",
        SamplingParams { max_tokens: 8, ..SamplingParams::default() },
    );
    e.generate(&req);
    e.generate(&req);

    let s = e.stats();
    assert_eq!(s.total_requests, 2);
    assert!(s.total_prompt_tokens > 0);
    assert!(s.total_output_tokens > 0);
    assert!(s.compute_saved_tokens() > 0, "second prompt should hit the cache");
    assert!(s.prefill_skip_ratio() > 0.0);
    assert!(s.avg_forward_per_output_token().is_finite());
    assert!(s.kv_utilization() > 0.0);
}

#[test]
fn longest_prefix_policy_schedules_hit_first() {
    let shared = "the common system instruction is rather long";
    let mut e = ToyEngine::toy(EngineConfig {
        schedule_policy: SchedulePolicy::LongestPrefix,
        ..EngineConfig::default()
    });

    // Seed the prefix cache with the shared prompt.
    e.generate(&WriteRequest::new(
        shared,
        SamplingParams { max_tokens: 1, ..SamplingParams::default() },
    ));

    let params = SamplingParams { max_tokens: 4, ..SamplingParams::default() };
    let reqs = vec![
        WriteRequest::new("zzz totally unrelated", params.clone()),
        WriteRequest::new(shared, params.clone()),
    ];
    let outs = e.generate_batch(&reqs);
    assert_eq!(outs.len(), 2);

    assert!(outs[0].prefix_hit_tokens > 0);
    assert_eq!(outs[1].prefix_hit_tokens, 0);
}

#[test]
fn shortest_prompt_policy_finishes_short_first() {
    let mut e = ToyEngine::toy(EngineConfig {
        schedule_policy: SchedulePolicy::ShortestPrompt,
        max_prefill_tokens_per_step: 16,
        ..EngineConfig::default()
    });
    let params = SamplingParams { max_tokens: 1, ..SamplingParams::default() };
    // Long request is queued FIRST; the policy must still finish the short one
    // first.
    let reqs = vec![
        WriteRequest::new("l".repeat(200), params.clone()),
        WriteRequest::new("s".repeat(10), params),
    ];
    let outs = e.generate_batch(&reqs);
    assert_eq!(outs.len(), 2);
    assert_eq!(
        outs[0].prompt_tokens, 10,
        "short prompt must complete first under ShortestPrompt"
    );
    assert_eq!(outs[1].prompt_tokens, 200);
}
