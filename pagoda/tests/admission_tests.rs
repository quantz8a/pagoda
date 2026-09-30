// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Admission control and SLO guardrails for the batch engine.

use pagoda::{
    EngineConfig, FinishReason, RejectReason, SamplingParams, ToyEngine, WriteRequest,
};

#[test]
fn empty_prompt_is_rejected() {
    let mut e = ToyEngine::toy(EngineConfig::default());
    let out = e.generate(&WriteRequest::new("", SamplingParams::default()));
    assert_eq!(out.finish_reason, FinishReason::Rejected);
    assert_eq!(out.rejection, Some(RejectReason::EmptyPrompt));
    assert!(out.output_token_ids.is_empty());
    assert_eq!(e.stats().rejected_requests, 1);
}

#[test]
fn oversized_request_rejected_by_slo_guard() {
    let mut e = ToyEngine::toy(EngineConfig {
        max_total_tokens: 16,
        ..EngineConfig::default()
    });

    // 11 prompt tokens + 100 requested output tokens exceeds the 16-token cap.
    let rejected = e.generate(&WriteRequest::new(
        "hello world",
        SamplingParams { max_tokens: 100, ..SamplingParams::default() },
    ));
    assert_eq!(rejected.finish_reason, FinishReason::Rejected);
    assert_eq!(rejected.rejection, Some(RejectReason::TooLong));

    // A request inside the cap is admitted and runs to length.
    let accepted = e.generate(&WriteRequest::new(
        "hi",
        SamplingParams { max_tokens: 4, ..SamplingParams::default() },
    ));
    assert_eq!(accepted.finish_reason, FinishReason::Length);
    assert_eq!(accepted.output_token_ids.len(), 4);

    assert_eq!(e.stats().rejected_requests, 1);
    assert_eq!(e.stats().total_requests, 1);
}

#[test]
fn queue_cap_rejects_excess_requests() {
    let mut e = ToyEngine::toy(EngineConfig {
        max_waiting_requests: 2,
        ..EngineConfig::default()
    });

    let params = SamplingParams { max_tokens: 4, ..SamplingParams::default() };
    let reqs = vec![
        WriteRequest::new("one", params.clone()),
        WriteRequest::new("two", params.clone()),
        WriteRequest::new("three", params.clone()),
        WriteRequest::new("four", params.clone()),
    ];
    let outs = e.generate_batch(&reqs);

    assert_eq!(outs.len(), 4);
    let rejected = outs.iter().filter(|o| o.finish_reason == FinishReason::Rejected).count();
    let completed = outs.iter().filter(|o| o.finish_reason == FinishReason::Length).count();
    assert_eq!(rejected, 2);
    assert_eq!(completed, 2);
    assert!(outs.iter().filter(|o| o.finish_reason == FinishReason::Rejected).all(|o| o.rejection == Some(RejectReason::QueueFull)));

    assert_eq!(e.stats().rejected_requests, 2);
    assert_eq!(e.stats().total_requests, 2);
}

