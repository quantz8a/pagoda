// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Request-level fault isolation: a model backend that returns invalid logits
//! for a specific request must terminate only that request, never panic or
//! corrupt the rest of the batch.

use pagoda::{
    ByteTokenizer, Engine, EngineConfig, FinishReason, ModelEngine, SamplingParams, WriteRequest,
};

/// Returns NaN logits whenever the token context contains `sentinel`, and
/// healthy uniform logits otherwise. Byte tokens start at id 3, so the
/// sentinel for byte `\x01` is 4.
struct SentinelFaultModel {
    sentinel: u32,
    vocab: usize,
}

impl ModelEngine for SentinelFaultModel {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn forward(&self, context: &[u32]) -> Vec<f32> {
        if context.contains(&self.sentinel) {
            vec![f32::NAN; self.vocab]
        } else {
            vec![0.0f32; self.vocab]
        }
    }

    fn name(&self) -> &'static str {
        "sentinel-fault"
    }
}

fn engine() -> Engine<SentinelFaultModel, ByteTokenizer> {
    Engine::new(
        ByteTokenizer::new(),
        SentinelFaultModel { sentinel: 4, vocab: 259 },
        EngineConfig {
            num_kv_blocks: 128,
            block_size: 8,
            ..EngineConfig::default()
        },
    )
}

#[test]
fn faulty_model_yields_fault_reason_instead_of_panicking() {
    // The \x01 byte maps to token id 4 and makes this request fault.
    let mut e = engine();
    let out = e.generate(&WriteRequest::new(
        "a\u{1}b",
        SamplingParams { max_tokens: 8, ..SamplingParams::default() },
    ));
    assert_eq!(out.finish_reason, FinishReason::Fault);
    assert!(out.output_token_ids.is_empty());
    assert_eq!(e.stats().faulted_requests, 1);
}

#[test]
fn fault_isolates_to_one_request_in_a_batch() {
    let mut e = engine();
    let params = SamplingParams { max_tokens: 4, ..SamplingParams::default() };
    let reqs = vec![
        WriteRequest::new("alpha", params.clone()),
        WriteRequest::new("a\u{1}b", params.clone()),
    ];
    let outs = e.generate_batch(&reqs);

    assert_eq!(outs.len(), 2);
    let faults = outs.iter().filter(|o| o.finish_reason == FinishReason::Fault).count();
    let lengths = outs.iter().filter(|o| o.finish_reason == FinishReason::Length).count();
    assert_eq!(faults, 1);
    assert_eq!(lengths, 1);

    let healthy = outs.iter().find(|o| o.finish_reason == FinishReason::Length).unwrap();
    assert_eq!(healthy.output_token_ids.len(), 4);

    assert_eq!(e.stats().faulted_requests, 1);
}

