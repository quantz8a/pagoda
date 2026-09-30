// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Tests for batched decode: the engine must fuse same-shape session feeds
//! (exactly one fresh token each) into a single model call when the backend
//! supports it, keep outputs token-identical to the per-sequence path, and
//! degrade to per-session forwards when the backend cannot batch.

use std::sync::{Arc, Mutex};

use pagoda::{
    ByteTokenizer, Engine, EngineConfig, FinishReason, ModelEngine, ModelSession, NGramModel,
    SamplingParams, WriteRequest,
};

type TestEngine = Engine<BatchableModel, ByteTokenizer>;

#[derive(Default)]
struct Counters {
    /// Physical batched calls (`session_forward_batch` invocations).
    batch_calls: usize,
    /// Sessions served inside batched calls.
    batch_sessions_served: usize,
    /// Per-session (unbatched) forwards with a non-empty feed.
    single_forwards: usize,
    /// Stateless full-replay forwards.
    stateless_forwards: usize,
}

/// An n-gram model whose sessions can advance in one batched call. The
/// batched path computes exactly what per-session forwards would, so greedy
/// outputs must match the unbatched engine token-for-token.
struct BatchableModel {
    inner: NGramModel,
    batching: bool,
    counters: Arc<Mutex<Counters>>,
}

struct BatchableSession {
    inner: NGramModel,
    history: Vec<u32>,
    last_logits: Option<Vec<f32>>,
    counters: Arc<Mutex<Counters>>,
}

impl BatchableModel {
    fn new(batching: bool) -> (Self, Arc<Mutex<Counters>>) {
        let counters = Arc::new(Mutex::new(Counters::default()));
        (
            Self {
                inner: NGramModel::default(),
                batching,
                counters: Arc::clone(&counters),
            },
            counters,
        )
    }
}

impl ModelEngine for BatchableModel {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn forward(&self, context: &[u32]) -> Vec<f32> {
        self.counters.lock().unwrap().stateless_forwards += 1;
        self.inner.forward(context)
    }

    fn name(&self) -> &'static str {
        "batchable-ngram"
    }

    fn begin_session(&self) -> Option<Box<dyn ModelSession>> {
        Some(Box::new(BatchableSession {
            inner: NGramModel::default(),
            history: Vec::new(),
            last_logits: None,
            counters: Arc::clone(&self.counters),
        }))
    }

    fn session_forward_batch(
        &self,
        sessions: &mut [Box<dyn ModelSession>],
        feeds: &[&[u32]],
    ) -> Option<Vec<Vec<f32>>> {
        if !self.batching {
            return None;
        }
        assert_eq!(sessions.len(), feeds.len());
        {
            let mut c = self.counters.lock().unwrap();
            c.batch_calls += 1;
            c.batch_sessions_served += sessions.len();
        }
        let mut out = Vec::with_capacity(sessions.len());
        for (session, feed) in sessions.iter_mut().zip(feeds) {
            let session = session
                .as_any_mut()
                .and_then(|s| s.downcast_mut::<BatchableSession>())
                .expect("batchable model received a foreign session");
            assert!(
                !feed.is_empty(),
                "engine must not route empty feeds into the batch"
            );
            session.history.extend_from_slice(feed);
            let logits = session.inner.forward(&session.history);
            session.last_logits = Some(logits.clone());
            out.push(logits);
        }
        Some(out)
    }
}

impl ModelSession for BatchableSession {
    fn context_len(&self) -> usize {
        self.history.len()
    }

    fn forward(&mut self, new_tokens: &[u32]) -> Vec<f32> {
        if !new_tokens.is_empty() {
            self.counters.lock().unwrap().single_forwards += 1;
            self.history.extend_from_slice(new_tokens);
            let logits = self.inner.forward(&self.history);
            self.last_logits = Some(logits.clone());
            return logits;
        }
        self.last_logits.clone().unwrap_or_default()
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn fork(&self) -> Option<Box<dyn ModelSession>> {
        Some(Box::new(Self {
            inner: NGramModel::default(),
            history: self.history.clone(),
            last_logits: self.last_logits.clone(),
            counters: Arc::clone(&self.counters),
        }))
    }
}

fn config(max_prefill_tokens_per_step: usize) -> EngineConfig {
    EngineConfig {
        num_kv_blocks: 256,
        block_size: 8,
        max_running_requests: 32,
        max_prefill_tokens_per_step,
        seed: 0,
        ..EngineConfig::default()
    }
}

fn greedy(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        max_tokens,
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

fn engine(batching: bool, max_prefill_tokens_per_step: usize) -> (TestEngine, Arc<Mutex<Counters>>) {
    let (model, counters) = BatchableModel::new(batching);
    (
        Engine::new(ByteTokenizer::default(), model, config(max_prefill_tokens_per_step)),
        counters,
    )
}

fn requests(n: usize, max_tokens: usize) -> Vec<WriteRequest> {
    (0..n)
        .map(|i| {
            WriteRequest::new(
                format!("request number {i} asks about the weather today and"),
                greedy(max_tokens),
            )
        })
        .collect()
}

/// Batched decode fuses same-shape session feeds into one physical call per
/// step: with 6 sequences x 8 tokens the engine performs 6 lone prefill
/// forwards (first step, multi-token feeds) plus 7 batched calls, not 48
/// individual forwards.
#[test]
fn batched_decode_shares_model_calls() {
    let (mut e, counters) = engine(true, 1024);
    let outs = e.generate_batch(&requests(6, 8));
    assert_eq!(outs.len(), 6);
    assert!(
        outs.iter().all(|o| o.finish_reason == FinishReason::Length),
        "every request should finish at its token budget"
    );

    let stats = e.stats();
    assert_eq!(stats.total_decode_steps, 6 * 8);
    let c = counters.lock().unwrap();
    assert_eq!(c.single_forwards, 6, "one lone prefill forward per sequence");
    assert_eq!(c.batch_calls, 7, "one batched call per remaining step");
    assert_eq!(c.batch_sessions_served, 6 * 7);
    assert_eq!(c.stateless_forwards, 0);
    assert_eq!(stats.total_decode_calls, 13);
    assert!(
        stats.decode_batch_factor() > 3.0,
        "batch factor should approach the batch size, got {:.2}",
        stats.decode_batch_factor()
    );
}

/// Batching must never change what the model says: greedy outputs from a
/// batching backend are token-identical to an unbatched backend, even with
/// staggered prefill joins (small per-step prefill budget) mixing feed
/// shapes inside the same decode step.
#[test]
fn batched_and_unbatched_produce_identical_outputs() {
    for max_prefill in [1024, 5] {
        let (mut batched, _) = engine(true, max_prefill);
        let batched_outs = batched.generate_batch(&requests(6, 8));

        let (mut lone, counters) = engine(false, max_prefill);
        let lone_outs = lone.generate_batch(&requests(6, 8));

        for (a, b) in batched_outs.iter().zip(&lone_outs) {
            assert_eq!(
                a.output_token_ids, b.output_token_ids,
                "batched decode changed the outputs (prefill budget {max_prefill})"
            );
            assert_eq!(a.finish_reason, b.finish_reason);
        }
        let c = counters.lock().unwrap();
        assert_eq!(c.batch_calls, 0, "unbatchable backend must stay unbatched");
        let stats = lone.stats();
        assert_eq!(stats.total_decode_calls, stats.total_decode_steps);
        assert_eq!(stats.decode_batch_factor(), 1.0);
    }
}

/// A branch forked from a checkpoint starts with an empty feed: the engine
/// must keep it off the batch path and serve it from the session's cached
/// logits, alongside ordinary batched sequences.
#[test]
fn empty_feed_branch_still_served() {
    let (mut e, counters) = engine(true, 1024);
    let trunk = e.create_checkpoint("the shared trunk of an agent program");
    let a = e
        .generate_from_checkpoint(trunk, "", greedy(4))
        .expect("checkpoint exists");
    let b = e
        .generate_from_checkpoint(trunk, "", greedy(4))
        .expect("checkpoint exists");
    assert_eq!(a.finish_reason, FinishReason::Length);
    assert_eq!(
        a.output_token_ids, b.output_token_ids,
        "branches off the same trunk must agree"
    );
    assert_eq!(a.prefix_hit_tokens, e.checkpoint_tokens(trunk).unwrap());

    // The trunk session feeds once; branch forks reuse its cached logits for
    // the empty first step instead of paying a model call.
    let c = counters.lock().unwrap();
    assert_eq!(c.stateless_forwards, 0);
}
