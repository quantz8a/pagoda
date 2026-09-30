// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: AGPL-3.0-only
//! Tests for the incremental KV-session contract: the engine must feed each
//! token to the model exactly once (prompt once, then one fresh token per
//! step), keep sessions private per sequence, fork checkpoint trunks instead
//! of recomputing them, and degrade safely when a session misbehaves.

use std::sync::{Arc, Mutex};

use pagoda::{
    ByteTokenizer, Engine, EngineConfig, FinishReason, ModelEngine, ModelSession, NGramModel,
    SamplingParams, Tokenizer, WriteRequest,
};

type TestEngine = Engine<SessionfulModel, ByteTokenizer>;

/// A model whose sessions record exactly how many tokens each `forward` call
/// received. Logits come from a deterministic inner n-gram over the
/// reconstructed full context, so session and stateless paths agree
/// token-for-token.
struct SessionfulModel {
    inner: NGramModel,
    feeds: Arc<Mutex<Vec<usize>>>,
}

impl SessionfulModel {
    fn new() -> (Self, Arc<Mutex<Vec<usize>>>) {
        let feeds = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                inner: NGramModel::default(),
                feeds: Arc::clone(&feeds),
            },
            feeds,
        )
    }
}

struct RecordingSession {
    inner: NGramModel,
    history: Vec<u32>,
    last_logits: Option<Vec<f32>>,
    feeds: Arc<Mutex<Vec<usize>>>,
}

impl ModelEngine for SessionfulModel {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn forward(&self, context: &[u32]) -> Vec<f32> {
        self.feeds.lock().unwrap().push(context.len());
        self.inner.forward(context)
    }

    fn name(&self) -> &'static str {
        "sessionful-ngram"
    }

    fn begin_session(&self) -> Option<Box<dyn ModelSession>> {
        Some(Box::new(RecordingSession {
            inner: NGramModel::default(),
            history: Vec::new(),
            last_logits: None,
            feeds: Arc::clone(&self.feeds),
        }))
    }
}

impl ModelSession for RecordingSession {
    fn context_len(&self) -> usize {
        self.history.len()
    }

    fn forward(&mut self, new_tokens: &[u32]) -> Vec<f32> {
        self.feeds.lock().unwrap().push(new_tokens.len());
        if new_tokens.is_empty() {
            return self.last_logits.clone().unwrap_or_default();
        }
        self.history.extend_from_slice(new_tokens);
        let logits = self.inner.forward(&self.history);
        self.last_logits = Some(logits.clone());
        logits
    }

    fn fork(&self) -> Option<Box<dyn ModelSession>> {
        Some(Box::new(Self {
            inner: NGramModel::default(),
            history: self.history.clone(),
            last_logits: self.last_logits.clone(),
            feeds: Arc::clone(&self.feeds),
        }))
    }
}

fn config() -> EngineConfig {
    EngineConfig {
        num_kv_blocks: 256,
        block_size: 8,
        max_running_requests: 32,
        max_prefill_tokens_per_step: 32,
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

fn engine() -> (TestEngine, Arc<Mutex<Vec<usize>>>) {
    let (model, feeds) = SessionfulModel::new();
    (Engine::new(ByteTokenizer::default(), model, config()), feeds)
}

#[test]
fn session_feeds_each_token_exactly_once() {
    let (mut e, feeds) = engine();
    let prompt = "OpenAI serves";
    let prompt_len = ByteTokenizer::default().encode(prompt).len();
    let out = e.generate(&WriteRequest::new(prompt, greedy(8)));
    assert_eq!(out.finish_reason, FinishReason::Length);

    let feeds = feeds.lock().unwrap().clone();
    let steps = out.output_token_ids.len();
    assert_eq!(feeds.len(), steps, "one forward per decode step");
    assert_eq!(feeds[0], prompt_len, "first feed is the whole prompt");
    assert!(
        feeds[1..].iter().all(|&n| n == 1),
        "every later step feeds exactly one fresh token: {feeds:?}"
    );
}

#[test]
fn session_output_matches_stateless_output() {
    let req = WriteRequest::new("SGLang is a fast serving framework", greedy(12));
    let mut stateless = pagoda::ToyEngine::toy(config());
    let expected = stateless.generate(&req);

    let (mut sessioned, _) = engine();
    let got = sessioned.generate(&req);
    assert_eq!(got.finish_reason, FinishReason::Length);
    assert_eq!(
        got.output_token_ids, expected.output_token_ids,
        "incremental session decoding must equal stateless full replay"
    );
}

#[test]
fn batched_sessions_stay_private_per_sequence() {
    let (mut e, feeds) = engine();
    let reqs = [
        WriteRequest::new("OpenAI serves", greedy(6)),
        WriteRequest::new("SGLang is", greedy(6)),
    ];
    let outs = e.generate_batch(&reqs);
    assert_eq!(outs.len(), 2);
    assert!(outs.iter().all(|o| o.finish_reason == FinishReason::Length));

    let feeds = feeds.lock().unwrap().clone();
    let prompt_feeds = feeds.iter().filter(|&&n| n > 1).count();
    let decode_feeds = feeds.iter().filter(|&&n| n == 1).count();
    assert_eq!(prompt_feeds, 2, "each sequence prefills exactly once");
    let expected_decodes: usize = outs.iter().map(|o| o.output_token_ids.len() - 1).sum();
    assert_eq!(decode_feeds, expected_decodes);

    // Cross-check against plain stateless runs.
    let mut a = pagoda::ToyEngine::toy(config());
    let mut b = pagoda::ToyEngine::toy(config());
    assert_eq!(outs[0].output_token_ids, a.generate(&reqs[0]).output_token_ids);
    assert_eq!(outs[1].output_token_ids, b.generate(&reqs[1]).output_token_ids);
}

#[test]
fn misbehaving_session_falls_back_without_panic() {
    /// A session that claims an impossible context length must be dropped,
    /// not crash the engine.
    struct Bogus;
    struct BogusSession;
    impl ModelEngine for Bogus {
        fn vocab_size(&self) -> usize {
            NGramModel::default().vocab_size()
        }
        fn forward(&self, context: &[u32]) -> Vec<f32> {
            NGramModel::default().forward(context)
        }
        fn name(&self) -> &'static str {
            "bogus"
        }
        fn begin_session(&self) -> Option<Box<dyn ModelSession>> {
            Some(Box::new(BogusSession))
        }
    }
    impl ModelSession for BogusSession {
        fn context_len(&self) -> usize {
            usize::MAX
        }
        fn forward(&mut self, _new: &[u32]) -> Vec<f32> {
            unreachable!("engine must drop the session before calling forward")
        }
    }

    let mut e = Engine::new(ByteTokenizer::default(), Bogus, config());
    let req = WriteRequest::new("OpenAI serves", greedy(6));
    let got = e.generate(&req);
    assert_eq!(got.finish_reason, FinishReason::Length);

    let mut reference = pagoda::ToyEngine::toy(config());
    assert_eq!(
        got.output_token_ids,
        reference.generate(&req).output_token_ids,
        "fallback must produce the stateless output"
    );
}

#[test]
fn checkpoint_branch_forks_kv_instead_of_recomputing_trunk() {
    let (mut e, feeds) = engine();
    let trunk = "OpenAI serves large language models";
    let trunk_len = ByteTokenizer::default().encode(trunk).len();
    let ckpt = e.create_checkpoint(trunk);

    // Creating the checkpoint prefills the trunk exactly once.
    assert_eq!(feeds.lock().unwrap().clone(), vec![trunk_len]);
    feeds.lock().unwrap().clear();

    let out = e
        .generate_from_checkpoint(ckpt, " at scale", greedy(6))
        .expect("checkpoint exists");
    assert_eq!(out.finish_reason, FinishReason::Length);
    assert_eq!(out.prefix_hit_tokens, trunk_len, "trunk is a guaranteed hit");

    let continuation_len = ByteTokenizer::default().encode(" at scale").len();
    let feeds = feeds.lock().unwrap().clone();
    assert_eq!(
        feeds[0], continuation_len,
        "branch pays only for the continuation, not the trunk"
    );
    assert!(feeds[1..].iter().all(|&n| n == 1));

    // Deterministic across branches of the same trunk.
    let again = e
        .generate_from_checkpoint(ckpt, " at scale", greedy(6))
        .expect("checkpoint exists");
    assert_eq!(out.output_token_ids, again.output_token_ids);
}

#[test]
fn empty_continuation_reuses_cached_trunk_logits() {
    let (mut e, feeds) = engine();
    let ckpt = e.create_checkpoint("OpenAI serves");
    feeds.lock().unwrap().clear();

    let out = e
        .generate_from_checkpoint(ckpt, "", greedy(4))
        .expect("checkpoint exists");
    assert_ne!(out.finish_reason, FinishReason::Fault);
    assert!(!out.output_token_ids.is_empty());
    // First branch step fed nothing: it sampled from the trunk's cached logits.
    assert_eq!(feeds.lock().unwrap()[0], 0);
}