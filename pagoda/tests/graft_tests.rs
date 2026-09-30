// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Tests for tensor-level prefix grafting (true RadixAttention): a backend
//! that retains finished sessions' KV must let later requests start from the
//! cached prefix instead of recomputing it, without ever changing outputs,
//! and must never retain KV from faulted sequences.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pagoda::{
    ByteTokenizer, CacheBackend, Engine, EngineConfig, FinishReason, ModelEngine, ModelSession,
    NGramModel, SamplingParams, Tokenizer, WriteRequest,
};

type TestEngine = Engine<GraftModel, ByteTokenizer>;

#[derive(Default)]
struct Counters {
    /// Sessions created from a grafted KV prefix.
    grafts: usize,
    /// Finished sessions offered into the vault.
    offers: usize,
    /// Per-session forwards with a non-empty feed.
    single_forwards: usize,
}

/// An n-gram model with a token-path keyed KV vault. A session's "KV" is its
/// fed token history, so a graft is just a cloned prefix of that history --
/// the mock stands in for a sliced tensor cache.
struct GraftModel {
    inner: NGramModel,
    vault: Arc<Mutex<HashMap<Vec<u32>, Vec<u32>>>>,
    grafting: bool,
    poisoned: Arc<AtomicBool>,
    counters: Arc<Mutex<Counters>>,
}

type Vault = Arc<Mutex<HashMap<Vec<u32>, Vec<u32>>>>;

struct GraftSession {
    inner: NGramModel,
    history: Vec<u32>,
    last_logits: Option<Vec<f32>>,
    poisoned: Arc<AtomicBool>,
    counters: Arc<Mutex<Counters>>,
}

impl GraftModel {
    fn new(grafting: bool) -> (Self, Vault, Arc<AtomicBool>, Arc<Mutex<Counters>>) {
        let vault: Vault = Arc::new(Mutex::new(HashMap::new()));
        let poisoned = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(Mutex::new(Counters::default()));
        (
            Self {
                inner: NGramModel::default(),
                vault: Arc::clone(&vault),
                grafting,
                poisoned: Arc::clone(&poisoned),
                counters: Arc::clone(&counters),
            },
            vault,
            poisoned,
            counters,
        )
    }
}

impl ModelEngine for GraftModel {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn forward(&self, context: &[u32]) -> Vec<f32> {
        self.inner.forward(context)
    }

    fn name(&self) -> &'static str {
        "graft-ngram"
    }

    fn begin_session(&self) -> Option<Box<dyn ModelSession>> {
        Some(Box::new(GraftSession {
            inner: NGramModel::default(),
            history: Vec::new(),
            last_logits: None,
            poisoned: Arc::clone(&self.poisoned),
            counters: Arc::clone(&self.counters),
        }))
    }

    fn graft_session(&self, prompt: &[u32], max_len: usize) -> Option<Box<dyn ModelSession>> {
        if !self.grafting {
            return None;
        }
        let cap = max_len.min(prompt.len());
        let vault = self.vault.lock().unwrap();
        let best = vault
            .keys()
            .filter(|k| k.len() <= prompt.len() && k.as_slice() == &prompt[..k.len()])
            .max_by_key(|k| k.len().min(cap))?
            .clone();
        let covered = best.len().min(cap);
        if covered == 0 {
            return None;
        }
        let history = vault.get(&best)?[..covered].to_vec();
        drop(vault);
        self.counters.lock().unwrap().grafts += 1;
        Some(Box::new(GraftSession {
            inner: NGramModel::default(),
            history,
            last_logits: None,
            poisoned: Arc::clone(&self.poisoned),
            counters: Arc::clone(&self.counters),
        }))
    }

    fn offer_session_kv(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        session: Box<dyn ModelSession>,
    ) {
        if !self.grafting {
            return;
        }
        let Some(gs) = session
            .as_any()
            .and_then(|a| a.downcast_ref::<GraftSession>())
        else {
            return;
        };
        let fed = gs.history.len().min(tokens.len());
        if fed == 0 {
            return;
        }
        let mut vault = self.vault.lock().unwrap();
        if prompt_len.min(fed) > 0 {
            vault.insert(
                tokens[..prompt_len.min(fed)].to_vec(),
                gs.history[..prompt_len.min(fed)].to_vec(),
            );
        }
        if fed > prompt_len.min(fed) {
            vault.insert(tokens[..fed].to_vec(), gs.history.clone());
        }
        self.counters.lock().unwrap().offers += 1;
    }
}

impl ModelSession for GraftSession {
    fn context_len(&self) -> usize {
        self.history.len()
    }

    fn forward(&mut self, new_tokens: &[u32]) -> Vec<f32> {
        if self.poisoned.swap(false, Ordering::SeqCst) {
            return vec![f32::NAN; self.inner.vocab_size()];
        }
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

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

fn config(cache_backend: CacheBackend) -> EngineConfig {
    EngineConfig {
        num_kv_blocks: 256,
        block_size: 8,
        max_running_requests: 32,
        max_prefill_tokens_per_step: 1024,
        seed: 0,
        cache_backend,
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

fn engine(
    grafting: bool,
    cache_backend: CacheBackend,
) -> (TestEngine, Vault, Arc<AtomicBool>, Arc<Mutex<Counters>>) {
    let (model, vault, poisoned, counters) = GraftModel::new(grafting);
    (
        Engine::new(ByteTokenizer::default(), model, config(cache_backend)),
        vault,
        poisoned,
        counters,
    )
}

/// A repeat request must graft the first request's prompt KV: it skips
/// `prompt_len - 1` physical prompt tokens (the cap keeps one fed token so
/// first-step logits exist) and produces the same output as a cold engine.
#[test]
fn warm_request_grafts_prompt_kv() {
    let (mut e, _, _, counters) = engine(true, CacheBackend::Radix);
    let req = || WriteRequest::new("the shared system prompt of an agent", greedy(8));

    let first = e.generate_batch(&[req()]);
    assert_eq!(first[0].finish_reason, FinishReason::Length);
    assert_eq!(e.stats().model_graft_tokens, 0, "cold request grafts nothing");

    let second = e.generate_batch(&[req()]);
    let prompt_len = ByteTokenizer::default()
        .encode("the shared system prompt of an agent")
        .len();
    assert_eq!(
        e.stats().model_graft_tokens as usize,
        prompt_len - 1,
        "warm request should graft all but one prompt token"
    );
    assert_eq!(first[0].output_token_ids, second[0].output_token_ids);
    let c = counters.lock().unwrap();
    assert_eq!(c.grafts, 1);
    assert!(c.offers >= 2, "both finished sequences offered their KV");
}

/// Grafting must never change what the model says: a mixed workload with
/// shared prefixes produces token-identical outputs on grafting and
/// non-grafting engines.
#[test]
fn graft_preserves_outputs() {
    // Wave 1 populates the vault; wave 2 repeats one prompt exactly and adds
    // a fresh one, so grafting applies mid-workload.
    let wave1: Vec<WriteRequest> = [
        "agent alpha plans the task and",
        "agent alpha plans the task but",
        "agent beta reviews the plan and",
    ]
    .iter()
    .map(|p| WriteRequest::new(*p, greedy(6)))
    .collect();
    let wave2: Vec<WriteRequest> = [
        "agent alpha plans the task and", // exact repeat of the first
        "a completely different prompt",
    ]
    .iter()
    .map(|p| WriteRequest::new(*p, greedy(6)))
    .collect();

    let (mut grafting, _, _, _) = engine(true, CacheBackend::Radix);
    let mut grafted = grafting.generate_batch(&wave1);
    grafted.extend(grafting.generate_batch(&wave2));

    let (mut cold, _, _, _) = engine(false, CacheBackend::Radix);
    let mut plain = cold.generate_batch(&wave1);
    plain.extend(cold.generate_batch(&wave2));

    for (a, b) in grafted.iter().zip(&plain) {
        assert_eq!(a.output_token_ids, b.output_token_ids);
        assert_eq!(a.finish_reason, b.finish_reason);
    }
    assert!(grafting.stats().model_graft_tokens > 0);
    assert_eq!(cold.stats().model_graft_tokens, 0);
}

/// A faulted sequence's KV may be corrupt: it must never enter the vault.
/// The next identical request grafts nothing; once that one completes
/// cleanly, its KV becomes graftable.
#[test]
fn faulted_sequence_kv_is_not_retained() {
    let (mut e, vault, poisoned, _) = engine(true, CacheBackend::Radix);
    let req = || WriteRequest::new("a prompt whose model run faults", greedy(8));
    let prompt_len = ByteTokenizer::default()
        .encode("a prompt whose model run faults")
        .len();

    poisoned.store(true, Ordering::SeqCst);
    let faulted = e.generate_batch(&[req()]);
    assert_eq!(faulted[0].finish_reason, FinishReason::Fault);
    assert_eq!(
        vault.lock().unwrap().len(),
        0,
        "faulted KV must be dropped"
    );

    let clean = e.generate_batch(&[req()]);
    assert_eq!(clean[0].finish_reason, FinishReason::Length);
    assert_eq!(e.stats().model_graft_tokens, 0, "nothing graftable yet");

    let warm = e.generate_batch(&[req()]);
    assert_eq!(warm[0].finish_reason, FinishReason::Length);
    assert_eq!(e.stats().model_graft_tokens as usize, prompt_len - 1);
    assert_eq!(clean[0].output_token_ids, warm[0].output_token_ids);
}

/// Grafting is orthogonal to the engine's logical prefix cache: the APC
/// (block-hash) backend gets the same graft hits and identical outputs.
#[test]
fn graft_works_with_apc_backend() {
    let req = || WriteRequest::new("apc backend still grafts tensor prefixes", greedy(6));
    let (mut e, _, _, _) = engine(true, CacheBackend::Apc);
    let first = e.generate_batch(&[req()]);
    let second = e.generate_batch(&[req()]);
    let prompt_len = ByteTokenizer::default()
        .encode("apc backend still grafts tensor prefixes")
        .len();
    assert_eq!(e.stats().model_graft_tokens as usize, prompt_len - 1);
    assert_eq!(first[0].output_token_ids, second[0].output_token_ids);
}
