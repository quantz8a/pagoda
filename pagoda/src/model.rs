// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Model backend abstraction plus a deterministic n-gram toy language model.

use std::collections::HashMap;

use crate::tokenizer::{BYTE_VOCAB_SIZE, FIRST_BYTE_ID};

/// A pluggable "forward" backend. A real deployment implements this with a
/// tensor engine (GPU kernels, GGML/Candle, an external inference server, ...).
pub trait ModelEngine: Send + Sync {
    /// Vocabulary width of the produced logits.
    fn vocab_size(&self) -> usize;
    /// Produce next-token logits given the full token context so far.
    fn forward(&self, context: &[u32]) -> Vec<f32>;
    fn name(&self) -> &'static str;

    /// Advance many decode sessions in **one** model call.
    ///
    /// `feeds[i]` is the not-yet-seen token suffix of session `i` (usually
    /// one freshly sampled token during continuous batching). On success the
    /// return value holds one last-position logits vector per session, in the
    /// same order. Return `None` when this backend cannot batch — the engine
    /// then loops over [`ModelSession::forward`], which is always correct.
    ///
    /// Why it matters: decode is weight-bandwidth-bound, so B single-sequence
    /// forwards cost roughly B full weight reads. One batched `[B, 1]`
    /// forward reads the weights once per step for the whole batch.
    ///
    /// The sessions arrive as the boxed trait objects the engine owns (a
    /// `&mut [&mut dyn ModelSession]` slice would force every borrow to the
    /// trait object's `'static` bound through `&mut` invariance).
    fn session_forward_batch(
        &self,
        _sessions: &mut [Box<dyn ModelSession>],
        _feeds: &[&[u32]],
    ) -> Option<Vec<Vec<f32>>> {
        None
    }

    /// Try to mint a session whose KV cache already covers a prefix of
    /// `prompt` — the tensor-level half of prefix caching (true
    /// RadixAttention): the covered tokens are never re-fed to the model.
    ///
    /// `max_len` caps the covered length; the engine passes
    /// `prompt.len() - 1` so at least one token is always fed (which is what
    /// produces the first-step logits). The covered length is read back from
    /// `session.context_len()`. Causality makes this exact: the KV at
    /// position `i` depends only on tokens `0..=i`, so a cached prefix is
    /// bit-identical to a recomputed one.
    ///
    /// Default `None`: no cross-request KV reuse; the engine falls back to a
    /// fresh session (or the stateless path) and stays correct.
    fn graft_session(&self, _prompt: &[u32], _max_len: usize) -> Option<Box<dyn ModelSession>> {
        None
    }

    /// Offer a finished sequence's session KV for future grafts. `tokens` is
    /// the full token path (prompt + output); `prompt_len` marks where the
    /// prompt ended so the backend can index both the prompt prefix and the
    /// full path as graftable. The engine calls this when a sequence
    /// completes (never for faulted ones — their KV may be corrupt).
    /// Default: no-op (the backend retains nothing).
    fn offer_session_kv(
        &self,
        _tokens: &[u32],
        _prompt_len: usize,
        _session: Box<dyn ModelSession>,
    ) {
    }

    /// Hand out an incremental decode session for one sequence.
    ///
    /// Stateless backends keep the default (`None`) and the engine replays the
    /// full context into [`ModelEngine::forward`] every step — correct but
    /// O(n^2). A backend with a real key/value cache overrides this so the
    /// engine can feed each token exactly once (see [`ModelSession`]).
    fn begin_session(&self) -> Option<Box<dyn ModelSession>> {
        None
    }
}

/// A stateful, per-sequence decode session backed by a real KV cache.
///
/// Contract with the engine:
///
/// * The engine feeds exactly the tokens the session has not seen yet:
///   `context[session.context_len()..]`. The first call carries the whole
///   prompt; later calls usually carry one freshly sampled token, but any
///   non-empty suffix must be accepted.
/// * [`ModelSession::forward`] always returns the logits of the **last fed
///   position**. An empty `new_tokens` slice must return the most recently
///   computed logits again (implementations cache them), so a branch that
///   adds nothing yet can still sample its first token.
/// * Sessions are request-private: the engine never shares one session
///   between sequences. Branching goes through [`ModelSession::fork`].
pub trait ModelSession: Send {
    /// Tokens already pushed through the model.
    fn context_len(&self) -> usize;
    /// Feed the not-yet-seen suffix; return last-position logits.
    fn forward(&mut self, new_tokens: &[u32]) -> Vec<f32>;
    /// Downcast hook so a batching backend can recover its concrete session
    /// type inside [`ModelEngine::session_forward_batch`]. Backends that
    /// never batch keep the default (`None`).
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
    /// Read-only downcast hook so the backend can retain a finished
    /// session's KV ([`ModelEngine::offer_session_kv`]). Default `None`.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
    /// Cheap snapshot for agent-tree branching: the fork shares the
    /// already-computed KV state and continues independently. Backends whose
    /// cache cannot be snapshotted keep the default (`None`); the engine then
    /// falls back to a fresh session that replays the prefix.
    fn fork(&self) -> Option<Box<dyn ModelSession>> {
        None
    }
}

/// Deterministic byte n-gram language model trained on an embedded corpus.
/// Greedy decoding produces a stable corpus-plausible continuation; sampling
/// (temperature > 0) produces varied, still corpus-shaped text. It is a
/// stand-in that keeps the engine testable without GPU compute, not a trained
/// LLM.
pub struct NGramModel {
    order: usize,
    vocab_size: usize,
    /// Context bytes of length `1..=order` to smoothed next-byte log-counts.
    table: HashMap<Vec<u8>, Vec<f32>>,
    /// Unigram fallback (ln(count + 1)) for out-of-corpus contexts.
    unigram: Vec<f32>,
}

impl NGramModel {
    pub fn new(order: usize, corpus: &str) -> Self {
        assert!(order >= 1);
        let mut table: HashMap<Vec<u8>, Vec<f32>> = HashMap::new();
        let bytes = corpus.as_bytes();
        // Train contexts of every order so lookup can back off gracefully.
        for k in 1..=order {
            for window in bytes.windows(k + 1) {
                let ctx = window[..k].to_vec();
                let next = window[k];
                let counts = table
                    .entry(ctx)
                    .or_insert_with(|| vec![1.0f32; BYTE_VOCAB_SIZE]); // add-one
                counts[next as usize] += 1.0;
            }
        }

        // Unigram distribution for the fallback path.
        let mut raw_counts = [1.0f32; BYTE_VOCAB_SIZE];
        for &b in bytes {
            raw_counts[b as usize] += 1.0;
        }
        let mut unigram: Vec<f32> = raw_counts.iter().map(|&c| c.ln()).collect();
        // Keep generated bytes printable / valid: suppress control bytes and
        // non-ASCII (which would break the ASCII corpus shape).
        for i in 0..=31 {
            if i != b'\n' as usize && i != b'\t' as usize {
                unigram[i] = -1e9;
            }
        }
        for i in 127..BYTE_VOCAB_SIZE {
            unigram[i] = -1e9;
        }

        Self {
            order,
            vocab_size: 3 + BYTE_VOCAB_SIZE,
            table,
            unigram,
        }
    }

    pub fn order(&self) -> usize {
        self.order
    }

    /// Back off from the longest available context, ending in the unigram.
    fn counts_for(&self, context: &[u32]) -> (Option<&Vec<f32>>, Option<u8>) {
        let recover = |t: u32| -> Option<u8> {
            if t >= FIRST_BYTE_ID && t < FIRST_BYTE_ID + BYTE_VOCAB_SIZE as u32 {
                Some((t - FIRST_BYTE_ID) as u8)
            } else {
                None
            }
        };

        for k in (1..=self.order).rev() {
            let tail: Option<Vec<u8>> = context
                .iter()
                .rev()
                .take(k)
                .map(|&t| recover(t))
                .collect::<Option<Vec<u8>>>()
                .map(|mut v| {
                    v.reverse();
                    v
                });
            if let Some(key) = tail {
                if let Some(counts) = self.table.get(&key) {
                    return (Some(counts), key.last().copied());
                }
            }
        }
        (None, None)
    }
}

/// Force the model to emit only printable ASCII (plus tab/newline) so the byte
/// tokenizer always reconstructs clean UTF-8.
fn mask_non_printable(logits: &mut [f32]) {
    for i in 0..BYTE_VOCAB_SIZE {
        let keep = i == b'\n' as usize || i == b'\t' as usize || (32..127).contains(&i);
        if !keep {
            logits[FIRST_BYTE_ID as usize + i] = -1e9;
        }
    }
}

impl Default for NGramModel {
    fn default() -> Self {
        NGramModel::new(4, DEFAULT_CORPUS)
    }
}

impl ModelEngine for NGramModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn forward(&self, context: &[u32]) -> Vec<f32> {
        let mut logits = vec![-1e9f32; self.vocab_size];
        match self.counts_for(context) {
            (Some(counts), _) => {
                for (b, &c) in counts.iter().enumerate() {
                    logits[FIRST_BYTE_ID as usize + b] = c.ln();
                }
            }
            (None, _) => {
                for (b, &c) in self.unigram.iter().enumerate() {
                    logits[FIRST_BYTE_ID as usize + b] = c;
                }
            }
        }
        mask_non_printable(&mut logits);
        logits
    }

    fn name(&self) -> &'static str {
        "ngram"
    }
}

/// A short English corpus used to train the reference toy model.
pub const DEFAULT_CORPUS: &str = "OpenAI serves large language models at scale \
with prefixed attention and paged memory to make inference fast and cheap. \
SGLang is a fast serving framework for large language models and vision \
language models. It uses a radix tree to cache the key and value tensors of \
repeated prompt prefixes, so common instructions are computed only once. \
The scheduler runs continuous batching to merge many requests into one forward \
pass and keep the hardware busy. A paged key value cache stores memory in fixed \
size blocks and uses copy on write when sequences diverge. The frontend language \
lets users write programs with generation, selection, and fork for parallel \
search. Sampling controls temperature and top p to trade quality and diversity. \
The server exposes OpenAI compatible chat completions and a native generate \
endpoint. Reliability comes from metrics and structured logging. \
";

