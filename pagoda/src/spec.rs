// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: AGPL-3.0-only
//! Public data model shared across the runtime and the SGLang-style frontend.

use std::fmt;

use crate::grammar::Grammar;

/// Token identifier. A `u32` to match the vocabulary sizes of mainstream
/// tokenizers (HF BPE, GGUF, SentencePiece, ...).
pub type TokenId = u32;

/// Why a request stopped generating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token or stop string was produced.
    Stop,
    /// Reached `max_tokens`.
    Length,
    /// The model backend produced invalid logits for this request; generation
    /// was aborted without affecting other requests in the batch.
    Fault,
    /// The request was rejected during admission (never started generating).
    Rejected,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::Fault => write!(f, "fault"),
            FinishReason::Rejected => write!(f, "rejected"),
        }
    }
}

/// Why the engine refused to admit a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// The waiting queue was full.
    QueueFull,
    /// `prompt + max_tokens` exceeded the engine's per-request token cap.
    TooLong,
    /// The prompt tokenized to zero tokens.
    EmptyPrompt,
    /// The request uses a feature this backend does not support (e.g. a
    /// byte-level grammar on a non-byte tokenizer).
    Unsupported,
}

impl fmt::Display for RejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RejectReason::QueueFull => write!(f, "queue_full"),
            RejectReason::TooLong => write!(f, "too_long"),
            RejectReason::EmptyPrompt => write!(f, "empty_prompt"),
            RejectReason::Unsupported => write!(f, "unsupported"),
        }
    }
}

/// Sampling knobs. Mirrors the subset of
/// `sglang.srt.sampling_params.SamplingParams` implemented by this prototype.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    /// Maximum number of newly generated tokens.
    pub max_tokens: usize,
    /// Softmax temperature; values below `1e-5` select greedy decoding.
    pub temperature: f32,
    /// Nucleus (top-p) cumulative-probability threshold.
    pub top_p: f32,
    /// Keep only the `top_k` most probable tokens (0/`usize::MAX` disables).
    pub top_k: usize,
    /// Decoded substrings that stop generation when the output ends with them.
    pub stop: Vec<String>,
    /// Raw token ids that stop generation.
    pub stop_token_ids: Vec<TokenId>,
    /// Frequency penalty applied per occurrence of a token (`logits -= f * count`).
    pub frequency_penalty: f32,
    /// Presence penalty applied to any generated token (`logits -= p`).
    pub presence_penalty: f32,
    /// RNG seed for reproducible sampling.
    pub seed: u64,
    /// Optional constrained-decoding grammar. When present, every sampled
    /// byte token is restricted to a legal continuation of the partial output.
    pub grammar: Option<Grammar>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.0,
            top_p: 1.0,
            top_k: usize::MAX,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 42,
            grammar: None,
        }
    }
}

/// A single request into the offline engine.
#[derive(Clone, Debug)]
pub struct WriteRequest {
    pub text: String,
    pub sampling: SamplingParams,
}

impl WriteRequest {
    pub fn new(text: impl Into<String>, sampling: SamplingParams) -> Self {
        Self {
            text: text.into(),
            sampling,
        }
    }
}

/// Result of one generation.
#[derive(Clone, Debug)]
pub struct GenerationOutput {
    /// Decoded text of only the newly generated tokens.
    pub text: String,
    /// Decoded prefix + generation, when available.
    pub full_text: Option<String>,
    /// The generated token ids (excluding the prompt).
    pub output_token_ids: Vec<TokenId>,
    pub finish_reason: FinishReason,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Number of prompt tokens served from the Radix prefix cache.
    pub prefix_hit_tokens: usize,
    /// Forward-pass units charged to this request (prefill + decode).
    pub forward_count: usize,
    /// Set when the request was rejected before generation; `None` otherwise.
    pub rejection: Option<RejectReason>,
}
