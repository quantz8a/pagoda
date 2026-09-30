// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! An SGLang-inspired LLM serving runtime, implemented from scratch in Rust.
//!
//! This crate distills the architecture of SGLang (sgl-project/sglang) into a
//! compact, dependency-free reference implementation:
//!
//! * [`engine`]     — offline batch engine + continuous-batching scheduler
//! * [`radix_cache`]— RadixAttention-style prefix cache (compute-skip)
//! * [`kv_cache`]   — paged KV cache with reference counting and copy-on-write
//! * [`sampler`]    — temperature / top-k / top-p / penalty sampling
//! * [`grammar`]    — regex / JSON constrained-decoding logit mask
//! * [`dsl`]        — an SGLang-style frontend language (gen/select/fork)
//! * [`model`]      — model backend trait + deterministic n-gram toy LM
//! * [`tokenizer`]  — tokenizer trait + deterministic byte-level tokenizer
//! * [`server`]     — minimal dependency-free HTTP serving frontend
//!
//! The implementation deliberately keeps external dependencies to zero so the
//! whole runtime compiles and runs entirely offline. See `docs/DESIGN.md` for
//! the architecture, the mapping to SGLang, and the parity roadmap.

#![allow(dead_code)]

pub mod apc;
pub mod dsl;
pub mod engine;
pub mod grammar;
pub mod json;
pub mod kv_cache;
pub mod model;
pub mod radix_cache;
pub mod rng;
pub mod sampler;
pub mod server;
pub mod spec;
pub mod tokenizer;

pub use apc::ApcCache;
pub use dsl::{Op, Program, StreamResult};
pub use engine::{CacheBackend, CheckpointId, Engine, EngineConfig, EngineStats, SchedulePolicy, ToyEngine};
pub use grammar::{mask_logits, ByteRegex, Grammar};
pub use kv_cache::{BlockId, PagedKvCache};
pub use model::{ModelEngine, ModelSession, NGramModel};
pub use radix_cache::RadixCache;
pub use sampler::Sampler;
pub use spec::{FinishReason, GenerationOutput, RejectReason, SamplingParams, TokenId, WriteRequest};
pub use tokenizer::{ByteTokenizer, Tokenizer};
