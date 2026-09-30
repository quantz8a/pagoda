// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Real tokenizer + weight backends for [`pagoda`].
//!
//! `pagoda` is deliberately dependency-free and ships only deterministic toy
//! stand-ins. This companion crate plugs **real** artifacts back in behind the
//! exact same trait surface:
//!
//! * [`HfTokenizer`] — a HuggingFace BPE / Unigram tokenizer backed by the
//!   `tokenizers` crate (`tokenizer.json`).
//! * [`CandleModel`] — a [`pagoda::ModelEngine`] adapter over a Candle
//!   causal LM (Llama family), loading real `config.json` + safetensors
//!   weights and returning raw logits.
//! * [`CandleSession`] — the incremental half: each engine sequence gets its
//!   own KV-cache session and feeds every token exactly once (prompt once,
//!   then one fresh token per decode step) instead of replaying the full
//!   context. Sessions fork cheaply (shared tensor storage), which is what
//!   powers checkpoint branching for agent-tree workloads.
//! * **Batched decode** — same-shape sessions (one fresh token each) advance
//!   in a single `[B, 1]` forward through the vendored [`llama`] model, so a
//!   batch reads the weights once per step instead of once per sequence.
//! * **Cross-request KV grafting** — finished sessions' KV snapshots live in
//!   a bounded [`KvVault`] keyed by token path; a later request whose prompt
//!   shares a prefix grafts the cached tensors instead of recomputing them
//!   (the physical half of RadixAttention).
//!
//! The model forward is a vendored, batch-capable Llama built on candle-nn
//! primitives (see [`mod@llama`]): candle-transformers' own `Llama` keeps its
//! KV cache private and drives one sequence per forward, which cannot batch.
//!
//! # Sandbox status
//!
//! Unlike `pagoda`, this crate **needs network access** to fetch its crates
//! (`tokenizers`, Candle, `hf-hub`, ...) and, at runtime, its `tokenizer.json`
//! and safetensors weights. It is therefore **not compiled or executed** as part
//! of the offline `cargo build --offline` / `cargo test --offline` loop. Build it
//! from a networked machine with:
//!
//! ```text
//! cd pagoda-hf
//! cargo build
//! ```

#![allow(unused_unsafe)] // `VarBuilder::from_mmaped_safetensors` is `unsafe` on some Candle releases.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

pub use candle_core::{DType, Device};
pub use candle_nn::VarBuilder;
pub use candle_transformers::models::llama::LlamaConfig;

pub mod llama;
pub use llama::{OwnedLlama, SessionKv};
pub mod vault;
pub use vault::KvVault;

use pagoda::model::{ModelEngine, ModelSession};
use pagoda::tokenizer::Tokenizer as SglTokenizer;

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// A real HuggingFace tokenizer (`tokenizer.json`) implementing
/// [`pagoda::Tokenizer`].
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    eos: u32,
    bos: Option<u32>,
    vocab_size: usize,
}

impl HfTokenizer {
    /// Load a `tokenizer.json` from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;
        Self::finish(inner)
    }

    /// Download `tokenizer.json` from a HuggingFace repo and load it.
    pub fn from_hub(repo: &str) -> Result<Self> {
        let path = Self::download(repo, "tokenizer.json")?;
        Self::from_file(path)
    }

    /// Download one file from a HuggingFace repo into the default HF cache.
    pub fn download(repo: &str, file: &str) -> Result<std::path::PathBuf> {
        let api = hf_hub::api::sync::Api::new()
            .context("failed to initialise the HuggingFace cache")?;
        let api = api.model(repo.to_string());
        api.get(file)
            .with_context(|| format!("failed to download {repo}/{file}"))
    }

    fn finish(inner: tokenizers::Tokenizer) -> Result<Self> {
        let vocab = inner.get_vocab(true);
        let eos = ["</s>", "<|endoftext|>", "<|eot_id|>", "<|end|>"]
            .iter()
            .find_map(|tok| vocab.get(*tok).copied())
            .unwrap_or(0);
        let bos = ["<s>", "<|begin_of_text|>"]
            .iter()
            .find_map(|tok| vocab.get(*tok).copied());
        let vocab_size = inner.get_vocab_size(true);
        Ok(Self {
            inner,
            eos,
            bos,
            vocab_size,
        })
    }
}

impl SglTokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let ids = self
            .inner
            .encode(text, true)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default();
        if let Some(bos) = self.bos {
            if ids.first() != Some(&bos) {
                let mut out = Vec::with_capacity(ids.len() + 1);
                out.push(bos);
                out.extend(ids);
                return out;
            }
        }
        ids
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.inner.decode(tokens, true).unwrap_or_default()
    }

    fn eos_token_id(&self) -> u32 {
        self.eos
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn name(&self) -> &'static str {
        "huggingface"
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A Candle causal-LM forward contract. Kept as a thin trait so the exact
/// model family (and its per-session KV cache type) lives in one spot and the
/// [`ModelEngine`] adapter below stays independent of it.
///
/// The KV cache is explicit and external: one cache per decode session, minted
/// by [`CandleCausalLM::new_cache`]. `Cache: Clone` is required so checkpoint
/// trunks can fork branches over shared tensor storage.
pub trait CandleCausalLM: Send + Sync {
    /// Per-session KV cache type.
    type Cache: Send + Clone;
    /// Mint a fresh, empty KV cache for one decode session.
    fn new_cache(&self, device: &Device) -> Result<Self::Cache>;
    /// Tokens already pushed through `cache`.
    fn cache_len(cache: &Self::Cache) -> usize;
    /// Push `tokens` through the model at the cache's current position and
    /// return the last-position logits. `tokens` may be a whole prompt
    /// (prefill) or any non-empty suffix.
    fn forward_tokens(&self, tokens: &[u32], cache: &mut Self::Cache) -> Result<Vec<f32>>;
    /// Advance many same-shape sessions (exactly one fresh token each) in a
    /// single batched forward, returning one logits vector per session in
    /// order. Default `None`: the engine falls back to per-session forwards.
    fn session_forward_batch(
        &self,
        _sessions: &mut [Box<dyn ModelSession>],
        _feeds: &[&[u32]],
    ) -> Option<Vec<Vec<f32>>> {
        None
    }
    /// Slice `cache` to its first `len` tokens for a prefix graft
    /// (tensor-level RadixAttention). Causal attention makes a prefix view
    /// bit-identical to a freshly computed one. Default `None`: the backend
    /// offers no cross-request grafts.
    fn cache_prefix(_cache: &Self::Cache, _len: usize) -> Option<Self::Cache> {
        None
    }
}

/// Llama family over the vendored [`OwnedLlama`]: per-session [`SessionKv`]
/// caches plus a batched decode step.
pub struct LlamaCausalLM {
    model: OwnedLlama,
}

impl CandleCausalLM for LlamaCausalLM {
    type Cache = SessionKv;

    fn new_cache(&self, _device: &Device) -> Result<Self::Cache> {
        Ok(self.model.new_cache())
    }

    fn cache_len(cache: &Self::Cache) -> usize {
        cache.len()
    }

    fn cache_prefix(cache: &Self::Cache, len: usize) -> Option<Self::Cache> {
        cache.prefix(len)
    }

    fn forward_tokens(&self, tokens: &[u32], cache: &mut Self::Cache) -> Result<Vec<f32>> {
        self.model.forward_tokens(tokens, cache)
    }

    fn session_forward_batch(
        &self,
        sessions: &mut [Box<dyn ModelSession>],
        feeds: &[&[u32]],
    ) -> Option<Vec<Vec<f32>>> {
        if sessions.len() != feeds.len() || sessions.len() < 2 {
            return None;
        }
        // The batched kernel advances exactly one token per session.
        if feeds.iter().any(|f| f.len() != 1) {
            return None;
        }
        // Recover the concrete sessions and their caches; leave the batch to
        // the per-sequence path if any session is foreign or inconsistent.
        let mut caches: Vec<&mut SessionKv> = Vec::with_capacity(sessions.len());
        for s in sessions.iter_mut() {
            let cs = s
                .as_any_mut()?
                .downcast_mut::<CandleSession<LlamaCausalLM>>()?;
            if cs.fed != Self::cache_len(&cs.cache) {
                return None;
            }
            caches.push(&mut cs.cache);
        }
        let tokens: Vec<u32> = feeds.iter().map(|f| f[0]).collect();
        match self.model.batch_decode(&tokens, &mut caches) {
            Ok(all) if all.len() == sessions.len() => {
                // Bookkeeping pass: sessions were fed, advance counters and
                // cache the logits for empty-feed re-reads.
                for (s, logits) in sessions.iter_mut().zip(all.iter()) {
                    let Some(cs) = s
                        .as_any_mut()
                        .and_then(|a| a.downcast_mut::<CandleSession<LlamaCausalLM>>())
                    else {
                        continue;
                    };
                    cs.fed += 1;
                    cs.last_logits = Some(logits.clone());
                    cs.tokens_fed.fetch_add(1, Ordering::Relaxed);
                }
                Some(all)
            }
            other => {
                if let Err(e) = &other {
                    eprintln!("[pagoda-hf] batched decode failed, using per-session forwards: {e:#}");
                }
                // Mid-forward failures (OOM, device loss) may leave some
                // caches partially updated; the per-session path degrades
                // those sequences to uniform logits instead of panicking.
                None
            }
        }
    }
}

/// [`ModelEngine`] adapter over a Candle causal LM.
///
/// Three contracts are served:
///
/// * **Stateless** [`ModelEngine::forward`]: full-context replay with a fresh
///   cache per call (used by `Engine::score` and as the universal fallback).
/// * **Incremental** [`ModelEngine::begin_session`]: each engine sequence gets
///   a [`CandleSession`] with its own KV cache; the prompt is pushed once and
///   every decode step feeds exactly one token, so total model compute drops
///   from O(n^2) replay to O(n) feed.
/// * **Batched** [`ModelEngine::session_forward_batch`]: same-shape sessions
///   advance in one `[B, 1]` forward (Llama family only, for now).
pub struct CandleModel<M: CandleCausalLM = LlamaCausalLM> {
    model: Arc<M>,
    device: Device,
    vocab_size: usize,
    /// Ground-truth model compute: tokens actually pushed through forward
    /// calls (stateless + all sessions). Engine-level stats model a shared-KV
    /// backend; this counter is what the current backend really paid.
    tokens_fed: Arc<AtomicU64>,
    /// Cross-request KV store: finished sessions' caches keyed by token path.
    /// Later requests graft the longest matching prefix instead of
    /// recomputing it (the physical half of RadixAttention; the core engine's
    /// radix/APC caches are the logical half).
    vault: Arc<Mutex<KvVault<M::Cache>>>,
}

/// Default cross-request KV vault capacity (entries, not tokens). Override
/// with `PAGODA_KV_VAULT_ENTRIES`.
const DEFAULT_KV_VAULT_ENTRIES: usize = 128;

fn kv_vault_entries_from_env() -> usize {
    std::env::var("PAGODA_KV_VAULT_ENTRIES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_KV_VAULT_ENTRIES)
}

impl CandleModel<LlamaCausalLM> {
    /// Pick the compute device from `PAGODA_DEVICE`: `cpu` (default) or
    /// `cuda` / `cuda:N`. CUDA requires building with `--features cuda`.
    pub fn device_from_env() -> Result<Device> {
        match std::env::var("PAGODA_DEVICE").as_deref() {
            Ok("cuda") => Device::new_cuda(0).context("PAGODA_DEVICE=cuda but no CUDA device"),
            Ok(other) if other.starts_with("cuda:") => {
                let idx: usize = other[5..].parse().context("bad PAGODA_DEVICE index")?;
                Device::new_cuda(idx).with_context(|| format!("no CUDA device {idx}"))
            }
            _ => Ok(Device::Cpu),
        }
    }

    /// Load a Llama-family `config.json` and (possibly sharded) safetensors
    /// weights on the given device.
    pub fn llama_from_safetensors(
        config_path: impl AsRef<Path>,
        weight_paths: &[std::path::PathBuf],
        device: Device,
    ) -> Result<Self> {
        Self::llama_from_safetensors_as(config_path, weight_paths, device, DType::F32)
    }

    /// [`CandleModel::llama_from_safetensors`] with an explicit compute dtype
    /// (F16 halves bandwidth and unlocks tensor cores on GPU).
    pub fn llama_from_safetensors_as(
        config_path: impl AsRef<Path>,
        weight_paths: &[std::path::PathBuf],
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        // `LlamaConfig` is the serde shape of config.json; `into_config`
        // converts it into Candle's runtime `Config` (flash-attn stays off).
        let hf_config: LlamaConfig =
            serde_json::from_reader(std::fs::File::open(config_path)?)
                .context("failed to parse Llama config.json")?;
        let config = hf_config.into_config(false);
        let vocab_size = config.vocab_size;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weight_paths, dtype, &device)?
        };
        let model = OwnedLlama::load(vb, &config, dtype, &device)
            .context("failed to load Llama weights")?;
        Ok(Self {
            model: Arc::new(LlamaCausalLM { model }),
            device,
            vocab_size,
            tokens_fed: Arc::new(AtomicU64::new(0)),
            vault: Arc::new(Mutex::new(KvVault::new(kv_vault_entries_from_env()))),
        })
    }

    /// Compute dtype from `PAGODA_DTYPE`: `f32` (default) or `f16` / `bf16`.
    pub fn dtype_from_env() -> Result<DType> {
        match std::env::var("PAGODA_DTYPE").as_deref() {
            Ok("f16") => Ok(DType::F16),
            Ok("bf16") => Ok(DType::BF16),
            Ok("f32") | Err(_) => Ok(DType::F32),
            Ok(other) => anyhow::bail!("unknown PAGODA_DTYPE {other:?} (want f32/f16/bf16)"),
        }
    }

    /// Download `config.json` + `model.safetensors` from a HuggingFace repo and
    /// load them on the CPU.
    pub fn llama_from_hub(repo: &str) -> Result<Self> {
        Self::llama_from_hub_on(repo, Device::Cpu)
    }

    /// Same as [`CandleModel::llama_from_hub`] but on an explicit device.
    pub fn llama_from_hub_on(repo: &str, device: Device) -> Result<Self> {
        let config_path = HfTokenizer::download(repo, "config.json")?;
        let weights_path = HfTokenizer::download(repo, "model.safetensors")?;
        Self::llama_from_safetensors_as(config_path, &[weights_path], device, Self::dtype_from_env()?)
    }
}

impl<M: CandleCausalLM + 'static> CandleModel<M> {
    /// Errors degrade to a uniform logits vector so the serving loop still
    /// produces a token instead of panicking.
    fn uniform_fallback(&self) -> Vec<f32> {
        vec![0.0f32; self.vocab_size]
    }

    /// Shared handle to the ground-truth "tokens fed to the model" counter.
    pub fn tokens_fed_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.tokens_fed)
    }

    /// The real forward chain, with errors surfaced instead of swallowed.
    fn try_forward(&self, context: &[u32]) -> Result<Vec<f32>> {
        if context.is_empty() {
            anyhow::bail!("empty context");
        }
        let mut cache = self.model.new_cache(&self.device)?;
        let logits = self.model.forward_tokens(context, &mut cache)?;
        self.tokens_fed
            .fetch_add(context.len() as u64, Ordering::Relaxed);
        Ok(logits)
    }
}

impl<M: CandleCausalLM + 'static> ModelEngine for CandleModel<M> {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn forward(&self, context: &[u32]) -> Vec<f32> {
        match self.try_forward(context) {
            Ok(logits) => logits,
            Err(e) => {
                eprintln!("[pagoda-hf] forward fell back to uniform logits: {e:#}");
                self.uniform_fallback()
            }
        }
    }

    fn name(&self) -> &'static str {
        "candle-llama"
    }

    fn begin_session(&self) -> Option<Box<dyn ModelSession>> {
        match self.model.new_cache(&self.device) {
            Ok(cache) => Some(Box::new(CandleSession {
                model: Arc::clone(&self.model),
                cache,
                fed: 0,
                vocab_size: self.vocab_size,
                last_logits: None,
                tokens_fed: Arc::clone(&self.tokens_fed),
            })),
            Err(e) => {
                eprintln!("[pagoda-hf] failed to mint a session cache: {e:#}");
                None
            }
        }
    }

    fn session_forward_batch(
        &self,
        sessions: &mut [Box<dyn ModelSession>],
        feeds: &[&[u32]],
    ) -> Option<Vec<Vec<f32>>> {
        self.model.session_forward_batch(sessions, feeds)
    }

    fn graft_session(&self, prompt: &[u32], max_len: usize) -> Option<Box<dyn ModelSession>> {
        let (cache, covered) = self.vault.lock().ok()?.longest_prefix(prompt, max_len)?;
        // The stored snapshot may cover more than the cap allows (a full-path
        // key matching a shorter prompt); slice down to exactly `covered`.
        let cache = M::cache_prefix(&cache, covered)?;
        Some(Box::new(CandleSession {
            model: Arc::clone(&self.model),
            cache,
            fed: covered,
            vocab_size: self.vocab_size,
            last_logits: None,
            tokens_fed: Arc::clone(&self.tokens_fed),
        }))
    }

    fn offer_session_kv(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        session: Box<dyn ModelSession>,
    ) {
        let Some(cs) = session
            .as_any()
            .and_then(|a| a.downcast_ref::<CandleSession<M>>())
        else {
            return;
        };
        // Key entries by the physically fed path: the final sampled token is
        // appended to the sequence but never pushed through the model, so the
        // cache is usually one token shorter than `tokens`. Keying by the fed
        // path guarantees a graft never overshoots the cached tensors, and it
        // makes continue-this-conversation prompts (old path + new suffix)
        // graft the entire previous turn.
        let fed = M::cache_len(&cs.cache).min(tokens.len());
        if fed == 0 {
            return;
        }
        if let Ok(mut vault) = self.vault.lock() {
            vault.offer(&tokens[..fed], prompt_len.min(fed), cs.cache.clone());
        }
    }
}

/// One sequence's incremental decode state: its own KV cache plus the number
/// of tokens already pushed through the model.
///
/// Feeding discipline: the first call carries the whole prompt (one masked
/// prefill pass), later calls carry the fresh suffix (one token during
/// continuous batching, but any non-empty suffix is accepted). Same-shape
/// sessions are advanced by the batched path instead; this per-session
/// forward handles everything else.
pub struct CandleSession<M: CandleCausalLM = LlamaCausalLM> {
    model: Arc<M>,
    cache: M::Cache,
    fed: usize,
    vocab_size: usize,
    last_logits: Option<Vec<f32>>,
    tokens_fed: Arc<AtomicU64>,
}

impl<M: CandleCausalLM> CandleSession<M> {
    fn try_forward(&mut self, new_tokens: &[u32]) -> Result<Vec<f32>> {
        if new_tokens.is_empty() {
            return self
                .last_logits
                .clone()
                .context("session has no cached logits yet");
        }
        let logits = self.model.forward_tokens(new_tokens, &mut self.cache)?;
        self.tokens_fed
            .fetch_add(new_tokens.len() as u64, Ordering::Relaxed);
        self.fed += new_tokens.len();
        Ok(logits)
    }
}

impl<M: CandleCausalLM + 'static> ModelSession for CandleSession<M> {
    fn context_len(&self) -> usize {
        self.fed
    }

    fn forward(&mut self, new_tokens: &[u32]) -> Vec<f32> {
        match self.try_forward(new_tokens) {
            Ok(logits) => {
                self.last_logits = Some(logits.clone());
                logits
            }
            Err(e) => {
                eprintln!("[pagoda-hf] session forward fell back to uniform logits: {e:#}");
                vec![0.0f32; self.vocab_size]
            }
        }
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn fork(&self) -> Option<Box<dyn ModelSession>> {
        // Candle tensors are Arc-shared and immutable; appends concatenate
        // into fresh tensors, so the cloned cache is an independent branch
        // that shares everything computed so far.
        Some(Box::new(Self {
            model: Arc::clone(&self.model),
            cache: self.cache.clone(),
            fed: self.fed,
            vocab_size: self.vocab_size,
            last_logits: self.last_logits.clone(),
            tokens_fed: Arc::clone(&self.tokens_fed),
        }))
    }
}
