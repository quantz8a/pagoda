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
//!   causal LM (Llama family by default), loading real `config.json` +
//!   safetensors weights and returning raw logits.
//! * [`CandleSession`] — the incremental half: each engine sequence gets its
//!   own Candle KV-cache session and feeds every token exactly once (prompt
//!   once, then one fresh token per decode step) instead of replaying the
//!   full context. Sessions fork cheaply (shared tensor storage), which is
//!   what powers checkpoint branching for agent-tree workloads.
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
use std::sync::Arc;

use anyhow::{Context, Result};

pub use candle_core::{DType, Device, Tensor};
pub use candle_nn::VarBuilder;
pub use candle_transformers::models::llama::{Cache as LlamaCache, Llama, LlamaConfig};

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
/// Candle release's `Llama` / `Mistral` / `Gemma` API (and its per-family KV
/// cache type) lives in one spot and the [`ModelEngine`] adapter below stays
/// independent of it.
///
/// The KV cache is explicit and external: one cache per decode session, minted
/// by [`CandleCausalLM::new_cache`]. `Cache: Clone` is required so checkpoint
/// trunks can fork branches over shared tensor storage.
pub trait CandleCausalLM: Send + Sync {
    /// Per-session KV cache type.
    type Cache: Send + Clone;
    /// Mint a fresh, empty KV cache for one decode session.
    fn new_cache(&self, device: &Device) -> Result<Self::Cache>;
    /// `input` has shape `[1, seq]`; `index_pos` is the position of the first
    /// input token (RoPE offset + KV-cache append point). Returns logits whose
    /// **last position** is what callers want (the exact rank varies across
    /// Candle releases; callers flatten and take the trailing `vocab_size`).
    fn forward(&self, input: &Tensor, index_pos: usize, cache: &mut Self::Cache)
        -> Result<Tensor>;
}

/// Candle 0.8's `Llama::forward` takes `&self` plus an external
/// `&mut llama::Cache`; the runtime config is retained here so fresh
/// per-session caches can be minted on demand.
pub struct LlamaCausalLM {
    model: Llama,
    config: candle_transformers::models::llama::Config,
    dtype: DType,
}

impl CandleCausalLM for LlamaCausalLM {
    type Cache = LlamaCache;

    fn new_cache(&self, device: &Device) -> Result<Self::Cache> {
        LlamaCache::new(true, self.dtype, &self.config, device)
            .context("failed to build the Llama KV cache")
    }

    fn forward(
        &self,
        input: &Tensor,
        index_pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        self.model
            .forward(input, index_pos, cache)
            .map_err(anyhow::Error::from)
    }
}

/// Reduce a Candle logits tensor of any rank to the last-position
/// `[vocab_size]` vector. Candle 0.8 already reduces to `[b_sz, vocab]`;
/// older releases return `[1, seq, vocab]`. The last-position logits are the
/// trailing `vocab_size` values either way.
fn last_position_logits(logits: &Tensor, vocab_size: usize) -> Result<Vec<f32>> {
    let flat = logits
        .contiguous()
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(DType::F32))
        .context("failed to flatten logits")?;
    let n = flat.elem_count();
    if n < vocab_size {
        anyhow::bail!("logits too small: {n} elements, vocab is {vocab_size}");
    }
    flat.narrow(0, n - vocab_size, vocab_size)?
        .to_vec1::<f32>()
        .context("failed to materialise logits as Vec<f32>")
}

/// [`ModelEngine`] adapter over a Candle causal LM.
///
/// Two contracts are served:
///
/// * **Stateless** [`ModelEngine::forward`]: full-context replay with a fresh
///   cache per call (used by `Engine::score` and as the universal fallback).
/// * **Incremental** [`ModelEngine::begin_session`]: each engine sequence gets
///   a [`CandleSession`] with its own KV cache; the prompt is pushed once and
///   every decode step feeds exactly one token, so total model compute drops
///   from O(n^2) replay to O(n) feed.
pub struct CandleModel<M: CandleCausalLM = LlamaCausalLM> {
    model: Arc<M>,
    device: Device,
    vocab_size: usize,
    /// Ground-truth model compute: tokens actually pushed through forward
    /// calls (stateless + all sessions). Engine-level stats model a shared-KV
    /// backend; this counter is what the current backend really paid.
    tokens_fed: Arc<AtomicU64>,
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
        let model = Llama::load(vb, &config).context("failed to load Llama weights")?;
        Ok(Self {
            model: Arc::new(LlamaCausalLM {
                model,
                config,
                dtype,
            }),
            device,
            vocab_size,
            tokens_fed: Arc::new(AtomicU64::new(0)),
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
        self.tokens_fed
            .fetch_add(context.len() as u64, Ordering::Relaxed);

        // Stateless path: fresh cache, single full-context pass at position 0.
        let mut cache = self.model.new_cache(&self.device)?;
        let input = Tensor::new(context, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .context("failed to build the [1, seq] input tensor")?;
        let logits = self
            .model
            .forward(&input, 0, &mut cache)
            .context("candle forward failed")?;
        last_position_logits(&logits, self.vocab_size)
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
                device: self.device.clone(),
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
}

/// One sequence's incremental decode state: its own Candle KV cache plus the
/// number of tokens already pushed through the model.
///
/// Feeding discipline (driven by candle-transformers 0.8's mask handling):
///
/// * **First call** (empty cache): the whole prompt goes through in one pass
///   at `index_pos = 0` — the causal mask is square, everything lines up.
/// * **Later calls** (non-empty cache): tokens are fed one at a time. Candle's
///   cached mask is `[seq, seq]` and cannot stretch over a longer KV history
///   (`seq_len > 1` with `index_pos > 0` breaks broadcasting), while
///   `seq_len == 1` skips masking entirely — which is exactly the decode
///   shape. Multi-token suffixes are looped.
pub struct CandleSession<M: CandleCausalLM = LlamaCausalLM> {
    model: Arc<M>,
    cache: M::Cache,
    fed: usize,
    device: Device,
    vocab_size: usize,
    last_logits: Option<Vec<f32>>,
    tokens_fed: Arc<AtomicU64>,
}

impl<M: CandleCausalLM> CandleSession<M> {
    /// Run one Candle forward over `tokens` starting at `index_pos` and return
    /// last-position logits.
    fn run(&mut self, tokens: &[u32], index_pos: usize) -> Result<Vec<f32>> {
        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .context("failed to build the [1, seq] input tensor")?;
        let logits = self
            .model
            .forward(&input, index_pos, &mut self.cache)
            .context("candle forward failed")?;
        last_position_logits(&logits, self.vocab_size)
    }

    fn try_forward(&mut self, new_tokens: &[u32]) -> Result<Vec<f32>> {
        if new_tokens.is_empty() {
            return self
                .last_logits
                .clone()
                .context("session has no cached logits yet");
        }
        self.tokens_fed
            .fetch_add(new_tokens.len() as u64, Ordering::Relaxed);

        let logits = if self.fed == 0 {
            // Single-shot prefill: square causal mask over an empty cache.
            self.run(new_tokens, 0)?
        } else {
            // Decode shape: one token per call so candle 0.8's [seq, seq]
            // mask never has to cover a longer KV history.
            let mut logits = Vec::new();
            for (i, tok) in new_tokens.iter().enumerate() {
                logits = self.run(std::slice::from_ref(tok), self.fed + i)?;
            }
            logits
        };
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

    fn fork(&self) -> Option<Box<dyn ModelSession>> {
        // Candle tensors are Arc-shared and immutable; appends concatenate
        // into fresh tensors, so the cloned cache is an independent branch
        // that shares everything computed so far.
        Some(Box::new(Self {
            model: Arc::clone(&self.model),
            cache: self.cache.clone(),
            fed: self.fed,
            device: self.device.clone(),
            vocab_size: self.vocab_size,
            last_logits: self.last_logits.clone(),
            tokens_fed: Arc::clone(&self.tokens_fed),
        }))
    }
}
