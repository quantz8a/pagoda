// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! MLX-serve HTTP adapter implementing pagoda's [`ModelEngine`] /
//! [`ModelSession`] contracts.
//!
//! GPU inference stays in `mlx-serve` (via the gpu gate). pagoda's Engine owns
//! scheduling, DSL, and checkpoints. Because the HTTP API returns sampled
//! tokens (and optional logprobs) rather than a full vocab logit vector, each
//! `forward` step requests `max_tokens=1` and builds a **sparse** logits row
//! peaked at the returned token (remaining entries a large finite penalty —
//! pagoda's Engine rejects non-finite logits as faults).
//!
//! Limitations (Phase 1):
//! * Local grammar / temperature only see the sparse row; prefer remote
//!   sampling params for quality.
//! * `fork` clones the token transcript; GPU KV is not copied (mlx prefix
//!   cache absorbs repeated prefixes).
//! * No tensor-level `graft_session` / `offer_session_kv`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use pagoda::http_client::{self, DEFAULT_TIMEOUT};
use pagoda::json::{self, Value};
use pagoda::model::{ModelEngine, ModelSession};
use pagoda::tokenizer::Tokenizer as SglTokenizer;

use crate::HfTokenizer;

/// Encode / decode hooks so tests can inject a tiny vocabulary without HF.
pub trait MlxTokenCodec: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
    fn vocab_size(&self) -> usize;
}

impl MlxTokenCodec for HfTokenizer {
    /// Map a model continuation fragment → ids (no special-token injection).
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }
    fn decode(&self, tokens: &[u32]) -> String {
        SglTokenizer::decode(self, tokens)
    }
    fn vocab_size(&self) -> usize {
        SglTokenizer::vocab_size(self)
    }
}

/// Which wire shape to POST when stepping the remote model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlxApi {
    /// OpenAI-style `POST /v1/completions`.
    Completions,
    /// SGLang-style `POST /generate` with `text` + `sampling_params`.
    Generate,
}

struct SharedRemote {
    addr: String,
    api: MlxApi,
    codec: Arc<dyn MlxTokenCodec>,
    timeout: Duration,
    prompt_tokens: Arc<AtomicU64>,
    cached_tokens: Arc<AtomicU64>,
    remote_steps: Arc<AtomicU64>,
}

impl Clone for SharedRemote {
    fn clone(&self) -> Self {
        Self {
            addr: self.addr.clone(),
            api: self.api,
            codec: Arc::clone(&self.codec),
            timeout: self.timeout,
            prompt_tokens: Arc::clone(&self.prompt_tokens),
            cached_tokens: Arc::clone(&self.cached_tokens),
            remote_steps: Arc::clone(&self.remote_steps),
        }
    }
}

/// Remote MLX backend for pagoda's Engine.
pub struct MlxHttpEngine {
    inner: Arc<SharedRemote>,
}

impl MlxHttpEngine {
    pub fn new(upstream_url: &str, codec: Arc<dyn MlxTokenCodec>, api: MlxApi) -> Result<Self> {
        let addr = http_client::parse_http_addr(upstream_url)
            .ok_or_else(|| anyhow!("bad upstream url {upstream_url:?} (want http://host[:port])"))?;
        Ok(Self {
            inner: Arc::new(SharedRemote {
                addr,
                api,
                codec,
                timeout: DEFAULT_TIMEOUT,
                prompt_tokens: Arc::new(AtomicU64::new(0)),
                cached_tokens: Arc::new(AtomicU64::new(0)),
                remote_steps: Arc::new(AtomicU64::new(0)),
            }),
        })
    }

    pub fn from_tokenizer(upstream_url: &str, tok: HfTokenizer, api: MlxApi) -> Result<Self> {
        Self::new(upstream_url, Arc::new(tok), api)
    }

    pub fn with_timeout(self, timeout: Duration) -> Self {
        let mut inner = (*self.inner).clone();
        inner.timeout = timeout;
        Self {
            inner: Arc::new(inner),
        }
    }

    pub fn prompt_tokens(&self) -> u64 {
        self.inner.prompt_tokens.load(Ordering::Relaxed)
    }
    pub fn cached_tokens(&self) -> u64 {
        self.inner.cached_tokens.load(Ordering::Relaxed)
    }
    pub fn remote_steps(&self) -> u64 {
        self.inner.remote_steps.load(Ordering::Relaxed)
    }

    /// Share usage counters with a [`ServingEngine`] wrapper for `/stats`.
    pub fn usage_handles(&self) -> MlxUsageHandles {
        MlxUsageHandles {
            upstream: format!("http://{}", self.inner.addr),
            api: self.inner.api,
            prompt_tokens: Arc::clone(&self.inner.prompt_tokens),
            cached_tokens: Arc::clone(&self.inner.cached_tokens),
            remote_steps: Arc::clone(&self.inner.remote_steps),
        }
    }

    fn step_logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        step_logits_shared(&self.inner, tokens)
    }
}

/// Arc handles for mlx remote usage (survive after the engine moves into
/// [`pagoda::Engine`]).
#[derive(Clone)]
pub struct MlxUsageHandles {
    pub upstream: String,
    pub api: MlxApi,
    prompt_tokens: Arc<AtomicU64>,
    cached_tokens: Arc<AtomicU64>,
    remote_steps: Arc<AtomicU64>,
}

impl MlxUsageHandles {
    pub fn prompt_tokens(&self) -> u64 {
        self.prompt_tokens.load(Ordering::Relaxed)
    }
    pub fn cached_tokens(&self) -> u64 {
        self.cached_tokens.load(Ordering::Relaxed)
    }
    pub fn remote_steps(&self) -> u64 {
        self.remote_steps.load(Ordering::Relaxed)
    }
    pub fn cache_hit_ratio(&self) -> f64 {
        let p = self.prompt_tokens();
        if p == 0 {
            0.0
        } else {
            self.cached_tokens() as f64 / p as f64
        }
    }

    /// Fields merged into `GET /stats` via [`pagoda::server::ServingEngine::extra_stats`].
    pub fn extra_stats(&self) -> Vec<(String, Value)> {
        vec![
            ("backend".to_string(), Value::String("mlx-http".into())),
            ("upstream".to_string(), Value::String(self.upstream.clone())),
            (
                "mlx_api".to_string(),
                Value::String(match self.api {
                    MlxApi::Completions => "completions".into(),
                    MlxApi::Generate => "generate".into(),
                }),
            ),
            (
                "upstream_prompt_tokens".to_string(),
                Value::Number(self.prompt_tokens() as f64),
            ),
            (
                "upstream_cached_tokens".to_string(),
                Value::Number(self.cached_tokens() as f64),
            ),
            (
                "upstream_cache_hit_ratio".to_string(),
                Value::Number(self.cache_hit_ratio()),
            ),
            (
                "mlx_remote_steps".to_string(),
                Value::Number(self.remote_steps() as f64),
            ),
        ]
    }
}

impl ModelEngine for MlxHttpEngine {
    fn vocab_size(&self) -> usize {
        self.inner.codec.vocab_size()
    }

    fn name(&self) -> &'static str {
        "mlx-http"
    }

    fn forward(&self, context: &[u32]) -> Vec<f32> {
        match self.step_logits(context) {
            Ok(logits) => logits,
            Err(e) => {
                eprintln!("[mlx-http] stateless forward failed: {e:#}");
                uniform_logits(self.inner.codec.vocab_size())
            }
        }
    }

    fn begin_session(&self) -> Option<Box<dyn ModelSession>> {
        Some(Box::new(MlxHttpSession {
            remote: Arc::clone(&self.inner),
            tokens: Vec::new(),
            last_logits: None,
        }))
    }
}

/// Per-sequence remote session.
pub struct MlxHttpSession {
    remote: Arc<SharedRemote>,
    tokens: Vec<u32>,
    last_logits: Option<Vec<f32>>,
}

impl ModelSession for MlxHttpSession {
    fn context_len(&self) -> usize {
        self.tokens.len()
    }

    fn forward(&mut self, new_tokens: &[u32]) -> Vec<f32> {
        if new_tokens.is_empty() {
            if let Some(logits) = &self.last_logits {
                return logits.clone();
            }
            return uniform_logits(self.remote.codec.vocab_size());
        }
        self.tokens.extend_from_slice(new_tokens);
        // Reuse the previous logits allocation (Qwen3.8 vocab ≈ 248k ≈ 1 MiB).
        let mut buf = self.last_logits.take().unwrap_or_default();
        match step_logits_into(&self.remote, &self.tokens, &mut buf) {
            Ok(()) => {
                self.last_logits = Some(buf.clone());
                buf
            }
            Err(e) => {
                eprintln!("[mlx-http] session forward failed: {e:#}");
                fill_uniform(&mut buf, self.remote.codec.vocab_size());
                self.last_logits = Some(buf.clone());
                buf
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
        Some(Box::new(MlxHttpSession {
            remote: Arc::clone(&self.remote),
            tokens: self.tokens.clone(),
            last_logits: self.last_logits.clone(),
        }))
    }
}

fn step_logits_shared(remote: &SharedRemote, tokens: &[u32]) -> Result<Vec<f32>> {
    let mut buf = Vec::new();
    step_logits_into(remote, tokens, &mut buf)?;
    Ok(buf)
}

fn step_logits_into(remote: &SharedRemote, tokens: &[u32], buf: &mut Vec<f32>) -> Result<()> {
    let prompt = remote.codec.decode(tokens);
    let (path, body) = match remote.api {
        MlxApi::Completions => (
            "/v1/completions",
            format!(
                r#"{{"prompt":{},"max_tokens":1,"temperature":0,"logprobs":5}}"#,
                json_string(&prompt)
            ),
        ),
        MlxApi::Generate => (
            "/generate",
            format!(
                r#"{{"text":{},"sampling_params":{{"max_tokens":1,"temperature":0.0}}}}"#,
                json_string(&prompt)
            ),
        ),
    };
    let (status, resp) =
        http_client::request(&remote.addr, "POST", path, Some(&body), remote.timeout)
            .with_context(|| format!("mlx upstream {}{path} unreachable", remote.addr))?;
    if status >= 400 {
        bail!(
            "mlx upstream HTTP {status}: {}",
            resp.chars().take(240).collect::<String>()
        );
    }
    if let Some((prompt_n, cached)) = scrape_usage(&resp) {
        remote.prompt_tokens.fetch_add(prompt_n, Ordering::Relaxed);
        remote.cached_tokens.fetch_add(cached, Ordering::Relaxed);
    }
    let (token_id, logprob) = parse_next_token(&resp, remote.codec.as_ref())?;
    remote.remote_steps.fetch_add(1, Ordering::Relaxed);
    fill_sparse_logits(buf, remote.codec.vocab_size(), token_id, logprob);
    Ok(())
}

fn json_string(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}

/// Floor for non-selected vocab rows. Must stay finite: [`pagoda::Engine`]
/// faults on NaN/Inf logits.
const SPARSE_FLOOR: f32 = -1.0e9;

fn fill_sparse_logits(buf: &mut Vec<f32>, vocab: usize, token_id: u32, logprob: f32) {
    if buf.len() == vocab {
        buf.fill(SPARSE_FLOOR);
    } else {
        buf.clear();
        buf.resize(vocab, SPARSE_FLOOR);
    }
    let idx = token_id as usize;
    // Ensure the peak is strictly above the floor even if remote logprob is tiny.
    let peak = if logprob.is_finite() {
        logprob.max(SPARSE_FLOOR + 1.0)
    } else {
        0.0
    };
    if idx < vocab {
        buf[idx] = peak;
    } else if vocab > 0 {
        buf[0] = peak;
    }
}

fn uniform_logits(vocab: usize) -> Vec<f32> {
    vec![0.0f32; vocab.max(1)]
}

fn fill_uniform(buf: &mut Vec<f32>, vocab: usize) {
    let n = vocab.max(1);
    if buf.len() == n {
        buf.fill(0.0);
    } else {
        buf.clear();
        buf.resize(n, 0.0);
    }
}

fn scrape_usage(body: &str) -> Option<(u64, u64)> {
    let v = json::parse(body).ok()?;
    let usage = v.get("usage")?;
    let prompt = usage
        .get("prompt_tokens")
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as u64;
    let cached = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .or_else(|| usage.get("cached_tokens"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as u64;
    if prompt == 0 && cached == 0 {
        return None;
    }
    Some((prompt, cached))
}

fn parse_next_token(body: &str, codec: &dyn MlxTokenCodec) -> Result<(u32, f32)> {
    let v = json::parse(body).map_err(|e| anyhow!("mlx response is not JSON: {e}"))?;

    if let Some(Value::Array(choices)) = v.get("choices") {
        if let Some(c0) = choices.first() {
            if let Some(text) = c0
                .get("text")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                let ids = codec.encode(text);
                let id = *ids
                    .first()
                    .ok_or_else(|| anyhow!("empty encode for {text:?}"))?;
                let lp = c0
                    .get("logprobs")
                    .and_then(|lp| lp.get("token_logprobs"))
                    .and_then(|t| match t {
                        Value::Array(a) => a.first().and_then(Value::as_f64),
                        _ => None,
                    })
                    .unwrap_or(0.0) as f32;
                return Ok((id, lp));
            }
            if let Some(text) = c0
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                let ids = codec.encode(text);
                let id = *ids
                    .first()
                    .ok_or_else(|| anyhow!("empty encode for {text:?}"))?;
                return Ok((id, 0.0));
            }
        }
    }

    if let Some(text) = v.get("text").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        let ids = codec.encode(text);
        let id = *ids
            .first()
            .ok_or_else(|| anyhow!("empty encode for generate text"))?;
        return Ok((id, 0.0));
    }

    bail!(
        "could not parse next token from mlx body: {}",
        body.chars().take(200).collect::<String>()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TinyCodec;
    impl MlxTokenCodec for TinyCodec {
        fn encode(&self, text: &str) -> Vec<u32> {
            text.bytes().map(|b| b as u32).collect()
        }
        fn decode(&self, tokens: &[u32]) -> String {
            tokens.iter().map(|&t| t as u8 as char).collect()
        }
        fn vocab_size(&self) -> usize {
            256
        }
    }

    #[test]
    fn sparse_peaks_at_token() {
        let mut logits = Vec::new();
        fill_sparse_logits(&mut logits, 10, 3, -0.5);
        assert_eq!(logits[3], -0.5);
        assert!(logits.iter().all(|x| x.is_finite()));
        assert!(logits[3] > logits[0]);
        // Reuse path must not realloc when vocab matches.
        let ptr = logits.as_ptr();
        fill_sparse_logits(&mut logits, 10, 7, -0.2);
        assert_eq!(logits.as_ptr(), ptr);
        assert_eq!(logits[7], -0.2);
        assert_eq!(logits[3], SPARSE_FLOOR);
    }

    #[test]
    fn parse_completions_text() {
        let body = r#"{"choices":[{"text":"A","logprobs":{"token_logprobs":[-0.1]}}],"usage":{"prompt_tokens":4,"prompt_tokens_details":{"cached_tokens":2}}}"#;
        let (id, lp) = parse_next_token(body, &TinyCodec).unwrap();
        assert_eq!(id, b'A' as u32);
        assert!((lp + 0.1).abs() < 1e-5);
        assert_eq!(scrape_usage(body), Some((4, 2)));
    }
}
