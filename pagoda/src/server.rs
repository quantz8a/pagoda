// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! A minimal, dependency-free HTTP serving frontend.
//!
//! Implements a tiny subset of the HTTP/1.1 protocol on top of
//! `std::net::TcpListener`. Production deployments would replace this with a
//! full async server (axum/hyper), with the same request-handling logic.
//!
//! The frontend is backend-agnostic: it only requires the [`ServingEngine`]
//! capability (`generate` + `stats`), which the toy [`crate::engine::Engine`]
//! and real backends (e.g. the companion `pagoda-hf` crate) both implement.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::actor::{EngineHandle, StreamEvent};
use crate::engine::{CheckpointId, Engine, EngineStats};
use crate::grammar::Grammar;
use crate::http_client;
use crate::json::{self, Value};
use crate::model::ModelEngine;
use crate::pd::{PdError, PdRole, PrefillReceipt};
use crate::spec::{FinishReason, GenerationOutput, SamplingParams, WriteRequest};
use crate::tokenizer::Tokenizer;

/// Reverse-proxy target: an upstream inference server (e.g. SGLang's
/// OpenAI-compatible HTTP server) that receives the heavy generation
/// traffic while pagoda keeps the control plane (`/health`, `/stats`,
/// `/checkpoint*`) local.
pub struct Proxy {
    /// `host:port` for dialing.
    addr: String,
    /// Original `http://...` URL, surfaced in `/stats`.
    url: String,
    /// Requests forwarded upstream so far.
    proxied: AtomicU64,
    /// Of those, requests relayed as a live stream (SSE/chunked).
    streamed: AtomicU64,
    /// Of those, requests sent to a department-specific route (not default).
    routed: AtomicU64,
    /// Department-specific upstreams from --route dept=url (Laya routes).
    routes: Vec<Route>,
    /// Prefix-affine pool over multiple default upstreams (comma-separated
    /// --upstream). None for a single default.
    prefix_pool: Option<Mutex<crate::pd::PrefixRouter>>,
}

/// One --route entry: a Laya department name and its dedicated upstream.
struct Route {
    dept: String,
    url: String,
    addr: String,
}

impl Proxy {
    /// Build from `http://host[:port]`; anything else returns `None`.
    /// A comma-separated list (`http://a:1,http://b:2`) builds a
    /// prefix-affine pool: prompts route to the worker that served their
    /// longest matching prefix (cache affinity); the first URL is the
    /// default for non-generation traffic.
    pub fn from_url(url: &str) -> Option<Self> {
        let urls: Vec<String> = url
            .split(',')
            .map(|u| u.trim().trim_end_matches('/').to_string())
            .filter(|u| !u.is_empty())
            .collect();
        let first = urls.first()?.clone();
        let addr = http_client::parse_http_addr(&first)?;
        let prefix_pool = if urls.len() > 1 {
            crate::pd::PrefixRouter::new(&urls).map(Mutex::new)
        } else {
            None
        };
        Some(Self {
            addr,
            url: first,
            proxied: AtomicU64::new(0),
            streamed: AtomicU64::new(0),
            routed: AtomicU64::new(0),
            routes: Vec::new(),
            prefix_pool,
        })
    }

    /// Build with department routes on top of the default upstream:
    /// Laya's department pick decides which worker serves the request.
    pub fn with_routes(url: &str, routes: &[(&str, &str)]) -> Option<Self> {
        let mut proxy = Self::from_url(url)?;
        for (dept, route_url) in routes {
            let addr = http_client::parse_http_addr(route_url)?;
            proxy.routes.push(Route {
                dept: dept.to_string(),
                url: route_url.trim_end_matches('/').to_string(),
                addr,
            });
        }
        Some(proxy)
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// Configured routes as "dept=url" pairs (surfaced in /stats).
    pub fn route_urls(&self) -> Vec<(&str, &str)> {
        self.routes
            .iter()
            .map(|r| (r.dept.as_str(), r.url.as_str()))
            .collect()
    }

    pub fn proxied_requests(&self) -> u64 {
        self.proxied.load(Ordering::Relaxed)
    }

    pub fn streamed_requests(&self) -> u64 {
        self.streamed.load(Ordering::Relaxed)
    }

    pub fn routed_requests(&self) -> u64 {
        self.routed.load(Ordering::Relaxed)
    }

    /// Pool member URLs when built from a comma-separated list (empty for a
    /// single default upstream).
    pub fn pool_urls(&self) -> Vec<String> {
        self.prefix_pool
            .as_ref()
            .map(|p| p.lock().expect("prefix router poisoned").urls().to_vec())
            .unwrap_or_default()
    }

    /// Per-pool-member routed request counts (aligned with pool_urls).
    pub fn pool_routed(&self) -> Vec<u64> {
        self.prefix_pool
            .as_ref()
            .map(|p| p.lock().expect("prefix router poisoned").routed_counts().to_vec())
            .unwrap_or_default()
    }

    /// Pick the upstream address: a department-specific route wins; then the
    /// prefix-affine pool (recording the assignment so same-prefix prompts
    /// stick); otherwise the default upstream. bool = matched a department
    /// route.
    fn pick(&self, dept: Option<&str>, prompt: Option<&str>) -> (String, bool) {
        if let Some(d) = dept {
            for route in &self.routes {
                if route.dept == d {
                    return (route.addr.clone(), true);
                }
            }
        }
        if let (Some(pool), Some(text)) = (&self.prefix_pool, prompt) {
            let mut router = pool.lock().expect("prefix router poisoned");
            let idx = router.pick(text);
            router.record(text, idx);
            return (router.addr(idx).to_string(), false);
        }
        (self.addr.clone(), false)
    }

    /// Forward one request and relay the response as a live stream: upstream
    /// chunks are written to the client socket as they arrive (SSE
    /// passthrough). A non-streaming upstream transparently falls back to the
    /// buffered path; an unreachable one degrades to 502.
    pub fn forward_streaming(
        &self,
        client: &mut TcpStream,
        method: &str,
        path: &str,
        body: &str,
        dept: Option<&str>,
    ) -> std::io::Result<()> {
        let prompt = extract_user_text(path, body);
        let (addr, matched) = self.pick(dept, prompt.as_deref());
        if matched {
            self.routed.fetch_add(1, Ordering::Relaxed);
        }
        let up = http_client::request_open(
            &addr,
            method,
            path,
            Some(body),
            http_client::DEFAULT_TIMEOUT,
        );
        let mut up = match up {
            Ok(up) => up,
            Err(e) => {
                return write_http_response(
                    client,
                    &HttpResponse::json(
                        502,
                        &format!(
                            r#"{{"error":"upstream_unreachable","upstream":"{}","detail":"{e}"}}"#,
                            addr
                        ),
                    ),
                );
            }
        };
        self.proxied.fetch_add(1, Ordering::Relaxed);
        if !up.chunked {
            let mut buf = Vec::new();
            while let Some(payload) = up.next_payload()? {
                buf.extend_from_slice(&payload);
            }
            let resp = HttpResponse {
                status: up.status,
                body: String::from_utf8_lossy(&buf).into_owned(),
            };
            return write_http_response(client, &resp);
        }
        self.streamed.fetch_add(1, Ordering::Relaxed);
        let content_type = up.content_type.as_deref().unwrap_or("text/event-stream");
        let head = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            up.status,
            reason_phrase(up.status),
            content_type
        );
        client.write_all(head.as_bytes())?;
        while let Some(payload) = up.next_payload()? {
            client.write_all(format!("{:x}\r\n", payload.len()).as_bytes())?;
            client.write_all(&payload)?;
            client.write_all(b"\r\n")?;
        }
        client.write_all(b"0\r\n\r\n")
    }

    /// Forward one JSON request verbatim; passthrough (status, body).
    /// Upstream failures degrade to a 502 instead of poisoning the server.
    fn forward(&self, method: &str, path: &str, body: Option<&str>, dept: Option<&str>) -> HttpResponse {
        let prompt = body.and_then(|b| extract_user_text(path, b));
        let (addr, matched) = self.pick(dept, prompt.as_deref());
        if matched {
            self.routed.fetch_add(1, Ordering::Relaxed);
        }
        match http_client::request(&addr, method, path, body, http_client::DEFAULT_TIMEOUT) {
            Ok((status, body)) => {
                self.proxied.fetch_add(1, Ordering::Relaxed);
                HttpResponse { status, body }
            }
            Err(e) => HttpResponse::json(
                502,
                &format!(
                    r#"{{"error":"upstream_unreachable","upstream":"{}","detail":"{e}"}}"#,
                    addr
                ),
            ),
        }
    }
}

/// A raw HTTP response ready to be serialized on the wire.
#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

impl HttpResponse {
    pub fn json(status: u16, body: &str) -> Self {
        Self {
            status,
            body: body.to_string(),
        }
    }
}

/// The minimal capability the HTTP frontend needs from a serving engine.
///
/// Implemented for every [`ModelEngine`] + [`Tokenizer`] pair via the blanket
/// impl below, so the server surface is independent of the concrete backend.
pub trait ServingEngine {
    fn generate(&mut self, req: &WriteRequest) -> GenerationOutput;
    fn stats(&self) -> EngineStats;

    /// Create a pinned KV checkpoint from `text`. Default: unsupported.
    fn create_checkpoint(&mut self, _text: &str) -> Option<u64> {
        None
    }
    /// Release a pinned checkpoint. Default: `false` (unsupported).
    fn drop_checkpoint(&mut self, _id: u64) -> bool {
        false
    }
    /// Branch generation off a pinned checkpoint. Default: unsupported.
    fn generate_from_checkpoint(
        &mut self,
        _id: u64,
        _continuation: &str,
        _sampling: SamplingParams,
    ) -> Option<GenerationOutput> {
        None
    }

    /// PD serving role ([`PdRole::Unified`] when PD is off).
    fn pd_role(&self) -> PdRole {
        PdRole::Unified
    }
    /// Run the prefill half and publish the KV bundle to the store.
    /// Default: `None` (PD not enabled).
    fn pd_prefill(
        &mut self,
        _req: &WriteRequest,
    ) -> Option<Result<PrefillReceipt, PdError>> {
        None
    }
    /// Decode from a KV bundle pulled out of the store.
    /// Default: `None` (PD not enabled).
    fn pd_decode(
        &mut self,
        _kv_key: &str,
        _sampling: SamplingParams,
    ) -> Option<Result<GenerationOutput, PdError>> {
        None
    }
    /// [`pd_decode`] with a per-token sink for SSE streaming.
    /// Default: `None` (PD not enabled).
    fn pd_decode_stream(
        &mut self,
        _kv_key: &str,
        _sampling: SamplingParams,
        _sink: &mut dyn FnMut(usize, &str),
    ) -> Option<Result<GenerationOutput, PdError>> {
        None
    }
}

impl<M: ModelEngine, T: Tokenizer> ServingEngine for Engine<M, T> {
    fn generate(&mut self, req: &WriteRequest) -> GenerationOutput {
        Engine::generate(self, req)
    }

    fn stats(&self) -> EngineStats {
        Engine::stats(self)
    }

    fn create_checkpoint(&mut self, text: &str) -> Option<u64> {
        Some(Engine::create_checkpoint(self, text).0)
    }

    fn drop_checkpoint(&mut self, id: u64) -> bool {
        Engine::drop_checkpoint(self, CheckpointId(id))
    }

    fn generate_from_checkpoint(
        &mut self,
        id: u64,
        continuation: &str,
        sampling: SamplingParams,
    ) -> Option<GenerationOutput> {
        Engine::generate_from_checkpoint(self, CheckpointId(id), continuation, sampling)
    }

    fn pd_role(&self) -> PdRole {
        Engine::pd_role(self)
    }

    fn pd_prefill(
        &mut self,
        req: &WriteRequest,
    ) -> Option<Result<PrefillReceipt, PdError>> {
        self.pd_store()
            .is_some()
            .then(|| Engine::prefill_only(self, req))
    }

    fn pd_decode(
        &mut self,
        kv_key: &str,
        sampling: SamplingParams,
    ) -> Option<Result<GenerationOutput, PdError>> {
        self.pd_store()
            .is_some()
            .then(|| Engine::decode_from_kv(self, kv_key, sampling))
    }

    fn pd_decode_stream(
        &mut self,
        kv_key: &str,
        sampling: SamplingParams,
        sink: &mut dyn FnMut(usize, &str),
    ) -> Option<Result<GenerationOutput, PdError>> {
        self.pd_store()
            .is_some()
            .then(|| Engine::decode_from_kv_streaming(self, kv_key, sampling, sink))
    }
}

/// PD conductor target: the prefill worker a decode-role server forwards
/// plain-text `/generate` requests to, before decoding from the returned KV
/// key locally (Mooncake's scheduler/conductor role).
pub struct Conductor {
    prefill_url: String,
    addr: String,
}

impl Conductor {
    pub fn from_url(url: &str) -> Option<Self> {
        http_client::parse_http_addr(url).map(|addr| Self {
            prefill_url: url.trim_end_matches('/').to_string(),
            addr,
        })
    }

    pub fn prefill_url(&self) -> &str {
        &self.prefill_url
    }
}

/// Routing + request handling, factored out so it can be unit-tested without
/// a socket.
pub fn handle<E: ServingEngine>(
    engine: &mut E,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> HttpResponse {
    handle_with_proxy(engine, None, method, path, body)
}

/// The endpoints that forward to the upstream worker in proxy mode.
fn is_generation_endpoint(method: &str, path: &str) -> bool {
    method == "POST" && (path == "/generate" || path == "/v1/chat/completions")
}

/// Does this request body ask for server-sent events (a stream:true field)?
fn wants_stream(body: Option<&str>) -> bool {
    body.and_then(|b| json::parse(b).ok())
        .and_then(|v| v.get("stream").and_then(Value::as_bool))
        .unwrap_or(false)
}

/// Laya gate shared by the buffered and streaming paths. Returns the
/// decided department (for --route upstream selection) plus, when the
/// request must not pass, the human-handoff reply to short-circuit with.
fn triage_decide(
    triage: Option<&crate::triage::Triage>,
    path: &str,
    body: Option<&str>,
) -> (Option<String>, Option<HttpResponse>) {
    let Some(t) = triage else {
        return (None, None);
    };
    let Some(text) = body.and_then(|b| extract_user_text(path, b)) else {
        return (None, None);
    };
    let out = t.check(&text);
    if out.escalate && !t.shadow {
        (None, Some(HttpResponse::json(200, &t.escalation_body(&out))))
    } else {
        (Some(out.department), None)
    }
}

/// Like [`handle`], but with an optional upstream [`Proxy`]: generation
/// endpoints (`/generate`, `/v1/chat/completions`) forward verbatim to the
/// upstream worker; the control plane stays local.
pub fn handle_with_proxy<E: ServingEngine>(
    engine: &mut E,
    proxy: Option<&Proxy>,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> HttpResponse {
    handle_with_triage(engine, proxy, None, method, path, body)
}

/// Pull the user's text out of a request body: `text` for /generate, the
/// last `role=="user"` message content for /v1/chat/completions.
fn extract_user_text(path: &str, body: &str) -> Option<String> {
    let v = crate::json::parse(body).ok()?;
    if path == "/generate" {
        return v
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string);
    }
    if let Some(Value::Array(items)) = v.get("messages") {
        for m in items.iter().rev() {
            if m.get("role").and_then(Value::as_str) == Some("user") {
                if let Some(Value::String(s)) = m.get("content") {
                    return Some(s.clone());
                }
            }
        }
    }
    None
}

/// [`handle_with_proxy`] plus an optional Laya [`Triage`] gate: generation
/// endpoints are judged by the System-1 decision model first; escalated
/// requests get a human-handoff reply without touching the upstream worker.
pub fn handle_with_triage<E: ServingEngine>(
    engine: &mut E,
    proxy: Option<&Proxy>,
    triage: Option<&crate::triage::Triage>,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> HttpResponse {
    handle_with_pd(engine, proxy, triage, None, method, path, body)
}

/// The engine-counter fields of `GET /stats` (proxy/triage extras are
/// appended by the caller). Shared by the lock-based server and the
/// concurrent actor server so both expose the same metrics schema.
fn base_stats_fields(s: &crate::engine::EngineStats) -> Vec<(String, Value)> {
    vec![
        ("total_requests".to_string(), Value::Number(s.total_requests as f64)),
        ("total_prompt_tokens".to_string(), Value::Number(s.total_prompt_tokens as f64)),
        ("total_prefill_tokens".to_string(), Value::Number(s.total_prefill_tokens as f64)),
        ("total_output_tokens".to_string(), Value::Number(s.total_output_tokens as f64)),
        ("total_forward".to_string(), Value::Number(s.total_forward as f64)),
        (
            "total_decode_steps".to_string(),
            Value::Number(s.total_decode_steps as f64),
        ),
        (
            "total_decode_calls".to_string(),
            Value::Number(s.total_decode_calls as f64),
        ),
        (
            "decode_batch_factor".to_string(),
            Value::Number(s.decode_batch_factor()),
        ),
        (
            "model_graft_tokens".to_string(),
            Value::Number(s.model_graft_tokens as f64),
        ),
        ("radix_nodes".to_string(), Value::Number(s.radix_nodes as f64)),
        (
            "radix_hit_tokens".to_string(),
            Value::Number(s.radix_hit_tokens as f64),
        ),
        ("apc_blocks".to_string(), Value::Number(s.apc_blocks as f64)),
        (
            "apc_hit_tokens".to_string(),
            Value::Number(s.apc_hit_tokens as f64),
        ),
        (
            "active_checkpoints".to_string(),
            Value::Number(s.active_checkpoints as f64),
        ),
        (
            "checkpoint_hit_tokens".to_string(),
            Value::Number(s.checkpoint_hit_tokens as f64),
        ),
        (
            "compute_saved_tokens".to_string(),
            Value::Number(s.compute_saved_tokens() as f64),
        ),
        (
            "prefill_skip_ratio".to_string(),
            Value::Number(s.prefill_skip_ratio()),
        ),
        (
            "avg_forward_per_output_token".to_string(),
            Value::Number(s.avg_forward_per_output_token()),
        ),
        (
            "kv_blocks".to_string(),
            Value::Number(s.kv_blocks as f64),
        ),
        ("kv_free_blocks".to_string(), Value::Number(s.kv_free_blocks as f64)),
        ("kv_allocations".to_string(), Value::Number(s.kv_allocations as f64)),
        ("kv_frees".to_string(), Value::Number(s.kv_frees as f64)),
        ("kv_utilization".to_string(), Value::Number(s.kv_utilization())),
        ("faulted_requests".to_string(), Value::Number(s.faulted_requests as f64)),
        ("rejected_requests".to_string(), Value::Number(s.rejected_requests as f64)),
        ("aborted_requests".to_string(), Value::Number(s.aborted_requests as f64)),
        (
            "pd_role".to_string(),
            Value::String(s.pd_role.as_str().to_string()),
        ),
        (
            "pd_prefill_requests".to_string(),
            Value::Number(s.pd_prefill_requests as f64),
        ),
        (
            "pd_decode_requests".to_string(),
            Value::Number(s.pd_decode_requests as f64),
        ),
        (
            "pd_kv_bytes".to_string(),
            Value::Number(s.pd_kv_bytes as f64),
        ),
        ("savings".to_string(), {
            // Four-axis savings ledger in three units. tokens: prompt tokens
            // never re-materialized, per mechanism. pages: the same in KV
            // blocks (floor of tokens / block size). forwards: one saved
            // prefill forward per saved token, plus decode calls the batcher
            // folded away (logical steps minus physical calls).
            let bs = (s.kv_block_size.max(1)) as u64;
            let radix = s.radix_hit_tokens;
            let apc = s.apc_hit_tokens;
            let ckpt = s.checkpoint_hit_tokens;
            let graft = s.model_graft_tokens;
            let tokens_total = radix + apc + ckpt + graft;
            let batch_saved = s.total_decode_steps.saturating_sub(s.total_decode_calls);
            let tokens_axis = Value::Object(vec![
                ("radix".to_string(), Value::Number(radix as f64)),
                ("apc".to_string(), Value::Number(apc as f64)),
                ("checkpoint".to_string(), Value::Number(ckpt as f64)),
                ("graft".to_string(), Value::Number(graft as f64)),
                ("total".to_string(), Value::Number(tokens_total as f64)),
            ]);
            let pages_axis = Value::Object(vec![
                ("radix".to_string(), Value::Number((radix / bs) as f64)),
                ("apc".to_string(), Value::Number((apc / bs) as f64)),
                ("checkpoint".to_string(), Value::Number((ckpt / bs) as f64)),
                ("graft".to_string(), Value::Number((graft / bs) as f64)),
                ("total".to_string(), Value::Number((tokens_total / bs) as f64)),
            ]);
            let forwards_axis = Value::Object(vec![
                ("cache".to_string(), Value::Number(tokens_total as f64)),
                ("batch".to_string(), Value::Number(batch_saved as f64)),
                (
                    "total".to_string(),
                    Value::Number((tokens_total + batch_saved) as f64),
                ),
            ]);
            Value::Object(vec![
                ("tokens".to_string(), tokens_axis),
                ("pages".to_string(), pages_axis),
                ("forwards".to_string(), forwards_axis),
            ])
        }),
    ]
}
/// [`handle_with_triage`] plus an optional PD [`Conductor`]: on a decode-role
/// server a plain-text `/generate` is conducted — POSTed to the prefill
/// worker, whose KV bundle key is then decoded locally.
pub fn handle_with_pd<E: ServingEngine>(
    engine: &mut E,
    proxy: Option<&Proxy>,
    triage: Option<&crate::triage::Triage>,
    conductor: Option<&Conductor>,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> HttpResponse {
    if let Some(proxy) = proxy {
        // Heavy generation goes to the upstream worker. Everything else
        // (health, stats, checkpoints) is pagoda-local control plane.
        if is_generation_endpoint(method, path) {
            let (dept, escalation) = triage_decide(triage, path, body);
            if let Some(escalation) = escalation {
                return escalation;
            }
            return proxy.forward(method, path, body, dept.as_deref());
        }
    }
    match (method, path) {
        ("GET", "/health") => HttpResponse::json(200, r#"{"status":"ok"}"#),
        ("GET", "/stats") => {
            let s = engine.stats();
            let mut fields = base_stats_fields(&s);
            if let Some(proxy) = proxy {
                fields.push((
                    "upstream".to_string(),
                    Value::String(proxy.url().to_string()),
                ));
                fields.push((
                    "proxied_requests".to_string(),
                    Value::Number(proxy.proxied_requests() as f64),
                ));
                fields.push((
                    "streamed_requests".to_string(),
                    Value::Number(proxy.streamed_requests() as f64),
                ));
                fields.push((
                    "routed_requests".to_string(),
                    Value::Number(proxy.routed_requests() as f64),
                ));
                let pool_urls = proxy.pool_urls();
                if !pool_urls.is_empty() {
                    fields.push((
                        "upstream_pool".to_string(),
                        Value::Array(pool_urls.into_iter().map(Value::String).collect()),
                    ));
                    fields.push((
                        "upstream_pool_routed".to_string(),
                        Value::Array(
                            proxy
                                .pool_routed()
                                .into_iter()
                                .map(|n| Value::Number(n as f64))
                                .collect(),
                        ),
                    ));
                }
                let routes: Vec<Value> = proxy
                    .route_urls()
                    .into_iter()
                    .map(|(dept, url)| {
                        Value::Object(vec![
                            ("department".to_string(), Value::String(dept.to_string())),
                            ("upstream".to_string(), Value::String(url.to_string())),
                        ])
                    })
                    .collect();
                if !routes.is_empty() {
                    fields.push(("routes".to_string(), Value::Array(routes)));
                }
            }
            if let Some(t) = triage {
                fields.push((
                    "triage".to_string(),
                    Value::String(t.url().to_string()),
                ));
                fields.push((
                    "triaged_requests".to_string(),
                    Value::Number(t.triaged() as f64),
                ));
                fields.push((
                    "escalated_requests".to_string(),
                    Value::Number(t.escalated() as f64),
                ));
                fields.push((
                    "triage_unavailable".to_string(),
                    Value::Number(t.unavailable() as f64),
                ));
            }
            let body = Value::Object(fields).to_json();
            HttpResponse::json(200, &body)
        }
        ("POST", "/generate") => {
            let parsed = body.and_then(|b| json::parse(b).ok());
            // PD decode path: {"kv_key": "kv-..."} pulls a prefilled bundle
            // out of the store and decodes without re-prefill.
            if let Some(key) = parsed
                .as_ref()
                .and_then(|v| v.get("kv_key"))
                .and_then(Value::as_str)
            {
                let sampling = parsed
                    .as_ref()
                    .and_then(|v| v.get("sampling_params"))
                    .map(parse_sampling)
                    .unwrap_or_default();
                return pd_decode_response(engine, key, sampling);
            }
            let req = match parsed.and_then(build_generate_request) {
                Some(req) => req,
                None => {
                    return HttpResponse::json(
                        400,
                        r#"{"error":"invalid request; expected {\"text\": ...}}"#,
                    )
                }
            };
            // PD conductor path: hand the prompt to the prefill worker, then
            // decode from the returned KV bundle.
            if let Some(c) = conductor {
                return conduct(engine, c, &req);
            }
            if engine.pd_role() == PdRole::Prefill {
                return HttpResponse::json(
                    400,
                    r#"{"error":"prefill-only worker; use POST /prefill"}"#,
                );
            }
            let out = engine.generate(&req);
            if out.finish_reason == FinishReason::Rejected {
                return rejection_response(&out);
            }
            HttpResponse::json(200, &generate_response(&out).to_json())
        }
        ("POST", "/prefill") => {
            if engine.pd_role() == PdRole::Decode {
                return HttpResponse::json(
                    400,
                    r#"{"error":"decode-only worker; use POST /generate"}"#,
                );
            }
            let req = match body
                .and_then(|b| json::parse(b).ok())
                .and_then(build_generate_request)
            {
                Some(req) => req,
                None => {
                    return HttpResponse::json(
                        400,
                        r#"{"error":"invalid request; expected {\"text\": ...}}"#,
                    )
                }
            };
            match engine.pd_prefill(&req) {
                Some(Ok(r)) => HttpResponse::json(
                    200,
                    &Value::Object(vec![
                        ("kv_key".to_string(), Value::String(r.kv_key)),
                        (
                            "prompt_tokens".to_string(),
                            Value::Number(r.prompt_tokens as f64),
                        ),
                        (
                            "prefill_tokens".to_string(),
                            Value::Number(r.prefill_tokens as f64),
                        ),
                        ("kv_bytes".to_string(), Value::Number(r.kv_bytes as f64)),
                        ("store_hit".to_string(), Value::Bool(r.store_hit)),
                    ])
                    .to_json(),
                ),
                Some(Err(PdError::Rejected(out))) => rejection_response(&out),
                Some(Err(e)) => HttpResponse::json(
                    502,
                    &Value::Object(vec![
                        ("error".to_string(), Value::String("pd_prefill_failed".into())),
                        ("detail".to_string(), Value::String(e.to_string())),
                    ])
                    .to_json(),
                ),
                None => HttpResponse::json(
                    400,
                    r#"{"error":"pd not enabled; start with --role prefill --store http://host:port"}"#,
                ),
            }
        }
        ("POST", "/v1/chat/completions") => {
            let req = match body
                .and_then(|b| json::parse(b).ok())
                .and_then(build_chat_request)
            {
                Some(req) => req,
                None => {
                    return HttpResponse::json(
                        400,
                        r#"{"error":"invalid messages payload"}"#,
                    )
                }
            };
            let out = engine.generate(&req);
            if out.finish_reason == FinishReason::Rejected {
                return rejection_response(&out);
            }
            HttpResponse::json(200, &chat_response(&out).to_json())
        }
        ("POST", "/checkpoint") => {
            let text = body
                .and_then(|b| json::parse(b).ok())
                .and_then(|v| v.get("text").and_then(Value::as_str).map(str::to_string));
            let Some(text) = text else {
                return HttpResponse::json(400, r#"{"error":"expected {\"text\": ...}"}"#);
            };
            match engine.create_checkpoint(&text) {
                Some(id) => HttpResponse::json(
                    200,
                    &Value::Object(vec![(
                        "checkpoint_id".to_string(),
                        Value::Number(id as f64),
                    )])
                    .to_json(),
                ),
                None => HttpResponse::json(
                    400,
                    r#"{"error":"checkpoints not supported by this backend"}"#,
                ),
            }
        }
        ("POST", "/checkpoint/generate") => {
            let parsed = body.and_then(|b| json::parse(b).ok()).and_then(|v| {
                let id = v.get("checkpoint_id")?.as_usize()? as u64;
                let text = v.get("text")?.as_str()?.to_string();
                let sampling = v
                    .get("sampling_params")
                    .map(parse_sampling)
                    .unwrap_or_default();
                Some((id, text, sampling))
            });
            let Some((id, text, sampling)) = parsed else {
                return HttpResponse::json(
                    400,
                    r#"{"error":"expected {\"checkpoint_id\": N, \"text\": ...}"}"#,
                );
            };
            match engine.generate_from_checkpoint(id, &text, sampling) {
                Some(out) => {
                    if out.finish_reason == FinishReason::Rejected {
                        return rejection_response(&out);
                    }
                    let mut resp = generate_response(&out);
                    if let Value::Object(ref mut entries) = resp {
                        entries.push(("checkpoint_id".to_string(), Value::Number(id as f64)));
                    }
                    HttpResponse::json(200, &resp.to_json())
                }
                None => HttpResponse::json(404, r#"{"error":"checkpoint not found"}"#),
            }
        }
        ("POST", "/checkpoint/delete") => {
            let id = body
                .and_then(|b| json::parse(b).ok())
                .and_then(|v| v.get("checkpoint_id").and_then(Value::as_usize))
                .map(|x| x as u64);
            let Some(id) = id else {
                return HttpResponse::json(400, r#"{"error":"expected {\"checkpoint_id\": N}"}"#);
            };
            let deleted = engine.drop_checkpoint(id);
            HttpResponse::json(
                200,
                &Value::Object(vec![("deleted".to_string(), Value::Bool(deleted))]).to_json(),
            )
        }
        _ => HttpResponse::json(404, r#"{"error":"not found"}"#),
    }
}

fn build_generate_request(v: Value) -> Option<WriteRequest> {
    let text = v.get("text")?.as_str()?.to_string();
    let sampling = v
        .get("sampling_params")
        .map(parse_sampling)
        .unwrap_or_default();
    Some(WriteRequest::new(text, sampling))
}

/// Decode from a store-pulled KV bundle, mapping PD failures to HTTP codes.
fn pd_decode_response<E: ServingEngine>(
    engine: &mut E,
    kv_key: &str,
    sampling: SamplingParams,
) -> HttpResponse {
    match engine.pd_decode(kv_key, sampling) {
        Some(Ok(out)) => {
            if out.finish_reason == FinishReason::Rejected {
                return rejection_response(&out);
            }
            let mut resp = generate_response(&out);
            if let Value::Object(ref mut entries) = resp {
                entries.push(("kv_key".to_string(), Value::String(kv_key.to_string())));
            }
            HttpResponse::json(200, &resp.to_json())
        }
        Some(Err(e)) => pd_error_response(kv_key, e),
        None => HttpResponse::json(
            400,
            r#"{"error":"pd not enabled; start with --role decode --store http://host:port"}"#,
        ),
    }
}

/// SSE decode from a KV bundle: one `data:` frame per sampled token, then a
/// final frame carrying the finish reason and usage, then `[DONE]`. The head
/// is written lazily on the first frame so pre-decode failures (kv miss,
/// store down) still surface as plain JSON errors.
fn pd_stream_decode<E: ServingEngine>(
    engine: &mut E,
    kv_key: &str,
    sampling: SamplingParams,
    stream: &mut TcpStream,
) -> std::io::Result<()> {
    const SSE_HEAD: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";

    let mut wrote_head = false;
    let mut sink = |id: usize, piece: &str| {
        if !wrote_head {
            if stream.write_all(SSE_HEAD.as_bytes()).is_err() {
                return;
            }
            wrote_head = true;
        }
        let frame = format!(
            "data: {}\n\n",
            Value::Object(vec![
                ("id".to_string(), Value::Number(id as f64)),
                ("token".to_string(), Value::String(piece.to_string())),
            ])
            .to_json()
        );
        let _ = write_chunk(stream, frame.as_bytes());
    };
    match engine.pd_decode_stream(kv_key, sampling, &mut sink) {
        None => write_http_response(
            stream,
            &HttpResponse::json(
                400,
                r#"{"error":"pd not enabled; start with --role decode --store http://host:port"}"#,
            ),
        ),
        Some(Err(e)) => write_http_response(stream, &pd_error_response(kv_key, e)),
        Some(Ok(out)) => {
            if !wrote_head {
                stream.write_all(SSE_HEAD.as_bytes())?;
            }
            let final_frame = format!(
                "data: {}\n\n",
                Value::Object(vec![
                    (
                        "finish_reason".to_string(),
                        Value::String(out.finish_reason.to_string()),
                    ),
                    (
                        "prompt_tokens".to_string(),
                        Value::Number(out.prompt_tokens as f64),
                    ),
                    (
                        "output_tokens".to_string(),
                        Value::Number(out.output_token_ids.len() as f64),
                    ),
                ])
                .to_json()
            );
            write_chunk(stream, final_frame.as_bytes())?;
            write_chunk(stream, b"data: [DONE]\n\n")?;
            stream.write_all(b"0\r\n\r\n")
        }
    }
}

/// PD streaming entry point: returns `Some(..)` when the request was handled
/// (or answered with an error), `None` to fall through to the buffered path.
fn pd_stream<E: ServingEngine>(
    engine: &mut E,
    conductor: Option<&Conductor>,
    body: &str,
    stream: &mut TcpStream,
) -> Option<std::io::Result<()>> {
    let v = json::parse(body).ok()?;
    let sampling = v.get("sampling_params").map(parse_sampling).unwrap_or_default();
    let key = match v.get("kv_key").and_then(Value::as_str) {
        Some(k) => k.to_string(),
        None => {
            // Conductor path: plain text + stream:true — prefill round trip
            // (buffered), then stream the decode.
            let conductor = conductor?;
            let text = v.get("text").and_then(Value::as_str)?;
            match prefill_roundtrip(conductor, text, &sampling) {
                Ok(key) => key,
                Err(resp) => return Some(write_http_response(stream, &resp)),
            }
        }
    };
    Some(pd_stream_decode(engine, &key, sampling, stream))
}

/// Conductor round trip: prefill the prompt on the prefill worker, then
/// decode locally from the returned KV bundle key.
fn conduct<E: ServingEngine>(engine: &mut E, conductor: &Conductor, req: &WriteRequest) -> HttpResponse {
    match prefill_roundtrip(conductor, &req.text, &req.sampling) {
        Ok(key) => pd_decode_response(engine, &key, req.sampling.clone()),
        Err(resp) => resp,
    }
}

/// POST the prompt to the prefill worker and return its KV bundle key.
fn prefill_roundtrip(
    conductor: &Conductor,
    text: &str,
    sampling: &SamplingParams,
) -> Result<String, HttpResponse> {
    let body = Value::Object(vec![
        ("text".to_string(), Value::String(text.to_string())),
        ("sampling_params".to_string(), sampling_json(sampling)),
    ])
    .to_json();
    let (status, resp) = match http_client::request(
        &conductor.addr,
        "POST",
        "/prefill",
        Some(&body),
        http_client::DEFAULT_TIMEOUT,
    ) {
        Ok(pair) => pair,
        Err(e) => {
            return Err(HttpResponse::json(
                502,
                &Value::Object(vec![
                    ("error".to_string(), Value::String("prefill_unreachable".into())),
                    ("detail".to_string(), Value::String(e.to_string())),
                ])
                .to_json(),
            ))
        }
    };
    if status != 200 {
        return Err(HttpResponse::json(
            502,
            &Value::Object(vec![
                ("error".to_string(), Value::String("prefill_failed".into())),
                ("status".to_string(), Value::Number(status as f64)),
                ("detail".to_string(), Value::String(resp)),
            ])
            .to_json(),
        ));
    }
    json::parse(&resp)
        .ok()
        .and_then(|v| v.get("kv_key").and_then(Value::as_str).map(str::to_string))
        .ok_or_else(|| {
            HttpResponse::json(
                502,
                r#"{"error":"prefill_failed","detail":"no kv_key in prefill response"}"#,
            )
        })
}

/// The `sampling_params` JSON shape the prefill worker parses back.
fn sampling_json(p: &SamplingParams) -> Value {
    Value::Object(vec![
        ("max_tokens".to_string(), Value::Number(p.max_tokens as f64)),
        ("temperature".to_string(), Value::Number(p.temperature as f64)),
        ("top_p".to_string(), Value::Number(p.top_p as f64)),
        ("top_k".to_string(), Value::Number(p.top_k as f64)),
        (
            "frequency_penalty".to_string(),
            Value::Number(p.frequency_penalty as f64),
        ),
        (
            "presence_penalty".to_string(),
            Value::Number(p.presence_penalty as f64),
        ),
        ("seed".to_string(), Value::Number(p.seed as f64)),
    ])
}

/// Map a PD error to the matching HTTP response.
fn pd_error_response(kv_key: &str, e: PdError) -> HttpResponse {
    match e {
        PdError::KvMiss(_) => HttpResponse::json(
            404,
            &Value::Object(vec![
                ("error".to_string(), Value::String("kv_miss".into())),
                ("kv_key".to_string(), Value::String(kv_key.to_string())),
            ])
            .to_json(),
        ),
        PdError::Rejected(out) => rejection_response(&out),
        other => HttpResponse::json(
            502,
            &Value::Object(vec![
                ("error".to_string(), Value::String("pd_decode_failed".into())),
                ("detail".to_string(), Value::String(other.to_string())),
            ])
            .to_json(),
        ),
    }
}

fn build_chat_request(v: Value) -> Option<WriteRequest> {
    let messages = v.get("messages")?;
    let Value::Array(items) = messages else { return None };
    let mut prompt = String::new();
    for item in items {
        let role = item.get("role")?.as_str()?;
        let content = item.get("content")?.as_str()?.to_string();
        match role {
            "system" => prompt.push_str(&format!("[SYS] {content}\n")),
            "assistant" => prompt.push_str(&format!("[ASSISTANT] {content}\n")),
            _ => prompt.push_str(&format!("[USER] {content}\n")),
        }
    }
    let mut sampling = parse_sampling(&v);
    if sampling.stop.is_empty() {
        sampling.stop = vec!["\n".to_string()]; // chat-completions convention
    }
    Some(WriteRequest::new(prompt.trim_end().to_string(), sampling))
}

fn parse_sampling(v: &Value) -> SamplingParams {
    let mut p = SamplingParams::default();
    if let Some(n) = v.get("max_tokens").and_then(Value::as_usize) {
        p.max_tokens = n;
    }
    if let Some(t) = v.get("temperature").and_then(Value::as_f64) {
        p.temperature = t as f32;
    }
    if let Some(t) = v.get("top_p").and_then(Value::as_f64) {
        p.top_p = t as f32;
    }
    if let Some(k) = v.get("top_k").and_then(Value::as_usize) {
        p.top_k = k;
    }
    if let Some(f) = v.get("frequency_penalty").and_then(Value::as_f64) {
        p.frequency_penalty = f as f32;
    }
    if let Some(f) = v.get("presence_penalty").and_then(Value::as_f64) {
        p.presence_penalty = f as f32;
    }
    if let Some(seed) = v.get("seed").and_then(Value::as_f64) {
        p.seed = seed as u64;
    }
    if let Some(Value::Array(stops)) = v.get("stop") {
        p.stop = stops
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
    }
    if let Some(Value::Array(ids)) = v.get("stop_token_ids") {
        p.stop_token_ids = ids.iter().filter_map(Value::as_usize).map(|x| x as u32).collect();
    }
    if let Some(Value::Object(entries)) = v.get("grammar") {
        let typ = entries
            .iter()
            .find(|(k, _)| k == "type")
            .and_then(|(_, val)| val.as_str());
        match typ {
            Some("json") => p.grammar = Some(Grammar::json()),
            Some("regex") => {
                if let Some(pattern) = entries
                    .iter()
                    .find(|(k, _)| k == "pattern")
                    .and_then(|(_, val)| val.as_str())
                {
                    p.grammar = Grammar::regex(pattern).ok();
                }
            }
            _ => {}
        }
    }
    p
}

fn rejection_response(out: &GenerationOutput) -> HttpResponse {
    let reason = out
        .rejection
        .map(|r| r.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    HttpResponse::json(
        400,
        &Value::Object(vec![(
            "error".to_string(),
            Value::String(format!("request rejected: {reason}")),
        )])
        .to_json(),
    )
}

fn generate_response(out: &GenerationOutput) -> Value {
    Value::Object(vec![
        ("text".to_string(), Value::String(out.text.clone())),
        (
            "finish_reason".to_string(),
            Value::String(out.finish_reason.to_string()),
        ),
        (
            "prompt_tokens".to_string(),
            Value::Number(out.prompt_tokens as f64),
        ),
        (
            "prefix_hit_tokens".to_string(),
            Value::Number(out.prefix_hit_tokens as f64),
        ),
        ("forward_count".to_string(), Value::Number(out.forward_count as f64)),
    ])
}

fn chat_response(out: &GenerationOutput) -> Value {
    let finish = match out.finish_reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Fault => "error",
        FinishReason::Rejected => "error",
    };
    Value::Object(vec![
        ("object".to_string(), Value::String("chat.completion".to_string())),
        (
            "choices".to_string(),
            Value::Array(vec![Value::Object(vec![
                (
                    "message".to_string(),
                    Value::Object(vec![
                        ("role".to_string(), Value::String("assistant".to_string())),
                        ("content".to_string(), Value::String(out.text.clone())),
                    ]),
                ),
                ("finish_reason".to_string(), Value::String(finish.to_string())),
            ])]),
        ),
    ])
}

/// Read one HTTP/1.1 request from a connection. Public so companion crates
/// Connection-level handler: same routing as the buffered handler, but a
/// proxied request with stream:true relays the upstream SSE stream
/// chunk-by-chunk onto the client socket instead of buffering. The engine
/// lock is only taken for the local control plane, so long-lived streams
/// never block /stats or /health.
pub fn handle_conn<E: ServingEngine>(
    engine: &std::sync::Mutex<E>,
    proxy: Option<&Proxy>,
    triage: Option<&crate::triage::Triage>,
    conductor: Option<&Conductor>,
    method: &str,
    path: &str,
    body: &str,
    stream: &mut TcpStream,
) -> std::io::Result<()> {
    if let Some(p) = proxy {
        if is_generation_endpoint(method, path) {
            let (dept, escalation) = triage_decide(triage, path, Some(body));
            if let Some(escalation) = escalation {
                return write_http_response(stream, &escalation);
            }
            if wants_stream(Some(body)) {
                return p.forward_streaming(stream, method, path, body, dept.as_deref());
            }
            return write_http_response(
                stream,
                &p.forward(method, path, Some(body), dept.as_deref()),
            );
        }
    }
    // PD conductor hoist: run the prefill round trip WITHOUT holding the
    // engine lock, so a long prefill never blocks in-flight decodes on this
    // worker (measured: a 738-token prefill added ~37s to a queued request
    // when the conductor prefetched under the lock). The body is rewritten
    // from {text, ...} to {kv_key, ...} and falls through to the normal
    // locked decode path below.
    let mut owned_body = body.to_string();
    if proxy.is_none() && conductor.is_some() && method == "POST" && path == "/generate" {
        if let Ok(v) = json::parse(body) {
            let has_key = v.get("kv_key").and_then(Value::as_str).is_some();
            let text = v.get("text").and_then(Value::as_str).map(str::to_string);
            if !has_key {
                if let (Some(c), Some(text)) = (conductor, text) {
                    let sampling = v
                        .get("sampling_params")
                        .map(parse_sampling)
                        .unwrap_or_default();
                    match prefill_roundtrip(c, &text, &sampling) {
                        Ok(key) => {
                            let mut fields: Vec<(String, Value)> = match v {
                                Value::Object(fields) => fields
                                    .into_iter()
                                    .filter(|(k, _)| k != "text")
                                    .collect(),
                                _ => Vec::new(),
                            };
                            fields.push(("kv_key".to_string(), Value::String(key)));
                            owned_body = Value::Object(fields).to_json();
                        }
                        Err(resp) => return write_http_response(stream, &resp),
                    }
                }
            }
        }
    }
    let body = owned_body.as_str();
    let resp = {
        let mut guard = engine.lock().unwrap();
        // PD streaming: /generate with stream:true on a PD-wired engine emits
        // SSE frames per token instead of one buffered JSON.
        if proxy.is_none() && method == "POST" && path == "/generate" && wants_stream(Some(body)) {
            if let Some(result) = pd_stream(&mut *guard, conductor, body, stream) {
                return result;
            }
        }
        handle_with_pd(&mut *guard, proxy, triage, conductor, method, path, Some(body))
    };
    write_http_response(stream, &resp)
}

/// (e.g. pagoda-hf's Laya server) can reuse the same tiny wire protocol.
pub fn read_http_request(stream: &mut TcpStream) -> std::io::Result<(String, String, String)> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        if header == "\r\n" || header == "\n" || header.is_empty() {
            break;
        }
        let lower = header.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok((method, path, String::from_utf8_lossy(&body).into_owned()))
}

/// Chunked framing so relays (the PD router) can forward frames live
/// instead of buffering to EOF.
fn write_chunk(stream: &mut TcpStream, payload: &[u8]) -> std::io::Result<()> {
    stream.write_all(format!("{:x}\r\n", payload.len()).as_bytes())?;
    stream.write_all(payload)?;
    stream.write_all(b"\r\n")
}

/// Status-line reason phrase for the codes pagoda emits or relays.
fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        502 => "Bad Gateway",
        _ => "Internal Server Error",
    }
}

/// Write one HTTP/1.1 response (JSON content type, connection close).
pub fn write_http_response(stream: &mut TcpStream, resp: &HttpResponse) -> std::io::Result<()> {
    let reason = reason_phrase(resp.status);
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        resp.status,
        reason,
        resp.body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(resp.body.as_bytes())
}

/// Concurrent serving: the engine moves onto a scheduler-actor thread
/// ([Engine::into_actor]) and HTTP handlers submit generation requests
/// through it, so in-flight requests interleave in one continuous-batching
/// loop instead of serializing behind a process-wide mutex. Unified-role
/// engines only; PD roles keep the lock-based [run_full] path (PD
/// concurrency comes from process separation, not intra-process overlap).
///
/// Endpoints: GET /health, GET /stats, POST /generate and
/// /v1/chat/completions (buffered JSON, or SSE with `"stream":true` on
/// /generate — same frame format as the PD stream path), plus the
/// /checkpoint* lifecycle. An optional triage gate (Laya System-1) runs on
/// the generation endpoints before any compute is spent.
pub fn run_concurrent<M, T>(
    engine: Engine<M, T>,
    addr: &str,
    triage: Option<Arc<crate::triage::Triage>>,
) -> std::io::Result<()>
where
    M: ModelEngine + Send + 'static,
    T: Tokenizer + Send + 'static,
{
    if engine.pd_role() != PdRole::Unified {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "run_concurrent serves the unified role; PD roles use run_full",
        ));
    }
    let handle = engine.into_actor();
    let listener = TcpListener::bind(addr)?;
    eprintln!(
        "pagoda serving on http://{} (concurrent scheduler)",
        listener.local_addr()?
    );
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                let handle = handle.clone();
                let triage = triage.clone();
                std::thread::spawn(move || {
                    let _ = stream.set_nodelay(true);
                    let parsed = read_http_request(&mut stream);
                    let result = match parsed {
                        Ok((method, path, body)) => handle_conn_concurrent(
                            &handle,
                            triage.as_deref(),
                            &method,
                            &path,
                            &body,
                            &mut stream,
                        ),
                        Err(_) => write_http_response(
                            &mut stream,
                            &HttpResponse::json(400, r#"{"error":"bad request"}"#),
                        ),
                    };
                    let _ = result;
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

fn handle_conn_concurrent(
    handle: &EngineHandle,
    triage: Option<&crate::triage::Triage>,
    method: &str,
    path: &str,
    body: &str,
    stream: &mut TcpStream,
) -> std::io::Result<()> {
    match (method, path) {
        ("GET", "/health") => {
            write_http_response(stream, &HttpResponse::json(200, r#"{"status":"ok"}"#))
        }
        ("GET", "/stats") => {
            let s = handle.stats();
            let mut fields = base_stats_fields(&s);
            fields.push(("mode".to_string(), Value::String("concurrent".to_string())));
            let body = Value::Object(fields).to_json();
            write_http_response(stream, &HttpResponse::json(200, &body))
        }
        ("POST", "/generate") => {
            let (_, escalation) = triage_decide(triage, path, Some(body));
            if let Some(escalation) = escalation {
                return write_http_response(stream, &escalation);
            }
            let stream_wanted = wants_stream(Some(body));
            let req = json::parse(body).ok().and_then(build_generate_request);
            let Some(req) = req else {
                return write_http_response(
                    stream,
                    &HttpResponse::json(
                        400,
                        r#"{"error":"invalid request; expected {\"text\": ...}"}"#,
                    ),
                );
            };
            if stream_wanted {
                return actor_stream(handle.subscribe(req), stream);
            }
            let out = handle.generate(req);
            if out.finish_reason == FinishReason::Rejected {
                return write_http_response(stream, &rejection_response(&out));
            }
            write_http_response(
                stream,
                &HttpResponse::json(200, &generate_response(&out).to_json()),
            )
        }
        ("POST", "/v1/chat/completions") => {
            let (_, escalation) = triage_decide(triage, path, Some(body));
            if let Some(escalation) = escalation {
                return write_http_response(stream, &escalation);
            }
            let req = json::parse(body).ok().and_then(build_chat_request);
            let Some(req) = req else {
                return write_http_response(
                    stream,
                    &HttpResponse::json(400, r#"{"error":"invalid messages payload"}"#),
                );
            };
            let out = handle.generate(req);
            if out.finish_reason == FinishReason::Rejected {
                return write_http_response(stream, &rejection_response(&out));
            }
            write_http_response(
                stream,
                &HttpResponse::json(200, &chat_response(&out).to_json()),
            )
        }
        ("POST", "/checkpoint") => {
            let text = json::parse(body)
                .ok()
                .and_then(|v| v.get("text").and_then(Value::as_str).map(str::to_string));
            let Some(text) = text else {
                return write_http_response(
                    stream,
                    &HttpResponse::json(400, r#"{"error":"expected {\"text\": ...}"}"#),
                );
            };
            let Some(id) = handle.checkpoint_create(text) else {
                return write_http_response(
                    stream,
                    &HttpResponse::json(503, r#"{"error":"engine unavailable"}"#),
                );
            };
            write_http_response(
                stream,
                &HttpResponse::json(
                    200,
                    &Value::Object(vec![(
                        "checkpoint_id".to_string(),
                        Value::Number(id as f64),
                    )])
                    .to_json(),
                ),
            )
        }
        ("POST", "/checkpoint/delete") => {
            let id = json::parse(body)
                .ok()
                .and_then(|v| v.get("checkpoint_id").and_then(Value::as_usize))
                .map(|x| x as u64);
            let Some(id) = id else {
                return write_http_response(
                    stream,
                    &HttpResponse::json(400, r#"{"error":"expected {\"checkpoint_id\": N}"}"#),
                );
            };
            let deleted = handle.checkpoint_drop(id);
            write_http_response(
                stream,
                &HttpResponse::json(
                    200,
                    &Value::Object(vec![("deleted".to_string(), Value::Bool(deleted))]).to_json(),
                ),
            )
        }
        ("POST", "/checkpoint/generate") => {
            let parsed = json::parse(body).ok().and_then(|v| {
                let id = v.get("checkpoint_id")?.as_usize()? as u64;
                let text = v.get("text")?.as_str()?.to_string();
                let sampling = v
                    .get("sampling_params")
                    .map(parse_sampling)
                    .unwrap_or_default();
                Some((id, text, sampling))
            });
            let Some((id, text, sampling)) = parsed else {
                return write_http_response(
                    stream,
                    &HttpResponse::json(
                        400,
                        r#"{"error":"expected {\"checkpoint_id\": N, \"text\": ...}"}"#,
                    ),
                );
            };
            let Some(events) = handle.checkpoint_generate(id, text, sampling) else {
                return write_http_response(
                    stream,
                    &HttpResponse::json(404, r#"{"error":"checkpoint not found"}"#),
                );
            };
            if wants_stream(Some(body)) {
                return actor_stream(events, stream);
            }
            let mut done = None;
            for event in events {
                if let StreamEvent::Done(out) = event {
                    done = Some(out);
                }
            }
            let out = done.expect("actor delivers Done before closing");
            if out.finish_reason == FinishReason::Rejected {
                return write_http_response(stream, &rejection_response(&out));
            }
            let mut resp = generate_response(&out);
            if let Value::Object(ref mut entries) = resp {
                entries.push(("checkpoint_id".to_string(), Value::Number(id as f64)));
            }
            write_http_response(stream, &HttpResponse::json(200, &resp.to_json()))
        }
        _ => write_http_response(stream, &HttpResponse::json(404, r#"{"error":"not found"}"#)),
    }
}

/// SSE relay for the concurrent server: subscribe to the actor's per-token
/// events and forward each as one chunked frame. A rejected request produces
/// a plain JSON error (head unwritten), mirroring the PD stream path.
fn actor_stream(
    events: std::sync::mpsc::Receiver<StreamEvent>,
    stream: &mut TcpStream,
) -> std::io::Result<()> {
    const SSE_HEAD: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    let mut wrote_head = false;
    let mut frame_id = 0usize;
    for event in events {
        match event {
            StreamEvent::Token(piece) => {
                if !wrote_head {
                    stream.write_all(SSE_HEAD.as_bytes())?;
                    wrote_head = true;
                }
                let frame = format!(
                    "data: {}\n\n",
                    Value::Object(vec![
                        ("id".to_string(), Value::Number(frame_id as f64)),
                        ("token".to_string(), Value::String(piece)),
                    ])
                    .to_json()
                );
                write_chunk(stream, frame.as_bytes())?;
                frame_id += 1;
            }
            StreamEvent::Done(out) => {
                if out.finish_reason == FinishReason::Rejected && !wrote_head {
                    return write_http_response(stream, &rejection_response(&out));
                }
                if !wrote_head {
                    stream.write_all(SSE_HEAD.as_bytes())?;
                }
                let final_frame = format!(
                    "data: {}\n\n",
                    Value::Object(vec![
                        (
                            "finish_reason".to_string(),
                            Value::String(out.finish_reason.to_string()),
                        ),
                        (
                            "prompt_tokens".to_string(),
                            Value::Number(out.prompt_tokens as f64),
                        ),
                        (
                            "output_tokens".to_string(),
                            Value::Number(out.output_token_ids.len() as f64),
                        ),
                    ])
                    .to_json()
                );
                write_chunk(stream, final_frame.as_bytes())?;
                write_chunk(stream, b"data: [DONE]\n\n")?;
                return stream.write_all(b"0\r\n\r\n");
            }
        }
    }
    // Channel closed without Done: the actor went away mid-request.
    if wrote_head {
        write_chunk(stream, b"data: {\"finish_reason\":\"Fault\"}\n\n")?;
        write_chunk(stream, b"data: [DONE]\n\n")?;
        stream.write_all(b"0\r\n\r\n")
    } else {
        write_http_response(
            stream,
            &HttpResponse::json(500, r#"{"error":"scheduler actor unavailable"}"#),
        )
    }
}

/// Blocking accept loop. Spawns one thread per connection.
pub fn run<E>(engine: Arc<Mutex<E>>, addr: &str) -> std::io::Result<()>
where
    E: ServingEngine + Send + 'static,
{
    run_with_proxy(engine, addr, None)
}

/// [`run`] with an upstream [`Proxy`]: generation endpoints forward to the
/// upstream worker, the control plane stays local.
pub fn run_with_proxy<E>(
    engine: Arc<Mutex<E>>,
    addr: &str,
    proxy: Option<Arc<Proxy>>,
) -> std::io::Result<()>
where
    E: ServingEngine + Send + 'static,
{
    run_with_triage(engine, addr, proxy, None)
}

/// [`run_with_proxy`] plus an optional Laya [`crate::triage::Triage`] gate.
pub fn run_with_triage<E>(
    engine: Arc<Mutex<E>>,
    addr: &str,
    proxy: Option<Arc<Proxy>>,
    triage: Option<Arc<crate::triage::Triage>>,
) -> std::io::Result<()>
where
    E: ServingEngine + Send + 'static,
{
    run_full(engine, addr, proxy, triage, None)
}

/// [`run_with_triage`] plus an optional PD [`Conductor`] for decode-role
/// servers that conduct plain-text requests through a prefill worker.
pub fn run_full<E>(
    engine: Arc<Mutex<E>>,
    addr: &str,
    proxy: Option<Arc<Proxy>>,
    triage: Option<Arc<crate::triage::Triage>>,
    conductor: Option<Arc<Conductor>>,
) -> std::io::Result<()>
where
    E: ServingEngine + Send + 'static,
{
    let listener = TcpListener::bind(addr)?;
    let local = listener.local_addr()?;
    match &proxy {
        Some(p) => eprintln!("pagoda serving on http://{local} (upstream: {})", p.url()),
        None => eprintln!("pagoda serving on http://{local}"),
    }
    if let Some(t) = &triage {
        eprintln!(
            "triage: laya at {}{}",
            t.url(),
            if t.shadow { " (shadow)" } else { "" }
        );
    }
    if let Some(c) = &conductor {
        eprintln!("pd conductor: prefill via {}", c.prefill_url());
    }
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                let engine = Arc::clone(&engine);
                let proxy = proxy.clone();
                let triage = triage.clone();
                let conductor = conductor.clone();
                std::thread::spawn(move || {
                    let _ = stream.set_nodelay(true);
                    let result = read_http_request(&mut stream);
                    let (method, path, body) = match result {
                        Ok(req) => req,
                        Err(_) => {
                            let _ = write_http_response(
                                &mut stream,
                                &HttpResponse::json(400, r#"{"error":"bad request"}"#),
                            );
                            return;
                        }
                    };
                    let _ = handle_conn(
                        &engine,
                        proxy.as_deref(),
                        triage.as_deref(),
                        conductor.as_deref(),
                        &method,
                        &path,
                        &body,
                        &mut stream,
                    );
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineConfig, ToyEngine};

    #[test]
    fn health_and_generate() {
        let mut engine = ToyEngine::toy(EngineConfig {
            num_kv_blocks: 64,
            block_size: 8,
            ..EngineConfig::default()
        });
        let health = handle(&mut engine, "GET", "/health", None);
        assert_eq!(health.status, 200);

        let resp = handle(
            &mut engine,
            "POST",
            "/generate",
            Some(r#"{"text":"Once upon","sampling_params":{"max_tokens":12}}"#),
        );
        assert_eq!(resp.status, 200);
        let v = json::parse(&resp.body).unwrap();
        assert!(v.get("text").unwrap().as_str().unwrap().len() > 0);
    }

    #[test]
    fn parse_sampling_accepts_grammar() {
        let v = json::parse(
            r#"{"max_tokens":10,"grammar":{"type":"json"}}"#,
        )
        .unwrap();
        assert_eq!(parse_sampling(&v).grammar, Some(Grammar::json()));

        let v = json::parse(
            r#"{"grammar":{"type":"regex","pattern":"\"[a-z]{3}\""}}"#,
        )
        .unwrap();
        assert!(parse_sampling(&v).grammar.is_some());
    }
}
